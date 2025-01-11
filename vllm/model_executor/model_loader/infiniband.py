from typing import Tuple, Generator

import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe
from vllm.logger import init_logger

logger = init_logger(__name__)


class InfinibandModelLoader:
    def __init__(self):
        pass

    def _send_tensor(self, pipe: PyNcclPipe, name: str, tensor: torch.Tensor):
        while True:
            torch.cuda.synchronize()
            check_sum = torch.sum(tensor, dtype=tensor.dtype).to(device="cpu")
            torch.cuda.synchronize()
            logger.debug(f"Sending tensor {name}, {tensor.shape}, {tensor.dtype}, {check_sum.dtype}, {check_sum}")
            meta = pipe.send_tensor_with_response(tensor, metadata={
                "finished": torch.zeros((1,), dtype=torch.bool, device='cpu'),
                "name": torch.tensor(list(name.encode('u8')), dtype=torch.uint8, device="cpu"),
                "check_sum": check_sum
            })
            if meta["success"].numpy():
                break

    def _send_finish(self, pipe: PyNcclPipe):
        torch.cuda.synchronize()
        pipe.send_tensor(torch.ones((1,), dtype=torch.bfloat16, device="cuda"), metadata={
            "finished": torch.ones((1,), dtype=torch.bool, device='cpu'),
        })

    def send_stream(self, stream: Generator[Tuple[str, torch.Tensor], None, None]):
        logger.debug("Starting sending tensors")
        config = KVTransferConfig(
            kv_connector='PyNcclConnector',
            kv_buffer_device='cuda',
            kv_buffer_size=1e9,
            kv_rank=0,
            kv_role="kv_both",  # this arg doesn't matter in this test
            kv_parallel_size=2,
            kv_ip="192.168.0.145",
            kv_port=29503,
        )
        logger.debug("Here: pipe = ")
        pipe = PyNcclPipe(
            local_rank=0,
            config=config,
            device="cuda",
            wait_for_workers=False,
        )

        logger.debug("Here: for name, tensor in stream: ")
        for name, tensor in stream:
            self._send_tensor(pipe, name, tensor.to(device="cuda"))

        self._send_finish(pipe)
        torch.cuda.synchronize()
        pipe.group.barrier()
        pipe.close()

    def load_tensors(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        logger.debug("Starting load tensors")
        config = KVTransferConfig(
            kv_connector='PyNcclConnector',
            kv_buffer_device='cuda',
            kv_buffer_size=1e9,
            kv_rank=1,
            kv_role="kv_both",  # this arg doesn't matter in this test
            kv_parallel_size=2,
            kv_ip="192.168.0.145",
            kv_port=29503,
        )

        pipe = PyNcclPipe(
            local_rank=0,
            config=config,
            # TODO : pass actual device
            # device=device,
        )
        while True:
            tensor, metadata = pipe.recv_tensor()
            done = metadata['finished']
            if done.numpy()[0]:
                break
            name_raw = metadata['name']
            name = bytes(name_raw.numpy()).decode('u8')
            check_sum = metadata['check_sum']
            torch.cuda.synchronize()
            real_sum = torch.sum(tensor).to(device="cpu")
            torch.cuda.synchronize()
            logger.debug(f"Receiving tensor {name}, {tensor.shape}, {tensor.dtype}, {check_sum.dtype}, {check_sum}, {real_sum.dtype}, {real_sum}")
            logger.debug("Check sum difference: {}".format(check_sum - real_sum))
            if abs(check_sum - real_sum) < 1e-6:
                pipe.send_metadata_only({
                    'success': torch.ones((1,), dtype=torch.bool, device='cpu')
                })
                yield name, tensor
            else:
                pipe.send_metadata_only({
                    'success': torch.zeros((1,), dtype=torch.bool, device='cpu')
                })

        logger.debug("Finished loading tensors")
        pipe.group.barrier()
        pipe.close()
        logger.debug("Closed remote pipes")
