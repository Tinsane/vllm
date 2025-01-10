from typing import Tuple, Generator, Iterable

import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe
from vllm.logger import init_logger

logger = init_logger(__name__)


class InfinibandModelLoader:
    def __init__(self):
        pass

    def _send_tensor(self, signal_pipe: PyNcclPipe, pipe: PyNcclPipe, name: str, tensor: torch.Tensor):
        signal_pipe.send_tensor(torch.zeros((1,), dtype=torch.bool, device='cpu'))
        signal_pipe.send_tensor(torch.tensor(list(name.encode('u8')), dtype=torch.uint8, device="cpu"))
        check_sum = torch.sum(tensor, dtype=tensor.dtype).to(device="cpu")
        signal_pipe.send_tensor(check_sum)
        logger.debug(f"Sending tensor {name}, {tensor.shape}, {tensor.dtype}, {check_sum.dtype}, {check_sum}")
        pipe.group.barrier()
        signal_pipe.group.barrier()
        pipe.send_tensor(tensor)

    def _send_finish(self, signal_pipe: PyNcclPipe):
        signal_pipe.send_tensor(torch.ones((1,), dtype=torch.bool, device='cpu'))

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
        logger.debug("Here: signal_pipe = ")
        signal_pipe = PyNcclPipe(
            local_rank=0,
            config=config,
            port_offset=1,
            device="cpu",
            wait_for_workers=False,
        )

        logger.debug("Here: for name, tensor in stream: ")
        for name, tensor in stream:
            self._send_tensor(signal_pipe, pipe, name, tensor)
            # self._send_tensor(signal_pipe, pipe, name, tensor.to(torch.device("cuda")))

        self._send_finish(signal_pipe)
        signal_pipe.close()
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
        signal_pipe = PyNcclPipe(
            local_rank=0,
            config=config,
            port_offset=1,
            device="cpu",
        )
        while True:
            done = signal_pipe.recv_tensor()
            if done.numpy()[0]:
                break
            name_raw = signal_pipe.recv_tensor()
            check_sum = signal_pipe.recv_tensor()
            name = bytes(name_raw.numpy()).decode('u8')
            tensor = pipe.recv_tensor()
            real_sum = torch.sum(tensor, dtype=tensor.dtype).to(device="cpu")
            logger.debug(f"Receiving tensor {name}, {tensor.shape}, {tensor.dtype}, {check_sum.dtype}, {check_sum}, {real_sum.dtype}, {real_sum}")
            logger.debug("Check sum difference: {}".format(check_sum - real_sum))
            pipe.group.barrier()
            signal_pipe.group.barrier()
            yield name, tensor

        logger.debug("Finished loading tensors")
        signal_pipe.close()
        pipe.close()
        logger.debug("Closed remote pipes")
