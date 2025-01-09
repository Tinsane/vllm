from typing import Tuple, Generator, Iterable

import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe


class InfinibandModelLoader:
    def __init__(self):
        pass

    def _send_tensor(self, signal_pipe: PyNcclPipe, pipe: PyNcclPipe, name: str, tensor: torch.Tensor):
        signal_pipe.send_tensor(torch.zeros((1,), device='cpu'))
        signal_pipe.send_tensor(torch.tensor(list(name.encode('u8')), dtype=torch.uint8, device="cpu"))
        pipe.send_tensor(tensor)

    def _send_finish(self, signal_pipe: PyNcclPipe):
        signal_pipe.send_tensor(torch.ones((1,), device='cpu'))

    def send_stream(self, stream: Iterable[Tuple[str, torch.Tensor]]):
        config = KVTransferConfig(
            kv_connector='PyNcclConnector',
            kv_buffer_device='cuda',
            kv_buffer_size=1e9,
            kv_rank=0,
            kv_role="kv_both",  # this arg doesn't matter in this test
            kv_parallel_size=2,
            kv_ip="192.168.0.28",
            kv_port=29500,
        )

        pipe = PyNcclPipe(
            local_rank=0,
            config=config,
            device="cuda",
        )
        signal_pipe = PyNcclPipe(
            local_rank=0,
            config=config,
            port_offset=1,
            device="cpu",
        )

        for name, tensor in stream:
            self._send_tensor(signal_pipe, pipe, name, tensor)
            # self._send_tensor(signal_pipe, pipe, name, tensor.to(torch.device("cuda")))

        self._send_finish(signal_pipe)
        signal_pipe.close()
        pipe.close()

    def load_tensors(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        config = KVTransferConfig(
            kv_connector='PyNcclConnector',
            kv_buffer_device='cuda',
            kv_buffer_size=1e9,
            kv_rank=1,
            kv_role="kv_both",  # this arg doesn't matter in this test
            kv_parallel_size=2,
            kv_ip="192.168.0.28",
            kv_port=29500,
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
            name = signal_pipe.recv_tensor()
            tensor = pipe.recv_tensor()
            yield bytes(name.numpy()).decode('u8'), tensor

        signal_pipe.close()
        pipe.close()
