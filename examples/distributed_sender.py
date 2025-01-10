import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe


device = torch.device("cuda:0")

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
    local_rank=device.index,
    config=config,
    device="cuda",
)

tensor_name = "kekw"
name_bytes = tensor_name.encode('u8')
tensor = torch.randn(131072, 131072, device=device)
pipe.send_tensor(tensor, metadata={
    "name": torch.tensor(list(name_bytes), dtype=torch.uint8, device="cpu"),
    "check_sum": torch.sum(tensor, dtype=tensor.dtype).to(device="cpu"),
})
print("Sent tensor with shape: {}".format(tensor.shape))
