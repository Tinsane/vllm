import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe

import re


device = torch.device("cuda:0")

config = KVTransferConfig(
    kv_connector='PyNcclConnector',
    kv_buffer_device='cuda',
    kv_buffer_size=1e9,
    kv_rank=1,
    kv_role="kv_both",  # this arg doesn't matter in this test
    kv_parallel_size=2,
    kv_ip="192.168.0.145",
    kv_port=29500,
)

pipe = PyNcclPipe(
    local_rank=device.index,
    config=config,
    device="cuda",
)

with open('distributed_sender.py', 'r') as log:
    log_data = log.read()

tensor_regex = re.compile(
    r"Sending tensor (\S+), torch\.Size\((\[[\d, ]+\])\), (\S+), (\S+), ([\d\.\-]+)"
)

for match in tensor_regex.finditer(log_data):
    name = match.group(1)
    shape = eval(match.group(2))  # Convert string to list
    dtype = match.group(3)  # e.g., "torch.bfloat16"

    tensor = torch.randn(shape, dtype=dtype, device=device)
    check_sum = torch.sum(tensor, dtype=tensor.dtype).to(device="cpu")
    pipe.send_tensor(tensor, metadata={
        "finished": torch.zeros((1,), dtype=torch.bool, device='cpu'),
        "name": torch.tensor(list(name.encode('u8')), dtype=torch.uint8, device="cpu"),
        "check_sum": check_sum,
    })
    print(f"Sending tensor {name}, {tensor.shape}, {tensor.dtype}, {check_sum.dtype}, {check_sum}")

pipe.send_tensor(torch.zeros((1,), device=device), metadata={
    'finished': torch.ones((1,), dtype=torch.bool, device='cpu'),
})
pipe.close()
