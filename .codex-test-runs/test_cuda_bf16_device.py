import torch

print(torch.cuda.get_device_name(0))
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
left = torch.ones((2, 2), device="cuda:0", dtype=torch.bfloat16)
print(left @ left)
