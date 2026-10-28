set -euo pipefail

export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd
/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
import torch

assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "Quadro RTX 8000"
value = torch.ones(1, dtype=torch.float32, device="cuda:0")
assert value.item() == 1.0
props = torch.cuda.get_device_properties(0)
print(
    "GPU_UUID_PREFLIGHT_COMPLETE",
    torch.cuda.get_device_name(0),
    f"cc={props.major}.{props.minor}",
    f"bytes={props.total_memory}",
)
PY
