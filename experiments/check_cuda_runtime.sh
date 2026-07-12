#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
import torch

print(
    {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
    }
)
value = torch.ones(4, device="cuda")
print(value)
PY
