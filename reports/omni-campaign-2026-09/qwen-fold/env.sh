#!/usr/bin/env bash
# Lane environment.  Sourced by every queue script; contains no secrets and no
# model paths that are not on LOCAL disk -- the CIFS share reads at under
# 3 MB/s and a capture that streams from it prices the whole wave wrong.
set -euo pipefail

export W=${W:-/home/cereal/omni-campaign/qwen-fold}
export PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
export PARENT=${PARENT:-/home/cereal/models/Qwen3.5-4B}
export EVIDENCE=${EVIDENCE:-$W/evidence/omni-evidence-v1.json}
export REVIEW_GPU_CAP=${REVIEW_GPU_CAP:-"8.6 NVIDIA GeForce RTX 3090"}
export LLAMA_CPP=${LLAMA_CPP:-/home/cereal/llama.cpp}

mkdir -p "$W/children" "$W/logs" "$W/cells" "$W/shards" "$W/evidence"

# GPU preflight.  PCI bus order, because CUDA's default order is by capability
# and a queue that asks for "device 0" without this can land on a P40.
gpu_preflight() {
  local want="${1:-$REVIEW_GPU_CAP}"
  local seen
  seen=$(CUDA_DEVICE_ORDER=PCI_BUS_ID "$PY" - <<'PYEOF'
import torch
index = 0
major, minor = torch.cuda.get_device_capability(index)
print(f"{major}.{minor} {torch.cuda.get_device_name(index)}")
PYEOF
)
  if [ "$seen" != "$want" ]; then
    echo "[qf] GPU0 is '$seen', this queue wants '$want'" >&2
    return 1
  fi
  echo "[qf] GPU0 preflight ok: $seen"
}
