set -euo pipefail

CUDA_VISIBLE_DEVICES=0 /home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
import torch

theta = 1_000_000.0
head_dim = 64
cpu = 1.0 / (
    theta
    ** (
        torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float32)
        / head_dim
    )
)
cuda = 1.0 / (
    theta
    ** (
        torch.arange(0, head_dim, 2, dtype=torch.int64, device='cuda').to(torch.float32)
        / head_dim
    )
)
moved = cpu.to('cuda')
delta = (moved - cuda).abs()
print('CPU_MOVED_EQUALS_CUDA_RECOMPUTE', torch.equal(moved, cuda))
print('DIFFERING_ELEMENTS', int(torch.count_nonzero(delta)))
print('MAX_ABS_DIFF', float(delta.max()))
print('MAX_REL_DIFF', float((delta / moved.abs().clamp_min(torch.finfo(torch.float32).tiny)).max()))
print('CPU_ROUNDTRIP_EXACT', torch.equal(moved.cpu(), cpu))
PY
