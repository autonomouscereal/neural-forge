#!/usr/bin/env python3
"""Run qwen-fold/qf_build.py UNCHANGED on a Pascal card.

qf_build's Harness only switches the qwen3_5 fused kernels (Triton GDN, causal-conv1d, fused gated RMSNorm) to their
torch fallbacks when the device is not CUDA.  The P40 (sm61) is CUDA without Triton, so this wrapper calls the same
qf_arch.force_torch_kernels() first and then hands argv to qf_build.main.  The switch is process-wide and idempotent;
qf_build's own receipt will show kernels_disabled = {} for the harness, so the switch is printed here and the queue
log keeps it.

  qf_build_p40.py <qwen-fold dir> <qf_build args...>
"""
import json
import sys

qf_dir, rest = sys.argv[1], sys.argv[2:]
sys.path.insert(0, qf_dir)
import qf_arch  # noqa: E402

disabled = qf_arch.force_torch_kernels()
print("[qf_build_p40] " + json.dumps({"kernels_disabled_before_build": disabled}), flush=True)
import qf_build  # noqa: E402

raise SystemExit(qf_build.main(rest))
