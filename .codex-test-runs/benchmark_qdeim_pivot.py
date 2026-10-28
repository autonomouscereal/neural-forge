import math
import time

import torch

from neural_forge.hf_lfm2_mlp_qdeim import _deterministic_pivoted_qr_indices


torch.set_num_threads(min(16, torch.get_num_threads()))
source = 2560
rows = torch.arange(source, dtype=torch.float64).unsqueeze(1) + 0.5
for modes in (64, 128, 256, 797):
    columns = torch.arange(modes, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(math.pi * rows * columns / source)
    basis[:, 0] /= math.sqrt(source)
    if modes > 1:
        basis[:, 1:] *= math.sqrt(2.0 / source)
    started = time.perf_counter()
    if modes == 797:
        try:
            import scipy.linalg
        except ImportError:
            print("scipy unavailable", flush=True)
            break
        _, _, pivots = scipy.linalg.qr(
            basis.T.numpy(), mode="economic", pivoting=True, check_finite=False
        )
        elapsed = time.perf_counter() - started
        print(
            f"scipy_qr modes={modes} seconds={elapsed:.3f} unique={len(set(pivots[:modes]))}",
            flush=True,
        )
        break
    pivots, _ = _deterministic_pivoted_qr_indices(basis)
    elapsed = time.perf_counter() - started
    print(
        f"current modes={modes} seconds={elapsed:.3f} unique={len(set(pivots))}",
        flush=True,
    )
    if elapsed > 20.0:
        break
