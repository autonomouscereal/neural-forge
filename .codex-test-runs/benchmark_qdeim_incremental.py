import math
import time

import torch

from neural_forge.hf_lfm2_mlp_qdeim import _deterministic_pivoted_qr_indices


def incremental_pivots(basis: torch.Tensor):
    source_neurons, modes = map(int, basis.shape)
    matrix = basis.T.contiguous()
    q_columns = torch.empty((modes, modes), dtype=torch.float64)
    scores = matrix.square().sum(dim=0)
    selected_mask = torch.zeros(source_neurons, dtype=torch.bool)
    selected = []
    norms = []
    epsilon = torch.finfo(torch.float64).eps
    for step in range(modes):
        eligible = scores.masked_fill(selected_mask, -1.0)
        maximum = float(eligible.max().item())
        tie_tolerance = 128.0 * epsilon * max(1.0, abs(maximum))
        candidates = torch.nonzero(
            (~selected_mask) & (scores >= maximum - tie_tolerance),
            as_tuple=False,
        ).flatten()
        if not int(candidates.numel()):
            raise RuntimeError("no pivot")
        pivot = int(candidates.min().item())
        vector = matrix[:, pivot].clone()
        if step:
            prior = q_columns[:, :step]
            vector -= prior @ (prior.T @ vector)
            vector -= prior @ (prior.T @ vector)
        norm = float(torch.linalg.vector_norm(vector).item())
        if not math.isfinite(norm) or norm <= epsilon:
            raise RuntimeError("singular")
        q = vector / norm
        q_columns[:, step].copy_(q)
        projection = q @ matrix
        scores.sub_(projection.square()).clamp_min_(0.0)
        selected_mask[pivot] = True
        selected.append(pivot)
        norms.append(norm)
    return tuple(selected), torch.tensor(norms, dtype=torch.float64)


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
    incremental, incremental_norms = incremental_pivots(basis)
    elapsed = time.perf_counter() - started
    print(
        f"incremental modes={modes} seconds={elapsed:.3f} "
        f"unique={len(set(incremental))}",
        flush=True,
    )

    if modes <= 256:
        started = time.perf_counter()
        current, current_norms = _deterministic_pivoted_qr_indices(basis)
        current_elapsed = time.perf_counter() - started
        print(
            f"current modes={modes} seconds={current_elapsed:.3f} "
            f"same_pivots={current == incremental} "
            f"max_norm_delta={float((current_norms - incremental_norms).abs().max()):.3e}",
            flush=True,
        )
