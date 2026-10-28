import json
from pathlib import Path

import torch
from safetensors import safe_open


CACHE = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_complete_causal_evidence_20260720_a0003/parent/"
    "dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b"
)
MODEL = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M/model.safetensors")
FIT_BATCHES = tuple(range(8))
ALL_BATCHES = tuple(range(16))
LAYERS = tuple(range(14))
TARGET_INTERMEDIATE = 797


def snapshots(layer: int, batches: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    post_rows = []
    gate_rows = []
    up_rows = []
    for batch in batches:
        path = CACHE / "batches" / f"batch-{batch:06d}" / "trace.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            suffix = f"layer.{layer:03d}.mlp.post_swiglu"
            post_keys = [key for key in handle.keys() if key.endswith(suffix)]
            for post_key in post_keys:
                prefix = post_key[: -len("post_swiglu")]
                post_rows.append(handle.get_tensor(post_key).reshape(-1, 2560))
                gate_rows.append(handle.get_tensor(prefix + "gate").reshape(-1, 2560))
                up_rows.append(handle.get_tensor(prefix + "up").reshape(-1, 2560))
    return tuple(torch.cat(rows, dim=0).to(torch.float64) for rows in (post_rows, gate_rows, up_rows))


def spectrum(matrix: torch.Tensor) -> dict[str, object]:
    gram = matrix @ matrix.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min_(0).flip(0)
    singular = torch.sqrt(eigenvalues)
    maximum = float(singular[0].item())
    energy = eigenvalues / eigenvalues.sum()
    cumulative = energy.cumsum(0)

    def energy_rank(fraction: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(fraction, dtype=cumulative.dtype)).item()) + 1

    return {
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "rank_relative_1e_3": int((singular > maximum * 1e-3).sum().item()),
        "rank_relative_1e_4": int((singular > maximum * 1e-4).sum().item()),
        "rank_relative_1e_6": int((singular > maximum * 1e-6).sum().item()),
        "energy_rank_90": energy_rank(0.90),
        "energy_rank_99": energy_rank(0.99),
        "energy_rank_999": energy_rank(0.999),
        "target_797_supported_at_1e_6": bool((singular > maximum * 1e-6).sum().item() >= TARGET_INTERMEDIATE),
    }


with safe_open(MODEL, framework="pt", device="cpu") as weights:
    for layer in LAYERS:
        fit_post, fit_gate, fit_up = snapshots(layer, FIT_BATCHES)
        all_post, _, _ = snapshots(layer, ALL_BATCHES)
        row_norms = torch.linalg.vector_norm(fit_post, dim=1)
        median_norm = row_norms.median()
        anchor = int((row_norms - median_norm).abs().argmin().item())
        gate = fit_gate[anchor]
        up = fit_up[anchor]
        sigmoid = torch.sigmoid(gate)
        silu = gate * sigmoid
        silu_prime = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        w1 = weights.get_tensor(f"model.layers.{layer}.feed_forward.w1.weight").to(torch.float64)
        w2 = weights.get_tensor(f"model.layers.{layer}.feed_forward.w2.weight").to(torch.float64)
        w3 = weights.get_tensor(f"model.layers.{layer}.feed_forward.w3.weight").to(torch.float64)
        jacobian = w1.T * (silu_prime * up).unsqueeze(0) + w3.T * silu.unsqueeze(0)
        writer_scale = torch.linalg.vector_norm(w2, dim=0).clamp_min(torch.finfo(torch.float64).tiny)
        writer_scale = writer_scale / writer_scale.median()
        result = {
            "layer": layer,
            "fit": spectrum(fit_post),
            "all_cached": spectrum(all_post),
            "fit_plus_full_parent_tangent": spectrum(torch.cat((fit_post, jacobian), dim=0)),
            "writer_balanced_fit_plus_full_parent_tangent": spectrum(
                torch.cat((fit_post, jacobian), dim=0) * writer_scale.unsqueeze(0)
            ),
            "tangent_anchor_fit_row": anchor,
            "tangent_rows": int(jacobian.shape[0]),
            "target_intermediate": TARGET_INTERMEDIATE,
        }
        print(json.dumps(result, sort_keys=True), flush=True)
