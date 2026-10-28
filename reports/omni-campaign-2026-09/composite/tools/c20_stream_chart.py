#!/usr/bin/env python3
"""C20 -- the global stream chart of the v2 fold (V2_FOLD_INIT_PLAN_2026-10-21 section 4.1).

The v2 residual stream is DEFINED as the image of the source trunk's stream under one fixed
orthonormal width chart C [H_src x d]:  z = alpha . h . C.  One basis for the whole stream, not
per-layer charts: a write made in basis C_k and read through C_{k+1} would put C_k C_{k+1}^+ - I
into every token (plan section 4.1).

C is the top-d eigenbasis of the UNCENTRED second moment of the source's residual rows, each row
first divided by its own RMS, pooled over every source layer.  Two choices, both recorded:

* Uncentred, not centred-plus-mean (the plan's wording).  The fold needs h ~ (h C) C^T for the
  row INCLUDING its mean (v2 has no biases); the rank-d subspace that best reconstructs the
  uncentred rows is exactly the top-d eigenbasis of the uncentred moment, so the mean is kept by
  construction rather than by a separate direction.  The energy the mean carries is recorded.
* RMS-normalised rows.  Every read of the stream is RMS-normalised, so what a chart must keep is
  the DIRECTION of each row; without the normalisation the late layers (whose stream norm is
  10-100x the early ones') would choose the basis alone.

Everything here is a diagnostic or a closed form: retained energy and the mean's energy are
recorded here, the per-layer retained energy by c21 (a per-layer [H x H] moment would cost
65 x 210 MB at H = 5120); none of them stops a fold.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

TOOL_NAME = "c20_stream_chart"
TOOL_VERSION = "1.0.0"
SCHEMA = "chimera_omni.composite.v2_stream_chart.v1"


class SecondMoment:
    """fp64 accumulator of sum(r r^T) and sum(r) over RMS-normalised rows r, per source layer too."""

    def __init__(self, width: int, device: torch.device | str = "cpu") -> None:
        self.width = int(width)
        self.gram = torch.zeros(self.width, self.width, dtype=torch.float64, device=device)
        self.total = torch.zeros(self.width, dtype=torch.float64, device=device)
        self.rows = 0
        self.layer_rows: dict[int, int] = {}

    def add(self, rows: torch.Tensor, layer: int, eps: float = 1.0e-6) -> None:
        """rows [n x width]; each is divided by its own RMS before it is pooled."""

        if rows.ndim != 2 or rows.shape[1] != self.width:
            raise ValueError(f"rows must be [n x {self.width}], got {tuple(rows.shape)}")
        if rows.shape[0] == 0:
            return
        r = rows.float()
        r = r * torch.rsqrt(r.pow(2).mean(dim=-1, keepdim=True) + eps)
        gram = (r.T @ r).to(self.gram.device, torch.float64)
        self.gram += gram
        self.total += r.sum(dim=0).to(self.total.device, torch.float64)
        self.rows += int(r.shape[0])
        self.layer_rows[layer] = self.layer_rows.get(layer, 0) + int(r.shape[0])


def fit_chart(moment: SecondMoment, rank: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """C [width x rank], orthonormal columns, descending eigenvalue order; plus its diagnostics."""

    if moment.rows == 0:
        raise ValueError("no rows were accumulated")
    if not 0 < rank <= moment.width:
        raise ValueError(f"rank must be in (0, {moment.width}], got {rank}")
    gram = moment.gram.cpu()
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("the second moment carries non-finite values")
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    chart = evecs[:, :rank].contiguous()
    # A deterministic sign per column (eigh's sign is arbitrary): the largest-magnitude entry is positive.
    pivot = chart.abs().argmax(dim=0)
    signs = torch.sign(chart[pivot, torch.arange(rank)])
    signs[signs == 0] = 1.0
    chart = chart * signs
    total = float(evals.clamp_min(0).sum())
    kept = float(evals[:rank].clamp_min(0).sum())
    mean = moment.total.cpu() / moment.rows
    mean_energy = float(mean @ mean) * moment.rows
    mean_kept = float(((mean @ chart) ** 2).sum() / max(float(mean @ mean), 1e-300))
    record = {
        "schema": SCHEMA,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "width": moment.width,
        "rank": int(rank),
        "rows": moment.rows,
        "rows_normalised": "each row divided by its own RMS (eps 1e-6) before pooling",
        "moment": "uncentred second moment (the mean is kept by the rank-d subspace, not a separate direction)",
        "retained_energy": kept / total if total > 0 else None,
        "mean_energy_fraction": mean_energy / total if total > 0 else None,
        "mean_retained_fraction": mean_kept,
        "rows_per_layer": {str(k): v for k, v in sorted(moment.layer_rows.items())},
        "retained_energy_per_layer": "measured by the fold itself (c21 records relL2 of h vs h C C^T per source layer)",
        "eigenvalue_head": [float(v) for v in evals[: min(8, rank)]],
        "eigenvalue_at_rank": float(evals[rank - 1]),
        "orthonormality_error": float((chart.T @ chart - torch.eye(rank, dtype=chart.dtype)).abs().max()),
    }
    return chart, record


def chart_sha256(chart: torch.Tensor) -> str:
    return hashlib.sha256(chart.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()).hexdigest()


def save_chart(path: str | Path, chart: torch.Tensor, record: dict[str, Any]) -> str:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = chart_sha256(chart)
    save_file({"C": chart.detach().cpu().to(torch.float64).contiguous()}, str(path),
              metadata={"schema": SCHEMA, "chart_sha256": sha, "record": json.dumps(record, sort_keys=True)})
    path.with_suffix(".json").write_text(json.dumps({**record, "chart_sha256": sha}, indent=1) + "\n", encoding="utf-8")
    return sha


def load_chart(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as handle:
        chart = handle.get_tensor("C")
        meta = handle.metadata() or {}
    sha = chart_sha256(chart)
    if meta.get("chart_sha256") and meta["chart_sha256"] != sha:
        raise ValueError(f"{path}: chart bytes do not match the recorded sha256")
    record = json.loads(meta.get("record", "{}"))
    record["chart_sha256"] = sha
    return chart, record
