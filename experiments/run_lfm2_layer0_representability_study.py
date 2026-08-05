#!/usr/bin/env python3
"""Single-boundary refit study of one LFM2 operator boundary (default layer 0).

R13 follow-up to NOTES-W13A.  The sealed r12 child's wall is layer 0: its
short-conv branch write has held-out cosine 0.171 against the chart-projected
parent write and the post-operator residual sits at relative L2 0.836, against
a predicted per-layer defect of 0.0965.  This runner decides whether that
deficit is REPRESENTATIONAL (the exact-4x layout cannot express the parent's
layer-0 function under any closed-form solve) or SOLVER-SIDE (a representable
target was fit badly), by re-running exactly the production closed-form solve
for that one boundary under a matrix of arms.

Nothing here trains.  Every arm is one closed-form producer fit plus one
closed-form consumer fit, both through the production compilers in
``neural_forge.hf_lfm2_structural_residual_attention_fold``.  No parent forward
call, no child forward call, no gradient, no optimizer step.  The evidence is
the sealed parent trace cache and the sealed balanced-basis phase the r12 body
fold itself consumed, both opened through their authenticated sessions.

Exact-input framing
-------------------
The first-divergence ledger measures ``c_in`` -- the child residual entering
layer 0 -- at relative L2 0.000000 and cosine 1.000000 over all 2,304 observed
states.  The child's layer-0 input therefore *is* the chart projection of the
parent's, bit for bit at fp64 print precision, so this runner constructs it as
``encode(parent residual.pre)`` and needs no child replay at all.  Arm
``published_child`` re-measures the sealed published child's own layer-0 write
under that construction and is required to reproduce the sealed diagnosis
numbers; if it does not, the harness is measuring something else and the run
stops before any solve arm is believed.

Metrics
-------
Every arm reports the branch write ``operator.output`` and the composed
``residual.post_operator``, both against the chart-projected parent, on the
held-out batches only, in two row sets:

* ``last_token`` -- one row per (record, phase), the exact 576 held-out states
  the first-divergence ledger aggregates.  This is the number to compare with
  NOTES-W13A.
* ``all_rows`` -- every full-prefix held-out row, the natural solve metric.

``relative_l2 = 1.0`` is the emit-zero baseline (R13 item 2).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

import neural_forge.hf_lfm2_balanced_basis_phase_cache as basis_phase_module
import neural_forge.hf_lfm2_circuit_trace_cache as trace_module
import neural_forge.hf_lfm2_structural_residual_attention_fold as fold_module
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache
from neural_forge.hf_closed_form_fit import fit_diagonal_norm
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (
    PhaseAnchorAuthority,
    load_existing_lfm2_circuit_balanced_basis_phase,
)
from neural_forge.hf_lfm2_circuit_trace_cache import open_lfm2_circuit_trace_cache
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (
    Lfm2DynamicClosedLoopAdapter,
)
from neural_forge.hf_lfm2_exact4x_first_divergence import (
    authoritative_split_by_batch,
    phase_names,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2SparseResidualAssignment,
    cached_short_conv_case_from_lfm2_trace,
    compile_lfm2_structural_short_conv_consumer_fold,
    compile_lfm2_structural_short_conv_producer_fold,
)

SCHEMA = "neural-forge-lfm2-layer0-representability-study-v1"


class Lfm2Layer0StudyError(RuntimeError):
    """The single-boundary study failed closed."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Lfm2Layer0StudyError(f"{name} must be a sha256 hex digest.")
    return value.lower()


def _read_raw_hmac_key(path: Path) -> bytes:
    resolved = Path(path).expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise Lfm2Layer0StudyError("HMAC key file must be an ordinary file.")
    payload = resolved.read_bytes()
    if not payload:
        raise Lfm2Layer0StudyError("HMAC key file is empty.")
    return payload


def _module_sha256(module: Any) -> str:
    path = Path(module.__file__).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, body: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def _row_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Per-row cosine/relative-L2 summary matching the first-divergence ledger.

    ``_metrics`` in ``hf_lfm2_exact4x_first_divergence`` is per single row; the
    ledger then averages those rows.  This does the same thing vectorized.
    """

    left = candidate.to(dtype=torch.float64)
    right = reference.to(dtype=torch.float64)
    delta = left - right
    l2 = torch.linalg.vector_norm(delta, dim=-1)
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    tiny = torch.finfo(torch.float64).tiny
    relative = l2 / right_norm.clamp_min(tiny)
    denominator = (left_norm * right_norm).clamp_min(tiny)
    cosine = ((left * right).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
    zero = (left_norm == 0.0) | (right_norm == 0.0)
    if bool(zero.any().item()):
        cosine = torch.where(zero, torch.zeros_like(cosine), cosine)
    irreducible = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
    ratio = left_norm / right_norm.clamp_min(tiny)
    return {
        "rows": int(l2.numel()),
        "mean_cosine": float(cosine.mean().item()),
        "mean_relative_l2": float(relative.mean().item()),
        "median_relative_l2": float(relative.median().item()),
        "mean_norm_ratio": float(ratio.mean().item()),
        "mean_irreducible": float(irreducible.mean().item()),
        "aggregate_relative_l2": float(
            (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(right)).item()
        ),
        "fraction_rows_beating_zero_baseline": float(
            (relative < 1.0).to(dtype=torch.float64).mean().item()
        ),
    }


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

_SUFFIXES = (
    "operator.input",
    "operator.output",
    "residual.pre",
    "residual.post_operator",
    "short_conv.b",
    "short_conv.c",
    "short_conv.x",
    "short_conv.context",
)


class BoundaryEvidence:
    """Sealed layer-local parent rows plus the authenticated FIT/heldout split."""

    def __init__(
        self,
        *,
        layer_index: int,
        phases: tuple[str, ...],
        split_by_batch: Mapping[int, str],
        batches: dict[int, dict[str, torch.Tensor]],
        source_width: int,
        selection_sha256: str,
    ) -> None:
        self.layer_index = layer_index
        self.phases = phases
        self.split_by_batch = dict(split_by_batch)
        self.batches = batches
        self.source_width = source_width
        self.selection_sha256 = selection_sha256
        self.fit_indices = tuple(
            index for index, split in sorted(self.split_by_batch.items())
            if split == "fit" and index in batches
        )
        self.heldout_indices = tuple(
            index for index, split in sorted(self.split_by_batch.items())
            if split != "fit" and index in batches
        )

    def rows(self, indices: Sequence[int]) -> int:
        total = 0
        for index in indices:
            for phase in self.phases:
                total += int(
                    self.batches[index][f"{phase}.residual.pre"].shape[1]
                )
        return total


def load_boundary_evidence(
    session: Any,
    *,
    layer_index: int,
    phases: tuple[str, ...],
    split_by_batch: Mapping[int, str],
    fit_batch_limit: int | None,
    heldout_batch_limit: int | None,
    source_width: int,
) -> BoundaryEvidence:
    # The sealed selective session requires canonical sorted selection names.
    names = tuple(
        sorted(
            f"{phase}.layer.{layer_index:03d}.{suffix}"
            for phase in phases
            for suffix in _SUFFIXES
        )
    )
    ordered_fit = [i for i, s in sorted(split_by_batch.items()) if s == "fit"]
    ordered_heldout = [i for i, s in sorted(split_by_batch.items()) if s != "fit"]
    if fit_batch_limit is not None:
        ordered_fit = ordered_fit[:fit_batch_limit]
    if heldout_batch_limit is not None:
        ordered_heldout = ordered_heldout[:heldout_batch_limit]
    selected = sorted(set(ordered_fit) | set(ordered_heldout))
    batches: dict[int, dict[str, torch.Tensor]] = {}
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode("ascii"))
    for count, index in enumerate(selected):
        payload = session.load_selected_tensors(index, names)
        try:
            digest.update(payload.selection_sha256.encode("ascii"))
            local: dict[str, torch.Tensor] = {}
            for phase in phases:
                for suffix in _SUFFIXES:
                    key = f"{phase}.layer.{layer_index:03d}.{suffix}"
                    value = payload.tensors[key]
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.ndim != 3
                        or int(value.shape[0]) != 1
                        or int(value.shape[2]) != source_width
                        or not bool(torch.isfinite(value).all().item())
                    ):
                        raise Lfm2Layer0StudyError(
                            f"Sealed parent tensor {key!r} has invalid geometry."
                        )
                    local[f"{phase}.{suffix}"] = (
                        value.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                        .clone()
                    )
            batches[index] = local
        finally:
            session.release_selected_tensors(payload)
        if (count + 1) % 32 == 0:
            _log(f"evidence: {count + 1}/{len(selected)} batches loaded")
    return BoundaryEvidence(
        layer_index=layer_index,
        phases=phases,
        split_by_batch={i: split_by_batch[i] for i in selected},
        batches=batches,
        source_width=source_width,
        selection_sha256=digest.hexdigest(),
    )


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def save_evidence_cache(
    directory: Path,
    evidence: BoundaryEvidence,
    sealed_basis: torch.Tensor,
    binding: Mapping[str, Any],
) -> None:
    """Persist the already-authenticated selection so re-runs are cheap.

    Re-opening the sealed selective session content-hashes 85 GB of trace
    payload, which costs ~15 minutes before a single arm can run.  The first
    pass is always the authoritative one: it opens the sealed session, loads
    the sealed balanced-basis phase bound to that session, and writes this
    cache.  A later pass may reuse the cache only after re-deriving the payload
    digest and the sealed chart tensor digest recorded here, and only when the
    trace cache key still matches; it never widens what the first pass was
    allowed to read.
    """

    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, torch.Tensor] = {"__chart__": sealed_basis}
    for index, local in evidence.batches.items():
        for key, value in local.items():
            payload[f"b{index:06d}|{key}"] = value
    payload_path = directory / "payload.pt"
    torch.save(payload, payload_path)
    body = {
        **dict(binding),
        "payload_sha256": _file_sha256(payload_path),
        "sealed_basis_sha256": _tensor_sha256(sealed_basis),
        "selection_sha256": evidence.selection_sha256,
        "split_by_batch": {
            str(k): v for k, v in sorted(evidence.split_by_batch.items())
        },
    }
    _atomic_write_json(directory / "binding.json", body)


def load_evidence_cache(
    directory: Path,
    *,
    trace_cache_key: str,
    layer_index: int,
    phases: tuple[str, ...],
    source_width: int,
) -> tuple[BoundaryEvidence, torch.Tensor, Mapping[str, Any]]:
    binding = json.loads((directory / "binding.json").read_text(encoding="utf-8"))
    payload_path = directory / "payload.pt"
    if _file_sha256(payload_path) != binding["payload_sha256"]:
        raise Lfm2Layer0StudyError("Cached evidence payload digest changed.")
    if (
        binding["trace_cache_key"] != trace_cache_key
        or int(binding["layer_index"]) != layer_index
        or tuple(binding["phases"]) != phases
        or int(binding["source_width"]) != source_width
    ):
        raise Lfm2Layer0StudyError("Cached evidence does not bind this request.")
    payload = torch.load(payload_path, weights_only=True)
    chart = payload.pop("__chart__").to(dtype=torch.float64).contiguous()
    if _tensor_sha256(chart) != binding["sealed_basis_sha256"]:
        raise Lfm2Layer0StudyError("Cached sealed chart digest changed.")
    batches: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in payload.items():
        prefix, name = key.split("|", 1)
        batches.setdefault(int(prefix[1:]), {})[name] = value
    evidence = BoundaryEvidence(
        layer_index=layer_index,
        phases=phases,
        split_by_batch={int(k): v for k, v in binding["split_by_batch"].items()},
        batches=batches,
        source_width=source_width,
        selection_sha256=str(binding["selection_sha256"]),
    )
    return evidence, chart, binding


def nested_chart(
    sealed_basis: torch.Tensor,
    evidence: BoundaryEvidence,
    target_width: int,
) -> torch.Tensor:
    """Return a 1024xW orthonormal chart whose first 816 columns are sealed.

    The width ladder has to isolate width from chart identity.  Rebuilding a
    balanced basis per width would re-canonicalize the subspace and confound
    the two, so the ladder instead *extends* the sealed r12 chart with the
    residual-energy-optimal directions of its own orthogonal complement,
    measured on the same FIT rows.  At the sealed width the chart is returned
    unchanged, so the ladder's bottom rung is bit-identical to the r12 arm, and
    every rung strictly contains the one below it.
    """

    source_width, sealed_width = map(int, sealed_basis.shape)
    if target_width == sealed_width:
        return sealed_basis
    if target_width < sealed_width or target_width > source_width:
        raise Lfm2Layer0StudyError(
            "Nested chart widths must lie in [sealed_width, source_width]."
        )
    gram = torch.zeros((source_width, source_width), dtype=torch.float64)
    for index in evidence.fit_indices:
        for phase in evidence.phases:
            rows = (
                evidence.batches[index][f"{phase}.residual.pre"]
                .to(dtype=torch.float64)
                .reshape(-1, source_width)
            )
            gram += rows.T @ rows
    projector = torch.eye(source_width, dtype=torch.float64) - (
        sealed_basis @ sealed_basis.T
    )
    complement = projector @ gram @ projector
    complement = ((complement + complement.T) * 0.5).contiguous()
    values, vectors = torch.linalg.eigh(complement)
    order = torch.argsort(values, descending=True)
    selected = [sealed_basis[:, column] for column in range(sealed_width)]
    for column in order.tolist():
        if len(selected) == target_width:
            break
        candidate = vectors[:, column].clone()
        for _pass in range(2):
            for existing in selected:
                candidate = candidate - existing * torch.dot(existing, candidate)
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= 1e-8:
            continue
        candidate = candidate / norm
        pivot = int(torch.argmax(candidate.abs()).item())
        if float(candidate[pivot].item()) < 0.0:
            candidate = -candidate
        selected.append(candidate.contiguous())
    if len(selected) != target_width:
        raise Lfm2Layer0StudyError("Nested chart extension ran out of directions.")
    basis = torch.stack(selected, dim=1).contiguous()
    if not torch.allclose(
        basis.T @ basis,
        torch.eye(target_width, dtype=torch.float64),
        rtol=1e-9,
        atol=1e-10,
    ):
        raise Lfm2Layer0StudyError("Nested chart is not orthonormal.")
    if not torch.allclose(basis[:, :sealed_width], sealed_basis, rtol=0.0, atol=0.0):
        raise Lfm2Layer0StudyError("Nested chart did not retain the sealed columns.")
    return basis


class _ChartShim:
    """Minimal dense-coordinate-map view over one orthonormal chart.

    ``Lfm2DynamicClosedLoopAdapter._build_short_conv_assignment`` is the
    production rule that turns a dense chart into the sparse operator-local
    B/C/x transport the short-conv compiler consumes.  It is reused here
    unbound rather than reimplemented, so the r12 arm gets the production
    assignment by construction.
    """

    def __init__(self, basis: torch.Tensor) -> None:
        self._basis = basis.to(dtype=torch.float64).contiguous()
        self.coordinate_map = self

    @property
    def source_width(self) -> int:
        return int(self._basis.shape[0])

    @property
    def target_width(self) -> int:
        return int(self._basis.shape[1])

    def materialize_dense(
        self,
        *,
        max_elements: int,
        dtype: Any,
        device: Any,
    ) -> torch.Tensor:
        if max_elements < int(self._basis.numel()):
            raise Lfm2Layer0StudyError("Chart materialization budget is too small.")
        return self._basis.to(device=device, dtype=dtype).contiguous()

    def encode(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.to(dtype=torch.float64) @ self._basis

    _coordinate_matrix = Lfm2DynamicClosedLoopAdapter._coordinate_matrix
    _build_short_conv_assignment = (
        Lfm2DynamicClosedLoopAdapter._build_short_conv_assignment
    )


def dense_residual_assignment(basis: torch.Tensor) -> Lfm2SparseResidualAssignment:
    return _ChartShim(basis)._build_short_conv_assignment()


# --------------------------------------------------------------------------
# parent weights
# --------------------------------------------------------------------------


def load_layer_short_conv_weights(
    model_directory: Path,
    layer_index: int,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    path = model_directory / "model.safetensors"
    if path.is_symlink() or not path.is_file():
        raise Lfm2Layer0StudyError("Parent model.safetensors is missing.")
    wanted = {
        "in_proj": f"model.layers.{layer_index}.conv.in_proj.weight",
        "conv": f"model.layers.{layer_index}.conv.conv.weight",
        "out_proj": f"model.layers.{layer_index}.conv.out_proj.weight",
        "operator_norm": f"model.layers.{layer_index}.operator_norm.weight",
    }
    values: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for role, key in wanted.items():
            if key not in available:
                raise Lfm2Layer0StudyError(f"Parent tensor {key!r} is missing.")
            values[role] = handle.get_tensor(key).detach().to(
                dtype=torch.float64
            ).contiguous()
    return values


def load_child_short_conv_weights(
    child_directory: Path,
    layer_index: int,
) -> dict[str, torch.Tensor]:
    """Read one layer's short-conv weights from a child package or capsule.

    Accepts either a published child directory or a restart-capsule
    ``child-state.safetensors`` file directly, which is what makes the
    stage-attribution arms possible: the capsule holds the child exactly as the
    per-layer closed-loop boundary solves left it, before the moving-gauge body
    reclosure re-solved every writer.
    """

    from safetensors import safe_open

    path = (
        child_directory
        if child_directory.is_file()
        else child_directory / "model.safetensors"
    )
    if path.is_symlink() or not path.is_file():
        raise Lfm2Layer0StudyError(f"Child state {path} is missing.")
    wanted = {
        "in_proj": f"model.layers.{layer_index}.conv.in_proj.weight",
        "conv": f"model.layers.{layer_index}.conv.conv.weight",
        "out_proj": f"model.layers.{layer_index}.conv.out_proj.weight",
        "operator_norm": f"model.layers.{layer_index}.operator_norm.weight",
    }
    values: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for role, key in wanted.items():
            if key not in available:
                raise Lfm2Layer0StudyError(f"Child tensor {key!r} is missing.")
            values[role] = handle.get_tensor(key).detach().to(
                dtype=torch.float64
            ).contiguous()
    return values


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------


def _rms_materialize(
    rows: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    normalized = rows * torch.rsqrt(
        rows.square().mean(dim=-1, keepdim=True) + epsilon
    )
    return (normalized * weight).contiguous()


def _build_cases(
    evidence: BoundaryEvidence,
    indices: Sequence[int],
    *,
    basis: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_epsilon: float,
    layer_index: int,
) -> tuple[list[Any], list[tuple[int, str]]]:
    cases: list[Any] = []
    origins: list[tuple[int, str]] = []
    for index in indices:
        local = evidence.batches[index]
        for phase in evidence.phases:
            pre = local[f"{phase}.residual.pre"].to(dtype=torch.float64)
            child_pre = pre @ basis
            target_input = _rms_materialize(child_pre, norm_weight, norm_epsilon)
            endpoint = local[f"{phase}.residual.post_operator"].to(
                dtype=torch.float64
            ) @ basis
            tensors = {
                f"{phase}.layer.{layer_index:03d}.short_conv.b": local[
                    f"{phase}.short_conv.b"
                ].to(dtype=torch.float64),
                f"{phase}.layer.{layer_index:03d}.short_conv.c": local[
                    f"{phase}.short_conv.c"
                ].to(dtype=torch.float64),
                f"{phase}.layer.{layer_index:03d}.short_conv.x": local[
                    f"{phase}.short_conv.x"
                ].to(dtype=torch.float64),
                f"{phase}.layer.{layer_index:03d}.short_conv.context": local[
                    f"{phase}.short_conv.context"
                ].to(dtype=torch.float64),
            }
            cases.append(
                cached_short_conv_case_from_lfm2_trace(
                    tensors,
                    phase=phase,
                    layer_index=layer_index,
                    target_input=target_input,
                    parent_post_operator_residual_target_space=endpoint,
                    target_pre_operator_residual=child_pre,
                )
            )
            origins.append((index, phase))
    return cases, origins


def _fit_operator_norm(
    evidence: BoundaryEvidence,
    *,
    basis: torch.Tensor,
    norm_epsilon: float,
    norm_ridge: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    inputs = []
    targets = []
    validation_inputs = []
    validation_targets = []
    for index in evidence.fit_indices + evidence.heldout_indices:
        bucket_inputs = inputs if index in evidence.fit_indices else validation_inputs
        bucket_targets = (
            targets if index in evidence.fit_indices else validation_targets
        )
        local = evidence.batches[index]
        for phase in evidence.phases:
            pre = local[f"{phase}.residual.pre"].to(dtype=torch.float64)
            bucket_inputs.append((pre @ basis).reshape(-1, int(basis.shape[1])))
            bucket_targets.append(
                (
                    local[f"{phase}.operator.input"].to(dtype=torch.float64) @ basis
                ).reshape(-1, int(basis.shape[1]))
            )
    fit = fit_diagonal_norm(
        torch.cat(inputs, dim=0),
        torch.cat(targets, dim=0),
        kind="rms_norm",
        epsilon=norm_epsilon,
        has_bias=False,
        ridge=norm_ridge,
        validation_input_rows=torch.cat(validation_inputs, dim=0),
        validation_target_rows=torch.cat(validation_targets, dim=0),
    )
    weight = fit.weight.detach().to(dtype=torch.float64).contiguous()
    report = {
        "epsilon": norm_epsilon,
        "ridge": norm_ridge,
        "fit_relative_rmse": float(fit.train_metrics.relative_rmse),
        "validation_relative_rmse": (
            None
            if fit.validation_metrics is None
            else float(fit.validation_metrics.relative_rmse)
        ),
        "weight_sha256": _tensor_sha256(weight),
    }
    return weight, report


def _measure(
    contexts: Sequence[torch.Tensor],
    origins: Sequence[tuple[int, str]],
    cases: Sequence[Any],
    out_weight: torch.Tensor,
    out_bias: torch.Tensor | None,
) -> dict[str, Any]:
    write_rows = []
    target_rows = []
    post_rows = []
    post_target_rows = []
    last_write = []
    last_target = []
    last_post = []
    last_post_target = []
    for context, case in zip(contexts, cases, strict=True):
        width = int(context.shape[-1])
        flat = context.reshape(-1, width)
        write = flat @ out_weight.T
        if out_bias is not None:
            write = write + out_bias
        pre = case.target_pre_operator_residual.reshape(-1, int(out_weight.shape[0]))
        endpoint = case.parent_post_operator_residual_target_space.reshape(
            -1, int(out_weight.shape[0])
        )
        target_write = endpoint - pre
        write_rows.append(write)
        target_rows.append(target_write)
        post_rows.append(pre + write)
        post_target_rows.append(endpoint)
        last_write.append(write[-1:])
        last_target.append(target_write[-1:])
        last_post.append((pre + write)[-1:])
        last_post_target.append(endpoint[-1:])
    del origins
    return {
        "all_rows": {
            "operator.output": _row_metrics(
                torch.cat(write_rows, dim=0), torch.cat(target_rows, dim=0)
            ),
            "residual.post_operator": _row_metrics(
                torch.cat(post_rows, dim=0), torch.cat(post_target_rows, dim=0)
            ),
        },
        "last_token": {
            "operator.output": _row_metrics(
                torch.cat(last_write, dim=0), torch.cat(last_target, dim=0)
            ),
            "residual.post_operator": _row_metrics(
                torch.cat(last_post, dim=0), torch.cat(last_post_target, dim=0)
            ),
        },
    }


def _prior_report(report: Any) -> dict[str, Any]:
    return {
        "role": report.role,
        "rows": report.rows,
        "input_width": report.input_width,
        "output_width": report.output_width,
        "prior_provided": report.prior_provided,
        "prior_norm": report.prior_norm,
        "correction_norm": report.correction_norm,
        "update_scale": report.update_scale,
        "clamped": report.update_scale < 1.0,
        "underdetermined": report.underdetermined,
    }


def run_arm(
    *,
    name: str,
    evidence: BoundaryEvidence,
    sealed_basis: torch.Tensor,
    parent_weights: Mapping[str, torch.Tensor],
    target_width: int,
    ridge: float,
    max_prior_delta_ratio: float | None,
    prior_mode: str,
    norm_epsilon: float,
    norm_ridge: float,
    chart_cache: dict[int, torch.Tensor],
    layer_index: int,
    solve_dtype: str,
    prior_child_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    _log(f"arm {name}: begin (width={target_width}, prior={prior_mode})")
    if target_width not in chart_cache:
        chart_cache[target_width] = nested_chart(
            sealed_basis, evidence, target_width
        )
    basis = chart_cache[target_width]
    norm_weight, norm_report = _fit_operator_norm(
        evidence,
        basis=basis,
        norm_epsilon=norm_epsilon,
        norm_ridge=norm_ridge,
    )
    fit_cases, fit_origins = _build_cases(
        evidence,
        evidence.fit_indices,
        basis=basis,
        norm_weight=norm_weight,
        norm_epsilon=norm_epsilon,
        layer_index=layer_index,
    )
    heldout_cases, heldout_origins = _build_cases(
        evidence,
        evidence.heldout_indices,
        basis=basis,
        norm_weight=norm_weight,
        norm_epsilon=norm_epsilon,
        layer_index=layer_index,
    )
    source_channels = int(parent_weights["conv"].shape[0])
    channel_assignment = Lfm2SparseResidualAssignment(
        target_width=source_channels,
        source_to_target=tuple(range(source_channels)),
        coefficients=(1.0,) * source_channels,
    )
    external_assignment = dense_residual_assignment(basis)
    template_dtype = torch.float32 if solve_dtype == "float32" else torch.float64
    in_template = torch.zeros(
        (3 * source_channels, target_width), dtype=template_dtype
    )
    conv_template = torch.zeros(
        (source_channels, 1, int(parent_weights["conv"].shape[2])),
        dtype=template_dtype,
    )
    out_template = torch.zeros(
        (target_width, source_channels), dtype=template_dtype
    )

    # The production consumer anchors on the live child out_proj, which the
    # structural pass had already set to the source-derived value the producer
    # compiler derives internally.  ``compile_..._producer_fold`` derives its
    # own in_proj prior when none is supplied, but the consumer compiler has no
    # derivation of its own, so the source-derived arm has to rebuild exactly
    # the ``residual_reduce @ source_out @ channel_lift`` prior the structural
    # fold used.  ``channel_lift`` is the identity here because this boundary's
    # internal channel width is not contracted.
    reduce_matrix = torch.zeros(
        (target_width, int(basis.shape[0])), dtype=torch.float64
    )
    for source, target in enumerate(external_assignment.source_to_target):
        reduce_matrix[target, source] = external_assignment.coefficients[source]
    derived_out_prior = (reduce_matrix @ parent_weights["out_proj"]).contiguous()

    in_prior = None
    out_prior = derived_out_prior
    if prior_mode == "child_state":
        # Anchor on a real pipeline state -- the restart capsule's own weights,
        # for instance -- to ask whether that state is already the closed-form
        # optimum or merely a point the production run could not leave.
        if prior_child_state is None:
            raise Lfm2Layer0StudyError(
                "prior_mode 'child_state' needs --prior-child-state."
            )
        in_prior = prior_child_state["in_proj"].to(dtype=torch.float64)
        out_prior = prior_child_state["out_proj"].to(dtype=torch.float64)
    elif prior_mode == "dense_chart":
        blocks = parent_weights["in_proj"].chunk(3, dim=0)
        in_prior = torch.cat([block @ basis for block in blocks], dim=0).contiguous()
        out_prior = (basis.T @ parent_weights["out_proj"]).contiguous()
    elif prior_mode == "none":
        in_prior = torch.zeros((3 * source_channels, target_width), dtype=torch.float64)
        out_prior = torch.zeros((target_width, source_channels), dtype=torch.float64)
    elif prior_mode != "source_derived":
        raise Lfm2Layer0StudyError(f"Unknown prior_mode {prior_mode!r}.")

    producer = compile_lfm2_structural_short_conv_producer_fold(
        source_in_proj_weight=parent_weights["in_proj"],
        source_conv_weight=parent_weights["conv"],
        source_out_proj_weight=parent_weights["out_proj"],
        target_in_proj_weight_template=in_template,
        target_conv_weight_template=conv_template,
        target_out_proj_weight_template=out_template,
        cases=tuple(fit_cases),
        residual_assignment=channel_assignment,
        external_residual_assignment=external_assignment,
        in_proj_weight_prior=in_prior,
        out_proj_weight_prior=out_prior,
        ridge=ridge,
        max_prior_delta_ratio=max_prior_delta_ratio,
    )
    in_weight = producer.in_proj_weight.to(dtype=torch.float64)
    conv_weight = producer.conv_weight.to(dtype=torch.float64)

    def contexts_for(cases: Sequence[Any]) -> list[torch.Tensor]:
        normalized = [
            {
                "target_input": case.target_input,
                "reset_state": case.reset_state,
            }
            for case in cases
        ]
        return fold_module._run_compiled_short_conv_contexts(
            normalized,
            in_proj_weight=in_weight,
            in_proj_bias=None,
            conv_weight=conv_weight,
            conv_bias=None,
        )

    fit_contexts = contexts_for(fit_cases)
    consumer = compile_lfm2_structural_short_conv_consumer_fold(
        target_out_proj_weight_template=out_template,
        actual_child_contexts=tuple(fit_contexts),
        cases=tuple(fit_cases),
        out_proj_weight_prior=out_prior,
        ridge=ridge,
        max_prior_delta_ratio=max_prior_delta_ratio,
    )
    out_weight = consumer.out_proj_weight.to(dtype=torch.float64)
    heldout_contexts = contexts_for(heldout_cases)
    metrics = {
        "heldout": _measure(
            heldout_contexts, heldout_origins, heldout_cases, out_weight, None
        ),
        "fit": _measure(fit_contexts, fit_origins, fit_cases, out_weight, None),
    }
    record = {
        "arm": name,
        "layer_index": layer_index,
        "target_width": target_width,
        "source_width": evidence.source_width,
        "internal_channel_width": source_channels,
        "prior_mode": prior_mode,
        "ridge": ridge,
        "max_prior_delta_ratio": max_prior_delta_ratio,
        "solve_dtype": solve_dtype,
        "norm_fit": norm_report,
        "external_residual_assignment_sha256": (
            external_assignment.assignment_sha256
        ),
        "producer": {
            "compiled_state_sha256": producer.compiled_state_sha256,
            "in_proj_fit": _prior_report(producer.prior_correction_report),
        },
        "consumer": {
            "compiled_state_sha256": consumer.compiled_state_sha256,
            "fit_row_relative_rmse": consumer.residual_endpoint_relative_rmse,
            "out_proj_fit": _prior_report(consumer.prior_correction_report),
        },
        "fit_batches": len(evidence.fit_indices),
        "heldout_batches": len(evidence.heldout_indices),
        "metrics": metrics,
        "parent_forward_calls": 0,
        "child_forward_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    heldout_write = metrics["heldout"]["last_token"]["operator.output"]
    heldout_post = metrics["heldout"]["last_token"]["residual.post_operator"]
    _log(
        f"arm {name}: last-token heldout write cos="
        f"{heldout_write['mean_cosine']:.4f} relL2="
        f"{heldout_write['mean_relative_l2']:.4f} | post_operator relL2="
        f"{heldout_post['mean_relative_l2']:.4f} | in_scale="
        f"{producer.prior_correction_report.update_scale:.4g} out_scale="
        f"{consumer.prior_correction_report.update_scale:.4g}"
    )
    return record


def run_published_child_arm(
    *,
    evidence: BoundaryEvidence,
    basis: torch.Tensor,
    child_weights: Mapping[str, torch.Tensor],
    layer_index: int,
    norm_epsilon: float,
    name: str = "published_child",
) -> dict[str, Any]:
    """Re-measure the sealed published child's own layer-0 write.

    This arm runs no solve.  It exists to prove the harness -- row selection,
    chart projection, RMSNorm semantics, short-conv semantics, and the metric
    -- reproduces the sealed first-divergence ledger before any solve result
    is believed.
    """

    _log(f"arm {name}: begin")
    target_width = int(basis.shape[1])
    heldout_cases, heldout_origins = _build_cases(
        evidence,
        evidence.heldout_indices,
        basis=basis,
        norm_weight=child_weights["operator_norm"],
        norm_epsilon=norm_epsilon,
        layer_index=layer_index,
    )
    fit_cases, fit_origins = _build_cases(
        evidence,
        evidence.fit_indices,
        basis=basis,
        norm_weight=child_weights["operator_norm"],
        norm_epsilon=norm_epsilon,
        layer_index=layer_index,
    )

    def contexts_for(cases: Sequence[Any]) -> list[torch.Tensor]:
        # The sealed cache stores decode phases as single rows, so the FIR
        # history has to carry across phases exactly the way the production
        # case adapter declares it.  Resetting per phase would run every decode
        # row against an empty filter window and understate the child badly.
        normalized = [
            {"target_input": case.target_input, "reset_state": case.reset_state}
            for case in cases
        ]
        return fold_module._run_compiled_short_conv_contexts(
            normalized,
            in_proj_weight=child_weights["in_proj"],
            in_proj_bias=None,
            conv_weight=child_weights["conv"],
            conv_bias=None,
        )

    out_weight = child_weights["out_proj"]
    metrics = {
        "heldout": _measure(
            contexts_for(heldout_cases),
            heldout_origins,
            heldout_cases,
            out_weight,
            None,
        ),
        "fit": _measure(
            contexts_for(fit_cases), fit_origins, fit_cases, out_weight, None
        ),
    }
    write = metrics["heldout"]["last_token"]["operator.output"]
    post = metrics["heldout"]["last_token"]["residual.post_operator"]
    _log(
        f"arm {name}: last-token heldout write cos="
        f"{write['mean_cosine']:.4f} relL2={write['mean_relative_l2']:.4f} "
        f"norm_ratio={write['mean_norm_ratio']:.4f} | post_operator relL2="
        f"{post['mean_relative_l2']:.4f}"
    )
    return {
        "arm": name,
        "layer_index": layer_index,
        "target_width": target_width,
        "solve": None,
        "metrics": metrics,
        "parent_forward_calls": 0,
        "child_forward_calls": 0,
        "training_steps": 0,
    }


# --------------------------------------------------------------------------
# arm matrix
# --------------------------------------------------------------------------


def default_arms(sealed_width: int) -> tuple[dict[str, Any], ...]:
    arms: list[dict[str, Any]] = [
        # 1. reproduction of the r12 production configuration
        {
            "name": "r12_reproduction",
            "target_width": sealed_width,
            "ridge": 1e-6,
            "max_prior_delta_ratio": 1.0,
            "prior_mode": "source_derived",
        },
        # 3. solver levers at r12's width
        {
            "name": "solver_prior_delta_unclamped",
            "target_width": sealed_width,
            "ridge": 1e-6,
            "max_prior_delta_ratio": None,
            "prior_mode": "source_derived",
        },
        {
            "name": "solver_prior_delta_4",
            "target_width": sealed_width,
            "ridge": 1e-6,
            "max_prior_delta_ratio": 4.0,
            "prior_mode": "source_derived",
        },
        {
            "name": "solver_prior_delta_16",
            "target_width": sealed_width,
            "ridge": 1e-6,
            "max_prior_delta_ratio": 16.0,
            "prior_mode": "source_derived",
        },
        {
            "name": "solver_chart_priors_clamped",
            "target_width": sealed_width,
            "ridge": 1e-6,
            "max_prior_delta_ratio": 1.0,
            "prior_mode": "dense_chart",
        },
        {
            "name": "solver_chart_priors_unclamped",
            "target_width": sealed_width,
            "ridge": 1e-6,
            "max_prior_delta_ratio": None,
            "prior_mode": "dense_chart",
        },
        {
            "name": "solver_zero_prior_unclamped",
            "target_width": sealed_width,
            "ridge": 1e-6,
            "max_prior_delta_ratio": None,
            "prior_mode": "none",
        },
    ]
    for ridge in (0.0, 1e-8, 1e-4, 1e-2):
        arms.append(
            {
                "name": f"solver_ridge_{ridge:g}",
                "target_width": sealed_width,
                "ridge": ridge,
                "max_prior_delta_ratio": None,
                "prior_mode": "dense_chart",
            }
        )
    # 2. width ladder at the r12 solver and at the best-known solver
    for width in (864, 912, 960, 1008, 1023):
        arms.append(
            {
                "name": f"width_{width}_r12_solver",
                "target_width": width,
                "ridge": 1e-6,
                "max_prior_delta_ratio": 1.0,
                "prior_mode": "source_derived",
            }
        )
        arms.append(
            {
                "name": f"width_{width}_free_solver",
                "target_width": width,
                "ridge": 1e-6,
                "max_prior_delta_ratio": None,
                "prior_mode": "dense_chart",
            }
        )
    return tuple(arms)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-cache", type=Path, required=True)
    parser.add_argument("--basis-phase-cache-root", type=Path, required=True)
    parser.add_argument("--basis-phase-key-sha256", required=True)
    parser.add_argument("--basis-phase-hmac-key-file", type=Path, required=True)
    parser.add_argument("--basis-phase-hmac-key-id", required=True)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--published-child-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--target-hidden-size", type=int, default=816)
    parser.add_argument("--forced-decode-count", type=int, default=8)
    parser.add_argument("--fit-batch-limit", type=int)
    parser.add_argument("--heldout-batch-limit", type=int)
    parser.add_argument("--norm-epsilon", type=float, default=1e-5)
    parser.add_argument("--norm-ridge", type=float, default=1e-6)
    parser.add_argument("--arms", type=Path)
    parser.add_argument("--mode", choices=("inventory", "study"), default="study")
    parser.add_argument("--skip-published-child", action="store_true")
    parser.add_argument("--evidence-cache", type=Path)
    # Stage attribution: NAME=PATH, where PATH is a child package directory or a
    # restart-capsule child-state.safetensors.  Each one is measured exactly the
    # way ``published_child`` is, so the difference between two pipeline stages
    # is read off the same metric.
    parser.add_argument("--extra-child-state", action="append", default=[])
    parser.add_argument("--prior-child-state", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    torch.set_grad_enabled(False)
    output = Path(args.output_directory).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise Lfm2Layer0StudyError("Output directory cannot be a symlink.")
    phases = phase_names(args.forced_decode_count)
    _log(f"phases: {phases}")

    trace_cache = open_lfm2_circuit_trace_cache(
        Path(args.trace_cache).expanduser().absolute(), verify=False
    )
    manifest = trace_cache.manifest
    topology = manifest.get("layer_topology")
    layers = topology.get("layers") if isinstance(topology, Mapping) else None
    if not isinstance(layers, list) or not layers:
        raise Lfm2Layer0StudyError("Trace cache has no layer topology.")
    widths = {
        int(layer["contract"]["hidden_size"])
        for layer in layers
        if isinstance(layer, Mapping) and isinstance(layer.get("contract"), Mapping)
    }
    if len(widths) != 1:
        raise Lfm2Layer0StudyError("Trace topology has no single residual width.")
    source_width = int(next(iter(widths)))
    _log(f"trace cache {trace_cache.cache_key} batches={len(trace_cache)} "
         f"source_width={source_width}")

    cache_directory = (
        None
        if args.evidence_cache is None
        else Path(args.evidence_cache).expanduser().absolute()
    )
    basis_provenance: dict[str, Any] = {}
    if cache_directory is not None and (cache_directory / "binding.json").is_file():
        _log(f"reusing the authenticated evidence cache at {cache_directory}")
        evidence, sealed_basis, binding = load_evidence_cache(
            cache_directory,
            trace_cache_key=trace_cache.cache_key,
            layer_index=int(args.layer_index),
            phases=phases,
            source_width=source_width,
        )
        basis_provenance = {
            "basis_map_sha256": binding["basis_map_sha256"],
            "basis_completion_sha256": binding["basis_completion_sha256"],
            "trace_selective_session_sha256": binding[
                "trace_selective_session_sha256"
            ],
            "evidence_cache_reused": True,
        }
        _log(
            f"cache bound: chart {tuple(sealed_basis.shape)} "
            f"map {binding['basis_map_sha256']}"
        )
        return _study(
            args=args,
            output=output,
            evidence=evidence,
            sealed_basis=sealed_basis,
            phases=phases,
            source_width=source_width,
            trace_cache_key=trace_cache.cache_key,
            basis_provenance=basis_provenance,
        )

    secret = _read_raw_hmac_key(args.basis_phase_hmac_key_file)
    try:
        authority = PhaseAnchorAuthority(
            key_id=args.basis_phase_hmac_key_id, secret=secret
        )
    finally:
        del secret

    _log("opening the sealed selective trace session (hashes every batch)")
    with trace_cache.selective_authenticated_session() as session:
        _log(f"session sealed: {session.evidence_sha256}")
        basis_phase = load_existing_lfm2_circuit_balanced_basis_phase(
            DynamicFoldPhaseCache(
                Path(args.basis_phase_cache_root).expanduser().absolute()
            ),
            _digest(args.basis_phase_key_sha256, "basis_phase_key_sha256"),
            anchor_authority=authority,
            expected_parent_identity_sha256=trace_cache.parent_fingerprint,
            expected_trace_cache_key=trace_cache.cache_key,
            expected_trace_selective_session_sha256=session.evidence_sha256,
            expected_source_width=source_width,
            expected_target_width=int(args.target_hidden_size),
        )
        sealed_basis = (
            basis_phase.basis_result.basis.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        )
        _log(
            f"sealed chart {basis_phase.map_sha256} "
            f"{tuple(sealed_basis.shape)}"
        )
        references = session.batch_references
        split_by_batch = authoritative_split_by_batch(basis_phase.report, references)
        fit_total = sum(1 for value in split_by_batch.values() if value == "fit")
        _log(
            f"authoritative split: {fit_total} fit / "
            f"{len(split_by_batch) - fit_total} heldout"
        )
        evidence = load_boundary_evidence(
            session,
            layer_index=int(args.layer_index),
            phases=phases,
            split_by_batch=split_by_batch,
            fit_batch_limit=args.fit_batch_limit,
            heldout_batch_limit=args.heldout_batch_limit,
            source_width=source_width,
        )
        session.verify()
        session_sha256 = session.evidence_sha256
    basis_provenance = {
        "basis_map_sha256": basis_phase.map_sha256,
        "basis_completion_sha256": basis_phase.completion_sha256,
        "trace_selective_session_sha256": session_sha256,
        "evidence_cache_reused": False,
    }
    if cache_directory is not None:
        _log(f"writing the authenticated evidence cache to {cache_directory}")
        save_evidence_cache(
            cache_directory,
            evidence,
            sealed_basis,
            {
                "schema": SCHEMA,
                "kind": "evidence-cache-binding",
                "trace_cache_key": trace_cache.cache_key,
                "parent_identity_sha256": trace_cache.parent_fingerprint,
                "layer_index": int(args.layer_index),
                "phases": list(phases),
                "source_width": source_width,
                "sealed_target_width": int(args.target_hidden_size),
                **basis_provenance,
            },
        )
    return _study(
        args=args,
        output=output,
        evidence=evidence,
        sealed_basis=sealed_basis,
        phases=phases,
        source_width=source_width,
        trace_cache_key=trace_cache.cache_key,
        basis_provenance=basis_provenance,
    )


def _study(
    *,
    args: argparse.Namespace,
    output: Path,
    evidence: BoundaryEvidence,
    sealed_basis: torch.Tensor,
    phases: tuple[str, ...],
    source_width: int,
    trace_cache_key: str,
    basis_provenance: Mapping[str, Any],
) -> int:
    inventory = {
        "schema": SCHEMA,
        "kind": "evidence-inventory",
        "layer_index": int(args.layer_index),
        "phases": list(phases),
        "source_width": source_width,
        "sealed_target_width": int(args.target_hidden_size),
        "fit_batches": len(evidence.fit_indices),
        "heldout_batches": len(evidence.heldout_indices),
        "fit_rows": evidence.rows(evidence.fit_indices),
        "heldout_rows": evidence.rows(evidence.heldout_indices),
        "selection_sha256": evidence.selection_sha256,
        "trace_cache_key": trace_cache_key,
        "sealed_basis_sha256": _tensor_sha256(sealed_basis),
        **dict(basis_provenance),
        "implementation": {
            "study_runner_sha256": hashlib.sha256(
                Path(__file__).resolve().read_bytes()
            ).hexdigest(),
            "structural_fold_sha256": _module_sha256(fold_module),
            "balanced_basis_phase_sha256": _module_sha256(basis_phase_module),
            "circuit_trace_cache_sha256": _module_sha256(trace_module),
        },
    }
    _atomic_write_json(output / "evidence-inventory.json", inventory)
    _log(json.dumps({k: v for k, v in inventory.items() if k != "implementation"}))
    if args.mode == "inventory":
        return 0

    parent_weights = load_layer_short_conv_weights(
        Path(args.parent_model).expanduser().absolute(), int(args.layer_index)
    )
    _log(
        "parent layer weights: "
        + json.dumps({k: list(v.shape) for k, v in parent_weights.items()})
    )
    records: list[dict[str, Any]] = []
    if args.published_child_directory is not None and not args.skip_published_child:
        child_weights = load_child_short_conv_weights(
            Path(args.published_child_directory).expanduser().absolute(),
            int(args.layer_index),
        )
        records.append(
            run_published_child_arm(
                evidence=evidence,
                basis=sealed_basis,
                child_weights=child_weights,
                layer_index=int(args.layer_index),
                norm_epsilon=float(args.norm_epsilon),
            )
        )
        _atomic_write_json(
            output / "arm-published_child.json", records[-1]
        )
    for entry in args.extra_child_state:
        name, _, raw = str(entry).partition("=")
        if not name or not raw:
            raise Lfm2Layer0StudyError("--extra-child-state needs NAME=PATH.")
        destination = output / f"arm-{name}.json"
        if destination.exists():
            _log(f"arm {name}: already sealed, reusing")
            records.append(json.loads(destination.read_text(encoding="utf-8")))
            continue
        record = run_published_child_arm(
            evidence=evidence,
            basis=sealed_basis,
            child_weights=load_child_short_conv_weights(
                Path(raw).expanduser().absolute(), int(args.layer_index)
            ),
            layer_index=int(args.layer_index),
            norm_epsilon=float(args.norm_epsilon),
            name=name,
        )
        record["child_state_path"] = raw
        _atomic_write_json(destination, record)
        records.append(record)

    if args.arms is not None:
        specs = json.loads(Path(args.arms).read_text(encoding="utf-8"))
    else:
        specs = list(default_arms(int(args.target_hidden_size)))
    prior_child_state = (
        None
        if args.prior_child_state is None
        else load_child_short_conv_weights(
            Path(args.prior_child_state).expanduser().absolute(),
            int(args.layer_index),
        )
    )
    chart_cache: dict[int, torch.Tensor] = {
        int(args.target_hidden_size): sealed_basis
    }
    for spec in specs:
        name = str(spec["name"])
        destination = output / f"arm-{name}.json"
        if destination.exists():
            _log(f"arm {name}: already sealed, reusing")
            records.append(json.loads(destination.read_text(encoding="utf-8")))
            continue
        try:
            record = run_arm(
                name=name,
                evidence=evidence,
                sealed_basis=sealed_basis,
                parent_weights=parent_weights,
                target_width=int(spec.get("target_width", args.target_hidden_size)),
                ridge=float(spec.get("ridge", 1e-6)),
                max_prior_delta_ratio=spec.get("max_prior_delta_ratio", 1.0),
                prior_mode=str(spec.get("prior_mode", "source_derived")),
                norm_epsilon=float(args.norm_epsilon),
                norm_ridge=float(args.norm_ridge),
                chart_cache=chart_cache,
                layer_index=int(args.layer_index),
                solve_dtype=str(spec.get("solve_dtype", "float64")),
                prior_child_state=prior_child_state,
            )
        except Exception as exc:  # noqa: BLE001 - arms fail independently
            record = {
                "arm": name,
                "failed": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _log(f"arm {name}: FAILED {type(exc).__name__}: {exc}")
        record["evidence_selection_sha256"] = evidence.selection_sha256
        record["sealed_basis_map_sha256"] = basis_provenance["basis_map_sha256"]
        _atomic_write_json(destination, record)
        records.append(record)

    summary = {
        "schema": SCHEMA,
        "kind": "study-summary",
        "generated_utc": _now(),
        "inventory": inventory,
        "arms": records,
    }
    summary["summary_sha256"] = canonical_sha256(
        {"schema": SCHEMA, "arms": [r.get("arm") for r in records]}
    )
    _atomic_write_json(output / "study-summary.json", summary)
    _log(f"complete: {len(records)} arms -> {output}")
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
