#!/usr/bin/env python3
"""C10: drop attention heads and absorb them into the O projection.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §1.1 gives C10 the contraction axis
``heads`` and §1.1's level table spells it out -- "for C10 the levels are head
drops (-1 Q head / -1 KV head / -4 Q heads / 4x-layout head plan)".  Like C04 it
has never produced a number: wave 1 sealed `C10.T1.M16.L02` ``infra`` on a
calibration relL2 of 4.4955 (the generic gate multiplied ``operator.input`` by
``q_proj`` and compared it to the operator's output, which is a category error
that happens to typecheck), and wave 2 sealed it ``infra`` on ``T1's reference
kernel contracts linear maps; C10 has axis 'heads'`` (`NOTES-MX-F` §4).  This
module is the missing kernel.

The mechanism is the one the estate's per-head machinery already implies.  An
attention operator's write is

    o = context @ out_proj.T,   context = [head_0 | head_1 | ... ] (H * D wide)

so a head is a *contiguous block of columns* of ``out_proj`` and a contiguous
block of rows of ``q_proj``.  Dropping head ``j`` therefore means:

* ``out_proj[:, j*D:(j+1)*D] = 0`` -- the head no longer reaches the residual;
* ``q_proj[j*D:(j+1)*D, :] = 0`` -- the head's query is no longer stored (its
  context still computes, uniformly, and is multiplied by the zeroed columns,
  so it cannot reach anything);
* the surviving columns of ``out_proj`` are **re-solved by least squares
  against the parent's own write**, so the dropped head's contribution is
  absorbed by the heads that remain instead of simply deleted.

That last step is the whole technique.  Deleting a head is a 1-line edit and a
guaranteed loss; absorbing it asks whether the remaining heads' context span
already contains what the dropped head was writing.  The answer is a number
this campaign has never had.

Three properties worth knowing before reading a receipt.

1. **The parent relation is exact and is checked.**  Unlike every other C10
   formulation, ``o = context @ out_proj.T`` is a genuine single matmul from a
   sealed endpoint to a sealed endpoint, so this kernel *can* run the
   calibration the generic gate could not: it seals
   ``parent_matmul_relative_l2`` and refuses the cell if the parent does not
   reproduce its own write.  `run_lfm2_matrix_cell.NO_PARENT_MATMUL_PREDICTOR`
   still declares the harness's generic gate inapplicable for C10 -- correctly,
   since it reads ``weight_keys[0]``, which is ``q_proj`` -- so the receipt
   keeps that declaration and carries the real measurement under ``solve``.

2. **The identity gate is measured, not asserted.**  At zero drop the kept set
   is every head and the solve is the full-rank prior-anchored fit; run against
   a parent-derived target it must return ``out_proj`` exactly.  Every receipt
   seals ``identity_gate_max_abs_deviation``, computed that way, beside the
   real fit against the sealed rows.

3. **Storage actually contracts, which is new.**  `NOTES-MX-E` §4.1's trap is
   that a ``w x r`` basis plus an ``r x w`` factor stores ``r*(in+out)`` and so
   *costs* storage at every marginal level above M2 -- every T1/T2/T3 cell in
   this campaign stores more than the tensor it replaced.  A dropped head is
   not a factorization: it removes ``D`` rows of ``q_proj`` and ``D`` columns of
   ``out_proj`` outright.  One head of sixteen stores 0.958x the four-tensor
   parent group, two heads 0.917x, and nothing is stored twice.

The level axis needs no special reading here, unlike C04's.  ``own_width`` for
C10 is ``q_proj``'s output width, ``H * D``, and the marginal family lands on
head boundaries exactly: M16 of 1024 is 960 = 15 heads (drop one), M8 is 896 =
14 heads (drop two), M4 is 768 = 12 heads (drop four -- the design's own third
rung).  The kernel requires that divisibility rather than rounding to it, so a
level that does not name a whole number of heads refuses instead of silently
contracting something else.

GQA is respected but not exploited.  LFM2 pairs 16 query heads with 8 KV heads,
so dropping query heads can in principle orphan a KV group; the kernel reports
any group whose queries are all gone under ``dead_kv_groups`` and does **not**
count the ``k_proj``/``v_proj`` rows it could then drop as saved storage.
Claiming a saving the installed child does not take would be exactly the
storage dishonesty §4b exists to catch.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "C10_COMPONENT_ID",
    "c10_head_drop_technique",
    "describe_head_drop_technique",
]

C10_COMPONENT_ID = "C10"

#: Design §4.1 row 29 is ``C10.T1.M16.L02`` -- "drop one Q head (never
#: measured)" -- and both sealed C10 instances carry that id.  Registering under
#: a new technique id would orphan them from their successor, so this kernel is
#: T1 pinned to C10, the same shape as T1-at-C14.
C10_TECHNIQUE_ID = "T1"

#: Enumerating every drop set is exact and cheap while the count is small
#: (16 for one head, 120 for two, 1,820 for four).  Past the cap the kernel
#: falls back to a greedy sequential search and says so in the receipt rather
#: than spending a 20-minute cell cap on a combinatorial sweep.
_MAX_ENUMERATED_DROP_SETS = 512


def _torch() -> Any:
    import torch

    return torch


def _head_dimension(request: Any, spec: Any, output_width: int) -> int:
    """The attention head width, read off the parent rather than assumed.

    LFM2 carries a per-head-dim RMSNorm gain (``self_attn.q_layernorm.weight``)
    whose length *is* ``head_dim``, so the geometry comes from the checkpoint.
    The knob exists for a parent that has no such tensor; it is never used to
    override one that does, because a head width that disagrees with the
    checkpoint would silently reindex every block this kernel slices.
    """

    from .matrix_technique_registry import CellTechniqueError

    key = f"model.layers.{int(spec.layer_index)}.self_attn.q_layernorm.weight"
    gain = request.parent_weights.get(key)
    if gain is not None:
        head_dim = int(gain.shape[0])
    else:
        declared = request.knob("head_dim", None)
        if declared is None:
            raise CellTechniqueError(
                f"C10 cannot resolve the head width: {key} is absent and no "
                "head_dim knob was supplied."
            )
        head_dim = int(declared)
    if head_dim <= 0 or output_width % head_dim != 0:
        raise CellTechniqueError(
            f"C10 resolved head_dim {head_dim}, which does not divide the "
            f"{output_width}-wide query projection."
        )
    return head_dim


def _residual_after_drop(
    gram: Any,
    cross: Any,
    target_energy: float,
    kept_channels: Any,
    ridge: float,
) -> float:
    """Least-squares residual of ``Y ~ X[:, kept]``, from the full normal system.

    Scoring a candidate drop set does not need the rows, only the Gram they
    already produced, so a sweep over drop sets costs one Cholesky solve each
    instead of one pass over the evidence each.
    """

    torch = _torch()
    sub_gram = gram.index_select(0, kept_channels).index_select(1, kept_channels)
    sub_cross = cross.index_select(0, kept_channels)
    system = sub_gram + ridge * torch.eye(
        int(kept_channels.numel()), dtype=sub_gram.dtype
    )
    solved = torch.linalg.solve(system, sub_cross)
    explained = float((sub_cross * solved).sum().item())
    return max(target_energy - explained, 0.0)


def c10_head_drop_technique(request: Any) -> Any:
    """C10: drop ``H - k`` attention heads and re-solve the O projection."""

    from .hf_lfm2_structural_residual_attention_fold import _fit_linear
    from .matrix_technique_registry import (
        CellTechniqueError,
        CellTechniqueResult,
        sealable_report,
        sealable_solve_number,
    )

    torch = _torch()
    spec = request.component
    if spec.component_id != C10_COMPONENT_ID or spec.contraction_axis != "heads":
        raise CellTechniqueError(
            "c10_head_drop_technique is the attn_per_head kernel only; "
            f"{spec.component_id} has axis {spec.contraction_axis!r}."
        )
    if spec.layer_index is None:
        raise CellTechniqueError("C10 needs a layer index to name its tensors.")
    if spec.input_endpoint is None or spec.write_endpoint is None:
        raise CellTechniqueError("C10 has no sealed input/write endpoints.")

    layer = int(spec.layer_index)
    q_key = f"model.layers.{layer}.self_attn.q_proj.weight"
    o_key = f"model.layers.{layer}.self_attn.out_proj.weight"
    for name in (q_key, o_key):
        if name not in spec.weight_keys:
            raise CellTechniqueError(
                f"C10 at {spec.site} does not own {name}; the component's key "
                f"set is {list(spec.weight_keys)}."
            )
    q_proj = request.parent_weights[q_key].detach().to(dtype=torch.float64)
    o_proj = request.parent_weights[o_key].detach().to(dtype=torch.float64)
    query_width = int(q_proj.shape[0])
    if int(o_proj.shape[1]) != query_width:
        raise CellTechniqueError(
            f"C10 sees a {query_width}-wide query projection and a "
            f"{int(o_proj.shape[1])}-wide O projection input; the head blocks "
            "of the two tensors do not line up."
        )
    head_dim = _head_dimension(request, spec, query_width)
    heads = query_width // head_dim

    kept_width = int(request.level_width)
    if kept_width % head_dim != 0:
        raise CellTechniqueError(
            f"Level {request.cell.level} resolves to {kept_width} channels, "
            f"which is not a whole number of {head_dim}-wide heads; C10's level "
            "axis is head counts and this kernel will not round to one."
        )
    kept_heads = kept_width // head_dim
    dropped = heads - kept_heads
    if dropped < 1 or kept_heads < 1:
        raise CellTechniqueError(
            f"Level {request.cell.level} keeps {kept_heads} of {heads} heads, "
            "which is not a head-axis contraction."
        )

    fit_context = request.evidence.fit_rows(spec.input_endpoint).to(dtype=torch.float64)
    fit_write = request.evidence.fit_rows(spec.write_endpoint).to(dtype=torch.float64)
    held_context = request.evidence.heldout_rows(spec.input_endpoint).to(
        dtype=torch.float64
    )
    held_write = request.evidence.heldout_rows(spec.write_endpoint).to(
        dtype=torch.float64
    )
    if int(fit_context.shape[1]) != query_width:
        raise CellTechniqueError(
            f"C10's input endpoint {spec.input_endpoint!r} is "
            f"{int(fit_context.shape[1])} wide; the O projection reads "
            f"{query_width} concatenated head channels."
        )

    # The parent gate the generic harness gate could not run.  o = context @
    # out_proj.T is a real single matmul between two sealed endpoints, so a
    # parent that does not reproduce its own write means this cell is measuring
    # something other than the component.
    tolerance = float(
        request.knob("head_drop_parent_tolerance", request.knob("calibration_tolerance", 5e-3))
    )
    parent_relative = _relative_l2(held_context @ o_proj.T, held_write)
    request.log(
        f"C10 {request.cell.text}: {heads} heads of {head_dim}, keep "
        f"{kept_heads} (drop {dropped}); parent write replay relL2 "
        f"{parent_relative:.6g}"
    )
    if not math.isfinite(parent_relative) or parent_relative > tolerance:
        raise CellTechniqueError(
            f"C10 cannot reproduce {spec.write_endpoint!r} from "
            f"{spec.input_endpoint!r} through the parent O projection: relL2 "
            f"{parent_relative:.6g} against a {tolerance:g} tolerance."
        )

    ridge = float(request.knob("head_drop_ridge", 1e-6))
    max_delta = request.knob("max_prior_delta_ratio", 1.0)
    gram = (fit_context.T @ fit_context).contiguous()
    cross = (fit_context.T @ fit_write).contiguous()
    target_energy = float((fit_write * fit_write).sum().item())

    def channels_of(kept: Sequence[int]) -> Any:
        spans = [
            torch.arange(head * head_dim, (head + 1) * head_dim, dtype=torch.int64)
            for head in kept
        ]
        return torch.cat(spans) if spans else torch.zeros(0, dtype=torch.int64)

    total = math.comb(heads, dropped)
    exhaustive = total <= _MAX_ENUMERATED_DROP_SETS
    scores: dict[str, float] = {}
    if exhaustive:
        best: tuple[float, tuple[int, ...]] | None = None
        for drop_set in itertools.combinations(range(heads), dropped):
            kept = [head for head in range(heads) if head not in drop_set]
            residual = _residual_after_drop(
                gram, cross, target_energy, channels_of(kept), ridge
            )
            scores["+".join(str(v) for v in drop_set)] = residual
            if best is None or residual < best[0]:
                best = (residual, tuple(drop_set))
        assert best is not None
        selected_residual, drop_heads = best
        selection_policy = "exhaustive-drop-set-least-squares-residual"
    else:
        # Greedy sequential: drop the single head whose removal costs least,
        # re-score, repeat.  Recorded as a different policy because it is one --
        # a greedy path is not guaranteed to find the exhaustive optimum.
        remaining = list(range(heads))
        chosen: list[int] = []
        selected_residual = target_energy
        for _step in range(dropped):
            step_best: tuple[float, int] | None = None
            for head in remaining:
                kept = [value for value in remaining if value != head]
                residual = _residual_after_drop(
                    gram, cross, target_energy, channels_of(kept), ridge
                )
                if step_best is None or residual < step_best[0]:
                    step_best = (residual, head)
            assert step_best is not None
            selected_residual, victim = step_best
            scores["+".join(str(v) for v in sorted([*chosen, victim]))] = (
                selected_residual
            )
            chosen.append(victim)
            remaining.remove(victim)
        drop_heads = tuple(sorted(chosen))
        selection_policy = "greedy-sequential-least-squares-residual"

    kept_heads_list = [head for head in range(heads) if head not in drop_heads]
    kept_channels = channels_of(kept_heads_list)

    # The final coefficients come from the production solver, with the parent's
    # own surviving columns as the prior, so C10's telemetry is the same
    # `Lfm2PriorCorrectionReport` every dense cell seals.
    solved, report = _fit_linear(
        fit_context.index_select(1, kept_channels),
        fit_write,
        role=f"{spec.name}_c10_head_drop_o_absorption",
        ridge=ridge,
        prior=o_proj.index_select(1, kept_channels).contiguous(),
        max_prior_delta_ratio=max_delta,
        solver_diagnostics=True,
        relative_ridge_lambda=request.knob("relative_ridge_lambda", None),
    )
    child_o = torch.zeros_like(o_proj)
    child_o[:, kept_channels] = solved
    child_q = q_proj.clone()
    for head in drop_heads:
        child_q[head * head_dim : (head + 1) * head_dim, :] = 0.0

    # The identity gate: same solve, every head kept, against a parent-derived
    # target.  A prior-anchored fit with an identically zero residual returns
    # its prior, so this is a measurement of the kernel's own algebra.
    identity_solved, _identity_report = _fit_linear(
        fit_context,
        fit_context @ o_proj.T,
        role=f"{spec.name}_c10_identity_gate",
        ridge=ridge,
        prior=o_proj,
        max_prior_delta_ratio=max_delta,
        solver_diagnostics=False,
    )
    identity_deviation = float((identity_solved - o_proj).abs().max().item())

    group = heads // max(int(_kv_head_count(request, spec, head_dim) or heads), 1)
    dead_groups = sorted(
        {
            head // group
            for head in drop_heads
            if group > 0
            and all(
                member in drop_heads
                for member in range((head // group) * group, (head // group + 1) * group)
            )
        }
    )

    installed = {q_key: child_q, o_key: child_o}
    stored = int(
        child_q.shape[1] * kept_heads * head_dim
        + child_o.shape[0] * kept_heads * head_dim
        + sum(
            int(request.parent_weights[name].numel())
            for name in spec.weight_keys
            if name not in (q_key, o_key)
        )
    )

    telemetry = report.as_dict()
    conditioning = telemetry.get("solver_conditioning", {})
    clamp = telemetry.get("prior_delta_clamp", {})
    relative = telemetry.get("relative_ridge", {})
    condition = conditioning.get("system_condition_number")
    condition_reported = condition is not None and math.isfinite(float(condition))
    report_body, report_non_finite = sealable_report(telemetry)
    solve: dict[str, Any] = {
        "solver_path": conditioning.get("solver_path") or "unreported",
        "ridge_relative": float(relative.get("relative_ridge_lambda") or 0.0),
        "condition_number": sealable_solve_number(condition),
        "condition_cap_saturated": not condition_reported,
        "condition_number_unbounded_reason": (
            None
            if condition_reported
            else ("solver-reported-none" if condition is None else "singular-system")
        ),
        "prior_norm": float(telemetry.get("prior_norm", 0.0)),
        "correction_norm": float(telemetry.get("correction_norm", 0.0)),
        "requested_clamp_ratio": sealable_solve_number(
            clamp.get("requested_prior_delta_ratio")
        ),
        "allowed_clamp_ratio": sealable_solve_number(
            clamp.get("allowed_prior_delta_ratio")
        ),
        "clamp_engaged": bool(clamp.get("clamp_engaged") or False),
        "omp": None,
        "prior_correction_report": report_body,
        "prior_correction_report_non_finite": report_non_finite,
        "contraction_axis": "heads",
        "parent_heads": heads,
        "head_dim": head_dim,
        "kept_heads": kept_heads,
        "dropped_heads": list(drop_heads),
        "dropped_head_count": int(dropped),
        "kept_head_indices": kept_heads_list,
        "head_selection_policy": selection_policy,
        "head_drop_set_scores": scores,
        "head_drop_candidate_sets": int(total),
        "head_drop_enumerated": bool(exhaustive),
        # The denominator every score above is relative to, sealed so a reader
        # can rank the candidate drop sets without re-opening the evidence.
        "fit_write_energy": target_energy,
        "selected_drop_residual": float(selected_residual),
        "selected_drop_relative_residual": (
            None if target_energy <= 0.0 else float(selected_residual) / target_energy
        ),
        "parent_matmul_relative_l2": sealable_solve_number(parent_relative),
        "parent_matmul_tolerance": tolerance,
        "identity_gate_max_abs_deviation": identity_deviation,
        # Reported and deliberately not banked: the kv rows a dead group would
        # free are only a saving if the child actually stops storing them, and
        # this child stores them.
        "dead_kv_groups": dead_groups,
        "kv_group_size": int(group),
        "unclaimed_dead_kv_parameters": int(
            sum(
                int(request.parent_weights[name].shape[1]) * head_dim
                for name in spec.weight_keys
                if name.endswith(("k_proj.weight", "v_proj.weight"))
            )
            * len(dead_groups)
        ),
    }

    def predict(rows: Any) -> Any:
        return rows.to(dtype=torch.float64) @ child_o.T

    return CellTechniqueResult(
        installed_weights=installed,
        stored_parameter_count=stored,
        solve=solve,
        predict=predict,
        notes=(
            f"C10 head drop: {heads} -> {kept_heads} query heads (dropped "
            f"{list(drop_heads)}) with least-squares absorption into the "
            f"surviving O columns; parent replay relL2 {parent_relative:.6g}, "
            f"identity gate {identity_deviation:.3g}."
        ),
    )


def _kv_head_count(request: Any, spec: Any, head_dim: int) -> int | None:
    """KV head count from ``v_proj``'s own geometry, or ``None`` when absent."""

    key = f"model.layers.{int(spec.layer_index)}.self_attn.v_proj.weight"
    value = request.parent_weights.get(key)
    if value is None or head_dim <= 0:
        return None
    rows = int(value.shape[0])
    return rows // head_dim if rows % head_dim == 0 else None


def _relative_l2(candidate: Any, reference: Any) -> float:
    torch = _torch()
    numerator = float(torch.linalg.matrix_norm(candidate - reference).item())
    denominator = float(torch.linalg.matrix_norm(reference).item())
    if denominator <= 0.0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def _register_head_drop_technique() -> None:
    """Wire the C10 kernel into the harness dispatch table at import time."""

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    register(
        C10_TECHNIQUE_ID,
        c10_head_drop_technique,
        component=C10_COMPONENT_ID,
        replace=True,
    )


_register_head_drop_technique()


def describe_head_drop_technique() -> Mapping[str, Any]:
    """Serializable inventory of what this kernel wires, for a cell receipt."""

    return {
        "schema": "neural-forge-matrix-technique-head-drop-v1",
        "technique": C10_TECHNIQUE_ID,
        "component": C10_COMPONENT_ID,
        "contraction_axis": "heads",
        "selection": "exhaustive drop sets by least-squares residual, greedy past the cap",
        "absorption": "production _fit_linear over the surviving O-projection columns",
        "parent_gate": "context @ out_proj.T against the sealed attention write",
    }
