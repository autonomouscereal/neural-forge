#!/usr/bin/env python3
"""T9: progressive multi-pass folding -- does stepping beat jumping?

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §1.1 lists T9 as "progressive
multi-pass (schedule rho_t, child_{t-1} -> parent_t)", §2.2 as "a driver that
re-points 'parent' at a published cell child", and §1.2 closes with "T9 applies
to any row whose T1-T8 mark survives at M16: its cells are schedules, not
solves, and enter in wave 3".  It has never run.

The hypothesis is distillation memo #6's and it is one sentence: a component
contracted in two small steps, each step solved against the *previous step's
child*, may end up better than the same component contracted to the same final
width in one jump against the parent.  If it is true it changes the composition
wave -- mode M-B (sequential conditioned composition) is exactly this idea
applied across components instead of across levels, so the cheapest possible
test of the progressive hypothesis is also a direct read on M-B.

What this kernel is
-------------------

T9 is a **schedule around another technique**, not a new solver.  It owns no
linear algebra at all: it resolves the base technique out of the same dispatch
table the harness uses, calls it once per rung, and between rungs it does the
only two things "child_{t-1} becomes parent_t" can mean:

* the **prior** becomes the previous rung's materialized weight, by handing the
  base kernel a parent state whose one component tensor is that weight;
* the **target** becomes the previous rung's own write, by handing the base
  kernel an evidence view whose write endpoint returns
  ``rows @ child_{t-1}.T`` instead of the parent's sealed rows.

Everything else -- the input endpoint, the split, the selection digest, the row
order -- is the sealed evidence, untouched.  A rung that changed the *inputs*
would be measuring a different question (and would need evidence this campaign
does not have: the folded child's own activations at that site are not sealed,
and capturing them is banned by the closed-path rule).  So the honest scope of
this cell is stated up front: **T9 conditions the teacher, not the input
distribution.**  Within a single-component cell those are the same thing at the
component's own boundary, because every upstream neighbour is copy_exact and
its input rows are therefore identical in parent and child.  That is the exact
sense in which a one-component T9 cell is clean and a composed M-B run is not.

The schedule
------------

``progressive_intermediate_levels`` names the rungs *before* the last one; the
final width is always the cell's own level width.  The default ``["M16"]`` on a
1024-wide writer therefore runs 1024 -> 960 -> (the cell's level), so
``C05.T9.M8.L05`` stores rank 896 -- byte-for-byte the same rank as the direct
``C05.T1.M8.L05`` cell it is meant to be compared against.  That is deliberate:
a two-step schedule that stopped at ``floor(960 * 15/16) = 900`` would differ
from the direct cell in *two* ways at once (rank and teacher), and the only
question worth a cell here is the teacher.

One consequence to have in view when reading the receipt.  On the input axis
the base technique's POD basis is a function of the input rows alone, and the
input rows do not change between rungs, so the rung-2 basis is the top-``r2``
eigenvectors of the same Gram the direct cell would use -- literally the same
subspace.  The two cells therefore differ in exactly one thing: what the second
factor was solved against.  A T9 cell that matches its direct twin to four
decimals is not a bug, it is the finding that progressive scheduling buys
nothing at this component.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

__all__ = [
    "T9_TECHNIQUE_ID",
    "describe_progressive_technique",
    "t9_progressive_multipass_technique",
]

T9_TECHNIQUE_ID = "T9"

#: The components a schedule can wrap.  These are the body components whose
#: base techniques materialize a parent-shaped tensor for a single weight key;
#: the token interface publishes a whole child and C04/C10/C14 install more
#: than a plain channel map, so a schedule over them would need its own
#: rung-to-rung substitution rule and does not get a silent one here.
_COMPONENTS: tuple[str, ...] = (
    "C03",
    "C05",
    "C06",
    "C07",
    "C08",
    "C09",
    "C11",
    "C12",
    "C13",
)

_DEFAULT_BASE_TECHNIQUE = "T1"
_DEFAULT_INTERMEDIATE_LEVELS = ("M16",)


def _torch() -> Any:
    import torch

    return torch


class _TeacherEvidence:
    """The sealed evidence with one endpoint answered by the previous child.

    Implements `matrix_technique_registry.MatrixCellEvidence` by delegation.
    Every endpoint except the component's write is the sealed rows, unchanged
    and un-copied; the write endpoint returns the previous rung's own output on
    those same rows, which is what "child_{t-1} becomes parent_t" means for a
    least-squares target.  The selection digest is passed through untouched --
    the evidence identity a receipt seals is still the sealed base's, because
    that is still where every row came from.
    """

    def __init__(self, base: Any, endpoint: str, weight: Any, input_endpoint: str):
        self._base = base
        self._endpoint = str(endpoint)
        self._weight = weight
        self._input_endpoint = str(input_endpoint)

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(self._base.endpoints)

    @property
    def selection_sha256(self) -> str:
        return str(self._base.selection_sha256)

    @property
    def fit_row_count(self) -> int:
        return int(self._base.fit_row_count)

    @property
    def heldout_row_count(self) -> int:
        return int(self._base.heldout_row_count)

    def fit_rows(self, endpoint: str) -> Any:
        if endpoint == self._endpoint:
            return self._teach(self._base.fit_rows(self._input_endpoint))
        return self._base.fit_rows(endpoint)

    def heldout_rows(self, endpoint: str) -> Any:
        if endpoint == self._endpoint:
            return self._teach(self._base.heldout_rows(self._input_endpoint))
        return self._base.heldout_rows(endpoint)

    def _teach(self, rows: Any) -> Any:
        torch = _torch()
        return (rows.to(dtype=torch.float64) @ self._weight.T).contiguous()


def _rung_width(own_width: int, level: str) -> int:
    from .matrix_cell_receipt import LEVELS
    from .matrix_technique_registry import CellTechniqueError

    fraction = LEVELS.get(level)
    if not isinstance(fraction, float):
        raise CellTechniqueError(
            f"T9's schedule cannot use level {level!r}: it has no marginal "
            "fraction, so the rung width is not defined."
        )
    width = int(math.floor(int(own_width) * float(fraction)))
    if width < 1 or width >= int(own_width):
        raise CellTechniqueError(
            f"T9 rung {level} on width {own_width} resolves to {width}, which "
            "is not a contraction."
        )
    return width


def t9_progressive_multipass_technique(request: Any) -> Any:
    """T9: run the base technique once per rung, each rung taught by the last."""

    from .matrix_technique_registry import (
        CellTechniqueError,
        CellTechniqueResult,
        resolve,
        sealable_report,
        sealable_solve_number,
    )

    torch = _torch()
    spec = request.component
    if spec.contraction_axis not in ("input", "output"):
        raise CellTechniqueError(
            f"T9 schedules a channel-axis technique; {spec.component_id} has "
            f"axis {spec.contraction_axis!r} and needs its own schedule rule."
        )
    if spec.input_endpoint is None or spec.write_endpoint is None:
        raise CellTechniqueError(f"{spec.component_id} has no sealed endpoints.")
    base_id = str(request.knob("progressive_base_technique", _DEFAULT_BASE_TECHNIQUE))
    if base_id == T9_TECHNIQUE_ID:
        raise CellTechniqueError("T9 cannot schedule itself.")
    base = resolve(base_id, spec.component_id)
    intermediate = tuple(
        str(value)
        for value in request.knob(
            "progressive_intermediate_levels", list(_DEFAULT_INTERMEDIATE_LEVELS)
        )
    )
    if not intermediate:
        raise CellTechniqueError(
            "T9 with no intermediate rung is the direct cell under a different "
            "technique id; give it at least one rung or run the direct cell."
        )
    widths = [_rung_width(spec.own_width, level) for level in intermediate]
    widths.append(int(request.level_width))
    labels = [*intermediate, str(request.cell.level)]
    for previous, current in zip(widths[:-1], widths[1:], strict=True):
        if current >= previous:
            raise CellTechniqueError(
                f"T9's schedule is not monotone: rung width {current} does not "
                f"contract the previous rung's {previous}."
            )

    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach().to(dtype=torch.float64)
    request.log(
        f"T9 {request.cell.text}: base {base_id}, schedule "
        + " -> ".join(
            [str(spec.own_width)] + [f"{label}({width})" for label, width in zip(labels, widths, strict=True)]
        )
    )

    weights: Mapping[str, Any] = request.parent_weights
    evidence = request.evidence
    passes: list[dict[str, Any]] = []
    result: Any = None
    previous_weight = parent
    for index, (label, width) in enumerate(zip(labels, widths, strict=True)):
        rung_request = dataclasses.replace(
            request,
            level=label,
            level_width=int(width),
            parent_weights=weights,
            evidence=evidence,
        )
        result = base(rung_request)
        if set(result.installed_weights) != {key}:
            raise CellTechniqueError(
                f"T9's base technique {base_id} installed "
                f"{sorted(result.installed_weights)}; a schedule can only "
                "re-point a single materialized tensor."
            )
        current = result.installed_weights[key].detach().to(dtype=torch.float64)
        body, moved = sealable_report(dict(result.solve))
        passes.append(
            {
                "rung": index + 1,
                "level": label,
                "width": int(width),
                "stored_parameter_count": int(result.stored_parameter_count),
                "teacher": "parent" if index == 0 else f"rung-{index}",
                "step_relative_change": sealable_solve_number(
                    _relative_l2(current, previous_weight)
                ),
                "parent_relative_change": sealable_solve_number(
                    _relative_l2(current, parent)
                ),
                "solve": body,
                "solve_non_finite": moved,
            }
        )
        # child_{t} becomes parent_{t+1}: both the prior and the target move.
        weights = {**dict(request.parent_weights), key: current}
        evidence = _TeacherEvidence(
            request.evidence, spec.write_endpoint, current, spec.input_endpoint
        )
        previous_weight = current

    assert result is not None
    final = result.installed_weights[key].detach().to(dtype=torch.float64).contiguous()
    tail = dict(result.solve)
    solve: dict[str, Any] = {
        "solver_path": (
            "t9_progressive_multipass:"
            + base_id
            + ":"
            + "->".join(f"{label}({width})" for label, width in zip(labels, widths, strict=True))
        ),
        "ridge_relative": float(tail.get("ridge_relative") or 0.0),
        "condition_number": sealable_solve_number(tail.get("condition_number")),
        "condition_cap_saturated": bool(tail.get("condition_cap_saturated", False)),
        "prior_norm": float(torch.linalg.matrix_norm(parent).item()),
        "correction_norm": float(torch.linalg.matrix_norm(final - parent).item()),
        "requested_clamp_ratio": sealable_solve_number(tail.get("requested_clamp_ratio")),
        "allowed_clamp_ratio": sealable_solve_number(tail.get("allowed_clamp_ratio")),
        "clamp_engaged": bool(tail.get("clamp_engaged", False)),
        "omp": tail.get("omp"),
        "progressive_base_technique": base_id,
        "progressive_rungs": len(widths),
        "progressive_schedule": [
            {"level": label, "width": int(width)}
            for label, width in zip(labels, widths, strict=True)
        ],
        "progressive_final_width": int(widths[-1]),
        "progressive_matches_direct_level_width": bool(
            int(widths[-1]) == int(request.level_width)
        ),
        "progressive_passes": passes,
        # How far the schedule's own last step moved: a schedule whose final
        # rung barely differs from its predecessor spent a pass to change
        # nothing, and that is a result about the schedule, not about the cell.
        "progressive_final_step_relative_change": passes[-1]["step_relative_change"],
        "progressive_total_relative_change": sealable_solve_number(
            _relative_l2(final, parent)
        ),
        "contraction_basis_rank": int(widths[-1]),
    }

    def predict(rows: Any) -> Any:
        return rows.to(dtype=torch.float64) @ final.T

    return CellTechniqueResult(
        installed_weights={key: final},
        stored_parameter_count=int(result.stored_parameter_count),
        solve=solve,
        predict=predict,
        notes=(
            f"T9 progressive multi-pass over {base_id}: "
            + " -> ".join(f"{label}({width})" for label, width in zip(labels, widths, strict=True))
            + f"; final width {widths[-1]} equals the cell's own level width "
            f"{int(request.level_width)}."
        ),
    )


def _relative_l2(candidate: Any, reference: Any) -> float:
    torch = _torch()
    numerator = float(torch.linalg.matrix_norm(candidate - reference).item())
    denominator = float(torch.linalg.matrix_norm(reference).item())
    if denominator <= 0.0 or not math.isfinite(denominator):
        return float("nan")
    return numerator / denominator


def _register_progressive_technique() -> None:
    """Wire T9 into the harness dispatch table at import time."""

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    for component in _COMPONENTS:
        register(
            T9_TECHNIQUE_ID,
            t9_progressive_multipass_technique,
            component=component,
            replace=True,
        )


_register_progressive_technique()


def describe_progressive_technique() -> Mapping[str, Any]:
    """Serializable inventory of what this schedule wires, for a cell receipt."""

    return {
        "schema": "neural-forge-matrix-technique-progressive-v1",
        "technique": T9_TECHNIQUE_ID,
        "components": _COMPONENTS,
        "default_base_technique": _DEFAULT_BASE_TECHNIQUE,
        "default_intermediate_levels": _DEFAULT_INTERMEDIATE_LEVELS,
        "conditions": ("prior", "least-squares target"),
        "does_not_condition": ("input rows", "evidence split", "row order"),
    }
