"""T5 — sparse residual-OMP routes generalized to body components.

The A9/A11 token-interface routes are the only production configuration with
measured behavioral retention (matrix-campaign design S4).  Their mechanism is
narrow in one respect only: the *dictionary* was a planted anchor table of
vocabulary rows.  Everything else — full-dictionary residual OMP with
deterministic ties, k-sparse coefficients solved on the selected support, and
the A11 split where ``C_in`` and ``C_out`` share supports but are solved
separately — is indifferent to what a "row" means.

This module transplants that mechanism onto a body component's channel axis.

Geometry
--------
A body component is a linear map with exactly one *component axis* (the width
the campaign's level contracts) and one *interface axis* (the residual width,
which a single-component cell never touches).

* ``role="writer"`` — the component axis is the map's **input** (``mlp_down``
  2560→1024, ``attn_o`` 1024→1024, ``conv_out_proj`` 1024→1024).
* ``role="reader"`` — the component axis is the map's **output**
  (``mlp_gate``/``mlp_up`` 1024→2560, ``attn_q/k/v``, ``conv_in_proj``).

Contracting the component axis keeps ``m`` channels (the *anchors*, exact
one-hot routes, the direct-bridge rows of A9) and drops ``N-m``.  Each dropped
channel receives a k-sparse code over the kept channels, selected by residual
OMP on the sealed evidence columns of the component axis.

Where the two roles differ is where the code is *installed*:

* writer — the dropped channel's contribution ``x_d · W[d,:]`` is absorbed into
  the kept columns, ``W'[i,:] = W[kept_i,:] + Σ_d c_{d,i} W[drop_d,:]``.  No
  runtime route object survives; the route is folded into the narrower weight.
* reader — the kept rows of ``W`` are copied bit-exactly and a sparse
  ``m×N`` expansion carries the child's narrow output back to the width the
  copy_exact downstream consumer still expects.

The A11 split, transplanted
---------------------------
Both coefficient sets are always solved on the *same* OMP supports:

* ``read`` coefficients minimise the channel-value residual
  ``‖a_d − A_S c‖²`` — decoupled, one exact ridge solve per dropped channel.
  This is the ``C_in`` analogue.
* ``write`` coefficients minimise the *induced output* error
  ``‖(A_drop − A_kept Cᵀ) M_drop‖²_F`` where ``M`` is the write-side metric
  (the writer's own weight rows; for a reader, the downstream consumer's
  matrix if the harness supplies one).  Expanding the trace gives
  ``Σ_{d,e} G_{de} ⟨r_d, r_e⟩`` with ``G = M_drop M_dropᵀ``: the dropped
  channels are *coupled* whenever ``G`` is not diagonal, which is exactly why
  A11's ``C_out`` is a genuinely different solve on identical supports.  When
  ``G`` is diagonal the coupled solve collapses to the decoupled one in a
  single sweep and the receipt seals ``coupled: false`` rather than pretending
  a second lane ran.

Fold-only discipline
--------------------
Selection and both solves are OMP plus least squares on sealed evidence.  No
gradient, no optimizer, no parent forward, no model execution of any kind.
Every input tensor is rejected if it carries autograd state.  All linear
algebra runs inside ``torch.no_grad()`` and the sealed record carries the
prohibited-operation audit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256


MATRIX_TECHNIQUE_ENTRY_SCHEMA = "neural-forge-matrix-technique-entry-v1"
T5_BODY_ROUTE_RECORD_SCHEMA = "neural-forge-matrix-t5-body-route-record-v1"

T5_TECHNIQUE_ID = "T5"
T5_TECHNIQUE_NAME = "sparse_omp_routes"

#: Selection is a full-dictionary residual OMP over the kept-channel columns.
#: The incremental Cholesky pivot ``G_jj - ‖L⁻¹G_Sj‖²`` *is* the squared
#: double-MGS orthogonal residual of the candidate atom against the current
#: support, so the estate's stability test is preserved exactly while the
#: R-sized Q basis of ``hf_lfm2_sparse_semantic_codebook`` is never allocated.
T5_SELECTION_ALGORITHM = (
    "full-kept-channel-normalized-residual-omp-incremental-cholesky-pivot-"
    "stable-channel-id-ties-v1"
)
T5_READ_SOLVE_ALGORITHM = "shared-support-decoupled-ridge-value-residual-v1"
T5_WRITE_SOLVE_ALGORITHM = (
    "shared-support-output-metric-coupled-damped-jacobi-bounded-sweeps-v1"
)

#: ``level`` -> kept fraction of the component axis.  ``X4`` is the canonical
#: exact-4x layout width for the component and has no closed form here: the
#: option bank supplies it as an explicit ``kept_width``.
_LEVEL_KEEP_FRACTION: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"M16": (15, 16), "M8": (7, 8), "M4": (3, 4), "M1": (1, 1)}
)

#: Body components this technique covers, with the campaign's component ids.
#: ``C04`` (depthwise FIR kernel), ``C10`` (head counts), ``C14`` (diagonal
#: norms) have no channel-route geometry; ``C01``/``C02``/``C15`` are the token
#: interface, which is A9/A11's own already-sealed lane and is not re-derived
#: here.
_COMPONENTS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "C03": ("conv_in_proj", "reader"),
        "C05": ("conv_out_proj", "writer"),
        "C06": ("attn_q", "reader"),
        "C07": ("attn_k", "reader"),
        "C08": ("attn_v", "reader"),
        "C09": ("attn_o", "writer"),
        "C11": ("mlp_gate", "reader"),
        "C12": ("mlp_up", "reader"),
        "C13": ("mlp_down", "writer"),
    }
)

_COMPONENT_IDS_BY_NAME: Mapping[str, str] = MappingProxyType(
    {name: key for key, (name, _role) in _COMPONENTS.items()}
)

_CHANNEL_SELECTION_POLICIES = ("evidence-energy", "pivoted-qr", "explicit")

_MAX_DEPENDENT_ATOM_RETRIES = 4

_DAMPING_LADDER = (1.0, 0.5, 0.25)

_PROHIBITED_OPERATION_AUDIT: Mapping[str, Any] = MappingProxyType(
    {
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "model_forward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "network_access": False,
    }
)


class T5BodyRouteError(ValueError):
    """A T5 body-route request is outside the technique's proven envelope."""


class T5BodyRouteGeometryError(T5BodyRouteError):
    """Component geometry, level, or support width cannot be honoured."""


class T5BodyRouteEvidenceError(T5BodyRouteError):
    """Sealed evidence is missing, malformed, or carries no usable energy."""


class T5BodyRouteConditioningError(T5BodyRouteError):
    """The kept-channel dictionary or a shared-support solve is degenerate."""


class T5BodyRouteDisciplineError(T5BodyRouteError):
    """A fold-only discipline rule was violated by the request."""


class T5BodyRouteAuthenticationError(T5BodyRouteError):
    """A sealed technique record failed its seal or schema check."""


# ---------------------------------------------------------------------------
# Small validated primitives (estate idiom: refuse early, name the reason)
# ---------------------------------------------------------------------------


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise T5BodyRouteError(
            "PyTorch is required for T5 sparse residual-OMP body routes."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise T5BodyRouteGeometryError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise T5BodyRouteGeometryError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise T5BodyRouteGeometryError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise T5BodyRouteGeometryError(f"{name} must be finite and non-negative.")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise T5BodyRouteError(f"{name} must be a nonempty string.")
    return value.strip()


def _no_autograd(value: Any, name: str) -> None:
    torch = _torch()
    if isinstance(value, torch.Tensor) and (
        value.requires_grad or value.grad_fn is not None
    ):
        raise T5BodyRouteDisciplineError(
            f"{name} carries autograd state; T5 is a fold-only technique."
        )


def _matrix(value: Any, name: str, *, dtype: Any, device: Any) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise T5BodyRouteEvidenceError(f"{name} must be a nonempty real matrix.")
    _no_autograd(value, name)
    result = value.detach().to(device=device, dtype=dtype).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise T5BodyRouteEvidenceError(f"{name} is not finite.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype),
            "shape": [int(size) for size in tensor.shape],
            "bytes": tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex(),
        }
    )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise T5BodyRouteError("Sealed records cannot carry non-finite numbers.")
        return value
    raise T5BodyRouteError(
        f"Sealed record contains non-JSON value {type(value).__name__}."
    )


def _sealed(body: Mapping[str, Any], field: str = "record_sha256") -> dict[str, Any]:
    value = _plain_json(body)
    return {**value, field: canonical_sha256(value)}


def verify_sealed_body_route_record(
    value: Mapping[str, Any],
    *,
    field: str = "record_sha256",
) -> dict[str, Any]:
    """Re-check a sealed technique record before the harness embeds it."""

    if not isinstance(value, Mapping):
        raise T5BodyRouteAuthenticationError("A technique record must be a mapping.")
    result = _plain_json(value)
    supplied = result.pop(field, None)
    if supplied != canonical_sha256(result):
        raise T5BodyRouteAuthenticationError("Technique record seal is invalid.")
    if result.get("schema") != T5_BODY_ROUTE_RECORD_SCHEMA:
        raise T5BodyRouteAuthenticationError("Technique record schema is unsupported.")
    return {**result, field: supplied}


def _relative_l2(difference: Any, reference: Any) -> float:
    torch = _torch()
    numerator = float(torch.linalg.vector_norm(difference).item())
    denominator = float(torch.linalg.vector_norm(reference).item())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise T5BodyRouteEvidenceError(
            "A comparison reference has no energy; the evidence is empty."
        )
    return numerator / denominator


# ---------------------------------------------------------------------------
# Level and component resolution
# ---------------------------------------------------------------------------


def resolve_component(component: Any, role: Any = None) -> tuple[str, str, str]:
    """Return ``(component_id, component_name, role)`` for a campaign cell."""

    name = _text(component, "component")
    if name in _COMPONENTS:
        component_id = name
        component_name, default_role = _COMPONENTS[name]
    elif name in _COMPONENT_IDS_BY_NAME:
        component_id = _COMPONENT_IDS_BY_NAME[name]
        component_name, default_role = _COMPONENTS[component_id]
    elif name == "custom":
        component_id = "custom"
        component_name = "custom"
        default_role = None
    else:
        raise T5BodyRouteGeometryError(
            f"{name!r} has no channel-route geometry for T5.  Supported: "
            + ", ".join(sorted(_COMPONENTS))
            + " (or component='custom' with an explicit role)."
        )
    if role is None:
        if default_role is None:
            raise T5BodyRouteGeometryError(
                "component='custom' requires an explicit role of "
                "'reader' or 'writer'."
            )
        return component_id, component_name, default_role
    chosen = _text(role, "role")
    if chosen not in ("reader", "writer"):
        raise T5BodyRouteGeometryError("role must be 'reader' or 'writer'.")
    if default_role is not None and chosen != default_role:
        raise T5BodyRouteGeometryError(
            f"{component_name} is a {default_role} component; role={chosen!r} "
            "would mis-orient the component axis."
        )
    return component_id, component_name, chosen


def resolve_kept_width(
    *,
    level: Any,
    component_width: int,
    kept_width: Any = None,
) -> int:
    """Resolve the campaign level into a kept-channel count."""

    width = _positive_int(component_width, "component_width")
    name = _text(level, "level").upper()
    if kept_width is not None:
        resolved = _positive_int(kept_width, "kept_width")
    elif name in _LEVEL_KEEP_FRACTION:
        numerator, denominator = _LEVEL_KEEP_FRACTION[name]
        resolved = (width * numerator) // denominator
    elif name == "X4":
        raise T5BodyRouteGeometryError(
            "level='X4' has no closed-form width; supply the option bank's "
            "kept_width for this component."
        )
    else:
        raise T5BodyRouteGeometryError(
            f"level={name!r} is not a campaign level. Supported: "
            + ", ".join(sorted(_LEVEL_KEEP_FRACTION))
            + ", X4 (with kept_width)."
        )
    if resolved < 1 or resolved > width:
        raise T5BodyRouteGeometryError(
            f"level={name!r} yields kept_width={resolved} outside 1..{width}."
        )
    return resolved


# ---------------------------------------------------------------------------
# Deterministic kept-channel (anchor) selection
# ---------------------------------------------------------------------------


def select_kept_channels(
    component_rows: Any,
    *,
    kept_width: int,
    policy: str = "evidence-energy",
    explicit_ids: Sequence[int] | None = None,
) -> Any:
    """Choose the kept channels — the planted anchor basis of the A9 analogy.

    ``evidence-energy`` ranks channels by sealed-evidence column energy and is
    the default; ``pivoted-qr`` runs a deterministic norm-downdated pivoted
    Gram-Schmidt, which favours a well-conditioned dictionary over a
    high-energy one.  Both break ties on the lower channel id.  The returned
    ids are always sorted ascending so the child's channel order is the
    parent's channel order.
    """

    torch = _torch()
    rows = component_rows
    width = int(rows.shape[1])
    keep = _positive_int(kept_width, "kept_width")
    if keep > width:
        raise T5BodyRouteGeometryError("kept_width exceeds the component width.")
    if policy not in _CHANNEL_SELECTION_POLICIES:
        raise T5BodyRouteGeometryError(
            "channel_selection must be one of "
            + ", ".join(_CHANNEL_SELECTION_POLICIES)
        )
    if policy == "explicit":
        if explicit_ids is None:
            raise T5BodyRouteGeometryError(
                "channel_selection='explicit' requires kept_channel_ids."
            )
        ids = sorted({int(value) for value in explicit_ids})
        if len(ids) != keep:
            raise T5BodyRouteGeometryError(
                f"kept_channel_ids holds {len(ids)} distinct ids but "
                f"kept_width is {keep}."
            )
        if ids and (ids[0] < 0 or ids[-1] >= width):
            raise T5BodyRouteGeometryError(
                "kept_channel_ids fall outside the component axis."
            )
        return torch.tensor(ids, dtype=torch.int64)

    energy = (rows * rows).sum(dim=0).to(dtype=torch.float64)
    if policy == "evidence-energy":
        # Stable descending energy sort: ties resolve on the lower channel id.
        # ``matrix_technique_relation_matching.select_kept_channels`` restates
        # this rule so a T5/T8 contrast folds the same kept set.
        order = torch.argsort(-energy, stable=True)
        return torch.sort(order[:keep]).values

    # pivoted-qr: greedy maximum-residual-norm pivots with an exact downdate.
    residual_energy = energy.clone()
    basis: list[Any] = []
    chosen_ids: list[int] = []
    work = rows.to(dtype=torch.float64)
    floor = float(torch.finfo(torch.float64).eps) * float(energy.max().item()) * width
    for _step in range(keep):
        best = float(residual_energy.max().item())
        if not math.isfinite(best) or best <= floor:
            raise T5BodyRouteConditioningError(
                "pivoted-qr exhausted the evidence rank before kept_width; the "
                "component axis does not carry that many independent channels."
            )
        tie = residual_energy >= best
        pivot = int(
            torch.where(
                tie,
                torch.arange(width, dtype=torch.int64),
                torch.full((width,), width, dtype=torch.int64),
            )
            .min()
            .item()
        )
        column = work[:, pivot].clone()
        for atom in basis:
            column = column - atom * float(torch.dot(atom, column).item())
        norm = float(torch.linalg.vector_norm(column).item())
        if norm <= math.sqrt(floor):
            raise T5BodyRouteConditioningError(
                "pivoted-qr selected a numerically dependent channel."
            )
        atom = column / norm
        basis.append(atom)
        chosen_ids.append(pivot)
        projections = atom @ work
        residual_energy = (residual_energy - projections * projections).clamp_min(0.0)
        residual_energy[pivot] = 0.0
    return torch.sort(torch.tensor(sorted(chosen_ids), dtype=torch.int64)).values


# ---------------------------------------------------------------------------
# Residual OMP over kept-channel columns (Gram space, A9 semantics preserved)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OmpSupports:
    selected: Any  # (dropped, active_k) int64, -1 padding
    lengths: Any  # (dropped,) int64
    diagnostics: Mapping[str, Any]


def _omp_supports(
    *,
    kept_gram: Any,
    cross_gram: Any,
    dropped_energy: Any,
    active_k: int,
    channels_per_chunk: int,
    pivot_tolerance: float,
    residual_floor: float,
) -> _OmpSupports:
    torch = _torch()
    width = int(kept_gram.shape[0])
    dropped = int(cross_gram.shape[1])
    steps = min(int(active_k), width)
    dtype = kept_gram.dtype
    device = kept_gram.device
    kept_norms = torch.sqrt(torch.diagonal(kept_gram).clamp_min(0.0))
    if not bool((kept_norms > 0).all().item()):
        raise T5BodyRouteConditioningError(
            "A kept channel carries zero evidence energy; it cannot anchor a "
            "route.  Re-select the kept channels or shrink the level."
        )
    axis = torch.arange(width, device=device, dtype=torch.int64)
    selected_all = torch.full((dropped, steps), -1, dtype=torch.int64)
    dependent_rejections = 0
    exhausted_rows = 0
    step_records: list[dict[str, Any]] = []
    step_cosine_sums = [0.0] * steps
    step_cosine_counts = [0] * steps
    step_residual_sums = [0.0] * steps

    for start in range(0, dropped, channels_per_chunk):
        stop = min(start + channels_per_chunk, dropped)
        count = stop - start
        target = cross_gram[:, start:stop].contiguous()
        energy = dropped_energy[start:stop]
        row_axis = torch.arange(count, device=device, dtype=torch.int64)
        coefficients = torch.zeros((width, count), dtype=dtype, device=device)
        chol = torch.zeros((count, 0, 0), dtype=dtype, device=device)
        rhs = torch.zeros((count, 0), dtype=dtype, device=device)
        solved = torch.zeros((count, 0), dtype=dtype, device=device)
        selected = torch.full((count, steps), -1, dtype=torch.int64, device=device)
        masked = torch.zeros((count, width), dtype=torch.bool, device=device)
        active = energy > 0.0

        for step in range(steps):
            residual = (energy - (rhs * solved).sum(dim=1)).clamp_min(0.0)
            ratio = torch.where(
                energy > 0,
                torch.sqrt(residual / energy.clamp_min(torch.finfo(dtype).tiny)),
                torch.zeros_like(residual),
            )
            active = active & (ratio > residual_floor)
            correlation = (target - kept_gram @ coefficients).transpose(0, 1)
            score = (correlation.abs() / kept_norms).masked_fill(masked, -1.0)
            pending = active.clone()
            choice = torch.zeros((count,), dtype=torch.int64, device=device)
            pivot_value = torch.ones((count,), dtype=dtype, device=device)
            new_row = torch.zeros((count, step), dtype=dtype, device=device)
            accepted = torch.zeros((count,), dtype=torch.bool, device=device)
            for _retry in range(_MAX_DEPENDENT_ATOM_RETRIES + 1):
                if not bool(pending.any().item()):
                    break
                best = score.max(dim=1).values
                tie = score >= best.unsqueeze(1)
                candidate = torch.where(
                    tie, axis.unsqueeze(0), torch.full_like(tie, width, dtype=axis.dtype)
                ).min(dim=1).values
                usable = pending & (best > 0.0) & (candidate < width)
                pending = usable
                if not bool(pending.any().item()):
                    break
                candidate = candidate.clamp(max=width - 1)
                diagonal = torch.diagonal(kept_gram).index_select(0, candidate)
                if step == 0:
                    row = torch.zeros((count, 0), dtype=dtype, device=device)
                    pivot = diagonal
                else:
                    support = selected[:, :step].clamp_min(0)
                    column = kept_gram[support, candidate.unsqueeze(1)]
                    row = torch.linalg.solve_triangular(
                        chol, column.unsqueeze(2), upper=False
                    ).squeeze(2)
                    pivot = diagonal - (row * row).sum(dim=1)
                good = pending & (pivot > pivot_tolerance * diagonal)
                choice = torch.where(good, candidate, choice)
                pivot_value = torch.where(good, pivot, pivot_value)
                new_row = torch.where(good.unsqueeze(1), row, new_row)
                accepted = accepted | good
                rejected = pending & ~good
                if bool(rejected.any().item()):
                    dependent_rejections += int(rejected.sum().item())
                    reject_rows = row_axis[rejected]
                    reject_ids = candidate[rejected]
                    score[reject_rows, reject_ids] = -1.0
                    masked[reject_rows, reject_ids] = True
                pending = rejected
            exhausted = active & ~accepted
            exhausted_rows += int(exhausted.sum().item())
            active = accepted
            chol = torch.cat(
                (
                    torch.cat(
                        (chol, torch.zeros((count, step, 1), dtype=dtype, device=device)),
                        dim=2,
                    ),
                    torch.cat(
                        (
                            torch.where(
                                active.unsqueeze(1),
                                new_row,
                                torch.zeros_like(new_row),
                            ).unsqueeze(1),
                            torch.where(
                                active,
                                torch.sqrt(
                                    pivot_value.clamp_min(torch.finfo(dtype).tiny)
                                ),
                                torch.ones_like(pivot_value),
                            ).reshape(count, 1, 1),
                        ),
                        dim=2,
                    ),
                ),
                dim=1,
            )
            appended = torch.where(
                active,
                target[choice, row_axis],
                torch.zeros((count,), dtype=dtype, device=device),
            )
            rhs = torch.cat((rhs, appended.unsqueeze(1)), dim=1)
            intermediate = torch.linalg.solve_triangular(
                chol, rhs.unsqueeze(2), upper=False
            )
            solved = torch.linalg.solve_triangular(
                chol.transpose(1, 2), intermediate, upper=True
            ).squeeze(2)
            selected[:, step] = torch.where(
                active, choice, torch.full_like(choice, -1)
            )
            masked[row_axis[active], choice[active]] = True
            coefficients = torch.zeros((width, count), dtype=dtype, device=device)
            valid = selected[:, : step + 1] >= 0
            if bool(valid.any().item()):
                rows_index = selected[:, : step + 1].clamp_min(0)[valid]
                columns_index = row_axis.unsqueeze(1).expand(count, step + 1)[valid]
                coefficients.index_put_(
                    (rows_index, columns_index), solved[valid], accumulate=False
                )
            chosen_cos = torch.where(
                active,
                score[row_axis, choice].abs()
                / torch.sqrt(residual.clamp_min(torch.finfo(dtype).tiny)),
                torch.zeros_like(ratio),
            )
            step_cosine_sums[step] += float(chosen_cos[active].sum().item())
            step_cosine_counts[step] += int(active.sum().item())
            step_residual_sums[step] += float(ratio.sum().item())
        selected_all[start:stop] = selected.to(device="cpu")

    for step in range(steps):
        count = step_cosine_counts[step]
        step_records.append(
            {
                "omp_step": step,
                "selecting_row_count": count,
                "mean_selected_absolute_residual_cosine": (
                    step_cosine_sums[step] / count if count else 0.0
                ),
                "mean_relative_residual_before_step": (
                    step_residual_sums[step] / dropped if dropped else 0.0
                ),
            }
        )
    lengths = (selected_all >= 0).sum(dim=1).to(dtype=torch.int64)
    diagnostics = {
        "algorithm": T5_SELECTION_ALGORITHM,
        "requested_active_k": int(active_k),
        "effective_active_k": steps,
        "dropped_channel_count": dropped,
        "dependent_atom_rejections": dependent_rejections,
        "residual_floor_exhausted_rows": exhausted_rows,
        "double_mgs": True,
        "reselection_masked": True,
        "candidate_tie_break": (
            "higher-absolute-residual-cosine-then-lower-channel-id"
        ),
        "support_width_histogram": {
            str(int(value)): int((lengths == int(value)).sum().item())
            for value in sorted(set(lengths.tolist()))
        },
        "steps": step_records,
    }
    return _OmpSupports(selected_all, lengths, MappingProxyType(diagnostics))


# ---------------------------------------------------------------------------
# Shared-support coefficient solves (the A11 split)
# ---------------------------------------------------------------------------


def _grouped_support_solve(
    *,
    kept_gram: Any,
    right_hand_side: Any,
    supports: Any,
    lengths: Any,
    ridge: float,
) -> tuple[Any, float]:
    """Solve one decoupled ridge system per dropped channel on its support."""

    torch = _torch()
    dtype = kept_gram.dtype
    device = kept_gram.device
    dropped, steps = map(int, supports.shape)
    coefficients = torch.zeros((dropped, steps), dtype=dtype, device=device)
    condition_proxy = 1.0
    for width in sorted({int(value) for value in lengths.tolist()}):
        if width == 0:
            continue
        rows = torch.nonzero(lengths == width, as_tuple=False).flatten()
        support = supports[rows, :width].to(device=device)
        gram = kept_gram[support.unsqueeze(2), support.unsqueeze(1)]
        diagonal = torch.diagonal(gram, dim1=1, dim2=2)
        gram = gram + ridge * torch.eye(width, dtype=dtype, device=device)
        rhs = right_hand_side[support, rows.unsqueeze(1)].unsqueeze(2)
        solved, info = torch.linalg.solve_ex(gram, rhs, check_errors=False)
        failed = info.ne(0) | ~torch.isfinite(solved).all(dim=(1, 2))
        if bool(failed.any().item()):
            solved = solved.clone()
            solved[failed] = torch.linalg.lstsq(gram[failed], rhs[failed]).solution
        if not bool(torch.isfinite(solved).all().item()):
            raise T5BodyRouteConditioningError(
                "A shared-support coefficient solve produced non-finite values."
            )
        coefficients[rows, :width] = solved.squeeze(2)
        proxy = diagonal.max(dim=1).values / diagonal.min(dim=1).values.clamp_min(
            torch.finfo(dtype).tiny
        )
        condition_proxy = max(condition_proxy, float(proxy.max().item()))
    return coefficients, condition_proxy


def _dense_coefficients(
    *,
    supports: Any,
    lengths: Any,
    coefficients: Any,
    width: int,
) -> Any:
    """Scatter the k-sparse codes into a dense ``(kept, dropped)`` matrix."""

    torch = _torch()
    dropped, steps = map(int, supports.shape)
    dense = torch.zeros(
        (width, dropped), dtype=coefficients.dtype, device=coefficients.device
    )
    valid = supports >= 0
    if not bool(valid.any().item()):
        return dense
    rows = supports.clamp_min(0)[valid]
    columns = (
        torch.arange(dropped, device=coefficients.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(dropped, steps)[valid]
    )
    dense.index_put_((rows, columns), coefficients[valid], accumulate=False)
    return dense


def _write_side_solve(
    *,
    kept_gram: Any,
    cross_gram: Any,
    dropped_gram: Any,
    metric_gram: Any,
    supports: Any,
    lengths: Any,
    initial: Any,
    ridge: float,
    max_sweeps: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Refit the shared supports against the induced-output objective.

    ``J(C) = Σ_{d,e} G_{de} ⟨r_d, r_e⟩`` with ``r_d = a_d − A_S c_d`` and
    ``G = M_drop M_dropᵀ``.  A diagonal ``G`` decouples the channels and the
    read solve is already optimal; a coupled ``G`` gets bounded damped Jacobi
    sweeps seeded from the read solve, each accepted only if the objective
    strictly decreases.
    """

    torch = _torch()
    dtype = kept_gram.dtype
    dropped = int(cross_gram.shape[1])
    coupling = torch.diagonal(metric_gram)
    if dropped > 1:
        off_diagonal = float(
            (metric_gram - torch.diag(coupling)).abs().max().item()
        )
        scale = float(coupling.abs().max().item())
        coupled = off_diagonal > scale * float(torch.finfo(dtype).eps) * dropped
    else:
        coupled = False

    def objective(dense: Any) -> float:
        residual_gram = (
            dropped_gram
            - dense.transpose(0, 1) @ cross_gram
            - cross_gram.transpose(0, 1) @ dense
            + dense.transpose(0, 1) @ kept_gram @ dense
        )
        return float((residual_gram * metric_gram).sum().item())

    dense = _dense_coefficients(
        supports=supports,
        lengths=lengths,
        coefficients=initial,
        width=int(kept_gram.shape[0]),
    )
    start_objective = objective(dense)
    sweeps: list[dict[str, Any]] = []
    coefficients = initial
    best = start_objective
    if not coupled or max_sweeps <= 0 or dropped == 0:
        return coefficients, MappingProxyType(
            {
                "algorithm": T5_WRITE_SOLVE_ALGORITHM,
                "coupled": bool(coupled),
                "sweeps_run": 0,
                "sweeps_accepted": 0,
                "initial_objective": start_objective,
                "final_objective": start_objective,
                "objective_improved": False,
                "sweep_records": [],
            }
        )

    tiny = float(torch.finfo(dtype).tiny)
    usable = coupling > 0
    for sweep in range(max_sweeps):
        # Block-Newton right-hand side: -0.5 dJ/dc_d = Aᵀ(EG)[S_d, d], and the
        # block Hessian is 2 G_dd A_SᵀA_S, so the per-channel system divides
        # the gradient by its own diagonal coupling weight.
        projected = cross_gram - kept_gram @ dense
        rhs = (projected @ metric_gram) / coupling.clamp_min(tiny).unsqueeze(0)
        rhs = torch.where(usable.unsqueeze(0), rhs, torch.zeros_like(rhs))
        candidate, _proxy = _grouped_support_solve(
            kept_gram=kept_gram,
            right_hand_side=rhs,
            supports=supports,
            lengths=lengths,
            ridge=ridge,
        )
        # ``candidate`` solves the *correction* system; the Jacobi update is
        # the damped step from the current coefficients toward it.
        accepted = False
        chosen_damping = 0.0
        chosen_objective = best
        for damping in _DAMPING_LADDER:
            trial = coefficients + damping * candidate
            trial_dense = _dense_coefficients(
                supports=supports,
                lengths=lengths,
                coefficients=trial,
                width=int(kept_gram.shape[0]),
            )
            value = objective(trial_dense)
            if math.isfinite(value) and value < best:
                coefficients = trial
                dense = trial_dense
                best = value
                accepted = True
                chosen_damping = damping
                chosen_objective = value
                break
        sweeps.append(
            {
                "sweep": sweep,
                "accepted": accepted,
                "damping": chosen_damping,
                "objective": chosen_objective,
            }
        )
        if not accepted:
            break
    return coefficients, MappingProxyType(
        {
            "algorithm": T5_WRITE_SOLVE_ALGORITHM,
            "coupled": True,
            "sweeps_run": len(sweeps),
            "sweeps_accepted": sum(1 for record in sweeps if record["accepted"]),
            "initial_objective": start_objective,
            "final_objective": best,
            "objective_improved": best < start_objective,
            "sweep_records": sweeps,
        }
    )


# ---------------------------------------------------------------------------
# The technique
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyRouteFold:
    """One folded body component plus its sparse residual route."""

    component_id: str
    component: str
    role: str
    site: str
    level: str
    parent_width: int
    kept_width: int
    block_count: int
    kept_channel_ids: Any
    support_offsets: Any
    support_channel_ids: Any
    read_coefficients: Any
    write_coefficients: Any
    installed_coefficients: str
    child_weight: Any
    expansion_route: Any
    calibration: Mapping[str, Any]
    solve: Mapping[str, Any]
    record: Mapping[str, Any]

    @property
    def record_sha256(self) -> str:
        return str(self.record["record_sha256"])

    def apply(self, rows: Any) -> Any:
        """Parent-equivalent map of the folded component.

        ``writer`` — full-width component rows in, component output out.
        ``reader`` — interface rows in, full-width component rows out.
        """

        torch = _torch()
        if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
            raise T5BodyRouteEvidenceError("apply expects a 2-D row matrix.")
        _no_autograd(rows, "apply rows")
        weight = self.child_weight
        with torch.no_grad():
            values = rows.detach().to(dtype=weight.dtype, device=weight.device)
            if self.role == "writer":
                if int(values.shape[1]) != self.parent_width:
                    raise T5BodyRouteGeometryError(
                        "writer apply expects full-width component rows."
                    )
                narrow = values.index_select(
                    1, self.kept_channel_ids.to(device=weight.device)
                )
                return narrow @ weight.transpose(0, 1)
            narrow = values @ weight.transpose(0, 1)
            blocks = self.block_count
            if blocks == 1:
                return narrow @ self.expansion_route
            parts = narrow.reshape(int(narrow.shape[0]), blocks, self.kept_width)
            expanded = parts @ self.expansion_route
            return expanded.reshape(int(narrow.shape[0]), blocks * self.parent_width)


def fit_body_component_routes(
    *,
    parent_weight: Any,
    component_rows: Any,
    component: str,
    level: str,
    site: str = "global",
    role: str | None = None,
    kept_width: int | None = None,
    block_count: int = 1,
    active_k: int = 8,
    channel_selection: str = "evidence-energy",
    kept_channel_ids: Sequence[int] | None = None,
    write_metric: Any = None,
    install_coefficients: str | None = None,
    refit_ridge_relative: float = 1e-10,
    write_ridge_relative: float = 1e-10,
    write_solve_sweeps: int = 4,
    pivot_tolerance: float = 1e-10,
    residual_floor: float = 1e-12,
    channels_per_chunk: int = 256,
    heldout_component_rows: Any = None,
    solve_dtype: Any = None,
    solve_device: Any = "cpu",
) -> BodyRouteFold:
    """Fold one body component by contracting its channel axis with T5 routes.

    ``parent_weight`` is in ``torch.nn.Linear`` orientation
    ``(out_features, in_features)``.  ``component_rows`` are the sealed
    evidence activations on the *component axis*: a writer's input channels or
    a reader's output channels.
    """

    torch = _torch()
    dtype = torch.float64 if solve_dtype is None else solve_dtype
    component_id, component_name, resolved_role = resolve_component(component, role)
    site_name = _text(site, "site")
    level_name = _text(level, "level").upper()
    blocks = _positive_int(block_count, "block_count")
    if resolved_role == "writer" and blocks != 1:
        raise T5BodyRouteGeometryError(
            "block_count applies to reader components only; a writer reads a "
            "single component axis."
        )
    if channel_selection not in _CHANNEL_SELECTION_POLICIES:
        raise T5BodyRouteGeometryError(
            "channel_selection must be one of "
            + ", ".join(_CHANNEL_SELECTION_POLICIES)
        )
    install = install_coefficients
    if install is None:
        install = "write" if resolved_role == "writer" else "read"
    if install not in ("read", "write"):
        raise T5BodyRouteGeometryError(
            "install_coefficients must be 'read' or 'write'."
        )

    weight = _matrix(parent_weight, "parent_weight", dtype=dtype, device=solve_device)
    rows = _matrix(
        component_rows, "component_rows", dtype=dtype, device=solve_device
    )
    out_features, in_features = map(int, weight.shape)
    if resolved_role == "writer":
        width = in_features
    else:
        if out_features % blocks:
            raise T5BodyRouteGeometryError(
                f"parent_weight out_features={out_features} is not divisible by "
                f"block_count={blocks}."
            )
        width = out_features // blocks
    if int(rows.shape[1]) != width:
        raise T5BodyRouteEvidenceError(
            f"component_rows width {int(rows.shape[1])} does not match the "
            f"{component_name} component axis width {width}."
        )
    if int(rows.shape[0]) < 1:
        raise T5BodyRouteEvidenceError("component_rows carries no evidence rows.")

    keep = resolve_kept_width(
        level=level_name, component_width=width, kept_width=kept_width
    )
    kept_ids = select_kept_channels(
        rows,
        kept_width=keep,
        policy=channel_selection,
        explicit_ids=kept_channel_ids,
    ).to(device=rows.device)
    membership = torch.zeros((width,), dtype=torch.bool, device=rows.device)
    membership[kept_ids] = True
    dropped_ids = torch.nonzero(~membership, as_tuple=False).flatten()
    dropped_count = int(dropped_ids.numel())

    steps = min(_positive_int(active_k, "active_k"), keep)
    ridge_ratio = _nonnegative_float(refit_ridge_relative, "refit_ridge_relative")
    write_ratio = _nonnegative_float(write_ridge_relative, "write_ridge_relative")
    pivot_floor = _nonnegative_float(pivot_tolerance, "pivot_tolerance")
    floor = _nonnegative_float(residual_floor, "residual_floor")
    chunk = _positive_int(channels_per_chunk, "channels_per_chunk")
    sweeps = int(write_solve_sweeps)
    if isinstance(write_solve_sweeps, bool) or sweeps < 0:
        raise T5BodyRouteGeometryError("write_solve_sweeps must be >= 0.")

    with torch.no_grad():
        kept_columns = rows.index_select(1, kept_ids)
        dropped_columns = rows.index_select(1, dropped_ids)
        kept_gram = kept_columns.transpose(0, 1) @ kept_columns
        cross_gram = kept_columns.transpose(0, 1) @ dropped_columns
        dropped_gram = dropped_columns.transpose(0, 1) @ dropped_columns
        dropped_energy = torch.diagonal(dropped_gram).clone()
        gram_scale = float(torch.diagonal(kept_gram).mean().item())
        if not math.isfinite(gram_scale) or gram_scale <= 0.0:
            raise T5BodyRouteEvidenceError(
                "The kept-channel evidence Gram has no energy."
            )
        ridge = ridge_ratio * gram_scale
        write_ridge = write_ratio * gram_scale

        supports = _omp_supports(
            kept_gram=kept_gram,
            cross_gram=cross_gram,
            dropped_energy=dropped_energy,
            active_k=steps,
            channels_per_chunk=chunk,
            pivot_tolerance=pivot_floor,
            residual_floor=floor,
        )
        selected = supports.selected.to(device=rows.device)
        lengths = supports.lengths.to(device=rows.device)

        read_coefficients, condition_proxy = _grouped_support_solve(
            kept_gram=kept_gram,
            right_hand_side=cross_gram,
            supports=selected,
            lengths=lengths,
            ridge=ridge,
        )

        if write_metric is None:
            if resolved_role == "writer":
                metric = weight.transpose(0, 1).contiguous()
            else:
                metric = torch.eye(width, dtype=dtype, device=rows.device)
            metric_source = (
                "parent-writer-rows" if resolved_role == "writer" else "identity"
            )
        else:
            metric = _matrix(
                write_metric, "write_metric", dtype=dtype, device=solve_device
            )
            if int(metric.shape[0]) != width:
                raise T5BodyRouteGeometryError(
                    "write_metric rows must index the component axis."
                )
            metric_source = "caller-supplied"
        metric_dropped = metric.index_select(0, dropped_ids)
        metric_gram = metric_dropped @ metric_dropped.transpose(0, 1)

        write_coefficients, write_report = _write_side_solve(
            kept_gram=kept_gram,
            cross_gram=cross_gram,
            dropped_gram=dropped_gram,
            metric_gram=metric_gram,
            supports=selected,
            lengths=lengths,
            initial=read_coefficients,
            ridge=write_ridge,
            max_sweeps=sweeps,
        )

        installed = read_coefficients if install == "read" else write_coefficients
        dense_read = _dense_coefficients(
            supports=selected,
            lengths=lengths,
            coefficients=read_coefficients,
            width=keep,
        )
        dense_installed = _dense_coefficients(
            supports=selected,
            lengths=lengths,
            coefficients=installed,
            width=keep,
        )

        if resolved_role == "writer":
            parent_math = weight.transpose(0, 1)
            child_math = parent_math.index_select(0, kept_ids) + (
                dense_installed @ parent_math.index_select(0, dropped_ids)
            )
            child_weight = child_math.transpose(0, 1).contiguous()
            expansion = torch.zeros((keep, width), dtype=dtype, device=rows.device)
            expansion[torch.arange(keep, device=rows.device), kept_ids] = 1.0
            expansion[:, dropped_ids] = dense_installed
        else:
            gather = torch.cat(
                [kept_ids + index * width for index in range(blocks)], dim=0
            )
            child_weight = weight.index_select(0, gather).contiguous()
            expansion = torch.zeros((keep, width), dtype=dtype, device=rows.device)
            expansion[torch.arange(keep, device=rows.device), kept_ids] = 1.0
            expansion[:, dropped_ids] = dense_installed

        reconstruction = kept_columns @ dense_read
        read_relative = (
            _relative_l2(reconstruction - dropped_columns, dropped_columns)
            if dropped_count
            else 0.0
        )
        installed_reconstruction = kept_columns @ dense_installed
        parent_math = weight.transpose(0, 1)
        if resolved_role == "writer":
            parent_output = rows @ parent_math
            child_output = kept_columns @ child_weight.transpose(0, 1)
        else:
            expanded = kept_columns @ expansion
            parent_output = rows
            child_output = expanded
        component_relative = _relative_l2(
            child_output - parent_output, parent_output
        )
        component_cosine = float(
            torch.nn.functional.cosine_similarity(
                child_output.reshape(1, -1), parent_output.reshape(1, -1)
            ).item()
        )

        heldout_report: dict[str, Any] | None = None
        if heldout_component_rows is not None:
            held = _matrix(
                heldout_component_rows,
                "heldout_component_rows",
                dtype=dtype,
                device=solve_device,
            )
            if int(held.shape[1]) != width:
                raise T5BodyRouteEvidenceError(
                    "heldout_component_rows width does not match the component "
                    "axis."
                )
            held_kept = held.index_select(1, kept_ids)
            held_dropped = held.index_select(1, dropped_ids)
            if resolved_role == "writer":
                held_parent = held @ parent_math
                held_child = held_kept @ child_weight.transpose(0, 1)
            else:
                held_parent = held
                held_child = held_kept @ expansion
            heldout_report = {
                "rows": int(held.shape[0]),
                "channel_reconstruction_relative_l2": (
                    _relative_l2(
                        held_kept @ dense_read - held_dropped, held_dropped
                    )
                    if dropped_count
                    else 0.0
                ),
                "component_write_relative_l2": _relative_l2(
                    held_child - held_parent, held_parent
                ),
                "component_write_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        held_child.reshape(1, -1), held_parent.reshape(1, -1)
                    ).item()
                ),
            }

        offsets = torch.zeros((dropped_count + 1,), dtype=torch.int64)
        offsets[1:] = torch.cumsum(lengths.to(device="cpu"), dim=0)
        flat_valid = selected >= 0
        support_channel_ids = kept_ids.to(device="cpu")[
            selected.to(device="cpu")[flat_valid.to(device="cpu")]
        ]
        read_flat = read_coefficients[flat_valid].to(device="cpu")
        write_flat = write_coefficients[flat_valid].to(device="cpu")

        # ``prior`` is the plain truncation of the parent — the child this cell
        # would carry with no route at all.  ``correction`` is what the sparse
        # residual route actually moved.
        if resolved_role == "writer":
            prior_weight = weight.index_select(1, kept_ids)
        else:
            prior_weight = weight.index_select(
                0,
                torch.cat(
                    [kept_ids + index * width for index in range(blocks)], dim=0
                ),
            )
        prior_norm = float(torch.linalg.vector_norm(prior_weight).item())
        correction_norm = float(
            torch.linalg.vector_norm(child_weight - prior_weight).item()
        )
        route_coefficient_norm = float(
            torch.linalg.vector_norm(dense_installed).item()
        )

    parent_parameters = out_features * in_features
    child_parameters = int(child_weight.numel())
    route_parameters = int(read_flat.numel()) if resolved_role == "reader" else 0
    calibration = {
        "no_contraction": dropped_count == 0,
        "dropped_channel_count": dropped_count,
        "fit_rows": int(rows.shape[0]),
        "channel_reconstruction_relative_l2": read_relative,
        "installed_channel_reconstruction_relative_l2": (
            _relative_l2(
                installed_reconstruction - dropped_columns, dropped_columns
            )
            if dropped_count
            else 0.0
        ),
        "component_write_relative_l2": component_relative,
        "component_write_cosine": component_cosine,
        "heldout": heldout_report,
        "exact_at_no_contraction": bool(
            dropped_count > 0 or component_relative == 0.0
        ),
    }
    solve = {
        "solver_path": f"{__name__}.fit_body_component_routes",
        "ridge_relative": ridge_ratio,
        "condition_number": condition_proxy,
        "condition_cap_saturated": False,
        "prior_norm": prior_norm,
        "correction_norm": correction_norm,
        "route_coefficient_norm": route_coefficient_norm,
        "requested_clamp_ratio": 0.0,
        "allowed_clamp_ratio": 0.0,
        "clamp_engaged": False,
        "omp": {
            "k": steps,
            "support_reuse_prefix_steps": 0,
            "double_mgs": True,
            **{
                key: value
                for key, value in supports.diagnostics.items()
                if key != "steps"
            },
            "steps": list(supports.diagnostics["steps"]),
        },
        "read_solve": {
            "algorithm": T5_READ_SOLVE_ALGORITHM,
            "ridge_relative": ridge_ratio,
            "gram_condition_proxy": condition_proxy,
        },
        "write_solve": dict(write_report),
        "write_metric_source": metric_source,
        "installed_coefficients": install,
    }
    body = {
        "schema": T5_BODY_ROUTE_RECORD_SCHEMA,
        "technique_id": T5_TECHNIQUE_ID,
        "technique": T5_TECHNIQUE_NAME,
        "component_id": component_id,
        "component": component_name,
        "role": resolved_role,
        "site": site_name,
        "level": level_name,
        "parent_width": width,
        "kept_width": keep,
        "dropped_width": dropped_count,
        "block_count": blocks,
        "channel_selection": channel_selection,
        "kept_channel_ids_sha256": _tensor_sha256(kept_ids.to(dtype=torch.float64)),
        "support_channel_ids_sha256": _tensor_sha256(
            support_channel_ids.to(dtype=torch.float64)
        ),
        "read_coefficients_sha256": _tensor_sha256(read_flat),
        "write_coefficients_sha256": _tensor_sha256(write_flat),
        "child_weight_sha256": _tensor_sha256(child_weight),
        "parent_weight_sha256": _tensor_sha256(weight),
        "component_rows_sha256": _tensor_sha256(rows),
        "parent_parameters": parent_parameters,
        "child_parameters": child_parameters,
        "route_parameters": route_parameters,
        "solve_placement": {
            "device": str(rows.device),
            "dtype": str(rows.dtype).replace("torch.", ""),
        },
        "calibration": calibration,
        "solve": solve,
        "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
    }
    record = _sealed(body)
    return BodyRouteFold(
        component_id=component_id,
        component=component_name,
        role=resolved_role,
        site=site_name,
        level=level_name,
        parent_width=width,
        kept_width=keep,
        block_count=blocks,
        kept_channel_ids=kept_ids.to(device="cpu"),
        support_offsets=offsets,
        support_channel_ids=support_channel_ids,
        read_coefficients=read_flat,
        write_coefficients=write_flat,
        installed_coefficients=install,
        child_weight=child_weight,
        expansion_route=expansion,
        calibration=MappingProxyType(calibration),
        solve=MappingProxyType(solve),
        record=MappingProxyType(record),
    )


# ---------------------------------------------------------------------------
# Registry entry (matrix-campaign design §4.3 dispatch table)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixTechniqueEntry:
    """One dispatch-table row for ``matrix_technique_registry``."""

    schema: str
    technique_id: str
    technique: str
    module: str
    components: tuple[str, ...]
    levels: tuple[str, ...]
    fit: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    describe: Callable[[], Mapping[str, Any]]


def _describe() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": MATRIX_TECHNIQUE_ENTRY_SCHEMA,
            "technique_id": T5_TECHNIQUE_ID,
            "technique": T5_TECHNIQUE_NAME,
            "module": __name__,
            "record_schema": T5_BODY_ROUTE_RECORD_SCHEMA,
            "components": tuple(sorted(_COMPONENTS)),
            "component_names": MappingProxyType(
                {key: value[0] for key, value in _COMPONENTS.items()}
            ),
            "roles": MappingProxyType(
                {key: value[1] for key, value in _COMPONENTS.items()}
            ),
            "levels": ("M16", "M8", "M4", "X4"),
            "required_request_keys": (
                "parent_weight",
                "component_rows",
                "component",
                "level",
            ),
            "optional_request_keys": (
                "site",
                "role",
                "kept_width",
                "block_count",
                "active_k",
                "channel_selection",
                "kept_channel_ids",
                "write_metric",
                "install_coefficients",
                "refit_ridge_relative",
                "write_ridge_relative",
                "write_solve_sweeps",
                "pivot_tolerance",
                "residual_floor",
                "channels_per_chunk",
                "heldout_component_rows",
                "solve_dtype",
                "solve_device",
            ),
            "selection_algorithm": T5_SELECTION_ALGORITHM,
            "read_solve_algorithm": T5_READ_SOLVE_ALGORITHM,
            "write_solve_algorithm": T5_WRITE_SOLVE_ALGORITHM,
            "fold_only": True,
            "gradient_free": True,
        }
    )


_REQUEST_KEYS = frozenset(
    tuple(_describe()["required_request_keys"])
    + tuple(_describe()["optional_request_keys"])
)


def fit_from_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Registry-facing entry point: mapping in, mapping out."""

    if not isinstance(request, Mapping):
        raise T5BodyRouteError("A technique request must be a mapping.")
    unknown = sorted(set(request) - _REQUEST_KEYS)
    if unknown:
        raise T5BodyRouteError(
            "Unsupported T5 request keys: " + ", ".join(unknown)
        )
    missing = sorted(set(_describe()["required_request_keys"]) - set(request))
    if missing:
        raise T5BodyRouteError("Missing T5 request keys: " + ", ".join(missing))
    fold = fit_body_component_routes(**dict(request))
    return MappingProxyType(
        {
            "schema": T5_BODY_ROUTE_RECORD_SCHEMA,
            "technique_id": T5_TECHNIQUE_ID,
            "technique": T5_TECHNIQUE_NAME,
            "component": fold.component,
            "component_id": fold.component_id,
            "role": fold.role,
            "site": fold.site,
            "level": fold.level,
            "child_parameters": MappingProxyType({"weight": fold.child_weight}),
            "route": MappingProxyType(
                {
                    "kept_channel_ids": fold.kept_channel_ids,
                    "support_offsets": fold.support_offsets,
                    "support_channel_ids": fold.support_channel_ids,
                    "read_coefficients": fold.read_coefficients,
                    "write_coefficients": fold.write_coefficients,
                    "expansion_matrix": fold.expansion_route,
                    "installed_coefficients": fold.installed_coefficients,
                }
            ),
            "apply": fold.apply,
            "calibration": fold.calibration,
            "solve": fold.solve,
            "technique_record": fold.record,
            "fold": fold,
        }
    )


def matrix_technique_registry_entry() -> MatrixTechniqueEntry:
    """Build the registry entry the matrix-cell harness dispatches through."""

    description = _describe()
    return MatrixTechniqueEntry(
        schema=MATRIX_TECHNIQUE_ENTRY_SCHEMA,
        technique_id=T5_TECHNIQUE_ID,
        technique=T5_TECHNIQUE_NAME,
        module=__name__,
        components=tuple(description["components"]),
        levels=tuple(description["levels"]),
        fit=fit_from_request,
        describe=_describe,
    )


# ---------------------------------------------------------------------------
# Harness adapter: neural_forge.matrix_technique_registry contract
# ---------------------------------------------------------------------------
#
# The registry requires **parent-shaped** installed weights: the level is the
# width of the *stored* form, and the tensor that lands in the copy_exact child
# keeps the geometry its untouched neighbours were compiled against.  Both T5
# roles satisfy that exactly, not approximately:
#
# * writer — the child gathers only the kept input channels, so scattering the
#   child's columns back into the kept positions and zeroing the dropped ones
#   computes the identical function.  The dropped channels multiply zero.
# * reader — the child's narrow output is expanded by the sparse route before
#   the copy_exact consumer sees it, and expansion is linear, so the composed
#   map ``expansionᵀ @ child`` is a parent-shaped tensor computing the identical
#   function.
#
# ``stored_parameter_count`` is where the contraction is declared, per the
# registry's own rule.


def _cell_component_rows(request: Any, spec: Any, parent: Any, blocks: int) -> Any:
    """Sealed rows on the component's own channel axis, for either role."""

    torch = _torch()
    inputs = request.evidence.fit_rows(spec.input_endpoint).to(dtype=torch.float64)
    if spec.contraction_axis == "input":
        return inputs
    derived = inputs @ parent.transpose(0, 1)
    if blocks == 1:
        return derived
    # A block reader (conv in_proj emits B/C/x over one shared internal channel
    # axis) selects and routes on the internal channel, so every block's rows
    # stack into one dictionary and the route is shared by construction.
    count = int(derived.shape[0])
    width = int(derived.shape[1]) // blocks
    return (
        derived.reshape(count, blocks, width)
        .permute(1, 0, 2)
        .reshape(blocks * count, width)
        .contiguous()
    )


def _cell_heldout_rows(request: Any, spec: Any, parent: Any, blocks: int) -> Any:
    torch = _torch()
    if int(getattr(request.evidence, "heldout_row_count", 0)) <= 0:
        return None
    inputs = request.evidence.heldout_rows(spec.input_endpoint).to(
        dtype=torch.float64
    )
    if spec.contraction_axis == "input":
        return inputs
    derived = inputs @ parent.transpose(0, 1)
    if blocks == 1:
        return derived
    count = int(derived.shape[0])
    width = int(derived.shape[1]) // blocks
    return (
        derived.reshape(count, blocks, width)
        .permute(1, 0, 2)
        .reshape(blocks * count, width)
        .contiguous()
    )


def t5_body_route_technique(request: Any) -> Any:
    """T5 kernel for the matrix-cell harness: one body component, one site."""

    from .matrix_technique_registry import CellTechniqueError, CellTechniqueResult

    torch = _torch()
    spec = request.component
    if spec.contraction_axis not in ("input", "output"):
        raise CellTechniqueError(
            f"T5 body routes contract a linear channel axis; "
            f"{spec.component_id} has axis {spec.contraction_axis!r} and needs "
            "its own kernel (C04 FIR taps, C10 heads, C14 diagonal, and the "
            "C01/C02/C15 token routes are A9/A11's own already-sealed lane)."
        )
    if spec.input_endpoint is None:
        raise CellTechniqueError(
            f"{spec.component_id} has no sealed input endpoint to route from."
        )
    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach().to(dtype=torch.float64)
    blocks = 3 if spec.name == "conv_in_proj" else 1
    level_width = int(request.level_width)
    if level_width % blocks:
        raise CellTechniqueError(
            f"{spec.component_id} stores {blocks} projection blocks over one "
            f"internal channel axis; level width {level_width} is not "
            f"divisible by {blocks}."
        )
    kept_width = level_width // blocks

    try:
        rows = _cell_component_rows(request, spec, parent, blocks)
        heldout = _cell_heldout_rows(request, spec, parent, blocks)
        fold = fit_body_component_routes(
            parent_weight=parent,
            component_rows=rows,
            component=spec.component_id,
            level=request.cell.level,
            site=spec.site,
            kept_width=kept_width,
            block_count=blocks,
            active_k=int(request.knob("active_k", 8)),
            channel_selection=str(
                request.knob("channel_selection", "evidence-energy")
            ),
            write_solve_sweeps=int(request.knob("write_solve_sweeps", 4)),
            install_coefficients=request.knob("install_coefficients", None),
            refit_ridge_relative=float(
                request.knob("refit_ridge_relative", 1e-10)
            ),
            heldout_component_rows=heldout,
        )
    except T5BodyRouteError as exc:
        raise CellTechniqueError(f"{request.cell.text}: {exc}") from exc

    request.log(
        f"T5 {request.cell.text}: role={fold.role} kept={fold.kept_width} "
        f"dropped={fold.parent_width - fold.kept_width} blocks={blocks} "
        f"k={fold.solve['omp']['k']} rows={int(rows.shape[0])}"
    )

    kept_ids = fold.kept_channel_ids.to(device=parent.device)
    installed = torch.zeros_like(parent)
    if fold.role == "writer":
        installed[:, kept_ids] = fold.child_weight
        stored = int(fold.child_weight.numel())
    else:
        width = fold.parent_width
        expansion = fold.expansion_route.transpose(0, 1)
        for block in range(blocks):
            child_block = fold.child_weight[
                block * fold.kept_width : (block + 1) * fold.kept_width, :
            ]
            installed[block * width : (block + 1) * width, :] = (
                expansion @ child_block
            )
        stored = int(fold.child_weight.numel()) + int(
            fold.read_coefficients.numel()
        )

    solve = dict(fold.solve)
    solve["technique_record_sha256"] = fold.record_sha256
    solve["calibration"] = dict(fold.calibration)
    solve["kept_width"] = fold.kept_width
    solve["block_count"] = blocks
    solve["route_support_entries"] = int(fold.support_channel_ids.numel())
    solve["component_rows_source"] = (
        spec.input_endpoint
        if spec.contraction_axis == "input"
        else f"{spec.input_endpoint} @ parent^T"
    )

    def predict(values: Any) -> Any:
        return fold.apply(values.to(dtype=torch.float64))

    return CellTechniqueResult(
        installed_weights={key: installed},
        stored_parameter_count=stored,
        solve=solve,
        predict=predict,
        artifact_identities={"t5_body_route_record_sha256": fold.record_sha256},
        notes=(
            f"T5 sparse residual-OMP {fold.role} routes: {fold.kept_width} of "
            f"{fold.parent_width} channels kept per block, k="
            f"{fold.solve['omp']['k']}, installed "
            f"{fold.installed_coefficients}-side coefficients."
        ),
    )


def _register_body_route_technique() -> None:
    """Wire T5 into the harness dispatch table at import time."""

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    for component in _COMPONENTS:
        register("T5", t5_body_route_technique, component=component, replace=True)


_register_body_route_technique()


__all__ = [
    "MATRIX_TECHNIQUE_ENTRY_SCHEMA",
    "T5_BODY_ROUTE_RECORD_SCHEMA",
    "T5_READ_SOLVE_ALGORITHM",
    "T5_SELECTION_ALGORITHM",
    "T5_TECHNIQUE_ID",
    "T5_TECHNIQUE_NAME",
    "T5_WRITE_SOLVE_ALGORITHM",
    "BodyRouteFold",
    "MatrixTechniqueEntry",
    "T5BodyRouteAuthenticationError",
    "T5BodyRouteConditioningError",
    "T5BodyRouteDisciplineError",
    "T5BodyRouteError",
    "T5BodyRouteEvidenceError",
    "T5BodyRouteGeometryError",
    "fit_body_component_routes",
    "fit_from_request",
    "matrix_technique_registry_entry",
    "resolve_component",
    "resolve_kept_width",
    "select_kept_channels",
    "t5_body_route_technique",
    "verify_sealed_body_route_record",
]
