"""T7/T8 — relation matching for contracted components.

Both techniques in this module fit a contracted component against a *relation*
on the sealed evidence rather than against the component's own output values.
That is the point: the campaign's measured failure signature is directional
(F-class physics — "direction dies"), and a relation objective is invariant to
the symmetry group that failure lives in.

T7 — attention-relation matching
--------------------------------
A head's whole contribution to the attention scores is the bilinear form
``B_h = Wq_h Wk_hᵀ``; the scores are ``X B_h Xᵀ / sqrt(d)`` and nothing else
about the Q/K factors is observable through the softmax.  Contracting the head
dimension to ``d' < d`` therefore is not "shrink Q and K", it is "find the best
rank-``d'`` bilinear form under the evidence metric", and the factors are
whatever gauge you pick afterwards.

* the evidence metric is ``M = XᵀX``; the score-kernel error is exactly
  ``‖M^{1/2}(B' − B)M^{1/2}‖_F``, so the ``R×R`` score matrix is never formed;
* Q and K are fitted *jointly* by alternating two-sided Stein ridge solves —
  the primitive the design names, ``(AᵀA) S (BBᵀ) + λS = Aᵀ E Bᵀ`` via two
  ``eigh`` — with a bounded iteration cap and a strict-decrease acceptance test
  on the unregularized score-kernel objective;
* grouped-query attention is respected: one child K per key/value head, fitted
  against the summed normal equations of its whole query group;
* the residual ``O(d')`` gauge freedom is fixed by the dormant Procrustes lane
  (``hf_cpcf_boundary_chart::_weighted_orthogonal_procrustes``), aligning the
  child factors to the parent's — a transformation that leaves ``B'``
  bit-invariant, so the gauge is free to choose and worth choosing;
* the softmax scale changes with ``d'``.  Unless the caller says otherwise the
  compensation ``sqrt(d'/d)`` is folded into the child Q so the child's
  ``q'k'ᵀ/sqrt(d')`` reproduces the parent's ``qkᵀ/sqrt(d)`` rather than a
  silently rescaled version of it.

T8 — Gram/relation matching at operator outputs
-----------------------------------------------
Fit the contracted component so the child's representation Gram on the
evidence rows matches the parent's.  Row-normalizing first makes the objective
scale-free, which is the whole bet: it scores direction agreement and refuses
to trade direction for magnitude.

The closed form is short.  With ``Ỹ = D Y`` the row-normalized parent output,
``K = ỸỸᵀ``, and any child ``Ỹ'`` of rank ``≤ r``, ``‖Ỹ'Ỹ'ᵀ − K‖_F`` is
minimized by the rank-``r`` truncation of ``Ỹ`` itself, in the parent's own
right-singular gauge — that is the Procrustes gauge choice, and it makes the
target ``T = Y V_r V_rᵀ``.  One ``D²``-weighted ridge solve from the
contracted design rows to ``T`` finishes it.

Two separate mechanisms are at work and the receipt distinguishes them,
because only one of them is always live:

* the ``D²`` row weighting — every evidence row enters with its *direction*
  at equal weight instead of its magnitude squared.  This is the scale-free
  part and it is live at every level.  Without it T8 would be numerically
  identical to a dense ridge whenever ``r`` reaches the parent's relation
  rank, which is the common case for a wide reader feeding a narrow writer;
  ``solve.row_weighting`` records which weighting ran.
* the rank truncation ``V_rV_rᵀ`` — live only when the child cannot carry the
  parent's relation rank.  ``solve.gram_truncation_active`` says so plainly
  rather than letting an inert projector look like a technique.

At no contraction, ``V_rV_rᵀ`` is the identity on the row space and the
weighted normal equations are exactly solvable, so the solve returns the
parent weight bit-for-bit.  ``normalization='none'`` turns the weighting off
and is the deliberate dense-ridge control arm.

Fold-only discipline
--------------------
Eigendecomposition, SVD, Procrustes, and ridge least squares on sealed
evidence.  No gradient, no optimizer, no parent forward.  Every input tensor
carrying autograd state is refused.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256


MATRIX_TECHNIQUE_ENTRY_SCHEMA = "neural-forge-matrix-technique-entry-v1"
T7_RELATION_RECORD_SCHEMA = "neural-forge-matrix-t7-attention-relation-record-v1"
T8_RELATION_RECORD_SCHEMA = "neural-forge-matrix-t8-gram-relation-record-v1"

T7_TECHNIQUE_ID = "T7"
T7_TECHNIQUE_NAME = "attention_relation_matching"
T8_TECHNIQUE_ID = "T8"
T8_TECHNIQUE_NAME = "gram_relation_matching"

T7_ALGORITHM = (
    "per-head-lifted-evidence-gram-alternating-two-sided-stein-ridge-"
    "procrustes-gauge-v1"
)
T8_ALGORITHM = (
    "row-normalized-output-gram-eigen-truncation-parent-gauge-ridge-refit-v1"
)
STEIN_ALGORITHM = "two-sided-stein-ridge-double-eigh-pseudoinverse-floor-v1"

#: Design §2.3 gives no numeric cap, so the module registers one: eight
#: alternating half-sweeps, each accepted only on strict objective decrease.
T7_DEFAULT_ITERATION_CAP = 8

_LEVEL_KEEP_FRACTION: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"M16": (15, 16), "M8": (7, 8), "M4": (3, 4), "M1": (1, 1)}
)

#: T7 is the bilinear Q/K cell; T8 covers every component whose output rows the
#: harness can hand over, including the short_conv writer — the measured wall.
_T7_COMPONENTS = ("C06", "C07", "C06+C07")
_T8_COMPONENTS = (
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


class RelationMatchError(ValueError):
    """A relation-matching request is outside the technique's envelope."""


class RelationMatchGeometryError(RelationMatchError):
    """Head geometry, level, or rank cannot be honoured."""


class RelationMatchEvidenceError(RelationMatchError):
    """Sealed evidence is missing, malformed, or carries no usable energy."""


class RelationMatchConditioningError(RelationMatchError):
    """An evidence Gram or a relation solve is degenerate."""


class RelationMatchDisciplineError(RelationMatchError):
    """A fold-only discipline rule was violated by the request."""


class RelationMatchAuthenticationError(RelationMatchError):
    """A sealed technique record failed its seal or schema check."""


# ---------------------------------------------------------------------------
# Validated primitives
# ---------------------------------------------------------------------------


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RelationMatchError(
            "PyTorch is required for T7/T8 relation matching."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RelationMatchGeometryError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RelationMatchGeometryError(f"{name} must be a non-negative integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise RelationMatchGeometryError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RelationMatchGeometryError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise RelationMatchGeometryError(f"{name} must be finite and non-negative.")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelationMatchError(f"{name} must be a nonempty string.")
    return value.strip()


def _no_autograd(value: Any, name: str) -> None:
    torch = _torch()
    if isinstance(value, torch.Tensor) and (
        value.requires_grad or value.grad_fn is not None
    ):
        raise RelationMatchDisciplineError(
            f"{name} carries autograd state; relation matching is fold-only."
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
        raise RelationMatchEvidenceError(f"{name} must be a nonempty real matrix.")
    _no_autograd(value, name)
    result = value.detach().to(device=device, dtype=dtype).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise RelationMatchEvidenceError(f"{name} is not finite.")
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
            raise RelationMatchError("Sealed records cannot carry non-finite numbers.")
        return value
    raise RelationMatchError(
        f"Sealed record contains non-JSON value {type(value).__name__}."
    )


def _sealed(body: Mapping[str, Any], field: str = "record_sha256") -> dict[str, Any]:
    value = _plain_json(body)
    return {**value, field: canonical_sha256(value)}


def verify_sealed_relation_record(
    value: Mapping[str, Any],
    *,
    schema: str,
    field: str = "record_sha256",
) -> dict[str, Any]:
    """Re-check a sealed technique record before the harness embeds it."""

    if not isinstance(value, Mapping):
        raise RelationMatchAuthenticationError("A technique record must be a mapping.")
    result = _plain_json(value)
    supplied = result.pop(field, None)
    if supplied != canonical_sha256(result):
        raise RelationMatchAuthenticationError("Technique record seal is invalid.")
    if result.get("schema") != schema:
        raise RelationMatchAuthenticationError(
            "Technique record schema is unsupported."
        )
    return {**result, field: supplied}


def resolve_kept_width(
    *,
    level: Any,
    component_width: int,
    kept_width: Any = None,
) -> int:
    """Resolve a campaign level into a kept width (shared level algebra)."""

    width = _positive_int(component_width, "component_width")
    name = _text(level, "level").upper()
    if kept_width is not None:
        resolved = _positive_int(kept_width, "kept_width")
    elif name in _LEVEL_KEEP_FRACTION:
        numerator, denominator = _LEVEL_KEEP_FRACTION[name]
        resolved = (width * numerator) // denominator
    elif name == "X4":
        raise RelationMatchGeometryError(
            "level='X4' has no closed-form width; supply kept_width."
        )
    else:
        raise RelationMatchGeometryError(
            f"level={name!r} is not a campaign level. Supported: "
            + ", ".join(sorted(_LEVEL_KEEP_FRACTION))
            + ", X4 (with kept_width)."
        )
    if resolved < 1 or resolved > width:
        raise RelationMatchGeometryError(
            f"level={name!r} yields kept_width={resolved} outside 1..{width}."
        )
    return resolved


def select_kept_channels(
    component_rows: Any,
    *,
    kept_width: int,
    explicit_ids: Sequence[int] | None = None,
) -> Any:
    """Deterministic evidence-energy channel keep set.

    This is byte-for-byte the same rule as
    ``matrix_technique_body_routes.select_kept_channels`` with
    ``policy='evidence-energy'``, restated here so the module stays
    self-contained.  A T5-vs-T8 contrast at the same level therefore folds the
    *same* kept channels and differs only in objective, which is the only way
    the wave-1 technique contrast means anything.  ``tests`` assert the two
    implementations agree.
    """

    torch = _torch()
    width = int(component_rows.shape[1])
    keep = _positive_int(kept_width, "kept_width")
    if keep > width:
        raise RelationMatchGeometryError("kept_width exceeds the component width.")
    if explicit_ids is not None:
        ids = sorted({int(value) for value in explicit_ids})
        if len(ids) != keep:
            raise RelationMatchGeometryError(
                f"kept_channel_ids holds {len(ids)} distinct ids but kept_width "
                f"is {keep}."
            )
        if ids and (ids[0] < 0 or ids[-1] >= width):
            raise RelationMatchGeometryError(
                "kept_channel_ids fall outside the component axis."
            )
        return torch.tensor(ids, dtype=torch.int64)
    energy = (component_rows * component_rows).sum(dim=0).to(dtype=torch.float64)
    order = torch.argsort(-energy, stable=True)
    return torch.sort(order[:keep]).values


# ---------------------------------------------------------------------------
# The shared primitive: two-sided Stein ridge
# ---------------------------------------------------------------------------


def _symmetric_eigh(value: Any, name: str) -> tuple[Any, Any]:
    torch = _torch()
    symmetric = (value + value.transpose(0, 1)) * 0.5
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    except RuntimeError as exc:  # pragma: no cover - LAPACK failure boundary
        raise RelationMatchConditioningError(
            f"The symmetric eigendecomposition of {name} failed."
        ) from exc
    if not bool(torch.isfinite(eigenvalues).all().item()):
        raise RelationMatchConditioningError(f"{name} has non-finite spectrum.")
    return eigenvalues.clamp_min(0.0), eigenvectors


def solve_two_sided_stein_ridge_gram(
    left_gram: Any,
    right_gram: Any,
    cross: Any,
    *,
    ridge: float = 0.0,
    left_eigen: tuple[Any, Any] | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Solve ``L S R + λS = cross`` for ``S`` with two ``eigh`` calls.

    ``left_gram`` is ``AᵀA``, ``right_gram`` is ``BBᵀ``, ``cross`` is
    ``Aᵀ E Bᵀ`` — the design's literal Stein normal equations.  Directions
    whose combined eigenvalue product falls below a machine-epsilon floor are
    resolved with pseudo-inverse semantics (coefficient set to zero) rather
    than by inventing a regularizer, so ``ridge=0`` stays exact wherever the
    right-hand side also vanishes.

    ``left_eigen`` lets a caller hand over an already-computed
    ``eigh(left_gram)``.  T7 reuses one residual-width decomposition across
    every head and every alternating half-sweep; recomputing it would put an
    ``O(H³)`` term inside the iteration and blow the per-cell budget.
    """

    torch = _torch()
    if left_eigen is None:
        left_values, left_vectors = _symmetric_eigh(left_gram, "left_gram")
    else:
        left_values, left_vectors = left_eigen
    right_values, right_vectors = _symmetric_eigh(right_gram, "right_gram")
    lam = _nonnegative_float(ridge, "ridge")
    rotated = left_vectors.transpose(0, 1) @ cross @ right_vectors
    denominator = left_values.unsqueeze(1) * right_values.unsqueeze(0) + lam
    scale = float(denominator.max().item()) if denominator.numel() else 0.0
    floor = float(torch.finfo(left_gram.dtype).eps) * max(scale, 0.0) * max(
        int(denominator.shape[0]), int(denominator.shape[1])
    )
    usable = denominator > floor
    solved = torch.where(
        usable, rotated / denominator.clamp_min(torch.finfo(left_gram.dtype).tiny),
        torch.zeros_like(rotated),
    )
    result = left_vectors @ solved @ right_vectors.transpose(0, 1)
    if not bool(torch.isfinite(result).all().item()):
        raise RelationMatchConditioningError(
            "The two-sided Stein ridge solve produced non-finite values."
        )
    report = {
        "algorithm": STEIN_ALGORITHM,
        "ridge": lam,
        "pseudoinverse_floor": floor,
        "suppressed_direction_count": int((~usable).sum().item()),
        "left_rank": int((left_values > floor).sum().item()),
        "right_rank": int((right_values > floor).sum().item()),
    }
    return result, MappingProxyType(report)


def solve_two_sided_stein_ridge(
    left: Any,
    right: Any,
    target: Any,
    *,
    ridge: float = 0.0,
) -> tuple[Any, Mapping[str, Any]]:
    """Minimize ``‖left S right − target‖² + ridge‖S‖²`` in closed form."""

    torch = _torch()
    if int(left.shape[0]) != int(target.shape[0]):
        raise RelationMatchGeometryError("left and target row counts differ.")
    if int(right.shape[1]) != int(target.shape[1]):
        raise RelationMatchGeometryError("right and target column counts differ.")
    left_gram = left.transpose(0, 1) @ left
    right_gram = right @ right.transpose(0, 1)
    cross = left.transpose(0, 1) @ target @ right.transpose(0, 1)
    del torch
    return solve_two_sided_stein_ridge_gram(
        left_gram, right_gram, cross, ridge=ridge
    )


def _matrix_square_roots(
    gram: Any,
    *,
    name: str,
    eigen: tuple[Any, Any] | None = None,
) -> tuple[Any, Any, int, float]:
    """Return ``(M^{1/2}, M^{-1/2}, rank, floor)`` for a PSD evidence Gram."""

    torch = _torch()
    if eigen is None:
        values, vectors = _symmetric_eigh(gram, name)
    else:
        values, vectors = eigen
    largest = float(values.max().item()) if values.numel() else 0.0
    if not math.isfinite(largest) or largest <= 0.0:
        raise RelationMatchEvidenceError(f"{name} carries no evidence energy.")
    floor = float(torch.finfo(gram.dtype).eps) * largest * int(values.numel())
    usable = values > floor
    roots = torch.where(usable, torch.sqrt(values), torch.zeros_like(values))
    inverse_roots = torch.where(
        usable, 1.0 / torch.sqrt(values.clamp_min(floor)), torch.zeros_like(values)
    )
    forward = vectors @ torch.diag(roots) @ vectors.transpose(0, 1)
    backward = vectors @ torch.diag(inverse_roots) @ vectors.transpose(0, 1)
    return forward, backward, int(usable.sum().item()), floor


def _orthogonal_procrustes(cross: Any) -> Any:
    """Deterministic ``argmin_R ‖A R − B‖`` over orthogonal ``R``.

    A compact restatement of ``hf_cpcf_boundary_chart::
    _weighted_orthogonal_procrustes`` for the unweighted square case: the SVD
    rotation, with the sign of each singular direction pinned so a rank
    deficiency cannot make the gauge depend on LAPACK tie order.
    """

    torch = _torch()
    left, singular, right_h = torch.linalg.svd(cross, full_matrices=True)
    largest = float(singular.max().item()) if singular.numel() else 0.0
    tolerance = (
        max(int(cross.shape[0]), int(cross.shape[1]))
        * float(torch.finfo(cross.dtype).eps)
        * largest
    )
    rank = int((singular > tolerance).sum().item())
    pivots = left.abs().argmax(dim=0)
    signs = torch.sign(left[pivots, torch.arange(int(left.shape[1]))])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    if rank < int(singular.numel()):
        # Degenerate directions get the canonical sign convention rather than
        # whatever the decomposition happened to emit.
        left = left * signs.unsqueeze(0)
        right_h = right_h * signs.unsqueeze(1)
    return (left @ right_h).contiguous()


# ---------------------------------------------------------------------------
# T7 — attention relation matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionRelationFit:
    """Contracted per-head Q/K fitted to the parent's score kernels."""

    site: str
    level: str
    head_dim: int
    child_head_dim: int
    query_heads: int
    key_value_heads: int
    query_weight: Any
    key_weight: Any
    softmax_scale_compensation: float
    per_head: tuple[Mapping[str, Any], ...]
    realized: Mapping[str, Mapping[str, Any]]
    calibration: Mapping[str, Any]
    solve: Mapping[str, Any]
    record: Mapping[str, Any]

    @property
    def record_sha256(self) -> str:
        return str(self.record["record_sha256"])

    def score_kernel(self, rows: Any, *, head: int) -> Any:
        """Child score kernel ``(X Wq')(X Wk')ᵀ / sqrt(d')`` for one q head."""

        torch = _torch()
        _no_autograd(rows, "rows")
        with torch.no_grad():
            values = rows.detach().to(
                dtype=self.query_weight.dtype, device=self.query_weight.device
            )
            group = self.query_heads // self.key_value_heads
            width = self.child_head_dim
            query = self.query_weight[head * width : (head + 1) * width, :]
            key_head = head // group
            key = self.key_weight[
                key_head * width : (key_head + 1) * width, :
            ]
            return (values @ query.transpose(0, 1)) @ (
                values @ key.transpose(0, 1)
            ).transpose(0, 1) / math.sqrt(width)


def fit_attention_relation_match(
    *,
    query_weight: Any,
    key_weight: Any,
    evidence_rows: Any,
    head_dim: int,
    query_heads: int,
    key_value_heads: int,
    level: str,
    site: str = "global",
    child_head_dim: int | None = None,
    ridge_relative: float = 0.0,
    iteration_cap: int = T7_DEFAULT_ITERATION_CAP,
    procrustes_gauge: bool = True,
    balance_factors: bool = True,
    compensate_softmax_scale: bool = True,
    realize_sides: Sequence[str] = (),
    heldout_rows: Any = None,
    solve_dtype: Any = None,
    solve_device: Any = "cpu",
) -> AttentionRelationFit:
    """Fit contracted Q/K jointly so child score kernels match the parent's.

    ``realize_sides`` additionally solves, for each named side, a *parent-shaped*
    weight for that side alone whose bilinear form against the **untouched
    parent partner** reproduces the jointly fitted rank-``d'`` score kernel.

    That is not a consolation prize, it is exact: the fitted kernel's row space
    already lies inside the parent partner's column space (it was built from
    it), so a single-side realization can carry the joint optimum.  It is what
    makes a one-component T7 cell honest — the neighbour stays bit-identical,
    the softmax scale stays at the parent's ``head_dim``, and the contraction
    shows up as the *rank* of the installed tensor rather than its shape.
    """

    torch = _torch()
    dtype = torch.float64 if solve_dtype is None else solve_dtype
    site_name = _text(site, "site")
    level_name = _text(level, "level").upper()
    heads = _positive_int(query_heads, "query_heads")
    kv_heads = _positive_int(key_value_heads, "key_value_heads")
    width = _positive_int(head_dim, "head_dim")
    if heads % kv_heads:
        raise RelationMatchGeometryError(
            f"query_heads={heads} is not divisible by key_value_heads={kv_heads}."
        )
    group = heads // kv_heads
    child_width = resolve_kept_width(
        level=level_name, component_width=width, kept_width=child_head_dim
    )
    cap = _nonnegative_int(iteration_cap, "iteration_cap")
    lam_ratio = _nonnegative_float(ridge_relative, "ridge_relative")

    query = _matrix(query_weight, "query_weight", dtype=dtype, device=solve_device)
    key = _matrix(key_weight, "key_weight", dtype=dtype, device=solve_device)
    rows = _matrix(evidence_rows, "evidence_rows", dtype=dtype, device=solve_device)
    hidden = int(rows.shape[1])
    if int(query.shape[0]) != heads * width or int(query.shape[1]) != hidden:
        raise RelationMatchGeometryError(
            f"query_weight must be ({heads * width}, {hidden})."
        )
    if int(key.shape[0]) != kv_heads * width or int(key.shape[1]) != hidden:
        raise RelationMatchGeometryError(
            f"key_weight must be ({kv_heads * width}, {hidden})."
        )

    with torch.no_grad():
        metric = rows.transpose(0, 1) @ rows
        metric_eigen = _symmetric_eigh(metric, "evidence Gram XᵀX")
        forward, backward, evidence_rank, floor = _matrix_square_roots(
            metric, name="evidence Gram XᵀX", eigen=metric_eigen
        )
        if evidence_rank < child_width:
            raise RelationMatchConditioningError(
                f"The evidence Gram has rank {evidence_rank}; a rank-"
                f"{child_width} score kernel is not identifiable from it."
            )
        lam = lam_ratio * float(torch.diagonal(metric).mean().item())

        # The bilinear form ``B_h = Wq_h Wk_gᵀ`` is never materialized: every
        # quantity the objective and the Stein solves need factors through the
        # thin ``H×d`` blocks, which keeps the whole technique O(H²d) instead
        # of O(H³) per head per sweep.
        parent_query = [
            query[index * width : (index + 1) * width, :].transpose(0, 1)
            for index in range(heads)
        ]
        parent_key = [
            key[index * width : (index + 1) * width, :].transpose(0, 1)
            for index in range(kv_heads)
        ]
        lifted_query = [forward @ value for value in parent_query]
        lifted_key = [forward @ value for value in parent_key]
        metric_query = [metric @ value for value in parent_query]
        metric_key = [metric @ value for value in parent_key]
        reference = []
        for index in range(heads):
            left_gram = lifted_query[index].transpose(0, 1) @ lifted_query[index]
            right_gram = (
                lifted_key[index // group].transpose(0, 1)
                @ lifted_key[index // group]
            )
            reference.append(
                math.sqrt(max(float((left_gram * right_gram).sum().item()), 0.0))
            )

        def objective(
            child_query: list[Any], child_key: list[Any]
        ) -> tuple[float, list[float]]:
            errors: list[float] = []
            total = 0.0
            for index in range(heads):
                factor_q = child_query[index]
                factor_k = child_key[index // group]
                lifted_q = factor_q.transpose(0, 1) @ metric @ factor_q
                lifted_k = factor_k.transpose(0, 1) @ metric @ factor_k
                child_energy = float((lifted_q * lifted_k.transpose(0, 1)).sum().item())
                overlap = float(
                    (
                        (factor_q.transpose(0, 1) @ metric_query[index])
                        * (
                            factor_k.transpose(0, 1)
                            @ metric_key[index // group]
                        )
                    )
                    .sum()
                    .item()
                )
                squared = max(
                    child_energy - 2.0 * overlap + reference[index] ** 2, 0.0
                )
                value = math.sqrt(squared)
                errors.append(
                    value / reference[index] if reference[index] > 0 else 0.0
                )
                total += squared
            return total, errors

        child_key: list[Any] = []
        for kv_index in range(kv_heads):
            # The stacked lifted kernels share the right factor ``F Wk_g``, so
            # their right singular subspace is the top eigenspace of
            # ``A S Aᵀ`` with ``A = F Wk_g`` and ``S = Σ_h (F Wq_h)ᵀ(F Wq_h)``.
            anchor = lifted_key[kv_index]
            inner = torch.zeros((width, width), dtype=dtype, device=rows.device)
            for offset in range(group):
                factor = lifted_query[kv_index * group + offset]
                inner = inner + factor.transpose(0, 1) @ factor
            orthonormal, triangular = torch.linalg.qr(anchor, mode="reduced")
            small = triangular @ inner @ triangular.transpose(0, 1)
            _values, vectors = _symmetric_eigh(small, "lifted key kernel")
            basis = orthonormal @ vectors[:, width - child_width :]
            child_key.append(backward @ basis)

        stein_reports: list[Mapping[str, Any]] = []

        def solve_query(keys: list[Any]) -> list[Any]:
            result: list[Any] = []
            for index in range(heads):
                key_factor = keys[index // group]
                right_gram = key_factor.transpose(0, 1) @ metric @ key_factor
                cross = metric_query[index] @ (
                    metric_key[index // group].transpose(0, 1) @ key_factor
                )
                solved, report = solve_two_sided_stein_ridge_gram(
                    metric,
                    right_gram,
                    cross,
                    ridge=lam,
                    left_eigen=metric_eigen,
                )
                stein_reports.append(report)
                result.append(solved)
            return result

        def solve_key(queries: list[Any]) -> list[Any]:
            result: list[Any] = []
            for kv_index in range(kv_heads):
                right_gram = torch.zeros(
                    (child_width, child_width), dtype=dtype, device=rows.device
                )
                inner = torch.zeros(
                    (width, child_width), dtype=dtype, device=rows.device
                )
                for offset in range(group):
                    head_index = kv_index * group + offset
                    factor = queries[head_index]
                    right_gram = right_gram + factor.transpose(0, 1) @ metric @ factor
                    inner = (
                        inner
                        + metric_query[head_index].transpose(0, 1) @ factor
                    )
                cross = metric_key[kv_index] @ inner
                solved, report = solve_two_sided_stein_ridge_gram(
                    metric,
                    right_gram,
                    cross,
                    ridge=lam,
                    left_eigen=metric_eigen,
                )
                stein_reports.append(report)
                result.append(solved)
            return result

        child_query = solve_query(child_key)
        best_total, _initial_errors = objective(child_query, child_key)
        best_query = [value.clone() for value in child_query]
        best_key = [value.clone() for value in child_key]
        iterations: list[dict[str, Any]] = []
        for step in range(cap):
            candidate_key = solve_key(child_query)
            candidate_query = solve_query(candidate_key)
            total, _errors = objective(candidate_query, candidate_key)
            improved = math.isfinite(total) and total < best_total
            iterations.append(
                {
                    "iteration": step,
                    "objective": total,
                    "accepted": bool(improved),
                }
            )
            child_key = candidate_key
            child_query = candidate_query
            if improved:
                best_total = total
                best_query = [value.clone() for value in candidate_query]
                best_key = [value.clone() for value in candidate_key]
            else:
                break
        child_query = best_query
        child_key = best_key

        balancing: list[float] = []
        if balance_factors:
            for kv_index in range(kv_heads):
                key_norm = float(
                    torch.linalg.matrix_norm(child_key[kv_index], ord="fro").item()
                )
                query_norm = math.sqrt(
                    sum(
                        float(
                            torch.linalg.matrix_norm(
                                child_query[kv_index * group + offset], ord="fro"
                            ).item()
                        )
                        ** 2
                        for offset in range(group)
                    )
                    / group
                )
                if key_norm > 0.0 and query_norm > 0.0:
                    factor = math.sqrt(query_norm / key_norm)
                else:
                    factor = 1.0
                balancing.append(factor)
                child_key[kv_index] = child_key[kv_index] * factor
                for offset in range(group):
                    head_index = kv_index * group + offset
                    child_query[head_index] = child_query[head_index] / factor

        gauge_report: list[dict[str, Any]] = []
        if procrustes_gauge:
            for kv_index in range(kv_heads):
                stacked_child = torch.cat(
                    [child_query[kv_index * group + offset] for offset in range(group)]
                    + [child_key[kv_index]],
                    dim=0,
                )
                stacked_parent = torch.cat(
                    [
                        parent_query[kv_index * group + offset][:, :child_width]
                        for offset in range(group)
                    ]
                    + [parent_key[kv_index][:, :child_width]],
                    dim=0,
                )
                rotation = _orthogonal_procrustes(
                    stacked_child.transpose(0, 1) @ stacked_parent
                )
                deviation = float(
                    torch.linalg.matrix_norm(
                        rotation.transpose(0, 1) @ rotation
                        - torch.eye(child_width, dtype=dtype, device=rows.device),
                        ord="fro",
                    ).item()
                )
                if deviation > 1e-8:
                    raise RelationMatchConditioningError(
                        "The Procrustes gauge is not orthogonal; the child "
                        "factors would not preserve the score kernel."
                    )
                for offset in range(group):
                    head_index = kv_index * group + offset
                    child_query[head_index] = child_query[head_index] @ rotation
                child_key[kv_index] = child_key[kv_index] @ rotation
                gauge_report.append(
                    {
                        "key_value_head": kv_index,
                        "orthogonality_deviation": deviation,
                    }
                )

        gauge_total, gauge_errors = objective(child_query, child_key)
        # The invariance tolerance is absolute against the parent's own lifted
        # energy, not relative against the objective: at no contraction the
        # objective is ~0 and a relative test would trip on rounding noise.
        parent_energy = sum(value * value for value in reference)
        if abs(gauge_total - best_total) > 1e-9 * max(parent_energy, 1.0):
            raise RelationMatchConditioningError(
                "Gauge fixing changed the score-kernel objective; the "
                "transformation was not score-invariant."
            )

        sides = tuple(_text(value, "realize_sides entry") for value in realize_sides)
        unknown_sides = sorted(set(sides) - {"query", "key"})
        if unknown_sides:
            raise RelationMatchGeometryError(
                "realize_sides accepts 'query' and 'key'; got "
                + ", ".join(unknown_sides)
            )
        realized: dict[str, dict[str, Any]] = {}
        for side in sides:
            if side == "query":
                factors: list[Any] = []
                for index in range(heads):
                    partner = parent_key[index // group]
                    right_gram = partner.transpose(0, 1) @ metric @ partner
                    cross = (metric @ child_query[index]) @ (
                        child_key[index // group].transpose(0, 1) @ metric @ partner
                    )
                    solved, report = solve_two_sided_stein_ridge_gram(
                        metric,
                        right_gram,
                        cross,
                        ridge=lam,
                        left_eigen=metric_eigen,
                    )
                    stein_reports.append(report)
                    factors.append(solved)
                _total, errors = objective(factors, parent_key)
                published = torch.cat(
                    [value.transpose(0, 1) for value in factors], dim=0
                ).contiguous()
                stored = heads * child_width * (hidden + width)
            else:
                factors = []
                for kv_index in range(kv_heads):
                    right_gram = torch.zeros(
                        (width, width), dtype=dtype, device=rows.device
                    )
                    cross = torch.zeros(
                        (hidden, width), dtype=dtype, device=rows.device
                    )
                    lifted = metric @ child_key[kv_index]
                    for offset in range(group):
                        head_index = kv_index * group + offset
                        partner = parent_query[head_index]
                        right_gram = (
                            right_gram + partner.transpose(0, 1) @ metric @ partner
                        )
                        cross = cross + lifted @ (
                            child_query[head_index].transpose(0, 1) @ metric @ partner
                        )
                    solved, report = solve_two_sided_stein_ridge_gram(
                        metric,
                        right_gram,
                        cross,
                        ridge=lam,
                        left_eigen=metric_eigen,
                    )
                    stein_reports.append(report)
                    factors.append(solved)
                _total, errors = objective(parent_query, factors)
                published = torch.cat(
                    [value.transpose(0, 1) for value in factors], dim=0
                ).contiguous()
                stored = kv_heads * child_width * (hidden + width)
            realized[side] = {
                "weight": published,
                "partner": "parent",
                "stored_parameter_count": int(stored),
                "effective_rank": child_width,
                "score_kernel_relative_error_max": max(errors),
                "score_kernel_relative_error_mean": sum(errors) / heads,
                "per_head_score_kernel_relative_error": list(errors),
                "softmax_scale_unchanged": True,
            }

        compensation = (
            math.sqrt(child_width / width) if compensate_softmax_scale else 1.0
        )
        child_query = [value * compensation for value in child_query]

        published_query = torch.cat(
            [value.transpose(0, 1) for value in child_query], dim=0
        ).contiguous()
        published_key = torch.cat(
            [value.transpose(0, 1) for value in child_key], dim=0
        ).contiguous()

        heldout_report: dict[str, Any] | None = None
        if heldout_rows is not None:
            held = _matrix(
                heldout_rows, "heldout_rows", dtype=dtype, device=solve_device
            )
            if int(held.shape[1]) != hidden:
                raise RelationMatchEvidenceError(
                    "heldout_rows width does not match the residual width."
                )
            held_errors: list[float] = []
            for index in range(heads):
                parent_scores = (held @ parent_query[index]) @ (
                    held @ parent_key[index // group]
                ).transpose(0, 1)
                child_scores = (
                    (held @ (child_query[index] / compensation))
                    @ (held @ child_key[index // group]).transpose(0, 1)
                )
                denominator = float(
                    torch.linalg.matrix_norm(parent_scores, ord="fro").item()
                )
                held_errors.append(
                    float(
                        torch.linalg.matrix_norm(
                            child_scores - parent_scores, ord="fro"
                        ).item()
                    )
                    / denominator
                    if denominator > 0.0
                    else 0.0
                )
            heldout_report = {
                "rows": int(held.shape[0]),
                "score_kernel_relative_error_max": max(held_errors),
                "score_kernel_relative_error_mean": sum(held_errors) / heads,
            }

    # Mandatory receipt norms for a *relation* technique: the prior is the
    # parent's own lifted score kernel, and the correction is how far the
    # contracted kernel moved from it in the evidence metric.  Reporting factor
    # norms here would be meaningless — the factors carry a gauge the relation
    # does not see.
    prior_norm = math.sqrt(sum(value * value for value in reference))
    correction_norm = math.sqrt(max(gauge_total, 0.0))
    per_head = tuple(
        MappingProxyType(
            {
                "query_head": index,
                "key_value_head": index // group,
                "score_kernel_relative_error": gauge_errors[index],
                "parent_lifted_norm": reference[index],
            }
        )
        for index in range(heads)
    )
    calibration = {
        "no_contraction": child_width == width,
        "score_kernel_relative_error_max": max(gauge_errors),
        "score_kernel_relative_error_mean": sum(gauge_errors) / heads,
        "fit_rows": int(rows.shape[0]),
        "evidence_rank": evidence_rank,
        "heldout": heldout_report,
    }
    solve = {
        "solver_path": f"{__name__}.fit_attention_relation_match",
        "algorithm": T7_ALGORITHM,
        "stein_algorithm": STEIN_ALGORITHM,
        "ridge_relative": lam_ratio,
        "ridge_absolute": lam,
        "condition_number": evidence_rank,
        "condition_cap_saturated": False,
        "evidence_gram_floor": floor,
        "iteration_cap": cap,
        "iterations": iterations,
        "iterations_accepted": sum(
            1 for record in iterations if record["accepted"]
        ),
        # At zero ridge the closed-form initializer is already the global
        # optimum (Eckart-Young on the lifted kernel, shared right subspace
        # across a GQA group), so a zero here is a confirmed optimum, not a
        # stalled solver.  Under ridge the alternating lane is what moves.
        "alternating_improved_on_closed_form": any(
            record["accepted"] for record in iterations
        ),
        "final_objective": best_total,
        "procrustes_gauge": bool(procrustes_gauge),
        "procrustes": gauge_report,
        "balance_factors": bool(balance_factors),
        "balancing_factors": balancing,
        "softmax_scale_compensation": compensation,
        # Mandatory receipt fields (design 3): the "prior" a relation fit moves
        # away from is the parent's own factor pair, and the correction is what
        # the contracted kernel actually changed about it.
        "prior_norm": prior_norm,
        "correction_norm": correction_norm,
        "realized_sides": {
            side: {
                key: value
                for key, value in report.items()
                if key not in ("weight", "per_head_score_kernel_relative_error")
            }
            for side, report in realized.items()
        },
        "requested_clamp_ratio": 0.0,
        "allowed_clamp_ratio": 0.0,
        "clamp_engaged": False,
        "stein_solve_calls": len(stein_reports),
        "stein_suppressed_directions_max": max(
            (int(report["suppressed_direction_count"]) for report in stein_reports),
            default=0,
        ),
    }
    body = {
        "schema": T7_RELATION_RECORD_SCHEMA,
        "technique_id": T7_TECHNIQUE_ID,
        "technique": T7_TECHNIQUE_NAME,
        "component_id": "C06+C07",
        "component": "attn_qk_score_kernel",
        "site": site_name,
        "level": level_name,
        "head_dim": width,
        "child_head_dim": child_width,
        "query_heads": heads,
        "key_value_heads": kv_heads,
        "query_weight_sha256": _tensor_sha256(published_query),
        "key_weight_sha256": _tensor_sha256(published_key),
        "parent_query_weight_sha256": _tensor_sha256(query),
        "parent_key_weight_sha256": _tensor_sha256(key),
        "evidence_rows_sha256": _tensor_sha256(rows),
        "parent_parameters": int(query.numel() + key.numel()),
        "child_parameters": int(published_query.numel() + published_key.numel()),
        "solve_placement": {
            "device": str(rows.device),
            "dtype": str(rows.dtype).replace("torch.", ""),
        },
        "per_head": [dict(value) for value in per_head],
        "calibration": calibration,
        "solve": solve,
        "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
    }
    return AttentionRelationFit(
        site=site_name,
        level=level_name,
        head_dim=width,
        child_head_dim=child_width,
        query_heads=heads,
        key_value_heads=kv_heads,
        query_weight=published_query,
        key_weight=published_key,
        softmax_scale_compensation=compensation,
        per_head=per_head,
        realized=MappingProxyType(
            {side: MappingProxyType(report) for side, report in realized.items()}
        ),
        calibration=MappingProxyType(calibration),
        solve=MappingProxyType(solve),
        record=MappingProxyType(_sealed(body)),
    )


# ---------------------------------------------------------------------------
# T8 — Gram/relation matching at operator outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GramRelationFit:
    """A contracted component fitted to the parent's representation Gram."""

    component: str
    component_id: str
    site: str
    level: str
    kept_width: int
    parent_width: int
    kept_channel_ids: Any
    child_weight: Any
    gram_rank: int
    calibration: Mapping[str, Any]
    solve: Mapping[str, Any]
    record: Mapping[str, Any]

    @property
    def record_sha256(self) -> str:
        return str(self.record["record_sha256"])

    def apply(self, rows: Any) -> Any:
        """Child component output for contracted or full-width design rows."""

        torch = _torch()
        if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
            raise RelationMatchEvidenceError("apply expects a 2-D row matrix.")
        _no_autograd(rows, "apply rows")
        weight = self.child_weight
        with torch.no_grad():
            values = rows.detach().to(dtype=weight.dtype, device=weight.device)
            columns = int(values.shape[1])
            if columns == self.parent_width:
                values = values.index_select(
                    1, self.kept_channel_ids.to(device=weight.device)
                )
            elif columns != self.kept_width:
                raise RelationMatchGeometryError(
                    "apply expects contracted or full-width component rows."
                )
            return values @ weight.transpose(0, 1)


def _row_weights(
    output_rows: Any, *, normalization: str
) -> tuple[Any, Any, int]:
    torch = _torch()
    if normalization == "none":
        ones = torch.ones(
            (int(output_rows.shape[0]),),
            dtype=output_rows.dtype,
            device=output_rows.device,
        )
        return ones, ones, 0
    if normalization != "row_unit":
        raise RelationMatchGeometryError(
            "normalization must be 'row_unit' or 'none'."
        )
    norms = torch.linalg.vector_norm(output_rows, dim=1)
    zero = norms <= 0
    scale = torch.where(zero, torch.zeros_like(norms), 1.0 / norms.clamp_min(
        torch.finfo(output_rows.dtype).tiny
    ))
    inverse = torch.where(zero, torch.zeros_like(norms), norms)
    return scale, inverse, int(zero.sum().item())


def fit_gram_relation_match(
    *,
    parent_output_rows: Any,
    contracted_design_rows: Any | None = None,
    parent_input_rows: Any | None = None,
    kept_channel_ids: Sequence[int] | None = None,
    level: str | None = None,
    kept_width: int | None = None,
    component: str = "custom",
    component_id: str = "custom",
    site: str = "global",
    normalization: str = "row_unit",
    ridge_relative: float = 0.0,
    gram_rank: int | None = None,
    clamp_gram_rank: bool = False,
    heldout_design_rows: Any = None,
    heldout_output_rows: Any = None,
    solve_dtype: Any = None,
    solve_device: Any = "cpu",
) -> GramRelationFit:
    """Fit a contracted component to the parent's row Gram, scale-free.

    Either supply ``contracted_design_rows`` directly (the harness already
    holds the contracted conv state or the kept MLP channels), or supply
    ``parent_input_rows`` with a ``level``/``kept_width`` and let the module
    apply the shared evidence-energy keep rule.
    """

    torch = _torch()
    dtype = torch.float64 if solve_dtype is None else solve_dtype
    site_name = _text(site, "site")
    level_name = _text(level, "level").upper() if level is not None else "EXPLICIT"
    output = _matrix(
        parent_output_rows, "parent_output_rows", dtype=dtype, device=solve_device
    )
    if contracted_design_rows is not None:
        if parent_input_rows is not None:
            raise RelationMatchGeometryError(
                "Supply contracted_design_rows or parent_input_rows, not both."
            )
        design = _matrix(
            contracted_design_rows,
            "contracted_design_rows",
            dtype=dtype,
            device=solve_device,
        )
        parent_width = int(design.shape[1])
        keep = parent_width
        kept_ids = torch.arange(keep, dtype=torch.int64)
    else:
        if parent_input_rows is None:
            raise RelationMatchGeometryError(
                "Supply contracted_design_rows or parent_input_rows."
            )
        full = _matrix(
            parent_input_rows, "parent_input_rows", dtype=dtype, device=solve_device
        )
        parent_width = int(full.shape[1])
        if level is None and kept_width is None and kept_channel_ids is None:
            raise RelationMatchGeometryError(
                "parent_input_rows needs a level, kept_width, or "
                "kept_channel_ids to contract with."
            )
        if kept_channel_ids is not None:
            keep = len(sorted({int(value) for value in kept_channel_ids}))
        else:
            keep = resolve_kept_width(
                level=level_name if level is not None else "M1",
                component_width=parent_width,
                kept_width=kept_width,
            )
        kept_ids = select_kept_channels(
            full, kept_width=keep, explicit_ids=kept_channel_ids
        )
        design = full.index_select(1, kept_ids.to(device=full.device))
    if int(design.shape[0]) != int(output.shape[0]):
        raise RelationMatchEvidenceError(
            "Design and parent output rows disagree on the evidence row count."
        )
    out_features = int(output.shape[1])
    lam_ratio = _nonnegative_float(ridge_relative, "ridge_relative")

    with torch.no_grad():
        scale, inverse, zero_rows = _row_weights(output, normalization=normalization)
        if zero_rows == int(output.shape[0]):
            raise RelationMatchEvidenceError(
                "Every parent output row has zero norm; the Gram is empty."
            )
        normalized = output * scale.unsqueeze(1)
        output_gram = normalized.transpose(0, 1) @ normalized
        values, vectors = _symmetric_eigh(output_gram, "row-normalized output Gram")
        largest = float(values.max().item())
        if not math.isfinite(largest) or largest <= 0.0:
            raise RelationMatchEvidenceError(
                "The row-normalized output Gram carries no energy."
            )
        floor = float(torch.finfo(dtype).eps) * largest * out_features
        available = int((values > floor).sum().item())
        # The whole technique lives in this weighting.  Fitting against
        # ``D Y`` — every evidence row scaled to unit output norm — is what
        # makes the objective scale-free: each row contributes its *direction*
        # with equal weight instead of contributing its magnitude squared.
        # A rank truncation on top of that only bites when the child cannot
        # carry the parent's relation rank, and the receipt says which of the
        # two mechanisms actually did anything.
        weighted_design = design * (scale * scale).unsqueeze(1)
        design_gram = weighted_design.transpose(0, 1) @ design
        design_values, _design_vectors = _symmetric_eigh(
            design_gram, "row-weighted contracted design Gram"
        )
        design_largest = float(design_values.max().item())
        if not math.isfinite(design_largest) or design_largest <= 0.0:
            raise RelationMatchEvidenceError(
                "The contracted design rows carry no energy."
            )
        design_floor = float(torch.finfo(dtype).eps) * design_largest * keep
        design_rank = int((design_values > design_floor).sum().item())
        rank = min(keep, out_features, available, design_rank)
        requested_rank = rank
        rank_clamped = False
        if gram_rank is not None:
            requested_rank = _positive_int(gram_rank, "gram_rank")
            if requested_rank > rank:
                if not clamp_gram_rank:
                    raise RelationMatchGeometryError(
                        f"gram_rank={requested_rank} exceeds the achievable "
                        f"relation rank {rank}."
                    )
                # An output-axis level can ask for a stored width wider than the
                # component's own input, in which case the rank cap is simply
                # not the binding constraint.  Clamping and *saying so* beats
                # refusing a cell whose scale-free weighting still has content.
                rank_clamped = True
            else:
                rank = requested_rank
        if rank < 1:
            raise RelationMatchConditioningError(
                "No usable relation rank survives the evidence and the level."
            )
        # eigh returns ascending eigenvalues; the top-``rank`` right-singular
        # subspace of the row-normalized output is its trailing block.
        basis = vectors[:, out_features - rank :]
        projector = basis @ basis.transpose(0, 1)
        target = output @ projector

        lam = lam_ratio * float(torch.diagonal(design_gram).mean().item())
        system = design_gram + lam * torch.eye(keep, dtype=dtype, device=design.device)
        rhs = weighted_design.transpose(0, 1) @ target
        solved, info = torch.linalg.solve_ex(system, rhs, check_errors=False)
        if int(info.item()) != 0 or not bool(torch.isfinite(solved).all().item()):
            solved = torch.linalg.lstsq(system, rhs).solution
        if not bool(torch.isfinite(solved).all().item()):
            raise RelationMatchConditioningError(
                "The Gram-relation ridge refit produced non-finite values."
            )
        child_weight = solved.transpose(0, 1).contiguous()

        child_output = design @ solved
        child_scale, _child_inverse, _child_zero = _row_weights(
            child_output, normalization=normalization
        )
        child_normalized = child_output * child_scale.unsqueeze(1)
        parent_self = normalized.transpose(0, 1) @ normalized
        child_self = child_normalized.transpose(0, 1) @ child_normalized
        cross = normalized.transpose(0, 1) @ child_normalized
        parent_energy = float((parent_self * parent_self).sum().item())
        child_energy = float((child_self * child_self).sum().item())
        cross_energy = float((cross * cross).sum().item())
        gram_error = math.sqrt(
            max(child_energy - 2.0 * cross_energy + parent_energy, 0.0)
        )
        gram_relative = (
            gram_error / math.sqrt(parent_energy) if parent_energy > 0.0 else 0.0
        )
        cosine = torch.nn.functional.cosine_similarity(child_output, output, dim=1)
        parent_norms = torch.linalg.vector_norm(output, dim=1)
        child_norms = torch.linalg.vector_norm(child_output, dim=1)
        usable = parent_norms > 0
        magnitude = torch.where(
            usable,
            child_norms / parent_norms.clamp_min(torch.finfo(dtype).tiny),
            torch.ones_like(parent_norms),
        )
        value_relative = float(
            torch.linalg.matrix_norm(child_output - output, ord="fro").item()
        ) / max(
            float(torch.linalg.matrix_norm(output, ord="fro").item()),
            float(torch.finfo(dtype).tiny),
        )

        heldout_report: dict[str, Any] | None = None
        if heldout_design_rows is not None or heldout_output_rows is not None:
            if heldout_design_rows is None or heldout_output_rows is None:
                raise RelationMatchEvidenceError(
                    "Held-out scoring needs both design and output rows."
                )
            held_design = _matrix(
                heldout_design_rows,
                "heldout_design_rows",
                dtype=dtype,
                device=solve_device,
            )
            held_output = _matrix(
                heldout_output_rows,
                "heldout_output_rows",
                dtype=dtype,
                device=solve_device,
            )
            if int(held_design.shape[1]) == parent_width and keep != parent_width:
                held_design = held_design.index_select(
                    1, kept_ids.to(device=held_design.device)
                )
            if int(held_design.shape[1]) != keep:
                raise RelationMatchGeometryError(
                    "heldout_design_rows width does not match the contraction."
                )
            if int(held_design.shape[0]) != int(held_output.shape[0]):
                raise RelationMatchEvidenceError(
                    "Held-out design and output rows disagree on row count."
                )
            held_child = held_design @ solved
            held_cos = torch.nn.functional.cosine_similarity(
                held_child, held_output, dim=1
            )
            heldout_report = {
                "rows": int(held_design.shape[0]),
                "direction_cosine_mean": float(held_cos.mean().item()),
                "value_relative_l2": float(
                    torch.linalg.matrix_norm(
                        held_child - held_output, ord="fro"
                    ).item()
                )
                / max(
                    float(
                        torch.linalg.matrix_norm(held_output, ord="fro").item()
                    ),
                    float(torch.finfo(dtype).tiny),
                ),
            }

    calibration = {
        "no_contraction": keep == parent_width and rank >= available,
        "gram_relative_error": gram_relative,
        "direction_cosine_mean": float(cosine.mean().item()),
        "direction_cosine_min": float(cosine.min().item()),
        "magnitude_ratio_mean": float(magnitude.mean().item()),
        "value_relative_l2": value_relative,
        "fit_rows": int(design.shape[0]),
        "zero_norm_output_rows": zero_rows,
        "heldout": heldout_report,
    }
    solve = {
        "solver_path": f"{__name__}.fit_gram_relation_match",
        "algorithm": T8_ALGORITHM,
        "normalization": normalization,
        "ridge_relative": lam_ratio,
        "ridge_absolute": lam,
        # Bounded ratio: a rank-deficient design would otherwise seal an
        # infinity and fail the record's JSON contract.
        "condition_number": float(
            design_values.max().item()
            / max(
                float(design_values.min().item()),
                float(design_values.max().item()) * float(torch.finfo(dtype).eps),
            )
        ),
        "condition_cap_saturated": False,
        "row_weighting": (
            "inverse-parent-row-norm-squared"
            if normalization == "row_unit"
            else "uniform"
        ),
        "relation_rank": rank,
        "requested_relation_rank": requested_rank,
        "relation_rank_clamped": rank_clamped,
        "available_output_rank": available,
        "design_rank": design_rank,
        "gram_truncation_active": rank < available,
        "scale_free_objective": normalization == "row_unit",
        "requested_clamp_ratio": 0.0,
        "allowed_clamp_ratio": 0.0,
        "clamp_engaged": False,
        "prior_norm": float(
            torch.linalg.matrix_norm(output, ord="fro").item()
        ),
        "correction_norm": float(
            torch.linalg.matrix_norm(child_output - output, ord="fro").item()
        ),
    }
    body = {
        "schema": T8_RELATION_RECORD_SCHEMA,
        "technique_id": T8_TECHNIQUE_ID,
        "technique": T8_TECHNIQUE_NAME,
        "component_id": _text(component_id, "component_id"),
        "component": _text(component, "component"),
        "site": site_name,
        "level": level_name,
        "parent_width": parent_width,
        "kept_width": keep,
        "out_features": out_features,
        "kept_channel_ids_sha256": _tensor_sha256(
            kept_ids.to(dtype=torch.float64)
        ),
        "child_weight_sha256": _tensor_sha256(child_weight),
        "parent_output_rows_sha256": _tensor_sha256(output),
        "child_parameters": int(child_weight.numel()),
        "solve_placement": {
            "device": str(design.device),
            "dtype": str(design.dtype).replace("torch.", ""),
        },
        "calibration": calibration,
        "solve": solve,
        "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
    }
    return GramRelationFit(
        component=_text(component, "component"),
        component_id=_text(component_id, "component_id"),
        site=site_name,
        level=level_name,
        kept_width=keep,
        parent_width=parent_width,
        kept_channel_ids=kept_ids.to(device="cpu"),
        child_weight=child_weight,
        gram_rank=rank,
        calibration=MappingProxyType(calibration),
        solve=MappingProxyType(solve),
        record=MappingProxyType(_sealed(body)),
    )


# ---------------------------------------------------------------------------
# Registry entries (matrix-campaign design §4.3 dispatch table)
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


_T7_REQUIRED = (
    "query_weight",
    "key_weight",
    "evidence_rows",
    "head_dim",
    "query_heads",
    "key_value_heads",
    "level",
)
_T7_OPTIONAL = (
    "site",
    "child_head_dim",
    "ridge_relative",
    "iteration_cap",
    "procrustes_gauge",
    "balance_factors",
    "compensate_softmax_scale",
    "realize_sides",
    "heldout_rows",
    "solve_dtype",
    "solve_device",
)
_T8_REQUIRED = ("parent_output_rows",)
_T8_OPTIONAL = (
    "contracted_design_rows",
    "parent_input_rows",
    "kept_channel_ids",
    "level",
    "kept_width",
    "component",
    "component_id",
    "site",
    "normalization",
    "ridge_relative",
    "gram_rank",
    "clamp_gram_rank",
    "heldout_design_rows",
    "heldout_output_rows",
    "solve_dtype",
    "solve_device",
)


def _describe_t7() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": MATRIX_TECHNIQUE_ENTRY_SCHEMA,
            "technique_id": T7_TECHNIQUE_ID,
            "technique": T7_TECHNIQUE_NAME,
            "module": __name__,
            "record_schema": T7_RELATION_RECORD_SCHEMA,
            "components": _T7_COMPONENTS,
            "levels": ("M16", "M8", "M4", "X4"),
            "required_request_keys": _T7_REQUIRED,
            "optional_request_keys": _T7_OPTIONAL,
            "algorithm": T7_ALGORITHM,
            "stein_algorithm": STEIN_ALGORITHM,
            "iteration_cap_default": T7_DEFAULT_ITERATION_CAP,
            "joint_query_key_solve": True,
            "fold_only": True,
            "gradient_free": True,
        }
    )


def _describe_t8() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": MATRIX_TECHNIQUE_ENTRY_SCHEMA,
            "technique_id": T8_TECHNIQUE_ID,
            "technique": T8_TECHNIQUE_NAME,
            "module": __name__,
            "record_schema": T8_RELATION_RECORD_SCHEMA,
            "components": _T8_COMPONENTS,
            "levels": ("M16", "M8", "M4", "X4"),
            "required_request_keys": _T8_REQUIRED,
            "optional_request_keys": _T8_OPTIONAL,
            "algorithm": T8_ALGORITHM,
            "scale_free": True,
            "covers_short_conv_outputs": True,
            "fold_only": True,
            "gradient_free": True,
        }
    )


def _checked(
    request: Mapping[str, Any],
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise RelationMatchError("A technique request must be a mapping.")
    allowed = frozenset(required + optional)
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise RelationMatchError(
            f"Unsupported {label} request keys: " + ", ".join(unknown)
        )
    missing = sorted(set(required) - set(request))
    if missing:
        raise RelationMatchError(
            f"Missing {label} request keys: " + ", ".join(missing)
        )
    return dict(request)


def fit_attention_relation_from_request(
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Registry-facing T7 entry point: mapping in, mapping out."""

    fit = fit_attention_relation_match(
        **_checked(
            request, required=_T7_REQUIRED, optional=_T7_OPTIONAL, label="T7"
        )
    )
    return MappingProxyType(
        {
            "schema": T7_RELATION_RECORD_SCHEMA,
            "technique_id": T7_TECHNIQUE_ID,
            "technique": T7_TECHNIQUE_NAME,
            "component": "attn_qk_score_kernel",
            "component_id": "C06+C07",
            "site": fit.site,
            "level": fit.level,
            "child_parameters": MappingProxyType(
                {
                    "query_weight": fit.query_weight,
                    "key_weight": fit.key_weight,
                }
            ),
            "route": MappingProxyType(
                {
                    "head_dim": fit.child_head_dim,
                    "softmax_scale_compensation": (
                        fit.softmax_scale_compensation
                    ),
                }
            ),
            "apply": fit.score_kernel,
            "calibration": fit.calibration,
            "solve": fit.solve,
            "technique_record": fit.record,
            "fit": fit,
        }
    )


def fit_gram_relation_from_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Registry-facing T8 entry point: mapping in, mapping out."""

    fit = fit_gram_relation_match(
        **_checked(
            request, required=_T8_REQUIRED, optional=_T8_OPTIONAL, label="T8"
        )
    )
    return MappingProxyType(
        {
            "schema": T8_RELATION_RECORD_SCHEMA,
            "technique_id": T8_TECHNIQUE_ID,
            "technique": T8_TECHNIQUE_NAME,
            "component": fit.component,
            "component_id": fit.component_id,
            "site": fit.site,
            "level": fit.level,
            "child_parameters": MappingProxyType({"weight": fit.child_weight}),
            "route": MappingProxyType(
                {
                    "kept_channel_ids": fit.kept_channel_ids,
                    "relation_rank": fit.gram_rank,
                }
            ),
            "apply": fit.apply,
            "calibration": fit.calibration,
            "solve": fit.solve,
            "technique_record": fit.record,
            "fit": fit,
        }
    )


def matrix_technique_registry_entries() -> tuple[MatrixTechniqueEntry, ...]:
    """Build both relation-matching registry entries (T7 then T8)."""

    seven = _describe_t7()
    eight = _describe_t8()
    return (
        MatrixTechniqueEntry(
            schema=MATRIX_TECHNIQUE_ENTRY_SCHEMA,
            technique_id=T7_TECHNIQUE_ID,
            technique=T7_TECHNIQUE_NAME,
            module=__name__,
            components=tuple(seven["components"]),
            levels=tuple(seven["levels"]),
            fit=fit_attention_relation_from_request,
            describe=_describe_t7,
        ),
        MatrixTechniqueEntry(
            schema=MATRIX_TECHNIQUE_ENTRY_SCHEMA,
            technique_id=T8_TECHNIQUE_ID,
            technique=T8_TECHNIQUE_NAME,
            module=__name__,
            components=tuple(eight["components"]),
            levels=tuple(eight["levels"]),
            fit=fit_gram_relation_from_request,
            describe=_describe_t8,
        ),
    )


def matrix_technique_registry_entry(technique_id: str = T7_TECHNIQUE_ID):
    """Return one relation-matching entry by technique id."""

    wanted = _text(technique_id, "technique_id").upper()
    for entry in matrix_technique_registry_entries():
        if entry.technique_id == wanted:
            return entry
    raise RelationMatchError(
        f"{wanted!r} is not a relation-matching technique; expected T7 or T8."
    )


# ---------------------------------------------------------------------------
# Harness adapters: neural_forge.matrix_technique_registry contract
# ---------------------------------------------------------------------------


def _attention_geometry(request: Any, spec: Any) -> tuple[int, int, int, int]:
    """Head geometry for one attention site, from knobs with LFM2 defaults."""

    head_dim = int(request.knob("head_dim", 64))
    query_heads = int(request.knob("query_heads", 16))
    key_value_heads = int(request.knob("key_value_heads", 8))
    hidden = int(spec.parent_input_width)
    return head_dim, query_heads, key_value_heads, hidden


def t7_attention_relation_technique(request: Any) -> Any:
    """T7 kernel for the matrix-cell harness: one side of the QK score kernel.

    The design's cell is written ``C06+C07`` — Q and K fitted jointly — while
    the registry (correctly, per §1.2 and §5 rule 6) lets a cell install only
    the tensors its own component owns.  Both are satisfied at once, and not by
    compromise:

    the joint fit runs in full, and then the *installed* side is re-solved as a
    parent-shaped weight whose bilinear form against the **bit-identical parent
    partner** reproduces that jointly fitted kernel exactly.  The partner stays
    copy_exact, the softmax scale stays at the parent's ``head_dim``, and the
    contraction appears as the rank of the installed tensor.  Running the C06
    cell and the C07 cell therefore measures the same relation from two sides,
    which is more information than one two-component cell would have given.
    """

    from .matrix_technique_registry import CellTechniqueError, CellTechniqueResult

    torch = _torch()
    spec = request.component
    side = {"C06": "query", "C07": "key"}.get(spec.component_id)
    if side is None:
        raise CellTechniqueError(
            f"T7 is the bilinear Q/K cell; {spec.component_id} is not C06 or "
            "C07.  Every other relation cell is T8."
        )
    if spec.input_endpoint is None:
        raise CellTechniqueError(
            f"{spec.component_id} has no sealed input endpoint to relate on."
        )
    head_dim, query_heads, key_value_heads, hidden = _attention_geometry(
        request, spec
    )
    layer = spec.layer_index
    if layer is None:
        raise CellTechniqueError("T7 needs a layer index to find its partner.")
    query_key = f"model.layers.{layer}.self_attn.q_proj.weight"
    key_key = f"model.layers.{layer}.self_attn.k_proj.weight"
    for name in (query_key, key_key):
        if name not in request.parent_weights:
            raise CellTechniqueError(f"Parent state is missing {name} for T7.")
    installed_key = query_key if side == "query" else key_key
    if installed_key not in set(spec.weight_keys):
        raise CellTechniqueError(
            f"{installed_key} is not owned by {spec.component_id}."
        )

    rows = request.evidence.fit_rows(spec.input_endpoint).to(dtype=torch.float64)
    heldout = (
        request.evidence.heldout_rows(spec.input_endpoint).to(dtype=torch.float64)
        if int(getattr(request.evidence, "heldout_row_count", 0)) > 0
        else None
    )
    try:
        fit = fit_attention_relation_match(
            query_weight=request.parent_weights[query_key]
            .detach()
            .to(dtype=torch.float64),
            key_weight=request.parent_weights[key_key]
            .detach()
            .to(dtype=torch.float64),
            evidence_rows=rows,
            head_dim=head_dim,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            level=request.cell.level,
            site=spec.site,
            child_head_dim=int(request.level_width) // max(
                query_heads if side == "query" else key_value_heads, 1
            ),
            ridge_relative=float(request.knob("ridge_relative", 0.0)),
            iteration_cap=int(
                request.knob("iteration_cap", T7_DEFAULT_ITERATION_CAP)
            ),
            procrustes_gauge=bool(request.knob("procrustes_gauge", True)),
            balance_factors=bool(request.knob("balance_factors", True)),
            # The installed child keeps the parent's head_dim, so the parent's
            # softmax scale is already the right one; compensating would
            # rescale a kernel that needs no rescaling.
            compensate_softmax_scale=False,
            realize_sides=(side,),
            heldout_rows=heldout,
        )
    except RelationMatchError as exc:
        raise CellTechniqueError(f"{request.cell.text}: {exc}") from exc

    realization = fit.realized[side]
    installed = realization["weight"]
    parent = request.parent_weights[installed_key]
    if tuple(int(v) for v in installed.shape) != tuple(
        int(v) for v in parent.shape
    ):
        raise CellTechniqueError(
            f"T7 realized {installed_key} at {tuple(installed.shape)} against a "
            f"parent {tuple(parent.shape)}; check head_dim/query_heads knobs."
        )
    request.log(
        f"T7 {request.cell.text}: side={side} d'={fit.child_head_dim} of "
        f"{head_dim}, joint score err "
        f"{fit.calibration['score_kernel_relative_error_max']:.5f}, realized "
        f"{realization['score_kernel_relative_error_max']:.5f}"
    )

    solve = dict(fit.solve)
    solve["technique_record_sha256"] = str(fit.record["record_sha256"])
    solve["calibration"] = dict(fit.calibration)
    solve["installed_side"] = side
    solve["partner_component"] = "C07" if side == "query" else "C06"
    solve["partner_is_copy_exact"] = True
    solve["realized_score_kernel_relative_error_max"] = float(
        realization["score_kernel_relative_error_max"]
    )
    solve["omp"] = None

    return CellTechniqueResult(
        installed_weights={installed_key: installed},
        stored_parameter_count=int(realization["stored_parameter_count"]),
        solve=solve,
        predict=None,
        artifact_identities={
            "t7_attention_relation_record_sha256": str(
                fit.record["record_sha256"]
            )
        },
        notes=(
            f"T7 joint QK score-kernel fit at head_dim {fit.child_head_dim} of "
            f"{head_dim}, realized into {side} against the bit-identical parent "
            "partner; softmax scale unchanged."
        ),
    )


def t8_gram_relation_technique(request: Any) -> Any:
    """T8 kernel for the matrix-cell harness: Gram matching at any linear site.

    Both contraction axes are the same closed form with a different contracted
    object.  On the ``input`` axis the design rows are the component's kept
    input channels; on the ``output`` axis the design rows are the full input
    and the level caps the *relation rank* the child may carry.  Either way the
    materialized tensor is parent-shaped and ``stored_parameter_count`` carries
    the contraction.

    **The target.**  By default the kernel synthesizes its own target by
    applying the parent operator to the design rows, which is the right target
    when the rows *are* the parent's: "be this map where the parent's data
    lives".  Wave 11's ALS re-solve needs the other target -- the parent's own
    sealed write rows, against design rows captured through a partially folded
    child -- because that is the only objective that can *cancel* upstream
    drift rather than re-fit the operator to it (ADDENDUM 9 §2).  The knob
    ``relation_target_endpoint`` names the evidence endpoint the target rows
    come from; with the knob absent, this function is byte-for-byte the
    function every wave 1-10 receipt was sealed under, which a test pins.
    """

    from .matrix_technique_registry import CellTechniqueError, CellTechniqueResult

    torch = _torch()
    spec = request.component
    if spec.contraction_axis not in ("input", "output"):
        raise CellTechniqueError(
            f"T8 relates a linear map's outputs; {spec.component_id} has axis "
            f"{spec.contraction_axis!r} and needs its own kernel."
        )
    if spec.input_endpoint is None:
        raise CellTechniqueError(
            f"{spec.component_id} has no sealed input endpoint to relate on."
        )
    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach().to(dtype=torch.float64)
    rows = request.evidence.fit_rows(spec.input_endpoint).to(dtype=torch.float64)
    target_endpoint = request.knob("relation_target_endpoint", None)
    if target_endpoint is None:
        output = rows @ parent.transpose(0, 1)
    else:
        output = request.evidence.fit_rows(str(target_endpoint)).to(
            dtype=torch.float64
        )
        if int(output.shape[0]) != int(rows.shape[0]) or int(
            output.shape[1]
        ) != int(parent.shape[0]):
            raise CellTechniqueError(
                f"{request.cell.text}: target endpoint {target_endpoint!r} "
                f"supplies {tuple(int(v) for v in output.shape)} rows, which "
                f"does not match the design rows ({int(rows.shape[0])}) and "
                f"the component's write width ({int(parent.shape[0])}).  A "
                "target that is not row-aligned with the design is a fit "
                "against two different corpora."
            )
    level_width = int(request.level_width)
    normalization = str(request.knob("normalization", "row_unit"))
    ridge = float(request.knob("ridge_relative", 0.0))
    heldout_design = None
    heldout_output = None
    if int(getattr(request.evidence, "heldout_row_count", 0)) > 0:
        heldout_design = request.evidence.heldout_rows(spec.input_endpoint).to(
            dtype=torch.float64
        )
        heldout_output = (
            heldout_design @ parent.transpose(0, 1)
            if target_endpoint is None
            else request.evidence.heldout_rows(str(target_endpoint)).to(
                dtype=torch.float64
            )
        )

    try:
        if spec.contraction_axis == "input":
            fit = fit_gram_relation_match(
                parent_output_rows=output,
                parent_input_rows=rows,
                kept_width=level_width,
                component=spec.name,
                component_id=spec.component_id,
                site=spec.site,
                level=request.cell.level,
                normalization=normalization,
                ridge_relative=ridge,
                heldout_design_rows=heldout_design,
                heldout_output_rows=heldout_output,
            )
            kept_ids = fit.kept_channel_ids.to(device=parent.device)
            installed = torch.zeros_like(parent)
            installed[:, kept_ids] = fit.child_weight
            # The kept ids are load-bearing: without them the narrow child
            # weight does not say which columns of the parent-shaped tensor it
            # occupies.  ADDENDUM 6 §6.2 counts every tensor the child must keep,
            # index side included, so the selection rides in the count.
            stored = int(fit.child_weight.numel()) + int(kept_ids.numel())
        else:
            fit = fit_gram_relation_match(
                parent_output_rows=output,
                contracted_design_rows=rows,
                gram_rank=level_width,
                clamp_gram_rank=True,
                component=spec.name,
                component_id=spec.component_id,
                site=spec.site,
                level=request.cell.level,
                normalization=normalization,
                ridge_relative=ridge,
                heldout_design_rows=heldout_design,
                heldout_output_rows=heldout_output,
            )
            installed = fit.child_weight
            stored = fit.gram_rank * (
                int(parent.shape[0]) + int(parent.shape[1])
            )
    except RelationMatchError as exc:
        raise CellTechniqueError(f"{request.cell.text}: {exc}") from exc

    request.log(
        f"T8 {request.cell.text}: axis={spec.contraction_axis} "
        f"rank={fit.gram_rank} gram={fit.calibration['gram_relative_error']:.5f} "
        f"cos={fit.calibration['direction_cosine_mean']:.5f} "
        f"truncation={fit.solve['gram_truncation_active']} "
        f"target={'parent-operator-on-design' if target_endpoint is None else target_endpoint}"
    )

    solve = dict(fit.solve)
    solve["technique_record_sha256"] = str(fit.record["record_sha256"])
    solve["calibration"] = dict(fit.calibration)
    solve["omp"] = None
    solve["relation_target_endpoint"] = (
        None if target_endpoint is None else str(target_endpoint)
    )

    def predict(values: Any) -> Any:
        return values.to(dtype=torch.float64) @ installed.transpose(0, 1)

    return CellTechniqueResult(
        installed_weights={key: installed},
        stored_parameter_count=int(stored),
        solve=solve,
        predict=predict,
        artifact_identities={
            "t8_gram_relation_record_sha256": str(fit.record["record_sha256"])
        },
        notes=(
            f"T8 scale-free Gram matching on the {spec.contraction_axis} axis, "
            f"relation rank {fit.gram_rank}, row weighting "
            f"{fit.solve['row_weighting']}, truncation active "
            f"{fit.solve['gram_truncation_active']}."
        ),
    )


def _register_relation_techniques() -> None:
    """Wire T7 and T8 into the harness dispatch table at import time."""

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    for component in ("C06", "C07"):
        register(
            "T7", t7_attention_relation_technique, component=component, replace=True
        )
    for component in _T8_COMPONENTS:
        register("T8", t8_gram_relation_technique, component=component, replace=True)


_register_relation_techniques()


__all__ = [
    "MATRIX_TECHNIQUE_ENTRY_SCHEMA",
    "STEIN_ALGORITHM",
    "T7_ALGORITHM",
    "T7_DEFAULT_ITERATION_CAP",
    "T7_RELATION_RECORD_SCHEMA",
    "T7_TECHNIQUE_ID",
    "T7_TECHNIQUE_NAME",
    "T8_ALGORITHM",
    "T8_RELATION_RECORD_SCHEMA",
    "T8_TECHNIQUE_ID",
    "T8_TECHNIQUE_NAME",
    "AttentionRelationFit",
    "GramRelationFit",
    "MatrixTechniqueEntry",
    "RelationMatchAuthenticationError",
    "RelationMatchConditioningError",
    "RelationMatchDisciplineError",
    "RelationMatchError",
    "RelationMatchEvidenceError",
    "RelationMatchGeometryError",
    "fit_attention_relation_from_request",
    "fit_attention_relation_match",
    "fit_gram_relation_from_request",
    "fit_gram_relation_match",
    "matrix_technique_registry_entries",
    "matrix_technique_registry_entry",
    "resolve_kept_width",
    "select_kept_channels",
    "solve_two_sided_stein_ridge",
    "solve_two_sided_stein_ridge_gram",
    "t7_attention_relation_technique",
    "t8_gram_relation_technique",
    "verify_sealed_relation_record",
]
