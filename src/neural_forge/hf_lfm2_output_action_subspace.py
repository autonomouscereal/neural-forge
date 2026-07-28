"""Authenticated, fold-only POD for cached LFM2 branch output actions.

This module compiles a residual-output bottleneck from evidence that has
already been captured.  For every FIT row it consumes either a pre/post
operator residual pair or a directly supplied branch action.  In the former
case the action is derived exactly once in CPU FP64 as ``Y = post - pre``.

The weighted FIT action covariance is decomposed into a deterministic rank-h
orthonormal basis ``Q`` and the corresponding projector ``P = Q Q.T``.  FIT
is the *only* source used to construct ``Q``.  A disjoint held-out split is
optional and contributes diagnostics only.  Degenerate eigenspaces are
canonicalized from their projectors, so arbitrary eigenvector rotations and
signs do not enter the artifact identity.

No model object is accepted by this API.  Compilation and materialization do
not execute a model, capture a trace, build an autograd graph, run an
optimizer, train, prune, quantize, access the filesystem, or access a network.
The materialized form is two independent ordinary dense FP32/FP64 factors; it
does not retain the dense projector as a runtime parameter.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_OUTPUT_ACTION_ROWS_SCHEMA = "neural-forge-lfm2-output-action-rows-v1"
LFM2_OUTPUT_ACTION_EVIDENCE_SCHEMA = "neural-forge-lfm2-output-action-evidence-v1"
LFM2_OUTPUT_ACTION_SUBSPACE_SCHEMA = "neural-forge-lfm2-output-action-pod-subspace-v1"
LFM2_OUTPUT_ACTION_MATERIALIZATION_SCHEMA = (
    "neural-forge-lfm2-output-action-pod-materialization-v1"
)
LFM2_OUTPUT_ACTION_RANK_DEFICIT_SCHEMA = (
    "neural-forge-lfm2-output-action-pod-rank-deficit-v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")
_FLOAT_DTYPES = frozenset(("float16", "bfloat16", "float32", "float64"))
_ROW_ROLES = frozenset(("fit", "heldout"))
_SOURCE_MODES = frozenset(("post_minus_pre", "sealed_branch_action"))

_ZERO_OPERATION_AUDIT = MappingProxyType(
    {
        "model_objects_accepted": 0,
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "candidate_model_forward_calls": 0,
        "model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "new_parent_trace_rows": 0,
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "runtime_parameter_aliases_created": 0,
        "runtime_parameter_sharing_used": False,
        "parent_sidecars_required": 0,
        "filesystem_reads": 0,
        "filesystem_writes": 0,
        "network_access": False,
    }
)


class Lfm2OutputActionSubspaceError(ValueError):
    """Cached evidence or compiled output-action geometry is invalid."""


class Lfm2OutputActionRankDeficitError(Lfm2OutputActionSubspaceError):
    """FIT actions cannot support the requested output-action rank."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = MappingProxyType(dict(report))
        observed = int(report.get("observed_rank", 0))
        requested = int(report.get("requested_rank", 0))
        super().__init__(
            "Output-action POD failed closed: FIT action rank "
            f"{observed} is below requested rank {requested}."
        )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError("Output-action POD requires PyTorch.") from exc
    return torch


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_deep_thaw(item) for item in value]
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise Lfm2OutputActionSubspaceError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not _IDENTIFIER.fullmatch(value)
        or len(value) > 512
    ):
        raise Lfm2OutputActionSubspaceError(
            f"{label} must be one stable nonempty identifier."
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2OutputActionSubspaceError(f"{label} must be one positive integer.")
    return value


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2OutputActionSubspaceError(
            f"{label} must be one finite nonnegative number."
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2OutputActionSubspaceError(
            f"{label} must be one finite nonnegative number."
        )
    return result


def _row_ids(value: Any, *, role: str) -> tuple[tuple[str, ...], Any]:
    torch = _torch()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise Lfm2OutputActionSubspaceError(
            f"{role}_row_ids must be one nonempty sequence."
        )
    identifiers: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item or len(item) > 1024:
            raise Lfm2OutputActionSubspaceError(
                f"{role}_row_ids[{index}] must be one nonempty string."
            )
        identifiers.append(item)
    if len(set(identifiers)) != len(identifiers):
        raise Lfm2OutputActionSubspaceError(
            f"{role}_row_ids contain duplicate row identities."
        )
    order = sorted(range(len(identifiers)), key=identifiers.__getitem__)
    canonical = tuple(identifiers[index] for index in order)
    return canonical, torch.tensor(order, dtype=torch.long)


def _source_rows(value: Any, *, label: str) -> tuple[Any, Any]:
    torch = _torch()
    if (
        type(value) is not torch.Tensor
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.ndim != 2
        or any(int(size) <= 0 for size in value.shape)
        or str(value.dtype).removeprefix("torch.") not in _FLOAT_DTYPES
        or bool(getattr(value, "is_quantized", False))
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2OutputActionSubspaceError(
            f"{label} must be one exact, detached, dense CPU floating rank-2 Tensor."
        )
    source = value.detach().contiguous().clone()
    if not bool(torch.isfinite(source).all().item()):
        raise Lfm2OutputActionSubspaceError(f"{label} contains NaN or infinity.")
    return source, source.to(dtype=torch.float64).contiguous()


def _case_weights(
    value: Any | None,
    *,
    row_count: int,
    label: str,
) -> tuple[Any, bool]:
    torch = _torch()
    if value is None:
        return torch.ones(row_count, dtype=torch.float64), False
    if type(value) is torch.Tensor:
        if (
            value.layout != torch.strided
            or value.device.type != "cpu"
            or value.ndim not in (1, 2)
            or (value.ndim == 2 and int(value.shape[1]) != 1)
            or value.numel() != row_count
            or str(value.dtype).removeprefix("torch.") not in _FLOAT_DTYPES
            or bool(getattr(value, "is_quantized", False))
            or value.requires_grad
            or value.grad_fn is not None
        ):
            raise Lfm2OutputActionSubspaceError(
                f"{label} must contain exactly one detached CPU floating "
                "weight per row."
            )
        result = value.detach().to(dtype=torch.float64).reshape(-1).contiguous()
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise Lfm2OutputActionSubspaceError(
                f"{label} must contain exactly one positive weight per row."
            )
        numbers: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise Lfm2OutputActionSubspaceError(
                    f"{label} must contain only finite positive numbers."
                )
            numbers.append(float(item))
        if len(numbers) != row_count:
            raise Lfm2OutputActionSubspaceError(
                f"{label} must contain exactly one positive weight per row."
            )
        result = torch.tensor(numbers, dtype=torch.float64)
    if not bool(torch.isfinite(result).all().item()) or not bool(
        (result > 0.0).all().item()
    ):
        raise Lfm2OutputActionSubspaceError(
            f"{label} must contain only finite strictly positive weights."
        )
    return result.clone(), True


def _tensor_record(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": [int(size) for size in value.shape],
        "sha256": lfm2_authenticated_tensor_sha256(value),
    }


@dataclass(frozen=True, slots=True)
class Lfm2SealedOutputActionRows:
    """Canonical cached action rows for one authenticated evidence split."""

    role: Literal["fit", "heldout"]
    row_ids: tuple[str, ...]
    action_rows: Any = field(repr=False, compare=False)
    case_weights: Any = field(repr=False, compare=False)
    source_mode: Literal["post_minus_pre", "sealed_branch_action"]
    source_tensor_records: tuple[tuple[str, str], ...]
    weights_supplied: bool
    receipt: Mapping[str, Any]

    @property
    def row_count(self) -> int:
        return int(self.action_rows.shape[0])

    @property
    def width(self) -> int:
        return int(self.action_rows.shape[1])

    @property
    def seal_sha256(self) -> str:
        return str(self.receipt["seal_sha256"])

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-compatible copy of the sealed receipt."""

        _authenticate_rows(self)
        return _deep_thaw(self.receipt)


def _rows_receipt_body(rows: Lfm2SealedOutputActionRows) -> dict[str, Any]:
    return {
        "schema": LFM2_OUTPUT_ACTION_ROWS_SCHEMA,
        "role": rows.role,
        "source_mode": rows.source_mode,
        "action_derivation": (
            "cpu-fp64-post-minus-pre"
            if rows.source_mode == "post_minus_pre"
            else "cpu-fp64-direct-sealed-action"
        ),
        "row_order": "lexicographic-row-id",
        "row_count": rows.row_count,
        "width": rows.width,
        "row_ids": list(rows.row_ids),
        "row_ids_sha256": canonical_sha256(list(rows.row_ids)),
        "source_tensor_sha256s": {
            name: digest for name, digest in rows.source_tensor_records
        },
        "action_rows": _tensor_record("action_rows", rows.action_rows),
        "case_weights": _tensor_record("case_weights", rows.case_weights),
        "case_weights_supplied": rows.weights_supplied,
        "case_weights_contract": "strictly-positive-per-row",
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }


def _authenticate_rows(
    value: Any,
    *,
    required_role: Literal["fit", "heldout"] | None = None,
) -> Lfm2SealedOutputActionRows:
    torch = _torch()
    if not isinstance(value, Lfm2SealedOutputActionRows):
        raise Lfm2OutputActionSubspaceError(
            "Output-action evidence must use sealed row objects."
        )
    if value.role not in _ROW_ROLES or (
        required_role is not None and value.role != required_role
    ):
        raise Lfm2OutputActionSubspaceError("Sealed row role is invalid.")
    if value.source_mode not in _SOURCE_MODES:
        raise Lfm2OutputActionSubspaceError("Sealed row source mode is invalid.")
    expected_source_names = (
        ("post_rows", "pre_rows")
        if value.source_mode == "post_minus_pre"
        else ("branch_action_rows",)
    )
    if tuple(name for name, _digest_value in value.source_tensor_records) != tuple(
        sorted(expected_source_names)
    ):
        raise Lfm2OutputActionSubspaceError(
            "Sealed row source tensor namespace is invalid."
        )
    for name, digest in value.source_tensor_records:
        _digest(digest, f"source_tensor_records[{name!r}]")
    if (
        type(value.action_rows) is not torch.Tensor
        or value.action_rows.dtype != torch.float64
        or value.action_rows.device.type != "cpu"
        or value.action_rows.layout != torch.strided
        or value.action_rows.ndim != 2
        or not value.action_rows.is_contiguous()
        or value.action_rows.requires_grad
        or value.action_rows.grad_fn is not None
        or type(value.case_weights) is not torch.Tensor
        or value.case_weights.dtype != torch.float64
        or value.case_weights.device.type != "cpu"
        or value.case_weights.layout != torch.strided
        or value.case_weights.ndim != 1
        or not value.case_weights.is_contiguous()
        or value.case_weights.requires_grad
        or value.case_weights.grad_fn is not None
        or value.action_rows.shape[0] != value.case_weights.shape[0]
        or value.action_rows.numel() == 0
        or len(value.row_ids) != int(value.action_rows.shape[0])
        or tuple(sorted(value.row_ids)) != value.row_ids
        or len(set(value.row_ids)) != len(value.row_ids)
        or not bool(torch.isfinite(value.action_rows).all().item())
        or not bool(torch.isfinite(value.case_weights).all().item())
        or not bool((value.case_weights > 0.0).all().item())
    ):
        raise Lfm2OutputActionSubspaceError(
            "Sealed output-action rows were mutated or malformed."
        )
    if not isinstance(value.weights_supplied, bool):
        raise Lfm2OutputActionSubspaceError(
            "Sealed row weight provenance is malformed."
        )
    body = _rows_receipt_body(value)
    expected = {**body, "seal_sha256": canonical_sha256(body)}
    if _deep_thaw(value.receipt) != expected:
        raise Lfm2OutputActionSubspaceError(
            "Sealed output-action row receipt authentication failed."
        )
    return value


def seal_lfm2_output_action_rows(
    *,
    role: Literal["fit", "heldout"],
    row_ids: Sequence[str],
    pre_rows: Any | None = None,
    post_rows: Any | None = None,
    branch_action_rows: Any | None = None,
    case_weights: Any | None = None,
) -> Lfm2SealedOutputActionRows:
    """Seal one cached split from pre/post rows or direct branch actions.

    Exactly one source form is legal.  Rows and weights are reordered by their
    stable row IDs before being cloned into private CPU FP64 storage.
    """

    torch = _torch()
    if role not in _ROW_ROLES:
        raise Lfm2OutputActionSubspaceError("role must be 'fit' or 'heldout'.")
    canonical_ids, order = _row_ids(row_ids, role=role)
    pair_requested = pre_rows is not None or post_rows is not None
    action_requested = branch_action_rows is not None
    if pair_requested == action_requested or (
        pair_requested and (pre_rows is None or post_rows is None)
    ):
        raise Lfm2OutputActionSubspaceError(
            "Supply exactly one complete pre/post pair or branch_action_rows."
        )

    with torch.no_grad():
        if action_requested:
            source_action, action_fp64 = _source_rows(
                branch_action_rows,
                label=f"{role}.branch_action_rows",
            )
            if int(source_action.shape[0]) != len(canonical_ids):
                raise Lfm2OutputActionSubspaceError(
                    "Row identities and branch actions have different counts."
                )
            source_action = source_action.index_select(0, order).contiguous()
            actions = action_fp64.index_select(0, order).contiguous()
            source_mode = "sealed_branch_action"
            source_records = (
                (
                    "branch_action_rows",
                    lfm2_authenticated_tensor_sha256(source_action),
                ),
            )
        else:
            source_pre, pre_fp64 = _source_rows(
                pre_rows,
                label=f"{role}.pre_rows",
            )
            source_post, post_fp64 = _source_rows(
                post_rows,
                label=f"{role}.post_rows",
            )
            if tuple(source_pre.shape) != tuple(source_post.shape) or int(
                source_pre.shape[0]
            ) != len(canonical_ids):
                raise Lfm2OutputActionSubspaceError(
                    "Pre/post residual rows and row identities must align."
                )
            source_pre = source_pre.index_select(0, order).contiguous()
            source_post = source_post.index_select(0, order).contiguous()
            actions = (
                post_fp64.index_select(0, order) - pre_fp64.index_select(0, order)
            ).contiguous()
            source_mode = "post_minus_pre"
            source_records = tuple(
                sorted(
                    (
                        (
                            "pre_rows",
                            lfm2_authenticated_tensor_sha256(source_pre),
                        ),
                        (
                            "post_rows",
                            lfm2_authenticated_tensor_sha256(source_post),
                        ),
                    )
                )
            )
        if not bool(torch.isfinite(actions).all().item()):
            raise Lfm2OutputActionSubspaceError(
                "Derived branch actions contain NaN or infinity."
            )
        weights, supplied = _case_weights(
            case_weights,
            row_count=len(canonical_ids),
            label=f"{role}.case_weights",
        )
        weights = weights.index_select(0, order).contiguous()

        provisional = Lfm2SealedOutputActionRows(
            role=role,
            row_ids=canonical_ids,
            action_rows=actions.clone(),
            case_weights=weights.clone(),
            source_mode=source_mode,
            source_tensor_records=source_records,
            weights_supplied=supplied,
            receipt=MappingProxyType({}),
        )
        body = _rows_receipt_body(provisional)
        sealed = Lfm2SealedOutputActionRows(
            role=provisional.role,
            row_ids=provisional.row_ids,
            action_rows=provisional.action_rows,
            case_weights=provisional.case_weights,
            source_mode=provisional.source_mode,
            source_tensor_records=provisional.source_tensor_records,
            weights_supplied=provisional.weights_supplied,
            receipt=_deep_freeze({**body, "seal_sha256": canonical_sha256(body)}),
        )
    return _authenticate_rows(sealed, required_role=role)


@dataclass(frozen=True, slots=True)
class Lfm2OutputActionEvidence:
    """Authenticated FIT evidence and optional diagnostic-only held-out rows."""

    parent_state_sha256: str
    cache_sha256: str
    component_id: str
    fit: Lfm2SealedOutputActionRows
    heldout: Lfm2SealedOutputActionRows | None
    receipt: Mapping[str, Any]

    @property
    def evidence_sha256(self) -> str:
        return str(self.receipt["evidence_sha256"])

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-compatible copy of the evidence receipt."""

        _authenticate_evidence(self)
        return _deep_thaw(self.receipt)


def _evidence_receipt_body(
    evidence: Lfm2OutputActionEvidence,
) -> dict[str, Any]:
    return {
        "schema": LFM2_OUTPUT_ACTION_EVIDENCE_SCHEMA,
        "parent_state_sha256": evidence.parent_state_sha256,
        "cache_sha256": evidence.cache_sha256,
        "component_id": evidence.component_id,
        "fit_rows_seal_sha256": evidence.fit.seal_sha256,
        "heldout_rows_seal_sha256": (
            None if evidence.heldout is None else evidence.heldout.seal_sha256
        ),
        "fit_heldout_overlap_count": 0,
        "heldout_policy": "diagnostic-only-no-basis-influence",
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }


def _authenticate_evidence(value: Any) -> Lfm2OutputActionEvidence:
    if not isinstance(value, Lfm2OutputActionEvidence):
        raise Lfm2OutputActionSubspaceError(
            "Compiler input must be sealed Lfm2OutputActionEvidence."
        )
    _digest(value.parent_state_sha256, "parent_state_sha256")
    _digest(value.cache_sha256, "cache_sha256")
    _identifier(value.component_id, "component_id")
    fit = _authenticate_rows(value.fit, required_role="fit")
    heldout = (
        None
        if value.heldout is None
        else _authenticate_rows(value.heldout, required_role="heldout")
    )
    if heldout is not None:
        overlap = set(fit.row_ids).intersection(heldout.row_ids)
        if overlap:
            raise Lfm2OutputActionSubspaceError(
                "FIT and held-out row identities overlap."
            )
        if fit.width != heldout.width:
            raise Lfm2OutputActionSubspaceError(
                "FIT and held-out branch actions have different widths."
            )
    body = _evidence_receipt_body(value)
    expected = {**body, "evidence_sha256": canonical_sha256(body)}
    if _deep_thaw(value.receipt) != expected:
        raise Lfm2OutputActionSubspaceError(
            "Output-action evidence receipt authentication failed."
        )
    return value


def seal_lfm2_output_action_evidence(
    *,
    parent_state_sha256: str,
    cache_sha256: str,
    component_id: str,
    fit: Lfm2SealedOutputActionRows,
    heldout: Lfm2SealedOutputActionRows | None = None,
) -> Lfm2OutputActionEvidence:
    """Bind sealed cached rows to one parent, cache, and LFM2 component."""

    _digest(parent_state_sha256, "parent_state_sha256")
    _digest(cache_sha256, "cache_sha256")
    _identifier(component_id, "component_id")
    _authenticate_rows(fit, required_role="fit")
    if heldout is not None:
        _authenticate_rows(heldout, required_role="heldout")
        overlap = set(fit.row_ids).intersection(heldout.row_ids)
        if overlap:
            raise Lfm2OutputActionSubspaceError(
                "FIT and held-out row identities overlap."
            )
        if fit.width != heldout.width:
            raise Lfm2OutputActionSubspaceError(
                "FIT and held-out branch actions have different widths."
            )
    provisional = Lfm2OutputActionEvidence(
        parent_state_sha256=parent_state_sha256,
        cache_sha256=cache_sha256,
        component_id=component_id,
        fit=fit,
        heldout=heldout,
        receipt=MappingProxyType({}),
    )
    body = _evidence_receipt_body(provisional)
    sealed = Lfm2OutputActionEvidence(
        parent_state_sha256=parent_state_sha256,
        cache_sha256=cache_sha256,
        component_id=component_id,
        fit=fit,
        heldout=heldout,
        receipt=_deep_freeze({**body, "evidence_sha256": canonical_sha256(body)}),
    )
    return _authenticate_evidence(sealed)


def _orthogonal_residual(value: Any, accepted: Sequence[Any]) -> Any:
    result = value.clone()
    for _pass in range(2):
        for basis in accepted:
            result = result - basis * (basis @ result)
    return result


def _sign_canonicalize(value: Any) -> Any:
    pivot = int(value.abs().argmax().item())
    if float(value[pivot].item()) < 0.0:
        value = -value
    return value.contiguous()


def _canonical_projector_basis(
    projector: Any,
    dimension: int,
    *,
    tolerance: float,
) -> tuple[Any, ...]:
    accepted: list[Any] = []
    for axis in range(int(projector.shape[0])):
        candidate = _orthogonal_residual(projector[:, axis], accepted)
        norm = float(candidate.norm().item())
        if norm <= tolerance:
            continue
        accepted.append(_sign_canonicalize(candidate / norm))
        if len(accepted) == dimension:
            break
    if len(accepted) != dimension:
        raise Lfm2OutputActionSubspaceError(
            "Could not canonicalize a degenerate output-action eigenspace."
        )
    return tuple(accepted)


def _canonical_top_basis(
    covariance: Any,
    *,
    eigenvalues: Any,
    eigenvectors: Any,
    target_rank: int,
    degeneracy_relative_tolerance: float,
    degeneracy_absolute_tolerance: float,
) -> tuple[Any, Any, tuple[dict[str, Any], ...], Any]:
    torch = _torch()
    scale = max(
        torch.finfo(torch.float64).tiny,
        float(eigenvalues.abs().max().item()),
    )
    group_tolerance = max(
        degeneracy_absolute_tolerance,
        degeneracy_relative_tolerance * scale,
        torch.finfo(torch.float64).eps * scale * int(covariance.shape[0]) * 64.0,
    )
    vector_tolerance = max(
        1e-14,
        torch.finfo(torch.float64).eps * int(covariance.shape[0]) * 32.0,
    )

    selected: list[Any] = []
    groups: list[dict[str, Any]] = []
    start = 0
    width = int(eigenvalues.numel())
    while start < width and len(selected) < target_rank:
        stop = start + 1
        while (
            stop < width
            and abs(float(eigenvalues[stop].item() - eigenvalues[start].item()))
            <= group_tolerance
        ):
            stop += 1
        group_vectors = eigenvectors[:, start:stop]
        group_projector = (group_vectors @ group_vectors.T).contiguous()
        if stop - start == 1:
            canonical = (_sign_canonicalize(group_vectors[:, 0].clone()),)
        else:
            canonical = _canonical_projector_basis(
                group_projector,
                stop - start,
                tolerance=vector_tolerance,
            )
        take = min(target_rank - len(selected), len(canonical))
        for candidate in canonical[:take]:
            candidate = _orthogonal_residual(candidate, selected)
            norm = float(candidate.norm().item())
            if norm <= vector_tolerance:
                raise Lfm2OutputActionSubspaceError(
                    "Canonical output-action modes lost orthogonality."
                )
            selected.append(_sign_canonicalize(candidate / norm))
        groups.append(
            {
                "start": start,
                "stop": stop,
                "multiplicity": stop - start,
                "selected": take,
                "eigenvalue_max": float(eigenvalues[start].item()),
                "eigenvalue_min": float(eigenvalues[stop - 1].item()),
                "projector_sha256": lfm2_authenticated_tensor_sha256(group_projector),
                "canonicalized_from_projector": stop - start > 1,
            }
        )
        start = stop
    if len(selected) != target_rank:
        raise Lfm2OutputActionSubspaceError(
            "Eigensolver did not yield the requested number of modes."
        )
    basis = torch.stack(selected, dim=1).contiguous()
    identity = torch.eye(target_rank, dtype=torch.float64)
    if not torch.allclose(
        basis.T @ basis,
        identity,
        atol=2e-12,
        rtol=2e-11,
    ):
        raise Lfm2OutputActionSubspaceError(
            "Canonical output-action basis is not orthonormal."
        )
    energies = torch.sum(basis * (covariance @ basis), dim=0).contiguous()
    if not bool(torch.isfinite(energies).all().item()):
        raise Lfm2OutputActionSubspaceError(
            "Canonical output-action energies are non-finite."
        )
    projector = (basis @ basis.T).contiguous()
    return basis, energies, tuple(groups), projector


def _normalized_weights(value: Any) -> Any:
    torch = _torch()
    maximum = float(value.max().item())
    scaled = value / maximum
    if not bool((scaled > 0.0).all().item()):
        raise Lfm2OutputActionSubspaceError(
            "Case-weight dynamic range underflowed deterministic FP64."
        )
    total = float(scaled.sum().item())
    if not math.isfinite(total) or total <= 0.0:
        raise Lfm2OutputActionSubspaceError(
            "Case weights cannot be normalized safely in FP64."
        )
    normalized = (scaled / total).contiguous()
    if not bool(torch.isfinite(normalized).all().item()):
        raise Lfm2OutputActionSubspaceError("Normalized case weights are non-finite.")
    return normalized


def _projection_diagnostics(
    actions: Any, weights: Any, projector: Any
) -> dict[str, Any]:
    projected = (actions @ projector).contiguous()
    residual = actions - projected
    normalized_weights = _normalized_weights(weights)
    target_energy = float(
        (normalized_weights * actions.square().sum(dim=1)).sum().item()
    )
    residual_energy = float(
        (normalized_weights * residual.square().sum(dim=1)).sum().item()
    )
    projected_energy = float(
        (normalized_weights * projected.square().sum(dim=1)).sum().item()
    )
    tiny = _torch().finfo(_torch().float64).tiny
    if target_energy <= tiny:
        relative_rmse = 0.0 if residual_energy <= tiny else math.inf
        retained = 1.0 if residual_energy <= tiny else 0.0
    else:
        relative_rmse = math.sqrt(max(residual_energy, 0.0) / target_energy)
        retained = max(0.0, min(1.0, projected_energy / target_energy))
    numerator = float(
        (normalized_weights * (actions * projected).sum(dim=1)).sum().item()
    )
    denominator = math.sqrt(max(target_energy, 0.0) * max(projected_energy, 0.0))
    cosine = (
        1.0
        if denominator <= tiny and residual_energy <= tiny
        else (0.0 if denominator <= tiny else numerator / denominator)
    )
    values = (
        target_energy,
        residual_energy,
        projected_energy,
        relative_rmse,
        retained,
        cosine,
    )
    if not all(math.isfinite(item) for item in values):
        raise Lfm2OutputActionSubspaceError("Projection diagnostics are non-finite.")
    return {
        "row_count": int(actions.shape[0]),
        "weighted_target_energy": target_energy,
        "weighted_projected_energy": projected_energy,
        "weighted_residual_energy": residual_energy,
        "relative_rmse": relative_rmse,
        "retained_energy_fraction": retained,
        "flattened_weighted_cosine": max(-1.0, min(1.0, cosine)),
    }


@dataclass(frozen=True, slots=True)
class Lfm2CompiledOutputActionSubspace:
    """Canonical FP64 output-action basis and its authenticated projector."""

    output_basis: Any = field(repr=False, compare=False)
    projector: Any = field(repr=False, compare=False)
    component_energies: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    @property
    def source_width(self) -> int:
        return int(self.output_basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.output_basis.shape[1])

    @property
    def basis_receipt_sha256(self) -> str:
        return str(self.report["basis_receipt_sha256"])

    @property
    def result_sha256(self) -> str:
        return str(self.report["result_sha256"])

    def encode(self, action_rows: Any) -> Any:
        """Encode residual action rows into the compact rank-h coordinates."""

        return encode_lfm2_output_actions(self, action_rows)

    def decode(self, coefficients: Any) -> Any:
        """Decode rank-h coordinates into the parent residual width."""

        return decode_lfm2_output_actions(self, coefficients)

    def project(self, action_rows: Any) -> Any:
        """Project residual action rows through ``P = Q Q.T``."""

        return project_lfm2_output_actions(self, action_rows)

    def materialize(self, *, dtype: Any | None = None) -> Any:
        """Materialize independent dense encoder/decoder factors."""

        return materialize_lfm2_output_action_subspace(self, dtype=dtype)

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-compatible copy of the result receipt."""

        _authenticate_compiled(self)
        return _deep_thaw(self.report)


def _authenticate_compiled(
    value: Any,
) -> Lfm2CompiledOutputActionSubspace:
    torch = _torch()
    if not isinstance(value, Lfm2CompiledOutputActionSubspace):
        raise Lfm2OutputActionSubspaceError(
            "Expected a compiled output-action subspace."
        )
    basis = value.output_basis
    projector = value.projector
    energies = value.component_energies
    if (
        type(basis) is not torch.Tensor
        or basis.dtype != torch.float64
        or basis.device.type != "cpu"
        or basis.ndim != 2
        or not basis.is_contiguous()
        or basis.requires_grad
        or basis.grad_fn is not None
        or type(projector) is not torch.Tensor
        or projector.dtype != torch.float64
        or projector.device.type != "cpu"
        or projector.ndim != 2
        or not projector.is_contiguous()
        or projector.requires_grad
        or projector.grad_fn is not None
        or tuple(projector.shape) != (int(basis.shape[0]), int(basis.shape[0]))
        or type(energies) is not torch.Tensor
        or energies.dtype != torch.float64
        or energies.device.type != "cpu"
        or energies.ndim != 1
        or int(energies.shape[0]) != int(basis.shape[1])
        or not energies.is_contiguous()
        or energies.requires_grad
        or energies.grad_fn is not None
        or not bool(torch.isfinite(basis).all().item())
        or not bool(torch.isfinite(projector).all().item())
        or not bool(torch.isfinite(energies).all().item())
    ):
        raise Lfm2OutputActionSubspaceError(
            "Compiled output-action tensors were mutated or malformed."
        )
    identity = torch.eye(int(basis.shape[1]), dtype=torch.float64)
    expected_projector = basis @ basis.T
    if not torch.allclose(basis.T @ basis, identity, atol=2e-12, rtol=2e-11):
        raise Lfm2OutputActionSubspaceError(
            "Compiled output-action basis lost orthonormality."
        )
    if not torch.equal(projector, expected_projector.contiguous()):
        raise Lfm2OutputActionSubspaceError(
            "Compiled output-action projector no longer equals Q Q.T."
        )
    report = _deep_thaw(value.report)
    if report.get("schema") != LFM2_OUTPUT_ACTION_SUBSPACE_SCHEMA:
        raise Lfm2OutputActionSubspaceError(
            "Compiled output-action result schema is invalid."
        )
    if report.get("output_basis") != _tensor_record("output_basis", basis):
        raise Lfm2OutputActionSubspaceError(
            "Compiled output-action basis seal is invalid."
        )
    if report.get("projector") != _tensor_record("projector", projector):
        raise Lfm2OutputActionSubspaceError(
            "Compiled output-action projector seal is invalid."
        )
    if report.get("component_energies") != _tensor_record(
        "component_energies", energies
    ):
        raise Lfm2OutputActionSubspaceError(
            "Compiled output-action energy seal is invalid."
        )
    expected_result = report.pop("result_sha256", None)
    if expected_result != canonical_sha256(report):
        raise Lfm2OutputActionSubspaceError(
            "Compiled output-action result receipt authentication failed."
        )
    return value


def compile_lfm2_output_action_subspace(
    evidence: Lfm2OutputActionEvidence,
    *,
    target_rank: int,
    rank_relative_tolerance: float = 1e-12,
    degeneracy_relative_tolerance: float = 1e-12,
    degeneracy_absolute_tolerance: float = 0.0,
) -> Lfm2CompiledOutputActionSubspace:
    """Compile a canonical weighted FIT-only POD output-action subspace."""

    torch = _torch()
    evidence = _authenticate_evidence(evidence)
    target_rank = _positive_int(target_rank, "target_rank")
    rank_relative_tolerance = _nonnegative_float(
        rank_relative_tolerance,
        "rank_relative_tolerance",
    )
    degeneracy_relative_tolerance = _nonnegative_float(
        degeneracy_relative_tolerance,
        "degeneracy_relative_tolerance",
    )
    degeneracy_absolute_tolerance = _nonnegative_float(
        degeneracy_absolute_tolerance,
        "degeneracy_absolute_tolerance",
    )
    width = evidence.fit.width
    if target_rank > width:
        body = {
            "schema": LFM2_OUTPUT_ACTION_RANK_DEFICIT_SCHEMA,
            "status": "requested-rank-exceeds-output-width",
            "evidence_sha256": evidence.evidence_sha256,
            "requested_rank": target_rank,
            "observed_rank": min(evidence.fit.row_count, width),
            "source_width": width,
        }
        raise Lfm2OutputActionRankDeficitError(
            {**body, "report_sha256": canonical_sha256(body)}
        )

    with torch.no_grad():
        fit_weights = _normalized_weights(evidence.fit.case_weights)
        weighted_actions = (
            evidence.fit.action_rows * fit_weights.sqrt().unsqueeze(1)
        ).contiguous()
        covariance = (weighted_actions.T @ weighted_actions).contiguous()
        covariance = ((covariance + covariance.T) * 0.5).contiguous()
        if not bool(torch.isfinite(covariance).all().item()):
            raise Lfm2OutputActionSubspaceError(
                "Weighted FIT action covariance is non-finite."
            )
        try:
            all_eigenvalues, all_eigenvectors = torch.linalg.eigh(covariance)
        except RuntimeError as exc:
            raise Lfm2OutputActionSubspaceError(
                "CPU FP64 FIT rank audit failed."
            ) from exc
        all_eigenvalues = all_eigenvalues.flip(0).contiguous()
        all_eigenvectors = all_eigenvectors.flip(1).contiguous()
        if not bool(torch.isfinite(all_eigenvalues).all().item()) or not bool(
            torch.isfinite(all_eigenvectors).all().item()
        ):
            raise Lfm2OutputActionSubspaceError(
                "FIT rank audit produced non-finite eigenpairs."
            )
        maximum = max(0.0, float(all_eigenvalues[0].item()))
        numerical_floor = (
            torch.finfo(torch.float64).eps * max(evidence.fit.row_count, width) * 64.0
        )
        rank_threshold = maximum * max(
            rank_relative_tolerance,
            numerical_floor,
        )
        negative_tolerance = max(
            1e-15,
            maximum * numerical_floor,
        )
        if float(all_eigenvalues[-1].item()) < -negative_tolerance:
            raise Lfm2OutputActionSubspaceError(
                "Weighted FIT covariance is materially non-positive-semidefinite."
            )
        observed_rank = int((all_eigenvalues > rank_threshold).sum().item())
        if observed_rank < target_rank:
            body = {
                "schema": LFM2_OUTPUT_ACTION_RANK_DEFICIT_SCHEMA,
                "status": "fit-action-rank-deficit",
                "evidence_sha256": evidence.evidence_sha256,
                "fit_rows_seal_sha256": evidence.fit.seal_sha256,
                "requested_rank": target_rank,
                "observed_rank": observed_rank,
                "source_width": width,
                "fit_row_count": evidence.fit.row_count,
                "largest_eigenvalue": maximum,
                "rank_threshold": rank_threshold,
                "rank_relative_tolerance": rank_relative_tolerance,
                "covariance_sha256": lfm2_authenticated_tensor_sha256(covariance),
            }
            raise Lfm2OutputActionRankDeficitError(
                {**body, "report_sha256": canonical_sha256(body)}
            )
        basis, energies, groups, projector = _canonical_top_basis(
            covariance,
            eigenvalues=all_eigenvalues,
            eigenvectors=all_eigenvectors,
            target_rank=target_rank,
            degeneracy_relative_tolerance=degeneracy_relative_tolerance,
            degeneracy_absolute_tolerance=degeneracy_absolute_tolerance,
        )
        fit_diagnostics = _projection_diagnostics(
            evidence.fit.action_rows,
            evidence.fit.case_weights,
            projector,
        )
        heldout_diagnostics = (
            None
            if evidence.heldout is None
            else _projection_diagnostics(
                evidence.heldout.action_rows,
                evidence.heldout.case_weights,
                projector,
            )
        )

        derivation = {
            "schema": LFM2_OUTPUT_ACTION_SUBSPACE_SCHEMA,
            "method": "weighted-fit-action-covariance-pod",
            "fit_rows_seal_sha256": evidence.fit.seal_sha256,
            "target_rank": target_rank,
            "rank_relative_tolerance": rank_relative_tolerance,
            "degeneracy_relative_tolerance": degeneracy_relative_tolerance,
            "degeneracy_absolute_tolerance": degeneracy_absolute_tolerance,
            "compute_contract": "deterministic-cpu-fp64-no-autograd",
            "heldout_influences_basis": False,
        }
        basis_derivation_sha256 = canonical_sha256(derivation)
        basis_receipt = {
            "basis_derivation_sha256": basis_derivation_sha256,
            "output_basis": _tensor_record("output_basis", basis),
            "projector": _tensor_record("projector", projector),
            "component_energies": _tensor_record("component_energies", energies),
        }
        basis_receipt_sha256 = canonical_sha256(basis_receipt)
        report = {
            "schema": LFM2_OUTPUT_ACTION_SUBSPACE_SCHEMA,
            "method": "weighted-fit-action-covariance-pod",
            "parent_state_sha256": evidence.parent_state_sha256,
            "cache_sha256": evidence.cache_sha256,
            "component_id": evidence.component_id,
            "evidence_sha256": evidence.evidence_sha256,
            "fit_rows_seal_sha256": evidence.fit.seal_sha256,
            "heldout_rows_seal_sha256": (
                None if evidence.heldout is None else evidence.heldout.seal_sha256
            ),
            "source_width": width,
            "target_rank": target_rank,
            "fit_row_count": evidence.fit.row_count,
            "heldout_row_count": (
                0 if evidence.heldout is None else evidence.heldout.row_count
            ),
            "fit_case_weights_supplied": evidence.fit.weights_supplied,
            "heldout_case_weights_supplied": (
                False if evidence.heldout is None else evidence.heldout.weights_supplied
            ),
            "compute_device": "cpu",
            "compute_dtype": "float64",
            "observed_fit_action_rank": observed_rank,
            "rank_threshold": rank_threshold,
            "weighted_fit_covariance": _tensor_record(
                "weighted_fit_covariance", covariance
            ),
            "all_covariance_eigenvalues": _tensor_record(
                "all_covariance_eigenvalues", all_eigenvalues
            ),
            "canonical_eigenspace_groups": list(groups),
            "output_basis": _tensor_record("output_basis", basis),
            "projector": _tensor_record("projector", projector),
            "component_energies": _tensor_record("component_energies", energies),
            "basis_derivation_sha256": basis_derivation_sha256,
            "basis_receipt_sha256": basis_receipt_sha256,
            "fit_diagnostics": fit_diagnostics,
            "heldout_diagnostics": heldout_diagnostics,
            "heldout_policy": "diagnostic-only-no-basis-influence",
            "heldout_influences_basis": False,
            "training_or_optimization_used": False,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        result = Lfm2CompiledOutputActionSubspace(
            output_basis=basis.clone(),
            projector=projector.clone(),
            component_energies=energies.clone(),
            report=_deep_freeze({**report, "result_sha256": canonical_sha256(report)}),
        )
    return _authenticate_compiled(result)


def _operator_rows(value: Any, *, width: int, label: str) -> Any:
    _source, fp64 = _source_rows(value, label=label)
    if int(fp64.shape[1]) != width:
        raise Lfm2OutputActionSubspaceError(
            f"{label} must have trailing width {width}."
        )
    return fp64


def encode_lfm2_output_actions(
    compiled: Lfm2CompiledOutputActionSubspace,
    action_rows: Any,
) -> Any:
    """Apply ``Y Q`` without constructing a model or autograd graph."""

    torch = _torch()
    compiled = _authenticate_compiled(compiled)
    rows = _operator_rows(
        action_rows,
        width=compiled.source_width,
        label="action_rows",
    )
    with torch.no_grad():
        result = (rows @ compiled.output_basis).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2OutputActionSubspaceError("Encoded actions are non-finite.")
    return result


def decode_lfm2_output_actions(
    compiled: Lfm2CompiledOutputActionSubspace,
    coefficients: Any,
) -> Any:
    """Apply ``C Q.T`` without constructing a model or autograd graph."""

    torch = _torch()
    compiled = _authenticate_compiled(compiled)
    rows = _operator_rows(
        coefficients,
        width=compiled.rank,
        label="coefficients",
    )
    with torch.no_grad():
        result = (rows @ compiled.output_basis.T).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2OutputActionSubspaceError("Decoded actions are non-finite.")
    return result


def project_lfm2_output_actions(
    compiled: Lfm2CompiledOutputActionSubspace,
    action_rows: Any,
) -> Any:
    """Apply the authenticated projector and verify factor/direct agreement."""

    torch = _torch()
    compiled = _authenticate_compiled(compiled)
    rows = _operator_rows(
        action_rows,
        width=compiled.source_width,
        label="action_rows",
    )
    with torch.no_grad():
        factored = (
            (rows @ compiled.output_basis) @ compiled.output_basis.T
        ).contiguous()
        direct = (rows @ compiled.projector).contiguous()
    if not torch.allclose(factored, direct, atol=2e-12, rtol=2e-11):
        raise Lfm2OutputActionSubspaceError(
            "Factored and direct output-action projection disagree."
        )
    return factored


@dataclass(frozen=True, slots=True)
class Lfm2MaterializedOutputActionSubspace:
    """Independent ordinary dense factors for a physical action bottleneck."""

    encoder_weight: Any = field(repr=False, compare=False)
    decoder_weight: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    @property
    def source_width(self) -> int:
        return int(self.encoder_weight.shape[1])

    @property
    def rank(self) -> int:
        return int(self.encoder_weight.shape[0])

    @property
    def materialization_sha256(self) -> str:
        return str(self.report["materialization_sha256"])

    def state_dict(self) -> Mapping[str, Any]:
        """Return independent copies of the dense encoder and decoder."""

        _authenticate_materialized(self)
        return MappingProxyType(
            {
                "decoder.weight": self.decoder_weight.detach()
                .to(device="cpu")
                .contiguous()
                .clone(),
                "encoder.weight": self.encoder_weight.detach()
                .to(device="cpu")
                .contiguous()
                .clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable JSON-compatible materialization receipt."""

        _authenticate_materialized(self)
        return _deep_thaw(self.report)


def _authenticate_materialized(
    value: Any,
) -> Lfm2MaterializedOutputActionSubspace:
    torch = _torch()
    if not isinstance(value, Lfm2MaterializedOutputActionSubspace):
        raise Lfm2OutputActionSubspaceError(
            "Expected one materialized output-action subspace."
        )
    encoder = value.encoder_weight
    decoder = value.decoder_weight
    if (
        type(encoder) is not torch.Tensor
        or type(decoder) is not torch.Tensor
        or encoder.dtype not in (torch.float32, torch.float64)
        or decoder.dtype != encoder.dtype
        or encoder.device.type != "cpu"
        or decoder.device.type != "cpu"
        or encoder.layout != torch.strided
        or decoder.layout != torch.strided
        or encoder.ndim != 2
        or decoder.ndim != 2
        or tuple(decoder.shape) != tuple(reversed(encoder.shape))
        or not encoder.is_contiguous()
        or not decoder.is_contiguous()
        or encoder.requires_grad
        or decoder.requires_grad
        or encoder.grad_fn is not None
        or decoder.grad_fn is not None
        or not bool(torch.isfinite(encoder).all().item())
        or not bool(torch.isfinite(decoder).all().item())
        or encoder.untyped_storage().data_ptr() == decoder.untyped_storage().data_ptr()
        or not torch.equal(decoder, encoder.T.contiguous())
    ):
        raise Lfm2OutputActionSubspaceError(
            "Materialized output-action factors were mutated or malformed."
        )
    report = _deep_thaw(value.report)
    if report.get("schema") != LFM2_OUTPUT_ACTION_MATERIALIZATION_SCHEMA:
        raise Lfm2OutputActionSubspaceError(
            "Output-action materialization schema is invalid."
        )
    if report.get("encoder_weight") != _tensor_record("encoder.weight", encoder):
        raise Lfm2OutputActionSubspaceError("Output-action encoder seal is invalid.")
    if report.get("decoder_weight") != _tensor_record("decoder.weight", decoder):
        raise Lfm2OutputActionSubspaceError("Output-action decoder seal is invalid.")
    if report.get("materialized_parameter_count") != int(
        encoder.numel() + decoder.numel()
    ):
        raise Lfm2OutputActionSubspaceError(
            "Output-action materialized parameter accounting is invalid."
        )
    expected = report.pop("materialization_sha256", None)
    if expected != canonical_sha256(report):
        raise Lfm2OutputActionSubspaceError(
            "Output-action materialization receipt authentication failed."
        )
    return value


def materialize_lfm2_output_action_subspace(
    compiled: Lfm2CompiledOutputActionSubspace,
    *,
    dtype: Any | None = None,
) -> Lfm2MaterializedOutputActionSubspace:
    """Materialize independent dense ``Q.T`` encoder and ``Q`` decoder weights."""

    torch = _torch()
    compiled = _authenticate_compiled(compiled)
    resolved_dtype = torch.float32 if dtype is None else dtype
    if resolved_dtype not in (torch.float32, torch.float64):
        raise Lfm2OutputActionSubspaceError(
            "Materialization dtype must be ordinary dense FP32 or FP64."
        )
    with torch.no_grad():
        encoder = compiled.output_basis.T.to(dtype=resolved_dtype).contiguous().clone()
        decoder = compiled.output_basis.to(dtype=resolved_dtype).contiguous().clone()
    if encoder.untyped_storage().data_ptr() == decoder.untyped_storage().data_ptr():
        raise Lfm2OutputActionSubspaceError(
            "Materialized encoder and decoder unexpectedly share storage."
        )
    parameter_count = int(encoder.numel() + decoder.numel())
    dense_projector_count = compiled.source_width * compiled.source_width
    body = {
        "schema": LFM2_OUTPUT_ACTION_MATERIALIZATION_SCHEMA,
        "method": "independent-dense-orthogonal-bottleneck-factors",
        "source_result_sha256": compiled.result_sha256,
        "source_basis_receipt_sha256": compiled.basis_receipt_sha256,
        "source_width": compiled.source_width,
        "target_rank": compiled.rank,
        "dtype": str(resolved_dtype).removeprefix("torch."),
        "encoder_weight": _tensor_record("encoder.weight", encoder),
        "decoder_weight": _tensor_record("decoder.weight", decoder),
        "materialized_parameter_count": parameter_count,
        "dense_projector_parameter_count": dense_projector_count,
        "dense_projector_retained_as_parameter": False,
        "parameter_aliases": [],
        "runtime_parameter_sharing_used": False,
        "training_or_optimization_used": False,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    result = Lfm2MaterializedOutputActionSubspace(
        encoder_weight=encoder,
        decoder_weight=decoder,
        report=_deep_freeze({**body, "materialization_sha256": canonical_sha256(body)}),
    )
    # Authenticate the state before returning it without invoking any model.
    if result.encoder_weight.shape != (compiled.rank, compiled.source_width):
        raise Lfm2OutputActionSubspaceError("Encoder materialization shape drifted.")
    if result.decoder_weight.shape != (compiled.source_width, compiled.rank):
        raise Lfm2OutputActionSubspaceError("Decoder materialization shape drifted.")
    return _authenticate_materialized(result)


__all__ = [
    "LFM2_OUTPUT_ACTION_EVIDENCE_SCHEMA",
    "LFM2_OUTPUT_ACTION_MATERIALIZATION_SCHEMA",
    "LFM2_OUTPUT_ACTION_RANK_DEFICIT_SCHEMA",
    "LFM2_OUTPUT_ACTION_ROWS_SCHEMA",
    "LFM2_OUTPUT_ACTION_SUBSPACE_SCHEMA",
    "Lfm2CompiledOutputActionSubspace",
    "Lfm2MaterializedOutputActionSubspace",
    "Lfm2OutputActionEvidence",
    "Lfm2OutputActionRankDeficitError",
    "Lfm2OutputActionSubspaceError",
    "Lfm2SealedOutputActionRows",
    "compile_lfm2_output_action_subspace",
    "decode_lfm2_output_actions",
    "encode_lfm2_output_actions",
    "materialize_lfm2_output_action_subspace",
    "project_lfm2_output_actions",
    "seal_lfm2_output_action_evidence",
    "seal_lfm2_output_action_rows",
]
