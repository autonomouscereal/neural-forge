"""Standalone nonlinear Q-DEIM compiler for one dense LFM2 SwiGLU MLP.

This module is deliberately not wired into the semantic body-recipe registry.
It compiles two reviewable dense candidates from authenticated FIT evidence:

* a structural Q-DEIM prior whose W2 contains the dense reconstruction of
  every source post-SwiGLU feature; and
* an optional, separately identified FIT-only analytic W2 correction.

Numeric diagnostics describe the construction.  They never select, accept,
publish, or reject a semantic candidate.  Insufficient empirical feature rank
is a structural exception because a k-mode Q-DEIM basis does not exist; no
random or silent basis completion is permitted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)
from neural_forge.hf_lfm2_model_fold import Lfm2ModelFoldSpan


LFM2_MLP_QDEIM_SCHEMA = "neural-forge-lfm2-mlp-nonlinear-qdeim-v1"
LFM2_MLP_QDEIM_RANK_DEFICIT_SCHEMA = (
    "neural-forge-lfm2-mlp-nonlinear-qdeim-rank-deficit-v1"
)
LFM2_MLP_QDEIM_FUNCTIONAL_SNAPSHOT_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-mlp-qdeim-functional-snapshot-evidence-v1"
)
LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE = "qdeim_structural_prior"
LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE = "qdeim_fit_residual_corrected"
LFM2_MLP_SPAN_QDEIM_SCHEMA = "neural-forge-lfm2-mlp-span-coupled-qdeim-v1"
LFM2_MLP_SPAN_QDEIM_FUNCTIONAL_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-mlp-span-coupled-complete-bundle-evidence-v2"
)

_ALLOWED_FUNCTIONAL_SNAPSHOT_KINDS = frozenset(
    {
        "analytic_child_lifted_parent_swiglu_tangent",
        "authenticated_structural",
    }
)


class Lfm2MlpQdeimError(ValueError):
    """Authenticated nonlinear Q-DEIM compilation failed closed."""


class Lfm2MlpQdeimInsufficientRankError(Lfm2MlpQdeimError):
    """Authenticated functional snapshots cannot support the Q-DEIM width."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = MappingProxyType(dict(report))
        super().__init__(
            "Authenticated functional snapshot rank "
            f"{report['observed_snapshot_rank']} is below required Q-DEIM rank "
            f"{report['minimum_snapshot_rank']}; supply additional authenticated "
            "child-lifted tangent or structural snapshot blocks."
        )


@dataclass(frozen=True, slots=True)
class Lfm2MlpQdeimFitCase:
    """One aligned FIT case used by the functional basis and W2 correction.

    ``target_normalized_input`` is the child-side normalized MLP input.
    ``parent_post_swiglu`` is the cached source nonlinear feature matrix.
    The two residual tensors define the desired target-space MLP delta for the
    optional analytic correction.
    """

    target_normalized_input: Any = field(repr=False, compare=False)
    parent_post_swiglu: Any = field(repr=False, compare=False)
    parent_post_mlp_residual_target_space: Any = field(
        repr=False,
        compare=False,
    )
    target_pre_mlp_residual: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Lfm2MlpQdeimFunctionalSnapshotBlock:
    """Externally generated structural rows for functional POD completion.

    Rows are already lifted into the parent post-SwiGLU neuron coordinates,
    so ``functional_rows`` has shape ``[row_count, source_intermediate]``.
    ``block_weight`` and ``row_sample_weights`` are energy weights: a desired
    row-amplitude multiplier ``s`` must be supplied as weight ``s**2`` because
    POD multiplies rows by the square root of their effective weight.
    The compiler authenticates but does not generate these rows.  FIT rows and
    structural rows remain separately receipted, and only FIT rows may enter
    the optional endpoint W2 correction.
    """

    block_id: str
    snapshot_kind: str
    functional_rows: Any = field(repr=False, compare=False)
    generator_receipt_sha256: str
    block_weight: float = 1.0
    row_sample_weights: Sequence[float] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledMlpQdeim:
    """Authenticated dense structural and FIT-corrected Q-DEIM candidates."""

    selected_source_neurons: tuple[int, ...]
    gate_weight: Any = field(repr=False, compare=False)
    up_weight: Any = field(repr=False, compare=False)
    down_weight_structural_prior: Any = field(repr=False, compare=False)
    down_weight_fit_corrected: Any | None = field(repr=False, compare=False)
    functional_pod_basis: Any = field(repr=False, compare=False)
    interpolation_reconstruction: Any = field(repr=False, compare=False)
    normalized_input_decoder: Any = field(repr=False, compare=False)
    residual_output_chart: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        values = [LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE]
        if self.down_weight_fit_corrected is not None:
            values.append(LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE)
        return tuple(values)

    def state_dict(self, candidate_id: str) -> Mapping[str, Any]:
        if candidate_id == LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE:
            down = self.down_weight_structural_prior
        elif (
            candidate_id == LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE
            and self.down_weight_fit_corrected is not None
        ):
            down = self.down_weight_fit_corrected
        else:
            raise KeyError(f"Unknown materialized Q-DEIM candidate: {candidate_id!r}.")
        return MappingProxyType(
            {
                "w1.weight": self.gate_weight.detach().cpu().contiguous().clone(),
                "w3.weight": self.up_weight.detach().cpu().contiguous().clone(),
                "w2.weight": down.detach().cpu().contiguous().clone(),
            }
        )

    def candidate_state_dicts(self) -> Mapping[str, Mapping[str, Any]]:
        return MappingProxyType(
            {
                candidate_id: self.state_dict(candidate_id)
                for candidate_id in self.candidate_ids
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


@dataclass(frozen=True, slots=True)
class Lfm2MlpQdeimSpanSource:
    """One tensor/hash-bound source-layer contribution to a depth span.

    The semantic layer-stack adapter is expected to assemble one value for
    every index in :class:`Lfm2ModelFoldSpan`.  Functional rows are aligned
    across the span and remain in each layer's native post-SwiGLU coordinate
    system.  The decoder pulls a selected W1/W3 producer into the child input
    chart; the output chart pushes that layer's complete W2 writer into the
    child residual chart.  ``cache_receipt_sha256`` binds a receipt digest into
    this compiler's evidence hash; the upstream adapter must authenticate the
    receipt itself before constructing this value.
    """

    source_layer_index: int
    source_gate_weight: Any = field(repr=False, compare=False)
    source_up_weight: Any = field(repr=False, compare=False)
    source_down_weight: Any = field(repr=False, compare=False)
    expected_source_state_sha256: str
    cached_parent_post_swiglu_rows: Any = field(repr=False, compare=False)
    expected_cached_parent_post_swiglu_sha256: str
    cache_receipt_sha256: str
    normalized_input_decoder: Any = field(repr=False, compare=False)
    expected_normalized_input_decoder_sha256: str
    residual_output_chart: Any = field(repr=False, compare=False)
    expected_residual_output_chart_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.source_layer_index, bool):
            raise Lfm2MlpQdeimError(
                "source_layer_index must be a non-negative integer."
            )
        try:
            parsed_index = int(self.source_layer_index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Lfm2MlpQdeimError(
                "source_layer_index must be a non-negative integer."
            ) from exc
        if parsed_index < 0 or parsed_index != self.source_layer_index:
            raise Lfm2MlpQdeimError(
                "source_layer_index must be a non-negative integer."
            )
        object.__setattr__(self, "source_layer_index", parsed_index)
        for name in (
            "expected_source_state_sha256",
            "expected_cached_parent_post_swiglu_sha256",
            "cache_receipt_sha256",
            "expected_normalized_input_decoder_sha256",
            "expected_residual_output_chart_sha256",
        ):
            _digest(getattr(self, name), label=name)


@dataclass(frozen=True, slots=True)
class Lfm2CompiledSpanMlpQdeim:
    """One dense child SwiGLU compiled from an ordered parent-layer span."""

    span: Lfm2ModelFoldSpan
    selected_source_neurons: tuple[tuple[int, int], ...]
    gate_weight: Any = field(repr=False, compare=False)
    up_weight: Any = field(repr=False, compare=False)
    down_weight: Any = field(repr=False, compare=False)
    functional_pod_basis: Any = field(repr=False, compare=False)
    interpolation_reconstruction: Any = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "w1.weight": self.gate_weight.detach().cpu().contiguous().clone(),
                "w3.weight": self.up_weight.detach().cpu().contiguous().clone(),
                "w2.weight": self.down_weight.detach().cpu().contiguous().clone(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.report)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("PyTorch is required for LFM2 Q-DEIM compilation.") from exc
    return torch


def _resolve_qdeim_compute_device(torch: Any, compute_device: Any) -> Any:
    """Resolve one explicit fold-only algebra device without silent fallback."""

    if compute_device == "cpu":
        return torch.device("cpu")
    if not isinstance(compute_device, str):
        raise Lfm2MlpQdeimError(
            "compute_device must be 'cpu' or an explicitly indexed CUDA device."
        )
    prefix, separator, index_text = compute_device.partition(":")
    if (
        prefix != "cuda"
        or separator != ":"
        or not index_text.isascii()
        or not index_text.isdecimal()
        or index_text != str(int(index_text))
    ):
        raise Lfm2MlpQdeimError(
            "CUDA compute_device must be explicitly indexed as 'cuda:N'."
        )
    index = int(index_text)
    if not torch.cuda.is_available() or index >= int(torch.cuda.device_count()):
        raise Lfm2MlpQdeimError(f"CUDA compute_device index {index} is unavailable.")
    return torch.device("cuda", index)


def _qdeim_compute_receipt(torch: Any, requested: str, resolved: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": "neural-forge-lfm2-mlp-qdeim-compute-v1",
        "requested_device": requested,
        "resolved_device": str(resolved),
        "backend": resolved.type,
        "compute_dtype": "float64",
        "authenticated_source_archive_device": "cpu",
        "active_layer_staged_to_compute_device": resolved.type == "cuda",
        "materialized_candidate_device": "cpu",
        "system_memory_is_immutable_archive": True,
        "gpu_is_ephemeral_dense_algebra_workspace": resolved.type == "cuda",
        "silent_device_fallback_performed": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    if resolved.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved)
        body.update(
            {
                "device_index": int(resolved.index),
                "device_name": str(properties.name),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
            }
        )
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]
    return value


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2MlpQdeimError(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _finite_float(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise Lfm2MlpQdeimError(f"{label} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2MlpQdeimError(f"{label} must be a finite number.") from exc
    if not math.isfinite(parsed) or (nonnegative and parsed < 0.0):
        qualifier = "nonnegative finite" if nonnegative else "finite"
        raise Lfm2MlpQdeimError(f"{label} must be {qualifier}.")
    return parsed


def _materialized_float_tensor(value: Any, *, label: str, ndim: int) -> Any:
    torch = _torch()
    allowed = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or value.dtype not in allowed
    ):
        raise Lfm2MlpQdeimError(
            f"{label} must be one dense materialized floating rank-{ndim} tensor."
        )
    detached = value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    if not bool(torch.isfinite(detached).all().item()):
        raise Lfm2MlpQdeimError(f"{label} contains non-finite values.")
    return detached


def _tensor_hash(value: Any, *, label: str) -> str:
    try:
        return lfm2_authenticated_tensor_sha256(value)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise Lfm2MlpQdeimError(
            f"{label} must be one hashable materialized tensor."
        ) from exc


def _state_sha256(values: Mapping[str, Any], *, schema: str) -> str:
    return canonical_sha256(
        {
            "schema": schema,
            "tensor_sha256": {
                name: _tensor_hash(value, label=name)
                for name, value in sorted(values.items())
            },
        }
    )


def lfm2_mlp_qdeim_source_state_sha256(
    source_gate_weight: Any,
    source_up_weight: Any,
    source_down_weight: Any,
) -> str:
    """Hash the coupled authenticated source W1/W3/W2 state."""

    return _state_sha256(
        {
            "w1.weight": source_gate_weight,
            "w3.weight": source_up_weight,
            "w2.weight": source_down_weight,
        },
        schema="neural-forge-lfm2-mlp-qdeim-source-state-v1",
    )


def _target_template_spec(
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
) -> dict[str, dict[str, Any]]:
    torch = _torch()
    values = {
        "w1.weight": target_gate_weight_template,
        "w3.weight": target_up_weight_template,
        "w2.weight": target_down_weight_template,
    }
    spec: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.ndim != 2
            or any(int(size) <= 0 for size in value.shape)
            or value.dtype != torch.bfloat16
        ):
            raise Lfm2MlpQdeimError(
                f"target {name} must be one dense materialized BF16 rank-2 template."
            )
        # Template values are intentionally never read.
        spec[name] = {
            "shape": [int(size) for size in value.shape],
            "dtype": "bfloat16",
        }
    return spec


def lfm2_mlp_qdeim_target_template_sha256(
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
) -> str:
    """Hash only target shape/dtype metadata; blank values are not consumed."""

    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-mlp-qdeim-target-template-v1",
            "tensors": _target_template_spec(
                target_gate_weight_template,
                target_up_weight_template,
                target_down_weight_template,
            ),
        }
    )


def _case_weights(
    cases: Sequence[Lfm2MlpQdeimFitCase],
    supplied: Sequence[float] | None,
) -> tuple[float, ...]:
    if not cases or not all(isinstance(case, Lfm2MlpQdeimFitCase) for case in cases):
        raise Lfm2MlpQdeimError(
            "fit_cases must contain at least one Lfm2MlpQdeimFitCase."
        )
    if supplied is None:
        return tuple(1.0 for _ in cases)
    if isinstance(supplied, (str, bytes)) or len(supplied) != len(cases):
        raise Lfm2MlpQdeimError(
            "case_sample_weights must contain one positive value per FIT case."
        )
    weights = []
    for index, value in enumerate(supplied):
        parsed = _finite_float(value, label=f"case_sample_weights[{index}]")
        if parsed <= 0.0:
            raise Lfm2MlpQdeimError(
                "case_sample_weights must contain positive finite values."
            )
        weights.append(parsed)
    return tuple(weights)


def _nonempty_token(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2MlpQdeimError(f"{label} must be one non-empty string.")
    return value.strip()


def _normalized_functional_snapshot_blocks(
    supplied: Sequence[Lfm2MlpQdeimFunctionalSnapshotBlock],
) -> tuple[dict[str, Any], ...]:
    torch = _torch()
    if isinstance(supplied, (str, bytes)):
        raise Lfm2MlpQdeimError(
            "functional_snapshot_blocks must be a sequence of snapshot blocks."
        )
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, block in enumerate(tuple(supplied)):
        prefix = f"functional_snapshot_blocks[{index}]"
        if not isinstance(block, Lfm2MlpQdeimFunctionalSnapshotBlock):
            raise Lfm2MlpQdeimError(
                f"{prefix} must be one Lfm2MlpQdeimFunctionalSnapshotBlock."
            )
        block_id = _nonempty_token(block.block_id, label=f"{prefix}.block_id")
        if block_id in seen_ids:
            raise Lfm2MlpQdeimError("functional snapshot block IDs must be unique.")
        seen_ids.add(block_id)
        kind = _nonempty_token(
            block.snapshot_kind,
            label=f"{prefix}.snapshot_kind",
        )
        if kind not in _ALLOWED_FUNCTIONAL_SNAPSHOT_KINDS:
            raise Lfm2MlpQdeimError(
                f"{prefix}.snapshot_kind must be an authenticated structural kind."
            )
        receipt_sha256 = _digest(
            block.generator_receipt_sha256,
            label=f"{prefix}.generator_receipt_sha256",
        )
        rows = _materialized_float_tensor(
            block.functional_rows,
            label=f"{prefix}.functional_rows",
            ndim=2,
        )
        row_count = int(rows.shape[0])
        block_weight = _finite_float(
            block.block_weight,
            label=f"{prefix}.block_weight",
        )
        if block_weight <= 0.0:
            raise Lfm2MlpQdeimError(
                "functional snapshot block weights must be positive and finite."
            )
        if block.row_sample_weights is None:
            row_weights = tuple(1.0 for _ in range(row_count))
        else:
            if (
                isinstance(block.row_sample_weights, (str, bytes))
                or len(block.row_sample_weights) != row_count
            ):
                raise Lfm2MlpQdeimError(
                    f"{prefix}.row_sample_weights must contain one positive "
                    "value per structural row."
                )
            parsed_weights = []
            for row_index, value in enumerate(block.row_sample_weights):
                parsed = _finite_float(
                    value,
                    label=f"{prefix}.row_sample_weights[{row_index}]",
                )
                if parsed <= 0.0:
                    raise Lfm2MlpQdeimError(
                        "functional snapshot row weights must be positive and finite."
                    )
                parsed_weights.append(parsed)
            row_weights = tuple(parsed_weights)
        effective_weights = torch.tensor(
            [block_weight * value for value in row_weights],
            dtype=torch.float64,
        )
        if not bool(torch.isfinite(effective_weights).all().item()):
            raise Lfm2MlpQdeimError(
                "effective functional snapshot weights must remain finite."
            )
        normalized.append(
            {
                "block_id": block_id,
                "snapshot_kind": kind,
                "functional_rows_source": block.functional_rows,
                "functional_rows": rows,
                "generator_receipt_sha256": receipt_sha256,
                "block_weight": block_weight,
                "row_sample_weights": row_weights,
                "effective_row_weights": effective_weights,
            }
        )
    return tuple(normalized)


def lfm2_mlp_qdeim_functional_snapshot_evidence_sha256(
    functional_snapshot_blocks: Sequence[Lfm2MlpQdeimFunctionalSnapshotBlock],
) -> str:
    """Hash authenticated tangent/structural POD rows and all row weights."""

    normalized = _normalized_functional_snapshot_blocks(functional_snapshot_blocks)
    return canonical_sha256(
        {
            "schema": LFM2_MLP_QDEIM_FUNCTIONAL_SNAPSHOT_EVIDENCE_SCHEMA,
            "blocks": [
                {
                    "block_id": value["block_id"],
                    "snapshot_kind": value["snapshot_kind"],
                    "functional_rows_sha256": _tensor_hash(
                        value["functional_rows_source"],
                        label=f"functional snapshot {value['block_id']} rows",
                    ),
                    "functional_rows_shape": [
                        int(size) for size in value["functional_rows"].shape
                    ],
                    "generator_receipt_sha256": value["generator_receipt_sha256"],
                    "block_weight": value["block_weight"],
                    "row_sample_weights": list(value["row_sample_weights"]),
                }
                for value in normalized
            ],
        }
    )


def lfm2_mlp_qdeim_fit_evidence_sha256(
    fit_cases: Sequence[Lfm2MlpQdeimFitCase],
    case_sample_weights: Sequence[float] | None = None,
) -> str:
    """Hash every FIT tensor and its case weight before compilation."""

    cases = tuple(fit_cases)
    weights = _case_weights(cases, case_sample_weights)
    rows = []
    for index, (case, weight) in enumerate(zip(cases, weights, strict=True)):
        tensors = {
            "target_normalized_input": case.target_normalized_input,
            "parent_post_swiglu": case.parent_post_swiglu,
            "parent_post_mlp_residual_target_space": (
                case.parent_post_mlp_residual_target_space
            ),
            "target_pre_mlp_residual": case.target_pre_mlp_residual,
        }
        rows.append(
            {
                "case_index": index,
                "case_sample_weight": weight,
                "tensor_sha256": {
                    name: _tensor_hash(value, label=f"fit_cases[{index}].{name}")
                    for name, value in sorted(tensors.items())
                },
            }
        )
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-mlp-qdeim-fit-evidence-v1",
            "cases": rows,
        }
    )


def _canonicalize_pod_signs(basis: Any) -> Any:
    torch = _torch()
    canonical = basis.contiguous().clone()
    for column in range(int(canonical.shape[1])):
        values = canonical[:, column]
        pivot = int(torch.argmax(values.abs()).item())
        if float(values[pivot].item()) < 0.0:
            canonical[:, column].neg_()
    return canonical.contiguous()


def _deterministic_pivoted_qr_indices(basis: Any) -> tuple[tuple[int, ...], Any]:
    """Incremental column-pivoted modified QR with stable index ties.

    The selected pivot column is still explicitly reorthogonalized twice.  The
    remaining-column norms are updated by the resulting rank-one projection,
    avoiding a full residual-matrix reconstruction at every pivot.
    """

    torch = _torch()
    source_neurons, modes = map(int, basis.shape)
    matrix = basis.T.contiguous()
    selected: list[int] = []
    q_columns = torch.empty(
        (modes, modes),
        dtype=torch.float64,
        device=matrix.device,
    )
    scores = matrix.square().sum(dim=0)
    selected_mask = torch.zeros(
        source_neurons,
        dtype=torch.bool,
        device=matrix.device,
    )
    pivot_residual_norms: list[float] = []
    epsilon = torch.finfo(torch.float64).eps
    for step in range(modes):
        eligible_scores = scores.masked_fill(selected_mask, -1.0)
        maximum = float(eligible_scores.max().item())
        tie_tolerance = 128.0 * epsilon * max(1.0, abs(maximum))
        candidates = torch.nonzero(
            (~selected_mask) & (scores >= maximum - tie_tolerance),
            as_tuple=False,
        ).flatten()
        if not int(candidates.numel()):
            raise Lfm2MlpQdeimError("Pivoted QR found no remaining source neuron.")
        pivot = int(candidates.min().item())
        vector = matrix[:, pivot].contiguous().clone()
        if step:
            prior = q_columns[:, :step]
            vector.sub_(prior @ (prior.T @ vector))
            vector.sub_(prior @ (prior.T @ vector))
        norm = float(torch.linalg.vector_norm(vector).item())
        if not math.isfinite(norm) or norm <= epsilon:
            raise Lfm2MlpQdeimError(
                "Pivoted QR encountered a structurally singular POD interpolation."
            )
        q = vector / norm
        q_columns[:, step].copy_(q)
        projection = q @ matrix
        scores.sub_(projection.square()).clamp_min_(0.0)
        selected_mask[pivot] = True
        selected.append(pivot)
        pivot_residual_norms.append(norm)
    return tuple(selected), torch.tensor(pivot_residual_norms, dtype=torch.float64)


def _deterministic_source_covering_pivoted_qr_indices(
    basis: Any,
    source_index_groups: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], Any]:
    """Find a covering independent seed, then extend it with global QRCP.

    The seed search is a deterministic complete search over one candidate per
    source group.  A branch is retained when its selected POD rows are linearly
    independent under twice-reorthogonalized FP64 QR.  At every covering leaf,
    deterministic QRCP fills remaining modes and the *final square sampled
    basis* is tested with the same SVD tolerance used by the compiler.  A weak
    above-epsilon seed therefore cannot terminate search and cause a later
    false rejection.  Completeness follows because every prefix of an
    independent transversal is independent: enumerating every independent
    choice in a fixed group order visits every covering seed, and accepts the
    first whose deterministic completion passes the declared final predicate.
    This is exponential in the adversarial case, but span layer counts are
    small and ordering the most constrained groups first limits normal search.

    Once one independent row from every source is fixed, ordinary global QRCP
    fills the remaining target modes.  Seeding first is essential: greedily
    spending spare pivots before source coverage can strand a final source in
    the span of earlier choices even though a covering full-rank basis exists.
    """

    torch = _torch()
    if (
        not isinstance(basis, torch.Tensor)
        or basis.ndim != 2
        or basis.dtype != torch.float64
        or basis.layout != torch.strided
        or basis.device.type == "meta"
    ):
        raise Lfm2MlpQdeimError(
            "A dense materialized FP64 POD basis is required for span Q-DEIM."
        )
    source_neurons, modes = map(int, basis.shape)
    try:
        groups = tuple(
            tuple(int(index) for index in group) for group in source_index_groups
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2MlpQdeimError(
            "source_index_groups must contain integer neuron indices."
        ) from exc
    if (
        not groups
        or any(not group for group in groups)
        or modes < len(groups)
        or tuple(sorted(index for group in groups for index in group))
        != tuple(range(source_neurons))
    ):
        raise Lfm2MlpQdeimError(
            "Source index groups must exactly partition the union neuron axis, "
            "and target width must retain at least one neuron per source layer."
        )
    groups = tuple(tuple(sorted(group)) for group in groups)
    for group in groups:
        if len(set(group)) != len(group):
            raise Lfm2MlpQdeimError(
                "Source index groups must not repeat neuron indices."
            )

    matrix = basis.T.contiguous()
    epsilon = torch.finfo(torch.float64).eps
    empty_q = torch.empty(
        (modes, 0),
        dtype=torch.float64,
        device=matrix.device,
    )

    def independent_extension(index: int, q_columns: Any) -> tuple[Any, float] | None:
        vector = matrix[:, index].contiguous().clone()
        if int(q_columns.shape[1]):
            vector.sub_(q_columns @ (q_columns.T @ vector))
            vector.sub_(q_columns @ (q_columns.T @ vector))
        norm = float(torch.linalg.vector_norm(vector).item())
        if not math.isfinite(norm) or norm <= epsilon:
            return None
        return (vector / norm).contiguous(), norm

    def independent_group_scores(
        group: tuple[int, ...],
        q_columns: Any,
    ) -> list[tuple[float, int]]:
        indices = torch.tensor(
            group,
            dtype=torch.long,
            device=matrix.device,
        )
        vectors = matrix.index_select(1, indices).contiguous()
        if int(q_columns.shape[1]):
            vectors.sub_(q_columns @ (q_columns.T @ vectors))
            vectors.sub_(q_columns @ (q_columns.T @ vectors))
        norms = torch.linalg.vector_norm(vectors, dim=0)
        finite = torch.isfinite(norms) & (norms > epsilon)
        valid_positions = torch.nonzero(finite, as_tuple=False).flatten()
        if not int(valid_positions.numel()):
            return []
        valid_norms = norms.index_select(0, valid_positions).detach().cpu().tolist()
        valid_indices = indices.index_select(0, valid_positions).detach().cpu().tolist()
        return sorted(
            (
                (-float(norm), int(index))
                for norm, index in zip(valid_norms, valid_indices, strict=True)
            ),
            key=lambda value: (value[0], value[1]),
        )

    # The smallest groups are the most constrained.  The original group index
    # is the stable tie-break, so the result is independent of set/hash order.
    group_order = tuple(
        sorted(range(len(groups)), key=lambda index: (len(groups[index]), index))
    )

    def stable_qrcp_completion(
        seed_selected: tuple[int, ...],
        seed_residual_norms: tuple[float, ...],
        seed_q: Any,
    ) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
        selected = list(seed_selected)
        pivot_residual_norms = list(seed_residual_norms)
        q_columns = torch.empty(
            (modes, modes),
            dtype=torch.float64,
            device=matrix.device,
        )
        q_columns[:, : len(selected)].copy_(seed_q)
        selected_mask = torch.zeros(
            source_neurons,
            dtype=torch.bool,
            device=matrix.device,
        )
        selected_mask[list(selected)] = True
        prior = q_columns[:, : len(selected)]
        residual = matrix - prior @ (prior.T @ matrix)
        residual.sub_(prior @ (prior.T @ residual))
        scores = residual.square().sum(dim=0)

        for step in range(len(selected), modes):
            eligible_scores = scores.masked_fill(selected_mask, -1.0)
            maximum = float(eligible_scores.max().item())
            tie_tolerance = 128.0 * epsilon * max(1.0, abs(maximum))
            candidates = torch.nonzero(
                (~selected_mask) & (scores >= maximum - tie_tolerance),
                as_tuple=False,
            ).flatten()
            if not int(candidates.numel()):
                return None
            pivot = int(candidates.min().item())
            extension = independent_extension(pivot, q_columns[:, :step])
            if extension is None:
                return None
            q, norm = extension
            q_columns[:, step].copy_(q)
            projection = q @ matrix
            scores.sub_(projection.square()).clamp_min_(0.0)
            selected_mask[pivot] = True
            selected.append(pivot)
            pivot_residual_norms.append(norm)

        sampled_basis = basis[list(selected), :].contiguous()
        sampled_singular = torch.linalg.svdvals(sampled_basis)
        sampled_minimum = float(sampled_singular.min().item())
        sampled_maximum = float(sampled_singular.max().item())
        sampled_tolerance = max(sampled_basis.shape) * epsilon * sampled_maximum
        if (
            not math.isfinite(sampled_minimum)
            or not math.isfinite(sampled_maximum)
            or sampled_minimum <= sampled_tolerance
        ):
            return None
        return tuple(selected), tuple(pivot_residual_norms)

    def covering_seed_search(
        depth: int,
        q_columns: Any,
        selected: tuple[int, ...],
        residual_norms: tuple[float, ...],
    ) -> tuple[tuple[int, ...], tuple[float, ...]] | None:
        if depth == len(group_order):
            return stable_qrcp_completion(selected, residual_norms, q_columns)
        group = groups[group_order[depth]]
        scored = independent_group_scores(group, q_columns)
        for _negative_norm, index in scored:
            extension = independent_extension(index, q_columns)
            if extension is None:  # pragma: no cover - q_columns is unchanged
                continue
            q_column, norm = extension
            next_q = torch.cat((q_columns, q_column.unsqueeze(1)), dim=1).contiguous()
            # Cheap sound pruning: every remaining source must still expose at
            # least one vector outside the current span.  Failure proves this
            # branch cannot complete, while success merely permits recursion.
            if any(
                not independent_group_scores(groups[remaining_group], next_q)
                for remaining_group in group_order[depth + 1 :]
            ):
                continue
            result = covering_seed_search(
                depth + 1,
                next_q,
                (*selected, index),
                (*residual_norms, norm),
            )
            if result is not None:
                return result
        return None

    selection = covering_seed_search(0, empty_q, (), ())
    if selection is None:
        raise Lfm2MlpQdeimError(
            "No source-covering Q-DEIM seed produced a final sampled basis "
            "that passes the compiler SVD stability criterion."
        )
    selected, pivot_residual_norms = selection
    return selected, torch.tensor(pivot_residual_norms, dtype=torch.float64)


def _relative_rmse(prediction: Any, target: Any) -> float:
    torch = _torch()
    residual = prediction - target
    numerator = torch.mean(residual.square())
    denominator = torch.mean(target.square()).clamp_min(torch.finfo(torch.float64).tiny)
    return float(torch.sqrt(numerator / denominator).item())


def _candidate_state_sha256(gate: Any, up: Any, down: Any) -> str:
    return _state_sha256(
        {"w1.weight": gate, "w3.weight": up, "w2.weight": down},
        schema="neural-forge-lfm2-mlp-qdeim-materialized-candidate-state-v1",
    )


def _rank_deficit_report(
    *,
    source_state_sha256: str,
    target_template_sha256: str,
    normalized_input_decoder_sha256: str,
    residual_output_chart_sha256: str,
    fit_evidence_sha256: str,
    functional_snapshot_evidence_sha256: str,
    weighted_features: Any,
    singular_values: Any,
    rank_tolerance: float,
    observed_rank: int,
    required_rank: int,
    source_intermediate: int,
    fit_snapshot_row_count: int,
    structural_snapshot_row_count: int,
    structural_snapshot_block_count: int,
    compute_execution: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": LFM2_MLP_QDEIM_RANK_DEFICIT_SCHEMA,
        "status": "insufficient-authenticated-fit-snapshot-rank",
        "source_state_sha256": source_state_sha256,
        "target_template_sha256": target_template_sha256,
        "normalized_input_decoder_sha256": normalized_input_decoder_sha256,
        "residual_output_chart_sha256": residual_output_chart_sha256,
        "fit_evidence_sha256": fit_evidence_sha256,
        "functional_snapshot_evidence_sha256": (functional_snapshot_evidence_sha256),
        "compute_execution": dict(compute_execution),
        "weighted_functional_snapshots_sha256": _tensor_hash(
            weighted_features,
            label="weighted FIT post-SwiGLU features",
        ),
        "pod_singular_values_sha256": _tensor_hash(
            singular_values,
            label="FIT POD singular values",
        ),
        "weighted_functional_snapshot_row_count": int(weighted_features.shape[0]),
        "fit_post_swiglu_row_count": fit_snapshot_row_count,
        "structural_completion_row_count": structural_snapshot_row_count,
        "structural_completion_block_count": structural_snapshot_block_count,
        "source_intermediate_size": source_intermediate,
        "target_intermediate_size": required_rank,
        "minimum_snapshot_rank": required_rank,
        "observed_snapshot_rank": observed_rank,
        "snapshot_rank_deficit": required_rank - observed_rank,
        "snapshot_rank_absolute_tolerance": rank_tolerance,
        "selected_source_neurons": [],
        "interpolation_matrix_materialized": False,
        "random_mode_completion_permitted": False,
        "silent_pseudoinverse_completion_permitted": False,
        "future_authenticated_completion_modes": [
            "analytic_child_lifted_parent_swiglu_tangent",
            "authenticated_structural",
        ],
        "structural_completion_input_contract": {
            "public_type": "Lfm2MlpQdeimFunctionalSnapshotBlock",
            "functional_rows_shape": ["row_count", source_intermediate],
            "coordinate_axis": "parent_post_swiglu_neuron",
            "positive_block_weight_required": True,
            "positive_row_sample_weights_required": True,
            "generator_receipt_sha256_required": True,
            "fit_endpoint_correction_eligibility": False,
            "heldout_evidence_eligibility": False,
        },
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _fit_w2_residual_delta(
    design: Any,
    residual_target: Any,
    *,
    ridge: float,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    left, singular, right_h = torch.linalg.svd(design, full_matrices=False)
    maximum = float(singular.max().item()) if singular.numel() else 0.0
    tolerance = max(design.shape) * torch.finfo(torch.float64).eps * maximum
    supported = singular > tolerance
    rank = int(supported.sum().item())
    if rank == 0:
        delta_t = torch.zeros(
            (int(design.shape[1]), int(residual_target.shape[1])),
            dtype=torch.float64,
            device=design.device,
        )
        scaled_ridge = 0.0
    else:
        energy_scale = float(singular[supported].square().mean().item())
        scaled_ridge = ridge * max(energy_scale, torch.finfo(torch.float64).tiny)
        active_singular = singular[supported]
        if scaled_ridge == 0.0:
            spectral_filter = active_singular.reciprocal()
        else:
            spectral_filter = active_singular / (
                active_singular.square() + scaled_ridge
            )
        delta_t = right_h[supported].T @ (
            spectral_filter.unsqueeze(1) * (left[:, supported].T @ residual_target)
        )
    report = {
        "method": "fit_only_fp64_svd_prior_residual_ridge",
        "fit_design_rank": rank,
        "fit_design_rank_tolerance": tolerance,
        "ridge_policy_value": ridge,
        "scaled_ridge": scaled_ridge,
        "fit_design_singular_values_sha256": _tensor_hash(
            singular,
            label="FIT correction singular values",
        ),
        "analytic_closed_form": True,
        "fit_rows_only": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    return delta_t.contiguous(), report


def compile_lfm2_mlp_nonlinear_qdeim(
    *,
    source_gate_weight: Any,
    source_up_weight: Any,
    source_down_weight: Any,
    expected_source_state_sha256: str,
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
    expected_target_template_sha256: str,
    normalized_input_decoder: Any,
    expected_normalized_input_decoder_sha256: str,
    residual_output_chart: Any,
    expected_residual_output_chart_sha256: str,
    fit_cases: Sequence[Lfm2MlpQdeimFitCase],
    expected_fit_evidence_sha256: str,
    case_sample_weights: Sequence[float] | None = None,
    functional_snapshot_blocks: Sequence[Lfm2MlpQdeimFunctionalSnapshotBlock] = (),
    expected_functional_snapshot_evidence_sha256: str | None = None,
    compute_fit_residual_correction: bool = True,
    fit_correction_ridge: float = 1e-8,
    pod_rank_relative_tolerance: float | None = None,
    compute_device: str = "cpu",
) -> Lfm2CompiledMlpQdeim:
    """Compile dense real-neuron Q-DEIM candidates without training.

    FIT post-SwiGLU rows and any separately authenticated structural snapshot
    blocks define an uncentered weighted FP64 functional POD. Deterministic
    column-pivoted QR of its transpose chooses exactly the target intermediate
    width of real source neurons. Selected W1/W3 rows are copied intact through
    ``normalized_input_decoder``. The dense Q-DEIM reconstruction is multiplied
    by every mapped source W2 column, so omitted nonlinear feature contributions
    are baked into the target structural W2. Structural snapshot rows never
    enter the optional FIT endpoint W2 correction.
    """

    torch = _torch()
    resolved_compute_device = _resolve_qdeim_compute_device(torch, compute_device)
    compute_execution = _qdeim_compute_receipt(
        torch,
        compute_device,
        resolved_compute_device,
    )
    if not isinstance(compute_fit_residual_correction, bool):
        raise Lfm2MlpQdeimError("compute_fit_residual_correction must be boolean.")
    fit_correction_ridge = _finite_float(
        fit_correction_ridge,
        label="fit_correction_ridge",
        nonnegative=True,
    )
    if pod_rank_relative_tolerance is not None:
        pod_rank_relative_tolerance = _finite_float(
            pod_rank_relative_tolerance,
            label="pod_rank_relative_tolerance",
            nonnegative=True,
        )

    expected_source = _digest(
        expected_source_state_sha256,
        label="expected_source_state_sha256",
    )
    expected_template = _digest(
        expected_target_template_sha256,
        label="expected_target_template_sha256",
    )
    expected_decoder = _digest(
        expected_normalized_input_decoder_sha256,
        label="expected_normalized_input_decoder_sha256",
    )
    expected_chart = _digest(
        expected_residual_output_chart_sha256,
        label="expected_residual_output_chart_sha256",
    )
    expected_fit = _digest(
        expected_fit_evidence_sha256,
        label="expected_fit_evidence_sha256",
    )

    cases = tuple(fit_cases)
    case_weights = _case_weights(cases, case_sample_weights)
    structural_blocks = _normalized_functional_snapshot_blocks(
        functional_snapshot_blocks
    )
    structural_snapshot_sha256 = lfm2_mlp_qdeim_functional_snapshot_evidence_sha256(
        functional_snapshot_blocks
    )
    if expected_functional_snapshot_evidence_sha256 is None:
        if structural_blocks:
            raise Lfm2MlpQdeimError(
                "Non-empty functional snapshot blocks require an expected "
                "evidence SHA-256 digest."
            )
    else:
        expected_structural_snapshot = _digest(
            expected_functional_snapshot_evidence_sha256,
            label="expected_functional_snapshot_evidence_sha256",
        )
        if expected_structural_snapshot != structural_snapshot_sha256:
            raise Lfm2MlpQdeimError(
                "Authenticated functional snapshot evidence changed."
            )
    source_before = lfm2_mlp_qdeim_source_state_sha256(
        source_gate_weight,
        source_up_weight,
        source_down_weight,
    )
    if source_before != expected_source:
        raise Lfm2MlpQdeimError("Authenticated source W1/W3/W2 state changed.")
    template_sha256 = lfm2_mlp_qdeim_target_template_sha256(
        target_gate_weight_template,
        target_up_weight_template,
        target_down_weight_template,
    )
    if template_sha256 != expected_template:
        raise Lfm2MlpQdeimError("Target BF16 template metadata changed.")
    decoder_sha256 = _tensor_hash(
        normalized_input_decoder,
        label="normalized_input_decoder",
    )
    chart_sha256 = _tensor_hash(
        residual_output_chart,
        label="residual_output_chart",
    )
    if decoder_sha256 != expected_decoder:
        raise Lfm2MlpQdeimError("Normalized-input decoder authentication failed.")
    if chart_sha256 != expected_chart:
        raise Lfm2MlpQdeimError("Residual output chart authentication failed.")
    fit_sha256 = lfm2_mlp_qdeim_fit_evidence_sha256(cases, case_weights)
    if fit_sha256 != expected_fit:
        raise Lfm2MlpQdeimError("Authenticated FIT Q-DEIM evidence changed.")

    source_gate = _materialized_float_tensor(
        source_gate_weight,
        label="source_gate_weight",
        ndim=2,
    )
    source_up = _materialized_float_tensor(
        source_up_weight,
        label="source_up_weight",
        ndim=2,
    )
    source_down = _materialized_float_tensor(
        source_down_weight,
        label="source_down_weight",
        ndim=2,
    )
    source_intermediate, source_hidden = map(int, source_gate.shape)
    if tuple(source_up.shape) != tuple(source_gate.shape) or tuple(
        source_down.shape
    ) != (source_hidden, source_intermediate):
        raise Lfm2MlpQdeimError(
            "Authenticated source tensors do not form one coupled LFM2 SwiGLU."
        )
    for block in structural_blocks:
        if int(block["functional_rows"].shape[1]) != source_intermediate:
            raise Lfm2MlpQdeimError(
                "Functional snapshot rows must use the authenticated parent "
                "post-SwiGLU neuron coordinate axis."
            )

    template_spec = _target_template_spec(
        target_gate_weight_template,
        target_up_weight_template,
        target_down_weight_template,
    )
    target_intermediate, target_hidden = template_spec["w1.weight"]["shape"]
    if template_spec["w3.weight"]["shape"] != [
        target_intermediate,
        target_hidden,
    ] or template_spec["w2.weight"]["shape"] != [target_hidden, target_intermediate]:
        raise Lfm2MlpQdeimError(
            "Target templates do not form one atomic dense LFM2 SwiGLU."
        )
    if target_intermediate >= source_intermediate or target_hidden > source_hidden:
        raise Lfm2MlpQdeimError(
            "Q-DEIM target must reduce intermediate width and cannot expand hidden width."
        )

    decoder = _materialized_float_tensor(
        normalized_input_decoder,
        label="normalized_input_decoder",
        ndim=2,
    )
    chart = _materialized_float_tensor(
        residual_output_chart,
        label="residual_output_chart",
        ndim=2,
    )
    if tuple(decoder.shape) != (target_hidden, source_hidden):
        raise Lfm2MlpQdeimError(
            "normalized_input_decoder must have shape [target_hidden, source_hidden]."
        )
    if tuple(chart.shape) != (source_hidden, target_hidden):
        raise Lfm2MlpQdeimError(
            "residual_output_chart must have shape [source_hidden, target_hidden]."
        )

    with torch.no_grad():
        # CPU tensors above remain immutable authenticated archives.  Stage
        # only this layer's active FP64 algebra to the requested accelerator.
        source_gate_work = source_gate.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        source_up_work = source_up.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        source_down_work = source_down.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        decoder_work = decoder.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        chart_work = chart.to(
            device=resolved_compute_device,
            dtype=torch.float64,
        )
        decoder_singular = torch.linalg.svdvals(decoder_work)
        chart_singular = torch.linalg.svdvals(chart_work)
        decoder_rank_tolerance = (
            max(decoder_work.shape)
            * torch.finfo(torch.float64).eps
            * decoder_singular.max()
        )
        chart_rank_tolerance = (
            max(chart_work.shape)
            * torch.finfo(torch.float64).eps
            * chart_singular.max()
        )
        decoder_rank = int((decoder_singular > decoder_rank_tolerance).sum().item())
        chart_rank = int((chart_singular > chart_rank_tolerance).sum().item())
        if decoder_rank != target_hidden or chart_rank != target_hidden:
            raise Lfm2MlpQdeimError(
                "Normalized-input decoder and residual chart must be structurally full rank."
            )

        normalized_cases: list[dict[str, Any]] = []
        for index, (case, case_weight) in enumerate(
            zip(cases, case_weights, strict=True)
        ):
            prefix = f"fit_cases[{index}]"
            target_input = _materialized_float_tensor(
                case.target_normalized_input,
                label=f"{prefix}.target_normalized_input",
                ndim=2,
            )
            parent_post = _materialized_float_tensor(
                case.parent_post_swiglu,
                label=f"{prefix}.parent_post_swiglu",
                ndim=2,
            )
            endpoint = _materialized_float_tensor(
                case.parent_post_mlp_residual_target_space,
                label=f"{prefix}.parent_post_mlp_residual_target_space",
                ndim=2,
            )
            target_residual = _materialized_float_tensor(
                case.target_pre_mlp_residual,
                label=f"{prefix}.target_pre_mlp_residual",
                ndim=2,
            )
            row_count = int(parent_post.shape[0])
            if (
                tuple(parent_post.shape) != (row_count, source_intermediate)
                or tuple(target_input.shape) != (row_count, target_hidden)
                or tuple(endpoint.shape) != (row_count, target_hidden)
                or tuple(target_residual.shape) != tuple(endpoint.shape)
            ):
                raise Lfm2MlpQdeimError(
                    f"{prefix} does not match the authenticated source/target topology."
                )
            normalized_cases.append(
                {
                    "target_input": target_input,
                    "parent_post": parent_post,
                    "endpoint": endpoint,
                    "target_residual": target_residual,
                    "case_weight": case_weight,
                }
            )

        fit_parent_features = torch.cat(
            [
                value["parent_post"].to(
                    device=resolved_compute_device,
                    dtype=torch.float64,
                )
                for value in normalized_cases
            ],
            dim=0,
        )
        target_inputs = torch.cat(
            [
                value["target_input"].to(
                    device=resolved_compute_device,
                    dtype=torch.float64,
                )
                for value in normalized_cases
            ],
            dim=0,
        )
        endpoint_target = torch.cat(
            [
                (value["endpoint"] - value["target_residual"]).to(
                    device=resolved_compute_device,
                    dtype=torch.float64,
                )
                for value in normalized_cases
            ],
            dim=0,
        )
        fit_row_weights = torch.cat(
            [
                torch.full(
                    (int(value["parent_post"].shape[0]),),
                    float(value["case_weight"]),
                    dtype=torch.float64,
                    device=resolved_compute_device,
                )
                for value in normalized_cases
            ]
        )
        structural_rows = [
            value["functional_rows"].to(
                device=resolved_compute_device,
                dtype=torch.float64,
            )
            for value in structural_blocks
        ]
        structural_row_weights = [
            value["effective_row_weights"].to(
                device=resolved_compute_device,
                dtype=torch.float64,
            )
            for value in structural_blocks
        ]
        pod_features = torch.cat(
            [fit_parent_features, *structural_rows],
            dim=0,
        )
        pod_row_weights = torch.cat(
            [fit_row_weights, *structural_row_weights],
            dim=0,
        )
        fit_sqrt_row_weights = torch.sqrt(fit_row_weights).unsqueeze(1)
        pod_sqrt_row_weights = torch.sqrt(pod_row_weights).unsqueeze(1)
        weighted_features = (pod_features * pod_sqrt_row_weights).contiguous()

        _pod_left, pod_singular, pod_right_h = torch.linalg.svd(
            weighted_features,
            full_matrices=False,
        )
        singular_maximum = float(pod_singular.max().item())
        default_rank_tolerance = (
            max(weighted_features.shape)
            * torch.finfo(torch.float64).eps
            * singular_maximum
        )
        rank_tolerance = (
            default_rank_tolerance
            if pod_rank_relative_tolerance is None
            else pod_rank_relative_tolerance * singular_maximum
        )
        observed_rank = int((pod_singular > rank_tolerance).sum().item())
        if observed_rank < target_intermediate:
            raise Lfm2MlpQdeimInsufficientRankError(
                _rank_deficit_report(
                    source_state_sha256=source_before,
                    target_template_sha256=template_sha256,
                    normalized_input_decoder_sha256=decoder_sha256,
                    residual_output_chart_sha256=chart_sha256,
                    fit_evidence_sha256=fit_sha256,
                    functional_snapshot_evidence_sha256=(structural_snapshot_sha256),
                    weighted_features=weighted_features,
                    singular_values=pod_singular,
                    rank_tolerance=rank_tolerance,
                    observed_rank=observed_rank,
                    required_rank=target_intermediate,
                    source_intermediate=source_intermediate,
                    fit_snapshot_row_count=int(fit_parent_features.shape[0]),
                    structural_snapshot_row_count=sum(
                        int(value["functional_rows"].shape[0])
                        for value in structural_blocks
                    ),
                    structural_snapshot_block_count=len(structural_blocks),
                    compute_execution=compute_execution,
                )
            )

        pod_basis = _canonicalize_pod_signs(
            pod_right_h[:target_intermediate].T.contiguous()
        )
        selected, pivot_residual_norms = _deterministic_pivoted_qr_indices(pod_basis)
        sampled_basis = pod_basis[list(selected), :].contiguous()
        interpolation_singular = torch.linalg.svdvals(sampled_basis)
        interpolation_minimum = float(interpolation_singular.min().item())
        interpolation_maximum = float(interpolation_singular.max().item())
        structural_interpolation_tolerance = (
            max(sampled_basis.shape)
            * torch.finfo(torch.float64).eps
            * interpolation_maximum
        )
        if (
            not math.isfinite(interpolation_minimum)
            or interpolation_minimum <= structural_interpolation_tolerance
        ):
            raise Lfm2MlpQdeimError(
                "Q-DEIM sampled POD basis is structurally singular."
            )
        reconstruction = torch.linalg.solve(
            sampled_basis.T,
            pod_basis.T,
        ).contiguous()
        reconstructed_features = pod_features[:, list(selected)] @ reconstruction
        interpolation_relative_rmse = _relative_rmse(
            reconstructed_features * pod_sqrt_row_weights,
            weighted_features,
        )
        fit_reconstructed_features = (
            fit_parent_features[:, list(selected)] @ reconstruction
        )
        fit_interpolation_relative_rmse = _relative_rmse(
            fit_reconstructed_features * fit_sqrt_row_weights,
            fit_parent_features * fit_sqrt_row_weights,
        )
        selected_identity = reconstruction[:, list(selected)]
        selected_identity_error = float(
            (
                selected_identity
                - torch.eye(
                    target_intermediate,
                    dtype=torch.float64,
                    device=resolved_compute_device,
                )
            )
            .abs()
            .max()
            .item()
        )

        selected_gate = source_gate_work[list(selected), :].contiguous()
        selected_up = source_up_work[list(selected), :].contiguous()
        gate_prior_fp64 = (selected_gate @ decoder_work.T).contiguous()
        up_prior_fp64 = (selected_up @ decoder_work.T).contiguous()
        mapped_source_down = (chart_work.T @ source_down_work).contiguous()
        down_prior_fp64 = (mapped_source_down @ reconstruction.T).contiguous()

        materialized_gate = gate_prior_fp64.to(dtype=torch.bfloat16).contiguous()
        materialized_up = up_prior_fp64.to(dtype=torch.bfloat16).contiguous()
        materialized_down_prior = down_prior_fp64.to(dtype=torch.bfloat16).contiguous()
        target_features = torch.nn.functional.silu(
            target_inputs @ materialized_gate.to(torch.float64).T
        ) * (target_inputs @ materialized_up.to(torch.float64).T)
        prior_prediction = target_features @ materialized_down_prior.to(torch.float64).T
        prior_fit_relative_rmse = _relative_rmse(
            prior_prediction * fit_sqrt_row_weights,
            endpoint_target * fit_sqrt_row_weights,
        )

        materialized_down_corrected = None
        correction_report: dict[str, Any] = {
            "enabled": False,
            "candidate_materialized": False,
        }
        corrected_fit_relative_rmse = None
        if compute_fit_residual_correction:
            weighted_design = target_features * fit_sqrt_row_weights
            weighted_residual_target = (
                endpoint_target - prior_prediction
            ) * fit_sqrt_row_weights
            delta_t, analytic_report = _fit_w2_residual_delta(
                weighted_design,
                weighted_residual_target,
                ridge=fit_correction_ridge,
            )
            corrected_fp64 = (
                materialized_down_prior.to(torch.float64) + delta_t.T
            ).contiguous()
            materialized_down_corrected = corrected_fp64.to(
                dtype=torch.bfloat16
            ).contiguous()
            corrected_prediction = (
                target_features @ materialized_down_corrected.to(torch.float64).T
            )
            corrected_fit_relative_rmse = _relative_rmse(
                corrected_prediction * fit_sqrt_row_weights,
                endpoint_target * fit_sqrt_row_weights,
            )
            prior_norm = float(
                torch.linalg.vector_norm(
                    materialized_down_prior.to(torch.float64)
                ).item()
            )
            correction_norm = float(torch.linalg.vector_norm(delta_t).item())
            correction_report = {
                "enabled": True,
                "candidate_materialized": True,
                **analytic_report,
                "correction_delta_sha256": _tensor_hash(
                    delta_t.T,
                    label="FIT W2 correction delta",
                ),
                "correction_delta_frobenius_norm": correction_norm,
                "correction_delta_to_prior_norm_ratio": correction_norm
                / max(prior_norm, torch.finfo(torch.float64).tiny),
                "prior_fit_relative_rmse": prior_fit_relative_rmse,
                "corrected_fit_relative_rmse": corrected_fit_relative_rmse,
                "numeric_fit_diagnostics_have_selection_authority": False,
            }

    source_after = lfm2_mlp_qdeim_source_state_sha256(
        source_gate_weight,
        source_up_weight,
        source_down_weight,
    )
    candidate_tensors = [
        materialized_gate,
        materialized_up,
        materialized_down_prior,
        pod_basis,
        reconstruction,
    ]
    if materialized_down_corrected is not None:
        candidate_tensors.append(materialized_down_corrected)
    if (
        source_after != source_before
        or len(selected) != target_intermediate
        or len(set(selected)) != target_intermediate
        or any(index < 0 or index >= source_intermediate for index in selected)
        or any(
            tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all().item())
            for tensor in candidate_tensors
        )
        or tuple(materialized_gate.shape) != (target_intermediate, target_hidden)
        or tuple(materialized_up.shape) != tuple(materialized_gate.shape)
        or tuple(materialized_down_prior.shape) != (target_hidden, target_intermediate)
        or materialized_gate.dtype != torch.bfloat16
        or materialized_up.dtype != torch.bfloat16
        or materialized_down_prior.dtype != torch.bfloat16
        or (
            materialized_down_corrected is not None
            and (
                tuple(materialized_down_corrected.shape)
                != (target_hidden, target_intermediate)
                or materialized_down_corrected.dtype != torch.bfloat16
            )
        )
    ):
        raise Lfm2MlpQdeimError(
            "Materialized Q-DEIM candidates failed dense/no-grad/source invariants."
        )

    structural_state_sha256 = _candidate_state_sha256(
        materialized_gate,
        materialized_up,
        materialized_down_prior,
    )
    corrected_state_sha256 = (
        None
        if materialized_down_corrected is None
        else _candidate_state_sha256(
            materialized_gate,
            materialized_up,
            materialized_down_corrected,
        )
    )
    interpolation_condition = interpolation_maximum / interpolation_minimum
    reconstruction_sha256 = _tensor_hash(
        reconstruction,
        label="Q-DEIM reconstruction",
    )
    writer_formula_body = {
        "formula": (
            "W2_target_prior = residual_output_chart.T @ W2_source @ "
            "qdeim_reconstruction.T"
        ),
        "residual_output_chart_sha256": chart_sha256,
        "source_w2_sha256": _tensor_hash(
            source_down_weight,
            label="source_down_weight",
        ),
        "qdeim_reconstruction_sha256": reconstruction_sha256,
        "fp64_result_sha256": _tensor_hash(
            down_prior_fp64,
            label="FP64 Q-DEIM W2 structural prior",
        ),
        "materialized_bfloat16_result_sha256": _tensor_hash(
            materialized_down_prior,
            label="BF16 Q-DEIM W2 structural prior",
        ),
    }
    writer_formula_sha256 = canonical_sha256(writer_formula_body)
    candidate_ids = [LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE]
    if materialized_down_corrected is not None:
        candidate_ids.append(LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE)
    omitted = sorted(set(range(source_intermediate)).difference(selected))
    omitted_nonzero_columns = sum(
        float(torch.linalg.vector_norm(reconstruction[:, index]).item()) > 0.0
        for index in omitted
    )
    report_body = {
        "schema": LFM2_MLP_QDEIM_SCHEMA,
        "status": "dense-candidates-ready-for-decoded-whole-response-review",
        "compute_execution": compute_execution,
        "source_state_sha256_before": source_before,
        "source_state_sha256_after": source_after,
        "source_state_immutable": True,
        "source_tensor_sha256": {
            "w1.weight": _tensor_hash(source_gate_weight, label="source_gate_weight"),
            "w3.weight": _tensor_hash(source_up_weight, label="source_up_weight"),
            "w2.weight": _tensor_hash(source_down_weight, label="source_down_weight"),
        },
        "target_template_sha256": template_sha256,
        "target_template_values_read": False,
        "normalized_input_decoder_sha256": decoder_sha256,
        "residual_output_chart_sha256": chart_sha256,
        "fit_evidence_sha256": fit_sha256,
        "functional_snapshot_evidence_sha256": structural_snapshot_sha256,
        "functional_snapshot_blocks": [
            {
                "block_id": value["block_id"],
                "snapshot_kind": value["snapshot_kind"],
                "generator_receipt_sha256": value["generator_receipt_sha256"],
                "row_count": int(value["functional_rows"].shape[0]),
                "block_weight": value["block_weight"],
                "row_sample_weights_sha256": _tensor_hash(
                    torch.tensor(
                        value["row_sample_weights"],
                        dtype=torch.float64,
                    ),
                    label=f"functional snapshot {value['block_id']} row weights",
                ),
                "effective_row_weights_sha256": _tensor_hash(
                    value["effective_row_weights"],
                    label=(
                        f"functional snapshot {value['block_id']} effective weights"
                    ),
                ),
            }
            for value in structural_blocks
        ],
        "functional_snapshot_blocks_are_heldout_evidence": False,
        "functional_snapshot_blocks_enter_fit_w2_correction": False,
        "source_hidden_size": source_hidden,
        "target_hidden_size": target_hidden,
        "source_intermediate_size": source_intermediate,
        "target_intermediate_size": target_intermediate,
        "minimum_snapshot_rank": target_intermediate,
        "observed_snapshot_rank": observed_rank,
        "fit_post_swiglu_row_count": int(fit_parent_features.shape[0]),
        "structural_completion_row_count": sum(
            int(value["functional_rows"].shape[0]) for value in structural_blocks
        ),
        "weighted_functional_snapshot_row_count": int(weighted_features.shape[0]),
        "snapshot_rank_absolute_tolerance": rank_tolerance,
        "pod_mode_source": (
            "joint_authenticated_fit_post_swiglu_and_structural_snapshot_rows"
            if structural_blocks
            else "authenticated_fit_parent_post_swiglu_only"
        ),
        "pod_centering_applied": False,
        "pod_compute_dtype": "float64",
        "pod_basis_sha256": _tensor_hash(pod_basis, label="functional POD basis"),
        "pod_singular_values_sha256": _tensor_hash(
            pod_singular,
            label="functional POD singular values",
        ),
        "selected_source_neurons": list(selected),
        "selected_source_neuron_count": len(selected),
        "selection_method": (
            "deterministic_incremental_column_pivoted_modified_qr_of_"
            "functional_pod_transpose"
        ),
        "selection_compute_diagnostic": {
            "schema": "neural-forge-lfm2-mlp-qdeim-selection-compute-v1",
            "basis_transpose_shape": [target_intermediate, source_intermediate],
            "preallocated_q_shape": [target_intermediate, target_intermediate],
            "pivot_step_count": target_intermediate,
            "initial_full_score_pass_count": 1,
            "rank_one_score_update_count": target_intermediate,
            "full_residual_matrix_reconstruction_count": 0,
            "selected_pivot_reorthogonalization_pass_count": (
                2 * max(0, target_intermediate - 1)
            ),
            "asymptotic_time_bound": ("O(pivot_steps*pod_modes*source_neurons)"),
            "asymptotic_auxiliary_storage_bound": (
                "O(pod_modes*source_neurons+pod_modes^2+source_neurons)"
            ),
            "wall_clock_timing_recorded": False,
            "timing_used_for_candidate_selection": False,
            "timing_used_for_semantic_acceptance_or_veto": False,
            "performance_diagnostics_have_semantic_authority": False,
        },
        "selection_sha256": canonical_sha256(list(selected)),
        "pivot_residual_norms_sha256": _tensor_hash(
            pivot_residual_norms,
            label="Q-DEIM pivot residual norms",
        ),
        "functional_interpolation": {
            "formula": "parent_post_swiglu ~= selected_post_swiglu @ R",
            "reconstruction_shape": [target_intermediate, source_intermediate],
            "reconstruction_sha256": reconstruction_sha256,
            "interpolation_condition_number": interpolation_condition,
            "interpolation_smallest_singular_value": interpolation_minimum,
            "interpolation_largest_singular_value": interpolation_maximum,
            "selected_identity_maximum_absolute_error": selected_identity_error,
            "weighted_joint_snapshot_relative_rmse": interpolation_relative_rmse,
            "weighted_fit_post_swiglu_relative_rmse": (fit_interpolation_relative_rmse),
            "numeric_diagnostics_have_selection_authority": False,
        },
        "producer_transport": {
            "method": "intact_selected_source_rows_through_normalized_input_decoder",
            "producer_averaging": False,
            "w1_formula": "W1_source[selected, :] @ normalized_input_decoder.T",
            "w3_formula": "W3_source[selected, :] @ normalized_input_decoder.T",
            "normalized_input_decoder_rank": decoder_rank,
            "normalized_input_decoder_condition_number": float(
                decoder_singular.max().item() / decoder_singular.min().item()
            ),
        },
        "writer_transport": {
            **writer_formula_body,
            "writer_formula_sha256": writer_formula_sha256,
            "all_source_writer_columns_consumed": True,
            "source_writer_column_count": source_intermediate,
            "selected_source_writer_column_count": target_intermediate,
            "omitted_source_writer_column_count": len(omitted),
            "omitted_reconstruction_nonzero_column_count": omitted_nonzero_columns,
            "residual_output_chart_rank": chart_rank,
            "residual_output_chart_condition_number": float(
                chart_singular.max().item() / chart_singular.min().item()
            ),
        },
        "fit_only_w2_residual_correction": correction_report,
        "materialized_candidates": {
            LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE: {
                "state_sha256": structural_state_sha256,
                "w2_sha256": _tensor_hash(
                    materialized_down_prior,
                    label="structural candidate W2",
                ),
                "fit_relative_rmse_diagnostic": prior_fit_relative_rmse,
            },
            **(
                {
                    LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE: {
                        "state_sha256": corrected_state_sha256,
                        "w2_sha256": _tensor_hash(
                            materialized_down_corrected,
                            label="corrected candidate W2",
                        ),
                        "fit_relative_rmse_diagnostic": (corrected_fit_relative_rmse),
                    }
                }
                if materialized_down_corrected is not None
                else {}
            ),
        },
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "automatic_candidate_selection_performed": False,
        "automatic_candidate_acceptance_performed": False,
        "candidate_published": False,
        "decoded_whole_response_review_required": True,
        "numeric_diagnostics_are_advisory_only": True,
        "rank_completion": {
            "joint_authenticated_pod_modes": target_intermediate,
            "mode_attribution": "not_separable_after_joint_weighted_pod",
            "fit_post_swiglu_snapshot_rows": int(fit_parent_features.shape[0]),
            "analytic_tangent_snapshot_rows": sum(
                int(value["functional_rows"].shape[0])
                for value in structural_blocks
                if value["snapshot_kind"]
                == "analytic_child_lifted_parent_swiglu_tangent"
            ),
            "authenticated_structural_snapshot_rows": sum(
                int(value["functional_rows"].shape[0])
                for value in structural_blocks
                if value["snapshot_kind"] == "authenticated_structural"
            ),
            "random_completion_modes": 0,
            "silent_completion_performed": False,
        },
        "dense_bfloat16_target_state": True,
        "source_parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "model_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    report = {
        **report_body,
        "result_sha256": canonical_sha256(report_body),
    }
    return Lfm2CompiledMlpQdeim(
        selected_source_neurons=selected,
        gate_weight=materialized_gate.detach().cpu().contiguous().clone(),
        up_weight=materialized_up.detach().cpu().contiguous().clone(),
        down_weight_structural_prior=(
            materialized_down_prior.detach().cpu().contiguous().clone()
        ),
        down_weight_fit_corrected=(
            None
            if materialized_down_corrected is None
            else materialized_down_corrected.detach().cpu().contiguous().clone()
        ),
        functional_pod_basis=pod_basis.detach().cpu().contiguous().clone(),
        interpolation_reconstruction=(
            reconstruction.detach().cpu().contiguous().clone()
        ),
        normalized_input_decoder=decoder.detach().cpu().contiguous().clone(),
        residual_output_chart=chart.detach().cpu().contiguous().clone(),
        report=MappingProxyType(report),
    )


def assemble_lfm2_mlp_qdeim_span_sources(
    *,
    span: Lfm2ModelFoldSpan,
    sources_by_layer: Mapping[int, Lfm2MlpQdeimSpanSource],
) -> tuple[Lfm2MlpQdeimSpanSource, ...]:
    """Order semantic layer-stack payloads by an existing model-fold span.

    This is the intentionally narrow adapter seam: the semantic stack supplies
    already-cached per-layer payloads whose cache receipts it authenticated
    upstream, while the model fold remains the sole owner of depth planning.
    No model or cache reader is accepted here, so this typed but not yet
    production-wired helper cannot authenticate a receipt, recapture a parent,
    or invent a span.
    """

    if not isinstance(span, Lfm2ModelFoldSpan):
        raise Lfm2MlpQdeimError("span must be one Lfm2ModelFoldSpan.")
    if not isinstance(sources_by_layer, Mapping):
        raise Lfm2MlpQdeimError(
            "sources_by_layer must map every span layer to tensor/hash-bound evidence."
        )
    expected = set(span.source_layer_indices)
    if set(sources_by_layer) != expected:
        raise Lfm2MlpQdeimError(
            "sources_by_layer must exactly cover the existing model-fold span."
        )
    ordered = tuple(sources_by_layer[index] for index in span.source_layer_indices)
    if any(
        not isinstance(source, Lfm2MlpQdeimSpanSource)
        or source.source_layer_index != expected_index
        for expected_index, source in zip(
            span.source_layer_indices,
            ordered,
            strict=True,
        )
    ):
        raise Lfm2MlpQdeimError(
            "Every span source payload must bind its declared source layer."
        )
    return ordered


def _ordered_lfm2_mlp_qdeim_span_sources(
    span: Lfm2ModelFoldSpan,
    sources: Sequence[Lfm2MlpQdeimSpanSource],
) -> tuple[Lfm2MlpQdeimSpanSource, ...]:
    if not isinstance(span, Lfm2ModelFoldSpan):
        raise Lfm2MlpQdeimError("span must be one Lfm2ModelFoldSpan.")
    if isinstance(sources, (str, bytes)):
        raise Lfm2MlpQdeimError(
            "sources must contain one tensor/hash-bound payload per span layer."
        )
    try:
        ordered = tuple(sources)
    except TypeError as exc:
        raise Lfm2MlpQdeimError(
            "sources must contain one tensor/hash-bound payload per span layer."
        ) from exc
    if (
        not ordered
        or any(not isinstance(value, Lfm2MlpQdeimSpanSource) for value in ordered)
        or tuple(value.source_layer_index for value in ordered)
        != span.source_layer_indices
    ):
        raise Lfm2MlpQdeimError(
            "Span sources must exactly cover Lfm2ModelFoldSpan in execution order."
        )
    return ordered


def _span_qdeim_row_weights(
    row_count: int,
    supplied: Sequence[float] | None,
) -> tuple[float, ...]:
    if supplied is None:
        return tuple(1.0 for _ in range(row_count))
    if isinstance(supplied, (str, bytes)) or len(supplied) != row_count:
        raise Lfm2MlpQdeimError(
            "row_sample_weights must contain one positive value per aligned row."
        )
    result = []
    for index, value in enumerate(supplied):
        parsed = _finite_float(value, label=f"row_sample_weights[{index}]")
        if parsed <= 0.0:
            raise Lfm2MlpQdeimError(
                "row_sample_weights must contain only positive finite values."
            )
        result.append(parsed)
    return tuple(result)


def lfm2_mlp_qdeim_span_functional_evidence_sha256(
    *,
    span: Lfm2ModelFoldSpan,
    sources: Sequence[Lfm2MlpQdeimSpanSource],
    row_sample_weights: Sequence[float] | None = None,
) -> str:
    """Root the complete span source bundle and supplied receipt digests.

    Actual W1/W3/W2 state, cached rows, per-layer decoder and output chart,
    every paired expected digest, and the opaque receipt digest are bound into
    one externally supplied root.  This still does not read or authenticate
    the upstream receipt itself; production integration must do that first.
    """

    ordered = _ordered_lfm2_mlp_qdeim_span_sources(span, sources)
    normalized_rows = [
        _materialized_float_tensor(
            source.cached_parent_post_swiglu_rows,
            label=(
                f"source layer {source.source_layer_index} cached parent "
                "post-SwiGLU rows"
            ),
            ndim=2,
        )
        for source in ordered
    ]
    row_count = int(normalized_rows[0].shape[0])
    if any(int(rows.shape[0]) != row_count for rows in normalized_rows):
        raise Lfm2MlpQdeimError(
            "Every source layer in a span must use the same aligned functional rows."
        )
    weights = _span_qdeim_row_weights(row_count, row_sample_weights)
    return canonical_sha256(
        {
            "schema": LFM2_MLP_SPAN_QDEIM_FUNCTIONAL_EVIDENCE_SCHEMA,
            "span": span.as_dict(),
            "root_scope": (
                "complete_source_w1_w3_w2_rows_decoder_chart_expected_hashes_"
                "and_opaque_receipt_bindings"
            ),
            "aligned_row_count": row_count,
            "row_sample_weights": list(weights),
            "sources": [
                {
                    "source_layer_index": source.source_layer_index,
                    "source_state_sha256": lfm2_mlp_qdeim_source_state_sha256(
                        source.source_gate_weight,
                        source.source_up_weight,
                        source.source_down_weight,
                    ),
                    "expected_source_state_sha256": (
                        source.expected_source_state_sha256
                    ),
                    "cached_parent_post_swiglu_rows_sha256": _tensor_hash(
                        source.cached_parent_post_swiglu_rows,
                        label=(
                            f"source layer {source.source_layer_index} cached "
                            "post-SwiGLU rows"
                        ),
                    ),
                    "expected_cached_parent_post_swiglu_sha256": (
                        source.expected_cached_parent_post_swiglu_sha256
                    ),
                    "normalized_input_decoder_sha256": _tensor_hash(
                        source.normalized_input_decoder,
                        label=(
                            f"source layer {source.source_layer_index} "
                            "normalized-input decoder"
                        ),
                    ),
                    "expected_normalized_input_decoder_sha256": (
                        source.expected_normalized_input_decoder_sha256
                    ),
                    "residual_output_chart_sha256": _tensor_hash(
                        source.residual_output_chart,
                        label=(
                            f"source layer {source.source_layer_index} "
                            "residual output chart"
                        ),
                    ),
                    "expected_residual_output_chart_sha256": (
                        source.expected_residual_output_chart_sha256
                    ),
                    "cache_receipt_sha256": source.cache_receipt_sha256,
                }
                for source in ordered
            ],
        }
    )


def compile_lfm2_mlp_span_coupled_nonlinear_qdeim(
    *,
    span: Lfm2ModelFoldSpan,
    sources: Sequence[Lfm2MlpQdeimSpanSource],
    expected_functional_evidence_sha256: str,
    target_gate_weight_template: Any,
    target_up_weight_template: Any,
    target_down_weight_template: Any,
    expected_target_template_sha256: str,
    row_sample_weights: Sequence[float] | None = None,
    pod_rank_relative_tolerance: float | None = None,
    compute_device: str = "cpu",
) -> Lfm2CompiledSpanMlpQdeim:
    """Fold every layer in one depth span into one dense child SwiGLU.

    Functional POD and Q-DEIM operate once on the concatenated post-SwiGLU
    neuron coordinates of the complete span.  Selection is constrained only
    enough to retain at least one real coupled gate/up producer from every
    source layer.  Each producer uses its source layer's normalized-input
    decoder.  The child W2 is the sum of every source layer's complete writer,
    transported by its own output chart and its slice of the one joint Q-DEIM
    reconstruction.  No fitting, optimization, model execution, or gradient
    operation occurs.  Supplied cache-receipt digests are evidence bindings,
    not locally verified receipts; upstream authentication is required.
    """

    torch = _torch()
    ordered = _ordered_lfm2_mlp_qdeim_span_sources(span, sources)
    expected_functional = _digest(
        expected_functional_evidence_sha256,
        label="expected_functional_evidence_sha256",
    )
    expected_template = _digest(
        expected_target_template_sha256,
        label="expected_target_template_sha256",
    )
    if pod_rank_relative_tolerance is not None:
        pod_rank_relative_tolerance = _finite_float(
            pod_rank_relative_tolerance,
            label="pod_rank_relative_tolerance",
            nonnegative=True,
        )
    resolved_compute_device = _resolve_qdeim_compute_device(torch, compute_device)
    compute_execution = _qdeim_compute_receipt(
        torch,
        compute_device,
        resolved_compute_device,
    )
    template_sha256 = lfm2_mlp_qdeim_target_template_sha256(
        target_gate_weight_template,
        target_up_weight_template,
        target_down_weight_template,
    )
    if template_sha256 != expected_template:
        raise Lfm2MlpQdeimError("Target BF16 template metadata changed.")
    template_spec = _target_template_spec(
        target_gate_weight_template,
        target_up_weight_template,
        target_down_weight_template,
    )
    target_intermediate, target_hidden = template_spec["w1.weight"]["shape"]
    if template_spec["w3.weight"]["shape"] != [
        target_intermediate,
        target_hidden,
    ] or template_spec["w2.weight"]["shape"] != [
        target_hidden,
        target_intermediate,
    ]:
        raise Lfm2MlpQdeimError(
            "Target templates do not form one atomic dense LFM2 SwiGLU."
        )
    if target_intermediate < len(ordered):
        raise Lfm2MlpQdeimError(
            "A span-coupled child needs at least one intermediate neuron per "
            "source layer."
        )

    functional_evidence_sha256 = lfm2_mlp_qdeim_span_functional_evidence_sha256(
        span=span,
        sources=ordered,
        row_sample_weights=row_sample_weights,
    )
    if functional_evidence_sha256 != expected_functional:
        raise Lfm2MlpQdeimError(
            "Authenticated span post-SwiGLU functional evidence changed."
        )

    normalized_sources: list[dict[str, Any]] = []
    row_count: int | None = None
    total_source_neurons = 0
    source_state_before: dict[int, str] = {}
    source_groups: list[tuple[int, ...]] = []
    for source in ordered:
        layer_index = source.source_layer_index
        prefix = f"source layer {layer_index}"
        expected_source_state = _digest(
            source.expected_source_state_sha256,
            label=f"{prefix} expected_source_state_sha256",
        )
        current_source_state = lfm2_mlp_qdeim_source_state_sha256(
            source.source_gate_weight,
            source.source_up_weight,
            source.source_down_weight,
        )
        if current_source_state != expected_source_state:
            raise Lfm2MlpQdeimError(f"Authenticated {prefix} W1/W3/W2 state changed.")
        source_state_before[layer_index] = current_source_state
        gate = _materialized_float_tensor(
            source.source_gate_weight,
            label=f"{prefix} gate weight",
            ndim=2,
        )
        up = _materialized_float_tensor(
            source.source_up_weight,
            label=f"{prefix} up weight",
            ndim=2,
        )
        down = _materialized_float_tensor(
            source.source_down_weight,
            label=f"{prefix} down weight",
            ndim=2,
        )
        source_intermediate, source_hidden = map(int, gate.shape)
        if tuple(up.shape) != tuple(gate.shape) or tuple(down.shape) != (
            source_hidden,
            source_intermediate,
        ):
            raise Lfm2MlpQdeimError(
                f"Authenticated {prefix} tensors do not form one coupled SwiGLU."
            )
        if target_hidden > source_hidden:
            raise Lfm2MlpQdeimError(
                "A span-coupled Q-DEIM source cannot expand target hidden width."
            )
        rows = _materialized_float_tensor(
            source.cached_parent_post_swiglu_rows,
            label=f"{prefix} cached parent post-SwiGLU rows",
            ndim=2,
        )
        if int(rows.shape[1]) != source_intermediate:
            raise Lfm2MlpQdeimError(
                f"{prefix} functional rows do not match its intermediate width."
            )
        if row_count is None:
            row_count = int(rows.shape[0])
        elif int(rows.shape[0]) != row_count:
            raise Lfm2MlpQdeimError(
                "Every source layer in a span must use aligned functional rows."
            )
        rows_sha256 = _tensor_hash(
            source.cached_parent_post_swiglu_rows,
            label=f"{prefix} cached parent post-SwiGLU rows",
        )
        if rows_sha256 != source.expected_cached_parent_post_swiglu_sha256:
            raise Lfm2MlpQdeimError(
                f"Authenticated {prefix} cached functional rows changed."
            )
        decoder_sha256 = _tensor_hash(
            source.normalized_input_decoder,
            label=f"{prefix} normalized-input decoder",
        )
        if decoder_sha256 != source.expected_normalized_input_decoder_sha256:
            raise Lfm2MlpQdeimError(
                f"Authenticated {prefix} normalized-input decoder changed."
            )
        chart_sha256 = _tensor_hash(
            source.residual_output_chart,
            label=f"{prefix} residual output chart",
        )
        if chart_sha256 != source.expected_residual_output_chart_sha256:
            raise Lfm2MlpQdeimError(
                f"Authenticated {prefix} residual output chart changed."
            )
        decoder = _materialized_float_tensor(
            source.normalized_input_decoder,
            label=f"{prefix} normalized-input decoder",
            ndim=2,
        )
        chart = _materialized_float_tensor(
            source.residual_output_chart,
            label=f"{prefix} residual output chart",
            ndim=2,
        )
        if tuple(decoder.shape) != (target_hidden, source_hidden):
            raise Lfm2MlpQdeimError(
                f"{prefix} normalized-input decoder must have shape "
                "[target_hidden, source_hidden]."
            )
        if tuple(chart.shape) != (source_hidden, target_hidden):
            raise Lfm2MlpQdeimError(
                f"{prefix} residual output chart must have shape "
                "[source_hidden, target_hidden]."
            )
        offset = total_source_neurons
        total_source_neurons += source_intermediate
        group = tuple(range(offset, total_source_neurons))
        source_groups.append(group)
        normalized_sources.append(
            {
                "source": source,
                "gate": gate,
                "up": up,
                "down": down,
                "rows": rows,
                "decoder": decoder,
                "chart": chart,
                "decoder_sha256": decoder_sha256,
                "chart_sha256": chart_sha256,
                "rows_sha256": rows_sha256,
                "source_intermediate": source_intermediate,
                "source_hidden": source_hidden,
                "union_start": offset,
                "union_stop": total_source_neurons,
            }
        )
    assert row_count is not None
    if target_intermediate >= total_source_neurons:
        raise Lfm2MlpQdeimError(
            "Span-coupled Q-DEIM must reduce the union intermediate width."
        )
    row_weights = _span_qdeim_row_weights(row_count, row_sample_weights)

    with torch.no_grad():
        staged_sources: list[dict[str, Any]] = []
        for source_value in normalized_sources:
            decoder_work = source_value["decoder"].to(
                device=resolved_compute_device,
                dtype=torch.float64,
            )
            chart_work = source_value["chart"].to(
                device=resolved_compute_device,
                dtype=torch.float64,
            )
            decoder_singular = torch.linalg.svdvals(decoder_work)
            chart_singular = torch.linalg.svdvals(chart_work)
            decoder_tolerance = (
                max(decoder_work.shape)
                * torch.finfo(torch.float64).eps
                * decoder_singular.max()
            )
            chart_tolerance = (
                max(chart_work.shape)
                * torch.finfo(torch.float64).eps
                * chart_singular.max()
            )
            decoder_rank = int((decoder_singular > decoder_tolerance).sum().item())
            chart_rank = int((chart_singular > chart_tolerance).sum().item())
            if decoder_rank != target_hidden or chart_rank != target_hidden:
                raise Lfm2MlpQdeimError(
                    "Every span decoder and output chart must be structurally "
                    "full target rank."
                )
            staged_sources.append(
                {
                    **source_value,
                    "gate_work": source_value["gate"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "up_work": source_value["up"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "down_work": source_value["down"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "rows_work": source_value["rows"].to(
                        device=resolved_compute_device,
                        dtype=torch.float64,
                    ),
                    "decoder_work": decoder_work,
                    "chart_work": chart_work,
                    "decoder_rank": decoder_rank,
                    "chart_rank": chart_rank,
                    "decoder_condition_number": float(
                        decoder_singular.max().item() / decoder_singular.min().item()
                    ),
                    "chart_condition_number": float(
                        chart_singular.max().item() / chart_singular.min().item()
                    ),
                }
            )

        union_features = torch.cat(
            [source_value["rows_work"] for source_value in staged_sources],
            dim=1,
        ).contiguous()
        weight_tensor = torch.tensor(
            row_weights,
            dtype=torch.float64,
            device=resolved_compute_device,
        )
        sqrt_weights = torch.sqrt(weight_tensor).unsqueeze(1)
        weighted_features = (union_features * sqrt_weights).contiguous()
        _pod_left, pod_singular, pod_right_h = torch.linalg.svd(
            weighted_features,
            full_matrices=False,
        )
        singular_maximum = float(pod_singular.max().item())
        default_rank_tolerance = (
            max(weighted_features.shape)
            * torch.finfo(torch.float64).eps
            * singular_maximum
        )
        rank_tolerance = (
            default_rank_tolerance
            if pod_rank_relative_tolerance is None
            else pod_rank_relative_tolerance * singular_maximum
        )
        observed_rank = int((pod_singular > rank_tolerance).sum().item())
        if observed_rank < target_intermediate:
            body = {
                "schema": LFM2_MLP_QDEIM_RANK_DEFICIT_SCHEMA,
                "status": "insufficient-hash-bound-span-snapshot-rank",
                "span": span.as_dict(),
                "minimum_snapshot_rank": target_intermediate,
                "observed_snapshot_rank": observed_rank,
                "snapshot_rank_deficit": target_intermediate - observed_rank,
                "source_layer_count": len(ordered),
                "union_source_intermediate_size": total_source_neurons,
                "functional_evidence_sha256": functional_evidence_sha256,
                "random_mode_completion_permitted": False,
                "silent_pseudoinverse_completion_permitted": False,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
            }
            raise Lfm2MlpQdeimInsufficientRankError(
                {**body, "result_sha256": canonical_sha256(body)}
            )

        pod_basis = _canonicalize_pod_signs(
            pod_right_h[:target_intermediate].T.contiguous()
        )
        selected_global, pivot_residual_norms = (
            _deterministic_source_covering_pivoted_qr_indices(
                pod_basis,
                source_groups,
            )
        )
        sampled_basis = pod_basis[list(selected_global), :].contiguous()
        interpolation_singular = torch.linalg.svdvals(sampled_basis)
        interpolation_minimum = float(interpolation_singular.min().item())
        interpolation_maximum = float(interpolation_singular.max().item())
        interpolation_tolerance = (
            max(sampled_basis.shape)
            * torch.finfo(torch.float64).eps
            * interpolation_maximum
        )
        if (
            not math.isfinite(interpolation_minimum)
            or interpolation_minimum <= interpolation_tolerance
        ):
            raise Lfm2MlpQdeimError(
                "Span-coupled Q-DEIM sampled POD basis is structurally singular."
            )
        reconstruction = torch.linalg.solve(
            sampled_basis.T,
            pod_basis.T,
        ).contiguous()
        reconstructed_features = (
            union_features[:, list(selected_global)] @ reconstruction
        )
        interpolation_relative_rmse = _relative_rmse(
            reconstructed_features * sqrt_weights,
            weighted_features,
        )
        selected_identity_error = float(
            (
                reconstruction[:, list(selected_global)]
                - torch.eye(
                    target_intermediate,
                    dtype=torch.float64,
                    device=resolved_compute_device,
                )
            )
            .abs()
            .max()
            .item()
        )

        selected_pairs: list[tuple[int, int]] = []
        selected_gate_rows = []
        selected_up_rows = []
        selected_counts: dict[int, int] = {
            source.source_layer_index: 0 for source in ordered
        }
        for global_index in selected_global:
            selected_source = next(
                value
                for value in staged_sources
                if value["union_start"] <= global_index < value["union_stop"]
            )
            layer_index = selected_source["source"].source_layer_index
            local_index = global_index - selected_source["union_start"]
            selected_pairs.append((layer_index, local_index))
            selected_counts[layer_index] += 1
            selected_gate_rows.append(
                selected_source["gate_work"][local_index]
                @ selected_source["decoder_work"].T
            )
            selected_up_rows.append(
                selected_source["up_work"][local_index]
                @ selected_source["decoder_work"].T
            )
        gate_fp64 = torch.stack(selected_gate_rows, dim=0).contiguous()
        up_fp64 = torch.stack(selected_up_rows, dim=0).contiguous()

        writer_contributions = []
        writer_reports = []
        for source_value in staged_sources:
            start = source_value["union_start"]
            stop = source_value["union_stop"]
            source_reconstruction = reconstruction[:, start:stop].contiguous()
            mapped_writer = (
                source_value["chart_work"].T @ source_value["down_work"]
            ).contiguous()
            contribution = (mapped_writer @ source_reconstruction.T).contiguous()
            writer_contributions.append(contribution)
            layer_index = source_value["source"].source_layer_index
            writer_reports.append(
                {
                    "source_layer_index": layer_index,
                    "source_writer_column_count": source_value["source_intermediate"],
                    "source_w2_sha256": _tensor_hash(
                        source_value["source"].source_down_weight,
                        label=f"source layer {layer_index} W2",
                    ),
                    "residual_output_chart_sha256": source_value["chart_sha256"],
                    "reconstruction_slice_sha256": _tensor_hash(
                        source_reconstruction,
                        label=f"source layer {layer_index} reconstruction slice",
                    ),
                    "mapped_writer_sha256": _tensor_hash(
                        mapped_writer,
                        label=f"source layer {layer_index} mapped writer",
                    ),
                    "writer_contribution_sha256": _tensor_hash(
                        contribution,
                        label=f"source layer {layer_index} writer contribution",
                    ),
                    "writer_contribution_frobenius_norm": float(
                        torch.linalg.vector_norm(contribution).item()
                    ),
                    "complete_source_w2_consumed": True,
                }
            )
        down_fp64 = torch.stack(writer_contributions, dim=0).sum(dim=0).contiguous()
        materialized_gate = gate_fp64.to(dtype=torch.bfloat16).contiguous()
        materialized_up = up_fp64.to(dtype=torch.bfloat16).contiguous()
        materialized_down = down_fp64.to(dtype=torch.bfloat16).contiguous()

    source_state_after = {
        source.source_layer_index: lfm2_mlp_qdeim_source_state_sha256(
            source.source_gate_weight,
            source.source_up_weight,
            source.source_down_weight,
        )
        for source in ordered
    }
    rows_after = {
        source.source_layer_index: _tensor_hash(
            source.cached_parent_post_swiglu_rows,
            label=(
                f"source layer {source.source_layer_index} cached parent "
                "post-SwiGLU rows"
            ),
        )
        for source in ordered
    }
    candidate_tensors = (
        materialized_gate,
        materialized_up,
        materialized_down,
        pod_basis,
        reconstruction,
    )
    if (
        source_state_after != source_state_before
        or any(
            rows_after[source.source_layer_index]
            != source.expected_cached_parent_post_swiglu_sha256
            for source in ordered
        )
        or any(count <= 0 for count in selected_counts.values())
        or len(selected_pairs) != target_intermediate
        or len(set(selected_pairs)) != target_intermediate
        or tuple(materialized_gate.shape) != (target_intermediate, target_hidden)
        or tuple(materialized_up.shape) != tuple(materialized_gate.shape)
        or tuple(materialized_down.shape) != (target_hidden, target_intermediate)
        or any(tensor.dtype != torch.bfloat16 for tensor in candidate_tensors[:3])
        or any(
            tensor.layout != torch.strided
            or tensor.device.type == "meta"
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all().item())
            for tensor in candidate_tensors
        )
    ):
        raise Lfm2MlpQdeimError(
            "Materialized span Q-DEIM failed dense, coverage, immutability, "
            "or zero-gradient invariants."
        )

    candidate_state_sha256 = _candidate_state_sha256(
        materialized_gate,
        materialized_up,
        materialized_down,
    )
    reconstruction_sha256 = _tensor_hash(
        reconstruction,
        label="span Q-DEIM reconstruction",
    )
    source_reports = []
    for source_value in staged_sources:
        layer_index = source_value["source"].source_layer_index
        source_reports.append(
            {
                "source_layer_index": layer_index,
                "source_state_sha256_before": source_state_before[layer_index],
                "source_state_sha256_after": source_state_after[layer_index],
                "source_state_immutable": True,
                "source_tensor_sha256": {
                    "w1.weight": _tensor_hash(
                        source_value["source"].source_gate_weight,
                        label=f"source layer {layer_index} W1",
                    ),
                    "w3.weight": _tensor_hash(
                        source_value["source"].source_up_weight,
                        label=f"source layer {layer_index} W3",
                    ),
                    "w2.weight": _tensor_hash(
                        source_value["source"].source_down_weight,
                        label=f"source layer {layer_index} W2",
                    ),
                },
                "cache_receipt_sha256": source_value["source"].cache_receipt_sha256,
                "cached_parent_post_swiglu_rows_sha256": source_value["rows_sha256"],
                "source_hidden_size": source_value["source_hidden"],
                "source_intermediate_size": source_value["source_intermediate"],
                "union_coordinate_range": [
                    source_value["union_start"],
                    source_value["union_stop"],
                ],
                "selected_real_neuron_count": selected_counts[layer_index],
                "source_layer_retained_by_real_producer": (
                    selected_counts[layer_index] > 0
                ),
                "normalized_input_decoder_sha256": source_value["decoder_sha256"],
                "normalized_input_decoder_rank": source_value.get(
                    "decoder_rank",
                    target_hidden,
                ),
                "normalized_input_decoder_condition_number": source_value[
                    "decoder_condition_number"
                ],
                "residual_output_chart_sha256": source_value["chart_sha256"],
                "residual_output_chart_rank": source_value.get(
                    "chart_rank",
                    target_hidden,
                ),
                "residual_output_chart_condition_number": source_value[
                    "chart_condition_number"
                ],
            }
        )
    report_body = {
        "schema": LFM2_MLP_SPAN_QDEIM_SCHEMA,
        "status": "dense-span-coupled-candidate-ready-for-decoded-review",
        "span": span.as_dict(),
        "target_layer_index": span.target_layer_index,
        "source_layer_indices": list(span.source_layer_indices),
        "source_layer_count": len(ordered),
        "all_source_layers_consumed": True,
        "all_source_w1_w3_w2_tensors_match_supplied_expected_sha256": True,
        "all_source_layers_retain_real_selected_producer": True,
        "selection_scope": "joint_union_of_all_parent_layer_neurons",
        "selection_constraint": (
            "complete_stability_aware_independent_transversal_seed_then_global_qrcp"
        ),
        "source_covering_seed": {
            "method": (
                "deterministic_complete_backtracking_over_one_real_neuron_per_"
                "source_with_final_sampled_basis_svd_admissibility"
            ),
            "selected_union_indices": list(selected_global[: len(ordered)]),
            "selected_count": len(ordered),
            "guarantee": (
                "enumerates_independent_covering_seeds_until_deterministic_qrcp_"
                "completion_passes_the_exact_compiler_svd_stability_predicate"
            ),
            "final_stability_predicate": (
                "sigma_min(sampled_basis) > max(sampled_basis.shape) * "
                "float64_eps * sigma_max(sampled_basis)"
            ),
            "weak_above_epsilon_seed_can_terminate_search": False,
            "group_order": "ascending_group_size_then_source_order",
            "candidate_order": "descending_residual_norm_then_union_index",
            "remaining_modes": (
                "ordinary_deterministic_global_qrcp_then_exact_svd_admissibility"
            ),
            "worst_case_search_complexity": "exponential_in_source_layer_count",
        },
        "selected_source_neurons": [
            {
                "source_layer_index": layer_index,
                "source_neuron_index": neuron_index,
            }
            for layer_index, neuron_index in selected_pairs
        ],
        "selected_source_neuron_count": len(selected_pairs),
        "selected_source_neuron_counts_by_layer": {
            str(layer_index): selected_counts[layer_index]
            for layer_index in span.source_layer_indices
        },
        "source_layers": source_reports,
        "target_template_sha256": template_sha256,
        "target_template_values_read": False,
        "functional_evidence_sha256": functional_evidence_sha256,
        "functional_evidence_schema": (LFM2_MLP_SPAN_QDEIM_FUNCTIONAL_EVIDENCE_SCHEMA),
        "complete_source_bundle_rooted_by_external_expected_evidence": True,
        "cache_receipt_binding": {
            "supplied_receipt_digests_bound_into_functional_evidence": True,
            "upstream_receipt_authentication_required": True,
            "upstream_receipt_authentication_performed_by_this_compiler": False,
        },
        "aligned_functional_row_count": row_count,
        "row_sample_weights_sha256": _tensor_hash(
            torch.tensor(row_weights, dtype=torch.float64),
            label="span row sample weights",
        ),
        "union_source_intermediate_size": total_source_neurons,
        "target_hidden_size": target_hidden,
        "target_intermediate_size": target_intermediate,
        "observed_snapshot_rank": observed_rank,
        "minimum_snapshot_rank": target_intermediate,
        "pod_basis_sha256": _tensor_hash(
            pod_basis,
            label="span functional POD basis",
        ),
        "pod_singular_values_sha256": _tensor_hash(
            pod_singular,
            label="span functional POD singular values",
        ),
        "pivot_residual_norms_sha256": _tensor_hash(
            pivot_residual_norms,
            label="span Q-DEIM pivot residual norms",
        ),
        "functional_interpolation": {
            "formula": (
                "concat(parent_layer_post_swiglu) ~= selected_union_features @ R"
            ),
            "reconstruction_shape": [target_intermediate, total_source_neurons],
            "reconstruction_sha256": reconstruction_sha256,
            "weighted_relative_rmse_diagnostic": interpolation_relative_rmse,
            "selected_identity_maximum_absolute_error": selected_identity_error,
            "condition_number": interpolation_maximum / interpolation_minimum,
            "numeric_diagnostics_have_semantic_authority": False,
        },
        "producer_transport": {
            "method": ("selected_real_source_gate_up_rows_through_their_own_decoders"),
            "w1_formula": "W1_layer[selected_local, :] @ decoder_layer.T",
            "w3_formula": "W3_layer[selected_local, :] @ decoder_layer.T",
            "producer_averaging": False,
            "cross_layer_weight_averaging": False,
        },
        "writer_transport": {
            "formula": ("W2_child = sum_layer(chart_layer.T @ W2_layer @ R_layer.T)"),
            "joint_reconstruction_sha256": reconstruction_sha256,
            "source_contributions": writer_reports,
            "all_source_w2_tensors_consumed": True,
            "all_source_writer_columns_consumed": True,
        },
        "materialized_candidate_state_sha256": candidate_state_sha256,
        "dense_ordinary_child_tensors": True,
        "dense_bfloat16_target_state": True,
        "parameter_sharing_used": False,
        "source_parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "semantic_layer_stack_adapter_seam": {
            "span_owner": "Lfm2ModelFoldSpan",
            "ordered_source_payload": "Lfm2MlpQdeimSpanSource",
            "assembly_helper": "assemble_lfm2_mlp_qdeim_span_sources",
            "typed": True,
            "production_wired": False,
            "planner_duplicated": False,
            "parent_recapture_required": False,
            "upstream_cache_receipt_authentication_required": True,
        },
        "compute_execution": compute_execution,
        "automatic_candidate_acceptance_performed": False,
        "decoded_whole_response_review_required": True,
        "numeric_diagnostics_are_advisory_only": True,
        "model_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    report = {**report_body, "result_sha256": canonical_sha256(report_body)}
    return Lfm2CompiledSpanMlpQdeim(
        span=span,
        selected_source_neurons=tuple(selected_pairs),
        gate_weight=materialized_gate.detach().cpu().contiguous().clone(),
        up_weight=materialized_up.detach().cpu().contiguous().clone(),
        down_weight=materialized_down.detach().cpu().contiguous().clone(),
        functional_pod_basis=pod_basis.detach().cpu().contiguous().clone(),
        interpolation_reconstruction=(
            reconstruction.detach().cpu().contiguous().clone()
        ),
        report=MappingProxyType(report),
    )


__all__ = [
    "LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE",
    "LFM2_MLP_QDEIM_FUNCTIONAL_SNAPSHOT_EVIDENCE_SCHEMA",
    "LFM2_MLP_QDEIM_RANK_DEFICIT_SCHEMA",
    "LFM2_MLP_QDEIM_SCHEMA",
    "LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE",
    "LFM2_MLP_SPAN_QDEIM_FUNCTIONAL_EVIDENCE_SCHEMA",
    "LFM2_MLP_SPAN_QDEIM_SCHEMA",
    "Lfm2CompiledMlpQdeim",
    "Lfm2CompiledSpanMlpQdeim",
    "Lfm2MlpQdeimError",
    "Lfm2MlpQdeimFitCase",
    "Lfm2MlpQdeimFunctionalSnapshotBlock",
    "Lfm2MlpQdeimInsufficientRankError",
    "Lfm2MlpQdeimSpanSource",
    "assemble_lfm2_mlp_qdeim_span_sources",
    "compile_lfm2_mlp_nonlinear_qdeim",
    "compile_lfm2_mlp_span_coupled_nonlinear_qdeim",
    "lfm2_mlp_qdeim_fit_evidence_sha256",
    "lfm2_mlp_qdeim_functional_snapshot_evidence_sha256",
    "lfm2_mlp_qdeim_source_state_sha256",
    "lfm2_mlp_qdeim_span_functional_evidence_sha256",
    "lfm2_mlp_qdeim_target_template_sha256",
]
