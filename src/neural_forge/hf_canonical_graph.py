"""Canonical simultaneous multi-parent execution graphs.

The graph is architecture neutral at its parent boundary.  Every parent
contributes row-aligned hidden trajectories and child-vocabulary logits under
an explicit token contract.  Parent order is canonicalized, all source depth
spans are represented, boundary zero is retained, and target probabilities are
fused simultaneously rather than applying donors sequentially.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .hf_canonical_kernel import (
    CanonicalBoundarySplits,
    CanonicalKernelError,
    CanonicalParentTrace,
    build_canonical_boundary_splits,
)


_SCHEMA = "neural-forge-canonical-execution-graph-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CanonicalGraphError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CanonicalChildSpec:
    family: str
    layer_count: int
    hidden_size: int
    parameter_budget: int
    vocabulary_size: int
    tokenizer_fingerprint: str
    tie_embeddings: bool
    position_policy: str
    architecture_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _text(self.family, "family"))
        object.__setattr__(
            self,
            "layer_count",
            _positive_int(self.layer_count, "layer_count"),
        )
        object.__setattr__(
            self,
            "hidden_size",
            _positive_int(self.hidden_size, "hidden_size"),
        )
        object.__setattr__(
            self,
            "parameter_budget",
            _positive_int(self.parameter_budget, "parameter_budget"),
        )
        object.__setattr__(
            self,
            "vocabulary_size",
            _positive_int(self.vocabulary_size, "vocabulary_size"),
        )
        object.__setattr__(
            self,
            "tokenizer_fingerprint",
            _sha256(self.tokenizer_fingerprint, "tokenizer_fingerprint"),
        )
        if not isinstance(self.tie_embeddings, bool):
            raise CanonicalGraphError("tie_embeddings must be boolean.")
        object.__setattr__(
            self,
            "position_policy",
            _text(self.position_policy, "position_policy"),
        )
        object.__setattr__(
            self,
            "architecture_fingerprint",
            _sha256(self.architecture_fingerprint, "architecture_fingerprint"),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.report())

    def report(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "parameter_budget": self.parameter_budget,
            "vocabulary_size": self.vocabulary_size,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "tie_embeddings": self.tie_embeddings,
            "position_policy": self.position_policy,
            "architecture_fingerprint": self.architecture_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class CanonicalTokenContract:
    parent_name: str
    parent_tokenizer_fingerprint: str
    child_tokenizer_fingerprint: str
    method: Literal["identity", "strict-row-map", "byte-span-likelihood"]
    mapping_fingerprint: str
    coverage_fraction: float = 1.0
    exact: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_name",
            _text(self.parent_name, "parent_name"),
        )
        object.__setattr__(
            self,
            "parent_tokenizer_fingerprint",
            _sha256(
                self.parent_tokenizer_fingerprint,
                "parent_tokenizer_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "child_tokenizer_fingerprint",
            _sha256(
                self.child_tokenizer_fingerprint,
                "child_tokenizer_fingerprint",
            ),
        )
        if self.method not in {"identity", "strict-row-map", "byte-span-likelihood"}:
            raise CanonicalGraphError("Unsupported canonical token-contract method.")
        object.__setattr__(
            self,
            "mapping_fingerprint",
            _sha256(self.mapping_fingerprint, "mapping_fingerprint"),
        )
        coverage = _finite_float(self.coverage_fraction, "coverage_fraction")
        if not 0.0 <= coverage <= 1.0:
            raise CanonicalGraphError("coverage_fraction must be in [0, 1].")
        object.__setattr__(self, "coverage_fraction", coverage)
        if not isinstance(self.exact, bool):
            raise CanonicalGraphError("exact must be boolean.")

    def report(self) -> dict[str, Any]:
        return {
            "parent_name": self.parent_name,
            "parent_tokenizer_fingerprint": self.parent_tokenizer_fingerprint,
            "child_tokenizer_fingerprint": self.child_tokenizer_fingerprint,
            "method": self.method,
            "mapping_fingerprint": self.mapping_fingerprint,
            "coverage_fraction": self.coverage_fraction,
            "exact": self.exact,
        }


@dataclass(frozen=True, slots=True)
class CanonicalParentLogits:
    parent_name: str
    calibration_logits: Any
    validation_logits: Any
    token_contract: CanonicalTokenContract


@dataclass(frozen=True, slots=True)
class CanonicalGraphEdge:
    target_layer_index: int
    parent_source_spans: tuple[tuple[str, tuple[int, ...]], ...]
    normalized_parent_weights: tuple[tuple[str, float], ...]
    calibration_input_sha256: str
    calibration_output_sha256: str
    validation_input_sha256: str
    validation_output_sha256: str
    retained_kernel_fraction: float
    effective_rank: int

    def report(self) -> dict[str, Any]:
        return {
            "target_layer_index": self.target_layer_index,
            "parent_source_spans": {
                name: list(span) for name, span in self.parent_source_spans
            },
            "normalized_parent_weights": {
                name: weight for name, weight in self.normalized_parent_weights
            },
            "calibration_input_sha256": self.calibration_input_sha256,
            "calibration_output_sha256": self.calibration_output_sha256,
            "validation_input_sha256": self.validation_input_sha256,
            "validation_output_sha256": self.validation_output_sha256,
            "retained_kernel_fraction": self.retained_kernel_fraction,
            "effective_rank": self.effective_rank,
            "all_source_spans_represented": True,
            "simultaneous_parent_contributions": True,
        }


@dataclass(frozen=True, slots=True)
class CanonicalExecutionGraph:
    child_spec: CanonicalChildSpec
    boundary_splits: CanonicalBoundarySplits
    calibration_target_log_probs: Any
    validation_target_log_probs: Any
    token_contracts: tuple[CanonicalTokenContract, ...]
    edges: tuple[CanonicalGraphEdge, ...]
    parent_logit_diagnostics: tuple[Mapping[str, Any], ...]
    fingerprint: str

    @property
    def calibration_boundaries(self) -> tuple[Any, ...]:
        return self.boundary_splits.calibration_rows

    @property
    def validation_boundaries(self) -> tuple[Any, ...]:
        return self.boundary_splits.validation_rows

    def bootstrap_inputs(self) -> dict[str, Any]:
        return {
            "canonical_calibration_boundaries": self.calibration_boundaries,
            "canonical_validation_boundaries": self.validation_boundaries,
            "canonical_calibration_target_log_probs": (
                self.calibration_target_log_probs
            ),
            "canonical_validation_target_log_probs": self.validation_target_log_probs,
            "canonical_execution_graph_fingerprint": self.fingerprint,
        }

    def report(self) -> dict[str, Any]:
        plan = self.boundary_splits.calibration_plan
        return {
            "schema": _SCHEMA,
            "method": "simultaneous-hidden-kernel-and-probability-mixture",
            "child_spec": self.child_spec.report(),
            "child_spec_fingerprint": self.child_spec.fingerprint,
            "parent_names": list(plan.parent_names),
            "normalized_parent_weights": list(plan.normalized_parent_weights),
            "token_contracts": [contract.report() for contract in self.token_contracts],
            "boundary_splits": self.boundary_splits.report(),
            "boundary_zero_consumed": True,
            "edges": [edge.report() for edge in self.edges],
            "calibration_target_log_probs_sha256": _tensor_sha256(
                self.calibration_target_log_probs
            ),
            "validation_target_log_probs_sha256": _tensor_sha256(
                self.validation_target_log_probs
            ),
            "parent_logit_diagnostics": [
                dict(value) for value in self.parent_logit_diagnostics
            ],
            "parent_application": "simultaneous-not-sequential",
            "single_authoritative_parent": False,
            "training_steps": 0,
            "gradients_used": False,
            "exactness_class": "finite-trace-approximate",
            "exactness_scope": (
                "fingerprinted calibration/validation trajectories and aligned "
                "child-vocabulary probability mixtures"
            ),
            "algebraically_exact": False,
            "fingerprint": self.fingerprint,
        }


def build_canonical_execution_graph(
    calibration_parents: Sequence[CanonicalParentTrace],
    validation_parents: Sequence[CanonicalParentTrace],
    parent_logits: Sequence[CanonicalParentLogits],
    *,
    child_spec: CanonicalChildSpec,
    ridge: float = 1e-6,
    center_rows: bool = True,
) -> CanonicalExecutionGraph:
    """Compile heterogeneous parents into one order-invariant child graph."""

    torch = _require_torch()
    if not isinstance(child_spec, CanonicalChildSpec):
        raise CanonicalGraphError("child_spec must be a CanonicalChildSpec.")
    try:
        splits = build_canonical_boundary_splits(
            calibration_parents,
            validation_parents,
            target_layer_count=child_spec.layer_count,
            target_hidden_size=child_spec.hidden_size,
            ridge=ridge,
            center_rows=center_rows,
        )
    except CanonicalKernelError as exc:
        raise CanonicalGraphError(str(exc)) from exc
    plan = splits.calibration_plan
    calibration_by_name = {parent.name: parent for parent in calibration_parents}
    validation_by_name = {parent.name: parent for parent in validation_parents}
    if set(calibration_by_name) != set(plan.parent_names) or set(validation_by_name) != set(
        plan.parent_names
    ):
        raise CanonicalGraphError("Canonical parent names are duplicated or inconsistent.")

    logits_by_name: dict[str, CanonicalParentLogits] = {}
    for index, value in enumerate(parent_logits):
        if not isinstance(value, CanonicalParentLogits):
            raise CanonicalGraphError(
                f"parent_logits[{index}] must be a CanonicalParentLogits."
            )
        name = _text(value.parent_name, f"parent_logits[{index}].parent_name")
        if name in logits_by_name:
            raise CanonicalGraphError("Canonical parent logits names must be unique.")
        logits_by_name[name] = value
    if set(logits_by_name) != set(plan.parent_names):
        raise CanonicalGraphError(
            "Every canonical parent must provide exactly one aligned logit pair."
        )

    calibration_rows = plan.row_count
    validation_rows = int(splits.validation_rows[0].shape[0])
    ordered_logits = []
    ordered_contracts = []
    for name in plan.parent_names:
        value = logits_by_name[name]
        contract = value.token_contract
        if not isinstance(contract, CanonicalTokenContract):
            raise CanonicalGraphError(f"Parent {name!r} has no valid token contract.")
        if contract.parent_name != name:
            raise CanonicalGraphError("Token-contract parent name mismatch.")
        if contract.child_tokenizer_fingerprint != child_spec.tokenizer_fingerprint:
            raise CanonicalGraphError("Token contract targets a different child tokenizer.")
        parent_tokenizer = calibration_by_name[name].row_identity.tokenizer_fingerprint
        if (
            contract.parent_tokenizer_fingerprint != parent_tokenizer
            or validation_by_name[name].row_identity.tokenizer_fingerprint
            != parent_tokenizer
        ):
            raise CanonicalGraphError("Parent tokenizer provenance is inconsistent.")
        if not contract.exact or contract.coverage_fraction != 1.0:
            raise CanonicalGraphError(
                "Canonical graph logits require exact full child-token coverage."
            )
        calibration = _logit_matrix(
            torch,
            value.calibration_logits,
            rows=calibration_rows,
            vocabulary=child_spec.vocabulary_size,
            name=f"{name}.calibration_logits",
        )
        validation = _logit_matrix(
            torch,
            value.validation_logits,
            rows=validation_rows,
            vocabulary=child_spec.vocabulary_size,
            name=f"{name}.validation_logits",
        )
        ordered_logits.append((calibration, validation))
        ordered_contracts.append(contract)

    calibration_target, calibration_diagnostics = _fuse_log_probs(
        torch,
        [value[0] for value in ordered_logits],
        plan.parent_names,
        plan.normalized_parent_weights,
    )
    validation_target, validation_diagnostics = _fuse_log_probs(
        torch,
        [value[1] for value in ordered_logits],
        plan.parent_names,
        plan.normalized_parent_weights,
    )
    diagnostics = tuple(
        {
            "parent_name": name,
            "calibration_mean_kl_to_mixture": calibration_diagnostics[index],
            "validation_mean_kl_to_mixture": validation_diagnostics[index],
        }
        for index, name in enumerate(plan.parent_names)
    )

    named_weights = tuple(zip(plan.parent_names, plan.normalized_parent_weights, strict=True))
    edges = []
    for target_index in range(child_spec.layer_count):
        source_spans = tuple(
            (name, cover.source_spans[target_index])
            for name, cover in zip(
                plan.parent_names,
                plan.source_depth_covers,
                strict=True,
            )
        )
        output_layer = plan.layers[target_index + 1]
        edges.append(
            CanonicalGraphEdge(
                target_layer_index=target_index,
                parent_source_spans=source_spans,
                normalized_parent_weights=named_weights,
                calibration_input_sha256=_tensor_sha256(
                    splits.calibration_rows[target_index]
                ),
                calibration_output_sha256=_tensor_sha256(
                    splits.calibration_rows[target_index + 1]
                ),
                validation_input_sha256=_tensor_sha256(
                    splits.validation_rows[target_index]
                ),
                validation_output_sha256=_tensor_sha256(
                    splits.validation_rows[target_index + 1]
                ),
                retained_kernel_fraction=output_layer.retained_kernel_fraction,
                effective_rank=output_layer.effective_rank,
            )
        )

    fingerprint_payload = {
        "schema": _SCHEMA,
        "child_spec_fingerprint": child_spec.fingerprint,
        "boundary_splits_fingerprint": splits.fingerprint,
        "token_contracts": [contract.report() for contract in ordered_contracts],
        "edges": [edge.report() for edge in edges],
        "calibration_target_log_probs_sha256": _tensor_sha256(calibration_target),
        "validation_target_log_probs_sha256": _tensor_sha256(validation_target),
        "parent_logit_diagnostics": list(diagnostics),
    }
    fingerprint = _fingerprint(fingerprint_payload)
    return CanonicalExecutionGraph(
        child_spec=child_spec,
        boundary_splits=splits,
        calibration_target_log_probs=calibration_target,
        validation_target_log_probs=validation_target,
        token_contracts=tuple(ordered_contracts),
        edges=tuple(edges),
        parent_logit_diagnostics=diagnostics,
        fingerprint=fingerprint,
    )


def _fuse_log_probs(
    torch: Any,
    logits: Sequence[Any],
    names: Sequence[str],
    weights: Sequence[float],
) -> tuple[Any, tuple[float, ...]]:
    stacked = torch.stack([torch.log_softmax(value, dim=-1) for value in logits])
    log_weights = torch.tensor(weights, dtype=torch.float64).log().reshape(-1, 1, 1)
    mixture = torch.logsumexp(stacked + log_weights, dim=0)
    mixture = mixture - torch.logsumexp(mixture, dim=-1, keepdim=True)
    if not bool(torch.isfinite(mixture).all()):
        raise CanonicalGraphError("Parent probability fusion produced non-finite values.")
    diagnostics = []
    for index, _ in enumerate(names):
        parent_log_prob = stacked[index]
        parent_probability = parent_log_prob.exp()
        kl = (parent_probability * (parent_log_prob - mixture)).sum(dim=-1).mean()
        diagnostics.append(float(kl))
    return mixture.contiguous(), tuple(diagnostics)


def _logit_matrix(
    torch: Any,
    value: Any,
    *,
    rows: int,
    vocabulary: int,
    name: str,
) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise CanonicalGraphError(f"{name} must be a matrix tensor.")
    if tuple(value.shape) != (rows, vocabulary):
        raise CanonicalGraphError(f"{name} must have shape ({rows}, {vocabulary}).")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise CanonicalGraphError(f"{name} contains NaN or infinity.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _require_torch()
    rows = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(size) for size in rows.shape)).encode("ascii"))
    digest.update(rows.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalGraphError(f"{name} must be a nonempty string.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CanonicalGraphError(f"{name} must be a positive integer.")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CanonicalGraphError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalGraphError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise CanonicalGraphError(f"{name} must be finite.")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CanonicalGraphError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Canonical execution graphs require PyTorch.") from exc
    return torch


__all__ = [
    "CanonicalChildSpec",
    "CanonicalExecutionGraph",
    "CanonicalGraphEdge",
    "CanonicalGraphError",
    "CanonicalParentLogits",
    "CanonicalTokenContract",
    "build_canonical_execution_graph",
]
