"""Closed-form causal diagnostics for one authenticated exact-4x LFM2 child.

This module is deliberately the small, runner-independent core of the v4
diagnostic.  It defines three-player interventions, exact cooperative-game
decompositions, record-clustered statistics, defined-vocabulary scoring, and
authenticated resumable checkpoints.  It never loads a parent, executes a
parent forward, mutates weights, fits a map, or performs optimization.

The optional transported-feature oracle is fail-closed.  Until an authority-
signed lineage is verified by rehashing its actual cache-derived artifacts, a
published child may still run C/S/M causal replay and the live-feature consumer
oracle, but it cannot make transported-feature or producer-mismatch claims.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .hf_lfm2_exact4x_first_divergence import (
    LFM2_EXACT4X_FIRST_DIVERGENCE_LEDGER_SCHEMA,
    SPLITS,
    authenticate_lfm2_exact4x_first_divergence_batch,
    authenticate_lfm2_exact4x_first_divergence_request,
)
from .hf_lfm2_moving_gauge_residual_reclosure import (
    solve_lfm2_trace_scaled_weighted_anchored_ridge,
)


LFM2_EXACT4X_CAUSAL_DIAGNOSTIC_REQUEST_SCHEMA = (
    "neural-forge-lfm2-exact4x-causal-diagnostic-request-v4"
)
LFM2_EXACT4X_CAUSAL_V3_BINDING_SCHEMA = (
    "neural-forge-lfm2-exact4x-causal-diagnostic-v3-binding-v1"
)
LFM2_EXACT4X_CAUSAL_OPERATOR_SPEC_SCHEMA = (
    "neural-forge-lfm2-exact4x-causal-operator-spec-v1"
)
LFM2_EXACT4X_CAUSAL_RECORD_BINDING_SCHEMA = (
    "neural-forge-lfm2-exact4x-causal-record-binding-v1"
)
LFM2_EXACT4X_CAUSAL_BATCH_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-exact4x-causal-batch-checkpoint-v4"
)
LFM2_EXACT4X_CAUSAL_BATCH_RESULT_SCHEMA = (
    "neural-forge-lfm2-exact4x-causal-batch-result-v4"
)
LFM2_EXACT4X_CAUSAL_LEDGER_SCHEMA = "neural-forge-lfm2-exact4x-causal-ledger-v4"
LFM2_EXACT4X_TRANSPORTED_FEATURE_LINEAGE_SCHEMA = (
    "neural-forge-lfm2-exact4x-transported-feature-lineage-v1"
)
LFM2_EXACT4X_TRANSPORTED_FEATURE_AVAILABILITY_SCHEMA = (
    "neural-forge-lfm2-exact4x-transported-feature-availability-v1"
)
LFM2_EXACT4X_CLUSTERED_BOOTSTRAP_SCHEMA = "neural-forge-record-clustered-bootstrap-v1"

CAUSAL_GROUP_COUNT = 3
CHECKPOINT_HMAC_ALGORITHM = "hmac-sha256"
_CHECKPOINT_HMAC_DOMAIN = b"neural-forge-lfm2-exact4x-causal-batch-checkpoint-v4\0"
_LEDGER_HMAC_DOMAIN = b"neural-forge-lfm2-exact4x-causal-ledger-v4\0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")
_HOOK_KINDS = frozenset({"module_input", "module_output"})

TERMINAL_ERROR_METRIC = "terminal_relative_l2_error"
DEFINED_VOCAB_KL_METRIC = "defined_vocab_kl"
PARENT_WINNER_MARGIN_METRIC = "parent_winner_margin"
TERMINAL_ERROR_BENEFIT = "terminal_error_reduction"
DEFINED_VOCAB_KL_BENEFIT = "defined_vocab_kl_reduction"
PARENT_WINNER_MARGIN_BENEFIT = "parent_winner_margin_gain"


class Lfm2Exact4xCausalDiagnosticError(RuntimeError):
    """The v4 causal request, statistic, or checkpoint is invalid."""


def _plain_json(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (TypeError, ValueError) as exc:
        raise Lfm2Exact4xCausalDiagnosticError(
            f"{name} must be finite canonical JSON."
        ) from exc


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2Exact4xCausalDiagnosticError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 256
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            f"{name} must be a nonempty bounded stable identifier."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2Exact4xCausalDiagnosticError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2Exact4xCausalDiagnosticError(f"{name} must be a nonnegative integer.")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2Exact4xCausalDiagnosticError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise Lfm2Exact4xCausalDiagnosticError(f"{name} must be finite.")
    return result


def _sealed(value: Mapping[str, Any], seal_field: str, schema: str) -> dict[str, Any]:
    row = _plain_json(dict(value), schema)
    if not isinstance(row, dict) or row.get("schema") != schema:
        raise Lfm2Exact4xCausalDiagnosticError(f"{schema} has the wrong schema.")
    claimed = row.pop(seal_field, None)
    if claimed != canonical_sha256(row):
        raise Lfm2Exact4xCausalDiagnosticError(
            f"{schema} does not verify its {seal_field}."
        )
    return {**row, seal_field: claimed}


@dataclass(frozen=True, slots=True)
class Lfm2CausalOperatorSpec:
    """One generic intervention site and its optional consumer feature site."""

    group_id: str
    stage_order: int
    layer_index: int
    component_kind: str
    parent_endpoint: str
    target_module_path: str
    hook_kind: str
    feature_endpoint: str | None = None
    consumer_module_path: str | None = None
    consumer_hook_kind: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.group_id, "group_id")
        _nonnegative_int(self.stage_order, "stage_order")
        _nonnegative_int(self.layer_index, "layer_index")
        _identifier(self.component_kind, "component_kind")
        _identifier(self.parent_endpoint, "parent_endpoint")
        _identifier(self.target_module_path, "target_module_path")
        if self.hook_kind not in _HOOK_KINDS:
            raise Lfm2Exact4xCausalDiagnosticError(
                "hook_kind must be module_input or module_output."
            )
        optional = (
            self.feature_endpoint,
            self.consumer_module_path,
            self.consumer_hook_kind,
        )
        if any(value is None for value in optional) != all(
            value is None for value in optional
        ):
            raise Lfm2Exact4xCausalDiagnosticError(
                "Consumer-oracle feature fields must be all present or all absent."
            )
        if self.feature_endpoint is not None:
            _identifier(self.feature_endpoint, "feature_endpoint")
            _identifier(self.consumer_module_path, "consumer_module_path")
            if self.consumer_hook_kind not in _HOOK_KINDS:
                raise Lfm2Exact4xCausalDiagnosticError(
                    "consumer_hook_kind must be module_input or module_output."
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_EXACT4X_CAUSAL_OPERATOR_SPEC_SCHEMA,
            "group_id": self.group_id,
            "stage_order": self.stage_order,
            "layer_index": self.layer_index,
            "component_kind": self.component_kind,
            "parent_endpoint": self.parent_endpoint,
            "target_module_path": self.target_module_path,
            "hook_kind": self.hook_kind,
            "feature_endpoint": self.feature_endpoint,
            "consumer_module_path": self.consumer_module_path,
            "consumer_hook_kind": self.consumer_hook_kind,
        }


def _operator_spec(value: Any) -> Lfm2CausalOperatorSpec:
    if isinstance(value, Lfm2CausalOperatorSpec):
        return value
    if not isinstance(value, Mapping):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal operator specs must be typed specs or mappings."
        )
    row = _plain_json(dict(value), "causal operator spec")
    expected = {
        "component_kind",
        "consumer_hook_kind",
        "consumer_module_path",
        "feature_endpoint",
        "group_id",
        "hook_kind",
        "layer_index",
        "parent_endpoint",
        "schema",
        "stage_order",
        "target_module_path",
    }
    if not isinstance(row, dict) or set(row) != expected:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal operator spec has a noncanonical field inventory."
        )
    if row.pop("schema") != LFM2_EXACT4X_CAUSAL_OPERATOR_SPEC_SCHEMA:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal operator spec has the wrong schema."
        )
    return Lfm2CausalOperatorSpec(**row)


def canonical_operator_specs(
    values: Sequence[Lfm2CausalOperatorSpec | Mapping[str, Any]],
) -> tuple[Lfm2CausalOperatorSpec, ...]:
    if isinstance(values, (str, bytes)):
        raise Lfm2Exact4xCausalDiagnosticError("operator_specs must be a sequence.")
    specs = tuple(_operator_spec(value) for value in values)
    if (
        len(specs) != CAUSAL_GROUP_COUNT
        or tuple(spec.stage_order for spec in specs) != tuple(range(CAUSAL_GROUP_COUNT))
        or len({spec.group_id for spec in specs}) != CAUSAL_GROUP_COUNT
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Exactly three unique operator groups in stage order 0,1,2 are required."
        )
    return specs


def default_lfm2_l0_csm_operator_specs() -> tuple[Lfm2CausalOperatorSpec, ...]:
    """Return C_in, layer-0 short-conv write, and layer-0 MLP write specs."""

    return (
        Lfm2CausalOperatorSpec(
            group_id="C",
            stage_order=0,
            layer_index=0,
            component_kind="residual_input",
            parent_endpoint="residual.pre",
            target_module_path="model.layers.0",
            hook_kind="module_input",
        ),
        Lfm2CausalOperatorSpec(
            group_id="S",
            stage_order=1,
            layer_index=0,
            component_kind="short_conv_write",
            parent_endpoint="operator.output",
            target_module_path="model.layers.0.conv",
            hook_kind="module_output",
            feature_endpoint="short_conv.context",
            consumer_module_path="model.layers.0.conv.out_proj",
            consumer_hook_kind="module_input",
        ),
        Lfm2CausalOperatorSpec(
            group_id="M",
            stage_order=2,
            layer_index=0,
            component_kind="mlp_write",
            parent_endpoint="mlp.down",
            target_module_path="model.layers.0.feed_forward.w2",
            hook_kind="module_output",
            feature_endpoint="mlp.post_swiglu",
            consumer_module_path="model.layers.0.feed_forward.w2",
            consumer_hook_kind="module_input",
        ),
    )


def canonical_intervention_subsets(
    group_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Return the eight subsets in deterministic low-bit-first mask order."""

    groups = tuple(_identifier(value, "group_id") for value in group_ids)
    if len(groups) != CAUSAL_GROUP_COUNT or len(set(groups)) != len(groups):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Canonical causal subsets require exactly three unique groups."
        )
    return tuple(
        {
            "mask": mask,
            "subset_id": "".join(
                "1" if mask & (1 << index) else "0"
                for index in range(CAUSAL_GROUP_COUNT)
            ),
            "members": [
                group for index, group in enumerate(groups) if mask & (1 << index)
            ],
        }
        for mask in range(1 << CAUSAL_GROUP_COUNT)
    )


def _coalition_key(value: Any, groups: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, bool):
        raise Lfm2Exact4xCausalDiagnosticError("Coalition keys cannot be boolean.")
    if isinstance(value, int):
        if value < 0 or value >= 1 << len(groups):
            raise Lfm2Exact4xCausalDiagnosticError("Coalition mask is out of range.")
        members = {group for index, group in enumerate(groups) if value & (1 << index)}
    elif (
        isinstance(value, str)
        and len(value) == len(groups)
        and set(value) <= {"0", "1"}
    ):
        members = {group for index, group in enumerate(groups) if value[index] == "1"}
    elif isinstance(value, (tuple, frozenset, set, list)):
        members = set(value)
    else:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Coalition keys must be masks, subset IDs, or group collections."
        )
    if any(not isinstance(group, str) for group in members) or not members.issubset(
        groups
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Coalition contains a group outside the causal request."
        )
    return tuple(group for group in groups if group in members)


def _coalition_values(
    values: Mapping[Any, Any], group_ids: Sequence[str]
) -> tuple[tuple[str, ...], dict[tuple[str, ...], float]]:
    groups = tuple(_identifier(value, "group_id") for value in group_ids)
    canonical = canonical_intervention_subsets(groups)
    result: dict[tuple[str, ...], float] = {}
    for key, value in values.items():
        coalition = _coalition_key(key, groups)
        if coalition in result:
            raise Lfm2Exact4xCausalDiagnosticError(
                "Coalition values contain duplicate subset identities."
            )
        result[coalition] = _finite(value, f"coalition {coalition!r}")
    expected = {tuple(row["members"]) for row in canonical}
    if set(result) != expected:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Coalition values must cover all eight canonical subsets exactly once."
        )
    return groups, result


def exact_three_player_shapley(
    benefits: Mapping[Any, Any], group_ids: Sequence[str]
) -> dict[str, float]:
    """Compute exact Shapley values from all eight higher-is-better benefits."""

    groups, values = _coalition_values(benefits, group_ids)
    factorial = math.factorial
    denominator = factorial(len(groups))
    result: dict[str, float] = {}
    for player in groups:
        others = tuple(group for group in groups if group != player)
        contribution = 0.0
        for mask in range(1 << len(others)):
            subset_members = {
                group for index, group in enumerate(others) if mask & (1 << index)
            }
            subset = tuple(group for group in groups if group in subset_members)
            with_player = tuple(
                group for group in groups if group in subset_members or group == player
            )
            size = len(subset)
            weight = factorial(size) * factorial(len(groups) - size - 1) / denominator
            contribution += weight * (values[with_player] - values[subset])
        result[player] = contribution
    return result


def recursive_harsanyi_dividends(
    benefits: Mapping[Any, Any], group_ids: Sequence[str]
) -> dict[tuple[str, ...], float]:
    """Compute every Harsanyi dividend by subset-lattice recursion."""

    groups, values = _coalition_values(benefits, group_ids)
    dividends: dict[tuple[str, ...], float] = {}
    for row in canonical_intervention_subsets(groups):
        coalition = tuple(row["members"])
        coalition_set = set(coalition)
        proper_sum = math.fsum(
            dividend
            for subset, dividend in dividends.items()
            if set(subset) < coalition_set
        )
        dividends[coalition] = values[coalition] - proper_sum
    return dividends


def three_player_causal_decomposition(
    benefits: Mapping[Any, Any], group_ids: Sequence[str]
) -> dict[str, Any]:
    """Return JSON-safe exact Shapley and recursive Harsanyi evidence."""

    groups, values = _coalition_values(benefits, group_ids)
    shapley = exact_three_player_shapley(values, groups)
    dividends = recursive_harsanyi_dividends(values, groups)
    empty = ()
    grand = groups
    total_gain = values[grand] - values[empty]
    body = {
        "schema": "neural-forge-exact-three-player-causal-decomposition-v1",
        "group_ids": list(groups),
        "subset_benefits": [
            {
                **row,
                "benefit": values[tuple(row["members"])],
            }
            for row in canonical_intervention_subsets(groups)
        ],
        "shapley_values": [
            {"group_id": group, "value": shapley[group]} for group in groups
        ],
        "harsanyi_dividends": [
            {
                **row,
                "dividend": dividends[tuple(row["members"])],
            }
            for row in canonical_intervention_subsets(groups)
        ],
        "grand_coalition_gain_over_empty": total_gain,
        "shapley_efficiency_remainder": total_gain - math.fsum(shapley.values()),
        "harsanyi_reconstruction_remainder": values[grand]
        - math.fsum(dividends.values()),
    }
    return {**body, "decomposition_sha256": canonical_sha256(body)}


def higher_is_better_benefits(
    subset_metrics: Mapping[Any, Mapping[str, Any]],
    group_ids: Sequence[str],
) -> dict[str, Mapping[tuple[str, ...], float]]:
    """Orient terminal error, defined-vocab KL, and margin as positive gains."""

    groups = tuple(group_ids)
    canonical_intervention_subsets(groups)
    metrics: dict[tuple[str, ...], dict[str, float]] = {}
    required = {
        TERMINAL_ERROR_METRIC,
        DEFINED_VOCAB_KL_METRIC,
        PARENT_WINNER_MARGIN_METRIC,
    }
    for key, value in subset_metrics.items():
        coalition = _coalition_key(key, groups)
        if (
            coalition in metrics
            or not isinstance(value, Mapping)
            or set(value) != required
        ):
            raise Lfm2Exact4xCausalDiagnosticError(
                "Subset metrics must cover each coalition once with the three raw metrics."
            )
        metrics[coalition] = {
            name: _finite(value[name], f"{coalition!r}.{name}") for name in required
        }
    expected = {tuple(row["members"]) for row in canonical_intervention_subsets(groups)}
    if set(metrics) != expected:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Subset metrics must cover all eight canonical coalitions."
        )
    baseline = metrics[()]
    return {
        TERMINAL_ERROR_BENEFIT: MappingProxyType(
            {
                coalition: baseline[TERMINAL_ERROR_METRIC] - row[TERMINAL_ERROR_METRIC]
                for coalition, row in metrics.items()
            }
        ),
        DEFINED_VOCAB_KL_BENEFIT: MappingProxyType(
            {
                coalition: baseline[DEFINED_VOCAB_KL_METRIC]
                - row[DEFINED_VOCAB_KL_METRIC]
                for coalition, row in metrics.items()
            }
        ),
        PARENT_WINNER_MARGIN_BENEFIT: MappingProxyType(
            {
                coalition: row[PARENT_WINNER_MARGIN_METRIC]
                - baseline[PARENT_WINNER_MARGIN_METRIC]
                for coalition, row in metrics.items()
            }
        ),
    }


def defined_vocab_kl_rows(
    parent_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    defined_semantic_row_count: int,
) -> torch.Tensor:
    """Return KL(parent || candidate) using only the defined vocabulary prefix."""

    count = _positive_int(defined_semantic_row_count, "defined_semantic_row_count")
    for value, name in (
        (parent_logits, "parent_logits"),
        (candidate_logits, "candidate_logits"),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 1
            or not torch.is_floating_point(value)
            or value.is_complex()
            or int(value.shape[-1]) < count
            or not bool(torch.isfinite(value.detach()).all())
        ):
            raise Lfm2Exact4xCausalDiagnosticError(
                f"{name} must be finite real logits covering the defined vocabulary."
            )
    if tuple(parent_logits.shape[:-1]) != tuple(candidate_logits.shape[:-1]):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Parent and candidate logits must have equal leading geometry."
        )
    parent = parent_logits.detach().to(device="cpu", dtype=torch.float64)[..., :count]
    candidate = candidate_logits.detach().to(device="cpu", dtype=torch.float64)[
        ..., :count
    ]
    parent_log_prob = torch.log_softmax(parent, dim=-1)
    candidate_log_prob = torch.log_softmax(candidate, dim=-1)
    rows = torch.sum(
        torch.exp(parent_log_prob) * (parent_log_prob - candidate_log_prob),
        dim=-1,
    )
    return rows.clamp_min(0.0)


def equal_record_mass_weights(record_ids: Sequence[str]) -> torch.Tensor:
    """Give every record equal mass and every row within it uniform mass."""

    if isinstance(record_ids, (str, bytes)):
        raise Lfm2Exact4xCausalDiagnosticError("record_ids must be a sequence.")
    records = tuple(_identifier(value, "record_id") for value in record_ids)
    if not records:
        raise Lfm2Exact4xCausalDiagnosticError("record_ids cannot be empty.")
    counts: dict[str, int] = {}
    for record in records:
        counts[record] = counts.get(record, 0) + 1
    record_mass = 1.0 / len(counts)
    weights = torch.tensor(
        [record_mass / counts[record] for record in records], dtype=torch.float64
    )
    return weights / weights.sum()


def aggregate_defined_vocab_kl(
    parent_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    defined_semantic_row_count: int,
    record_ids: Sequence[str] | None = None,
) -> float:
    rows = defined_vocab_kl_rows(
        parent_logits,
        candidate_logits,
        defined_semantic_row_count=defined_semantic_row_count,
    ).reshape(-1)
    if record_ids is None:
        return float(rows.mean().item())
    weights = equal_record_mass_weights(record_ids)
    if int(weights.numel()) != int(rows.numel()):
        raise Lfm2Exact4xCausalDiagnosticError(
            "record_ids must identify every KL row exactly once."
        )
    return float(torch.dot(rows, weights).item())


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(
        sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def deterministic_record_clustered_bootstrap(
    values: Sequence[float] | torch.Tensor,
    record_ids: Sequence[str],
    *,
    seed: str,
    resample_count: int = 4096,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap whole record means with replacement under a deterministic seed."""

    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or not bool(torch.isfinite(values.detach()).all()):
            raise Lfm2Exact4xCausalDiagnosticError(
                "Bootstrap tensor values must be a finite vector."
            )
        observed = tuple(
            float(value)
            for value in values.detach().to(device="cpu", dtype=torch.float64).tolist()
        )
    else:
        observed = tuple(_finite(value, "bootstrap value") for value in values)
    records = tuple(_identifier(value, "record_id") for value in record_ids)
    if not observed or len(observed) != len(records):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Bootstrap values and record IDs must be nonempty and row-aligned."
        )
    if not isinstance(seed, str) or not seed or len(seed) > 4096:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Bootstrap seed must be a nonempty bounded string."
        )
    draws = _positive_int(resample_count, "resample_count")
    confidence = _finite(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise Lfm2Exact4xCausalDiagnosticError(
            "confidence_level must lie strictly between zero and one."
        )
    by_record: dict[str, list[float]] = {}
    for record, value in zip(records, observed, strict=True):
        by_record.setdefault(record, []).append(value)
    ordered_records = tuple(sorted(by_record))
    record_means = tuple(
        math.fsum(sorted(by_record[record])) / len(by_record[record])
        for record in ordered_records
    )
    inventory = {
        "seed": seed,
        "ordered_record_ids": list(ordered_records),
        "row_counts": [len(by_record[record]) for record in ordered_records],
    }
    seed_sha256 = canonical_sha256(inventory)
    generator = random.Random(int(seed_sha256, 16))
    record_count = len(record_means)
    estimates = sorted(
        math.fsum(
            record_means[generator.randrange(record_count)] for _ in range(record_count)
        )
        / record_count
        for _ in range(draws)
    )
    alpha = (1.0 - confidence) / 2.0
    one_sided_alpha = 1.0 - confidence
    body = {
        "schema": LFM2_EXACT4X_CLUSTERED_BOOTSTRAP_SCHEMA,
        "seed_sha256": seed_sha256,
        "record_inventory_sha256": canonical_sha256(
            {
                "ordered_record_ids": list(ordered_records),
                "row_counts": [len(by_record[record]) for record in ordered_records],
            }
        ),
        "row_count": len(observed),
        "record_count": record_count,
        "resample_count": draws,
        "confidence_level": confidence,
        "point_estimate": math.fsum(record_means) / record_count,
        "lower_percentile": _percentile(estimates, alpha),
        "upper_percentile": _percentile(estimates, 1.0 - alpha),
        "central_two_sided_interval_confidence_level": confidence,
        "one_sided_lower_percentile": _percentile(estimates, one_sided_alpha),
        "one_sided_upper_percentile": _percentile(estimates, confidence),
        "one_sided_bound_confidence_level": confidence,
        "one_sided_bounds_are_diagnostic_not_gates": True,
        "cluster_unit": "whole-record",
        "record_total_mass": "equal",
        "within_record_row_mass": "uniform",
        "phase_rows_are_not_bootstrap_units": True,
    }
    return {**body, "bootstrap_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class Lfm2CausalRecordBinding:
    """One immutable cache record and its corresponding v3 batch audit."""

    batch_index: int
    trace_input_sha256: str
    trace_record_sha256: str
    split: str
    v3_batch_audit_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.batch_index, "batch_index")
        _digest(self.trace_input_sha256, "trace_input_sha256")
        _digest(self.trace_record_sha256, "trace_record_sha256")
        _digest(self.v3_batch_audit_sha256, "v3_batch_audit_sha256")
        if self.split not in SPLITS:
            raise Lfm2Exact4xCausalDiagnosticError(
                "Record split must be fit or heldout."
            )

    @property
    def record_id(self) -> str:
        return self.trace_input_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_EXACT4X_CAUSAL_RECORD_BINDING_SCHEMA,
            "batch_index": self.batch_index,
            "record_id": self.record_id,
            "trace_input_sha256": self.trace_input_sha256,
            "trace_record_sha256": self.trace_record_sha256,
            "split": self.split,
            "v3_batch_audit_sha256": self.v3_batch_audit_sha256,
        }


def _record_binding(value: Any) -> Lfm2CausalRecordBinding:
    if isinstance(value, Lfm2CausalRecordBinding):
        return value
    if not isinstance(value, Mapping):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal record bindings must be typed bindings or mappings."
        )
    row = _plain_json(dict(value), "causal record binding")
    expected = {
        "batch_index",
        "record_id",
        "schema",
        "split",
        "trace_input_sha256",
        "trace_record_sha256",
        "v3_batch_audit_sha256",
    }
    if not isinstance(row, dict) or set(row) != expected:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal record binding has a noncanonical field inventory."
        )
    if row.pop("schema") != LFM2_EXACT4X_CAUSAL_RECORD_BINDING_SCHEMA or row.pop(
        "record_id"
    ) != row.get("trace_input_sha256"):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal record binding schema or record identity is invalid."
        )
    return Lfm2CausalRecordBinding(**row)


def canonical_record_bindings(
    values: Sequence[Lfm2CausalRecordBinding | Mapping[str, Any]],
) -> tuple[Lfm2CausalRecordBinding, ...]:
    if isinstance(values, (str, bytes)):
        raise Lfm2Exact4xCausalDiagnosticError("record_bindings must be a sequence.")
    rows = tuple(_record_binding(value) for value in values)
    if (
        not rows
        or tuple(row.batch_index for row in rows) != tuple(range(len(rows)))
        or len({row.record_id for row in rows}) != len(rows)
        or set(row.split for row in rows) != set(SPLITS)
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Record bindings must be unique, contiguous, ordered, and cover both splits."
        )
    return rows


def authenticate_v3_first_divergence_ledger(
    value: Mapping[str, Any],
    *,
    expected_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the canonical v3 ledger and its embedded v3 request."""

    ledger = _sealed(
        value,
        "ledger_sha256",
        LFM2_EXACT4X_FIRST_DIVERGENCE_LEDGER_SCHEMA,
    )
    expected_fields = {
        "batch_count",
        "component_first_divergence_is_descriptive_not_thresholded",
        "first_discrete_c_out_parent_winner_mismatch",
        "free_generation_performed",
        "generation_cap_saturation_is_censored_not_failure",
        "ledger_sha256",
        "manual_whole_response_intelligence_review_is_authoritative",
        "numeric_acceptance_decision",
        "numeric_metrics_are_diagnostic_only",
        "operation_audit",
        "ordered_batch_audit_sha256s",
        "request",
        "safety_observations_are_separate_not_failures",
        "schema",
        "split_summaries",
        "status",
    }
    if set(ledger) != expected_fields:
        raise Lfm2Exact4xCausalDiagnosticError(
            "First-divergence ledger has a noncanonical field inventory."
        )
    request = authenticate_lfm2_exact4x_first_divergence_request(ledger["request"])
    if expected_request is not None:
        expected = authenticate_lfm2_exact4x_first_divergence_request(expected_request)
        if request["request_sha256"] != expected["request_sha256"]:
            raise Lfm2Exact4xCausalDiagnosticError(
                "First-divergence ledger belongs to a different v3 request."
            )
    count = _positive_int(ledger.get("batch_count"), "v3 ledger batch_count")
    batch_digests = ledger.get("ordered_batch_audit_sha256s")
    summaries = ledger.get("split_summaries")
    if (
        not isinstance(batch_digests, list)
        or len(batch_digests) != count
        or len(set(batch_digests)) != count
        or any(_digest(item, "v3 batch audit") != item for item in batch_digests)
        or not isinstance(summaries, Mapping)
        or set(summaries) != set(SPLITS)
        or any(not isinstance(summaries[split], Mapping) for split in SPLITS)
        or sum(
            _positive_int(
                summaries[split].get("batch_count"), f"v3 {split} batch_count"
            )
            for split in SPLITS
        )
        != count
        or ledger.get("status") != "complete-observation"
        or ledger.get("numeric_acceptance_decision") is not None
        or ledger.get("numeric_metrics_are_diagnostic_only") is not True
        or ledger.get("manual_whole_response_intelligence_review_is_authoritative")
        is not True
        or ledger.get("safety_observations_are_separate_not_failures") is not True
        or ledger.get("generation_cap_saturation_is_censored_not_failure") is not True
        or ledger.get("free_generation_performed") is not False
        or ledger.get("component_first_divergence_is_descriptive_not_thresholded")
        is not True
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "First-divergence ledger counts or diagnostic policy are invalid."
        )
    return ledger


def build_v3_first_divergence_binding(
    first_divergence_request: Mapping[str, Any],
    first_divergence_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    request = authenticate_lfm2_exact4x_first_divergence_request(
        first_divergence_request
    )
    ledger = authenticate_v3_first_divergence_ledger(
        first_divergence_ledger,
        expected_request=request,
    )
    body = {
        "schema": LFM2_EXACT4X_CAUSAL_V3_BINDING_SCHEMA,
        "first_divergence_request_sha256": request["request_sha256"],
        "first_divergence_ledger_sha256": ledger["ledger_sha256"],
        "child_package_sha256": request["child_package_sha256"],
        "child_state_sha256": request["child_state_sha256"],
        "trace_cache_key": request["trace_cache_key"],
        "trace_selective_session_sha256": request["trace_selective_session_sha256"],
        "basis_phase_key_sha256": request["basis_phase_key_sha256"],
        "basis_completion_sha256": request["basis_completion_sha256"],
        "basis_map_sha256": request["basis_map_sha256"],
        "parent_logit_domain_contract_sha256": request[
            "parent_logit_domain_contract_sha256"
        ],
        "defined_semantic_row_count": request["parent_logit_domain_contract"][
            "defined_semantic_row_count"
        ],
        "parent_physical_row_capacity": request["parent_logit_domain_contract"][
            "parent_physical_row_capacity"
        ],
        "layer_count": request["layer_count"],
        "runtime_device": request["runtime_device"],
        "runtime_dtype": request["runtime_dtype"],
        "batch_count": ledger["batch_count"],
        "ordered_batch_audit_sha256s_sha256": canonical_sha256(
            ledger["ordered_batch_audit_sha256s"]
        ),
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def authenticate_v3_first_divergence_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _sealed(
        value,
        "binding_sha256",
        LFM2_EXACT4X_CAUSAL_V3_BINDING_SCHEMA,
    )
    expected = {
        "basis_completion_sha256",
        "basis_map_sha256",
        "basis_phase_key_sha256",
        "batch_count",
        "binding_sha256",
        "child_package_sha256",
        "child_state_sha256",
        "defined_semantic_row_count",
        "first_divergence_ledger_sha256",
        "first_divergence_request_sha256",
        "layer_count",
        "ordered_batch_audit_sha256s_sha256",
        "parent_logit_domain_contract_sha256",
        "parent_physical_row_capacity",
        "runtime_device",
        "runtime_dtype",
        "schema",
        "trace_cache_key",
        "trace_selective_session_sha256",
    }
    if set(binding) != expected:
        raise Lfm2Exact4xCausalDiagnosticError(
            "First-divergence v3 binding is noncanonical."
        )
    for field in expected - {
        "schema",
        "batch_count",
        "defined_semantic_row_count",
        "layer_count",
        "parent_physical_row_capacity",
        "runtime_device",
        "runtime_dtype",
    }:
        _digest(binding.get(field), f"v3 binding {field}")
    _positive_int(binding.get("batch_count"), "v3 binding batch_count")
    defined_count = _positive_int(
        binding.get("defined_semantic_row_count"),
        "v3 binding defined_semantic_row_count",
    )
    physical_count = _positive_int(
        binding.get("parent_physical_row_capacity"),
        "v3 binding parent_physical_row_capacity",
    )
    _positive_int(binding.get("layer_count"), "v3 binding layer_count")
    if (
        physical_count < defined_count
        or not isinstance(binding.get("runtime_device"), str)
        or not binding["runtime_device"]
        or binding.get("runtime_dtype")
        not in {"float16", "bfloat16", "float32", "float64"}
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "First-divergence v3 numeric domain is invalid."
        )
    return binding


def authenticate_v3_batch_identity_manifest(
    first_divergence_request: Mapping[str, Any],
    first_divergence_ledger: Mapping[str, Any],
    first_divergence_batch_results: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[Lfm2CausalRecordBinding, ...],
    str,
]:
    """Bind v4 record identities to authenticated v3 batch result contents."""

    request = authenticate_lfm2_exact4x_first_divergence_request(
        first_divergence_request
    )
    ledger = authenticate_v3_first_divergence_ledger(
        first_divergence_ledger,
        expected_request=request,
    )
    if isinstance(first_divergence_batch_results, (str, bytes)):
        raise Lfm2Exact4xCausalDiagnosticError(
            "first_divergence_batch_results must be a sequence."
        )
    batches = tuple(
        authenticate_lfm2_exact4x_first_divergence_batch(
            value,
            expected_request=request,
        )
        for value in first_divergence_batch_results
    )
    if (
        len(batches) != ledger["batch_count"]
        or tuple(batch["batch_index"] for batch in batches)
        != tuple(range(len(batches)))
        or [batch["batch_audit_sha256"] for batch in batches]
        != ledger["ordered_batch_audit_sha256s"]
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Authenticated v3 batch results differ from the complete v3 ledger."
        )
    bindings = tuple(
        Lfm2CausalRecordBinding(
            batch_index=int(batch["batch_index"]),
            trace_input_sha256=str(batch["trace_input_sha256"]),
            trace_record_sha256=str(batch["trace_record_sha256"]),
            split=str(batch["split"]),
            v3_batch_audit_sha256=str(batch["batch_audit_sha256"]),
        )
        for batch in batches
    )
    canonical = canonical_record_bindings(bindings)
    manifest_sha256 = canonical_sha256([binding.as_dict() for binding in canonical])
    return batches, canonical, manifest_sha256


@dataclass(frozen=True, slots=True)
class Lfm2TransportedFeatureBinding:
    """Existing cache-derived feature coordinates for one consumer operator."""

    group_id: str
    source_feature_tensor_name: str
    target_consumer_module_path: str
    source_row_selection_sha256: str
    transport_tensor_sha256: str
    source_width: int
    target_width: int

    def __post_init__(self) -> None:
        _identifier(self.group_id, "transport group_id")
        _identifier(self.source_feature_tensor_name, "source_feature_tensor_name")
        _identifier(self.target_consumer_module_path, "target_consumer_module_path")
        _digest(self.source_row_selection_sha256, "source_row_selection_sha256")
        _digest(self.transport_tensor_sha256, "transport_tensor_sha256")
        _positive_int(self.source_width, "source_width")
        _positive_int(self.target_width, "target_width")

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "source_feature_tensor_name": self.source_feature_tensor_name,
            "target_consumer_module_path": self.target_consumer_module_path,
            "source_row_selection_sha256": self.source_row_selection_sha256,
            "transport_tensor_sha256": self.transport_tensor_sha256,
            "source_width": self.source_width,
            "target_width": self.target_width,
        }


def _transport_binding(value: Any) -> Lfm2TransportedFeatureBinding:
    if isinstance(value, Lfm2TransportedFeatureBinding):
        return value
    if not isinstance(value, Mapping):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Transport feature bindings must be typed bindings or mappings."
        )
    row = _plain_json(dict(value), "transport feature binding")
    expected = {
        "group_id",
        "source_feature_tensor_name",
        "source_row_selection_sha256",
        "source_width",
        "target_consumer_module_path",
        "target_width",
        "transport_tensor_sha256",
    }
    if not isinstance(row, dict) or set(row) != expected:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Transport feature binding has a noncanonical field inventory."
        )
    return Lfm2TransportedFeatureBinding(**row)


def build_lfm2_transported_feature_lineage(
    *,
    trace_cache_key: str,
    trace_selective_session_sha256: str,
    basis_map_sha256: str,
    operator_specs: Sequence[Lfm2CausalOperatorSpec | Mapping[str, Any]],
    derivation_implementation_sha256: str,
    feature_bindings: Sequence[Lfm2TransportedFeatureBinding | Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject the legacy self-asserted, unkeyed v1 lineage envelope.

    Its shape remains recognizable for migration diagnostics, but a canonical
    SHA over caller assertions cannot prove cache derivation.  Future support
    requires an authority-signed receipt and runner-side rehashing of the
    actual tensor and row-selection artifacts.
    """

    raise Lfm2Exact4xCausalDiagnosticError(
        "Legacy v1 transported-feature lineage is unsupported and unverifiable; "
        "an authority-signed receipt plus runner artifact rehashing is required."
    )


def authenticate_lfm2_transported_feature_lineage(
    value: Mapping[str, Any],
    *,
    operator_specs: Sequence[Lfm2CausalOperatorSpec | Mapping[str, Any]],
    trace_cache_key: str,
    trace_selective_session_sha256: str,
    basis_map_sha256: str,
) -> dict[str, Any]:
    raise Lfm2Exact4xCausalDiagnosticError(
        "Legacy v1 transported-feature lineage is unsupported and unverifiable; "
        "an authority-signed receipt plus runner artifact rehashing is required."
    )


def transported_feature_availability(
    lineage: Mapping[str, Any] | None,
    *,
    operator_specs: Sequence[Lfm2CausalOperatorSpec | Mapping[str, Any]],
    trace_cache_key: str,
    trace_selective_session_sha256: str,
    basis_map_sha256: str,
) -> dict[str, Any]:
    canonical_operator_specs(operator_specs)
    _digest(trace_cache_key, "trace_cache_key")
    _digest(trace_selective_session_sha256, "trace_selective_session_sha256")
    _digest(basis_map_sha256, "basis_map_sha256")
    if lineage is not None:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Legacy v1 transported-feature lineage is unsupported and unverifiable; "
            "an authority-signed receipt plus runner artifact rehashing is required."
        )
    body = {
        "schema": LFM2_EXACT4X_TRANSPORTED_FEATURE_AVAILABILITY_SCHEMA,
        "status": "unavailable",
        "lineage_sha256": None,
        "reason": "no-authority-signed-rehashed-transport-lineage",
        "transported_feature_oracle_available": False,
        "producer_mismatch_classification_available": False,
        "live_feature_consumer_oracle_available": True,
        "csm_causal_replay_available": True,
        "missing_coordinates_are_not_invented": True,
        "legacy_v1_lineage_supported": False,
        "future_authority_signed_receipt_required": True,
        "runner_artifact_rehash_required": True,
    }
    return {**body, "availability_sha256": canonical_sha256(body)}


def build_lfm2_exact4x_causal_diagnostic_request(
    *,
    first_divergence_request: Mapping[str, Any],
    first_divergence_ledger: Mapping[str, Any],
    first_divergence_batch_results: Sequence[Mapping[str, Any]],
    record_bindings: Sequence[Lfm2CausalRecordBinding | Mapping[str, Any]],
    operator_specs: Sequence[Lfm2CausalOperatorSpec | Mapping[str, Any]] = (),
    implementation_identity_sha256: str,
    checkpoint_hmac_key_id: str,
    bootstrap_seed: str,
    bootstrap_resample_count: int = 4096,
    bootstrap_confidence_level: float = 0.95,
    transported_feature_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable v4 request downstream of a complete v3 ledger."""

    specs = canonical_operator_specs(
        default_lfm2_l0_csm_operator_specs() if not operator_specs else operator_specs
    )
    records = canonical_record_bindings(record_bindings)
    v3_binding = build_v3_first_divergence_binding(
        first_divergence_request,
        first_divergence_ledger,
    )
    ledger = authenticate_v3_first_divergence_ledger(
        first_divergence_ledger,
        expected_request=first_divergence_request,
    )
    _v3_batches, authenticated_records, authenticated_record_manifest_sha256 = (
        authenticate_v3_batch_identity_manifest(
            first_divergence_request,
            first_divergence_ledger,
            first_divergence_batch_results,
        )
    )
    if (
        len(records) != v3_binding["batch_count"]
        or [row.v3_batch_audit_sha256 for row in records]
        != ledger["ordered_batch_audit_sha256s"]
        or records != authenticated_records
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal record bindings differ from authenticated v3 batch identities."
        )
    implementation_sha256 = _digest(
        implementation_identity_sha256, "implementation_identity_sha256"
    )
    key_id = _identifier(checkpoint_hmac_key_id, "checkpoint_hmac_key_id")
    if (
        not isinstance(bootstrap_seed, str)
        or not bootstrap_seed
        or len(bootstrap_seed) > 4096
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "bootstrap_seed must be a nonempty bounded string."
        )
    resamples = _positive_int(bootstrap_resample_count, "bootstrap_resample_count")
    confidence = _finite(bootstrap_confidence_level, "bootstrap_confidence_level")
    if not 0.0 < confidence < 1.0:
        raise Lfm2Exact4xCausalDiagnosticError(
            "bootstrap_confidence_level must lie strictly between zero and one."
        )
    specs_rows = [spec.as_dict() for spec in specs]
    availability = transported_feature_availability(
        transported_feature_lineage,
        operator_specs=specs,
        trace_cache_key=v3_binding["trace_cache_key"],
        trace_selective_session_sha256=v3_binding["trace_selective_session_sha256"],
        basis_map_sha256=v3_binding["basis_map_sha256"],
    )
    record_rows = [row.as_dict() for row in records]
    body = {
        "schema": LFM2_EXACT4X_CAUSAL_DIAGNOSTIC_REQUEST_SCHEMA,
        "v3_binding": v3_binding,
        "v3_binding_sha256": v3_binding["binding_sha256"],
        "record_bindings": record_rows,
        "record_manifest_sha256": canonical_sha256(record_rows),
        "authenticated_v3_record_manifest_sha256": (
            authenticated_record_manifest_sha256
        ),
        "record_bindings_authenticated_from_v3_batch_results": True,
        "record_count": len(records),
        "fit_record_count": sum(row.split == "fit" for row in records),
        "heldout_record_count": sum(row.split == "heldout" for row in records),
        "operator_specs": specs_rows,
        "operator_specs_sha256": canonical_sha256(specs_rows),
        "group_ids": [spec.group_id for spec in specs],
        "canonical_subsets": list(
            canonical_intervention_subsets([spec.group_id for spec in specs])
        ),
        "metric_contract": {
            "raw_metrics": [
                TERMINAL_ERROR_METRIC,
                DEFINED_VOCAB_KL_METRIC,
                PARENT_WINNER_MARGIN_METRIC,
            ],
            "higher_is_better_benefits": [
                TERMINAL_ERROR_BENEFIT,
                DEFINED_VOCAB_KL_BENEFIT,
                PARENT_WINNER_MARGIN_BENEFIT,
            ],
            "defined_semantic_row_count": v3_binding["defined_semantic_row_count"],
            "parent_physical_row_capacity": v3_binding["parent_physical_row_capacity"],
            "undefined_trailing_capacity_rows_excluded_from_kl": True,
            "top1_is_descriptive_only": True,
        },
        "bootstrap_contract": {
            "seed": bootstrap_seed,
            "seed_sha256": hashlib.sha256(bootstrap_seed.encode("utf-8")).hexdigest(),
            "resample_count": resamples,
            "confidence_level": confidence,
            "confidence_bound_contract": (
                "explicit-one-sided-lower-and-upper-plus-central-two-sided"
            ),
            "cluster_unit": "whole-record",
            "record_total_mass": "equal",
            "within_record_row_mass": "uniform",
            "split": "heldout",
        },
        "transported_feature_lineage": (
            None
            if transported_feature_lineage is None
            else _plain_json(
                dict(transported_feature_lineage), "transported feature lineage"
            )
        ),
        "transported_feature_availability": availability,
        "implementation_identity_sha256": implementation_sha256,
        "checkpoint_hmac_algorithm": CHECKPOINT_HMAC_ALGORITHM,
        "checkpoint_hmac_key_id": key_id,
        "policy": {
            "sealed_cached_parent_evidence_only": True,
            "parent_forward_calls": 0,
            "child_forward_only": True,
            "gradient_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "iterative_optimization_allowed": False,
            "global_residual_map_allowed": False,
            "exact_three_player_shapley": True,
            "recursive_harsanyi": True,
            "numeric_metrics_are_diagnostic_only": True,
            "manual_whole_response_intelligence_review_is_authoritative": True,
            "missing_transport_coordinates_must_not_be_invented": True,
        },
    }
    return {**body, "request_sha256": canonical_sha256(body)}


def authenticate_lfm2_exact4x_causal_diagnostic_request(
    value: Mapping[str, Any],
    *,
    expected_first_divergence_request: Mapping[str, Any] | None = None,
    expected_first_divergence_ledger: Mapping[str, Any] | None = None,
    expected_first_divergence_batch_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    request = _sealed(
        value,
        "request_sha256",
        LFM2_EXACT4X_CAUSAL_DIAGNOSTIC_REQUEST_SCHEMA,
    )
    expected = {
        "authenticated_v3_record_manifest_sha256",
        "bootstrap_contract",
        "canonical_subsets",
        "checkpoint_hmac_algorithm",
        "checkpoint_hmac_key_id",
        "fit_record_count",
        "group_ids",
        "heldout_record_count",
        "implementation_identity_sha256",
        "metric_contract",
        "operator_specs",
        "operator_specs_sha256",
        "policy",
        "record_bindings",
        "record_bindings_authenticated_from_v3_batch_results",
        "record_count",
        "record_manifest_sha256",
        "request_sha256",
        "schema",
        "transported_feature_availability",
        "transported_feature_lineage",
        "v3_binding",
        "v3_binding_sha256",
    }
    if set(request) != expected:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal diagnostic request has a noncanonical field inventory."
        )
    v3_binding = authenticate_v3_first_divergence_binding(request["v3_binding"])
    if request.get("v3_binding_sha256") != v3_binding["binding_sha256"]:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal request v3 binding digest is inconsistent."
        )
    expected_v3_presence = (
        expected_first_divergence_request is not None,
        expected_first_divergence_ledger is not None,
        expected_first_divergence_batch_results is not None,
    )
    if len(set(expected_v3_presence)) != 1:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Expected v3 request, ledger, and batch results must be supplied together."
        )
    if expected_first_divergence_request is not None:
        expected_v3 = build_v3_first_divergence_binding(
            expected_first_divergence_request,
            expected_first_divergence_ledger,
        )
        if expected_v3["binding_sha256"] != v3_binding["binding_sha256"]:
            raise Lfm2Exact4xCausalDiagnosticError(
                "Causal request differs from the expected v3 evidence."
            )
    specs_value = request.get("operator_specs")
    records_value = request.get("record_bindings")
    if not isinstance(specs_value, list) or not isinstance(records_value, list):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal request operator and record inventories must be lists."
        )
    specs = canonical_operator_specs(specs_value)
    records = canonical_record_bindings(records_value)
    group_ids = [spec.group_id for spec in specs]
    if (
        request.get("operator_specs_sha256")
        != canonical_sha256([spec.as_dict() for spec in specs])
        or request.get("group_ids") != group_ids
        or request.get("canonical_subsets")
        != list(canonical_intervention_subsets(group_ids))
        or request.get("record_manifest_sha256")
        != canonical_sha256([row.as_dict() for row in records])
        or request.get("authenticated_v3_record_manifest_sha256")
        != request.get("record_manifest_sha256")
        or request.get("record_bindings_authenticated_from_v3_batch_results")
        is not True
        or request.get("record_count") != len(records)
        or request.get("fit_record_count") != sum(row.split == "fit" for row in records)
        or request.get("heldout_record_count")
        != sum(row.split == "heldout" for row in records)
        or len(records) != v3_binding["batch_count"]
        or canonical_sha256([row.v3_batch_audit_sha256 for row in records])
        != v3_binding["ordered_batch_audit_sha256s_sha256"]
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal request operator, subset, or record binding is inconsistent."
        )
    if expected_first_divergence_batch_results is not None:
        _batches, expected_records, expected_manifest_sha256 = (
            authenticate_v3_batch_identity_manifest(
                expected_first_divergence_request,
                expected_first_divergence_ledger,
                expected_first_divergence_batch_results,
            )
        )
        if (
            records != expected_records
            or request["authenticated_v3_record_manifest_sha256"]
            != expected_manifest_sha256
        ):
            raise Lfm2Exact4xCausalDiagnosticError(
                "Causal request record identities differ from authenticated v3 batches."
            )
    metric = request.get("metric_contract")
    bootstrap = request.get("bootstrap_contract")
    policy = request.get("policy")
    if metric != {
        "raw_metrics": [
            TERMINAL_ERROR_METRIC,
            DEFINED_VOCAB_KL_METRIC,
            PARENT_WINNER_MARGIN_METRIC,
        ],
        "higher_is_better_benefits": [
            TERMINAL_ERROR_BENEFIT,
            DEFINED_VOCAB_KL_BENEFIT,
            PARENT_WINNER_MARGIN_BENEFIT,
        ],
        "defined_semantic_row_count": v3_binding["defined_semantic_row_count"],
        "parent_physical_row_capacity": v3_binding["parent_physical_row_capacity"],
        "undefined_trailing_capacity_rows_excluded_from_kl": True,
        "top1_is_descriptive_only": True,
    }:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal request metric contract is invalid."
        )
    if (
        not isinstance(bootstrap, Mapping)
        or set(bootstrap)
        != {
            "cluster_unit",
            "confidence_level",
            "confidence_bound_contract",
            "record_total_mass",
            "resample_count",
            "seed",
            "seed_sha256",
            "split",
            "within_record_row_mass",
        }
        or not isinstance(bootstrap.get("seed"), str)
        or not bootstrap["seed"]
        or bootstrap.get("seed_sha256")
        != hashlib.sha256(bootstrap["seed"].encode("utf-8")).hexdigest()
        or _positive_int(bootstrap.get("resample_count"), "bootstrap resample_count")
        != bootstrap["resample_count"]
        or not 0.0
        < _finite(bootstrap.get("confidence_level"), "bootstrap confidence_level")
        < 1.0
        or bootstrap.get("cluster_unit") != "whole-record"
        or bootstrap.get("record_total_mass") != "equal"
        or bootstrap.get("within_record_row_mass") != "uniform"
        or bootstrap.get("split") != "heldout"
        or bootstrap.get("confidence_bound_contract")
        != "explicit-one-sided-lower-and-upper-plus-central-two-sided"
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal request bootstrap contract is invalid."
        )
    expected_policy = {
        "sealed_cached_parent_evidence_only": True,
        "parent_forward_calls": 0,
        "child_forward_only": True,
        "gradient_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "iterative_optimization_allowed": False,
        "global_residual_map_allowed": False,
        "exact_three_player_shapley": True,
        "recursive_harsanyi": True,
        "numeric_metrics_are_diagnostic_only": True,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "missing_transport_coordinates_must_not_be_invented": True,
    }
    if (
        policy != expected_policy
        or request.get("checkpoint_hmac_algorithm") != CHECKPOINT_HMAC_ALGORITHM
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal request execution policy is invalid."
        )
    _identifier(request.get("checkpoint_hmac_key_id"), "checkpoint_hmac_key_id")
    _digest(
        request.get("implementation_identity_sha256"),
        "implementation_identity_sha256",
    )
    availability = transported_feature_availability(
        request.get("transported_feature_lineage"),
        operator_specs=specs,
        trace_cache_key=v3_binding["trace_cache_key"],
        trace_selective_session_sha256=v3_binding["trace_selective_session_sha256"],
        basis_map_sha256=v3_binding["basis_map_sha256"],
    )
    if request.get("transported_feature_availability") != availability:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal request transport availability is inconsistent."
        )
    return request


def _canonical_subset_metric_rows(
    subset_metrics: Mapping[Any, Mapping[str, Any]],
    group_ids: Sequence[str],
) -> list[dict[str, Any]]:
    groups = tuple(group_ids)
    benefits = higher_is_better_benefits(subset_metrics, groups)
    raw_by_coalition = {
        _coalition_key(key, groups): {
            name: _finite(value[name], f"subset metric {name}")
            for name in (
                TERMINAL_ERROR_METRIC,
                DEFINED_VOCAB_KL_METRIC,
                PARENT_WINNER_MARGIN_METRIC,
            )
        }
        for key, value in subset_metrics.items()
    }
    return [
        {
            **row,
            "metrics": raw_by_coalition[tuple(row["members"])],
            "benefits": {
                name: values[tuple(row["members"])] for name, values in benefits.items()
            },
        }
        for row in canonical_intervention_subsets(groups)
    ]


def _canonical_oracle_evidence(
    value: Mapping[str, Any] | None,
    *,
    request_sha256: str,
    batch_index: int,
    record: Lfm2CausalRecordBinding,
) -> dict[str, Any]:
    """Accept no oracle claims, or the one explicit live-feature receipt shape."""

    oracle = {} if value is None else _plain_json(dict(value), "oracle evidence")
    if not isinstance(oracle, dict):
        raise Lfm2Exact4xCausalDiagnosticError("oracle_evidence must be a mapping.")
    if not oracle:
        return oracle
    expected = {
        "all_prefill_and_decode_rows_intervened",
        "feature_receipt",
        "full_trajectory_intervention_row_count",
        "live_feature_consumer_oracle_capture_available",
        "producer_mismatch_classification",
        "selected_parent_logical_bytes",
        "selected_parent_payload_sha256",
        "semantic_metric_row_count",
        "semantic_positions_used_only_for_metrics",
        "transported_feature_oracle_evaluated",
    }
    receipt = oracle.get("feature_receipt")
    if (
        set(oracle) != expected
        or not isinstance(receipt, Mapping)
        or receipt.get("request_sha256") != request_sha256
        or receipt.get("batch_index") != batch_index
        or receipt.get("trace_input_sha256") != record.trace_input_sha256
        or receipt.get("trace_record_sha256") != record.trace_record_sha256
        or _digest(receipt.get("receipt_sha256"), "oracle feature receipt_sha256")
        != receipt["receipt_sha256"]
        or _digest(
            oracle.get("selected_parent_payload_sha256"),
            "selected_parent_payload_sha256",
        )
        != oracle["selected_parent_payload_sha256"]
        or _positive_int(
            oracle.get("selected_parent_logical_bytes"),
            "selected_parent_logical_bytes",
        )
        != oracle["selected_parent_logical_bytes"]
        or _positive_int(
            oracle.get("full_trajectory_intervention_row_count"),
            "full_trajectory_intervention_row_count",
        )
        != oracle["full_trajectory_intervention_row_count"]
        or _positive_int(
            oracle.get("semantic_metric_row_count"),
            "semantic_metric_row_count",
        )
        != oracle["semantic_metric_row_count"]
        or oracle["semantic_metric_row_count"]
        > oracle["full_trajectory_intervention_row_count"]
        or oracle.get("all_prefill_and_decode_rows_intervened") is not True
        or oracle.get("semantic_positions_used_only_for_metrics") is not True
        or oracle.get("live_feature_consumer_oracle_capture_available") is not True
        or oracle.get("transported_feature_oracle_evaluated") is not False
        or oracle.get("producer_mismatch_classification") is not None
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Oracle evidence has a noncanonical or unauthenticated field inventory."
        )
    return oracle


def build_lfm2_exact4x_causal_batch_result(
    request: Mapping[str, Any],
    *,
    batch_index: int,
    subset_metrics: Mapping[Any, Mapping[str, Any]],
    child_forward_calls: int,
    oracle_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal one record's eight-subset metrics and exact decompositions."""

    authenticated = authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    index = _nonnegative_int(batch_index, "batch_index")
    records = canonical_record_bindings(authenticated["record_bindings"])
    if index >= len(records):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch index is outside the requested record manifest."
        )
    record = records[index]
    calls = _positive_int(child_forward_calls, "child_forward_calls")
    group_ids = tuple(authenticated["group_ids"])
    rows = _canonical_subset_metric_rows(subset_metrics, group_ids)
    benefits_by_name = {
        name: {tuple(row["members"]): row["benefits"][name] for row in rows}
        for name in (
            TERMINAL_ERROR_BENEFIT,
            DEFINED_VOCAB_KL_BENEFIT,
            PARENT_WINNER_MARGIN_BENEFIT,
        )
    }
    decompositions = {
        name: three_player_causal_decomposition(values, group_ids)
        for name, values in benefits_by_name.items()
    }
    availability = authenticated["transported_feature_availability"]
    if availability["transported_feature_oracle_available"] is False and (
        isinstance(oracle_evidence, Mapping)
        and (
            oracle_evidence.get("transported_feature_oracle_evaluated") is True
            or oracle_evidence.get("producer_mismatch_classification") is not None
        )
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Transported-feature claims require an authenticated transport lineage."
        )
    oracle = _canonical_oracle_evidence(
        oracle_evidence,
        request_sha256=authenticated["request_sha256"],
        batch_index=index,
        record=record,
    )
    body = {
        "schema": LFM2_EXACT4X_CAUSAL_BATCH_RESULT_SCHEMA,
        "request_sha256": authenticated["request_sha256"],
        "batch_index": index,
        "record_id": record.record_id,
        "trace_input_sha256": record.trace_input_sha256,
        "trace_record_sha256": record.trace_record_sha256,
        "split": record.split,
        "v3_batch_audit_sha256": record.v3_batch_audit_sha256,
        "subset_metrics": rows,
        "decompositions": decompositions,
        "oracle_evidence": oracle,
        "numeric_acceptance_decision": None,
        "numeric_metrics_are_diagnostic_only": True,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "operation_audit": {
            "parent_forward_calls": 0,
            "child_forward_calls": calls,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "persistent_model_mutations": 0,
        },
    }
    return {**body, "batch_result_sha256": canonical_sha256(body)}


def authenticate_lfm2_exact4x_causal_batch_result(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    authenticated_request = authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    result = _sealed(
        value,
        "batch_result_sha256",
        LFM2_EXACT4X_CAUSAL_BATCH_RESULT_SCHEMA,
    )
    expected = {
        "batch_index",
        "batch_result_sha256",
        "decompositions",
        "manual_whole_response_intelligence_review_is_authoritative",
        "numeric_acceptance_decision",
        "numeric_metrics_are_diagnostic_only",
        "operation_audit",
        "oracle_evidence",
        "record_id",
        "request_sha256",
        "schema",
        "split",
        "subset_metrics",
        "trace_input_sha256",
        "trace_record_sha256",
        "v3_batch_audit_sha256",
    }
    index = result.get("batch_index")
    records = canonical_record_bindings(authenticated_request["record_bindings"])
    if (
        set(result) != expected
        or isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(records)
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch result identity is noncanonical."
        )
    record = records[index]
    if (
        result.get("request_sha256") != authenticated_request["request_sha256"]
        or result.get("record_id") != record.record_id
        or result.get("trace_input_sha256") != record.trace_input_sha256
        or result.get("trace_record_sha256") != record.trace_record_sha256
        or result.get("split") != record.split
        or result.get("v3_batch_audit_sha256") != record.v3_batch_audit_sha256
        or result.get("numeric_acceptance_decision") is not None
        or result.get("numeric_metrics_are_diagnostic_only") is not True
        or result.get("manual_whole_response_intelligence_review_is_authoritative")
        is not True
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch result differs from its request record."
        )
    rows = result.get("subset_metrics")
    if not isinstance(rows, list):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch subset metrics must be a list."
        )
    groups = tuple(authenticated_request["group_ids"])
    canonical = canonical_intervention_subsets(groups)
    if len(rows) != len(canonical):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch must contain all eight subset metrics."
        )
    subset_metrics: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row, subset in zip(rows, canonical, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"benefits", "mask", "members", "metrics", "subset_id"}
            or {key: row[key] for key in ("mask", "members", "subset_id")} != subset
            or not isinstance(row.get("metrics"), Mapping)
        ):
            raise Lfm2Exact4xCausalDiagnosticError(
                "Causal subset metric row is noncanonical."
            )
        subset_metrics[tuple(subset["members"])] = row["metrics"]
    rebuilt_rows = _canonical_subset_metric_rows(subset_metrics, groups)
    benefits_by_name = {
        name: {tuple(row["members"]): row["benefits"][name] for row in rebuilt_rows}
        for name in (
            TERMINAL_ERROR_BENEFIT,
            DEFINED_VOCAB_KL_BENEFIT,
            PARENT_WINNER_MARGIN_BENEFIT,
        )
    }
    expected_decompositions = {
        name: three_player_causal_decomposition(values, groups)
        for name, values in benefits_by_name.items()
    }
    if rows != rebuilt_rows or result.get("decompositions") != expected_decompositions:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch benefits or decompositions do not reconstruct."
        )
    operation = result.get("operation_audit")
    if (
        not isinstance(operation, Mapping)
        or set(operation)
        != {
            "backward_calls",
            "child_forward_calls",
            "optimizer_steps",
            "parent_forward_calls",
            "persistent_model_mutations",
            "training_steps",
        }
        or operation.get("parent_forward_calls") != 0
        or _positive_int(
            operation.get("child_forward_calls"), "batch child_forward_calls"
        )
        != operation["child_forward_calls"]
        or operation.get("training_steps") != 0
        or operation.get("backward_calls") != 0
        or operation.get("optimizer_steps") != 0
        or operation.get("persistent_model_mutations") != 0
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch operation audit violates inference-only execution."
        )
    oracle = result.get("oracle_evidence")
    if not isinstance(oracle, Mapping):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch oracle evidence must be a mapping."
        )
    availability = authenticated_request["transported_feature_availability"]
    if availability["transported_feature_oracle_available"] is False and (
        oracle.get("transported_feature_oracle_evaluated") is True
        or oracle.get("producer_mismatch_classification") is not None
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Unauthenticated transported-feature claims are forbidden."
        )
    if (
        _canonical_oracle_evidence(
            oracle,
            request_sha256=authenticated_request["request_sha256"],
            batch_index=index,
            record=record,
        )
        != oracle
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch oracle evidence does not reconstruct."
        )
    return result


class Lfm2CausalCheckpointAuthority:
    """Runtime-only, domain-separated HMAC authority for v4 batch checkpoints."""

    __slots__ = ("__secret", "key_id")

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        self.key_id = _identifier(key_id, "checkpoint HMAC key_id")
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise Lfm2Exact4xCausalDiagnosticError(
                "Checkpoint HMAC secret must contain at least 32 runtime bytes."
            )
        self.__secret = bytes(secret)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key_id={self.key_id!r})"

    def __reduce_ex__(self, _protocol: int) -> Any:
        raise TypeError("Lfm2CausalCheckpointAuthority cannot be serialized.")

    def _sign_domain(self, body: Mapping[str, Any], domain: bytes) -> str:
        return hmac.new(
            self.__secret,
            domain + canonical_json_bytes(dict(body)),
            hashlib.sha256,
        ).hexdigest()

    def sign(self, body: Mapping[str, Any]) -> str:
        return self._sign_domain(body, _CHECKPOINT_HMAC_DOMAIN)

    def verify(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return (
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign(body), claimed)
        )

    def sign_ledger(self, body: Mapping[str, Any]) -> str:
        return self._sign_domain(body, _LEDGER_HMAC_DOMAIN)

    def verify_ledger(self, body: Mapping[str, Any], claimed: Any) -> bool:
        return (
            isinstance(claimed, str)
            and _SHA256.fullmatch(claimed) is not None
            and hmac.compare_digest(self.sign_ledger(body), claimed)
        )


def build_lfm2_exact4x_causal_batch_checkpoint(
    request: Mapping[str, Any],
    batch_result: Mapping[str, Any],
    *,
    authority: Lfm2CausalCheckpointAuthority,
) -> dict[str, Any]:
    authenticated_request = authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    if (
        not isinstance(authority, Lfm2CausalCheckpointAuthority)
        or authority.key_id != authenticated_request["checkpoint_hmac_key_id"]
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Checkpoint authority differs from the causal request."
        )
    result = authenticate_lfm2_exact4x_causal_batch_result(
        batch_result,
        request=authenticated_request,
    )
    body = {
        "schema": LFM2_EXACT4X_CAUSAL_BATCH_CHECKPOINT_SCHEMA,
        "request_sha256": authenticated_request["request_sha256"],
        "batch_index": result["batch_index"],
        "record_id": result["record_id"],
        "trace_input_sha256": result["trace_input_sha256"],
        "trace_record_sha256": result["trace_record_sha256"],
        "split": result["split"],
        "batch_result_sha256": result["batch_result_sha256"],
        "checkpoint_hmac_algorithm": CHECKPOINT_HMAC_ALGORITHM,
        "checkpoint_hmac_key_id": authority.key_id,
        "result": result,
    }
    sealed = {**body, "checkpoint_sha256": canonical_sha256(body)}
    return {**sealed, "checkpoint_hmac_sha256": authority.sign(sealed)}


def authenticate_lfm2_exact4x_causal_batch_checkpoint(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    authority: Lfm2CausalCheckpointAuthority,
) -> dict[str, Any]:
    authenticated_request = authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    if (
        not isinstance(authority, Lfm2CausalCheckpointAuthority)
        or authority.key_id != authenticated_request["checkpoint_hmac_key_id"]
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Checkpoint authority differs from the causal request."
        )
    row = _plain_json(dict(value), "causal batch checkpoint")
    if not isinstance(row, dict):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch checkpoint must be a mapping."
        )
    claimed_hmac = row.pop("checkpoint_hmac_sha256", None)
    claimed_sha = row.pop("checkpoint_sha256", None)
    expected = {
        "batch_index",
        "batch_result_sha256",
        "checkpoint_hmac_algorithm",
        "checkpoint_hmac_key_id",
        "record_id",
        "request_sha256",
        "result",
        "schema",
        "split",
        "trace_input_sha256",
        "trace_record_sha256",
    }
    if (
        set(row) != expected
        or row.get("schema") != LFM2_EXACT4X_CAUSAL_BATCH_CHECKPOINT_SCHEMA
        or row.get("request_sha256") != authenticated_request["request_sha256"]
        or row.get("checkpoint_hmac_algorithm") != CHECKPOINT_HMAC_ALGORITHM
        or row.get("checkpoint_hmac_key_id") != authority.key_id
        or claimed_sha != canonical_sha256(row)
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch checkpoint differs from its request or seal."
        )
    sealed = {**row, "checkpoint_sha256": claimed_sha}
    if not authority.verify(sealed, claimed_hmac):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal batch checkpoint HMAC does not verify."
        )
    result_value = row.get("result")
    if not isinstance(result_value, Mapping):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal checkpoint result must be a mapping."
        )
    result = authenticate_lfm2_exact4x_causal_batch_result(
        result_value,
        request=authenticated_request,
    )
    for field in (
        "batch_index",
        "batch_result_sha256",
        "record_id",
        "split",
        "trace_input_sha256",
        "trace_record_sha256",
    ):
        if row.get(field) != result[field]:
            raise Lfm2Exact4xCausalDiagnosticError(
                "Causal checkpoint result identity does not verify."
            )
    return {**sealed, "checkpoint_hmac_sha256": claimed_hmac}


def authenticate_lfm2_exact4x_causal_resume(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    request: Mapping[str, Any],
    authority: Lfm2CausalCheckpointAuthority,
) -> tuple[dict[str, Any], ...]:
    """Validate a partial canonical checkpoint prefix for safe resume."""

    rows = tuple(
        authenticate_lfm2_exact4x_causal_batch_checkpoint(
            value,
            request=request,
            authority=authority,
        )
        for value in checkpoints
    )
    if tuple(row["batch_index"] for row in rows) != tuple(range(len(rows))):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Resume checkpoints must form one unique canonical batch prefix."
        )
    return rows


def _aggregate_lfm2_exact4x_causal_authenticated(
    request: Mapping[str, Any],
    batch_results: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    authority: Lfm2CausalCheckpointAuthority,
) -> dict[str, Any]:
    """Aggregate already-HMAC-authenticated checkpoints without a numeric gate."""

    authenticated_request = authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    if (
        not isinstance(authority, Lfm2CausalCheckpointAuthority)
        or authority.key_id != authenticated_request["checkpoint_hmac_key_id"]
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Ledger authority differs from the causal request."
        )
    results = tuple(
        authenticate_lfm2_exact4x_causal_batch_result(
            value,
            request=authenticated_request,
        )
        for value in batch_results
    )
    checkpoint_rows = tuple(checkpoints)
    if (
        len(results) != authenticated_request["record_count"]
        or len(checkpoint_rows) != len(results)
        or tuple(result["batch_index"] for result in results)
        != tuple(range(len(results)))
        or any(
            checkpoint["result"]["batch_result_sha256"] != result["batch_result_sha256"]
            for checkpoint, result in zip(checkpoint_rows, results, strict=True)
        )
    ):
        raise Lfm2Exact4xCausalDiagnosticError(
            "Final causal ledger requires every requested batch in canonical order."
        )
    groups = tuple(authenticated_request["group_ids"])
    subsets = canonical_intervention_subsets(groups)
    benefit_names = (
        TERMINAL_ERROR_BENEFIT,
        DEFINED_VOCAB_KL_BENEFIT,
        PARENT_WINNER_MARGIN_BENEFIT,
    )
    raw_names = (
        TERMINAL_ERROR_METRIC,
        DEFINED_VOCAB_KL_METRIC,
        PARENT_WINNER_MARGIN_METRIC,
    )
    split_summaries: dict[str, Any] = {}
    for split in SPLITS:
        selected = tuple(result for result in results if result["split"] == split)
        raw_means: dict[tuple[str, ...], dict[str, float]] = {}
        for subset_index, subset in enumerate(subsets):
            coalition = tuple(subset["members"])
            raw_means[coalition] = {
                metric: math.fsum(
                    float(result["subset_metrics"][subset_index]["metrics"][metric])
                    for result in selected
                )
                / len(selected)
                for metric in raw_names
            }
        benefit_maps = higher_is_better_benefits(raw_means, groups)
        split_summaries[split] = {
            "batch_count": len(selected),
            "record_count": len({result["record_id"] for result in selected}),
            "mean_subset_metrics": _canonical_subset_metric_rows(raw_means, groups),
            "decompositions": {
                name: three_player_causal_decomposition(values, groups)
                for name, values in benefit_maps.items()
            },
            "numeric_acceptance_decision": None,
        }
    heldout = tuple(result for result in results if result["split"] == "heldout")
    bootstrap_contract = authenticated_request["bootstrap_contract"]
    bootstrap_rows = []
    for benefit_name in benefit_names:
        for group in groups:
            values = []
            for result in heldout:
                shapley_rows = result["decompositions"][benefit_name]["shapley_values"]
                by_group = {row["group_id"]: row["value"] for row in shapley_rows}
                values.append(float(by_group[group]))
            receipt = deterministic_record_clustered_bootstrap(
                values,
                [result["record_id"] for result in heldout],
                seed=(
                    f"{bootstrap_contract['seed']}\0shapley\0{benefit_name}\0{group}"
                ),
                resample_count=int(bootstrap_contract["resample_count"]),
                confidence_level=float(bootstrap_contract["confidence_level"]),
            )
            bootstrap_rows.append(
                {
                    "statistic": "shapley_value",
                    "benefit_metric": benefit_name,
                    "group_id": group,
                    "bootstrap": receipt,
                }
            )
    body = {
        "schema": LFM2_EXACT4X_CAUSAL_LEDGER_SCHEMA,
        "status": "complete-observation",
        "request": authenticated_request,
        "ordered_batch_checkpoint_authentication": [
            {
                "batch_index": checkpoint["batch_index"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_hmac_sha256": checkpoint["checkpoint_hmac_sha256"],
                "batch_result_sha256": checkpoint["batch_result_sha256"],
            }
            for checkpoint in checkpoint_rows
        ],
        "ordered_batch_result_sha256s": [
            result["batch_result_sha256"] for result in results
        ],
        "checkpoint_hmac_algorithm": CHECKPOINT_HMAC_ALGORITHM,
        "checkpoint_hmac_key_id": authority.key_id,
        "checkpoint_chain_authenticated": True,
        "batch_count": len(results),
        "split_summaries": split_summaries,
        "heldout_record_clustered_shapley_bootstraps": bootstrap_rows,
        "numeric_metrics_are_diagnostic_only": True,
        "numeric_acceptance_decision": None,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "operation_audit": {
            "parent_forward_calls": 0,
            "child_forward_calls": sum(
                int(result["operation_audit"]["child_forward_calls"])
                for result in results
            ),
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "persistent_model_mutations": 0,
        },
    }
    sealed = {**body, "ledger_sha256": canonical_sha256(body)}
    return {**sealed, "ledger_hmac_sha256": authority.sign_ledger(sealed)}


def aggregate_lfm2_exact4x_causal_checkpoints(
    request: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    authority: Lfm2CausalCheckpointAuthority,
) -> dict[str, Any]:
    """Publish a final ledger only from the complete authenticated HMAC chain."""

    authenticated = authenticate_lfm2_exact4x_causal_resume(
        checkpoints,
        request=request,
        authority=authority,
    )
    authenticated_request = authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    if len(authenticated) != authenticated_request["record_count"]:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Final causal ledger requires the complete HMAC checkpoint chain."
        )
    return _aggregate_lfm2_exact4x_causal_authenticated(
        authenticated_request,
        [checkpoint["result"] for checkpoint in authenticated],
        authenticated,
        authority=authority,
    )


def aggregate_lfm2_exact4x_causal_results(
    request: Mapping[str, Any],
    batch_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject the legacy plain-result publication path."""

    raise Lfm2Exact4xCausalDiagnosticError(
        "Plain batch results cannot publish a causal ledger; complete authenticated "
        "HMAC checkpoints and their runtime authority are required."
    )


def authenticate_lfm2_exact4x_causal_ledger(
    value: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    authority: Lfm2CausalCheckpointAuthority,
) -> dict[str, Any]:
    """Authenticate the final ledger and its complete checkpoint HMAC chain."""

    authenticated_request = authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    authenticated_checkpoints = authenticate_lfm2_exact4x_causal_resume(
        checkpoints,
        request=authenticated_request,
        authority=authority,
    )
    submitted = _plain_json(dict(value), "causal ledger")
    if not isinstance(submitted, dict):
        raise Lfm2Exact4xCausalDiagnosticError("Causal ledger must be a mapping.")
    expected = _aggregate_lfm2_exact4x_causal_authenticated(
        authenticated_request,
        [checkpoint["result"] for checkpoint in authenticated_checkpoints],
        authenticated_checkpoints,
        authority=authority,
    )
    if submitted != expected:
        raise Lfm2Exact4xCausalDiagnosticError(
            "Causal ledger or its complete checkpoint HMAC chain does not verify."
        )
    return expected


__all__ = [
    "CAUSAL_GROUP_COUNT",
    "CHECKPOINT_HMAC_ALGORITHM",
    "DEFINED_VOCAB_KL_BENEFIT",
    "DEFINED_VOCAB_KL_METRIC",
    "LFM2_EXACT4X_CAUSAL_BATCH_CHECKPOINT_SCHEMA",
    "LFM2_EXACT4X_CAUSAL_BATCH_RESULT_SCHEMA",
    "LFM2_EXACT4X_CAUSAL_DIAGNOSTIC_REQUEST_SCHEMA",
    "LFM2_EXACT4X_CAUSAL_LEDGER_SCHEMA",
    "LFM2_EXACT4X_CAUSAL_OPERATOR_SPEC_SCHEMA",
    "LFM2_EXACT4X_CAUSAL_RECORD_BINDING_SCHEMA",
    "LFM2_EXACT4X_CAUSAL_V3_BINDING_SCHEMA",
    "LFM2_EXACT4X_CLUSTERED_BOOTSTRAP_SCHEMA",
    "LFM2_EXACT4X_TRANSPORTED_FEATURE_AVAILABILITY_SCHEMA",
    "LFM2_EXACT4X_TRANSPORTED_FEATURE_LINEAGE_SCHEMA",
    "Lfm2CausalCheckpointAuthority",
    "Lfm2CausalOperatorSpec",
    "Lfm2CausalRecordBinding",
    "Lfm2Exact4xCausalDiagnosticError",
    "Lfm2TransportedFeatureBinding",
    "PARENT_WINNER_MARGIN_BENEFIT",
    "PARENT_WINNER_MARGIN_METRIC",
    "TERMINAL_ERROR_BENEFIT",
    "TERMINAL_ERROR_METRIC",
    "aggregate_defined_vocab_kl",
    "aggregate_lfm2_exact4x_causal_checkpoints",
    "aggregate_lfm2_exact4x_causal_results",
    "authenticate_lfm2_exact4x_causal_batch_checkpoint",
    "authenticate_lfm2_exact4x_causal_batch_result",
    "authenticate_lfm2_exact4x_causal_diagnostic_request",
    "authenticate_lfm2_exact4x_causal_ledger",
    "authenticate_lfm2_exact4x_causal_resume",
    "authenticate_lfm2_transported_feature_lineage",
    "authenticate_v3_first_divergence_binding",
    "authenticate_v3_batch_identity_manifest",
    "authenticate_v3_first_divergence_ledger",
    "build_lfm2_exact4x_causal_batch_checkpoint",
    "build_lfm2_exact4x_causal_batch_result",
    "build_lfm2_exact4x_causal_diagnostic_request",
    "build_lfm2_transported_feature_lineage",
    "build_v3_first_divergence_binding",
    "canonical_intervention_subsets",
    "canonical_operator_specs",
    "canonical_record_bindings",
    "default_lfm2_l0_csm_operator_specs",
    "defined_vocab_kl_rows",
    "deterministic_record_clustered_bootstrap",
    "equal_record_mass_weights",
    "exact_three_player_shapley",
    "higher_is_better_benefits",
    "recursive_harsanyi_dividends",
    "solve_lfm2_trace_scaled_weighted_anchored_ridge",
    "three_player_causal_decomposition",
    "transported_feature_availability",
]
