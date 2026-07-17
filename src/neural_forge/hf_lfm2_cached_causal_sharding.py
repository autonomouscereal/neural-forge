"""Pure provenance contracts for N-way cached-causal LFM2 sharding.

This module deliberately contains no model, tokenizer, Torch, or producer
execution code.  It partitions whole LFM2 layers into deterministic contiguous
shards, seals the common run contract and per-shard ownership, authenticates
the already sealed candidate-record files, and validates the exact inventory
that a single frontier merger is allowed to consume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


RUN_CONTRACT_SCHEMA = "neural-forge-lfm2-cached-causal-run-contract-v1"
SHARD_SPEC_SCHEMA = "neural-forge-lfm2-cached-causal-shard-spec-v1"
SHARD_MANIFEST_SCHEMA = "neural-forge-lfm2-cached-causal-shard-manifest-v1"
MERGE_INVENTORY_SCHEMA = "neural-forge-lfm2-cached-causal-merge-inventory-v1"
OWNERSHIP_ALGORITHM = "contiguous-layer-floor-v1"
WEIGHTED_OWNERSHIP_ALGORITHM = "contiguous-layer-weighted-largest-remainder-v1"
CANDIDATE_RECORD_SCHEMA = "neural-forge-lfm2-cached-causal-candidate-record-v1"

_OPTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_ZERO_AUDIT_FIELDS = (
    "parent_forward_calls",
    "parent_generation_calls",
    "optimizer_steps",
    "backward_calls",
)
_RUN_ZERO_AUDIT_FIELDS = (
    "parent_forward_calls",
    "parent_cache_build_calls",
    "parent_generation_calls",
    "optimizer_steps",
    "backward_calls",
)
_PARENT_DIGEST_FIELDS = (
    "parent_identity_sha256",
    "parent_trace_cache_key",
    "parent_corpus_sha256",
    "parent_trace_rows_sha256",
    "parent_output_envelope_sha256",
    "parent_output_file_sha256",
    "tokenizer_fingerprint",
)
_EVIDENCE_DIGEST_FIELDS = (
    "split_sha256",
    "task_anchor_manifest_sha256",
    "variable_width_evidence_sha256",
)


class Lfm2CachedCausalShardingError(RuntimeError):
    """A shard contract, record inventory, or merge inventory is unsafe."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CachedCausalShardingError(
            "Shard provenance must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedCausalShardingError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CachedCausalShardingError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CachedCausalShardingError(f"{name} must be a non-negative integer.")
    return value


def _finite_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CachedCausalShardingError(
            f"{name} must be strictly between zero and one."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CachedCausalShardingError(
            f"{name} must be strictly between zero and one."
        ) from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise Lfm2CachedCausalShardingError(
            f"{name} must be strictly between zero and one."
        )
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CachedCausalShardingError(f"{name} must be a mapping.")
    # Round-tripping also rejects non-JSON keys and mutable exotic values.
    return json.loads(_canonical_json(dict(value)).decode("ascii"))


def _option_id(value: Any, name: str = "option_id") -> str:
    if not isinstance(value, str) or _OPTION_ID.fullmatch(value) is None:
        raise Lfm2CachedCausalShardingError(
            f"{name} must be a portable filename-safe identifier."
        )
    return value


def _normalize_widths(
    widths_by_layer: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    if isinstance(widths_by_layer, (str, bytes)) or not isinstance(
        widths_by_layer, Sequence
    ):
        raise Lfm2CachedCausalShardingError(
            "widths_by_layer must be a nonempty sequence."
        )
    result = []
    for layer_index, raw_widths in enumerate(widths_by_layer):
        if isinstance(raw_widths, (str, bytes)) or not isinstance(raw_widths, Sequence):
            raise Lfm2CachedCausalShardingError(
                f"Layer {layer_index} widths must be a sequence."
            )
        widths = tuple(raw_widths)
        if (
            len(widths) < 2
            or any(
                isinstance(width, bool) or not isinstance(width, int) or width <= 0
                for width in widths
            )
            or any(left <= right for left, right in zip(widths, widths[1:]))
        ):
            raise Lfm2CachedCausalShardingError(
                "Every layer needs a strictly descending source/target frontier."
            )
        result.append(widths)
    if not result:
        raise Lfm2CachedCausalShardingError(
            "widths_by_layer must cover at least one layer."
        )
    return tuple(result)


def _normalize_candidate_inventory(
    inventory: Sequence[Mapping[str, Any]],
    *,
    widths_by_layer: Sequence[Sequence[int]],
) -> tuple[dict[str, Any], ...]:
    if isinstance(inventory, (str, bytes)) or not isinstance(inventory, Sequence):
        raise Lfm2CachedCausalShardingError("candidate_inventory must be a sequence.")
    widths = _normalize_widths(widths_by_layer)
    normalized = []
    seen_ids: set[str] = set()
    seen_coordinates: set[tuple[int, int]] = set()
    for raw in inventory:
        value = _mapping(raw, "candidate_inventory entry")
        option_id = _option_id(value.get("option_id"))
        layer_index = _nonnegative_int(value.get("layer_index"), "layer_index")
        target_width = _positive_int(value.get("target_width"), "target_width")
        proposal_sha256 = _digest(
            value.get("proposal_sha256"),
            "proposal_sha256",
        )
        canonical_pack_manifest_sha256 = _digest(
            value.get("canonical_pack_manifest_sha256"),
            "canonical_pack_manifest_sha256",
        )
        canonical_pack_tensor_file_sha256 = _digest(
            value.get("canonical_pack_tensor_file_sha256"),
            "canonical_pack_tensor_file_sha256",
        )
        canonical_bundle_sha256 = _digest(
            value.get("canonical_bundle_sha256"),
            "canonical_bundle_sha256",
        )
        canonical_layer_binding_sha256 = _digest(
            value.get("canonical_layer_binding_sha256"),
            "canonical_layer_binding_sha256",
        )
        if value.get("canonical_payload_required") is not True:
            raise Lfm2CachedCausalShardingError(
                "Every sharded candidate requires a canonical physical payload."
            )
        if (
            layer_index >= len(widths)
            or target_width not in widths[layer_index][1:]
            or option_id in seen_ids
            or (layer_index, target_width) in seen_coordinates
        ):
            raise Lfm2CachedCausalShardingError(
                "Candidate inventory is duplicated or outside the width frontier."
            )
        seen_ids.add(option_id)
        seen_coordinates.add((layer_index, target_width))
        normalized.append(
            {
                "option_id": option_id,
                "layer_index": layer_index,
                "target_width": target_width,
                "proposal_sha256": proposal_sha256,
                "canonical_pack_manifest_sha256": (canonical_pack_manifest_sha256),
                "canonical_pack_tensor_file_sha256": (
                    canonical_pack_tensor_file_sha256
                ),
                "canonical_bundle_sha256": canonical_bundle_sha256,
                "canonical_layer_binding_sha256": (canonical_layer_binding_sha256),
                "canonical_payload_required": True,
            }
        )
    expected_coordinates = {
        (layer_index, target_width)
        for layer_index, layer_widths in enumerate(widths)
        for target_width in layer_widths[1:]
    }
    if seen_coordinates != expected_coordinates:
        raise Lfm2CachedCausalShardingError(
            "Candidate inventory does not exactly cover every layer/width option."
        )
    return tuple(sorted(normalized, key=lambda value: value["option_id"]))


def lfm2_contiguous_layer_bounds(
    layer_count: int,
    shard_count: int,
    shard_index: int,
    *,
    shard_weights: Sequence[int] | None = None,
) -> tuple[int, int]:
    """Return deterministic half-open whole-layer bounds for one shard."""

    layers = _positive_int(layer_count, "layer_count")
    shards = _positive_int(shard_count, "shard_count")
    index = _nonnegative_int(shard_index, "shard_index")
    if shards > layers:
        raise Lfm2CachedCausalShardingError(
            "shard_count cannot exceed layer_count; empty shards are forbidden."
        )
    if index >= shards:
        raise Lfm2CachedCausalShardingError(
            "shard_index must be smaller than shard_count."
        )
    allocations = lfm2_contiguous_layer_allocations(
        layers,
        shards,
        shard_weights=shard_weights,
    )
    start = sum(allocations[:index])
    stop = start + allocations[index]
    if not start < stop:
        raise Lfm2CachedCausalShardingError(
            "Contiguous ownership unexpectedly produced an empty shard."
        )
    return start, stop


def lfm2_contiguous_layer_allocations(
    layer_count: int,
    shard_count: int,
    *,
    shard_weights: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Apportion whole layers, optionally by sealed positive integer weights.

    Equal ownership preserves the original floor-boundary partition.  Weighted
    ownership uses Hamilton/largest-remainder apportionment with shard index as
    the deterministic tie-breaker.
    """

    layers = _positive_int(layer_count, "layer_count")
    shards = _positive_int(shard_count, "shard_count")
    if shards > layers:
        raise Lfm2CachedCausalShardingError(
            "shard_count cannot exceed layer_count; empty shards are forbidden."
        )
    if shard_weights is None:
        return tuple(
            (index + 1) * layers // shards - index * layers // shards
            for index in range(shards)
        )
    if isinstance(shard_weights, (str, bytes)) or not isinstance(
        shard_weights,
        Sequence,
    ):
        raise Lfm2CachedCausalShardingError(
            "shard_weights must be a positive integer sequence."
        )
    weights = tuple(shard_weights)
    if len(weights) != shards or any(
        isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
        for weight in weights
    ):
        raise Lfm2CachedCausalShardingError(
            "shard_weights must contain one positive integer per shard."
        )
    total_weight = sum(weights)
    allocations = [layers * weight // total_weight for weight in weights]
    remainders = [(layers * weight) % total_weight for weight in weights]
    remaining = layers - sum(allocations)
    order = sorted(
        range(shards),
        key=lambda index: (-remainders[index], index),
    )
    for index in order[:remaining]:
        allocations[index] += 1
    if any(allocation <= 0 for allocation in allocations):
        raise Lfm2CachedCausalShardingError(
            "Weighted ownership produced an empty shard; adjust shard weights."
        )
    return tuple(allocations)


def lfm2_contiguous_layer_indices(
    layer_count: int,
    shard_count: int,
    shard_index: int,
    *,
    shard_weights: Sequence[int] | None = None,
) -> tuple[int, ...]:
    start, stop = lfm2_contiguous_layer_bounds(
        layer_count,
        shard_count,
        shard_index,
        shard_weights=shard_weights,
    )
    return tuple(range(start, stop))


def lfm2_contiguous_layer_owner(
    layer_count: int,
    shard_count: int,
    layer_index: int,
    *,
    shard_weights: Sequence[int] | None = None,
) -> int:
    layers = _positive_int(layer_count, "layer_count")
    shards = _positive_int(shard_count, "shard_count")
    layer = _nonnegative_int(layer_index, "layer_index")
    if shards > layers or layer >= layers:
        raise Lfm2CachedCausalShardingError(
            "Layer ownership inputs are outside the nonempty shard range."
        )
    allocations = lfm2_contiguous_layer_allocations(
        layers,
        shards,
        shard_weights=shard_weights,
    )
    stop = 0
    for index, allocation in enumerate(allocations):
        stop += allocation
        if layer < stop:
            return index
    raise AssertionError("validated layer ownership was not assigned")


def seal_lfm2_cached_causal_run_contract(
    *,
    parent_binding: Mapping[str, Any],
    evidence_binding: Mapping[str, Any],
    target_fraction: float,
    widths_by_layer: Sequence[Sequence[int]],
    candidate_inventory: Sequence[Mapping[str, Any]],
    measurement_contract: Mapping[str, Any],
    producer_implementation_sha256: str,
) -> dict[str, Any]:
    """Seal the shard-independent numerical and evidence contract."""

    parent = _mapping(parent_binding, "parent_binding")
    for field in _PARENT_DIGEST_FIELDS:
        parent[field] = _digest(parent.get(field), f"parent_binding.{field}")
    parent["parent_unique_parameters"] = _positive_int(
        parent.get("parent_unique_parameters"),
        "parent_binding.parent_unique_parameters",
    )
    evidence = _mapping(evidence_binding, "evidence_binding")
    profile = evidence.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise Lfm2CachedCausalShardingError(
            "evidence_binding.profile must be nonempty."
        )
    evidence["profile"] = profile.strip()
    for field in _EVIDENCE_DIGEST_FIELDS:
        evidence[field] = _digest(evidence.get(field), f"evidence_binding.{field}")
    widths = _normalize_widths(widths_by_layer)
    candidates = _normalize_candidate_inventory(
        candidate_inventory,
        widths_by_layer=widths,
    )
    measurement = _mapping(measurement_contract, "measurement_contract")
    dtype = measurement.get("dtype")
    trace_mode = measurement.get("decode_trace_mode")
    if (
        not isinstance(dtype, str)
        or not dtype.strip()
        or not isinstance(trace_mode, str)
        or not trace_mode.strip()
        or measurement.get("forced_decode_use_cache") is not False
    ):
        raise Lfm2CachedCausalShardingError(
            "Measurement contract requires dtype, trace mode, and uncached decode."
        )
    for field in (
        "rollout_batch_size",
        "cluster_max_iterations",
        "cluster_distance_chunk_size",
    ):
        measurement[field] = _positive_int(
            measurement.get(field),
            f"measurement_contract.{field}",
        )
    ridge = measurement.get("cluster_ridge")
    if isinstance(ridge, bool):
        raise Lfm2CachedCausalShardingError(
            "measurement_contract.cluster_ridge must be finite and non-negative."
        )
    try:
        ridge_value = float(ridge)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CachedCausalShardingError(
            "measurement_contract.cluster_ridge must be finite and non-negative."
        ) from exc
    if not math.isfinite(ridge_value) or ridge_value < 0.0:
        raise Lfm2CachedCausalShardingError(
            "measurement_contract.cluster_ridge must be finite and non-negative."
        )
    measurement["cluster_ridge"] = ridge_value
    body = {
        "schema": RUN_CONTRACT_SCHEMA,
        "producer_implementation_sha256": _digest(
            producer_implementation_sha256,
            "producer_implementation_sha256",
        ),
        "parent_binding": parent,
        "evidence_binding": evidence,
        "frontier": {
            "target_fraction": _finite_fraction(
                target_fraction,
                "target_fraction",
            ),
            "layer_count": len(widths),
            "widths_by_layer": [list(value) for value in widths],
            "candidate_inventory": list(candidates),
            "candidate_inventory_sha256": _json_sha256(list(candidates)),
            "candidate_count": len(candidates),
        },
        "measurement_contract": measurement,
        "forbidden_operations": {field: 0 for field in _RUN_ZERO_AUDIT_FIELDS},
    }
    return {**body, "run_contract_sha256": _json_sha256(body)}


def verify_lfm2_cached_causal_run_contract(
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _mapping(contract, "run_contract")
    seal = value.pop("run_contract_sha256", None)
    if value.get("schema") != RUN_CONTRACT_SCHEMA or _digest(
        seal,
        "run_contract_sha256",
    ) != _json_sha256(value):
        raise Lfm2CachedCausalShardingError("Run contract seal is invalid.")
    parent = _mapping(value.get("parent_binding"), "parent_binding")
    for field in _PARENT_DIGEST_FIELDS:
        _digest(parent.get(field), f"parent_binding.{field}")
    _positive_int(
        parent.get("parent_unique_parameters"),
        "parent_binding.parent_unique_parameters",
    )
    evidence = _mapping(value.get("evidence_binding"), "evidence_binding")
    if not isinstance(evidence.get("profile"), str) or not evidence["profile"].strip():
        raise Lfm2CachedCausalShardingError("Run contract evidence profile is invalid.")
    for field in _EVIDENCE_DIGEST_FIELDS:
        _digest(evidence.get(field), f"evidence_binding.{field}")
    frontier = _mapping(value.get("frontier"), "frontier")
    widths = _normalize_widths(frontier.get("widths_by_layer"))
    if frontier.get("layer_count") != len(widths):
        raise Lfm2CachedCausalShardingError(
            "Run contract layer count differs from its width frontier."
        )
    candidates = _normalize_candidate_inventory(
        frontier.get("candidate_inventory"),
        widths_by_layer=widths,
    )
    if frontier.get("candidate_count") != len(candidates) or frontier.get(
        "candidate_inventory_sha256"
    ) != _json_sha256(list(candidates)):
        raise Lfm2CachedCausalShardingError(
            "Run contract candidate inventory changed after sealing."
        )
    _finite_fraction(frontier.get("target_fraction"), "target_fraction")
    measurement = _mapping(
        value.get("measurement_contract"),
        "measurement_contract",
    )
    if (
        not isinstance(measurement.get("dtype"), str)
        or not measurement["dtype"].strip()
        or not isinstance(measurement.get("decode_trace_mode"), str)
        or not measurement["decode_trace_mode"].strip()
        or measurement.get("forced_decode_use_cache") is not False
    ):
        raise Lfm2CachedCausalShardingError(
            "Run contract measurement settings are invalid."
        )
    for field in (
        "rollout_batch_size",
        "cluster_max_iterations",
        "cluster_distance_chunk_size",
    ):
        _positive_int(
            measurement.get(field),
            f"measurement_contract.{field}",
        )
    ridge = measurement.get("cluster_ridge")
    if isinstance(ridge, bool):
        raise Lfm2CachedCausalShardingError("Run contract cluster ridge is invalid.")
    try:
        ridge_value = float(ridge)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CachedCausalShardingError(
            "Run contract cluster ridge is invalid."
        ) from exc
    if not math.isfinite(ridge_value) or ridge_value < 0.0:
        raise Lfm2CachedCausalShardingError("Run contract cluster ridge is invalid.")
    forbidden = _mapping(
        value.get("forbidden_operations"),
        "forbidden_operations",
    )
    if any(forbidden.get(field) != 0 for field in _RUN_ZERO_AUDIT_FIELDS):
        raise Lfm2CachedCausalShardingError(
            "Run contract authorizes a forbidden operation."
        )
    _digest(
        value.get("producer_implementation_sha256"),
        "producer_implementation_sha256",
    )
    return contract


def seal_lfm2_cached_causal_shard_spec(
    run_contract: Mapping[str, Any],
    *,
    shard_count: int,
    shard_index: int,
    shard_weights: Sequence[int] | None = None,
) -> dict[str, Any]:
    verify_lfm2_cached_causal_run_contract(run_contract)
    frontier = run_contract["frontier"]
    layer_count = int(frontier["layer_count"])
    weights = None if shard_weights is None else tuple(shard_weights)
    owned_layers = lfm2_contiguous_layer_indices(
        layer_count,
        shard_count,
        shard_index,
        shard_weights=weights,
    )
    owned_layer_set = set(owned_layers)
    owned = tuple(
        value
        for value in frontier["candidate_inventory"]
        if int(value["layer_index"]) in owned_layer_set
    )
    body = {
        "schema": SHARD_SPEC_SCHEMA,
        "run_contract_sha256": run_contract["run_contract_sha256"],
        "ownership_policy": {
            "algorithm": (
                OWNERSHIP_ALGORITHM if weights is None else WEIGHTED_OWNERSHIP_ALGORITHM
            ),
            "weights": (
                [1 for _index in range(int(shard_count))]
                if weights is None
                else list(weights)
            ),
            "weighted": weights is not None,
        },
        "layer_count": layer_count,
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "owned_layer_indices": list(owned_layers),
        "expected_records": list(owned),
        "expected_candidate_count": len(owned),
        "global_candidate_inventory_sha256": (frontier["candidate_inventory_sha256"]),
        "owned_candidate_inventory_sha256": _json_sha256(list(owned)),
    }
    return {**body, "shard_spec_sha256": _json_sha256(body)}


def verify_lfm2_cached_causal_shard_spec(
    spec: Mapping[str, Any],
    run_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    verify_lfm2_cached_causal_run_contract(run_contract)
    value = _mapping(spec, "shard_spec")
    seal = value.pop("shard_spec_sha256", None)
    if value.get("schema") != SHARD_SPEC_SCHEMA or _digest(
        seal,
        "shard_spec_sha256",
    ) != _json_sha256(value):
        raise Lfm2CachedCausalShardingError("Shard spec seal is invalid.")
    if value.get("run_contract_sha256") != run_contract["run_contract_sha256"]:
        raise Lfm2CachedCausalShardingError("Shard spec binds another run contract.")
    policy = _mapping(value.get("ownership_policy"), "ownership_policy")
    weighted = policy.get("weighted")
    if not isinstance(weighted, bool):
        raise Lfm2CachedCausalShardingError(
            "Shard ownership policy must declare whether it is weighted."
        )
    raw_weights = policy.get("weights")
    if not isinstance(raw_weights, list):
        raise Lfm2CachedCausalShardingError(
            "Shard ownership policy weights are missing."
        )
    expected = seal_lfm2_cached_causal_shard_spec(
        run_contract,
        shard_count=_positive_int(value.get("shard_count"), "shard_count"),
        shard_index=_nonnegative_int(value.get("shard_index"), "shard_index"),
        shard_weights=(tuple(raw_weights) if weighted else None),
    )
    if dict(spec) != expected:
        raise Lfm2CachedCausalShardingError(
            "Shard spec ownership or expected inventory is invalid."
        )
    return spec


def _safe_record_path(
    shard_root: str | os.PathLike[str],
    relative_path: Any,
    *,
    expected_option_id: str,
) -> Path:
    root = Path(shard_root).expanduser()
    if not root.is_dir() or root.is_symlink():
        raise Lfm2CachedCausalShardingError(
            "Shard root must be an existing regular directory."
        )
    if (
        not isinstance(relative_path, str)
        or "\\" in relative_path
        or "://" in relative_path
    ):
        raise Lfm2CachedCausalShardingError(
            "Candidate record path must be a portable relative path."
        )
    portable = PurePosixPath(relative_path)
    expected = PurePosixPath("candidate-records") / f"{expected_option_id}.json"
    if (
        portable.is_absolute()
        or portable != expected
        or any(part in ("", ".", "..") for part in portable.parts)
    ):
        raise Lfm2CachedCausalShardingError(
            "Candidate record path is outside its canonical shard location."
        )
    root_resolved = root.resolve(strict=True)
    candidate = root.joinpath(*portable.parts)
    if not candidate.is_file() or candidate.is_symlink():
        raise Lfm2CachedCausalShardingError(
            "Candidate record is missing, non-regular, or symlinked."
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise Lfm2CachedCausalShardingError(
            "Candidate record resolves outside the shard root."
        ) from exc
    return resolved


def _read_candidate_record(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CachedCausalShardingError(
            "Candidate record is not readable canonical JSON."
        ) from exc
    return _mapping(document, "candidate_record")


def build_lfm2_cached_causal_record_inventory_entry(
    shard_root: str | os.PathLike[str],
    expected_record: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    *,
    relative_path: str | None = None,
) -> dict[str, Any]:
    """Authenticate one sealed candidate record and return merge metadata."""

    verify_lfm2_cached_causal_run_contract(run_contract)
    expected = _mapping(expected_record, "expected_record")
    option_id = _option_id(expected.get("option_id"))
    layer_index = _nonnegative_int(expected.get("layer_index"), "layer_index")
    target_width = _positive_int(expected.get("target_width"), "target_width")
    proposal_sha = _digest(
        expected.get("proposal_sha256"),
        "expected_record.proposal_sha256",
    )
    selected_path = (
        f"candidate-records/{option_id}.json"
        if relative_path is None
        else relative_path
    )
    path = _safe_record_path(
        shard_root,
        selected_path,
        expected_option_id=option_id,
    )
    document = _read_candidate_record(path)
    body = dict(document)
    candidate_record_sha = body.pop("candidate_record_sha256", None)
    if body.get("schema") != CANDIDATE_RECORD_SCHEMA or _digest(
        candidate_record_sha, "candidate_record_sha256"
    ) != _json_sha256(body):
        raise Lfm2CachedCausalShardingError("Candidate record seal is invalid.")
    proposal = _mapping(body.get("proposal"), "candidate_record.proposal")
    candidate = _mapping(body.get("candidate"), "candidate_record.candidate")
    measurement = _mapping(
        body.get("measurement"),
        "candidate_record.measurement",
    )
    event_sha = _digest(
        measurement.get("output_event_sha256"),
        "measurement.output_event_sha256",
    )
    measurement_body = dict(measurement)
    measurement_body.pop("output_event_sha256", None)
    if _json_sha256(measurement_body) != event_sha:
        raise Lfm2CachedCausalShardingError(
            "Candidate measurement output-event seal is invalid."
        )
    if (
        proposal.get("option_id") != option_id
        or proposal.get("layer_index") != layer_index
        or proposal.get("target_width") != target_width
        or proposal.get("proposal_sha256") != proposal_sha
        or proposal.get("source_evidence_sha256")
        != run_contract["evidence_binding"]["variable_width_evidence_sha256"]
        or candidate.get("option_id") != option_id
        or candidate.get("layer_index") != layer_index
        or candidate.get("target_units") != target_width
        or candidate.get("proposal_sha256") != proposal_sha
        or candidate.get("output_event_sha256") != event_sha
        or measurement.get("option_id") != option_id
        or body.get("split_sha256") != run_contract["evidence_binding"]["split_sha256"]
        or measurement.get("split_sha256")
        != run_contract["evidence_binding"]["split_sha256"]
    ):
        raise Lfm2CachedCausalShardingError(
            "Candidate record differs from its physical proposal or run binding."
        )
    for field in _ZERO_AUDIT_FIELDS:
        if body.get(field) != 0:
            raise Lfm2CachedCausalShardingError(
                f"Candidate record has nonzero {field}."
            )
    for field in _RUN_ZERO_AUDIT_FIELDS:
        if field in measurement and measurement.get(field) != 0:
            raise Lfm2CachedCausalShardingError(
                f"Candidate measurement has nonzero {field}."
            )
    child_trace_sha = _digest(
        candidate.get("measured_child_trace_sha256"),
        "candidate.measured_child_trace_sha256",
    )
    return {
        "option_id": option_id,
        "layer_index": layer_index,
        "target_width": target_width,
        "proposal_sha256": proposal_sha,
        "candidate_record_sha256": candidate_record_sha,
        "record_file_sha256": _file_sha256(path),
        "output_event_sha256": event_sha,
        "measured_child_trace_sha256": child_trace_sha,
        "relative_path": selected_path,
    }


def seal_lfm2_cached_causal_shard_manifest(
    shard_root: str | os.PathLike[str],
    run_contract: Mapping[str, Any],
    shard_spec: Mapping[str, Any],
    *,
    worker_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read, authenticate, and seal every record owned by one complete shard."""

    verify_lfm2_cached_causal_shard_spec(shard_spec, run_contract)
    records = tuple(
        build_lfm2_cached_causal_record_inventory_entry(
            shard_root,
            expected,
            run_contract,
        )
        for expected in shard_spec["expected_records"]
    )
    body = {
        "schema": SHARD_MANIFEST_SCHEMA,
        "run_contract_sha256": run_contract["run_contract_sha256"],
        "shard_spec": dict(shard_spec),
        "shard_spec_sha256": shard_spec["shard_spec_sha256"],
        "worker_provenance": (
            {}
            if worker_provenance is None
            else _mapping(worker_provenance, "worker_provenance")
        ),
        "records": list(records),
        "candidate_count": len(records),
        "record_inventory_sha256": _json_sha256(list(records)),
        "complete": True,
        "parent_forward_calls": 0,
        "parent_cache_build_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return {**body, "shard_manifest_sha256": _json_sha256(body)}


def verify_lfm2_cached_causal_shard_manifest(
    manifest: Mapping[str, Any],
    shard_root: str | os.PathLike[str],
    run_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    verify_lfm2_cached_causal_run_contract(run_contract)
    value = _mapping(manifest, "shard_manifest")
    seal = value.pop("shard_manifest_sha256", None)
    if value.get("schema") != SHARD_MANIFEST_SCHEMA or _digest(
        seal,
        "shard_manifest_sha256",
    ) != _json_sha256(value):
        raise Lfm2CachedCausalShardingError("Shard manifest seal is invalid.")
    if (
        value.get("run_contract_sha256") != run_contract["run_contract_sha256"]
        or value.get("complete") is not True
    ):
        raise Lfm2CachedCausalShardingError(
            "Shard manifest is incomplete or binds another run."
        )
    spec = value.get("shard_spec")
    if not isinstance(spec, Mapping):
        raise Lfm2CachedCausalShardingError("Shard manifest has no sealed shard spec.")
    verify_lfm2_cached_causal_shard_spec(spec, run_contract)
    if value.get("shard_spec_sha256") != spec["shard_spec_sha256"]:
        raise Lfm2CachedCausalShardingError("Shard manifest/spec binding is invalid.")
    records = value.get("records")
    if not isinstance(records, list):
        raise Lfm2CachedCausalShardingError("Shard manifest records must be a list.")
    expected_entries = tuple(
        build_lfm2_cached_causal_record_inventory_entry(
            shard_root,
            expected,
            run_contract,
        )
        for expected in spec["expected_records"]
    )
    if (
        records != list(expected_entries)
        or value.get("candidate_count") != len(expected_entries)
        or value.get("record_inventory_sha256") != _json_sha256(list(expected_entries))
    ):
        raise Lfm2CachedCausalShardingError(
            "Shard record inventory is incomplete, stale, or reordered."
        )
    if any(value.get(field) != 0 for field in _RUN_ZERO_AUDIT_FIELDS):
        raise Lfm2CachedCausalShardingError(
            "Shard manifest reports a forbidden operation."
        )
    return manifest


def validate_lfm2_cached_causal_merge_inventory(
    run_contract: Mapping[str, Any],
    shard_inputs: Sequence[tuple[str | os.PathLike[str], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Authenticate complete disjoint shards and seal their merge inventory."""

    verify_lfm2_cached_causal_run_contract(run_contract)
    if isinstance(shard_inputs, (str, bytes)) or not isinstance(shard_inputs, Sequence):
        raise Lfm2CachedCausalShardingError(
            "shard_inputs must be a sequence of (root, manifest) pairs."
        )
    verified: dict[int, Mapping[str, Any]] = {}
    for raw in shard_inputs:
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise Lfm2CachedCausalShardingError(
                "Each shard input must be a (root, manifest) tuple."
            )
        root, manifest = raw
        verify_lfm2_cached_causal_shard_manifest(
            manifest,
            root,
            run_contract,
        )
        index = int(manifest["shard_spec"]["shard_index"])
        if index in verified:
            raise Lfm2CachedCausalShardingError(
                "Merge inputs contain a duplicate shard index."
            )
        verified[index] = manifest
    if not verified:
        raise Lfm2CachedCausalShardingError("Merge needs at least one shard.")
    shard_counts = {
        int(value["shard_spec"]["shard_count"]) for value in verified.values()
    }
    if len(shard_counts) != 1:
        raise Lfm2CachedCausalShardingError("Merge inputs disagree about shard count.")
    shard_count = next(iter(shard_counts))
    if set(verified) != set(range(shard_count)):
        raise Lfm2CachedCausalShardingError(
            "Merge inputs do not provide every shard exactly once."
        )
    ownership_policies = {
        _json_sha256(value["shard_spec"]["ownership_policy"])
        for value in verified.values()
    }
    if len(ownership_policies) != 1:
        raise Lfm2CachedCausalShardingError(
            "Merge inputs disagree about the ownership policy or weights."
        )
    ownership_policy = verified[0]["shard_spec"]["ownership_policy"]
    layer_count = int(run_contract["frontier"]["layer_count"])
    owned_layers = [
        layer
        for index in range(shard_count)
        for layer in verified[index]["shard_spec"]["owned_layer_indices"]
    ]
    if sorted(owned_layers) != list(range(layer_count)) or len(
        set(owned_layers)
    ) != len(owned_layers):
        raise Lfm2CachedCausalShardingError(
            "Merge shard ownership is overlapping or incomplete."
        )
    merged_records = []
    seen_options: set[str] = set()
    for index in range(shard_count):
        for record in verified[index]["records"]:
            option_id = str(record["option_id"])
            if option_id in seen_options:
                raise Lfm2CachedCausalShardingError(
                    "Merge inputs contain a duplicate candidate option."
                )
            seen_options.add(option_id)
            merged_records.append(
                {
                    "source_shard_index": index,
                    **dict(record),
                }
            )
    expected_options = {
        str(value["option_id"])
        for value in run_contract["frontier"]["candidate_inventory"]
    }
    if seen_options != expected_options:
        raise Lfm2CachedCausalShardingError(
            "Merged records do not exactly cover the global candidate inventory."
        )
    ordered_records = sorted(
        merged_records,
        key=lambda value: value["option_id"],
    )
    manifest_seals = [
        {
            "shard_index": index,
            "shard_manifest_sha256": verified[index]["shard_manifest_sha256"],
        }
        for index in range(shard_count)
    ]
    body = {
        "schema": MERGE_INVENTORY_SCHEMA,
        "run_contract_sha256": run_contract["run_contract_sha256"],
        "ownership_policy": dict(ownership_policy),
        "shard_count": shard_count,
        "layer_count": layer_count,
        "input_shard_manifests": manifest_seals,
        "input_shard_manifests_sha256": _json_sha256(manifest_seals),
        "records": ordered_records,
        "candidate_count": len(ordered_records),
        "merged_record_inventory_sha256": _json_sha256(ordered_records),
        "parent_forward_calls": 0,
        "parent_cache_build_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return {**body, "merge_inventory_sha256": _json_sha256(body)}


__all__ = [
    "CANDIDATE_RECORD_SCHEMA",
    "Lfm2CachedCausalShardingError",
    "MERGE_INVENTORY_SCHEMA",
    "OWNERSHIP_ALGORITHM",
    "RUN_CONTRACT_SCHEMA",
    "SHARD_MANIFEST_SCHEMA",
    "SHARD_SPEC_SCHEMA",
    "WEIGHTED_OWNERSHIP_ALGORITHM",
    "build_lfm2_cached_causal_record_inventory_entry",
    "lfm2_contiguous_layer_allocations",
    "lfm2_contiguous_layer_bounds",
    "lfm2_contiguous_layer_indices",
    "lfm2_contiguous_layer_owner",
    "seal_lfm2_cached_causal_run_contract",
    "seal_lfm2_cached_causal_shard_manifest",
    "seal_lfm2_cached_causal_shard_spec",
    "validate_lfm2_cached_causal_merge_inventory",
    "verify_lfm2_cached_causal_run_contract",
    "verify_lfm2_cached_causal_shard_manifest",
    "verify_lfm2_cached_causal_shard_spec",
]
