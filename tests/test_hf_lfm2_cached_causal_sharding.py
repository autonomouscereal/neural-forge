from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neural_forge.hf_lfm2_cached_causal_sharding import (
    CANDIDATE_RECORD_SCHEMA,
    Lfm2CachedCausalShardingError,
    build_lfm2_cached_causal_record_inventory_entry,
    lfm2_contiguous_layer_allocations,
    lfm2_contiguous_layer_bounds,
    lfm2_contiguous_layer_indices,
    lfm2_contiguous_layer_owner,
    seal_lfm2_cached_causal_run_contract,
    seal_lfm2_cached_causal_shard_manifest,
    seal_lfm2_cached_causal_shard_spec,
    validate_lfm2_cached_causal_merge_inventory,
    verify_lfm2_cached_causal_run_contract,
    verify_lfm2_cached_causal_shard_manifest,
    verify_lfm2_cached_causal_shard_spec,
)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _json_sha256(value) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _candidate_inventory(target_widths=(8,)):
    return tuple(
        {
            "option_id": (
                f"ffn-l{layer_index:02d}-w{target_width:04d}-"
                f"{layer_index:08x}{target_width:08x}"
            ),
            "layer_index": layer_index,
            "target_width": target_width,
            "proposal_sha256": _digest(f"proposal-{layer_index}-{target_width}"),
            "canonical_pack_manifest_sha256": _digest(
                f"pack-manifest-{layer_index}-{target_width}"
            ),
            "canonical_pack_tensor_file_sha256": _digest(
                f"pack-tensors-{layer_index}-{target_width}"
            ),
            "canonical_bundle_sha256": _digest(f"bundle-{layer_index}-{target_width}"),
            "canonical_layer_binding_sha256": _digest(
                f"layer-binding-{layer_index}-{target_width}"
            ),
            "canonical_payload_required": True,
        }
        for layer_index in range(14)
        for target_width in target_widths
    )


def _run_contract(*, dtype: str = "float16", target_widths=(8,)):
    inventory = _candidate_inventory(target_widths)
    return seal_lfm2_cached_causal_run_contract(
        parent_binding={
            "parent_identity_sha256": _digest("parent"),
            "parent_unique_parameters": 229_693_184,
            "parent_trace_cache_key": _digest("cache-key"),
            "parent_corpus_sha256": _digest("corpus"),
            "parent_trace_rows_sha256": _digest("trace-rows"),
            "parent_output_envelope_sha256": _digest("envelope"),
            "parent_output_file_sha256": _digest("envelope-file"),
            "tokenizer_fingerprint": _digest("tokenizer"),
        },
        evidence_binding={
            "profile": "balanced",
            "split_sha256": _digest("split"),
            "task_anchor_manifest_sha256": _digest("anchors"),
            "variable_width_evidence_sha256": _digest("variable-width"),
        },
        target_fraction=0.35,
        widths_by_layer=tuple((10, *target_widths) for _ in range(14)),
        candidate_inventory=inventory,
        measurement_contract={
            "dtype": dtype,
            "decode_trace_mode": "full_prefix_canonical",
            "forced_decode_use_cache": False,
            "rollout_batch_size": 16,
            "cluster_ridge": 1e-8,
            "cluster_max_iterations": 8,
            "cluster_distance_chunk_size": 512,
        },
        producer_implementation_sha256=_digest("producer-implementation"),
    )


def _write_record(root: Path, expected, contract) -> None:
    option_id = expected["option_id"]
    event_body = {
        "schema": "test-cached-causal-measurement-v1",
        "option_id": option_id,
        "split_sha256": contract["evidence_binding"]["split_sha256"],
        "parent_forward_calls": 0,
        "parent_cache_build_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    event_sha = _json_sha256(event_body)
    measurement = {
        **event_body,
        "output_event_sha256": event_sha,
    }
    proposal = {
        "option_id": option_id,
        "layer_index": expected["layer_index"],
        "target_width": expected["target_width"],
        "proposal_sha256": expected["proposal_sha256"],
        "source_evidence_sha256": (
            contract["evidence_binding"]["variable_width_evidence_sha256"]
        ),
    }
    candidate = {
        "option_id": option_id,
        "layer_index": expected["layer_index"],
        "target_units": expected["target_width"],
        "proposal_sha256": expected["proposal_sha256"],
        "output_event_sha256": event_sha,
        "measured_child_trace_sha256": _digest(f"trace-{option_id}"),
    }
    body = {
        "schema": CANDIDATE_RECORD_SCHEMA,
        "proposal": proposal,
        "candidate": candidate,
        "measurement": measurement,
        "split_sha256": contract["evidence_binding"]["split_sha256"],
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    document = {
        **body,
        "candidate_record_sha256": _json_sha256(body),
    }
    path = root / "candidate-records" / f"{option_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(document) + b"\n")


def _complete_shards(
    tmp_path: Path,
    contract,
    *,
    shard_count: int = 6,
    shard_weights=None,
):
    result = []
    for index in range(shard_count):
        root = tmp_path / f"shard-{index:02d}-of-{shard_count:02d}"
        root.mkdir()
        spec = seal_lfm2_cached_causal_shard_spec(
            contract,
            shard_count=shard_count,
            shard_index=index,
            shard_weights=shard_weights,
        )
        for expected in spec["expected_records"]:
            _write_record(root, expected, contract)
        manifest = seal_lfm2_cached_causal_shard_manifest(
            root,
            contract,
            spec,
            worker_provenance={
                "device": f"cuda:{index}",
                "gpu_family": "test",
            },
        )
        result.append((root, manifest))
    return result


def test_six_way_contiguous_ownership_is_complete_and_deterministic() -> None:
    bounds = tuple(lfm2_contiguous_layer_bounds(14, 6, index) for index in range(6))
    assert bounds == ((0, 2), (2, 4), (4, 7), (7, 9), (9, 11), (11, 14))
    shards = tuple(lfm2_contiguous_layer_indices(14, 6, index) for index in range(6))
    assert tuple(len(value) for value in shards) == (2, 2, 3, 2, 2, 3)
    assert tuple(layer for shard in shards for layer in shard) == tuple(range(14))
    assert tuple(lfm2_contiguous_layer_owner(14, 6, layer) for layer in range(14)) == (
        0,
        0,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        5,
    )

    with pytest.raises(Lfm2CachedCausalShardingError, match="empty shards"):
        lfm2_contiguous_layer_bounds(4, 5, 0)


def test_three_way_weighted_ownership_gives_fast_workers_more_inventory(
    tmp_path: Path,
) -> None:
    weights = (3, 3, 2)
    assert lfm2_contiguous_layer_allocations(
        14,
        3,
        shard_weights=weights,
    ) == (5, 5, 4)
    shards = tuple(
        lfm2_contiguous_layer_indices(
            14,
            3,
            index,
            shard_weights=weights,
        )
        for index in range(3)
    )
    assert shards == (
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
        (10, 11, 12, 13),
    )
    assert tuple(
        lfm2_contiguous_layer_owner(
            14,
            3,
            layer,
            shard_weights=weights,
        )
        for layer in range(14)
    ) == (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2)

    contract = _run_contract(
        target_widths=(9, 8, 7, 6, 5, 4),
    )
    specs = tuple(
        seal_lfm2_cached_causal_shard_spec(
            contract,
            shard_count=3,
            shard_index=index,
            shard_weights=weights,
        )
        for index in range(3)
    )
    assert [spec["expected_candidate_count"] for spec in specs] == [30, 30, 24]
    assert all(
        spec["ownership_policy"]
        == {
            "algorithm": ("contiguous-layer-weighted-largest-remainder-v1"),
            "weights": [3, 3, 2],
            "weighted": True,
        }
        for spec in specs
    )
    owned_options = [
        row["option_id"] for spec in specs for row in spec["expected_records"]
    ]
    assert len(owned_options) == len(set(owned_options)) == 84
    shards_on_disk = _complete_shards(
        tmp_path,
        contract,
        shard_count=3,
        shard_weights=weights,
    )
    merged = validate_lfm2_cached_causal_merge_inventory(
        contract,
        shards_on_disk,
    )
    assert merged["candidate_count"] == 84
    assert merged["ownership_policy"] == specs[0]["ownership_policy"]
    assert [
        sum(record["source_shard_index"] == index for record in merged["records"])
        for index in range(3)
    ] == [30, 30, 24]


def test_run_contract_and_shard_specs_are_sealed() -> None:
    contract = _run_contract()
    assert verify_lfm2_cached_causal_run_contract(contract) is contract
    specs = tuple(
        seal_lfm2_cached_causal_shard_spec(
            contract,
            shard_count=6,
            shard_index=index,
        )
        for index in range(6)
    )
    assert [spec["expected_candidate_count"] for spec in specs] == [
        2,
        2,
        3,
        2,
        2,
        3,
    ]
    assert all(
        verify_lfm2_cached_causal_shard_spec(spec, contract) is spec for spec in specs
    )

    tampered = json.loads(json.dumps(contract))
    tampered["measurement_contract"]["dtype"] = "bfloat16"
    with pytest.raises(Lfm2CachedCausalShardingError, match="seal"):
        verify_lfm2_cached_causal_run_contract(tampered)


def test_record_and_manifest_authentication_rejects_tamper_and_escape(
    tmp_path: Path,
) -> None:
    contract = _run_contract()
    spec = seal_lfm2_cached_causal_shard_spec(
        contract,
        shard_count=6,
        shard_index=0,
    )
    root = tmp_path / "shard"
    root.mkdir()
    for expected in spec["expected_records"]:
        _write_record(root, expected, contract)
    first = spec["expected_records"][0]
    entry = build_lfm2_cached_causal_record_inventory_entry(
        root,
        first,
        contract,
    )
    assert entry["option_id"] == first["option_id"]
    manifest = seal_lfm2_cached_causal_shard_manifest(
        root,
        contract,
        spec,
    )
    assert (
        verify_lfm2_cached_causal_shard_manifest(
            manifest,
            root,
            contract,
        )
        is manifest
    )

    with pytest.raises(Lfm2CachedCausalShardingError, match="canonical"):
        build_lfm2_cached_causal_record_inventory_entry(
            root,
            first,
            contract,
            relative_path="../outside.json",
        )

    record_path = root / entry["relative_path"]
    record_path.write_bytes(
        record_path.read_bytes().replace(b'"backward_calls":0', b'"backward_calls":1')
    )
    with pytest.raises(Lfm2CachedCausalShardingError, match="seal"):
        verify_lfm2_cached_causal_shard_manifest(
            manifest,
            root,
            contract,
        )


def test_six_way_merge_is_exact_order_independent_and_fails_closed(
    tmp_path: Path,
) -> None:
    contract = _run_contract()
    shards = _complete_shards(tmp_path, contract)
    merged = validate_lfm2_cached_causal_merge_inventory(contract, shards)
    reversed_merge = validate_lfm2_cached_causal_merge_inventory(
        contract,
        list(reversed(shards)),
    )
    assert merged == reversed_merge
    assert merged["shard_count"] == 6
    assert merged["candidate_count"] == 14
    assert [row["option_id"] for row in merged["records"]] == sorted(
        row["option_id"] for row in merged["records"]
    )
    assert all(
        merged[field] == 0
        for field in (
            "parent_forward_calls",
            "parent_cache_build_calls",
            "parent_generation_calls",
            "optimizer_steps",
            "backward_calls",
        )
    )

    with pytest.raises(Lfm2CachedCausalShardingError, match="every shard"):
        validate_lfm2_cached_causal_merge_inventory(contract, shards[:-1])

    duplicate = [*shards[:-1], shards[0]]
    with pytest.raises(Lfm2CachedCausalShardingError, match="duplicate shard"):
        validate_lfm2_cached_causal_merge_inventory(contract, duplicate)

    incompatible = _run_contract(dtype="bfloat16")
    with pytest.raises(
        Lfm2CachedCausalShardingError,
        match="another run",
    ):
        validate_lfm2_cached_causal_merge_inventory(incompatible, shards)
