from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache
from neural_forge.hf_lfm2_circuit_trace_cache import (
    CHILD_REPLAY_BATCH_SCHEMA,
    CHILD_REPLAY_SCHEMA,
    Lfm2ChildEvidenceProfile,
    Lfm2ChildReplay,
    Lfm2ChildReplayBatch,
)
from neural_forge import hf_lfm2_moving_gauge_conditioned_recapture_phase_cache as recapture
from neural_forge.hf_lfm2_moving_gauge_conditioned_recapture_phase_cache import (
    AuthenticatedLfm2ConditionedRecapturePhase,
    LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS,
    Lfm2ConditionedRecaptureAuthenticationError,
    Lfm2ConditionedRecaptureFieldContract,
    Lfm2ConditionedRecapturePhaseAuthority,
    Lfm2ConditionedRecapturePhaseRequest,
    Lfm2ConditionedRecaptureProgress,
    derive_lfm2_conditioned_recapture_split_identity,
    derive_lfm2_frozen_conditioned_child_state_receipt_sha256,
    lfm2_conditioned_recapture_implementation_sha256,
    lfm2_conditioned_recapture_spec_sha256,
    load_or_capture_lfm2_conditioned_recapture_phase,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sealed(body: dict[str, Any], seal_name: str) -> dict[str, Any]:
    return {**body, seal_name: canonical_sha256(body)}


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _field_contracts() -> tuple[Lfm2ConditionedRecaptureFieldContract, ...]:
    widths = {
        "short_conv.context": ("float64", 3),
        "operator.output": ("float32", 2),
        "residual.pre": ("float32", 2),
        "residual.post_operator": ("float32", 2),
        "residual.post_layer": ("float32", 2),
        "mlp.post_swiglu": ("float32", 4),
        "mlp.down": ("float32", 2),
    }
    return tuple(
        Lfm2ConditionedRecaptureFieldContract(
            field_name=name,
            source_dtype=widths[name][0],
            feature_width=widths[name][1],
        )
        for name in LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS
    )


def _static_parts() -> dict[str, Any]:
    child_body = {
        "model_class": "tests.ConditionedChild",
        "config_sha256": _digest("conditioned-config"),
        "state_dict_sha256": _digest("conditioned-state"),
        "state_tensor_count": 7,
        "runtime_buffer_sha256": _digest("conditioned-buffers"),
        "runtime_buffer_count": 1,
    }
    child = {
        **child_body,
        "identity_sha256": canonical_sha256(
            {"schema": "neural-forge-lfm2-parent-fingerprint-v2", **child_body}
        ),
    }
    topology = _sealed(
        {
            "schema": "neural-forge-lfm2-layer-topology-v1",
            "layer_count": 1,
            "layers": [{"container_index": 0, "kind": "short_conv"}],
        },
        "layer_topology_sha256",
    )
    implementation = _sealed(
        {
            "schema": "neural-forge-lfm2-circuit-implementation-v-test",
            "collector_source": "tests",
        },
        "implementation_sha256",
    )
    capture = _sealed(
        {
            "sample_last_tokens": 1,
            "max_decode_steps": 1,
            "max_capture_bytes": 1024 * 1024,
            "logit_rtol": 1e-5,
            "logit_atol": 1e-6,
            "require_top1_agreement": True,
            "allow_top1_disagreement_within_numeric_error_envelope": False,
            "inference_mode": True,
            "model_eval_mode": True,
            "gradients_used": False,
            "training_steps": 0,
            "prefill_use_cache": False,
            "cached_prefill_use_cache": True,
            "forced_decode_use_cache": True,
            "full_prefix_verification_use_cache": False,
        },
        "selection_sha256",
    )
    profile = Lfm2ChildEvidenceProfile(
        profile_role="moving_gauge.conditioned_recapture",
        layer_suffixes=(
            (0, tuple(sorted(LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS))),
        ),
    )
    selected_topology = _sealed(
        {
            "schema": "neural-forge-lfm2-child-selected-topology-v1",
            "model_layer_count": 1,
            "selected_layer_indices": [0],
            "layers": topology["layers"],
        },
        "selected_topology_sha256",
    )
    selection = _sealed(
        {
            "schema": "neural-forge-lfm2-child-replay-selection-v2",
            "selected_layer_indices": [0],
            "capture": capture,
            "replay_capture_mode": "prefix_by_prefix_canonical",
            "selected_topology": selected_topology,
            "child_evidence_profile": profile.as_dict(),
            "child_evidence_profile_sha256": profile.profile_sha256,
        },
        "selection_sha256",
    )
    execution = _sealed(
        {
            "schema": "neural-forge-lfm2-circuit-execution-v1",
            "attention_backend": "eager",
            "fast_short_conv_available": False,
            "fast_short_conv_selected": False,
            "input_device_type": "cpu",
        },
        "execution_sha256",
    )
    numeric_execution = _sealed(
        {
            "schema": "neural-forge-numeric-execution-policy-test-v1",
            "target_inventory": {
                "floating_dtype_counts": {"float32": 6, "float64": 1}
            },
        },
        "policy_sha256",
    )
    envelope = _sealed(
        {
            "schema": "neural-forge-lfm2-child-replay-envelope-test-v1",
            "numeric_policy_sha256": numeric_execution["policy_sha256"],
        },
        "envelope_sha256",
    )
    numeric = _sealed(
        {
            "schema": "neural-forge-lfm2-child-replay-numeric-test-v1",
            "numeric_execution_policy": numeric_execution,
            "numeric_policy_sha256": numeric_execution["policy_sha256"],
            "non_logit_envelope_policy": envelope,
            "non_logit_envelope_sha256": envelope["envelope_sha256"],
        },
        "replay_numeric_policy_sha256",
    )
    return {
        "child": child,
        "topology": topology,
        "implementation": implementation,
        "capture": capture,
        "profile": profile,
        "selection": selection,
        "execution": execution,
        "numeric": numeric,
    }


@pytest.fixture(scope="module")
def parts() -> dict[str, Any]:
    return _static_parts()


def _request(parts: dict[str, Any]) -> Lfm2ConditionedRecapturePhaseRequest:
    input_sha256s = tuple(_digest(f"source-input-{index}") for index in range(256))
    record_ids = tuple(f"record-{index:03d}" for index in range(256))
    basis_split_by_batch = tuple(
        "heldout" if index % 4 == 3 else "fit" for index in range(256)
    )
    source_session = _digest("source-session")
    split = derive_lfm2_conditioned_recapture_split_identity(
        source_input_session_sha256=source_session,
        source_input_sha256s=input_sha256s,
        source_record_ids=record_ids,
        basis_split_by_batch=basis_split_by_batch,
    )
    operator = recapture._stored_tensor_descriptor(torch.eye(2, dtype=torch.float64))
    base_identity = _digest("base-child-identity")
    base_state = _digest("base-child-state")
    endpoint = _digest("endpoint-chain-receipt")
    solution = _digest("operator-solution")
    state_receipt = derive_lfm2_frozen_conditioned_child_state_receipt_sha256(
        base_child_identity_sha256=base_identity,
        base_child_state_dict_sha256=base_state,
        endpoint_chain_receipt_sha256=endpoint,
        operator_replacement_tensor_descriptor=operator,
        operator_solution_sha256=solution,
        conditioned_child_identity_sha256=parts["child"]["identity_sha256"],
        conditioned_child_state_dict_sha256=parts["child"]["state_dict_sha256"],
    )
    return Lfm2ConditionedRecapturePhaseRequest(
        v5_request_sha256=_digest("v5-request"),
        v5_lineage_sha256=_digest("v5-lineage"),
        base_child_identity_sha256=base_identity,
        base_child_state_dict_sha256=base_state,
        endpoint_chain_receipt_sha256=endpoint,
        operator_replacement_tensor_descriptor=operator,
        operator_solution_sha256=solution,
        frozen_conditioned_child_state_receipt_sha256=state_receipt,
        conditioned_child_identity_sha256=parts["child"]["identity_sha256"],
        conditioned_child_state_dict_sha256=parts["child"]["state_dict_sha256"],
        child_layer_topology_sha256=parts["topology"]["layer_topology_sha256"],
        chart_mode="E0R",
        child_replay_config_sha256=parts["capture"]["selection_sha256"],
        child_replay_implementation_sha256=parts["implementation"][
            "implementation_sha256"
        ],
        child_replay_execution_sha256=parts["execution"]["execution_sha256"],
        child_replay_numeric_policy_sha256=parts["numeric"][
            "replay_numeric_policy_sha256"
        ],
        child_replay_capture_mode="prefix_by_prefix_canonical",
        source_input_session_sha256=source_session,
        source_input_sha256s=input_sha256s,
        source_record_ids=record_ids,
        basis_split_by_batch=basis_split_by_batch,
        split_identity_sha256=split,
        field_contracts=_field_contracts(),
        runtime_device_type="cpu",
        runtime_floating_dtypes=("float32", "float64"),
        conditioned_recapture_spec_sha256=(
            lfm2_conditioned_recapture_spec_sha256()
        ),
        implementation_sha256=lfm2_conditioned_recapture_implementation_sha256(),
        checkpoint_hmac_key_id="test-conditioned-recapture",
    )


def _source_tensor(
    field_name: str, global_index: int, width: int, dtype: torch.dtype
) -> torch.Tensor:
    field_offset = LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS.index(field_name) * 10
    value = 1000 + field_offset + global_index
    return torch.full((1, 1, width), value, dtype=dtype)


def _runtime_capture(
    profile: Lfm2ChildEvidenceProfile,
    phases: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "capture_bytes": 4096,
        "decode_steps": len(phases) - 1,
        "prefill_sample_start": 0,
        "prefill_sample_count": 1,
        "phase_diagnostics": {phase: {} for phase in phases},
        "cached_full_prefix_verifications": [],
        "all_cached_logits_verified": None,
        "all_cached_top1_verified": None,
        "all_cached_top1_decisions_verified": None,
        "decode_trace_mode": "full_prefix_canonical",
        "canonical_trace_source": "use_cache_false_full_prefix",
        "cached_logits_accepted": False,
        "cached_forward_count": 0,
        "gradients_used": False,
        "training_steps": 0,
        "retained_capture_bytes": 4096,
        "transient_validation_peak_bytes": 1024,
        "estimated_peak_capture_bytes": 5120,
        "phase_evidence_accounting": {phase: {} for phase in phases},
        "child_evidence_profile": profile.as_dict(),
        "lm_head_input_captured_for_replay_equivalence": False,
        "child_replay_capture_mode": "prefix_by_prefix_canonical",
        "child_forward_count": len(phases),
        "canonical_prefix_forward_count": len(phases),
    }


def _replay(
    request: Lfm2ConditionedRecapturePhaseRequest,
    parts: dict[str, Any],
    start: int,
    stop: int,
    *,
    wrong_child: bool = False,
    wrong_input: bool = False,
) -> Lfm2ChildReplay:
    child = dict(parts["child"])
    if wrong_child:
        child_body = dict(child)
        child_body.pop("identity_sha256")
        child_body["config_sha256"] = _digest("wrong-conditioned-config")
        child = {
            **child_body,
            "identity_sha256": canonical_sha256(
                {
                    "schema": "neural-forge-lfm2-parent-fingerprint-v2",
                    **child_body,
                }
            ),
        }
    batches: list[Lfm2ChildReplayBatch] = []
    ordered: list[dict[str, Any]] = []
    source_inputs = list(request.source_input_sha256s[start:stop])
    if wrong_input:
        source_inputs[0] = _digest("wrong-source-input")
    for local_index, (global_index, input_sha256) in enumerate(
        zip(range(start, stop), source_inputs)
    ):
        phases = ("prefill",)
        tensors: dict[str, torch.Tensor] = {
            "input.input_ids": torch.tensor([[global_index + 1]], dtype=torch.int64),
            "input.attention_mask": torch.ones((1, 1), dtype=torch.int64),
            "input.forced_decode_ids": torch.tensor([[7]], dtype=torch.int64),
        }
        for contract in request.field_contracts:
            dtype = getattr(torch, contract.source_dtype)
            tensors[f"prefill.layer.000.{contract.field_name}"] = _source_tensor(
                contract.field_name, global_index, contract.feature_width, dtype
            )
        inventory = {
            name: recapture._trace_tensor_descriptor(tensor)
            for name, tensor in sorted(tensors.items())
        }
        record = _sealed(
            {
                "schema": CHILD_REPLAY_BATCH_SCHEMA,
                "batch_index": local_index,
                "input_sha256": input_sha256,
                "selected_layer_indices": [0],
                "phase_order": list(phases),
                "phase_rows": [
                    {
                        "phase": "prefill",
                        "prefix_length": 1,
                        "sample_start": 0,
                        "sample_count": 1,
                        "forced_decode_step": None,
                    }
                ],
                "child_evidence_profile": parts["profile"].as_dict(),
                "child_evidence_profile_sha256": parts["profile"].profile_sha256,
                "capture": _runtime_capture(parts["profile"], phases),
                "tensor_count": len(inventory),
                "tensors": inventory,
            },
            "replay_batch_sha256",
        )
        batches.append(
            Lfm2ChildReplayBatch(
                index=local_index,
                input_sha256=input_sha256,
                _tensors=MappingProxyType(tensors),
                _record=MappingProxyType(record),
            )
        )
        ordered.append(
            {
                "batch_index": local_index,
                "input_sha256": input_sha256,
                "replay_batch_sha256": record["replay_batch_sha256"],
                "phase_order": ["prefill"],
                "child_evidence_profile_sha256": parts["profile"].profile_sha256,
            }
        )
    corpus = _sealed(
        {
            "schema": "neural-forge-lfm2-circuit-corpus-v1",
            "batch_count": stop - start,
            "ordered_batch_input_sha256": source_inputs,
            "prompt_token_counts": [1] * (stop - start),
            "forced_decode_token_counts": [1] * (stop - start),
        },
        "corpus_sha256",
    )
    metadata = _sealed(
        {
            "schema": CHILD_REPLAY_SCHEMA,
            "child": child,
            "corpus": corpus,
            "implementation": parts["implementation"],
            "layer_topology": parts["topology"],
            "selection": parts["selection"],
            "execution": parts["execution"],
            "numeric_execution_policy": parts["numeric"],
            "batch_count": stop - start,
            "ordered_batches": ordered,
            "forward_counts": {"child": stop - start, "parent": 0},
            "replay_benchmark": {"actual_child_forwards": stop - start},
            "causal_equivalence": {"verified": True},
            "runtime": {
                "tokenizer_accessed": False,
                "parent_model_supplied": False,
                "parent_forward_count": 0,
                "disk_artifacts_written": 0,
                "intermediate_cache": "memory_only",
                "child_state_preserved": True,
            },
        },
        "replay_sha256",
    )
    return Lfm2ChildReplay(
        _batches=tuple(batches),
        _metadata=MappingProxyType(metadata),
    )


def test_runtime_batch_capture_is_distinct_from_config_and_authenticates(
    parts: dict[str, Any],
) -> None:
    request = _request(parts)
    replay = _replay(request, parts, 0, 1)

    assert replay.batches[0].record["capture"] != replay.metadata["selection"][
        "capture"
    ]
    metadata, batches = recapture._authenticate_source_replay(
        replay,
        request,
        batch_start=0,
        batch_stop=1,
    )

    assert metadata["selection"]["capture"] == parts["capture"]
    assert len(batches) == 1


@pytest.fixture
def authority() -> Lfm2ConditionedRecapturePhaseAuthority:
    return Lfm2ConditionedRecapturePhaseAuthority(
        key_id="test-conditioned-recapture", secret=b"r" * 32
    )


def test_interrupted_resume_hit_and_heldout_barrier(
    tmp_path: Path,
    parts: dict[str, Any],
    authority: Lfm2ConditionedRecapturePhaseAuthority,
) -> None:
    request = _request(parts)
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    calls: list[tuple[int, int]] = []

    def provider(start: int, stop: int) -> Lfm2ChildReplay:
        calls.append((start, stop))
        assert not torch.is_grad_enabled()
        return _replay(request, parts, start, stop)

    first = load_or_capture_lfm2_conditioned_recapture_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "work",
        range_provider=provider,
        maximum_new_checkpoints=1,
    )
    assert isinstance(first, Lfm2ConditionedRecaptureProgress)
    assert first.completed_checkpoints == 1
    second = load_or_capture_lfm2_conditioned_recapture_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "work",
        range_provider=provider,
        maximum_new_checkpoints=1,
    )
    assert isinstance(second, Lfm2ConditionedRecaptureProgress)
    assert calls == [(0, 64), (64, 128)]
    complete = load_or_capture_lfm2_conditioned_recapture_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "work",
        range_provider=provider,
    )
    assert isinstance(complete, AuthenticatedLfm2ConditionedRecapturePhase)
    assert calls == [(0, 64), (64, 128), (128, 192), (192, 256)]
    fit = complete.field_rows("mlp.post_swiglu", split="fit")
    assert fit.rows.shape == (192, 4)
    expected_fit_indices = tuple(
        index
        for index, split in enumerate(request.basis_split_by_batch)
        if split == "fit"
    )
    assert fit.global_batch_indices == expected_fit_indices
    baseline = torch.arange(192, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
    assert not torch.equal(fit.rows, baseline)
    assert fit.rows[0].tolist() == [1050.0] * 4
    with pytest.raises(Lfm2ConditionedRecaptureAuthenticationError):
        complete.field_rows("mlp.post_swiglu", split="heldout")
    capability = complete.issue_heldout_access_capability(
        frozen_candidate_barrier_sha256=_digest("candidate-frozen-after-fit")
    )
    heldout = complete.field_rows(
        "mlp.post_swiglu",
        split="heldout",
        heldout_capability=capability,
    )
    assert heldout.rows.shape == (64, 4)
    expected_heldout_indices = tuple(
        index
        for index, split in enumerate(request.basis_split_by_batch)
        if split == "heldout"
    )
    assert heldout.global_batch_indices == expected_heldout_indices
    assert heldout.rows[0].tolist() == [1053.0] * 4

    def forbidden(_start: int, _stop: int) -> Lfm2ChildReplay:
        raise AssertionError("completed hit called provider")

    class ExplodingPath:
        def __fspath__(self) -> str:
            raise AssertionError("completed hit touched checkpoint_root")

    hit = load_or_capture_lfm2_conditioned_recapture_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=ExplodingPath(),
        range_provider=forbidden,
    )
    assert isinstance(hit, AuthenticatedLfm2ConditionedRecapturePhase)
    assert all(value == 0 for value in hit.invocation_audit.values())


def test_provider_failure_does_not_poison_and_retry_starts_at_zero(
    tmp_path: Path,
    parts: dict[str, Any],
    authority: Lfm2ConditionedRecapturePhaseAuthority,
) -> None:
    request = _request(parts)
    cache = DynamicFoldPhaseCache(tmp_path / "cache")

    def failed(_start: int, _stop: int) -> Lfm2ChildReplay:
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        load_or_capture_lfm2_conditioned_recapture_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=tmp_path / "work",
            range_provider=failed,
        )
    assert cache.lookup(request.phase_key) is None
    assert not list((tmp_path / "work").rglob("checkpoint-*.json"))
    calls: list[tuple[int, int]] = []

    def retry(start: int, stop: int) -> Lfm2ChildReplay:
        calls.append((start, stop))
        return _replay(request, parts, start, stop)

    complete = load_or_capture_lfm2_conditioned_recapture_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "work",
        range_provider=retry,
    )
    assert isinstance(complete, AuthenticatedLfm2ConditionedRecapturePhase)
    assert calls[0] == (0, 64)


@pytest.mark.parametrize("tamper", ["json", "hmac", "shard", "tensor", "chain"])
def test_checkpoint_tampering_fails_closed(
    tmp_path: Path,
    parts: dict[str, Any],
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    tamper: str,
) -> None:
    request = _request(parts)
    cache = DynamicFoldPhaseCache(tmp_path / "cache")

    def provider(start: int, stop: int) -> Lfm2ChildReplay:
        return _replay(request, parts, start, stop)

    progress = load_or_capture_lfm2_conditioned_recapture_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "work",
        range_provider=provider,
        maximum_new_checkpoints=2,
    )
    assert isinstance(progress, Lfm2ConditionedRecaptureProgress)
    root = tmp_path / "work" / request.phase_key.phase_key_sha256
    checkpoint0 = root / "checkpoints" / "checkpoint-0000.json"
    checkpoint1 = root / "checkpoints" / "checkpoint-0001.json"
    shard0 = root / "shards" / "conditioned-recapture-0000.safetensors"
    if tamper == "json":
        checkpoint0.write_bytes(checkpoint0.read_bytes() + b" ")
    elif tamper == "hmac":
        document = json.loads(checkpoint0.read_text(encoding="utf-8"))
        document["checkpoint_hmac_sha256"] = "0" * 64
        _write_canonical(checkpoint0, document)
    elif tamper == "shard":
        value = bytearray(shard0.read_bytes())
        value[-1] ^= 1
        shard0.write_bytes(value)
    elif tamper == "tensor":
        from safetensors.torch import load_file, save_file

        tensors = load_file(str(shard0), device="cpu")
        tensors["mlp.post_swiglu"][0, 0] += 1
        save_file(
            tensors,
            str(shard0),
            metadata={
                "schema": recapture.LFM2_CONDITIONED_RECAPTURE_SHARD_SCHEMA,
                "request_sha256": request.request_sha256,
                "checkpoint_ordinal": "0",
            },
        )
    else:
        document = json.loads(checkpoint1.read_text(encoding="utf-8"))
        signed = dict(document)
        signed.pop("checkpoint_hmac_sha256")
        body = dict(signed)
        body.pop("checkpoint_sha256")
        body["previous_checkpoint_hmac_sha256"] = "f" * 64
        signed = {**body, "checkpoint_sha256": canonical_sha256(body)}
        document = {
            **signed,
            "checkpoint_hmac_sha256": authority.sign_checkpoint(signed),
        }
        _write_canonical(checkpoint1, document)
    with pytest.raises(Lfm2ConditionedRecaptureAuthenticationError):
        load_or_capture_lfm2_conditioned_recapture_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=tmp_path / "work",
            range_provider=provider,
        )
    assert cache.lookup(request.phase_key) is None


def test_wrong_child_and_input_replays_fail_without_completion(
    tmp_path: Path,
    parts: dict[str, Any],
    authority: Lfm2ConditionedRecapturePhaseAuthority,
) -> None:
    request = _request(parts)
    for name, kwargs in (
        ("child", {"wrong_child": True}),
        ("input", {"wrong_input": True}),
    ):
        cache = DynamicFoldPhaseCache(tmp_path / f"cache-{name}")

        def provider(
            start: int, stop: int, selected: dict[str, bool] = kwargs
        ) -> Lfm2ChildReplay:
            return _replay(request, parts, start, stop, **selected)

        with pytest.raises(Lfm2ConditionedRecaptureAuthenticationError):
            load_or_capture_lfm2_conditioned_recapture_phase(
                cache,
                request,
                authority=authority,
                checkpoint_root=tmp_path / f"work-{name}",
                range_provider=provider,
            )
        assert cache.lookup(request.phase_key) is None


def test_wrong_operator_and_split_bindings_are_rejected(
    parts: dict[str, Any],
) -> None:
    request = _request(parts)
    wrong_operator = recapture._stored_tensor_descriptor(
        torch.full((2, 2), 2.0, dtype=torch.float64)
    )
    with pytest.raises(
        Lfm2ConditionedRecaptureAuthenticationError,
        match="state receipt",
    ):
        replace(
            request,
            operator_replacement_tensor_descriptor=wrong_operator,
            request_sha256="",
        )
    with pytest.raises(
        Lfm2ConditionedRecaptureAuthenticationError,
        match="split identity",
    ):
        replace(request, split_identity_sha256=_digest("wrong-split"), request_sha256="")


@pytest.mark.parametrize(
    "substitution",
    [
        ("fit",) * 192 + ("heldout",) * 64,
        tuple(
            "heldout" if (index + 1) % 4 == 3 else "fit"
            for index in range(256)
        ),
    ],
    ids=("contiguous", "reordered"),
)
def test_interleaved_split_rejects_substitution_without_exact_identity(
    parts: dict[str, Any],
    substitution: tuple[str, ...],
) -> None:
    request = _request(parts)
    assert request.basis_split_by_batch != substitution
    with pytest.raises(
        Lfm2ConditionedRecaptureAuthenticationError,
        match="split identity",
    ):
        replace(
            request,
            basis_split_by_batch=substitution,
            request_sha256="",
        )

    matching_identity = derive_lfm2_conditioned_recapture_split_identity(
        source_input_session_sha256=request.source_input_session_sha256,
        source_input_sha256s=request.source_input_sha256s,
        source_record_ids=request.source_record_ids,
        basis_split_by_batch=substitution,
    )
    coherently_readdressed = replace(
        request,
        basis_split_by_batch=substitution,
        split_identity_sha256=matching_identity,
        request_sha256="",
    )
    assert coherently_readdressed.basis_split_by_batch == substitution
    assert coherently_readdressed.split_identity_sha256 == matching_identity
    assert coherently_readdressed.request_sha256 != request.request_sha256
    assert coherently_readdressed.phase_key != request.phase_key


def test_requires_grad_replay_tensor_is_rejected(
    tmp_path: Path,
    parts: dict[str, Any],
    authority: Lfm2ConditionedRecapturePhaseAuthority,
) -> None:
    request = _request(parts)

    def provider(start: int, stop: int) -> Lfm2ChildReplay:
        replay = _replay(request, parts, start, stop)
        replay.batches[0].tensors["prefill.layer.000.mlp.post_swiglu"].requires_grad_(
            True
        )
        return replay

    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    with pytest.raises(
        Lfm2ConditionedRecaptureAuthenticationError, match="no-grad"
    ):
        load_or_capture_lfm2_conditioned_recapture_phase(
            cache,
            request,
            authority=authority,
            checkpoint_root=tmp_path / "work",
            range_provider=provider,
        )
    assert cache.lookup(request.phase_key) is None


def test_publication_copy_failure_does_not_poison_or_recapture(
    tmp_path: Path,
    parts: dict[str, Any],
    authority: Lfm2ConditionedRecapturePhaseAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(parts)
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    calls: list[tuple[int, int]] = []

    def provider(start: int, stop: int) -> Lfm2ChildReplay:
        calls.append((start, stop))
        return _replay(request, parts, start, stop)

    original_copy = recapture.shutil.copyfile
    with monkeypatch.context() as patcher:
        patcher.setattr(
            recapture.shutil,
            "copyfile",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected copy failure")
            ),
        )
        with pytest.raises(OSError, match="injected copy failure"):
            load_or_capture_lfm2_conditioned_recapture_phase(
                cache,
                request,
                authority=authority,
                checkpoint_root=tmp_path / "work",
                range_provider=provider,
            )
    assert recapture.shutil.copyfile is original_copy
    assert cache.lookup(request.phase_key) is None
    assert calls == [(0, 64), (64, 128), (128, 192), (192, 256)]

    def forbidden(_start: int, _stop: int) -> Lfm2ChildReplay:
        raise AssertionError("full work prefix recaptured after copy failure")

    complete = load_or_capture_lfm2_conditioned_recapture_phase(
        cache,
        request,
        authority=authority,
        checkpoint_root=tmp_path / "work",
        range_provider=forbidden,
    )
    assert isinstance(complete, AuthenticatedLfm2ConditionedRecapturePhase)
