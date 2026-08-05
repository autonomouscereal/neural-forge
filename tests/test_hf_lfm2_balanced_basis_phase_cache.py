from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
safetensors_torch = pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.dynamic_fold_phase_cache import (  # noqa: E402
    DynamicFoldPhaseCache,
    FoldPhaseCacheIntegrityError,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
import neural_forge.dynamic_fold_phase_cache as phase_cache_module  # noqa: E402
import neural_forge.hf_lfm2_balanced_basis_phase_cache as basis_cache  # noqa: E402
import neural_forge.hf_lfm2_balanced_circuit_basis as balanced_basis  # noqa: E402
import neural_forge.hf_lfm2_dense_budget_model_fold as dense_fold  # noqa: E402
from neural_forge import (  # noqa: E402
    hf_lfm2_streaming_vocabulary_margin_adapter as margin_adapter,
)
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (  # noqa: E402
    LFM2_BALANCED_BASIS_PHASE,
    Lfm2BalancedBasisPhaseBuildRequest,
    Lfm2BalancedBasisPhaseCacheContext,
    Lfm2ExistingBalancedBasisPhaseReuseRequest,
    PhaseAnchorAuthority,
    load_or_build_lfm2_circuit_balanced_basis,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_balanced_circuit_basis import (  # noqa: E402
    Lfm2CircuitBalancedBasisError,
)
from neural_forge.hf_lfm2_dense_budget_model_fold import (  # noqa: E402
    Lfm2DenseBudgetFoldConfig,
    Lfm2DenseBudgetModelOperator,
    _build_residual_basis,
)
from neural_forge.hf_lfm2_sealed_evidence_set import (  # noqa: E402
    Lfm2SealedEvidenceRequest,
    Lfm2SealedEvidenceSet,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


_ANCHOR_SECRET = b"balanced-basis-phase-test-secret-v1"


def _parent(seed: int = 911):
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _batch(index: int):
    input_ids = torch.tensor(
        [[1, 2, 3, 4 + index]],
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor(
            [[10 + index, 15 + index]],
            dtype=torch.long,
        ),
    }


def _case(
    tmp_path: Path,
    *,
    vocabulary_margin_observability: bool = False,
    progress=None,
    key_id: str = "balanced-basis-test-key-v1",
    secret: bytes = _ANCHOR_SECRET,
    dependency_provider=None,
):
    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="balanced-basis-phase-parent",
        tokenizer_fingerprint=_digest("parent-tokenizer"),
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "trace-cache",
        parent,
        [_batch(index) for index in range(4)],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=2,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    phase_root = tmp_path / "derived-phases"
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=DynamicFoldPhaseCache(phase_root),
        anchor_authority=PhaseAnchorAuthority(key_id=key_id, secret=secret),
        progress=progress,
        dependency_provider=(
            dependency_provider
            if dependency_provider is not None
            else lambda: {"carrier.route": _digest("disabled-carrier-route")}
        ),
    )
    target = {
        "schema": "test-selected-target-v1",
        "source_graph_fingerprint": graph.fingerprint,
        "source_state_fingerprint": _digest("executor-state-binding"),
        "hidden_size": 4,
        "exact_target_unique_parameters": 1_337,
    }
    tokenizer = {
        "schema": "test-tokenizer-binding-v1",
        "parent_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "child_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "bridge_required": False,
    }
    def bindings():
        return {
            "selected_target": target,
            "tokenizer": tokenizer,
        }
    build_request = Lfm2BalancedBasisPhaseBuildRequest(
        adapter_source_sha256=_digest("test-dense-basis-adapter-source"),
        validation_fraction=0.25,
        split_seed="balanced-basis-phase-test-v1",
        max_rows_per_block=8,
        vocabulary_margin_observability=vocabulary_margin_observability,
        vocabulary_margin_top_k=(3 if vocabulary_margin_observability else 8),
    )

    session = cache.selective_authenticated_session()
    return (
        parent,
        graph,
        cache,
        session,
        context,
        phase_root,
        bindings,
        build_request,
    )


def _invoke(case):
    (
        parent,
        graph,
        cache,
        session,
        context,
        _phase_root,
        bindings,
        build_request,
    ) = case
    return load_or_build_lfm2_circuit_balanced_basis(
        context,
        parent,
        cache,
        4,
        circuit_graph=graph,
        selective_session=session,
        build_request=build_request,
        binding_provider=bindings,
        live_vocabulary_candidate_gate_required=False,
    )


def _with_context(case, context):
    values = list(case)
    values[4] = context
    return tuple(values)


def test_balanced_basis_phase_cold_reload_and_hit_are_exact_and_hit_loads_no_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _case(tmp_path)
    parent, graph, cache, session, context, phase_root, *rest = case
    original_live_bindings = basis_cache._live_bindings
    live_binding_calls: list[bool] = []

    def tracked_live_bindings(*args, **kwargs):
        live_binding_calls.append(kwargs.get("reuse_inventory_from") is not None)
        return original_live_bindings(*args, **kwargs)

    monkeypatch.setattr(basis_cache, "_live_bindings", tracked_live_bindings)
    try:
        cold = _invoke(case)
        assert live_binding_calls == [False, True, True]
        assert cold.vocabulary_margin_gate is None
        entry = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())
        assert {
            path.relative_to(entry).as_posix()
            for path in entry.rglob("*")
            if path.is_file()
        } == {
            "complete.json",
            "payload/basis.safetensors",
            "payload/receipt.json",
            "payload/source-anchor.json",
        }
        session.close()
        session = cache.selective_authenticated_session()
        case = (
            parent,
            graph,
            cache,
            session,
            context,
            phase_root,
            *rest,
        )

        def forbidden_forward(*_args, **_kwargs):
            raise AssertionError("basis phase hit executed the parent")

        def forbidden_trace_load(*_args, **_kwargs):
            raise AssertionError("basis phase hit loaded a trace tensor")

        def forbidden_builder():
            raise AssertionError("basis phase hit rebuilt the basis")

        parent.forward = forbidden_forward
        monkeypatch.setattr(
            Lfm2CircuitTraceSelectiveAuthenticatedSession,
            "load_selected_tensors",
            forbidden_trace_load,
        )
        monkeypatch.setattr(
            basis_cache._lfm2_balanced_module,
            "load_lfm2_circuit_trace_selected_rows",
            forbidden_trace_load,
        )
        monkeypatch.setattr(
            basis_cache,
            "build_lfm2_circuit_balanced_basis",
            forbidden_builder,
        )
        live_binding_calls.clear()
        hit = _invoke(case)
        assert live_binding_calls == [False, True]
        assert hit.vocabulary_margin_gate is None
        assert canonical_json_bytes(hit.as_dict()) == canonical_json_bytes(
            cold.as_dict()
        )
        assert canonical_json_bytes(
            hit.basis_result.as_dict()
        ) == canonical_json_bytes(cold.basis_result.as_dict())
        for name in basis_cache._TENSOR_NAMES:
            assert torch.equal(
                getattr(hit, name),
                getattr(cold, name),
            )
            assert getattr(hit, name).dtype == torch.float64
            assert getattr(hit, name).device.type == "cpu"
    finally:
        if not session.closed:
            session.close()


def test_balanced_basis_phase_resealed_semantic_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    session = case[3]
    phase_root = case[5]
    try:
        _invoke(case)
        entry = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())
        receipt_path = entry / "payload" / "receipt.json"
        completion_path = entry / "complete.json"
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        receipt["execution"]["training_steps"] = 1
        receipt_body = dict(receipt)
        receipt_body.pop("receipt_sha256")
        receipt["receipt_sha256"] = canonical_sha256(receipt_body)
        receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")

        completion = json.loads(completion_path.read_text(encoding="ascii"))
        completion["payload_inventory"] = phase_cache_module._payload_inventory(
            entry / "payload"
        )
        completion["result"]["receipt_sha256"] = receipt["receipt_sha256"]
        completion_body = dict(completion)
        completion_body.pop("completion_sha256")
        completion["completion_sha256"] = canonical_sha256(completion_body)
        completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")

        with pytest.raises(
            FoldPhaseCacheIntegrityError,
            match="receipt differs from its live v1 contract",
        ):
            _invoke(case)
    finally:
        session.close()


def test_fully_resealed_all_tensor_and_report_substitution_fails_source_hmac(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _case(tmp_path)
    session = case[3]
    phase_root = case[5]
    try:
        _invoke(case)
        entry = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())
        payload_root = entry / "payload"
        tensor_path = payload_root / "basis.safetensors"
        receipt_path = payload_root / "receipt.json"
        anchor_path = payload_root / "source-anchor.json"
        completion_path = entry / "complete.json"

        tensors = dict(safetensors_torch.load(tensor_path.read_bytes()))
        original_hashes = {
            name: basis_cache._balanced_tensor_sha256(value)
            for name, value in tensors.items()
        }
        permutation = torch.roll(torch.arange(8), shifts=1)
        coordinate_map = torch.eye(8, dtype=torch.float64)[permutation]
        tensors["basis"] = (coordinate_map @ tensors["basis"]).contiguous()
        tensors["balanced_decoder"] = (
            coordinate_map @ tensors["balanced_decoder"]
        ).contiguous()
        tensors["balanced_encoder"] = (
            tensors["balanced_encoder"] @ coordinate_map.T
        ).contiguous()
        for name in (
            "full_weight_gram",
            "controllability_gram",
            "observability_gram",
            "heldout_gram",
        ):
            tensors[name] = (
                coordinate_map @ tensors[name] @ coordinate_map.T
            ).contiguous()
        tensor_payload = safetensors_torch.save(
            {name: tensors[name] for name in basis_cache._TENSOR_NAMES},
            metadata=basis_cache._TENSOR_METADATA,
        )
        tensor_path.write_bytes(tensor_payload)
        descriptors = basis_cache._tensor_descriptors(
            tensors,
            source_width=8,
            target_width=4,
        )
        descriptor_by_name = {value["name"]: value for value in descriptors}
        assert all(
            descriptor_by_name[name]["sha256"] != original_hashes[name]
            for name in basis_cache._TENSOR_NAMES
        )

        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        nested = receipt["balanced_basis_report"]
        report_hash_names = {
            "basis": "basis_sha256",
            "balanced_encoder": "balanced_encoder_sha256",
            "balanced_decoder": "balanced_decoder_sha256",
            "full_weight_gram": "full_weight_gram_sha256",
            "controllability_gram": "controllability_gram_sha256",
            "observability_gram": "observability_gram_sha256",
            "heldout_gram": "heldout_gram_sha256",
        }
        for tensor_name, report_name in report_hash_names.items():
            nested["hashes"][report_name] = descriptor_by_name[tensor_name][
                "sha256"
            ]
        nested["hashes"]["projection_sha256"] = (
            basis_cache._balanced_tensor_sha256(
                tensors["basis"] @ tensors["basis"].T
            )
        )
        nested["whole_weight"]["raw_gram_sha256"] = descriptor_by_name[
            "full_weight_gram"
        ]["sha256"]
        nested["heldout"]["raw_gram_sha256"] = descriptor_by_name[
            "heldout_gram"
        ]["sha256"]
        regularization = nested["regularized_balanced_truncation"][
            "gram_regularization"
        ]
        ridge = torch.eye(8, dtype=torch.float64) * (regularization / 8)
        nested["controllability_blend"]["unregularized_gram_sha256"] = (
            basis_cache._balanced_tensor_sha256(
                tensors["controllability_gram"] - ridge
            )
        )
        nested["controllability_blend"]["regularized_gram_sha256"] = (
            descriptor_by_name["controllability_gram"]["sha256"]
        )
        selection_payload = {
            "schema": basis_cache._balanced_basis_module.BALANCED_CIRCUIT_BASIS_SCHEMA,
            "source_width": nested["source_width"],
            "target_width": nested["target_width"],
            "whole_weight_manifest_sha256": canonical_sha256(
                [
                    value["operator_sha256"]
                    for value in nested["whole_weight"]["operators"]
                ]
            ),
            "fit_manifest_sha256": nested["cached_controllability"][
                "block_manifest_sha256"
            ],
            "observability_manifest_sha256": canonical_sha256(
                [
                    value["evidence_sha256"]
                    for value in nested["downstream_observability"]["evidence"]
                ]
            ),
            "circuit_graph_fingerprint": nested["circuit_ir"][
                "graph_fingerprint"
            ],
            "structural_controllability_weight": nested[
                "controllability_blend"
            ]["whole_weight_role_balanced_weight"],
            "cached_controllability_weight": nested["controllability_blend"][
                "cached_residual_weight"
            ],
            "gram_regularization": regularization,
            "rank_tolerance": receipt["phase_key"]["numeric_contract"][
                "basis_config"
            ]["rank_tolerance"],
            "max_condition_number": receipt["phase_key"]["numeric_contract"][
                "basis_config"
            ]["max_condition_number"],
            "required_observability_kinds": nested["downstream_observability"][
                "required_circuit_kinds"
            ],
            "basis_sha256": descriptor_by_name["basis"]["sha256"],
            "balanced_encoder_sha256": descriptor_by_name[
                "balanced_encoder"
            ]["sha256"],
            "balanced_decoder_sha256": descriptor_by_name[
                "balanced_decoder"
            ]["sha256"],
        }
        nested["hashes"]["selection_sha256"] = canonical_sha256(
            selection_payload
        )
        nested_body = dict(nested)
        nested_body.pop("run_sha256")
        nested["run_sha256"] = canonical_sha256(nested_body)

        wrapper = receipt["basis_report"]
        wrapper["balanced_basis"] = nested
        wrapper["hashes"]["basis_sha256"] = descriptor_by_name["basis"][
            "sha256"
        ]
        wrapper_body = dict(wrapper)
        wrapper_body.pop("result_sha256")
        wrapper["result_sha256"] = canonical_sha256(wrapper_body)
        receipt["balanced_basis_report"] = nested
        receipt["basis_report"] = wrapper
        receipt["tensor_descriptors"] = descriptors
        receipt["tensor_file"]["size_bytes"] = len(tensor_payload)
        receipt["tensor_file"]["sha256"] = basis_cache.hashlib.sha256(
            tensor_payload
        ).hexdigest()
        receipt_body = dict(receipt)
        receipt_body.pop("receipt_sha256")
        receipt["receipt_sha256"] = canonical_sha256(receipt_body)
        receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")

        anchor = json.loads(anchor_path.read_text(encoding="ascii"))
        replacement_result = basis_cache._completion_result(receipt)
        for name in (
            "receipt_sha256",
            "tensor_file_sha256",
            "tensor_inventory_sha256",
            "basis_result_sha256",
            "balanced_run_sha256",
            "basis_map_sha256",
        ):
            anchor[name] = replacement_result[name]
        anchor_body = dict(anchor)
        anchor_body.pop("hmac_sha256")
        anchor["hmac_sha256"] = canonical_sha256(anchor_body)
        anchor_path.write_bytes(canonical_json_bytes(anchor) + b"\n")

        completion = json.loads(completion_path.read_text(encoding="ascii"))
        completion["payload_inventory"] = phase_cache_module._payload_inventory(
            payload_root
        )
        completion["result"] = replacement_result
        completion_body = dict(completion)
        completion_body.pop("completion_sha256")
        completion["completion_sha256"] = canonical_sha256(completion_body)
        completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")

        tensor_loads = []
        original_load_tensors = basis_cache._load_tensors

        def tracked_load_tensors(payload):
            tensor_loads.append(True)
            return original_load_tensors(payload)

        monkeypatch.setattr(basis_cache, "_load_tensors", tracked_load_tensors)
        with pytest.raises(
            FoldPhaseCacheIntegrityError,
            match="source anchor HMAC",
        ):
            _invoke(case)
        assert not tensor_loads

        monkeypatch.setattr(
            basis_cache,
            "_read_source_anchor",
            lambda *_args, **_kwargs: {},
        )
        unanchored = _invoke(case)
        assert tensor_loads
        for name in basis_cache._TENSOR_NAMES:
            assert torch.equal(getattr(unanchored, name), tensors[name])
    finally:
        session.close()


def test_explicit_historical_phase_reuse_rebinds_live_inventory_without_building(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _case(tmp_path)
    session = case[3]
    context = case[4]
    phase_root = case[5]
    try:
        cold = _invoke(case)
        phase_key_sha256 = next(
            (phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir()
        ).name
        progress_events = []

        def collect_progress(stage, details):
            progress_events.append((stage, dict(details)))

        reuse_context = Lfm2BalancedBasisPhaseCacheContext(
            cache=context.cache,
            anchor_authority=context.anchor_authority,
            progress=collect_progress,
            dependency_provider=context.dependency_provider,
            existing_phase_reuse_request=(
                Lfm2ExistingBalancedBasisPhaseReuseRequest(
                    phase_key_sha256=phase_key_sha256,
                    expected_map_sha256=cold.map_sha256,
                )
            ),
        )
        values = list(case)
        values[4] = reuse_context
        values[7] = replace(
            case[7],
            adapter_source_sha256=_digest("new-dense-consumer-source"),
        )
        monkeypatch.setattr(
            basis_cache,
            "build_lfm2_circuit_balanced_basis",
            lambda *_args, **_kwargs: pytest.fail(
                "explicit historical phase reuse attempted a cold build"
            ),
        )
        reused = _invoke(tuple(values))
        assert reused.map_sha256 == cold.map_sha256
        assert reused.as_dict() == cold.as_dict()
        assert reused.inventory.operators
        assert reused.inventory.as_dict() == cold.inventory.as_dict()
        assert reused.circuit_graph.fingerprint == cold.circuit_graph.fingerprint
        provenance = dict(reused.reuse_provenance)
        assert provenance["historical_phase_key_sha256"] == phase_key_sha256
        assert provenance["historical_map_sha256"] == cold.map_sha256
        assert provenance["historical_artifact_rewritten"] is False
        assert provenance["cache_miss_build_allowed"] is False
        assert provenance["semantic_bindings_exact"] is True
        assert provenance["basis_math_settings_exact"] is True
        assert provenance["trace_tensor_loads"] == 0
        assert provenance["parent_forward_calls"] == 0
        assert provenance["training_steps"] == 0
        assert provenance["reuse_sha256"] == canonical_sha256(
            {key: value for key, value in provenance.items() if key != "reuse_sha256"}
        )
        assert len(progress_events) == 3
        reuse_stage, reuse_progress = progress_events[-1]
        assert reuse_stage == "balanced_basis_existing_phase_reuse_hit"
        assert set(reuse_progress).isdisjoint(
            {
                "schema",
                "timestamp_utc",
                "configuration_sha256",
                "sequence",
                "stage",
            }
        )
        assert all(
            value is None or isinstance(value, (bool, int, float, str))
            for value in reuse_progress.values()
        )
        assert reuse_progress == {
            "basis_reuse_provenance_sha256": provenance["reuse_sha256"],
            "basis_reuse_historical_phase_key_sha256": phase_key_sha256,
            "basis_reuse_historical_completion_sha256": provenance[
                "historical_completion_sha256"
            ],
            "basis_reuse_historical_map_sha256": cold.map_sha256,
            "basis_reuse_semantic_bindings_exact": True,
            "basis_reuse_math_settings_exact": True,
            "basis_reuse_historical_artifact_rewritten": False,
            "basis_reuse_cache_miss_build_allowed": False,
            "basis_reuse_trace_tensor_loads": 0,
            "basis_reuse_parent_forward_calls": 0,
            "basis_reuse_training_steps": 0,
            "basis_reuse_optimizer_steps": 0,
            "basis_reuse_backward_calls": 0,
        }
        assert tuple(
            (phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir()
        ) == (
            phase_root / LFM2_BALANCED_BASIS_PHASE / phase_key_sha256,
        )
    finally:
        session.close()


def test_explicit_historical_phase_reuse_rejects_changed_basis_math(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    session = case[3]
    context = case[4]
    phase_root = case[5]
    try:
        _invoke(case)
        phase_key_sha256 = next(
            (phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir()
        ).name
        values = list(case)
        values[4] = Lfm2BalancedBasisPhaseCacheContext(
            cache=context.cache,
            anchor_authority=context.anchor_authority,
            dependency_provider=context.dependency_provider,
            existing_phase_reuse_request=(
                Lfm2ExistingBalancedBasisPhaseReuseRequest(
                    phase_key_sha256=phase_key_sha256,
                )
            ),
        )
        values[7] = replace(
            case[7],
            adapter_source_sha256=_digest("new-dense-consumer-source"),
            validation_fraction=0.5,
        )
        with pytest.raises(
            FoldPhaseCacheIntegrityError,
            match="different basis-math settings",
        ):
            _invoke(tuple(values))
    finally:
        session.close()


def test_explicit_historical_phase_reuse_miss_never_creates_or_builds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case = _case(tmp_path / "source")
    session = case[3]
    original_context = case[4]
    missing_root = tmp_path / "missing-phase-cache"
    try:
        values = list(case)
        values[4] = Lfm2BalancedBasisPhaseCacheContext(
            cache=DynamicFoldPhaseCache(missing_root),
            anchor_authority=original_context.anchor_authority,
            dependency_provider=original_context.dependency_provider,
            existing_phase_reuse_request=(
                Lfm2ExistingBalancedBasisPhaseReuseRequest(
                    phase_key_sha256="f" * 64,
                )
            ),
        )
        monkeypatch.setattr(
            basis_cache,
            "build_lfm2_circuit_balanced_basis",
            lambda *_args, **_kwargs: pytest.fail(
                "missing historical phase attempted a cold build"
            ),
        )
        with pytest.raises(
            basis_cache.Lfm2BalancedBasisPhaseCacheError,
            match="absent.*forbids building",
        ):
            _invoke(tuple(values))
        assert not missing_root.exists()
    finally:
        session.close()


def test_explicit_historical_phase_reuse_rejects_live_candidate_gate(
    tmp_path: Path,
) -> None:
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "phase-cache"),
        anchor_authority=PhaseAnchorAuthority(
            key_id="reuse-candidate-gate-v1",
            secret=_ANCHOR_SECRET,
        ),
        existing_phase_reuse_request=(
            Lfm2ExistingBalancedBasisPhaseReuseRequest(
                phase_key_sha256="f" * 64,
            )
        ),
    )
    with pytest.raises(
        basis_cache.Lfm2BalancedBasisPhaseCacheError,
        match="cannot restore the live vocabulary candidate gate",
    ):
        load_or_build_lfm2_circuit_balanced_basis(
            context,
            object(),
            object(),
            4,
            circuit_graph=object(),
            selective_session=None,
            build_request=Lfm2BalancedBasisPhaseBuildRequest(
                adapter_source_sha256=_digest("reuse-gate-adapter"),
            ),
            binding_provider=lambda: pytest.fail("reuse gate read bindings"),
            live_vocabulary_candidate_gate_required=True,
        )
    assert not context.cache.root.exists()


def test_anchor_key_rotation_separates_phase_identity_and_wrong_secret_fails(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    session = case[3]
    context = case[4]
    phase_root = case[5]
    try:
        _invoke(case)
        wrong_secret_context = Lfm2BalancedBasisPhaseCacheContext(
            cache=context.cache,
            anchor_authority=PhaseAnchorAuthority(
                key_id=context.anchor_authority.key_id,
                secret=b"wrong-balanced-basis-secret-key-0001",
            ),
            dependency_provider=context.dependency_provider,
        )
        with pytest.raises(
            FoldPhaseCacheIntegrityError,
            match="source anchor HMAC",
        ):
            _invoke(_with_context(case, wrong_secret_context))

        rotated_context = Lfm2BalancedBasisPhaseCacheContext(
            cache=context.cache,
            anchor_authority=PhaseAnchorAuthority(
                key_id="balanced-basis-test-key-v2",
                secret=_ANCHOR_SECRET,
            ),
            dependency_provider=context.dependency_provider,
        )
        _invoke(_with_context(case, rotated_context))
        entries = tuple((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())
        assert len(entries) == 2
        assert len({entry.name for entry in entries}) == 2
    finally:
        session.close()


def test_anchor_secret_is_private_and_never_persisted(tmp_path: Path) -> None:
    case = _case(tmp_path)
    session = case[3]
    context = case[4]
    phase_root = case[5]
    try:
        assert "secret" not in repr(context.anchor_authority).casefold()
        assert not hasattr(context.anchor_authority, "secret")
        _invoke(case)
        entry = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())
        for path in entry.rglob("*"):
            if path.is_file():
                assert _ANCHOR_SECRET not in path.read_bytes()
    finally:
        session.close()


def test_fresh_graph_reinspection_authenticates_executor_all_frozen_state(
    tmp_path: Path,
) -> None:
    observed = []

    def progress(stage, details):
        observed.append((stage, dict(details)))

    case = _case(tmp_path, progress=progress)
    parent, supplied_graph, _cache, session, _context, phase_root, *_rest = case
    try:
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        result = _invoke(case)
        assert result.circuit_graph.fingerprint != supplied_graph.fingerprint
        assert result.circuit_graph.as_structural_topology_dict() == (
            supplied_graph.as_structural_topology_dict()
        )

        entry = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())
        receipt = json.loads(
            (entry / "payload" / "receipt.json").read_text(encoding="ascii")
        )
        graph_binding = receipt["bindings"]["circuit_graph"]
        assert graph_binding["supplied_fingerprint"] == supplied_graph.fingerprint
        assert graph_binding["runtime_fingerprint"] == result.circuit_graph.fingerprint
        assert graph_binding["fingerprint"] == result.circuit_graph.fingerprint
        assert graph_binding["structural_topology_fingerprint"] == (
            supplied_graph.structural_topology_fingerprint
        )
        assert graph_binding["executor_freeze_normalized"] is True
        assert graph_binding["structural_topology_exactly_preserved"] is True
        assert graph_binding["structural_topology_excluded_parameter_fields"] == [
            "trainable"
        ]
        assert graph_binding["runtime_trainability"][
            "all_parameters_frozen"
        ] is True
        assert graph_binding["runtime_trainability"][
            "trainable_parameter_spec_count"
        ] == 0
        dependencies = receipt["phase_key"]["dependency_sha256s"]
        assert dependencies["source_graph"] == supplied_graph.fingerprint
        assert dependencies["runtime_source_graph"] == result.circuit_graph.fingerprint
        assert dependencies["source_structural_topology"] == (
            supplied_graph.structural_topology_fingerprint
        )
        assert any(
            stage == "balanced_basis_live_graph_authenticated"
            and details["executor_freeze_normalized"] is True
            and details["runtime_all_parameters_frozen"] is True
            for stage, details in observed
        )
    finally:
        session.close()


def test_fresh_graph_reinspection_rejects_partial_trainability_drift(
    tmp_path: Path,
) -> None:
    observed = []

    def progress(stage, details):
        observed.append((stage, dict(details)))

    case = _case(tmp_path, progress=progress)
    parent = case[0]
    session = case[3]
    try:
        next(parent.parameters()).requires_grad_(False)
        with pytest.raises(
            basis_cache.Lfm2BalancedBasisPhaseCacheError,
            match="all-frozen contract",
        ):
            _invoke(case)
        mismatch = next(
            details
            for stage, details in observed
            if stage == "balanced_basis_live_graph_mismatch"
        )
        assert mismatch["mismatch_kind"] == "trainability_not_executor_freeze"
        assert mismatch["runtime_all_parameters_frozen"] is False
        assert mismatch["runtime_trainable_parameter_spec_count"] > 0
    finally:
        session.close()


def test_fresh_graph_reinspection_rejects_structural_drift_with_safe_digests(
    tmp_path: Path,
) -> None:
    observed = []

    def progress(stage, details):
        observed.append((stage, dict(details)))

    case = _case(tmp_path, progress=progress)
    parent = case[0]
    session = case[3]
    try:
        parent.config.max_position_embeddings = (
            int(parent.config.max_position_embeddings) + 1
        )
        with pytest.raises(
            basis_cache.Lfm2BalancedBasisPhaseCacheError,
            match="differs structurally",
        ):
            _invoke(case)
        mismatch = next(
            details
            for stage, details in observed
            if stage == "balanced_basis_live_graph_mismatch"
        )
        assert mismatch["mismatch_kind"] == "structural_topology"
        for name in (
            "supplied_graph_fingerprint",
            "runtime_graph_fingerprint",
            "supplied_structural_topology_fingerprint",
            "runtime_structural_topology_fingerprint",
        ):
            assert len(mismatch[name]) == 64
    finally:
        session.close()


def test_cold_build_forwards_sampled_row_selection_progress_and_restores_logger(
    tmp_path: Path,
) -> None:
    observed = []

    def progress(stage, details):
        observed.append((stage, dict(details)))

    logger = basis_cache._lfm2_balanced_module._LOGGER
    original_level = logger.level
    original_disabled = logger.disabled
    original_handlers = tuple(logger.handlers)
    case = _case(tmp_path, progress=progress)
    session = case[3]
    try:
        _invoke(case)
        selected = [
            details
            for stage, details in observed
            if stage == "balanced_basis_row_selection_progress"
        ]
        assert selected
        assert all("schema" not in details for details in selected)
        assert all(
            details["row_selection_schema"]
            == basis_cache._lfm2_balanced_module._STREAMING_ROW_SELECTION_PROGRESS_SCHEMA
            for details in selected
        )
        assert any(details["completed_batch_count"] == 1 for details in selected)
        assert any(
            details["completed_batch_count"] == details["batch_count"]
            for details in selected
        )
    finally:
        session.close()
    assert logger.level == original_level
    assert logger.disabled is original_disabled
    assert tuple(logger.handlers) == original_handlers


def test_progress_logger_state_is_restored_when_cold_builder_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed = []
    logger = basis_cache._lfm2_balanced_module._LOGGER
    original_level = logger.level
    original_disabled = logger.disabled
    original_handlers = tuple(logger.handlers)
    case = _case(
        tmp_path,
        progress=lambda stage, details: observed.append((stage, dict(details))),
    )
    session = case[3]

    def fail_after_progress(*_args, **_kwargs):
        basis_cache._lfm2_balanced_module._emit_streaming_row_selection_progress(
            {
                "schema": basis_cache._lfm2_balanced_module._STREAMING_ROW_SELECTION_PROGRESS_SCHEMA,
                "split": "fit",
                "phase": "prefill",
                "batch_index": 0,
                "completed_batch_count": 1,
                "batch_count": 2,
                "selected_tensor_count": 1,
                "selected_row_count": 1,
                "retained_fp64_logical_bytes": 64,
                "phase_buffer_fp64_bytes": 64,
                "selected_source_payload_bytes": 32,
                "source_slice_or_hash_overhead_bytes": 0,
                "maximum_fp64_conversion_bytes": 64,
                "maximum_fp64_finiteness_scratch_bytes": 1,
                "maximum_block_fp64_bytes": 64,
                "block_owned_clone_count": 1,
                "block_hash_clone_count": 1,
                "block_hash_tobytes_copy_count": 1,
                "source_row_count": 2,
                "selected_logical_bytes": 32,
                "source_logical_bytes": 64,
                "maximum_materialized_source_rows": 1,
                "safetensors_get_slice_calls": 1,
                "coalesced_slice_count": 1,
                "authenticated_row_slice_path": True,
                "whole_tensor_cpu_clone_preceded_row_selection": False,
                "payload_released": True,
                "phase_windowed_fp64_retention": True,
                "all_snapshot_blocks_retained": False,
                "telemetry_excluded_from_basis_evidence_hashes": True,
            }
        )
        raise RuntimeError("synthetic cold-build failure")

    monkeypatch.setattr(
        basis_cache,
        "build_lfm2_circuit_balanced_basis",
        fail_after_progress,
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic cold-build failure"):
            _invoke(case)
    finally:
        session.close()
    assert any(
        stage == "balanced_basis_row_selection_progress"
        for stage, _details in observed
    )
    assert logger.level == original_level
    assert logger.disabled is original_disabled
    assert tuple(logger.handlers) == original_handlers


def test_live_external_dependency_provider_drift_fails_before_publication(
    tmp_path: Path,
) -> None:
    calls = 0

    def dependencies():
        nonlocal calls
        calls += 1
        return {
            "carrier.route": _digest(
                "stable-route" if calls == 1 else "changed-route"
            )
        }

    case = _case(tmp_path, dependency_provider=dependencies)
    session = case[3]
    phase_root = case[5]
    try:
        with pytest.raises(
            basis_cache.Lfm2BalancedBasisPhaseCacheError,
            match="changed during cold build",
        ):
            _invoke(case)
        phase_collection = phase_root / LFM2_BALANCED_BASIS_PHASE
        assert not phase_collection.exists() or not tuple(phase_collection.iterdir())
    finally:
        session.close()


def test_nonrequired_live_vocabulary_gate_is_discarded_with_exact_cached_report(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, vocabulary_margin_observability=True)
    session = case[3]
    try:
        cold = _invoke(case)
        assert cold.vocabulary_margin_gate is None
        assert cold.as_dict()["vocabulary_margin_observability"]["enabled"] is True

        def forbidden_builder():
            raise AssertionError("vocabulary-aware basis phase hit rebuilt")

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                basis_cache,
                "build_lfm2_circuit_balanced_basis",
                forbidden_builder,
            )
            hit = _invoke(case)
        assert hit.vocabulary_margin_gate is None
        assert canonical_json_bytes(hit.as_dict()) == canonical_json_bytes(
            cold.as_dict()
        )
    finally:
        session.close()


def test_live_vocabulary_candidate_gate_explicitly_bypasses_v1_cache(
    tmp_path: Path,
) -> None:
    marker = object()
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "phase-cache"),
        anchor_authority=PhaseAnchorAuthority(
            key_id="bypass-key-v1",
            secret=_ANCHOR_SECRET,
        ),
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            basis_cache,
            "build_lfm2_circuit_balanced_basis",
            lambda *_args, **_kwargs: marker,
        )
        result = load_or_build_lfm2_circuit_balanced_basis(
            context,
            object(),
            object(),
            4,
            circuit_graph=object(),
            selective_session=None,
            build_request=Lfm2BalancedBasisPhaseBuildRequest(
                adapter_source_sha256=_digest("bypass-adapter-source"),
            ),
            binding_provider=lambda: pytest.fail("bypass authenticated bindings"),
            live_vocabulary_candidate_gate_required=True,
        )
    assert result is marker
    assert not context.cache.root.exists()


def test_dense_basis_boundary_exposes_and_routes_optional_phase_cache_hook(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assert "residual_basis_phase_cache_context" in inspect.signature(
        Lfm2DenseBudgetModelOperator
    ).parameters
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "phase-cache"),
        anchor_authority=PhaseAnchorAuthority(
            key_id="dense-hook-key-v1",
            secret=_ANCHOR_SECRET,
        ),
    )
    marker = object()
    observed = {}

    def delegated(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return marker

    monkeypatch.setattr(
        dense_fold,
        "load_or_build_lfm2_circuit_balanced_basis",
        delegated,
    )
    def provider():
        return {
            "selected_target": {
                "source_graph_fingerprint": "a" * 64,
                "hidden_size": 4,
            },
            "tokenizer": {"schema": "test"},
        }
    result = _build_residual_basis(
        object(),
        object(),
        object(),
        4,
        Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="circuit_balanced",
            run_closed_loop=False,
        ),
        phase_cache_context=context,
        phase_cache_binding_provider=provider,
    )
    assert result is marker
    assert observed["args"][0] is context
    assert observed["kwargs"]["binding_provider"] is provider
    assert observed["kwargs"]["live_vocabulary_candidate_gate_required"] is False
    build_request = observed["kwargs"]["build_request"]
    assert isinstance(build_request, Lfm2BalancedBasisPhaseBuildRequest)
    assert build_request.validation_fraction == 0.25
    assert len(build_request.adapter_source_sha256) == 64


def test_dense_no_cache_path_performs_no_phase_source_file_io(
    monkeypatch,
) -> None:
    marker = object()
    monkeypatch.setattr(
        basis_cache._lfm2_balanced_module,
        "build_lfm2_circuit_balanced_basis",
        lambda *_args, **_kwargs: marker,
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("no-cache path hashed a phase source file"),
    )
    result = _build_residual_basis(
        object(),
        object(),
        object(),
        4,
        Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="circuit_balanced",
            run_closed_loop=False,
        ),
    )
    assert result is marker


# --------------------------------------------------------------------------
# Base + on-policy extension sealed evidence
# --------------------------------------------------------------------------


def _sealed_cache(tmp_path: Path, name: str, parent, indices):
    return build_lfm2_circuit_trace_cache(
        tmp_path / name,
        parent,
        [_batch(index) for index in indices],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=2,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )


def _evidence_request(
    cache,
    *,
    extension: bool,
    manifest: str | None = None,
    splits=None,
):
    splits = splits or (
        tuple("fit" for _ in range(cache.batch_count))
        if extension
        else ("selection", *("fit" for _ in range(cache.batch_count - 1)))
    )
    return Lfm2SealedEvidenceRequest(
        cache=cache,
        splits=splits,
        split_sha256=canonical_sha256(
            {"cache_key": cache.cache_key, "splits": list(splits)}
        ),
        round_manifest_sha256=(
            (manifest or _digest("round-manifest-0")) if extension else None
        ),
        all_rows_are_fit=extension,
    )


def _evidence_case(
    tmp_path: Path,
    *,
    extension_indices=(10, 11),
    extension_name: str = "extension-cache",
    round_manifest: str | None = None,
    base_splits=None,
    progress=None,
):
    """One base+extension merged evidence handle in ``_case`` shape."""

    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="balanced-basis-phase-parent",
        tokenizer_fingerprint=_digest("parent-tokenizer"),
    )
    base = _sealed_cache(tmp_path, "base-cache", parent, range(4))
    extension = _sealed_cache(
        tmp_path,
        extension_name,
        parent,
        extension_indices,
    )
    evidence = Lfm2SealedEvidenceSet.open(
        _evidence_request(base, extension=False, splits=base_splits),
        (
            _evidence_request(
                extension,
                extension=True,
                manifest=round_manifest,
            ),
        ),
    )
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "derived-phases"),
        anchor_authority=PhaseAnchorAuthority(
            key_id="merged-evidence-test-key-v1",
            secret=_ANCHOR_SECRET,
        ),
        progress=progress,
    )
    target = {
        "schema": "test-selected-target-v1",
        "source_graph_fingerprint": graph.fingerprint,
        "source_state_fingerprint": _digest("executor-state-binding"),
        "hidden_size": 4,
        "exact_target_unique_parameters": 1_337,
    }
    tokenizer = {
        "schema": "test-tokenizer-binding-v1",
        "parent_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "child_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "bridge_required": False,
    }

    def bindings():
        return {"selected_target": target, "tokenizer": tokenizer}

    build_request = Lfm2BalancedBasisPhaseBuildRequest(
        adapter_source_sha256=_digest("test-dense-basis-adapter-source"),
        validation_fraction=0.25,
        split_seed="balanced-basis-phase-test-v1",
        max_rows_per_block=8,
    )
    case = (
        parent,
        graph,
        evidence.merged_cache(),
        evidence.merged_session(),
        context,
        tmp_path / "derived-phases",
        bindings,
        build_request,
    )
    return case, evidence


def _case_phase_key(case):
    """The phase key one case would look up, without building a basis."""

    parent, graph, cache, session, context, _root, bindings, build_request = case
    live = basis_cache._live_bindings(
        parent,
        graph,
        cache,
        session,
        4,
        bindings,
        context,
    )
    return basis_cache._phase_key(
        live,
        4,
        basis_cache._canonical_mapping(
            build_request.as_basis_config(),
            "basis_config",
        ),
        context.anchor_authority,
    )


def test_plain_trace_cache_binding_is_byte_identical_to_the_pre_merge_form(
    tmp_path: Path,
) -> None:
    """A single sealed cache binds exactly what it bound before merged support.

    The reference body below is the pre-change ``_trace_cache_binding``
    construction verbatim.  Every phase entry sealed by that revision stays
    readable only while this holds.
    """

    case = _case(tmp_path)
    cache, session = case[2], case[3]
    try:
        manifest = cache.manifest
        completion = basis_cache._canonical_mapping(
            dict(cache._completion),
            "trace cache completion",
        )
        reference = {
            "cache_key": cache.cache_key,
            "manifest_sha256": manifest["manifest_sha256"],
            "completion_sha256": completion["completion_sha256"],
            "parent_identity_sha256": cache.parent_fingerprint,
            "corpus_sha256": cache.corpus_fingerprint,
            "layer_topology_sha256": cache.layer_topology_fingerprint,
            "implementation_sha256": cache.implementation_fingerprint,
            "collector_source_sha256": manifest["implementation"][
                "collector_source_sha256"
            ],
            "canonical_contract_source_sha256": manifest["implementation"][
                "canonical_contract_source_sha256"
            ],
            "transformers_lfm2_source_sha256": manifest["implementation"][
                "transformers_lfm2_source_sha256"
            ],
            "execution_sha256": manifest["execution"]["execution_sha256"],
            "capture_selection_sha256": manifest["capture"]["selection_sha256"],
            "selective_session_sha256": session.evidence_sha256,
            "batch_count": cache.batch_count,
        }
        binding = basis_cache._trace_cache_binding(cache, session)
        assert canonical_json_bytes(binding) == canonical_json_bytes(reference)
        assert basis_cache._MERGED_EVIDENCE_BINDING_FIELD not in binding
        key = _case_phase_key(case).as_dict()
        # The merged-evidence dependencies exist only for merged evidence, so a
        # single-cache key keeps exactly the dependency names it always had.
        assert basis_cache._MERGED_EVIDENCE_BINDING_FIELD not in (
            key["dependency_sha256s"]
        )
        assert "on_policy_evidence_identity" not in key["dependency_sha256s"]
    finally:
        session.close()


def test_merged_sealed_evidence_publishes_a_phase_binding_every_member(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The exact r13 failure shape: merged evidence through the phase cache."""

    events: list[tuple[str, dict]] = []
    case, evidence = _evidence_case(
        tmp_path,
        progress=lambda stage, details: events.append((stage, dict(details))),
    )
    phase_root = case[5]
    try:
        cold = _invoke(case)
        assert cold.vocabulary_margin_gate is None
        key = _case_phase_key(case)
        dependencies = key.as_dict()["dependency_sha256s"]
        assert dependencies["trace_cache"] == evidence.cache_key
        assert (
            dependencies["on_policy_evidence_identity"]
            == evidence.evidence_identity_sha256
        )
        merged_dependency = dependencies[basis_cache._MERGED_EVIDENCE_BINDING_FIELD]
        assert len(merged_dependency) == 64

        entry = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())
        receipt = json.loads(
            (entry / "payload" / "receipt.json").read_text(encoding="ascii")
        )
        document = receipt["bindings"]["trace_cache"][
            basis_cache._MERGED_EVIDENCE_BINDING_FIELD
        ]
        assert receipt["phase_key_sha256"] == key.phase_key_sha256
        assert document["schema"] == (
            basis_cache.LFM2_BALANCED_BASIS_MERGED_EVIDENCE_SCHEMA
        )
        assert document["evidence_set_cache_key"] == evidence.cache_key
        assert document["base_cache_key"] == evidence.base_cache.cache_key
        assert document["member_count"] == 2
        assert document["batch_count"] == evidence.batch_count
        assert [
            member["trace_cache"]["cache_key"]
            for member in document["ordered_members"]
        ] == [member.cache_key for member in evidence.members]
        assert [
            member["trace_cache"]["implementation_sha256"]
            for member in document["ordered_members"]
        ] == [member.cache.implementation_fingerprint for member in evidence.members]
        assert document["ordered_members"][1]["round_manifest_sha256"] == (
            evidence.round_manifest_sha256[0]
        )
        assert document["ordered_members"][1]["all_rows_are_fit_rows"] is True
        assert document["extension_batch_indices"] == list(
            evidence.extension_batch_indices
        )
        assert (
            canonical_sha256(
                {
                    name: value
                    for name, value in document.items()
                    if name
                    != f"{basis_cache._MERGED_EVIDENCE_BINDING_FIELD}_sha256"
                }
            )
            == merged_dependency
        )
        bound = [
            details
            for stage, details in events
            if stage == "balanced_basis_merged_evidence_bound"
        ]
        assert bound and bound[0]["evidence_set_cache_key"] == evidence.cache_key
        assert bound[0]["member_count"] == 2

        def forbidden_builder(*_args, **_kwargs):
            raise AssertionError("merged evidence phase hit rebuilt the basis")

        monkeypatch.setattr(
            basis_cache,
            "build_lfm2_circuit_balanced_basis",
            forbidden_builder,
        )
        hit = _invoke(case)
        assert canonical_json_bytes(hit.as_dict()) == canonical_json_bytes(
            cold.as_dict()
        )
        for name in basis_cache._TENSOR_NAMES:
            assert torch.equal(getattr(hit, name), getattr(cold, name))
    finally:
        evidence.close()


def test_merged_evidence_phase_key_binds_every_member_identity(
    tmp_path: Path,
) -> None:
    """No phase entry can be reused across a different evidence composition."""

    baseline, evidence = _evidence_case(tmp_path / "baseline")
    other_extension, other_evidence = _evidence_case(
        tmp_path / "other-extension",
        extension_indices=(12, 13),
    )
    other_manifest, manifest_evidence = _evidence_case(
        tmp_path / "other-manifest",
        round_manifest=_digest("round-manifest-1"),
    )
    # The evidence-set cache key covers member cache keys and the round
    # identity but not the split assignment, so this composition is the one
    # that proves the binding is strictly stronger than that key alone.
    other_split, split_evidence = _evidence_case(
        tmp_path / "other-split",
        base_splits=("fit", "selection", "fit", "fit"),
    )
    try:
        baseline_key = _case_phase_key(baseline).phase_key_sha256
        assert (
            split_evidence.cache_key == evidence.cache_key
            and split_evidence.base.split_sha256 != evidence.base.split_sha256
        )
        keys = {
            "other_extension_cache": _case_phase_key(
                other_extension
            ).phase_key_sha256,
            "other_round_manifest": _case_phase_key(
                other_manifest
            ).phase_key_sha256,
            "other_base_split": _case_phase_key(other_split).phase_key_sha256,
        }
        assert baseline_key not in keys.values()
        assert len(set(keys.values())) == len(keys)
        # Two different compositions never collide, and neither collides with
        # the base-only solve of the same base cache.
        base_only = (
            baseline[0],
            baseline[1],
            evidence.base_cache,
            evidence.base.session,
            baseline[4],
            baseline[5],
            baseline[6],
            baseline[7],
        )
        assert _case_phase_key(base_only).phase_key_sha256 != baseline_key
    finally:
        evidence.close()
        other_evidence.close()
        manifest_evidence.close()
        split_evidence.close()


def test_merged_evidence_binding_fails_closed_on_a_foreign_handle_pair(
    tmp_path: Path,
) -> None:
    case, evidence = _evidence_case(tmp_path)
    merged_cache, merged_session = case[2], case[3]
    try:
        with pytest.raises(
            basis_cache.Lfm2BalancedBasisPhaseCacheError,
            match="requires its merged authenticated session",
        ):
            basis_cache._trace_cache_binding(merged_cache, evidence.base.session)
        with pytest.raises(
            basis_cache.Lfm2BalancedBasisPhaseCacheError,
            match="A selective authenticated trace session is required",
        ):
            basis_cache._trace_cache_binding(evidence.base_cache, merged_session)
        with pytest.raises(
            basis_cache.Lfm2BalancedBasisPhaseCacheError,
            match="trace_cache must be Lfm2CircuitTraceCache",
        ):
            basis_cache._trace_cache_binding(object(), merged_session)
    finally:
        evidence.close()


def test_dense_residual_basis_routes_merged_evidence_through_the_phase_cache(
    tmp_path: Path,
) -> None:
    """The r13 chain: resolved dense handles into the PHASE-CACHED builder.

    ``_resolve_dense_evidence_handles`` hands a writer the merged cache and the
    merged session whenever a round contributed extension evidence, and
    ``_build_residual_basis`` routes to the phase-cached builder whenever a
    phase-cache context exists.  That pair is what failed closed with
    ``trace_cache must be Lfm2CircuitTraceCache``.
    """

    case, evidence = _evidence_case(tmp_path)
    parent, graph, merged_cache, merged_session, context, phase_root = case[:6]
    bindings = case[6]
    try:
        resolved_cache, resolved_session = dense_fold._resolve_dense_evidence_handles(
            evidence.base_cache,
            evidence.base.session,
            evidence,
        )
        assert resolved_cache is merged_cache
        assert resolved_session is merged_session
        basis = _build_residual_basis(
            parent,
            graph,
            resolved_cache,
            4,
            Lfm2DenseBudgetFoldConfig(
                residual_basis_mode="circuit_balanced",
                run_closed_loop=False,
                circuit_validation_fraction=0.25,
                circuit_split_seed="balanced-basis-phase-test-v1",
                circuit_max_rows_per_block=8,
            ),
            selective_session=resolved_session,
            phase_cache_context=context,
            phase_cache_binding_provider=bindings,
        )
        assert basis.source_width == 8
        assert basis.target_width == 4
        entry = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir())
        receipt = json.loads(
            (entry / "payload" / "receipt.json").read_text(encoding="ascii")
        )
        document = receipt["bindings"]["trace_cache"][
            basis_cache._MERGED_EVIDENCE_BINDING_FIELD
        ]
        assert document["evidence_set_cache_key"] == evidence.cache_key
        assert [
            member["trace_cache"]["cache_key"]
            for member in document["ordered_members"]
        ] == [member.cache_key for member in evidence.members]
    finally:
        evidence.close()


def test_merged_evidence_margin_modes_are_exactly_what_the_adapters_support(
    tmp_path: Path,
) -> None:
    """Both margin arms run on merged evidence through the phase cache.

    ``build_lfm2_streaming_vocabulary_margin_adapter_v2`` used to authenticate
    exactly one sealed cache's ``manifest_sha256``, which a merged corpus
    deliberately does not have, so the v2 arm was refused by name.  It now
    authenticates every member's own sealed digest instead, so both arms publish
    a phase entry and only a handle that is neither a sealed session nor a
    merged one is refused.
    """

    case, evidence = _evidence_case(tmp_path)
    parent, graph, cache, session, context = case[:5]
    bindings = case[6]
    try:
        observability = load_or_build_lfm2_circuit_balanced_basis(
            context,
            parent,
            cache,
            4,
            circuit_graph=graph,
            selective_session=session,
            build_request=Lfm2BalancedBasisPhaseBuildRequest(
                adapter_source_sha256=_digest("margin-observability-adapter"),
                validation_fraction=0.25,
                split_seed="balanced-basis-phase-test-v1",
                max_rows_per_block=8,
                vocabulary_margin_observability=True,
                vocabulary_margin_top_k=3,
            ),
            binding_provider=bindings,
            live_vocabulary_candidate_gate_required=False,
        )
        assert observability.target_width == 4
        streaming = load_or_build_lfm2_circuit_balanced_basis(
            context,
            parent,
            cache,
            4,
            circuit_graph=graph,
            selective_session=session,
            build_request=Lfm2BalancedBasisPhaseBuildRequest(
                adapter_source_sha256=_digest("margin-streaming-adapter"),
                validation_fraction=0.25,
                split_seed="balanced-basis-phase-test-v1",
                max_rows_per_block=8,
                vocabulary_margin_observability=True,
                vocabulary_margin_top_k=3,
                streaming_vocabulary_margin_v2=True,
                streaming_vocabulary_margin_pair_chunk_size=8,
            ),
            binding_provider=bindings,
            live_vocabulary_candidate_gate_required=False,
        )
        assert streaming.target_width == 4
        # The two arms are different bases, so the phase cache holds two
        # entries, and the streaming one names the evidence it was solved on.
        assert (
            streaming.basis_result.map_sha256
            != observability.basis_result.map_sha256
        )
        margin = streaming.as_dict()["vocabulary_margin_observability"]
        assert margin["streaming_v2"] is True
        assert margin["adapter"]["cache_authentication"][
            margin_adapter._MERGED_EVIDENCE_FIELD
        ]["evidence_set_cache_key"] == evidence.cache_key
        # Anything that is neither a sealed selective session nor a merged one
        # is still refused by name, before any evidence is read.
        with pytest.raises(
            Lfm2CircuitBalancedBasisError,
            match="one sealed cache session or one merged sealed evidence",
        ):
            balanced_basis.build_lfm2_circuit_balanced_basis(
                parent,
                cache,
                4,
                circuit_graph=graph,
                selective_session=object(),
                vocabulary_margin_observability=True,
                vocabulary_margin_top_k=3,
                streaming_vocabulary_margin_v2=True,
            )
    finally:
        evidence.close()


def test_plain_cache_streaming_v2_evidence_identity_is_byte_identical(
    tmp_path: Path,
) -> None:
    """One sealed cache's v2 margin evidence is the pre-merge form exactly.

    The reference body below is the pre-change ``cache_authentication``
    construction verbatim.  Merged support may add one field to a merged
    record and nothing at all to a single-cache one, because every margin
    receipt a single sealed cache has already produced stays comparable.
    """

    case = _case(tmp_path, vocabulary_margin_observability=True)
    parent, graph, cache, session = case[:4]
    try:
        result = balanced_basis.build_lfm2_circuit_balanced_basis(
            parent,
            cache,
            4,
            circuit_graph=graph,
            selective_session=session,
            validation_fraction=0.25,
            split_seed="balanced-basis-phase-test-v1",
            max_rows_per_block=8,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
            streaming_vocabulary_margin_v2=True,
            streaming_vocabulary_margin_pair_chunk_size=8,
        )
        adapter = result.as_dict()["vocabulary_margin_observability"]["adapter"]
        reference = {
            "cache_key": cache.cache_key,
            "manifest_sha256": cache.manifest["manifest_sha256"],
            "completion_sha256": session.as_dict()["completion_sha256"],
            "selective_session_sha256": session.evidence_sha256,
            "full_sealed_cache_file_and_inventory_integrity": True,
        }
        assert canonical_json_bytes(
            adapter["cache_authentication"]
        ) == canonical_json_bytes(reference)
        for record in (
            adapter["cache_authentication"],
            adapter["inventory"],
            adapter["fit"]["precomputed_observability"]["provenance"],
            adapter["heldout"]["gate"],
        ):
            assert margin_adapter._MERGED_EVIDENCE_FIELD not in record
        assert result.as_dict()["model_forward_calls"] == 0
    finally:
        session.close()


def test_merged_evidence_streaming_v2_publishes_sealed_margin_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The arm the r13 refold needs: merged evidence, streaming v2, sealed."""

    case, evidence = _evidence_case(tmp_path)
    parent, graph, merged_cache, merged_session = case[:4]
    session_type = type(merged_session)
    original_load = session_type.load_selected_tensor_rows
    original_release = session_type.release_selected_tensor_rows
    row_loads: list[int] = []
    row_releases: list[int] = []

    def counted_load(self, index, row_indices):
        row_loads.append(int(index))
        return original_load(self, index, row_indices)

    def counted_release(self, payload):
        row_releases.append(int(payload.batch_index))
        return original_release(self, payload)

    monkeypatch.setattr(session_type, "load_selected_tensor_rows", counted_load)
    monkeypatch.setattr(
        session_type,
        "release_selected_tensor_rows",
        counted_release,
    )
    try:
        result = balanced_basis.build_lfm2_circuit_balanced_basis(
            parent,
            merged_cache,
            4,
            circuit_graph=graph,
            selective_session=merged_session,
            validation_fraction=0.25,
            split_seed="balanced-basis-phase-test-v1",
            max_rows_per_block=8,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
            streaming_vocabulary_margin_v2=True,
            streaming_vocabulary_margin_pair_chunk_size=8,
        )
        margin = result.as_dict()["vocabulary_margin_observability"]
        adapter = margin["adapter"]
        authentication = adapter["cache_authentication"]
        document = authentication[margin_adapter._MERGED_EVIDENCE_FIELD]

        # The identity is the evidence set's own, and no sealed manifest or
        # completion digest is invented for a corpus that carries neither.
        assert authentication["cache_key"] == evidence.cache_key
        assert authentication["manifest_sha256"] is None
        assert authentication["completion_sha256"] is None
        assert authentication["selective_session_sha256"] == (
            merged_session.evidence_sha256
        )
        assert document["schema"] == (
            margin_adapter.LFM2_STREAMING_VOCABULARY_MARGIN_MERGED_EVIDENCE_SCHEMA
        )
        assert document["sealed_merged_manifest_digest_exists"] is False
        assert document["every_member_sealed_digest_authenticated"] is True

        # Which members, in which order, at which offsets, produced it.
        members = document["ordered_members"]
        assert [value["role"] for value in members] == [
            member.role for member in evidence.members
        ]
        assert [value["batch_offset"] for value in members] == [
            member.batch_offset for member in evidence.members
        ]
        assert [value["trace_cache"]["cache_key"] for value in members] == [
            member.cache_key for member in evidence.members
        ]
        # The real sealed digests, the ones a member file actually carries.
        assert [
            value["trace_cache"]["manifest_sha256"] for value in members
        ] == [
            member.cache.manifest["manifest_sha256"]
            for member in evidence.members
        ]
        assert members[1]["round_manifest_sha256"] == (
            evidence.round_manifest_sha256[0]
        )
        assert members[1]["all_rows_are_fit_rows"] is True
        assert document["extension_batch_indices"] == list(
            evidence.extension_batch_indices
        )
        assert document["on_policy_evidence_identity_sha256"] == (
            evidence.evidence_identity_sha256
        )
        assert document["merged_corpus_sha256"] == evidence.corpus_fingerprint
        assert document["merged_session_sha256"] == (
            merged_session.evidence_sha256
        )
        assert canonical_sha256(
            {
                name: value
                for name, value in document.items()
                if name != f"{margin_adapter._MERGED_EVIDENCE_FIELD}_sha256"
            }
        ) == document[f"{margin_adapter._MERGED_EVIDENCE_FIELD}_sha256"]

        # Every record this adapter emits carries the same composite.
        for record in (
            adapter["inventory"],
            adapter["fit"]["precomputed_observability"]["provenance"],
            adapter["heldout"]["gate"],
            margin["transaction_gate"],
        ):
            assert record[margin_adapter._MERGED_EVIDENCE_FIELD][
                f"{margin_adapter._MERGED_EVIDENCE_FIELD}_sha256"
            ] == document[f"{margin_adapter._MERGED_EVIDENCE_FIELD}_sha256"]

        assert margin["heldout_projection_gate"]["gate_passed"] is True
        assert margin["heldout_projection_gate"]["parent_forward_calls"] == 0
        assert adapter["fit"]["all_declared_rows_consumed"] is True
        assert adapter["fit"]["selection_pass"][
            "row_cap_or_subsample_applied"
        ] is False
        assert result.as_dict()["model_forward_calls"] == 0

        # v2's whole point: bounded row reads, every one released, and the
        # extension member's own batches among them.
        assert len(row_loads) == len(row_releases) > 0
        assert set(row_loads) <= set(range(evidence.batch_count))
        assert set(row_loads) & set(evidence.extension_batch_indices)
    finally:
        evidence.close()


def test_merged_evidence_streaming_v2_composite_binds_every_member_identity(
    tmp_path: Path,
) -> None:
    """No margin evidence can be mistaken for another composition's."""

    baseline, evidence = _evidence_case(tmp_path / "baseline")
    _other_extension, other_evidence = _evidence_case(
        tmp_path / "other-extension",
        extension_indices=(12, 13),
    )
    _other_manifest, manifest_evidence = _evidence_case(
        tmp_path / "other-manifest",
        round_manifest=_digest("round-manifest-1"),
    )
    _other_split, split_evidence = _evidence_case(
        tmp_path / "other-split",
        base_splits=("fit", "selection", "fit", "fit"),
    )

    def composite(handle) -> str:
        receipt = margin_adapter._authenticate_merged_evidence_session(
            handle.merged_cache(),
            handle.merged_session(),
        )
        return receipt[margin_adapter._MERGED_EVIDENCE_FIELD][
            f"{margin_adapter._MERGED_EVIDENCE_FIELD}_sha256"
        ]

    try:
        baseline_composite = composite(evidence)
        # The evidence-set cache key covers member cache keys and the round
        # identity but not the split assignment, so this composition is the one
        # that proves the composite is strictly stronger than that key alone.
        assert (
            split_evidence.cache_key == evidence.cache_key
            and split_evidence.base.split_sha256 != evidence.base.split_sha256
        )
        others = {
            "other_extension_cache": composite(other_evidence),
            "other_round_manifest": composite(manifest_evidence),
            "other_base_split": composite(split_evidence),
        }
        assert baseline_composite not in others.values()
        assert len(set(others.values())) == len(others)
        # Re-authenticating the same evidence reproduces the same composite.
        assert composite(evidence) == baseline_composite
    finally:
        evidence.close()
        other_evidence.close()
        manifest_evidence.close()
        split_evidence.close()


def _role_blind_heldout_record_ids(
    records,
    *,
    cache_key: str,
    split_seed: str,
    validation_fraction: float = 0.25,
) -> set[str]:
    """The pre-fix ranking, verbatim: no member role, no extension awareness."""

    count = len(records)
    heldout_count = max(
        1,
        min(count - 1, math.ceil(count * validation_fraction)),
    )
    ranked = sorted(
        records,
        key=lambda value: (
            hashlib.sha256(
                (
                    split_seed
                    + "\0"
                    + cache_key
                    + "\0"
                    + value.record_id
                ).encode("utf-8")
            ).hexdigest(),
            value.index,
        ),
    )
    return {value.record_id for value in ranked[:heldout_count]}


def test_plain_cache_split_is_byte_identical_to_the_role_blind_form(
    tmp_path: Path,
) -> None:
    """One sealed cache splits exactly as it did before role awareness.

    The reference below is the pre-change ``_split_records`` body verbatim.  A
    single cache has no extension members, so the narrowed held-out pool is the
    whole record set and neither the ranking nor the report may move -- every
    basis report digest a single-cache solve has already sealed depends on it.
    """

    case = _case(tmp_path)
    cache, session = case[2], case[3]
    split_seed = "balanced-basis-phase-test-v1"
    try:
        records = tuple(
            value.summary
            for value in balanced_basis._selective_record_metadata(
                session,
                source_width=8,
                layer_count=2,
            )
        )
        count = len(records)
        heldout_count = max(1, min(count - 1, math.ceil(count * 0.25)))
        heldout_ids = _role_blind_heldout_record_ids(
            records,
            cache_key=cache.cache_key,
            split_seed=split_seed,
        )
        reference_fit = tuple(
            value for value in records if value.record_id not in heldout_ids
        )
        reference_heldout = tuple(
            value for value in records if value.record_id in heldout_ids
        )
        reference_report = {
            "method": "whole-record-seeded-sha256-ranking",
            "split_seed_sha256": hashlib.sha256(
                split_seed.encode("utf-8")
            ).hexdigest(),
            "requested_validation_fraction": 0.25,
            "record_count": count,
            "fit_record_count": len(reference_fit),
            "heldout_record_count": len(reference_heldout),
            "actual_heldout_fraction": len(reference_heldout) / count,
            "fit_record_ids_sha256": canonical_sha256(
                sorted(value.record_id for value in reference_fit)
            ),
            "heldout_record_ids_sha256": canonical_sha256(
                sorted(value.record_id for value in reference_heldout)
            ),
            "whole_records_only": True,
            "record_ids_disjoint": True,
            "heldout_excluded_from_selection": True,
        }
        fit, heldout, report = balanced_basis._split_records(
            records,
            cache_key=cache.cache_key,
            validation_fraction=0.25,
            split_seed=split_seed,
        )
        assert len(heldout) == heldout_count
        assert [value.record_id for value in fit] == [
            value.record_id for value in reference_fit
        ]
        assert [value.record_id for value in heldout] == [
            value.record_id for value in reference_heldout
        ]
        assert canonical_json_bytes(report) == canonical_json_bytes(
            reference_report
        )
        # A single cache exposes no extension indices at all, which is what
        # selects the unchanged branch.
        assert balanced_basis._extension_batch_indices(cache) == frozenset()
        assert balanced_basis._extension_batch_indices(session) == frozenset()
    finally:
        session.close()


def test_merged_evidence_split_never_holds_out_a_child_visited_batch(
    tmp_path: Path,
) -> None:
    """Every on-policy extension row is a fit row, in the basis split too.

    The sealed evidence set refuses any non-fit split label for an extension
    member because "validation would stop being held out"
    (``Lfm2SealedEvidenceRequest``).  The basis's own fit/held-out ranking used
    to ignore member roles entirely, so a child-generated batch could land in
    the held-out projection gate and the gate would measure the child partly on
    its own outputs.  The seed picked here is one the role-blind ranking
    provably leaks on.
    """

    case, evidence = _evidence_case(tmp_path)
    merged_cache, merged_session = case[2], case[3]
    try:
        records = tuple(
            value.summary
            for value in balanced_basis._selective_record_metadata(
                merged_session,
                source_width=8,
                layer_count=2,
            )
        )
        extension_indices = frozenset(evidence.extension_batch_indices)
        assert extension_indices == frozenset({4, 5})
        extension_ids = {
            value.record_id
            for value in records
            if value.index in extension_indices
        }
        assert len(extension_ids) == 2

        # Pick a seed the pre-fix ranking demonstrably leaks on, so this test
        # fails if the fix is reverted instead of passing by luck.
        leaking_seed = next(
            seed
            for seed in (f"role-blind-leak-{index}" for index in range(64))
            if _role_blind_heldout_record_ids(
                records,
                cache_key=merged_cache.cache_key,
                split_seed=seed,
            )
            & extension_ids
        )
        assert (
            _role_blind_heldout_record_ids(
                records,
                cache_key=merged_cache.cache_key,
                split_seed=leaking_seed,
            )
            & extension_ids
        )

        for split_seed in (leaking_seed, "balanced-basis-phase-test-v1"):
            fit, heldout, report = balanced_basis._split_records(
                records,
                cache_key=merged_cache.cache_key,
                validation_fraction=0.25,
                split_seed=split_seed,
                extension_batch_indices=extension_indices,
            )
            heldout_ids = {value.record_id for value in heldout}
            assert heldout_ids & extension_ids == set()
            assert extension_ids <= {value.record_id for value in fit}
            assert heldout_ids
            assert report["heldout_extension_record_count"] == 0
            assert report["extension_records_are_never_heldout"] is True
            assert report["heldout_pool"] == "base-members-only"
            assert report["on_policy_extension_record_count"] == 2
            assert report["heldout_eligible_record_count"] == 4
            # Held-out is still drawn by the same ranking, only from the base
            # members, so the requested fraction is honoured where it can be.
            assert report["heldout_record_count"] == math.ceil(
                len(records) * 0.25
            )
    finally:
        evidence.close()


def test_merged_evidence_streaming_v2_refuses_a_heldout_extension_batch(
    tmp_path: Path,
) -> None:
    """The adapter refuses the claim rather than sealing it.

    The split is what makes this impossible; this is the authenticator's own
    check that the split it was handed really did keep every child-visited
    batch out of the held-out set.
    """

    case, evidence = _evidence_case(tmp_path)
    merged_cache, merged_session = case[2], case[3]
    try:
        batches = tuple(evidence.iter_batches())
        extension_input = batches[
            evidence.extension_batch_indices[0]
        ].input_sha256
        base_input = batches[0].input_sha256
        with pytest.raises(
            margin_adapter.Lfm2StreamingVocabularyMarginAdapterError,
            match="every extension row is a fit row",
        ):
            margin_adapter._authenticate_merged_evidence_session(
                merged_cache,
                merged_session,
                heldout_record_ids=(base_input, extension_input),
            )
        # A base-only held-out set is exactly what the split now produces, and
        # it authenticates to the same composite an empty set does, because the
        # enforced count is zero either way.
        with_base = margin_adapter._authenticate_merged_evidence_session(
            merged_cache,
            merged_session,
            heldout_record_ids=(base_input,),
        )
        without = margin_adapter._authenticate_merged_evidence_session(
            merged_cache,
            merged_session,
        )
        field = margin_adapter._MERGED_EVIDENCE_FIELD
        assert with_base[field][f"{field}_sha256"] == (
            without[field][f"{field}_sha256"]
        )
        assert with_base[field]["heldout_extension_batch_count"] == 0
        assert with_base[field][
            "extension_batches_are_never_heldout_evidence"
        ] is True
    finally:
        evidence.close()


def test_merged_evidence_gate_refuses_a_changed_composition(
    tmp_path: Path,
) -> None:
    """The gate's composite re-comparison is the composite's enforcement point.

    A member's split assignment enters neither the merged session digest nor
    the evidence-set cache key, so moving it after the gate was built passes
    every other check the gate makes and is caught only by comparing the
    composite the gate sealed against the one its live evidence now produces.
    """

    case, evidence = _evidence_case(tmp_path)
    parent, graph, merged_cache, merged_session = case[:4]
    try:
        result = balanced_basis.build_lfm2_circuit_balanced_basis(
            parent,
            merged_cache,
            4,
            circuit_graph=graph,
            selective_session=merged_session,
            validation_fraction=0.25,
            split_seed="balanced-basis-phase-test-v1",
            max_rows_per_block=8,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
            streaming_vocabulary_margin_v2=True,
            streaming_vocabulary_margin_pair_chunk_size=8,
        )
        gate = result.vocabulary_margin_gate
        assert gate.as_dict()["cache_key"] == evidence.cache_key

        base_member = evidence.members[0]
        original_split = base_member.split_sha256
        object.__setattr__(base_member, "split_sha256", _digest("moved-split"))
        try:
            # Everything else still authenticates: the merged session digest and
            # the evidence-set cache key do not cover a member's split.
            assert merged_session.evidence_sha256 == (
                gate.expected_session_sha256
            )
            assert merged_cache.cache_key == gate.cache_key
            with pytest.raises(
                margin_adapter.Lfm2StreamingVocabularyMarginAdapterError,
                match="composition changed after gate construction",
            ):
                gate.as_dict()
            with pytest.raises(
                margin_adapter.Lfm2StreamingVocabularyMarginAdapterError,
                match="composition changed after gate construction",
            ):
                gate.evaluate_projection(
                    result.basis,
                    minimum_retained_energy_fraction=0.0,
                    minimum_sign_agreement_fraction=0.0,
                    minimum_top_competitor_nonnegative_fraction=0.0,
                    minimum_all_top_k_nonnegative_fraction=0.0,
                )
        finally:
            object.__setattr__(base_member, "split_sha256", original_split)
        # Restored, the same gate authenticates again.
        assert gate.as_dict()["cache_key"] == evidence.cache_key
    finally:
        evidence.close()


def test_merged_evidence_streaming_v2_fails_closed_on_a_foreign_handle_pair(
    tmp_path: Path,
) -> None:
    case, evidence = _evidence_case(tmp_path)
    merged_cache, merged_session = case[2], case[3]
    try:
        with pytest.raises(
            margin_adapter.Lfm2StreamingVocabularyMarginAdapterError,
            match="requires its merged authenticated session",
        ):
            margin_adapter._authenticate_evidence_session(
                merged_cache,
                evidence.base.session,
            )
        with pytest.raises(
            margin_adapter.Lfm2StreamingVocabularyMarginAdapterError,
            match="requires its merged cache view",
        ):
            margin_adapter._authenticate_evidence_session(
                evidence.base_cache,
                merged_session,
            )
        with pytest.raises(
            margin_adapter.Lfm2StreamingVocabularyMarginAdapterError,
            match="requires its merged cache view",
        ):
            margin_adapter._authenticate_evidence_session(object(), merged_session)
        with pytest.raises(
            margin_adapter.Lfm2StreamingVocabularyMarginAdapterError,
            match="must expose verify\\(\\) and as_dict\\(\\)",
        ):
            margin_adapter._authenticate_evidence_session(object(), object())
        # The single-cache pair still routes to the single-cache authenticator
        # and still binds one real sealed manifest digest.
        receipt = margin_adapter._authenticate_evidence_session(
            evidence.base_cache,
            evidence.base.session,
        )
        assert receipt["manifest_sha256"] == (
            evidence.base_cache.manifest["manifest_sha256"]
        )
        assert margin_adapter._MERGED_EVIDENCE_FIELD not in receipt
    finally:
        evidence.close()
