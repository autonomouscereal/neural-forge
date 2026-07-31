from __future__ import annotations

import inspect
import json
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
import neural_forge.hf_lfm2_dense_budget_model_fold as dense_fold  # noqa: E402
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
from neural_forge.hf_lfm2_dense_budget_model_fold import (  # noqa: E402
    Lfm2DenseBudgetFoldConfig,
    Lfm2DenseBudgetModelOperator,
    _build_residual_basis,
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
        reuse_context = Lfm2BalancedBasisPhaseCacheContext(
            cache=context.cache,
            anchor_authority=context.anchor_authority,
            progress=context.progress,
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
