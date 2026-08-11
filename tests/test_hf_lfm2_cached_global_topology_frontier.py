from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (  # noqa: E402
    _parent,
)
from test_hf_lfm2_dynamic_closed_loop_adapter import _batch  # noqa: E402

from neural_forge.dynamic_behavior_gate import observe_model_execution  # noqa: E402
from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.dynamic_fold_phase_cache import (  # noqa: E402
    DynamicFoldPhaseCache,
    FoldPhaseCacheIntegrityError,
)
import neural_forge.dynamic_fold_phase_cache as phase_cache_module  # noqa: E402
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
import neural_forge.hf_lfm2_cached_global_topology_frontier as global_frontier  # noqa: E402
import neural_forge.hf_lfm2_global_topology_phase_compatibility as topology_compatibility  # noqa: E402
from neural_forge.hf_lfm2_cached_global_topology_frontier import (  # noqa: E402
    GLOBAL_TOPOLOGY_CAPACITY_PHASE,
    GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
    GLOBAL_TOPOLOGY_SELECTION_PHASE,
    Lfm2CachedGlobalTopologyFrontierError,
    Lfm2ExistingGlobalTopologyPhaseReuseRequest,
    Lfm2GlobalTargetRuntimePins,
    Lfm2GlobalTopologyCandidateReport,
    Lfm2GlobalTopologyPhaseCacheContext,
    build_lfm2_cached_global_topology_frontier,
    select_lfm2_global_topology_report,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (  # noqa: E402
    ATTENTION_GEOMETRY_ENABLED_POLICY,
    AUTOMATIC_MLP_OPTION_POLICY,
    CAPACITY_OPTION_BANK_SCHEMA,
    EXPLICIT_MLP_OPTION_POLICY,
    MLP_CAPACITY_RISK_METHOD,
    SHORT_CONV_WRITER_ACTION_RISK_METHOD,
    Lfm2CachedHeterogeneousCapacityOptionBank,
    Lfm2CachedLayerCapacityEvidence,
    Lfm2CachedMlpCapacityRisk,
    Lfm2CachedShortConvWriterRisk,
    Lfm2MlpWidthOptionPolicy,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CACHED_VERIFIED_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCache,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (  # noqa: E402
    build_lfm2_short_conv_layer_option,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


_PUBLISHED_GLOBAL_ALGORITHM_SHA256 = (
    "3e4531740792b3abe7ed54c02a49ca99caabc3ae94dd181c167913cbb5dbcab1"
)
_CURRENT_GLOBAL_FRONTIER_SOURCE_SHA256 = (
    "338d5b800bfac45dc8cee0017a98a850392610b34322db4cd55a0d8a0de73b12"
)
_CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256 = (
    "3ca319c165d81c7fffb2f57a55edf3fcab49c647dd73cefb5a9b8d955a4741a7"
)
_CURRENT_GLOBAL_ALGORITHM_SHA256 = (
    "d2ed9e076927dea1bfa44e47f3a2fff19b9617f486ca079e27948d212b6212dd"
)


def _compatibility_receipt_sha256() -> str:
    receipt = (
        topology_compatibility.explicit_global_topology_phase_compatibility_receipt(
            historical_algorithm_sha256=_PUBLISHED_GLOBAL_ALGORITHM_SHA256,
            current_global_frontier_source_sha256=(
                _CURRENT_GLOBAL_FRONTIER_SOURCE_SHA256
            ),
            current_functional_width_source_sha256=(
                _CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256
            ),
            current_algorithm_sha256=_CURRENT_GLOBAL_ALGORITHM_SHA256,
        )
    )
    assert receipt is not None
    return str(receipt["compatibility_receipt_sha256"])


def _runtime() -> Lfm2GlobalTargetRuntimePins:
    return Lfm2GlobalTargetRuntimePins(
        vocab_size=64,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        target_vocab_evidence_sha256=_digest("target-vocabulary"),
        token_bridge_evidence_sha256=_digest("identity-token-bridge"),
    )


def _phase_cache_case(tmp_path):
    parent = _parent()
    parent.config._attn_implementation = "eager"
    graph = inspect_model_circuits(
        parent,
        model_name="global-frontier-phase-cache-parent",
        tokenizer_fingerprint=_digest("global-frontier-phase-cache-tokenizer"),
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "parent-cache",
        parent,
        [_batch((1, 2, 3, 4), (5,))],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    kwargs = {
        "runtime": _runtime(),
        "exact_target_unique_parameters": 8_720,
        "minimum_hidden_size": 8,
        "hidden_size_multiple": 8,
        "short_conv_internal_widths": (4, 8),
        "attention_head_options": ((2, 1, 8), (4, 2, 8)),
        "maximum_functional_rows_per_role": 16,
    }
    return parent, graph, cache, kwargs


def _forge_resealed_phase_evidence(
    entry,
    evidence_field: str,
    mutate,
) -> tuple[bytes, bytes]:
    payload_path = entry / "payload" / "evidence.json"
    completion_path = entry / "complete.json"
    original_payload = payload_path.read_bytes()
    original_completion = completion_path.read_bytes()
    payload = json.loads(original_payload.decode("ascii"))
    evidence = payload[evidence_field]
    mutate(evidence)
    evidence_body = dict(evidence)
    evidence_body.pop("evidence_sha256")
    evidence["evidence_sha256"] = canonical_sha256(evidence_body)
    payload_body = dict(payload)
    payload_body.pop("payload_sha256")
    payload["payload_sha256"] = canonical_sha256(payload_body)
    payload_path.write_bytes(canonical_json_bytes(payload) + b"\n")

    completion = json.loads(original_completion.decode("ascii"))
    completion["payload_inventory"] = phase_cache_module._payload_inventory(
        entry / "payload"
    )
    completion["result"]["payload_sha256"] = payload["payload_sha256"]
    completion["result"]["evidence_sha256"] = evidence["evidence_sha256"]
    if evidence_field == "capacity_option_bank":
        completion["result"]["layer_count"] = len(evidence["layer_evidence"])
        completion["result"]["layer_option_count"] = len(evidence["layer_options"])
    else:
        completion["result"]["candidate_score_count"] = len(
            evidence["candidate_scores"]
        )
    completion_body = dict(completion)
    completion_body.pop("completion_sha256")
    completion["completion_sha256"] = canonical_sha256(completion_body)
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")
    return original_payload, original_completion


def test_global_frontier_scans_nonlinear_cache_once_and_selects_complete_exact_child(
    tmp_path,
    monkeypatch,
) -> None:
    parent = _parent()
    parent.config._attn_implementation = "eager"
    graph = inspect_model_circuits(
        parent,
        model_name="global-frontier-parent",
        tokenizer_fingerprint=_digest("global-frontier-tokenizer"),
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "parent-cache",
        parent,
        [_batch((1, 2, 3, 4), (5,))],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    nonlinear_builds = []
    nonlinear_banks = []
    original_builder = (
        global_frontier.build_lfm2_cached_heterogeneous_capacity_option_bank
    )

    def tracked_builder(*args, **kwargs):
        nonlinear_builds.append((args, kwargs))
        result = original_builder(*args, **kwargs)
        nonlinear_banks.append(result)
        return result

    monkeypatch.setattr(
        global_frontier,
        "build_lfm2_cached_heterogeneous_capacity_option_bank",
        tracked_builder,
    )
    with observe_model_execution(parent) as execution:
        frontier = build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            runtime=_runtime(),
            exact_target_unique_parameters=8_720,
            minimum_hidden_size=8,
            hidden_size_multiple=8,
            short_conv_internal_widths=(4, 8),
            attention_head_options=((2, 1, 8), (4, 2, 8)),
            maximum_functional_rows_per_role=16,
        )

    assert execution["root_forward_calls"] == 0
    assert execution["module_forward_calls"] == 0
    assert len(nonlinear_builds) == 1
    assert nonlinear_builds[0][1]["mlp_intermediate_sizes"] is not None
    assert nonlinear_builds[0][1]["attention_geometry_risk_enabled"] is True
    assert frontier.candidate_hidden_sizes == (8, 16, 24, 32)
    assert [value.hidden_size for value in frontier.candidates] == [8, 16, 24]
    assert [
        (value.hidden_size, value.reason_class) for value in frontier.rejected_widths
    ] == [(32, "no_exact_integer_lattice")]
    assert frontier.selected_input.target_pins.hidden_size == 8
    assert [
        value.role_balanced_functional_retention
        for value in frontier.functional_width_evidence.candidate_scores
    ] == pytest.approx(
        (0.8135463040140856, 0.9693938753607511, 0.9990537692196746, 1.0),
        rel=0.0,
        abs=5e-13,
    )
    assert all(value.exact_unique_parameters == 8_720 for value in frontier.candidates)
    assert all(value.target_layer_count == 4 for value in frontier.candidates)
    assert frontier.selected_capacity_option_bank.mlp_option_policy == (
        AUTOMATIC_MLP_OPTION_POLICY
    )
    assert frontier.selected_capacity_option_bank.attention_geometry_policy == (
        ATTENTION_GEOMETRY_ENABLED_POLICY
    )
    assert all(
        bool(evidence.attention_geometry_risks)
        == (evidence.layer_type == "full_attention")
        for evidence in frontier.selected_capacity_option_bank.layer_evidence
    )
    assert [
        evidence.attention_geometry_risks
        for evidence in frontier.selected_capacity_option_bank.layer_evidence
    ] == [
        evidence.attention_geometry_risks
        for evidence in nonlinear_banks[0].layer_evidence
    ]
    assert frontier.selected_layout.descriptor.accounting.unique_parameter_elements == (
        8_720
    )
    assert len(frontier.selected_layout.descriptor.layers) == 4
    assert (
        frontier.selected_input.target_pins.hidden_size
        == min(
            frontier.candidates,
            key=lambda value: (
                value.combined_predicted_risk,
                value.role_balanced_functional_loss,
                value.mean_selected_component_risk,
                value.candidate_sha256,
            ),
        ).hidden_size
    )
    assert [
        value.hidden_size
        for value in frontier.functional_width_evidence.candidate_scores
    ] == [8, 16, 24, 32]
    report = frontier.as_dict()
    assert report["selection_contract"] == {
        "complete_full_depth_child_per_feasible_width": True,
        "same_exact_parameter_count_per_feasible_width": True,
        "parameter_count_used_as_quality_score": False,
        "cached_parent_function_only": True,
        "cached_attention_geometry_allocation": True,
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "sparse_parameters": 0,
        "shared_parameters": 0,
    }


def test_required_hidden_filters_b1_b2_b3_and_authenticates_distinct_cache(
    tmp_path,
    monkeypatch,
) -> None:
    parent, graph, cache, kwargs = _phase_cache_case(tmp_path)
    phase_root = tmp_path / "derived-phases"
    capacity_hidden_sizes = []
    original_builder = (
        global_frontier.build_lfm2_cached_heterogeneous_capacity_option_bank
    )

    def tracked_builder(*args, **builder_kwargs):
        capacity_hidden_sizes.append(builder_kwargs["target_hidden_size"])
        return original_builder(*args, **builder_kwargs)

    monkeypatch.setattr(
        global_frontier,
        "build_lfm2_cached_heterogeneous_capacity_option_bank",
        tracked_builder,
    )
    context = Lfm2GlobalTopologyPhaseCacheContext(
        cache=DynamicFoldPhaseCache(phase_root),
        evidence_split_sha256=_digest("required-hidden-splits"),
    )
    with observe_model_execution(parent) as execution:
        dynamic = build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=context,
            **kwargs,
        )
        constrained = build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=context,
            required_target_hidden_size=16,
            **kwargs,
        )

    assert execution["root_forward_calls"] == 0
    assert execution["module_forward_calls"] == 0
    assert dynamic.required_target_hidden_size is None
    assert len(dynamic.candidate_hidden_sizes) > 1
    assert constrained.required_target_hidden_size == 16
    assert constrained.candidate_hidden_sizes == (16,)
    assert [value.hidden_size for value in constrained.candidates] == [16]
    assert constrained.rejected_widths == ()
    assert constrained.selected_input.target_pins.hidden_size == 16
    assert [
        value.hidden_size
        for value in constrained.functional_width_evidence.candidate_scores
    ] == [16]
    assert capacity_hidden_sizes == [32, 16]
    assert constrained.as_dict()["required_target_hidden_size"] == 16
    for phase_name in (
        GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        GLOBAL_TOPOLOGY_SELECTION_PHASE,
    ):
        assert len(list((phase_root / phase_name).iterdir())) == 2


def test_illegal_required_hidden_fails_before_cached_evidence(tmp_path) -> None:
    parent, graph, cache, kwargs = _phase_cache_case(tmp_path)

    with (
        observe_model_execution(parent) as execution,
        pytest.raises(
            Lfm2CachedGlobalTopologyFrontierError,
            match="required target hidden size is not legal.*H12",
        ),
    ):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            required_target_hidden_size=12,
            **kwargs,
        )

    assert execution["root_forward_calls"] == 0
    assert execution["module_forward_calls"] == 0


def test_historical_phase_compatibility_is_exact_directional_and_source_bound() -> None:
    active_global_source_sha256 = hashlib.sha256(
        Path(global_frontier.__file__).read_bytes()
    ).hexdigest()
    active_algorithm_sha256 = global_frontier._global_algorithm_fingerprint()
    # The v5 attention-risk schema intentionally leaves the last explicitly
    # approved v4 compatibility tuple behind.
    assert active_global_source_sha256 != _CURRENT_GLOBAL_FRONTIER_SOURCE_SHA256
    assert active_algorithm_sha256 != _CURRENT_GLOBAL_ALGORITHM_SHA256
    assert (
        hashlib.sha256(
            Path(global_frontier._cached_functional_width_module.__file__).read_bytes()
        ).hexdigest()
        == _CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256
    )
    exact = topology_compatibility.explicit_global_topology_phase_compatibility_receipt(
        historical_algorithm_sha256=_PUBLISHED_GLOBAL_ALGORITHM_SHA256,
        current_global_frontier_source_sha256=(_CURRENT_GLOBAL_FRONTIER_SOURCE_SHA256),
        current_functional_width_source_sha256=(
            _CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256
        ),
        current_algorithm_sha256=_CURRENT_GLOBAL_ALGORITHM_SHA256,
    )
    assert exact is not None
    receipt = dict(exact)
    body = dict(receipt)
    seal = body.pop("compatibility_receipt_sha256")
    assert seal == canonical_sha256(body)
    for field in (
        "historical_algorithm_sha256",
        "current_global_frontier_source_sha256",
        "current_functional_width_source_sha256",
        "current_algorithm_sha256",
    ):
        values = {
            "historical_algorithm_sha256": _PUBLISHED_GLOBAL_ALGORITHM_SHA256,
            "current_global_frontier_source_sha256": (
                _CURRENT_GLOBAL_FRONTIER_SOURCE_SHA256
            ),
            "current_functional_width_source_sha256": (
                _CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256
            ),
            "current_algorithm_sha256": _CURRENT_GLOBAL_ALGORITHM_SHA256,
        }
        values[field] = "f" * 64
        assert (
            topology_compatibility.explicit_global_topology_phase_compatibility_receipt(
                **values
            )
            is None
        )
    assert (
        topology_compatibility.explicit_global_topology_phase_compatibility_receipt(
            historical_algorithm_sha256=_CURRENT_GLOBAL_ALGORITHM_SHA256,
            current_global_frontier_source_sha256=(
                _CURRENT_GLOBAL_FRONTIER_SOURCE_SHA256
            ),
            current_functional_width_source_sha256=(
                _CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256
            ),
            current_algorithm_sha256=_PUBLISHED_GLOBAL_ALGORITHM_SHA256,
        )
        is None
    )
    assert (
        topology_compatibility.explicit_global_topology_phase_compatibility_receipt(
            historical_algorithm_sha256=_PUBLISHED_GLOBAL_ALGORITHM_SHA256,
            current_global_frontier_source_sha256=active_global_source_sha256,
            current_functional_width_source_sha256=(
                _CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256
            ),
            current_algorithm_sha256=active_algorithm_sha256,
        )
        is None
    )


def test_compatibility_receipt_reader_rejects_foreign_fields_and_reseals() -> None:
    exact = topology_compatibility.explicit_global_topology_phase_compatibility_receipt(
        historical_algorithm_sha256=_PUBLISHED_GLOBAL_ALGORITHM_SHA256,
        current_global_frontier_source_sha256=(_CURRENT_GLOBAL_FRONTIER_SOURCE_SHA256),
        current_functional_width_source_sha256=(
            _CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256
        ),
        current_algorithm_sha256=_CURRENT_GLOBAL_ALGORITHM_SHA256,
    )
    assert exact is not None
    validation = {
        "historical_algorithm_sha256": _PUBLISHED_GLOBAL_ALGORITHM_SHA256,
        "current_global_frontier_source_sha256": (
            _CURRENT_GLOBAL_FRONTIER_SOURCE_SHA256
        ),
        "current_functional_width_source_sha256": (
            _CURRENT_FUNCTIONAL_WIDTH_SOURCE_SHA256
        ),
        "current_algorithm_sha256": _CURRENT_GLOBAL_ALGORITHM_SHA256,
    }
    assert dict(
        global_frontier._validate_global_topology_phase_compatibility_receipt(
            exact,
            **validation,
        )
    ) == dict(exact)
    for field_name, foreign_value in (
        ("schema", "foreign-schema"),
        ("direction", "current-to-historical"),
        ("historical_artifact_rewritten", True),
        ("historical_algorithm_sha256", "f" * 64),
        ("current_global_frontier_source_sha256", "f" * 64),
        ("current_functional_width_source_sha256", "f" * 64),
        ("current_algorithm_sha256", "f" * 64),
    ):
        forged = dict(exact)
        forged[field_name] = foreign_value
        forged_body = dict(forged)
        forged_body.pop("compatibility_receipt_sha256")
        forged["compatibility_receipt_sha256"] = canonical_sha256(forged_body)
        with pytest.raises(FoldPhaseCacheIntegrityError):
            global_frontier._validate_global_topology_phase_compatibility_receipt(
                forged,
                **validation,
            )
    extra = {**dict(exact), "foreign": False}
    with pytest.raises(FoldPhaseCacheIntegrityError, match="fields"):
        global_frontier._validate_global_topology_phase_compatibility_receipt(
            extra,
            **validation,
        )
    bad_self_hash = {**dict(exact), "compatibility_receipt_sha256": "f" * 64}
    with pytest.raises(FoldPhaseCacheIntegrityError, match="self-hash"):
        global_frontier._validate_global_topology_phase_compatibility_receipt(
            bad_self_hash,
            **validation,
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    (
        (b'{"phase_key":{},"phase_key":{}}\n', "duplicate key"),
        (b'{"phase_key":NaN}\n', "non-finite value"),
    ),
)
def test_historical_completion_discovery_rejects_noncanonical_json(
    tmp_path,
    raw,
    message,
) -> None:
    phase_key_sha256 = "a" * 64
    cache = DynamicFoldPhaseCache(tmp_path / "phases")
    entry = cache.root / GLOBAL_TOPOLOGY_CAPACITY_PHASE / phase_key_sha256
    entry.mkdir(parents=True)
    (entry / phase_cache_module.COMPLETE_FILENAME).write_bytes(raw)
    with pytest.raises(FoldPhaseCacheIntegrityError, match=message):
        global_frontier._existing_global_topology_phase_key_from_digest(
            cache,
            phase_name=GLOBAL_TOPOLOGY_CAPACITY_PHASE,
            phase_key_sha256=phase_key_sha256,
        )


def test_historical_completion_discovery_is_bounded_and_replacement_stable(
    tmp_path,
    monkeypatch,
) -> None:
    phase_key_sha256 = "a" * 64
    cache = DynamicFoldPhaseCache(tmp_path / "phases")
    entry = cache.root / GLOBAL_TOPOLOGY_CAPACITY_PHASE / phase_key_sha256
    entry.mkdir(parents=True)
    completion = entry / phase_cache_module.COMPLETE_FILENAME
    completion.write_bytes(b" " * (phase_cache_module._MAX_JSON_BYTES + 1))
    with pytest.raises(FoldPhaseCacheIntegrityError, match="size limit"):
        global_frontier._existing_global_topology_phase_key_from_digest(
            cache,
            phase_name=GLOBAL_TOPOLOGY_CAPACITY_PHASE,
            phase_key_sha256=phase_key_sha256,
        )

    completion.write_bytes(b"{}\n")
    replacement = entry / "replacement.json"
    replacement.write_bytes(b"{}\n")
    real_open = os.open
    swapped = False

    def replacing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(os.fspath(path)) == completion:
            swapped = True
            os.replace(replacement, completion)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(phase_cache_module.os, "open", replacing_open)
    with pytest.raises(FoldPhaseCacheIntegrityError, match="changed while opening"):
        global_frontier._existing_global_topology_phase_key_from_digest(
            cache,
            phase_name=GLOBAL_TOPOLOGY_CAPACITY_PHASE,
            phase_key_sha256=phase_key_sha256,
        )


def test_requested_historical_phase_miss_is_fatal_and_never_builds(
    tmp_path,
    monkeypatch,
) -> None:
    _parent_model, graph, cache, kwargs = _phase_cache_case(tmp_path / "source")
    phase_root = tmp_path / "missing-derived-phases"
    current_algorithm_sha256 = global_frontier._global_algorithm_fingerprint()
    monkeypatch.setattr(
        global_frontier,
        "_global_algorithm_fingerprint",
        lambda: current_algorithm_sha256,
    )
    monkeypatch.setattr(
        global_frontier,
        "build_lfm2_cached_heterogeneous_capacity_option_bank",
        lambda *_args, **_kwargs: pytest.fail(
            "missing historical capacity phase attempted a cold build"
        ),
    )
    request = Lfm2ExistingGlobalTopologyPhaseReuseRequest(
        phase_name=GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        phase_key_sha256="a" * 64,
        expected_completion_sha256="b" * 64,
        expected_evidence_sha256="c" * 64,
        expected_compatibility_receipt_sha256="d" * 64,
    )
    with pytest.raises(
        Lfm2CachedGlobalTopologyFrontierError,
        match="absent.*hit-only loading forbids building",
    ):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest("missing-historical-splits"),
                selective_session_factory=lambda: pytest.fail(
                    "missing historical phase opened a selective session"
                ),
                existing_capacity_phase_reuse=request,
            ),
            **kwargs,
        )
    assert not phase_root.exists()


def test_callable_source_fingerprint_is_portable_across_bytecode_compilers(
    monkeypatch,
) -> None:
    def first(value):
        return value + 1

    def second(value):
        return value * 2

    for function in (first, second):
        function.__module__ = "portable_fixture"
        function.__qualname__ = "portable_fixture"
    monkeypatch.setattr(
        global_frontier.inspect,
        "getsource",
        lambda _value: "def portable_fixture(value):\n    return value\n",
    )

    assert first.__code__.co_code != second.__code__.co_code
    assert global_frontier._callable_source_fingerprint(
        first,
        role="portable",
    ) == global_frontier._callable_source_fingerprint(
        second,
        role="portable",
    )


def test_topology_trace_cache_fingerprint_ignores_child_replay_surface(
    tmp_path,
    monkeypatch,
) -> None:
    trace_cache_module = global_frontier._circuit_trace_cache_module
    interface_before = global_frontier._trace_cache_topology_interface_fingerprint()
    global_before = global_frontier._global_algorithm_fingerprint()

    original_source = Path(trace_cache_module.__file__).read_text(encoding="utf-8")
    changed_source = original_source.replace(
        'CHILD_REPLAY_BATCH_SCHEMA = "neural-forge-lfm2-child-circuit-replay-batch-v2"',
        'CHILD_REPLAY_BATCH_SCHEMA = "irrelevant-child-replay-batch-schema-v3"',
        1,
    )
    assert changed_source != original_source
    changed_module_path = tmp_path / "hf_lfm2_circuit_trace_cache.py"
    changed_module_path.write_text(changed_source, encoding="utf-8")

    def changed_child_replay_capture(*_args, **_kwargs):
        raise AssertionError("irrelevant child replay implementation")

    monkeypatch.setattr(
        trace_cache_module,
        "capture_lfm2_child_replay",
        changed_child_replay_capture,
    )
    monkeypatch.setattr(
        trace_cache_module,
        "CHILD_REPLAY_BATCH_SCHEMA",
        "irrelevant-child-replay-batch-schema-v3",
    )
    monkeypatch.setattr(trace_cache_module, "__file__", str(changed_module_path))

    assert (
        global_frontier._trace_cache_topology_interface_fingerprint()
        == interface_before
    )
    assert global_frontier._global_algorithm_fingerprint() == global_before


def test_topology_trace_cache_fingerprint_changes_with_selected_row_reader(
    monkeypatch,
) -> None:
    trace_cache_module = global_frontier._circuit_trace_cache_module
    session_type = trace_cache_module.Lfm2CircuitTraceSelectiveAuthenticatedSession
    original = session_type.load_selected_tensors
    interface_before = global_frontier._trace_cache_topology_interface_fingerprint()
    global_before = global_frontier._global_algorithm_fingerprint()

    def changed_selected_row_reader(self, index, tensor_names):
        return original(self, index, tensor_names)

    monkeypatch.setattr(
        session_type,
        "load_selected_tensors",
        changed_selected_row_reader,
    )

    assert (
        global_frontier._trace_cache_topology_interface_fingerprint()
        != interface_before
    )
    assert global_frontier._global_algorithm_fingerprint() != global_before


def test_topology_trace_cache_fingerprint_changes_with_reader_abi(
    monkeypatch,
) -> None:
    trace_cache_module = global_frontier._circuit_trace_cache_module
    interface_before = global_frontier._trace_cache_topology_interface_fingerprint()

    monkeypatch.setattr(
        trace_cache_module,
        "SELECTIVE_BATCH_PAYLOAD_SCHEMA",
        "topology-selected-batch-payload-v2",
    )

    assert (
        global_frontier._trace_cache_topology_interface_fingerprint()
        != interface_before
    )


def test_global_frontier_phase_cache_resumes_without_evidence_rescan_and_fails_closed(
    tmp_path,
    monkeypatch,
) -> None:
    parent, graph, cache, kwargs = _phase_cache_case(tmp_path)
    phase_root = tmp_path / "derived-phases"
    progress = []
    context = Lfm2GlobalTopologyPhaseCacheContext(
        cache=DynamicFoldPhaseCache(phase_root),
        evidence_split_sha256=_digest("phase-cache-splits"),
        progress=lambda stage, details: progress.append((stage, dict(details))),
    )
    with observe_model_execution(parent) as execution:
        first = build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=context,
            **kwargs,
        )
    assert execution["root_forward_calls"] == 0
    assert execution["module_forward_calls"] == 0
    assert {child.name for child in phase_root.iterdir() if child.is_dir()} == {
        GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        GLOBAL_TOPOLOGY_SELECTION_PHASE,
    }
    assert (
        sum(
            stage == "global_topology_phase_cache_published"
            for stage, _details in progress
        )
        == 3
    )
    assert {
        "functional_width_selection_started",
        "functional_width_selection_completed",
        "functional_width_gram_accumulation_started",
        "functional_width_gram_batch_progress",
        "functional_width_gram_accumulation_completed",
        "functional_width_eigensolve_started",
        "functional_width_eigensolve_completed",
    }.issubset({stage for stage, _details in progress})
    assert all(
        details["parent_model_calls"] == 0
        and details["training_steps"] == 0
        and details["optimizer_steps"] == 0
        for stage, details in progress
        if stage.startswith("functional_width_")
    )
    for phase_name in (
        GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        GLOBAL_TOPOLOGY_SELECTION_PHASE,
    ):
        entry = next((phase_root / phase_name).iterdir())
        assert {
            path.relative_to(entry).as_posix()
            for path in entry.rglob("*")
            if path.is_file()
        } == {"complete.json", "payload/evidence.json"}

    trace_cache_interface_sha256 = (
        global_frontier._trace_cache_topology_interface_fingerprint()
    )

    def forbidden_session(_self):
        raise AssertionError("selective session must not open on a complete hit")

    monkeypatch.setattr(
        global_frontier,
        "_trace_cache_topology_interface_fingerprint",
        lambda: trace_cache_interface_sha256,
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceCache,
        "selective_authenticated_session",
        forbidden_session,
    )
    resumed_progress = []
    second = build_lfm2_cached_global_topology_frontier(
        graph,
        cache,
        phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
            cache=DynamicFoldPhaseCache(phase_root),
            evidence_split_sha256=_digest("phase-cache-splits"),
            progress=lambda stage, details: resumed_progress.append(
                (stage, dict(details))
            ),
        ),
        **kwargs,
    )
    assert dict(second.as_dict()) == dict(first.as_dict())
    assert (
        sum(
            stage == "global_topology_phase_cache_hit"
            for stage, _details in resumed_progress
        )
        == 3
    )
    reconstructed = next(
        details
        for stage, details in resumed_progress
        if stage == "global_topology_selected_width_reconstructed"
    )
    assert reconstructed["selector_width_count"] == 1
    assert reconstructed["selected_hidden_size"] == (
        first.selected_input.target_pins.hidden_size
    )
    assert not any(
        stage == "global_topology_evidence_compute_started"
        for stage, _details in resumed_progress
    )

    changed_progress = []
    with pytest.raises(
        AssertionError,
        match="selective session must not open",
    ):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest("different-phase-cache-splits"),
                progress=lambda stage, details: changed_progress.append(
                    (stage, dict(details))
                ),
            ),
            **kwargs,
        )
    assert any(
        stage == "global_topology_phase_cache_miss"
        and details["phase_name"] == GLOBAL_TOPOLOGY_CAPACITY_PHASE
        for stage, details in changed_progress
    )

    capacity_entry = next((phase_root / GLOBAL_TOPOLOGY_CAPACITY_PHASE).iterdir())
    completion_path = capacity_entry / "complete.json"
    original_completion = completion_path.read_bytes()
    completion = json.loads(original_completion.decode("ascii"))
    completion_body = dict(completion)
    completion_body.pop("completion_sha256")
    completion_body["result"]["layer_count"] = True
    forged_completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    completion_path.write_bytes(canonical_json_bytes(forged_completion) + b"\n")
    with pytest.raises(FoldPhaseCacheIntegrityError):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest("phase-cache-splits"),
            ),
            **kwargs,
        )
    completion_path.write_bytes(original_completion)

    payload_path = capacity_entry / "payload" / "evidence.json"
    payload = bytearray(payload_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    payload_path.write_bytes(bytes(payload))
    with pytest.raises(FoldPhaseCacheIntegrityError):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest("phase-cache-splits"),
            ),
            **kwargs,
        )


def test_capacity_restore_requires_v5_rope_pair_risks_and_refuses_tamper(
    tmp_path,
) -> None:
    _parent_model, graph, cache, kwargs = _phase_cache_case(tmp_path)
    frontier = build_lfm2_cached_global_topology_frontier(graph, cache, **kwargs)
    raw = frontier.selected_capacity_option_bank.as_dict()

    restored = global_frontier._restore_capacity_option_bank(raw)

    assert restored.as_dict() == raw
    assert raw["schema"] == CAPACITY_OPTION_BANK_SCHEMA
    attention_row = next(
        row
        for layer in raw["layer_evidence"]
        for row in layer["attention_geometry_risks"]
    )
    assert "query_key_rope_pair_tail_fraction" in attention_row
    assert "value_context_rope_pair_tail_fraction" in attention_row

    stale = json.loads(json.dumps(dict(raw)))
    stale["schema"] = "neural-forge-lfm2-cached-capacity-option-bank-v4"
    with pytest.raises(
        FoldPhaseCacheIntegrityError,
        match="schema is unsupported",
    ):
        global_frontier._restore_capacity_option_bank(stale)

    missing = json.loads(json.dumps(dict(raw)))
    missing_row = next(
        row
        for layer in missing["layer_evidence"]
        for row in layer["attention_geometry_risks"]
    )
    missing_row.pop("query_key_rope_pair_tail_fraction")
    with pytest.raises(FoldPhaseCacheIntegrityError):
        global_frontier._restore_capacity_option_bank(missing)

    inconsistent = json.loads(json.dumps(dict(raw)))
    inconsistent_row = next(
        row
        for layer in inconsistent["layer_evidence"]
        for row in layer["attention_geometry_risks"]
    )
    inconsistent_row["predicted_defect"] = 0.0
    inconsistent_body = dict(inconsistent)
    inconsistent_body.pop("evidence_sha256")
    inconsistent["evidence_sha256"] = canonical_sha256(inconsistent_body)
    with pytest.raises(
        FoldPhaseCacheIntegrityError,
        match="failed typed authentication",
    ) as inconsistent_error:
        global_frontier._restore_capacity_option_bank(inconsistent)
    assert "predicted_defect must equal the maximum" in str(
        inconsistent_error.value.__cause__
    )

    full_dim_tamper = json.loads(json.dumps(dict(raw)))
    full_dim_row = next(
        row
        for layer in full_dim_tamper["layer_evidence"]
        for row in layer["attention_geometry_risks"]
    )
    full_dim_row["value_context_rope_pair_tail_fraction"] = 0.999
    full_dim_row["predicted_defect"] = 0.999
    full_dim_body = dict(full_dim_tamper)
    full_dim_body.pop("evidence_sha256")
    full_dim_tamper["evidence_sha256"] = canonical_sha256(full_dim_body)
    with pytest.raises(
        FoldPhaseCacheIntegrityError,
        match="failed typed authentication",
    ) as full_dim_error:
        global_frontier._restore_capacity_option_bank(full_dim_tamper)
    assert "exact zero RoPE-pair tails" in str(full_dim_error.value.__cause__)


def test_global_frontier_phase_cache_resumes_after_capacity_only_interruption(
    tmp_path,
) -> None:
    _parent_model, graph, cache, kwargs = _phase_cache_case(tmp_path)
    phase_root = tmp_path / "derived-phases"

    def interrupt_after_capacity(stage, details):
        if (
            stage == "global_topology_phase_cache_published"
            and details["phase_name"] == GLOBAL_TOPOLOGY_CAPACITY_PHASE
        ):
            raise RuntimeError("injected failure after capacity publication")

    context = Lfm2GlobalTopologyPhaseCacheContext(
        cache=DynamicFoldPhaseCache(phase_root),
        evidence_split_sha256=_digest("interrupted-phase-cache-splits"),
        progress=interrupt_after_capacity,
    )
    with pytest.raises(
        RuntimeError,
        match="injected failure after capacity publication",
    ):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=context,
            **kwargs,
        )
    assert len(list((phase_root / GLOBAL_TOPOLOGY_CAPACITY_PHASE).iterdir())) == 1
    assert not (phase_root / GLOBAL_TOPOLOGY_SELECTION_PHASE).exists()
    progress = []
    result = build_lfm2_cached_global_topology_frontier(
        graph,
        cache,
        phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
            cache=DynamicFoldPhaseCache(phase_root),
            evidence_split_sha256=_digest("interrupted-phase-cache-splits"),
            progress=lambda stage, details: progress.append((stage, dict(details))),
        ),
        **kwargs,
    )
    assert result.selected_layout.descriptor.accounting.unique_parameter_elements == (
        8_720
    )
    assert any(
        stage == "global_topology_phase_cache_hit"
        and details["phase_name"] == GLOBAL_TOPOLOGY_CAPACITY_PHASE
        for stage, details in progress
    )
    assert any(
        stage == "global_topology_phase_cache_published"
        and details["phase_name"] == GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE
        for stage, details in progress
    )


def test_historical_capacity_reuse_rejects_semantic_neighbor_and_payload_tamper(
    tmp_path,
    monkeypatch,
) -> None:
    _parent_model, graph, cache, kwargs = _phase_cache_case(tmp_path)
    phase_root = tmp_path / "historical-derived-phases"
    with monkeypatch.context() as historical_patch:
        historical_patch.setattr(
            global_frontier,
            "_global_algorithm_fingerprint",
            lambda: _PUBLISHED_GLOBAL_ALGORITHM_SHA256,
        )
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest("historical-context-splits"),
            ),
            **kwargs,
        )
    capacity_entry = next((phase_root / GLOBAL_TOPOLOGY_CAPACITY_PHASE).iterdir())
    completion = json.loads(
        (capacity_entry / "complete.json").read_text(encoding="ascii")
    )
    payload_path = capacity_entry / "payload" / "evidence.json"
    payload = json.loads(payload_path.read_text(encoding="ascii"))
    request = Lfm2ExistingGlobalTopologyPhaseReuseRequest(
        phase_name=GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        phase_key_sha256=capacity_entry.name,
        expected_completion_sha256=completion["completion_sha256"],
        expected_evidence_sha256=payload["capacity_option_bank"]["evidence_sha256"],
        expected_compatibility_receipt_sha256=_compatibility_receipt_sha256(),
    )
    current_algorithm_sha256 = global_frontier._global_algorithm_fingerprint()
    monkeypatch.setattr(
        global_frontier,
        "_global_algorithm_fingerprint",
        lambda: current_algorithm_sha256,
    )
    monkeypatch.setattr(
        global_frontier,
        "build_lfm2_cached_heterogeneous_capacity_option_bank",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid historical capacity phase attempted a cold build"
        ),
    )
    with monkeypatch.context() as receipt_patch:
        receipt_patch.setattr(
            global_frontier,
            "_read_phase_payload",
            lambda *_args, **_kwargs: pytest.fail(
                "compatibility receipt neighbor reached evidence loading"
            ),
        )
        with pytest.raises(
            FoldPhaseCacheIntegrityError,
            match="no exact directional compatibility receipt",
        ):
            build_lfm2_cached_global_topology_frontier(
                graph,
                cache,
                phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                    cache=DynamicFoldPhaseCache(phase_root),
                    evidence_split_sha256=_digest("historical-context-splits"),
                    selective_session_factory=lambda: pytest.fail(
                        "compatibility neighbor opened a selective session"
                    ),
                    existing_capacity_phase_reuse=replace(
                        request,
                        expected_compatibility_receipt_sha256="f" * 64,
                    ),
                ),
                **kwargs,
            )
    for field_name in (
        "expected_completion_sha256",
        "expected_evidence_sha256",
    ):
        with pytest.raises(
            FoldPhaseCacheIntegrityError,
            match="no exact directional compatibility receipt",
        ):
            build_lfm2_cached_global_topology_frontier(
                graph,
                cache,
                phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                    cache=DynamicFoldPhaseCache(phase_root),
                    evidence_split_sha256=_digest("historical-context-splits"),
                    selective_session_factory=lambda: pytest.fail(
                        "digest neighbor opened a selective session"
                    ),
                    existing_capacity_phase_reuse=replace(
                        request,
                        **{field_name: "f" * 64},
                    ),
                ),
                **kwargs,
            )
    with pytest.raises(
        FoldPhaseCacheIntegrityError,
        match="outside dependency_sha256s.global_topology_algorithms",
    ):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest("neighbor-context-splits"),
                selective_session_factory=lambda: pytest.fail(
                    "semantic neighbor opened a selective session"
                ),
                existing_capacity_phase_reuse=request,
            ),
            **kwargs,
        )
    raw_payload = bytearray(payload_path.read_bytes())
    raw_payload[len(raw_payload) // 2] ^= 1
    payload_path.write_bytes(bytes(raw_payload))
    with pytest.raises(FoldPhaseCacheIntegrityError):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest("historical-context-splits"),
                selective_session_factory=lambda: pytest.fail(
                    "tampered historical phase opened a selective session"
                ),
                existing_capacity_phase_reuse=request,
            ),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("phase_name", "evidence_field", "mutation"),
    (
        (
            GLOBAL_TOPOLOGY_CAPACITY_PHASE,
            "capacity_option_bank",
            lambda evidence: evidence.__setitem__(
                "source_graph_fingerprint",
                _digest("forged-source-graph"),
            ),
        ),
        (
            GLOBAL_TOPOLOGY_CAPACITY_PHASE,
            "capacity_option_bank",
            lambda evidence: evidence.__setitem__(
                "parent_cache_key",
                _digest("forged-parent-cache"),
            ),
        ),
        (
            GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
            "functional_width_evidence",
            lambda evidence: evidence.__setitem__(
                "source_graph_fingerprint",
                _digest("forged-functional-source-graph"),
            ),
        ),
        (
            GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
            "functional_width_evidence",
            lambda evidence: evidence.__setitem__(
                "parent_cache_key",
                _digest("forged-functional-parent-cache"),
            ),
        ),
        (
            GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
            "functional_width_evidence",
            lambda evidence: evidence.__setitem__(
                "source_hidden_size",
                evidence["source_hidden_size"] + 8,
            ),
        ),
        (
            GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
            "functional_width_evidence",
            lambda evidence: evidence["candidate_scores"].pop(),
        ),
    ),
)
def test_phase_cache_rejects_resealed_evidence_for_different_active_context(
    tmp_path,
    monkeypatch,
    phase_name,
    evidence_field,
    mutation,
) -> None:
    _parent_model, graph, cache, kwargs = _phase_cache_case(tmp_path)
    phase_root = tmp_path / "derived-phases"
    context = Lfm2GlobalTopologyPhaseCacheContext(
        cache=DynamicFoldPhaseCache(phase_root),
        evidence_split_sha256=_digest("context-binding-splits"),
    )
    build_lfm2_cached_global_topology_frontier(
        graph,
        cache,
        phase_cache_context=context,
        **kwargs,
    )
    entry = next((phase_root / phase_name).iterdir())
    original_payload, original_completion = _forge_resealed_phase_evidence(
        entry,
        evidence_field,
        mutation,
    )
    trace_cache_interface_sha256 = (
        global_frontier._trace_cache_topology_interface_fingerprint()
    )

    def forbidden_session(_self):
        raise AssertionError("invalid hot evidence must fail before session open")

    monkeypatch.setattr(
        global_frontier,
        "_trace_cache_topology_interface_fingerprint",
        lambda: trace_cache_interface_sha256,
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceCache,
        "selective_authenticated_session",
        forbidden_session,
    )
    with pytest.raises(
        FoldPhaseCacheIntegrityError,
        match=r"bound to (?:a )?different",
    ):
        build_lfm2_cached_global_topology_frontier(
            graph,
            cache,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest("context-binding-splits"),
            ),
            **kwargs,
        )
    (entry / "payload" / "evidence.json").write_bytes(original_payload)
    (entry / "complete.json").write_bytes(original_completion)


def _mlp_risks() -> tuple[Lfm2CachedMlpCapacityRisk, ...]:
    return tuple(
        Lfm2CachedMlpCapacityRisk(
            intermediate_size=width,
            source_intermediate_size=32,
            fold_method_id=MLP_CAPACITY_RISK_METHOD,
            post_swiglu_tail_fraction=defect * defect,
            signed_regime_tail_fraction=0.0,
            tangent_tail_fraction=0.0,
            downstream_weighted_tail_fraction=0.0,
            collision_fraction=0.0,
            predicted_defect=defect,
        )
        for width, defect in ((8, 0.2), (10, 0.1))
    )


def _writer_risk(rank: int, defect: float) -> Lfm2CachedShortConvWriterRisk:
    tail = defect * defect
    return Lfm2CachedShortConvWriterRisk(
        writer_rank=rank,
        source_output_width=16,
        fold_method_id=SHORT_CONV_WRITER_ACTION_RISK_METHOD,
        output_action_tail_fraction=tail,
        sampling_uncertainty_fraction=0.0,
        regularized_tail_fraction=tail,
        observed_effective_rank=8.0,
        predicted_defect=defect,
        sampled_output_row_count=32,
    )


def _synthetic_union_bank() -> tuple[
    Lfm2CachedHeterogeneousCapacityOptionBank,
    tuple[str, str],
]:
    sources = (_digest("factor-site-a"), _digest("factor-site-b"))
    writer_defects = (((2, 0.2), (4, 0.19)), ((2, 0.3), (4, 0.01)))
    evidence = tuple(
        Lfm2CachedLayerCapacityEvidence(
            layer_index=index,
            layer_type="conv",
            source_layer_sha256=source,
            operator_delta_energy_fraction=1.0,
            mlp_delta_energy_fraction=1.0,
            sampled_scalar_count=32,
            nonlinear_sampled_row_count=32,
            source_mlp_intermediate_size=32,
            mlp_capacity_risks=_mlp_risks(),
            short_conv_writer_risks=tuple(
                _writer_risk(rank, defect) for rank, defect in writer_defects[index]
            ),
            selection_evidence_sha256=_digest(f"factor-evidence-{index}"),
        )
        for index, source in enumerate(sources)
    )
    options = tuple(
        build_lfm2_short_conv_layer_option(
            source_layer_sha256=source,
            mlp_intermediate_size=width,
            internal_width=internal,
            selection_evidence_sha256=_digest(f"dense-{source}-{width}-{internal}"),
            structural_risk=0.2 if width == 8 else 0.1,
        )
        for source in sources
        for width in (8, 10)
        for internal in (8, 10)
    )
    values = {
        "source_graph_fingerprint": _digest("synthetic-source-graph"),
        "parent_cache_key": _digest("synthetic-parent-cache"),
        "mlp_option_policy": EXPLICIT_MLP_OPTION_POLICY,
        "attention_geometry_policy": ATTENTION_GEOMETRY_ENABLED_POLICY,
        "exact_budget_lattice_sha256": _digest("synthetic-union-lattice"),
        "layer_evidence": evidence,
        "layer_options": options,
    }
    provisional = object.__new__(Lfm2CachedHeterogeneousCapacityOptionBank)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    return (
        Lfm2CachedHeterogeneousCapacityOptionBank(
            **values,
            evidence_sha256=canonical_sha256(provisional._body()),
        ),
        sources,
    )


def test_global_retarget_reselects_factor_site_for_each_policy_rank() -> None:
    base, (source_a, source_b) = _synthetic_union_bank()

    def policy(rank: int) -> Lfm2MlpWidthOptionPolicy:
        return Lfm2MlpWidthOptionPolicy(
            policy_id=AUTOMATIC_MLP_OPTION_POLICY,
            widths_by_layer=((8, 10), (8, 10)),
            exact_budget_lattice_sha256=_digest(f"rank-{rank}-lattice"),
            short_conv_writer_ranks=(rank,),
            exact_witness_factor_writer_count=1,
            exact_witness_search_candidate_count=10,
            exact_witness_search_retained_count=2,
        )

    low_rank = global_frontier._retarget_capacity_option_bank(
        base,
        policy(2),
        target_hidden_size=16,
    )
    high_rank = global_frontier._retarget_capacity_option_bank(
        base,
        policy(4),
        target_hidden_size=16,
    )

    low_sources = {
        value.source_layer_sha256
        for value in low_rank.layer_options
        if value.short_conv_writer_rank is not None
    }
    high_sources = {
        value.source_layer_sha256
        for value in high_rank.layer_options
        if value.short_conv_writer_rank is not None
    }
    assert low_sources == {source_a}
    assert high_sources == {source_b}


def _candidate(
    hidden_size: int,
    *,
    functional_loss: float,
    component_risk: float,
    suffix: str,
) -> Lfm2GlobalTopologyCandidateReport:
    digests = [_digest(f"{suffix}-{index}") for index in range(6)]
    return Lfm2GlobalTopologyCandidateReport.build(
        hidden_size=hidden_size,
        exact_unique_parameters=10_000,
        target_layer_count=4,
        functional_retention=1.0 - functional_loss,
        functional_loss=functional_loss,
        mean_component_risk=component_risk,
        functional_score_sha256=digests[0],
        mlp_width_lattice_sha256=digests[1],
        capacity_option_bank_sha256=digests[2],
        selection_input_sha256=digests[3],
        descriptor_sha256=digests[4],
        layout_selection_receipt_sha256=digests[5],
    )


def test_global_selection_uses_cached_function_and_component_risk_not_width() -> None:
    narrow = _candidate(
        8,
        functional_loss=0.30,
        component_risk=0.02,
        suffix="narrow",
    )
    wide = _candidate(
        24,
        functional_loss=0.10,
        component_risk=0.03,
        suffix="wide",
    )
    assert select_lfm2_global_topology_report((narrow, wide)) is wide

    fragile_wide = _candidate(
        24,
        functional_loss=0.10,
        component_risk=0.50,
        suffix="fragile-wide",
    )
    assert select_lfm2_global_topology_report((narrow, fragile_wide)) is narrow


def test_global_candidate_seal_fails_closed_on_score_tampering() -> None:
    candidate = _candidate(
        16,
        functional_loss=0.20,
        component_risk=0.10,
        suffix="sealed",
    )
    with pytest.raises(
        Lfm2CachedGlobalTopologyFrontierError,
        match="Combined frontier risk differs|candidate seal is invalid",
    ):
        replace(candidate, combined_predicted_risk=0.01)
