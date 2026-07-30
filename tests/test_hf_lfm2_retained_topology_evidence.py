from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
from types import MappingProxyType, SimpleNamespace
import weakref

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from experiments import run_lfm2_residual_attention_fold as residual  # noqa: E402
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache  # noqa: E402
import neural_forge.hf_lfm2_cached_heterogeneous_capacity as capacity  # noqa: E402
from neural_forge.hf_lfm2_cached_functional_width import (  # noqa: E402
    Lfm2CachedFunctionalWidthError,
    build_lfm2_cached_functional_width_evidence,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (  # noqa: E402
    Lfm2CachedHeterogeneousCapacityError,
    build_lfm2_cached_heterogeneous_capacity_option_bank,
)
from neural_forge.hf_lfm2_cached_global_topology_frontier import (  # noqa: E402
    GLOBAL_TOPOLOGY_CAPACITY_PHASE,
    GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
    GLOBAL_TOPOLOGY_SELECTION_PHASE,
    Lfm2CachedGlobalTopologyEvidenceCacheMiss,
    Lfm2CachedGlobalTopologyFrontierError,
    Lfm2GlobalTargetRuntimePins,
    Lfm2GlobalTopologyPhaseCacheContext,
    build_lfm2_cached_global_topology_frontier,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CACHED_VERIFIED_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    build_lfm2_circuit_trace_cache,
)
from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (  # noqa: E402
    _parent,
)
from test_hf_lfm2_dynamic_closed_loop_adapter import _batch  # noqa: E402


def _phase_order(record) -> tuple[str, ...]:
    phases = record["capture"]["phase_diagnostics"]

    def key(phase: str) -> tuple[int, int]:
        if phase in {"prefill", "cached_prefill"}:
            return (0, 0)
        return (1, int(phase.removeprefix("decode.")))

    return tuple(sorted(phases, key=key))


def _capacity_kwargs() -> dict[str, object]:
    return {
        "short_conv_internal_widths": (12, 14),
        "short_conv_writer_ranks": (2, 4),
        "attention_head_options": ((2, 1, 8), (4, 2, 8)),
        "mlp_intermediate_sizes": (18, 20),
        "target_hidden_size": 24,
        "attention_geometry_risk_enabled": True,
    }


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _global_kwargs() -> dict[str, object]:
    return {
        "runtime": Lfm2GlobalTargetRuntimePins(
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
            target_vocab_evidence_sha256=_digest("retained-target-vocabulary"),
            token_bridge_evidence_sha256=_digest("retained-token-bridge"),
        ),
        "exact_target_unique_parameters": 8_720,
        "minimum_hidden_size": 8,
        "hidden_size_multiple": 8,
        "short_conv_internal_widths": (4, 8),
        "attention_head_options": ((2, 1, 8), (4, 2, 8)),
        "maximum_functional_rows_per_role": 24,
    }


@pytest.fixture(scope="module", params=(torch.float32, torch.bfloat16))
def retained_case(request, tmp_path_factory):
    dtype = request.param
    root = tmp_path_factory.mktemp(str(dtype).replace("torch.", ""))
    parent = _parent().to(dtype=dtype).eval()
    parent.config._attn_implementation = "eager"
    graph = inspect_model_circuits(
        parent,
        model_name=f"retained-topology-{dtype}",
        tokenizer_fingerprint=("1" if dtype == torch.float32 else "2") * 64,
    )
    cache = build_lfm2_circuit_trace_cache(
        root / "parent-cache",
        parent,
        [
            _batch((1, 2, 3, 4), (5,)),
            _batch((6, 7, 8, 9), (10,)),
        ],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            max_capture_bytes=128 * 1024 * 1024,
            decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    with cache.selective_authenticated_session() as session:
        rows = residual.extract_trace_rows(
            cache,
            batch_splits=("fit", "validation"),
            selective_session=session,
        )
    return SimpleNamespace(
        dtype=dtype,
        parent=parent,
        graph=graph,
        cache=cache,
        rows=rows,
    )


def test_hybrid_b1_b2_exact_parity_and_selection_counts(
    retained_case,
    monkeypatch,
) -> None:
    case = retained_case
    with case.cache.selective_authenticated_session() as session:
        legacy_capacity = build_lfm2_cached_heterogeneous_capacity_option_bank(
            case.graph,
            case.cache,
            selective_session=session,
            **_capacity_kwargs(),
        )
        legacy_functional = build_lfm2_cached_functional_width_evidence(
            case.graph,
            case.cache,
            (8, 16, 24, 32),
            selective_session=session,
            maximum_rows_per_role=24,
        )

    selections: list[tuple[int, tuple[str, ...]]] = []
    original_load = Lfm2CircuitTraceSelectiveAuthenticatedSession.load_selected_tensors

    def tracked_load(session, index, names):
        selections.append((index, tuple(names)))
        return original_load(session, index, names)

    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "load_selected_tensors",
        tracked_load,
    )
    hybrid_session = case.cache.selective_authenticated_session()
    try:
        hybrid_capacity = build_lfm2_cached_heterogeneous_capacity_option_bank(
            case.graph,
            case.cache,
            selective_session=hybrid_session,
            retained_parent_rows=case.rows,
            **_capacity_kwargs(),
        )
        assert len(selections) == case.cache.batch_count
        for batch_index, names in selections:
            reference = hybrid_session.batch_references[batch_index]
            expected = set()
            for phase in _phase_order(reference.record):
                for layer_index, layer_kind in enumerate(case.rows.layer_kinds):
                    prefix = f"{phase}.layer.{layer_index:03d}."
                    expected.update(
                        prefix + endpoint
                        for endpoint in (
                            "operator.input",
                            "mlp.input",
                            "mlp.down",
                        )
                    )
                    if layer_kind == "short_conv":
                        expected.add(prefix + "operator.output")
                    else:
                        expected.update(
                            prefix + endpoint
                            for endpoint in (
                                "attention.key_history",
                                "attention.value_history",
                                "attention.qk_scores",
                                "attention.additive_mask",
                                "attention.probabilities",
                                "attention.context_from_probabilities",
                            )
                        )
            assert names == tuple(sorted(expected))

        selections.clear()
        hybrid_functional = build_lfm2_cached_functional_width_evidence(
            case.graph,
            case.cache,
            (8, 16, 24, 32),
            selective_session=hybrid_session,
            retained_parent_rows=case.rows,
            maximum_rows_per_role=24,
        )
        assert selections == []
        assert hybrid_session.closed is False
    finally:
        hybrid_session.close()

    assert hybrid_capacity.as_dict() == legacy_capacity.as_dict()
    assert hybrid_capacity.evidence_sha256 == legacy_capacity.evidence_sha256
    assert hybrid_functional.as_dict() == legacy_functional.as_dict()
    assert hybrid_functional.evidence_sha256 == legacy_functional.evidence_sha256


def test_global_cold_path_defers_then_uses_rows_and_hot_path_opens_nothing(
    retained_case,
    monkeypatch,
    tmp_path,
) -> None:
    case = retained_case
    progress: list[tuple[str, dict[str, object]]] = []
    factory_calls = []

    def forbidden_cold_factory():
        factory_calls.append(True)
        raise AssertionError("deferred cold planning must not open a session")

    context = Lfm2GlobalTopologyPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "derived-phases"),
        evidence_split_sha256=_digest(f"retained-{case.dtype}-splits"),
        progress=lambda stage, details: progress.append((stage, dict(details))),
        selective_session_factory=forbidden_cold_factory,
    )
    with pytest.raises(
        Lfm2CachedGlobalTopologyEvidenceCacheMiss,
        match="retained parent rows are not materialized",
    ):
        build_lfm2_cached_global_topology_frontier(
            case.graph,
            case.cache,
            phase_cache_context=context,
            defer_evidence_cache_miss=True,
            **_global_kwargs(),
        )
    assert factory_calls == []
    assert any(
        stage == "global_topology_evidence_compute_deferred"
        and details["selective_session_opened"] is False
        for stage, details in progress
    )

    selections: list[tuple[int, tuple[str, ...]]] = []
    original_load = Lfm2CircuitTraceSelectiveAuthenticatedSession.load_selected_tensors

    def tracked_load(session, index, names):
        selections.append((index, tuple(names)))
        return original_load(session, index, names)

    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "load_selected_tensors",
        tracked_load,
    )
    session = case.cache.selective_authenticated_session()
    try:
        cold = build_lfm2_cached_global_topology_frontier(
            case.graph,
            case.cache,
            selective_session=session,
            retained_parent_rows=case.rows,
            phase_cache_context=context,
            **_global_kwargs(),
        )
        assert session.closed is False
    finally:
        session.close()
    assert len(selections) == case.cache.batch_count
    assert [index for index, _names in selections] == list(
        range(case.cache.batch_count)
    )
    capacity_progress = [
        details
        for stage, details in progress
        if stage == "global_topology_capacity_batch_progress"
    ]
    assert 0 < len(capacity_progress) <= 16
    assert capacity_progress[-1]["completed_batch_count"] == case.cache.batch_count
    assert capacity_progress[-1]["total_batch_count"] == case.cache.batch_count
    assert capacity_progress[-1]["retained_parent_rows"] is True
    assert capacity_progress[-1]["retained_tensor_name_count"] > 0
    assert capacity_progress[-1]["supplemental_tensor_name_count"] > 0
    assert capacity_progress[-1]["supplemental_selection_count"] == (
        case.cache.batch_count
    )
    assert capacity_progress[-1]["maximum_progress_event_count"] == 16

    selections.clear()
    legacy_root = tmp_path / "legacy-derived-phases"
    legacy = build_lfm2_cached_global_topology_frontier(
        case.graph,
        case.cache,
        phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
            cache=DynamicFoldPhaseCache(legacy_root),
            evidence_split_sha256=_digest(f"retained-{case.dtype}-splits"),
        ),
        **_global_kwargs(),
    )
    assert dict(legacy.as_dict()) == dict(cold.as_dict())
    for phase_name in (
        GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        GLOBAL_TOPOLOGY_SELECTION_PHASE,
    ):
        hybrid_keys = sorted(
            path.name
            for path in (tmp_path / "derived-phases" / phase_name).iterdir()
        )
        legacy_keys = sorted(path.name for path in (legacy_root / phase_name).iterdir())
        assert hybrid_keys == legacy_keys

    selections.clear()
    hot_factory_calls = []

    def forbidden_hot_factory():
        hot_factory_calls.append(True)
        raise AssertionError("stable phase-cache hit must not open a session")

    hot = build_lfm2_cached_global_topology_frontier(
        case.graph,
        case.cache,
        retained_parent_rows=case.rows,
        phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
            cache=DynamicFoldPhaseCache(tmp_path / "derived-phases"),
            evidence_split_sha256=_digest(f"retained-{case.dtype}-splits"),
            selective_session_factory=forbidden_hot_factory,
        ),
        **_global_kwargs(),
    )
    assert dict(hot.as_dict()) == dict(cold.as_dict())
    assert selections == []
    assert hot_factory_calls == []


def _mutated_rows(rows, defect: str):
    if defect == "shuffled_paths":
        paths = list(rows.paths)
        paths[0], paths[1] = paths[1], paths[0]
        return replace(rows, paths=tuple(paths))
    fields = {
        index: dict(values) for index, values in rows.layer_fields.items()
    }
    if defect == "missing_endpoint":
        fields[0].pop("residual.pre")
    elif defect == "foreign_endpoint":
        fields[0]["foreign.endpoint"] = fields[0]["residual.pre"]
    elif defect == "wrong_dtype":
        fields[0]["residual.pre"] = fields[0]["residual.pre"].to(torch.float64)
    elif defect == "wrong_shape":
        fields[0]["residual.pre"] = fields[0]["residual.pre"][:-1].contiguous()
    elif defect == "nonfinite":
        value = fields[0]["residual.pre"].clone()
        value.fill_(float("inf"))
        fields[0]["residual.pre"] = value
    elif defect == "finite_value":
        value = fields[0]["residual.pre"].clone()
        flat = value.reshape(-1)
        flat[0] = 123.0 if float(flat[0]) != 123.0 else 124.0
        fields[0]["residual.pre"] = value
    else:  # pragma: no cover - test helper contract
        raise AssertionError(defect)
    return replace(
        rows,
        layer_fields=MappingProxyType(
            {
                index: MappingProxyType(values)
                for index, values in fields.items()
            }
        ),
    )


@pytest.mark.parametrize(
    "defect",
    (
        "shuffled_paths",
        "missing_endpoint",
        "foreign_endpoint",
        "wrong_dtype",
        "wrong_shape",
        "nonfinite",
    ),
)
def test_retained_rows_fail_closed_on_lineage_and_tensor_defects(
    retained_case,
    defect,
) -> None:
    invalid = _mutated_rows(retained_case.rows, defect)
    with pytest.raises(
        (Lfm2CachedFunctionalWidthError, Lfm2CachedHeterogeneousCapacityError),
        match="Retained",
    ):
        build_lfm2_cached_functional_width_evidence(
            retained_case.graph,
            retained_case.cache,
            (8, 16, 24, 32),
            retained_parent_rows=invalid,
            maximum_rows_per_role=24,
        )


def test_finite_value_tamper_fails_b1_b2_and_publishes_no_phase_evidence(
    retained_case,
    tmp_path,
) -> None:
    case = retained_case
    invalid = _mutated_rows(case.rows, "finite_value")
    assert invalid.trace_rows_sha256 == case.rows.trace_rows_sha256
    assert torch.isfinite(invalid.layer_fields[0]["residual.pre"]).all()

    with pytest.raises(
        Lfm2CachedFunctionalWidthError,
        match="Retained.*(?:SHA|identity|authentication)",
    ):
        build_lfm2_cached_functional_width_evidence(
            case.graph,
            case.cache,
            (8, 16, 24, 32),
            retained_parent_rows=invalid,
            maximum_rows_per_role=24,
        )
    with pytest.raises(
        Lfm2CachedHeterogeneousCapacityError,
        match="Retained.*(?:SHA|identity|authentication)",
    ):
        build_lfm2_cached_heterogeneous_capacity_option_bank(
            case.graph,
            case.cache,
            retained_parent_rows=invalid,
            **_capacity_kwargs(),
        )

    phase_root = tmp_path / "tampered-derived-phases"
    with pytest.raises(
        Lfm2CachedGlobalTopologyFrontierError,
        match="Retained.*authentication",
    ):
        build_lfm2_cached_global_topology_frontier(
            case.graph,
            case.cache,
            retained_parent_rows=invalid,
            phase_cache_context=Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(phase_root),
                evidence_split_sha256=_digest(f"finite-tamper-{case.dtype}"),
            ),
            **_global_kwargs(),
        )
    assert list(phase_root.rglob("complete.json")) == []


def test_previous_supplemental_tensors_die_before_next_batch_load(
    retained_case,
    monkeypatch,
) -> None:
    previous_refs: list[weakref.ReferenceType] = []
    observed_prior_release = []
    original_load = Lfm2CircuitTraceSelectiveAuthenticatedSession.load_selected_tensors

    def tracked_load(session, index, names):
        if previous_refs:
            gc.collect()
            observed_prior_release.append(all(ref() is None for ref in previous_refs))
            assert observed_prior_release[-1] is True
        payload = original_load(session, index, names)
        previous_refs[:] = [weakref.ref(value) for value in payload.tensors.values()]
        return payload

    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "load_selected_tensors",
        tracked_load,
    )
    build_lfm2_cached_heterogeneous_capacity_option_bank(
        retained_case.graph,
        retained_case.cache,
        retained_parent_rows=retained_case.rows,
        **_capacity_kwargs(),
    )
    gc.collect()
    assert observed_prior_release == [True] * (retained_case.cache.batch_count - 1)
    assert all(ref() is None for ref in previous_refs)


def test_owned_hybrid_session_closes_and_releases_once_per_selection(
    retained_case,
    monkeypatch,
) -> None:
    close_calls = []
    release_calls = []
    supplemental_refs: list[weakref.ReferenceType] = []
    original_close = Lfm2CircuitTraceSelectiveAuthenticatedSession.close
    original_release = (
        Lfm2CircuitTraceSelectiveAuthenticatedSession.release_selected_tensors
    )
    original_load = Lfm2CircuitTraceSelectiveAuthenticatedSession.load_selected_tensors

    def tracked_close(session):
        close_calls.append(session)
        return original_close(session)

    def tracked_release(session, payload):
        release_calls.append((session, payload))
        return original_release(session, payload)

    def tracked_load(session, index, names):
        payload = original_load(session, index, names)
        supplemental_refs.extend(
            weakref.ref(value) for value in payload.tensors.values()
        )
        return payload

    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "close",
        tracked_close,
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "release_selected_tensors",
        tracked_release,
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "load_selected_tensors",
        tracked_load,
    )
    build_lfm2_cached_heterogeneous_capacity_option_bank(
        retained_case.graph,
        retained_case.cache,
        retained_parent_rows=retained_case.rows,
        **_capacity_kwargs(),
    )
    assert len(close_calls) == 1
    assert len(release_calls) == retained_case.cache.batch_count
    assert all(payload.released for _session, payload in release_calls)
    gc.collect()
    assert supplemental_refs
    assert all(ref() is None for ref in supplemental_refs)


def test_hybrid_processing_exception_releases_payload_and_closes_session_once(
    retained_case,
    monkeypatch,
) -> None:
    close_calls = []
    release_calls = []
    supplemental_refs: list[weakref.ReferenceType] = []
    original_close = Lfm2CircuitTraceSelectiveAuthenticatedSession.close
    original_release = (
        Lfm2CircuitTraceSelectiveAuthenticatedSession.release_selected_tensors
    )
    original_load = Lfm2CircuitTraceSelectiveAuthenticatedSession.load_selected_tensors

    def tracked_close(session):
        close_calls.append(session)
        return original_close(session)

    def tracked_release(session, payload):
        release_calls.append((session, payload))
        return original_release(session, payload)

    def tracked_load(session, index, names):
        payload = original_load(session, index, names)
        supplemental_refs.extend(
            weakref.ref(value) for value in payload.tensors.values()
        )
        return payload

    def injected_failure(**_kwargs):
        raise RuntimeError("injected hybrid selector consumer failure")

    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "close",
        tracked_close,
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "release_selected_tensors",
        tracked_release,
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
        "load_selected_tensors",
        tracked_load,
    )
    monkeypatch.setattr(
        capacity,
        "_cached_mlp_nonlinear_statistics",
        injected_failure,
    )
    with pytest.raises(RuntimeError, match="injected hybrid"):
        build_lfm2_cached_heterogeneous_capacity_option_bank(
            retained_case.graph,
            retained_case.cache,
            retained_parent_rows=retained_case.rows,
            **_capacity_kwargs(),
        )
    assert len(close_calls) == 1
    assert len(release_calls) == 1
    assert release_calls[0][1].released is True
    gc.collect()
    assert supplemental_refs
    assert all(ref() is None for ref in supplemental_refs)


def test_externally_supplied_hybrid_session_is_not_closed(
    retained_case,
) -> None:
    session = retained_case.cache.selective_authenticated_session()
    try:
        build_lfm2_cached_heterogeneous_capacity_option_bank(
            retained_case.graph,
            retained_case.cache,
            selective_session=session,
            retained_parent_rows=retained_case.rows,
            **_capacity_kwargs(),
        )
        assert session.closed is False
    finally:
        session.close()
