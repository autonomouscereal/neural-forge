from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from neural_forge.dynamic_behavior_gate import (  # noqa: E402
    observe_model_execution,
)
from neural_forge.dynamic_model_inspector import (  # noqa: E402
    inspect_model_circuits,
)
import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
from neural_forge.hf_lfm2_cached_functional_width import (  # noqa: E402
    LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY,
    Lfm2CachedFunctionalWidthError,
    build_lfm2_cached_functional_width_evidence,
    enumerate_lfm2_global_hidden_sizes,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (  # noqa: E402
    _parent,
)
from test_hf_lfm2_dynamic_closed_loop_adapter import _batch  # noqa: E402


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _case(tmp_path):
    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="functional-width-parent",
        tokenizer_fingerprint=_digest("functional-width-tokenizer"),
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "parent-cache",
        parent,
        [
            _batch((1, 2, 3, 4), (5, 6)),
            _batch((7, 8, 9, 10), (11, 12)),
        ],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=3,
            max_decode_steps=2,
            max_capture_bytes=128 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    return parent, graph, cache


def test_cached_functional_width_spectrum_is_bounded_authenticated_and_cache_only(
    tmp_path,
    monkeypatch,
) -> None:
    parent, graph, cache = _case(tmp_path)
    selections = []
    original_load = (
        trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession.load_selected_tensors
    )

    def tracked_load(session, index, names):
        payload = original_load(session, index, names)
        selections.append((index, tuple(names), payload))
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(
            trace_cache.Lfm2CircuitTraceSelectiveAuthenticatedSession,
            "load_selected_tensors",
            tracked_load,
        )
        with observe_model_execution(parent) as execution:
            evidence = build_lfm2_cached_functional_width_evidence(
                graph,
                cache,
                (8, 16, 24, 32),
                maximum_rows_per_role=24,
            )

    assert execution["root_forward_calls"] == 0
    assert execution["module_forward_calls"] == 0
    assert evidence.source_graph_fingerprint == graph.fingerprint
    assert evidence.parent_cache_key == cache.cache_key
    assert evidence.source_hidden_size == 32
    assert evidence.maximum_rows_per_role == 24
    sampled_rows = dict(evidence.sampled_rows_by_role)
    assert tuple(sampled_rows) == (
        "nonlinear_mlp_write",
        "operator_write",
        "residual_context",
        "terminal_context",
    )
    assert all(0 < value <= 24 for value in sampled_rows.values())
    assert [value.hidden_size for value in evidence.candidate_scores] == [
        8,
        16,
        24,
        32,
    ]
    retained = [
        value.role_balanced_functional_retention for value in evidence.candidate_scores
    ]
    assert retained == sorted(retained)
    assert retained[0] < retained[-1]
    assert retained[-1] == pytest.approx(1.0, abs=1e-12)
    assert evidence.score_for(16) == evidence.candidate_scores[1]
    report = evidence.as_dict()
    assert report["sampling"] == {
        "policy": LFM2_FUNCTIONAL_WIDTH_SAMPLING_POLICY,
        "maximum_rows_per_role": 24,
        "sampled_rows_by_role": dict(evidence.sampled_rows_by_role),
        "selection_manifest_sha256": evidence.selection_manifest_sha256,
        "row_l2_normalized_before_gram": True,
        "role_grams_normalized_to_unit_trace": True,
        "roles_weighted_equally": True,
        "raw_residual_energy_used_as_width_score": False,
    }
    assert report["selection_contract"]["nonlinear_mlp_write_is_a_scored_role"]
    assert report["selection_contract"]["parameter_count_is_not_a_quality_score"]
    assert len(selections) == cache.batch_count
    assert all(payload.released for _index, _names, payload in selections)
    assert all(
        names == tuple(sorted(set(names))) and names
        for _index, names, _payload in selections
    )


def test_functional_width_evidence_and_scores_fail_closed_on_tampering(
    tmp_path,
) -> None:
    _parent_model, graph, cache = _case(tmp_path)
    evidence = build_lfm2_cached_functional_width_evidence(
        graph,
        cache,
        (16, 32),
        maximum_rows_per_role=16,
    )
    score = evidence.candidate_scores[0]
    with pytest.raises(Lfm2CachedFunctionalWidthError, match="inconsistent"):
        replace(
            score, role_balanced_functional_loss=score.role_balanced_functional_loss / 2
        )
    with pytest.raises(Lfm2CachedFunctionalWidthError, match="seal is invalid"):
        replace(evidence, maximum_rows_per_role=17)
    with pytest.raises(Lfm2CachedFunctionalWidthError, match="No authenticated"):
        evidence.score_for(24)


def test_hidden_width_enumeration_is_dynamic_not_a_historical_shortlist() -> None:
    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="functional-width-enumeration-parent",
        tokenizer_fingerprint=_digest("functional-width-enumeration-tokenizer"),
    )
    widths = enumerate_lfm2_global_hidden_sizes(
        graph,
        exact_target_unique_parameters=9_910,
        target_vocabulary_size=64,
        minimum_hidden_size=8,
        hidden_size_multiple=8,
    )
    assert widths == (8, 16, 24, 32)
    assert enumerate_lfm2_global_hidden_sizes(
        graph,
        exact_target_unique_parameters=3_000,
        target_vocabulary_size=64,
        minimum_hidden_size=8,
        hidden_size_multiple=8,
    ) == (8, 16)
