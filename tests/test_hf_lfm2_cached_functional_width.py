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
import neural_forge.hf_lfm2_cached_functional_width as functional_width  # noqa: E402
from neural_forge.hf_lfm2_cached_functional_width import (  # noqa: E402
    LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA,
    LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE,
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
    progress = []
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
                progress=lambda stage, details: progress.append(
                    (stage, dict(details))
                ),
            )

    assert execution["root_forward_calls"] == 0
    assert execution["module_forward_calls"] == 0
    repeated_evidence = build_lfm2_cached_functional_width_evidence(
        graph,
        cache,
        (8, 16, 24, 32),
        maximum_rows_per_role=24,
    )
    assert dict(repeated_evidence.as_dict()) == dict(evidence.as_dict())
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
    assert retained == pytest.approx(
        (0.6245758481017962, 0.8608747661642241, 0.9660240314257886, 1.0),
        rel=0.0,
        abs=5e-13,
    )
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
        "gram_accumulation": {
            "schema": LFM2_FUNCTIONAL_WIDTH_GRAM_ACCUMULATION_SCHEMA,
            "method": "deterministic-role-buffered-row-batch-addmm",
            "calculation_device": "cpu",
            "calculation_dtype": "torch.float64",
            "row_batch_size": LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE,
            "canonical_row_order": (
                "batch-index-then-location-key-then-selected-row-index-v1"
            ),
            "intraop_threads": 1,
            "bounded_pending_role_buffers": True,
        },
    }
    assert report["selection_contract"]["nonlinear_mlp_write_is_a_scored_role"]
    assert report["selection_contract"]["parameter_count_is_not_a_quality_score"]
    assert len(selections) == cache.batch_count
    assert all(payload.released for _index, _names, payload in selections)
    assert all(
        names == tuple(sorted(set(names))) and names
        for _index, names, _payload in selections
    )
    stages = [stage for stage, _details in progress]
    assert stages == [
        "functional_width_selection_started",
        "functional_width_selection_completed",
        "functional_width_gram_accumulation_started",
        "functional_width_gram_batch_progress",
        "functional_width_gram_batch_progress",
        "functional_width_gram_accumulation_completed",
        "functional_width_eigensolve_started",
        "functional_width_eigensolve_completed",
    ]
    assert all(
        details["parent_model_calls"] == 0
        and details["child_model_calls"] == 0
        and details["trace_regeneration_calls"] == 0
        and details["training_steps"] == 0
        and details["backward_calls"] == 0
        and details["optimizer_steps"] == 0
        for _stage, details in progress
    )
    assert all(
        "schema" not in details
        and all(
            value is None or isinstance(value, (bool, int, float, str))
            for value in details.values()
        )
        for _stage, details in progress
    )
    gram_completion = next(
        details
        for stage, details in progress
        if stage == "functional_width_gram_accumulation_completed"
    )
    assert gram_completion["previous_location_matmul_count"] == 78
    assert gram_completion["batched_matmul_count"] == 4
    assert gram_completion["location_matmul_count_reduction"] == 74
    assert gram_completion["peak_pending_bytes"] <= gram_completion[
        "pending_byte_scratch_bound"
    ]


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


def test_batched_role_gram_is_deterministic_bounded_and_reduces_matmuls() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260730)
    rows_by_role = {
        role: torch.nn.functional.normalize(
            torch.randn((256, 64), generator=generator, dtype=torch.float64),
            dim=1,
        )
        for role in functional_width._ROLES
    }
    legacy = {
        role: torch.zeros((64, 64), dtype=torch.float64)
        for role in functional_width._ROLES
    }
    for role in functional_width._ROLES:
        for row in rows_by_role[role].split(1, dim=0):
            legacy[role].add_(row.T @ row)

    def accumulate():
        value = functional_width._DeterministicRoleBatchedGramAccumulator(
            64,
            row_batch_size=LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE,
        )
        with functional_width._single_threaded_torch_cpu_math(torch):
            for row_index in range(256):
                for role in functional_width._ROLES:
                    value.add(role, rows_by_role[role][row_index : row_index + 1])
            return value.finalize()

    first_grams, first_diagnostics = accumulate()
    second_grams, second_diagnostics = accumulate()
    for role in functional_width._ROLES:
        assert torch.equal(first_grams[role], second_grams[role])
        assert torch.allclose(
            first_grams[role],
            legacy[role],
            rtol=0.0,
            atol=5e-13,
        )
    assert dict(first_diagnostics) == dict(second_diagnostics)
    assert first_diagnostics["row_outer_product_equivalent_count"] == 1_024
    assert first_diagnostics["previous_location_matmul_count"] == 1_024
    assert first_diagnostics["batched_matmul_count"] == 16
    assert first_diagnostics["location_matmul_count_reduction"] == 1_008
    assert first_diagnostics["peak_pending_rows"] <= first_diagnostics[
        "pending_row_scratch_bound"
    ]
    assert first_diagnostics["peak_pending_bytes"] <= first_diagnostics[
        "pending_byte_scratch_bound"
    ]
    assert first_diagnostics["maximum_flush_rows"] == (
        LFM2_FUNCTIONAL_WIDTH_GRAM_ROW_BATCH_SIZE
    )


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
