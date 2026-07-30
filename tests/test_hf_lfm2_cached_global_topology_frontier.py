from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (  # noqa: E402
    _parent,
)
from test_hf_lfm2_dynamic_closed_loop_adapter import _batch  # noqa: E402

from neural_forge.dynamic_behavior_gate import observe_model_execution  # noqa: E402
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
import neural_forge.hf_lfm2_cached_global_topology_frontier as global_frontier  # noqa: E402
from neural_forge.hf_lfm2_cached_global_topology_frontier import (  # noqa: E402
    Lfm2CachedGlobalTopologyFrontierError,
    Lfm2GlobalTargetRuntimePins,
    Lfm2GlobalTopologyCandidateReport,
    build_lfm2_cached_global_topology_frontier,
    select_lfm2_global_topology_report,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (  # noqa: E402
    AUTOMATIC_MLP_OPTION_POLICY,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CACHED_VERIFIED_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


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
    original_builder = (
        global_frontier.build_lfm2_cached_heterogeneous_capacity_option_bank
    )

    def tracked_builder(*args, **kwargs):
        nonlinear_builds.append((args, kwargs))
        return original_builder(*args, **kwargs)

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
            exact_target_unique_parameters=6_776,
            minimum_hidden_size=8,
            hidden_size_multiple=8,
            short_conv_internal_widths=(4, 8),
            attention_head_options=((1, 1, 2), (2, 1, 4)),
            maximum_functional_rows_per_role=16,
        )

    assert execution["root_forward_calls"] == 0
    assert execution["module_forward_calls"] == 0
    assert len(nonlinear_builds) == 1
    assert nonlinear_builds[0][1]["mlp_intermediate_sizes"] is not None
    assert frontier.candidate_hidden_sizes == (8, 16, 24, 32)
    assert [value.hidden_size for value in frontier.candidates] == [8, 16, 24]
    assert [
        (value.hidden_size, value.reason_class) for value in frontier.rejected_widths
    ] == [(32, "no_exact_integer_lattice")]
    assert all(value.exact_unique_parameters == 6_776 for value in frontier.candidates)
    assert all(value.target_layer_count == 4 for value in frontier.candidates)
    assert frontier.selected_capacity_option_bank.mlp_option_policy == (
        AUTOMATIC_MLP_OPTION_POLICY
    )
    assert frontier.selected_layout.descriptor.accounting.unique_parameter_elements == (
        6_776
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
