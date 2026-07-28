from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from neural_forge import hf_lfm2_semantic_layer_stack_refold_adapter as stack
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    Lfm2MovingGaugeBoundaryEvidence,
    Lfm2MovingGaugeWriterEvidence,
    compile_lfm2_moving_gauge_residual_reclosure,
    fit_lfm2_cycle_coupled_moving_gauge_boundary,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionDownWeightPrior,
)
from neural_forge.hf_lfm2_semantic_postfold import (
    _fit_normalized_operator_input_decoder,
)
from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
    Lfm2SemanticShortConvFhvrSource,
)


def _boundary(name: str, role: str, *, locked: bool) -> object:
    generator = torch.Generator().manual_seed(8 if locked else 9)
    parent = torch.randn(20, 3, generator=generator, dtype=torch.float64)
    child = parent @ torch.tensor(
        [[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    evidence = Lfm2MovingGaugeBoundaryEvidence(
        boundary_id=name,
        interface_role=role,
        interface_locked=locked,
        base_encoder=torch.eye(3, dtype=torch.float64),
        fit_parent_rows=parent,
        fit_child_rows=child,
        fit_weights=torch.ones(20, dtype=torch.float64),
        fit_provenance=f"{name}:fit",
        row_alignment_sha256="1" * 64,
        cached_parent_evidence_sha256="2" * 64,
        live_child_recapture_sha256="3" * 64,
    )
    return fit_lfm2_cycle_coupled_moving_gauge_boundary(evidence)


def test_exact_moving_decoder_reaches_normalized_and_short_conv_consumers() -> None:
    chart = _boundary("layer.001.pre", "intermediate", locked=False)
    foundation = stack._sealed(
        {
            "schema": "neural-forge-analytic-coordinate-completion-candidate-v1",
            "coordinate_map_sha256": "4" * 64,
        }
    )
    candidate = stack._moving_gauge_coordinate_completion_candidate(
        chart,
        foundation_candidate_receipt=foundation,
    )
    exact_decoder_sha256 = stack._tensor_sha256(chart.decoder)
    assert candidate["moving_gauge_decoder_tensor_sha256"] == exact_decoder_sha256
    assert candidate["foundation_prior_candidate_sha256"] == foundation["result_sha256"]

    generator = torch.Generator().manual_seed(10)
    parents = tuple(
        torch.randn(7, 3, generator=generator, dtype=torch.float64) for _ in range(4)
    )
    children = tuple(value @ chart.encoder for value in parents)
    _decoder, normalized = _fit_normalized_operator_input_decoder(
        parent_raw_rows=parents,
        child_normalized_rows=children,
        parent_operator_norm_weight=torch.ones(3, dtype=torch.float64),
        target_operator_norm_weight=torch.ones(3, dtype=torch.float64),
        parent_norm_epsilon=1e-5,
        target_norm_epsilon=1e-5,
        residual_chart=chart.encoder,
        residual_chart_sha256=chart.report["report_sha256"],
        fit_trajectory_count=2,
        normalized_input_correction_rho=0.0,
        raw_structural_decoder=chart.decoder,
        raw_structural_decoder_sha256=exact_decoder_sha256,
        coordinate_completion_receipt=candidate,
        coordinate_completion_receipt_sha256=candidate["result_sha256"],
    )
    provenance = normalized["raw_structural_decoder_provenance"]
    assert provenance["raw_structural_decoder_sha256"] == exact_decoder_sha256
    assert (
        provenance["coordinate_completion_receipt_sha256"] == candidate["result_sha256"]
    )

    rectangular_parent = torch.randn(20, 4, dtype=torch.float64)
    rectangular_encoder = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    rectangular_evidence = Lfm2MovingGaugeBoundaryEvidence(
        boundary_id="short-conv-source",
        interface_role="intermediate",
        interface_locked=False,
        base_encoder=rectangular_encoder,
        fit_parent_rows=rectangular_parent,
        fit_child_rows=rectangular_parent @ rectangular_encoder,
        fit_weights=torch.ones(20, dtype=torch.float64),
        fit_provenance="short-conv-source:fit",
        row_alignment_sha256="b" * 64,
        cached_parent_evidence_sha256="c" * 64,
        live_child_recapture_sha256="d" * 64,
    )
    rectangular_chart = fit_lfm2_cycle_coupled_moving_gauge_boundary(
        rectangular_evidence
    )
    rectangular_candidate = stack._moving_gauge_coordinate_completion_candidate(
        rectangular_chart,
        foundation_candidate_receipt=foundation,
    )
    source = Lfm2SemanticShortConvFhvrSource(
        layer_index=1,
        cache_key="a" * 64,
        bridge_artifact_sha256="5" * 64,
        source_model_identity_sha256="6" * 64,
        parent_session_open_receipt_sha256="7" * 64,
        parent_identity_reconstruction_sha256="8" * 64,
        residual_chart_sha256="9" * 64,
        norm_epsilon=1e-5,
        source_operator_norm_weight=torch.ones(4),
        source_in_proj_weight=torch.randn(12, 4),
        source_in_proj_bias=None,
        source_fir_weight=torch.randn(4, 2),
        source_fir_bias=None,
        source_out_proj_weight=torch.randn(4, 4),
        source_out_proj_bias=None,
        residual_output_chart=rectangular_chart.encoder,
        raw_target_to_parent_decoder=rectangular_chart.decoder,
        coordinate_completion_candidate_receipt=rectangular_candidate,
    )
    assert torch.equal(source.raw_target_to_parent_decoder, rectangular_chart.decoder)
    assert (
        source.coordinate_completion_candidate_sha256
        == rectangular_candidate["result_sha256"]
    )


def test_noncommuting_norm_gains_require_input_producer_absorption() -> None:
    input_chart = _boundary("input", "input", locked=True)
    terminal_chart = _boundary("terminal", "terminal", locked=True)

    def linear(rows: int, columns: int) -> object:
        return SimpleNamespace(weight=torch.nn.Parameter(torch.randn(rows, columns)))

    layer = SimpleNamespace(
        is_attention_layer=True,
        self_attn=SimpleNamespace(
            q_proj=linear(3, 3),
            k_proj=linear(3, 3),
            v_proj=linear(3, 3),
        ),
        feed_forward=SimpleNamespace(w1=linear(5, 3), w3=linear(5, 3)),
        operator_norm=SimpleNamespace(
            weight=torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        ),
        ffn_norm=SimpleNamespace(
            weight=torch.nn.Parameter(torch.tensor([1.0, 1.5, 2.0]))
        ),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=(layer,)))
    old_rows = torch.randn(12, 3, dtype=torch.float64)
    new_rows = old_rows + 0.05
    evidence = []
    for parameter_id, role, parameter in (
        (
            "model.layers.0.self_attn.q_proj.weight",
            "attention_q",
            layer.self_attn.q_proj.weight,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "attention_k",
            layer.self_attn.k_proj.weight,
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            "attention_v",
            layer.self_attn.v_proj.weight,
        ),
        (
            "model.layers.0.feed_forward.w1.weight",
            "mlp_w1",
            layer.feed_forward.w1.weight,
        ),
        (
            "model.layers.0.feed_forward.w3.weight",
            "mlp_w3",
            layer.feed_forward.w3.weight,
        ),
    ):
        evidence.append(
            stack.Lfm2MovingGaugeInputProducerSiteEvidence(
                parameter_id=parameter_id,
                producer_group_id=(
                    "layer.000.mlp"
                    if role.startswith("mlp_")
                    else "layer.000.attention"
                ),
                producer_role=role,
                seed_weight=parameter.detach(),
                old_actual_normalized_input_rows=old_rows,
                new_actual_normalized_input_rows=new_rows,
                row_alignment_sha256="a" * 64,
                old_child_state_sha256="b" * 64,
                new_child_state_sha256="c" * 64,
            )
        )
    producer_state = stack.compile_lfm2_moving_gauge_input_producer_reclosure(evidence)
    foundation = stack._sealed(
        {
            "schema": "neural-forge-analytic-coordinate-completion-candidate-v1",
            "coordinate_map_sha256": "d" * 64,
        }
    )
    candidate = stack._moving_gauge_coordinate_completion_candidate(
        input_chart,
        foundation_candidate_receipt=foundation,
    )
    report = stack._moving_gauge_rmsnorm_transport_contract(
        model,
        boundary_charts=(input_chart, terminal_chart),
        input_producer_state=producer_state,
        coordinate_completion_candidates=(candidate,),
    )
    assert report["all_gains_exact_scalar"] is False
    assert report["noncommuting_gain_count"] == 2
    assert report["blanket_orthogonal_through_diagonal_gain_claim"] is False
    assert report["input_producer_reclosure_state_authenticated"] is True
    assert report["per_layer_exact_decoder_lineage_authenticated"] is True
    assert all(
        "adjacent_input_producer_fold" in value["transport"]
        for value in report["norm_gains"]
    )


def test_paired_physical_recaptures_build_every_input_producer_site() -> None:
    def linear(rows: int, columns: int) -> object:
        return SimpleNamespace(weight=torch.nn.Parameter(torch.randn(rows, columns)))

    layer = SimpleNamespace(
        is_attention_layer=True,
        self_attn=SimpleNamespace(
            q_proj=linear(3, 3),
            k_proj=linear(3, 3),
            v_proj=linear(3, 3),
        ),
        feed_forward=SimpleNamespace(w1=linear(5, 3), w3=linear(5, 3)),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=(layer,)))
    old_replay = SimpleNamespace(
        operator_input=torch.randn(1, 4, 3),
        mlp_input=torch.randn(1, 4, 3),
    )
    new_replay = SimpleNamespace(
        operator_input=old_replay.operator_input + 0.1,
        mlp_input=old_replay.mlp_input - 0.2,
    )
    old_bank = stack._LayerReplayBank(
        captured_child_model_state_sha256="a" * 64,
        layer_index=0,
        pre_residual_by_batch={0: torch.zeros(1, 4, 3)},
        replay_by_batch={0: old_replay},
        bank_sha256="b" * 64,
    )
    new_bank = stack._LayerReplayBank(
        captured_child_model_state_sha256="c" * 64,
        layer_index=0,
        pre_residual_by_batch={0: torch.zeros(1, 4, 3)},
        replay_by_batch={0: new_replay},
        bank_sha256="d" * 64,
    )
    old = stack.Lfm2LiveChildFullPrefixRecapture(
        active_layer_bank=old_bank,
        next_residual_by_batch={},
        receipt={"captured_child_model_state_sha256": "a" * 64},
        layer_banks=(old_bank,),
    )
    new = stack.Lfm2LiveChildFullPrefixRecapture(
        active_layer_bank=new_bank,
        next_residual_by_batch={},
        receipt={"captured_child_model_state_sha256": "c" * 64},
        layer_banks=(new_bank,),
    )
    alignment = SimpleNamespace(
        source_batch_index=0,
        phases=(SimpleNamespace(target_to_donor_boundary_positions=(0, 1, 2, 3)),),
    )
    evidence = stack._moving_gauge_input_producer_evidence(
        model,
        alignments=(alignment,),
        old_recapture=old,
        new_recapture=new,
        fit_batch_indices=(0,),
        row_alignment_sha256="e" * 64,
    )
    assert {value.producer_role for value in evidence} == {
        "attention_q",
        "attention_k",
        "attention_v",
        "mlp_w1",
        "mlp_w3",
    }
    state = stack.compile_lfm2_moving_gauge_input_producer_reclosure(evidence)
    assert state.verify_authentication()
    assert state.report["parameter_ids"] == sorted(
        value.parameter_id for value in evidence
    )


class _TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(3, 3, bias=False)
        self.k_proj = torch.nn.Linear(3, 3, bias=False)
        self.v_proj = torch.nn.Linear(3, 3, bias=False)
        self.out_proj = torch.nn.Linear(3, 3, bias=False)


class _TinyFeedForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(3, 5, bias=False)
        self.w3 = torch.nn.Linear(3, 5, bias=False)
        self.w2 = torch.nn.Linear(5, 3, bias=False)


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_attention_layer = True
        self.self_attn = _TinyAttention()
        self.feed_forward = _TinyFeedForward()
        self.operator_norm = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(3)))
        self.ffn_norm = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(3)))


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyLayer()])


def _finalizer_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[object, ...]:
    torch.manual_seed(33)
    model = _TinyModel()
    layer = model.model.layers[0]
    parent_rows = torch.randn(24, 3, dtype=torch.float64)
    rotation = torch.tensor(
        [[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    boundaries = tuple(
        Lfm2MovingGaugeBoundaryEvidence(
            boundary_id=name,
            interface_role=role,
            interface_locked=True,
            base_encoder=torch.eye(3, dtype=torch.float64),
            fit_parent_rows=parent_rows + offset,
            fit_child_rows=(parent_rows + offset) @ rotation,
            fit_weights=torch.ones(24, dtype=torch.float64),
            fit_provenance=f"{name}:fit",
            row_alignment_sha256="1" * 64,
            cached_parent_evidence_sha256=("2" if role == "input" else "3") * 64,
            live_child_recapture_sha256=("4" if role == "input" else "5") * 64,
        )
        for name, role, offset in (
            ("input", "input", 0.0),
            ("terminal", "terminal", 0.2),
        )
    )
    operator_rows = torch.randn(24, 3, dtype=torch.float64)
    mlp_rows = torch.randn(24, 5, dtype=torch.float64)
    writers = (
        Lfm2MovingGaugeWriterEvidence(
            parameter_id="model.layers.0.self_attn.out_proj.weight",
            transition_index=0,
            seed_weight=layer.self_attn.out_proj.weight.detach(),
            fit_input_rows=operator_rows,
            row_alignment_sha256="1" * 64,
            live_child_recapture_sha256="6" * 64,
            boundary_evidence_sha256=boundaries[0].report["report_sha256"],
        ),
        Lfm2MovingGaugeWriterEvidence(
            parameter_id="model.layers.0.feed_forward.w2.weight",
            transition_index=0,
            seed_weight=layer.feed_forward.w2.weight.detach(),
            fit_input_rows=mlp_rows,
            row_alignment_sha256="1" * 64,
            live_child_recapture_sha256="7" * 64,
            boundary_evidence_sha256=boundaries[0].report["report_sha256"],
            publication_kind="deferred_dense_prior",
        ),
    )
    writer_state = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        writers,
    )
    old_rows = torch.randn(18, 3, dtype=torch.float64)
    new_rows = old_rows @ rotation
    producer_evidence = []
    for parameter_id, role, parameter in (
        (
            "model.layers.0.self_attn.q_proj.weight",
            "attention_q",
            layer.self_attn.q_proj.weight,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "attention_k",
            layer.self_attn.k_proj.weight,
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            "attention_v",
            layer.self_attn.v_proj.weight,
        ),
        (
            "model.layers.0.feed_forward.w1.weight",
            "mlp_w1",
            layer.feed_forward.w1.weight,
        ),
        (
            "model.layers.0.feed_forward.w3.weight",
            "mlp_w3",
            layer.feed_forward.w3.weight,
        ),
    ):
        producer_evidence.append(
            stack.Lfm2MovingGaugeInputProducerSiteEvidence(
                parameter_id=parameter_id,
                producer_group_id=(
                    "layer.000.mlp"
                    if role.startswith("mlp_")
                    else "layer.000.attention"
                ),
                producer_role=role,
                seed_weight=parameter.detach(),
                old_actual_normalized_input_rows=old_rows,
                new_actual_normalized_input_rows=new_rows,
                row_alignment_sha256="1" * 64,
                old_child_state_sha256="8" * 64,
                new_child_state_sha256="9" * 64,
            )
        )
    producer_state = stack.compile_lfm2_moving_gauge_input_producer_reclosure(
        producer_evidence
    )
    final_id = "model.layers.0.feed_forward.w2.weight"
    corrected = writer_state.replacements_by_parameter_id[final_id]
    prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=corrected,
        body_recipe_id="nonlinear_qdeim_coupled_attention_fit_corrected",
        body_recipe_sha256="a" * 64,
        mlp_lane="nonlinear_qdeim_fit_corrected_priority",
        selected_role="full-w1-w3-w2",
        selected_candidate_sha256="b" * 64,
        preparation_sha256="c" * 64,
        mlp_installation_core_sha256="d" * 64,
        live_deferred_w2_sha256=stack.lfm2_cpcf_terminal_action_tensor_sha256(
            layer.feed_forward.w2.weight
        ),
    )
    causal = stack._sealed(
        {
            "schema": "test-causal-sweep",
            "steps": [
                {
                    "layer_index": 0,
                    "operator_parameter_ids": [
                        "model.layers.0.self_attn.q_proj.weight",
                        "model.layers.0.self_attn.k_proj.weight",
                        "model.layers.0.self_attn.v_proj.weight",
                    ],
                    "mlp_parameter_ids": [
                        "model.layers.0.feed_forward.w1.weight",
                        "model.layers.0.feed_forward.w3.weight",
                    ],
                }
            ],
        }
    )
    report_body = {
        "schema": "neural-forge-lfm2-moving-gauge-deferred-finalization-plan-v1",
        "dense_replacement_state_sha256": writer_state.report["report_sha256"],
        "input_producer_reclosure_state_sha256": producer_state.report["report_sha256"],
        "final_layer_mlp_down_weight_prior_authentication_sha256": (
            prior.authentication_sha256
        ),
        "causal_producer_sweep_sha256": causal["result_sha256"],
        "final_w2_parameter_id": final_id,
    }
    report = {
        **report_body,
        "report_sha256": stack.canonical_sha256(report_body),
    }
    plan = stack.Lfm2MovingGaugeDeferredFinalizationPlan(
        replacement_state=writer_state,
        input_producer_state=producer_state,
        corrected_final_w2_prior=prior,
        alignments=(SimpleNamespace(alignment_sha256="e" * 64),),
        boundary_evidence=boundaries,
        fit_batch_indices=(0,),
        causal_sweep_receipt=causal,
        report=report,
    )
    with torch.no_grad():
        layer.feed_forward.w2.weight.copy_(corrected)

    def state_sha(value: torch.nn.Module) -> str:
        return stack.canonical_sha256(
            {
                name: stack._tensor_sha256(parameter)
                for name, parameter in value.named_parameters()
            }
        )

    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda value: SimpleNamespace(identity_sha256=state_sha(value)),
    )
    recapture_count = {"value": 0}

    def recapture(value: torch.nn.Module, **_kwargs: object) -> object:
        recapture_count["value"] += 1
        state = state_sha(value)
        return SimpleNamespace(
            receipt={
                "recapture_sha256": stack.canonical_sha256(
                    {"count": recapture_count["value"], "state": state}
                ),
                "captured_child_model_state_sha256": state,
            }
        )

    monkeypatch.setattr(stack, "recapture_lfm2_live_child_full_prefix", recapture)
    monkeypatch.setattr(
        stack,
        "_moving_gauge_actual_recapture_diagnostics",
        lambda *_args, **_kwargs: stack._sealed(
            {
                "schema": "test-actual-final-recapture",
                "actual_physical_post_install_child_replayed": True,
            }
        ),
    )
    stack_receipt = stack._sealed(
        {
            "moving_gauge_deferred_finalization_plan_sha256": report["report_sha256"],
            "global_residual_closure_receipt_sha256": report["report_sha256"],
        }
    )
    w2_receipt = stack._sealed(
        {
            "semantic_layer_stack_result_sha256": stack_receipt["result_sha256"],
            "prior_authentication_sha256": prior.authentication_sha256,
            "prior_tensor_sha256": stack.lfm2_cpcf_terminal_action_tensor_sha256(
                corrected
            ),
            "live_w2_sha256_after": (
                stack.lfm2_cpcf_terminal_action_tensor_sha256(corrected)
            ),
            "model_state_after_sha256": state_sha(model),
        }
    )
    return model, plan, stack_receipt, w2_receipt, state_sha, recapture_count


def test_post_w2_finalizer_publishes_in_causal_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, plan, stack_receipt, w2_receipt, state_sha, recapture_count = (
        _finalizer_fixture(monkeypatch)
    )
    before = state_sha(model)
    final_w2_before = stack._tensor_sha256(model.model.layers[0].feed_forward.w2.weight)
    receipt = stack.finalize_lfm2_moving_gauge_after_deferred_w2(
        model,
        plan,
        layer_stack_receipt=stack_receipt,
        deferred_final_w2_installation_receipt=w2_receipt,
    )
    assert receipt["model_state_before_sha256"] == before
    assert receipt["model_state_after_sha256"] == state_sha(model)
    assert receipt["model_state_after_sha256"] != before
    assert receipt["causal_layer_step_count"] == 1
    assert receipt["stale_downstream_rows_used"] is False
    assert receipt["stale_same_layer_mlp_rows_used"] is False
    assert receipt["actual_final_recapture_includes_corrected_final_w2"] is True
    assert receipt["training_steps"] == 0
    assert recapture_count["value"] == 4
    assert (
        stack._tensor_sha256(model.model.layers[0].feed_forward.w2.weight)
        == final_w2_before
    )


def test_post_w2_finalizer_fault_rolls_back_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, plan, stack_receipt, w2_receipt, state_sha, _count = _finalizer_fixture(
        monkeypatch
    )
    stable_recapture = stack.recapture_lfm2_live_child_full_prefix
    calls = {"value": 0}

    def fail_during_sweep(*args: object, **kwargs: object) -> object:
        calls["value"] += 1
        if calls["value"] == 2:
            raise RuntimeError("injected causal replay failure")
        return stable_recapture(*args, **kwargs)

    monkeypatch.setattr(
        stack,
        "recapture_lfm2_live_child_full_prefix",
        fail_during_sweep,
    )
    before = state_sha(model)
    tensor_before = {
        name: stack._tensor_sha256(parameter)
        for name, parameter in model.named_parameters()
    }
    with pytest.raises(RuntimeError, match="injected causal replay failure"):
        stack.finalize_lfm2_moving_gauge_after_deferred_w2(
            model,
            plan,
            layer_stack_receipt=stack_receipt,
            deferred_final_w2_installation_receipt=w2_receipt,
        )
    assert state_sha(model) == before
    assert {
        name: stack._tensor_sha256(parameter)
        for name, parameter in model.named_parameters()
    } == tensor_before
