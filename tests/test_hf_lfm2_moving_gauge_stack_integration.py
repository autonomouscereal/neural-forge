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
    def __init__(self, layer_count: int = 1) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_TinyLayer() for _ in range(layer_count)]
        )


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
    causal_writer_step = stack._sealed(
        {
            "schema": "test-causal-fresh-primal-w2-step",
            "layer_index": 0,
            "parameter_id": final_id,
            "replacement_weight_sha256": (
                stack.fingerprint_lfm2_moving_gauge_tensor(corrected)
            ),
        }
    )
    causal_writer = stack.Lfm2MovingGaugeCausalWriterReclosureState(
        parameter_ids=(final_id,),
        replacement_weights=(corrected,),
        step_receipts=(causal_writer_step,),
    )
    report_body = {
        "schema": "neural-forge-lfm2-moving-gauge-deferred-finalization-plan-v1",
        "dense_replacement_state_sha256": writer_state.report["report_sha256"],
        "input_producer_reclosure_state_sha256": producer_state.report["report_sha256"],
        "causal_writer_reclosure_state_sha256": causal_writer.receipt[
            "result_sha256"
        ],
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
        causal_writer_reclosure_state=causal_writer,
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
    assert receipt["terminal_w2_replacement_authenticated_not_rewritten"] is True
    assert receipt["terminal_w2_exactly_once_boundary_preserved"] is True
    assert receipt["causal_fresh_primal_writer_steps"][0][
        "published_by_moving_finalizer"
    ] is False
    assert receipt["training_steps"] == 0
    assert recapture_count["value"] == 3
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


def test_fresh_primal_second_w2_pass_is_reproduced_without_rewriting_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(117)
    model = _TinyModel(layer_count=2)
    layers = tuple(model.model.layers)
    row_count = 9
    scale = row_count**0.5
    base_residual = torch.randn(1, row_count, 3)
    alignment = SimpleNamespace(
        source_batch_index=0,
        alignment_sha256="f" * 64,
        phases=(
            SimpleNamespace(
                target_to_donor_boundary_positions=tuple(range(row_count))
            ),
        ),
    )

    def state_sha(value: torch.nn.Module) -> str:
        return stack.canonical_sha256(
            {
                name: stack._tensor_sha256(parameter)
                for name, parameter in value.named_parameters()
            }
        )

    recapture_count = {"value": 0}

    def recapture(
        value: torch.nn.Module,
        *,
        active_layer_index: int,
        **_kwargs: object,
    ) -> stack.Lfm2LiveChildFullPrefixRecapture:
        recapture_count["value"] += 1
        residual = base_residual.clone()
        banks = []
        for layer_index, layer in enumerate(value.model.layers):
            pre = residual
            mlp_input = pre
            gate = torch.nn.functional.linear(mlp_input, layer.feed_forward.w1.weight)
            up = torch.nn.functional.linear(mlp_input, layer.feed_forward.w3.weight)
            writer_input = torch.nn.functional.silu(gate) * up
            delta = torch.nn.functional.linear(
                writer_input,
                layer.feed_forward.w2.weight,
            )
            residual = pre + delta
            replay = SimpleNamespace(
                mlp_input=mlp_input.detach().clone(),
                post_operator_residual=pre.detach().clone(),
                post_layer_residual=residual.detach().clone(),
            )
            banks.append(
                stack._LayerReplayBank(
                    captured_child_model_state_sha256=state_sha(value),
                    layer_index=layer_index,
                    pre_residual_by_batch={0: pre.detach().clone()},
                    replay_by_batch={0: replay},
                    bank_sha256=stack.canonical_sha256(
                        {
                            "layer_index": layer_index,
                            "count": recapture_count["value"],
                            "state": state_sha(value),
                        }
                    ),
                )
            )
        child_state = state_sha(value)
        recapture_sha = stack.canonical_sha256(
            {
                "count": recapture_count["value"],
                "state": child_state,
                "active_layer_index": active_layer_index,
            }
        )
        return stack.Lfm2LiveChildFullPrefixRecapture(
            active_layer_bank=banks[active_layer_index],
            next_residual_by_batch={0: residual.detach().clone()},
            receipt={
                "recapture_sha256": recapture_sha,
                "captured_child_model_state_sha256": child_state,
            },
            layer_banks=tuple(banks),
        )

    initial_recapture = recapture(model, active_layer_index=1)
    desired_rows = (
        base_residual[0].to(torch.float64) / scale,
        initial_recapture.layer_banks[0]
        .replay_by_batch[0]
        .post_layer_residual[0]
        .to(torch.float64)
        / scale
        + 0.75,
        initial_recapture.layer_banks[1]
        .replay_by_batch[0]
        .post_layer_residual[0]
        .to(torch.float64)
        / scale
        - 0.55,
    )
    current_rows = (
        base_residual[0].to(torch.float64) / scale,
        initial_recapture.layer_banks[0]
        .replay_by_batch[0]
        .post_layer_residual[0]
        .to(torch.float64)
        / scale,
        initial_recapture.layer_banks[1]
        .replay_by_batch[0]
        .post_layer_residual[0]
        .to(torch.float64)
        / scale,
    )
    boundaries = tuple(
        Lfm2MovingGaugeBoundaryEvidence(
            boundary_id=name,
            interface_role=role,
            interface_locked=True,
            base_encoder=torch.eye(3, dtype=torch.float64),
            fit_parent_rows=desired,
            fit_child_rows=current_rows[index],
            fit_weights=torch.ones(row_count, dtype=torch.float64),
            fit_provenance=f"{name}:fit",
            row_alignment_sha256="1" * 64,
            cached_parent_evidence_sha256=f"{index + 2:x}" * 64,
            live_child_recapture_sha256=f"{index + 5:x}" * 64,
        )
        for index, (name, role, desired) in enumerate(
            (
                ("input", "input", desired_rows[0]),
                ("layer.000.post", "intermediate", desired_rows[1]),
                ("terminal", "terminal", desired_rows[2]),
            )
        )
    )
    writers = []
    for layer_index, layer in enumerate(layers):
        writers.extend(
            (
                Lfm2MovingGaugeWriterEvidence(
                    parameter_id=(
                        f"model.layers.{layer_index}.self_attn.out_proj.weight"
                    ),
                    transition_index=layer_index,
                    seed_weight=layer.self_attn.out_proj.weight.detach(),
                    fit_input_rows=torch.randn(row_count, 3, dtype=torch.float64),
                    row_alignment_sha256="1" * 64,
                    live_child_recapture_sha256=f"{layer_index + 8:x}" * 64,
                    boundary_evidence_sha256=boundaries[layer_index].report[
                        "report_sha256"
                    ],
                ),
                Lfm2MovingGaugeWriterEvidence(
                    parameter_id=(
                        f"model.layers.{layer_index}.feed_forward.w2.weight"
                    ),
                    transition_index=layer_index,
                    seed_weight=layer.feed_forward.w2.weight.detach(),
                    fit_input_rows=torch.randn(row_count, 5, dtype=torch.float64),
                    row_alignment_sha256="1" * 64,
                    live_child_recapture_sha256=f"{layer_index + 10:x}" * 64,
                    boundary_evidence_sha256=boundaries[layer_index].report[
                        "report_sha256"
                    ],
                    publication_kind=(
                        "deferred_dense_prior"
                        if layer_index == len(layers) - 1
                        else "live_physical_parameter"
                    ),
                ),
            )
        )
    writer_state = compile_lfm2_moving_gauge_residual_reclosure(
        boundaries,
        tuple(writers),
    )
    producer_evidence = []
    for layer_index, layer in enumerate(layers):
        old_rows = torch.randn(12, 3, dtype=torch.float64)
        new_rows = old_rows * (1.6 + 0.1 * layer_index) + 0.35
        for parameter_id, role, parameter in (
            (
                f"model.layers.{layer_index}.self_attn.q_proj.weight",
                "attention_q",
                layer.self_attn.q_proj.weight,
            ),
            (
                f"model.layers.{layer_index}.self_attn.k_proj.weight",
                "attention_k",
                layer.self_attn.k_proj.weight,
            ),
            (
                f"model.layers.{layer_index}.self_attn.v_proj.weight",
                "attention_v",
                layer.self_attn.v_proj.weight,
            ),
            (
                f"model.layers.{layer_index}.feed_forward.w1.weight",
                "mlp_w1",
                layer.feed_forward.w1.weight,
            ),
            (
                f"model.layers.{layer_index}.feed_forward.w3.weight",
                "mlp_w3",
                layer.feed_forward.w3.weight,
            ),
        ):
            producer_evidence.append(
                stack.Lfm2MovingGaugeInputProducerSiteEvidence(
                    parameter_id=parameter_id,
                    producer_group_id=(
                        f"layer.{layer_index:03d}.mlp"
                        if role.startswith("mlp_")
                        else f"layer.{layer_index:03d}.attention"
                    ),
                    producer_role=role,
                    seed_weight=parameter.detach(),
                    old_actual_normalized_input_rows=old_rows,
                    new_actual_normalized_input_rows=new_rows,
                    row_alignment_sha256="1" * 64,
                    old_child_state_sha256="b" * 64,
                    new_child_state_sha256="c" * 64,
                )
            )
    producer_state = stack.compile_lfm2_moving_gauge_input_producer_reclosure(
        producer_evidence
    )
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    writer_replacements = writer_state.replacements_by_parameter_id
    with torch.no_grad():
        for name, parameter in stack._moving_gauge_live_writer_parameters(
            model
        ).items():
            parameter.copy_(writer_replacements[name])
    pre_producer = recapture(model, active_layer_index=1)
    pre_primal, _ = stack._moving_gauge_current_w2_writer_rows(
        model,
        pre_producer,
        alignments=(alignment,),
        fit_batch_indices=(0,),
        layer_index=0,
    )
    with torch.no_grad():
        for name, parameter in stack._moving_gauge_input_producer_parameters(
            model
        ).items():
            parameter.copy_(producer_state.replacements_by_parameter_id[name])
    post_producer = recapture(model, active_layer_index=1)
    post_primal, _ = stack._moving_gauge_current_w2_writer_rows(
        model,
        post_producer,
        alignments=(alignment,),
        fit_batch_indices=(0,),
        layer_index=0,
    )
    assert not torch.equal(pre_primal, post_primal)
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda value: SimpleNamespace(identity_sha256=state_sha(value)),
    )
    monkeypatch.setattr(stack, "recapture_lfm2_live_child_full_prefix", recapture)
    causal_writer_state, preview_final_recapture = (
        stack._compile_install_moving_gauge_causal_writer_reclosure(
            model,
            current_recapture=post_producer,
            alignments=(alignment,),
            boundary_evidence=boundaries,
            boundary_charts=writer_state.boundaries,
            fit_batch_indices=(0,),
            initial_final_w2_seed=writer_replacements[
                "model.layers.1.feed_forward.w2.weight"
            ],
            policy=stack.Lfm2MovingGaugeReclosurePolicy(),
        )
    )
    nonterminal_id = "model.layers.0.feed_forward.w2.weight"
    final_id = "model.layers.1.feed_forward.w2.weight"
    causal_replacements = causal_writer_state.replacements_by_parameter_id
    assert stack._tensor_sha256(causal_replacements[nonterminal_id]) != (
        stack._tensor_sha256(writer_replacements[nonterminal_id])
    )
    assert causal_writer_state.step_receipts[1]["pre_recapture_sha256"] == (
        causal_writer_state.step_receipts[0]["post_recapture_sha256"]
    )
    assert preview_final_recapture.receipt["recapture_sha256"] == (
        causal_writer_state.step_receipts[-1]["post_recapture_sha256"]
    )
    prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=causal_replacements[final_id],
        body_recipe_id="nonlinear_qdeim_coupled_attention_fit_corrected",
        body_recipe_sha256="a" * 64,
        mlp_lane="nonlinear_qdeim_fit_corrected_priority",
        selected_role="full-w1-w3-w2",
        selected_candidate_sha256="b" * 64,
        preparation_sha256="c" * 64,
        mlp_installation_core_sha256="d" * 64,
        live_deferred_w2_sha256=stack.lfm2_cpcf_terminal_action_tensor_sha256(
            initial_parameters[final_id]
        ),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(initial_parameters[name])
        layers[-1].feed_forward.w2.weight.copy_(prior.down_weight)

    producer_steps = []
    for layer_index in range(2):
        producer_steps.append(
            {
                "layer_index": layer_index,
                "operator_parameter_ids": [
                    f"model.layers.{layer_index}.self_attn.{suffix}.weight"
                    for suffix in ("q_proj", "k_proj", "v_proj")
                ],
                "mlp_parameter_ids": [
                    f"model.layers.{layer_index}.feed_forward.{suffix}.weight"
                    for suffix in ("w1", "w3")
                ],
            }
        )
    causal_producer = stack._sealed(
        {"schema": "test-two-layer-causal-producer", "steps": producer_steps}
    )
    report_body = {
        "schema": "neural-forge-lfm2-moving-gauge-deferred-finalization-plan-v1",
        "dense_replacement_state_sha256": writer_state.report["report_sha256"],
        "input_producer_reclosure_state_sha256": producer_state.report[
            "report_sha256"
        ],
        "causal_writer_reclosure_state_sha256": causal_writer_state.receipt[
            "result_sha256"
        ],
        "final_layer_mlp_down_weight_prior_authentication_sha256": (
            prior.authentication_sha256
        ),
        "causal_producer_sweep_sha256": causal_producer["result_sha256"],
        "final_w2_parameter_id": final_id,
    }
    plan_report = {
        **report_body,
        "report_sha256": stack.canonical_sha256(report_body),
    }
    plan = stack.Lfm2MovingGaugeDeferredFinalizationPlan(
        replacement_state=writer_state,
        input_producer_state=producer_state,
        causal_writer_reclosure_state=causal_writer_state,
        corrected_final_w2_prior=prior,
        alignments=(alignment,),
        boundary_evidence=boundaries,
        fit_batch_indices=(0,),
        causal_sweep_receipt=causal_producer,
        report=plan_report,
    )
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda value: SimpleNamespace(identity_sha256=state_sha(value)),
    )
    monkeypatch.setattr(stack, "recapture_lfm2_live_child_full_prefix", recapture)
    stack_receipt = stack._sealed(
        {
            "moving_gauge_deferred_finalization_plan_sha256": plan_report[
                "report_sha256"
            ],
            "global_residual_closure_receipt_sha256": plan_report["report_sha256"],
        }
    )
    final_sha = stack.lfm2_cpcf_terminal_action_tensor_sha256(prior.down_weight)
    w2_receipt = stack._sealed(
        {
            "semantic_layer_stack_result_sha256": stack_receipt["result_sha256"],
            "prior_authentication_sha256": prior.authentication_sha256,
            "prior_tensor_sha256": final_sha,
            "live_w2_sha256_after": final_sha,
            "model_state_after_sha256": state_sha(model),
        }
    )
    terminal_version_before = layers[-1].feed_forward.w2.weight._version
    receipt = stack.finalize_lfm2_moving_gauge_after_deferred_w2(
        model,
        plan,
        layer_stack_receipt=stack_receipt,
        deferred_final_w2_installation_receipt=w2_receipt,
    )
    assert stack._tensor_sha256(layers[0].feed_forward.w2.weight) == (
        stack._tensor_sha256(causal_replacements[nonterminal_id])
    )
    assert stack.lfm2_cpcf_terminal_action_tensor_sha256(
        layers[-1].feed_forward.w2.weight
    ) == final_sha
    assert layers[-1].feed_forward.w2.weight._version == terminal_version_before
    assert receipt["causal_fresh_primal_writer_steps"][0][
        "published_by_moving_finalizer"
    ] is True
    assert receipt["causal_fresh_primal_writer_steps"][-1][
        "published_by_moving_finalizer"
    ] is False
    assert receipt["final_full_stack_recapture_sha256"] == receipt[
        "causal_fresh_primal_writer_steps"
    ][0]["post_recapture_sha256"]
    assert receipt["chart_transition_used_as_physical_jacobian"] is False
    assert receipt["training_steps"] == 0


def test_terminal_boundary_uses_the_same_deferred_prior_as_its_writer_seed() -> None:
    torch.manual_seed(91)
    hidden = 3
    intermediate = 5
    prior_weight = torch.randn(hidden, intermediate, dtype=torch.float32)
    prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=prior_weight,
        body_recipe_id="nonlinear_qdeim_coupled_attention_fit_corrected",
        body_recipe_sha256="a" * 64,
        mlp_lane="nonlinear_qdeim_fit_corrected_priority",
        selected_role="full-w1-w3-w2",
        selected_candidate_sha256="b" * 64,
        preparation_sha256="c" * 64,
        mlp_installation_core_sha256="d" * 64,
        live_deferred_w2_sha256="e" * 64,
    )
    alignments = []
    parent = {}
    replay_by_batch = {}
    accumulators = {}
    live_rows = {}
    prior_rows = {}
    for batch in range(4):
        alignment_sha = f"{batch + 1:x}" * 64
        alignments.append(
            SimpleNamespace(
                source_batch_index=batch,
                alignment_sha256=alignment_sha,
                phases=(
                    SimpleNamespace(target_to_donor_boundary_positions=(0,)),
                ),
            )
        )
        parent_row = torch.tensor(
            [[[1.0 + batch, 2.0 + batch, 3.0 + batch]]],
            dtype=torch.float64,
        )
        live_row = torch.full(
            (1, 1, hidden),
            -100.0 - batch,
            dtype=torch.float64,
        )
        prior_row = torch.full(
            (1, 1, hidden),
            10.0 + batch,
            dtype=torch.float64,
        )
        parent[batch] = SimpleNamespace(
            evidence_sha256=f"{batch + 5:x}" * 64,
            final_histories={"residual.post_layer": parent_row},
        )
        replay_by_batch[batch] = SimpleNamespace(post_layer_residual=live_row)
        live_rows[batch] = live_row
        prior_rows[batch] = prior_row
        macrocell = SimpleNamespace(
            authentication_sha256=f"{batch + 9:x}" * 64,
            verify_authentication=lambda: True,
            mlp=SimpleNamespace(
                parameters=SimpleNamespace(
                    down_weight=prior_weight.to(dtype=torch.float64)
                ),
                output=prior_row,
            ),
        )
        accumulators[batch] = stack._GlobalResidualTrajectoryAccumulator(
            batch_index=batch,
            alignment_sha256=alignment_sha,
            selected_target_positions=torch.tensor([0], dtype=torch.int64),
            source_donor_positions=torch.tensor([0], dtype=torch.int64),
            compact_donor_positions=torch.tensor([0], dtype=torch.int64),
            canonical_boundary_row_indices=torch.tensor([batch], dtype=torch.int64),
            macrocells=[macrocell],
        )
    upstream_evidence = Lfm2MovingGaugeBoundaryEvidence(
        boundary_id="stack.input",
        interface_role="input",
        interface_locked=True,
        base_encoder=torch.eye(hidden, dtype=torch.float64),
        fit_parent_rows=torch.ones(1, hidden, dtype=torch.float64),
        fit_child_rows=torch.ones(1, hidden, dtype=torch.float64),
        fit_weights=torch.ones(1, dtype=torch.float64),
        fit_provenance="upstream-fit",
        row_alignment_sha256="f" * 64,
        cached_parent_evidence_sha256="a" * 64,
        live_child_recapture_sha256="b" * 64,
    )
    upstream_chart = fit_lfm2_cycle_coupled_moving_gauge_boundary(
        upstream_evidence
    )
    chart_receipt = stack._sealed(
        {
            "schema": "test-terminal-partitions",
            "partition_batch_indices": {
                "fit": [0],
                "fit_check": [1],
                "selection": [2],
                "validation": [3],
            },
        }
    )
    evidence, _chart = stack._moving_gauge_terminal_boundary(
        alignments=tuple(alignments),
        parent=parent,
        propagation_replay=SimpleNamespace(
            replay_by_batch=replay_by_batch,
            bank_sha256="c" * 64,
        ),
        accumulators=accumulators,
        deferred_final_w2_prior=prior,
        producer_encoder=torch.eye(hidden, dtype=torch.float64),
        chart_receipt=chart_receipt,
        row_alignment_sha256="f" * 64,
        upstream_chart=upstream_chart,
        upstream_evidence=upstream_evidence,
    )
    torch.testing.assert_close(
        evidence.fit_child_rows,
        prior_rows[0][0],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(evidence.fit_child_rows, live_rows[0][0])
    assert evidence.child_row_evidence_kind == (
        "deferred_prior_analytic_macrocell"
    )
    assert evidence.report["live_child_rows_already_recaptured"] is False
    assert evidence.report["deferred_prior_analytic_macrocell_rows"] is True
