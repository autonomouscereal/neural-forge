from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


class _TinyShortConvLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_attention_layer = False
        self.conv = torch.nn.Module()
        self.conv.out_proj = torch.nn.Linear(
            2,
            2,
            bias=False,
            dtype=torch.float32,
        )
        with torch.no_grad():
            self.conv.out_proj.weight.copy_(torch.eye(2, dtype=torch.float32))


class _TinyShortConvTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyShortConvLayer()])


def _state_identity(model: torch.nn.Module) -> str:
    return canonical_sha256(
        {
            name: stack._tensor_sha256(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def _recipe_fields(recipe) -> dict[str, object]:
    return {
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "recipe_component_lane": recipe.component_lanes["short_conv"],
    }


def test_source_local_short_conv_writer_is_wired_as_one_causal_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _TinyShortConvTarget()
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_state_identity(model)),
    )
    recipe = stack.resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected"
    )
    initial_state = _state_identity(target)
    parameter_count = stack._parameter_count(target)
    structural_body = {
        "schema": stack._SHORT_CONV_INSTALLATION_SCHEMA,
        **_recipe_fields(recipe),
        "layer_index": 0,
        "status": "installed-analytic-candidate",
        "installed": True,
        "install_count": 1,
        "model_state_before_sha256": "1" * 64,
        "model_state_after_sha256": initial_state,
        "changed_state_keys": ["model.layers.0.conv.out_proj.weight"],
        "parameter_element_count": parameter_count,
        "parameter_count_changed": False,
        "transport_evidence_sha256": ["2" * 64],
        "fold_compute_device": "cpu",
        "analytic_cached_evidence_only": True,
        "operation_audit": dict(stack._ZERO_OPERATIONS),
    }
    structural_receipt = stack._sealed(structural_body)

    replay = SimpleNamespace(
        operator_input=torch.zeros((1, 2, 2), dtype=torch.float32),
        operator_output_input=torch.eye(2, dtype=torch.float32).reshape(1, 2, 2),
        replay_sha256="3" * 64,
    )
    structural_replay = stack._LayerReplayBank(
        captured_child_model_state_sha256=initial_state,
        layer_index=0,
        pre_residual_by_batch=MappingProxyType({}),
        replay_by_batch=MappingProxyType({0: replay}),
        bank_sha256="4" * 64,
    )
    final_phase = SimpleNamespace(
        target_to_donor_boundary_positions=(0, 1),
        donor_full_prefix_tokens=2,
        target_full_prefix_tokens=2,
    )
    alignment = SimpleNamespace(
        source_batch_index=0,
        alignment_sha256="5" * 64,
        phases=(final_phase,),
    )
    parent = {
        0: SimpleNamespace(
            evidence_sha256="6" * 64,
            final_histories={
                "residual.pre": torch.zeros((1, 2, 2), dtype=torch.float32),
                "residual.post_operator": (
                    2.0 * torch.eye(2, dtype=torch.float32).reshape(1, 2, 2)
                ),
            },
        )
    }
    fit_cases, fit_receipt = stack._build_short_conv_source_local_writer_fit_cases(
        alignments=(alignment,),
        parent=parent,
        structural_replay=structural_replay,
        fit_batch_indices=(0,),
        fit_batch_sample_weights={0: 1.0},
        layer_index=0,
    )
    prior = target.model.layers[0].conv.out_proj.weight.detach().clone()

    compiled, installation = stack._compile_install_short_conv_source_local_writer(
        target_model=target,
        layer_index=0,
        structural_operator_receipt=structural_receipt,
        structural_replay=structural_replay,
        fit_cases=fit_cases,
        fit_cases_receipt=fit_receipt,
        residual_output_chart=torch.eye(2, dtype=torch.float32),
        parameter_count=parameter_count,
        fold_compute_device="cpu",
        body_recipe=recipe,
    )

    assert installation["installed"] is True
    assert installation["changed_state_keys"] == ["model.layers.0.conv.out_proj.weight"]
    assert not torch.equal(
        prior,
        target.model.layers[0].conv.out_proj.weight.detach(),
    )
    assert compiled.report["target"]["correction_target"] == (
        "mapped_parent_source_local_short_conv_action"
    )
    assert compiled.report["prohibited_operations"]["training_steps"] == 0

    final_state = _state_identity(target)
    post_replay = stack._LayerReplayBank(
        captured_child_model_state_sha256=final_state,
        layer_index=0,
        pre_residual_by_batch=MappingProxyType({}),
        replay_by_batch=MappingProxyType({0: replay}),
        bank_sha256="7" * 64,
    )
    composite = stack._compose_source_local_short_conv_operator_receipt(
        structural_operator_receipt=structural_receipt,
        source_local_compiled=compiled,
        source_local_installation_receipt=installation,
        fit_cases_receipt=fit_receipt,
        structural_replay=structural_replay,
        post_source_local_replay=post_replay,
        layer_index=0,
        parameter_count=parameter_count,
        body_recipe=recipe,
    )

    assert composite["schema"] == stack._SOURCE_LOCAL_SHORT_CONV_COMPOSITE_SCHEMA
    assert composite["model_state_before_sha256"] == "1" * 64
    assert composite["model_state_intermediate_sha256"] == initial_state
    assert composite["model_state_after_sha256"] == final_state
    assert composite["source_local_short_conv_output"] is True
    assert composite["physical_parameter_transaction_count"] == 2
    assert composite["transport_evidence_sha256"] == [
        "2" * 64,
        fit_receipt["fit_evidence_sha256"],
    ]
    assert composite["operation_audit"] == dict(stack._ZERO_OPERATIONS)


def test_source_local_short_conv_fit_target_is_offset_invariant() -> None:
    from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
        compile_lfm2_semantic_short_conv_source_local_writer,
        lfm2_semantic_short_conv_source_local_writer_fit_evidence_sha256,
        lfm2_semantic_short_conv_source_local_writer_state_sha256,
    )

    recipe = stack.resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_source_local_residual_fit_corrected"
    )
    del recipe
    prior = torch.eye(2, dtype=torch.float32)
    chart = torch.eye(2, dtype=torch.float32)
    design = torch.eye(2, dtype=torch.float32)

    def compile_with_offset(offset: float):
        from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
            Lfm2SemanticShortConvSourceLocalWriterFitCase,
        )

        pre = torch.full((2, 2), offset, dtype=torch.float32)
        case = Lfm2SemanticShortConvSourceLocalWriterFitCase(
            case_id="fit",
            parent_pre_operator_residual=pre,
            parent_post_operator_residual=(
                pre + 2.0 * torch.eye(2, dtype=torch.float32)
            ),
            live_child_pre_operator_residual=torch.zeros_like(pre),
            live_child_writer_design=design,
            row_sample_weights=(0.5, 0.5),
            parent_trajectory_receipt_sha256="8" * 64,
            live_child_replay_receipt_sha256="9" * 64,
            aligned_row_identity_sha256="a" * 64,
        )
        cases = (case,)
        return compile_lfm2_semantic_short_conv_source_local_writer(
            structural_out_weight=prior,
            expected_structural_writer_state_sha256=(
                lfm2_semantic_short_conv_source_local_writer_state_sha256(prior)
            ),
            residual_output_chart=chart,
            expected_residual_output_chart_sha256=stack._tensor_sha256(chart),
            fit_cases=cases,
            expected_fit_evidence_sha256=(
                lfm2_semantic_short_conv_source_local_writer_fit_evidence_sha256(cases)
            ),
            fold_compute_device="cpu",
        )

    baseline = compile_with_offset(0.0)
    shifted = compile_with_offset(1_000_000.0)

    torch.testing.assert_close(
        baseline.out_weight_source_local_corrected,
        shifted.out_weight_source_local_corrected,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        baseline.report["target"]["common_parent_pre_post_offset_changes_candidate"]
        is False
    )
