from __future__ import annotations

from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionDownWeightPrior,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_global_residual_closure import (  # noqa: E402
    fingerprint_lfm2_global_residual_tensor,
)
from neural_forge.hf_lfm2_postfold_terminal_w2_reclosure import (  # noqa: E402
    LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA,
    LFM2_POST_RECLOSURE_TERMINAL_W2_INSTALL_SCHEMA,
    Lfm2PostfoldTerminalW2ReclosureError,
    install_lfm2_post_reclosure_terminal_w2,
    rebase_lfm2_installed_terminal_w2_for_output_reclosure,
)
from neural_forge.hf_lfm2_semantic_global_residual_integration import (  # noqa: E402
    LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_mlp_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _sealed(body: dict[str, object], field: str = "result_sha256") -> MappingProxyType:
    return MappingProxyType({**body, field: canonical_sha256(body)})


class _Config:
    model_type = "lfm2"

    def to_dict(self) -> dict[str, object]:
        return {"model_type": "lfm2", "hidden_size": 2, "num_hidden_layers": 1}


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.out_proj.weight.copy_(torch.eye(2, dtype=torch.float32))


class _FeedForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w2 = torch.nn.Linear(3, 2, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.w2.weight.zero_()


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.feed_forward = _FeedForward()


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _Backbone()

    def forward(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("Terminal W2 handoff must never execute the model.")


def _source_receipts(model: _Model):
    final = model.model.layers[0].feed_forward.w2.weight
    original = final.detach().clone()
    original_sha = lfm2_cpcf_terminal_action_tensor_sha256(original)
    selected = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
    )
    prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=selected,
        body_recipe_id="nonlinear_qdeim_structural",
        body_recipe_sha256=_digest("recipe"),
        mlp_lane="nonlinear_qdeim_structural_priority",
        selected_role="full-w1-w3-w2",
        selected_candidate_sha256=_digest("candidate"),
        preparation_sha256=_digest("preparation"),
        mlp_installation_core_sha256=_digest("installation"),
        live_deferred_w2_sha256=original_sha,
    )
    frontier = _sealed(
        {
            "schema": LFM2_SEMANTIC_MLP_INSTALLATION_SCHEMA,
            "body_recipe_id": prior.body_recipe_id,
            "body_recipe_sha256": prior.body_recipe_sha256,
            "recipe_component_lane": prior.mlp_lane,
            "selected_role": prior.selected_role,
            "selected_candidate_sha256": prior.selected_candidate_sha256,
            "preparation_sha256": prior.preparation_sha256,
            "deferred_selected_w2_prior_mlp_installation_core_sha256": (
                prior.mlp_installation_core_sha256
            ),
            "deferred_selected_w2_prior_authentication_sha256": (
                prior.authentication_sha256
            ),
            "deferred_selected_w2_prior_candidate_sha256": (
                prior.selected_candidate_sha256
            ),
            "deferred_selected_w2_prior_terminal_action_sha256": (
                lfm2_cpcf_terminal_action_tensor_sha256(prior.down_weight)
            ),
            "deferred_selected_w2_prior_selected_role": prior.selected_role,
            "deferred_selected_w2_prior_installed_before_terminal": False,
        }
    )
    base_state = fingerprint_lfm2_parent(model).identity_sha256
    stack = _sealed(
        {
            "schema": LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "terminal_layer_index": 0,
            "layer_installations": [
                {
                    "layer_index": 0,
                    "mlp_complete_candidate_frontier_receipt": dict(frontier),
                }
            ],
            "model_state_after_sha256": base_state,
            "deferred_final_w2_sha256": original_sha,
            "final_layer_mlp_w2_deferred_to_fused_terminal": True,
            "final_layer_mlp_w2_install_count": 0,
            "final_layer_mlp_down_weight_prior_authentication_sha256": (
                prior.authentication_sha256
            ),
            "final_layer_mlp_down_weight_prior_tensor_sha256": (
                lfm2_cpcf_terminal_action_tensor_sha256(prior.down_weight)
            ),
            "final_layer_mlp_down_weight_prior_live_w2_sha256": original_sha,
            "final_layer_mlp_down_weight_prior_installed_before_terminal": False,
            "final_layer_mlp_down_weight_prior_install_count": 0,
        }
    )
    with torch.no_grad():
        final.copy_(selected)
    installed_state = fingerprint_lfm2_parent(model).identity_sha256
    installation = _sealed(
        {
            "schema": LFM2_DEFERRED_FINAL_W2_INSTALLATION_SCHEMA,
            "semantic_layer_stack_result_sha256": stack["result_sha256"],
            "prior_authentication_sha256": prior.authentication_sha256,
            "prior_tensor_sha256": lfm2_cpcf_terminal_action_tensor_sha256(
                selected
            ),
            "live_w2_sha256_before": original_sha,
            "live_w2_sha256_after": lfm2_cpcf_terminal_action_tensor_sha256(
                selected
            ),
            "model_state_before_sha256": base_state,
            "model_state_after_sha256": installed_state,
            "changed_state_keys": ["model.layers.0.feed_forward.w2.weight"],
            "final_w2_parameter_name": "model.layers.0.feed_forward.w2.weight",
            "effectful_installation": True,
        }
    )
    return original, selected, prior, stack, installation


def _reclosure_report(
    model: _Model,
    corrected: Lfm2CpcfTerminalActionDownWeightPrior,
) -> MappingProxyType:
    writer_name = "model.layers.0.self_attn.out_proj.weight"
    writer = model.model.layers[0].self_attn.out_proj.weight
    final = model.model.layers[0].feed_forward.w2.weight
    final_sha = lfm2_cpcf_terminal_action_tensor_sha256(final)
    corrected_sha = lfm2_cpcf_terminal_action_tensor_sha256(
        corrected.down_weight
    )
    return _sealed(
        {
            "schema": (
                LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA
            ),
            "source_local_boundary_mode": True,
            "absolute_endpoint_boundary_mode": False,
            "final_w2_unchanged": True,
            "final_w2_installation_count": 0,
            "final_w2_live_sha256_after": final_sha,
            "corrected_final_w2_prior_sha256": corrected.authentication_sha256,
            "final_layer_mlp_down_weight_prior_authentication_sha256": (
                corrected.authentication_sha256
            ),
            "final_layer_mlp_down_weight_prior_tensor_sha256": corrected_sha,
            "final_layer_mlp_down_weight_prior_live_w2_sha256": final_sha,
            "installed_writer_site_ids": [writer_name],
            "dense_live_writer_sha256_after": {
                writer_name: fingerprint_lfm2_global_residual_tensor(writer)
            },
        },
        "report_sha256",
    )


def test_rebase_and_post_reclosure_install_preserve_every_unrelated_tensor() -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    original, selected, prior, stack, installation = _source_receipts(model)
    out_before = model.model.layers[0].self_attn.out_proj.weight.detach().clone()

    rebase = rebase_lfm2_installed_terminal_w2_for_output_reclosure(
        model,
        semantic_layer_stack_receipt=stack,
        deferred_final_w2_installation_receipt=installation,
        original_deferred_w2=original,
    )

    assert rebase.verify()
    assert rebase.terminal_prior.authentication_sha256 == prior.authentication_sha256
    assert torch.equal(rebase.terminal_prior.down_weight, selected)
    assert torch.equal(model.model.layers[0].feed_forward.w2.weight, original)
    assert torch.equal(model.model.layers[0].self_attn.out_proj.weight, out_before)
    assert all(
        torch.equal(parameter.grad, torch.full_like(parameter, 0.25))
        for parameter in model.parameters()
    )

    corrected = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=rebase.terminal_prior.down_weight.add(0.5),
        body_recipe_id=rebase.terminal_prior.body_recipe_id,
        body_recipe_sha256=rebase.terminal_prior.body_recipe_sha256,
        mlp_lane=rebase.terminal_prior.mlp_lane,
        selected_role=rebase.terminal_prior.selected_role,
        selected_candidate_sha256=rebase.terminal_prior.selected_candidate_sha256,
        preparation_sha256=rebase.terminal_prior.preparation_sha256,
        mlp_installation_core_sha256=(
            rebase.terminal_prior.mlp_installation_core_sha256
        ),
        live_deferred_w2_sha256=(rebase.terminal_prior.live_deferred_w2_sha256),
    )
    report = _reclosure_report(model, corrected)
    result = install_lfm2_post_reclosure_terminal_w2(
        model,
        rebase=rebase,
        corrected_terminal_prior=corrected,
        source_local_global_reclosure_report=report,
    )

    assert result["schema"] == LFM2_POST_RECLOSURE_TERMINAL_W2_INSTALL_SCHEMA
    assert result["changed_state_keys"] == [
        "model.layers.0.feed_forward.w2.weight"
    ]
    assert result["effectful_installation"] is True
    assert result["installation_count"] == 1
    assert torch.equal(
        model.model.layers[0].feed_forward.w2.weight,
        corrected.down_weight,
    )
    assert torch.equal(model.model.layers[0].self_attn.out_proj.weight, out_before)
    assert all(
        torch.equal(parameter.grad, torch.full_like(parameter, 0.25))
        for parameter in model.parameters()
    )


def test_rebase_rejects_wrong_original_seed_without_mutation() -> None:
    model = _Model()
    _original, selected, _prior, stack, installation = _source_receipts(model)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(
        Lfm2PostfoldTerminalW2ReclosureError,
        match="does not authenticate",
    ):
        rebase_lfm2_installed_terminal_w2_for_output_reclosure(
            model,
            semantic_layer_stack_receipt=stack,
            deferred_final_w2_installation_receipt=installation,
            original_deferred_w2=torch.ones_like(selected),
        )

    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_final_install_rejects_stale_nonfinal_writer() -> None:
    model = _Model()
    original, _selected, _prior, stack, installation = _source_receipts(model)
    rebase = rebase_lfm2_installed_terminal_w2_for_output_reclosure(
        model,
        semantic_layer_stack_receipt=stack,
        deferred_final_w2_installation_receipt=installation,
        original_deferred_w2=original,
    )
    corrected = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=rebase.terminal_prior.down_weight.add(0.5),
        body_recipe_id=rebase.terminal_prior.body_recipe_id,
        body_recipe_sha256=rebase.terminal_prior.body_recipe_sha256,
        mlp_lane=rebase.terminal_prior.mlp_lane,
        selected_role=rebase.terminal_prior.selected_role,
        selected_candidate_sha256=rebase.terminal_prior.selected_candidate_sha256,
        preparation_sha256=rebase.terminal_prior.preparation_sha256,
        mlp_installation_core_sha256=(
            rebase.terminal_prior.mlp_installation_core_sha256
        ),
        live_deferred_w2_sha256=(rebase.terminal_prior.live_deferred_w2_sha256),
    )
    report = _reclosure_report(model, corrected)
    with torch.no_grad():
        model.model.layers[0].self_attn.out_proj.weight.add_(1.0)
    final_before = model.model.layers[0].feed_forward.w2.weight.detach().clone()

    with pytest.raises(
        Lfm2PostfoldTerminalW2ReclosureError,
        match="nonfinal writers differ",
    ):
        install_lfm2_post_reclosure_terminal_w2(
            model,
            rebase=rebase,
            corrected_terminal_prior=corrected,
            source_local_global_reclosure_report=report,
        )

    assert torch.equal(model.model.layers[0].feed_forward.w2.weight, final_before)
