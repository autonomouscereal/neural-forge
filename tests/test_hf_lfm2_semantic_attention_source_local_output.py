from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_semantic_attention_source_local_output import (  # noqa: E402
    LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA,
    Lfm2SemanticAttentionSourceLocalOutputError,
    compile_install_lfm2_semantic_attention_source_local_output,
)
from tests.test_hf_lfm2_attention_group_qdeim import (  # noqa: E402
    _source_local_output_fit_case,
)
from tests.test_hf_lfm2_semantic_attention_group_qdeim_installer import (  # noqa: E402
    _Model,
    _arguments,
    _candidate,
)


def _attention_state(model: _Model) -> dict[str, torch.Tensor]:
    attention = model.model.layers[0].self_attn
    return {
        "q_proj.weight": attention.q_proj.weight.detach().cpu().clone(),
        "k_proj.weight": attention.k_proj.weight.detach().cpu().clone(),
        "v_proj.weight": attention.v_proj.weight.detach().cpu().clone(),
        "out_proj.weight": attention.out_proj.weight.detach().cpu().clone(),
        "q_layernorm.weight": attention.q_layernorm.weight.detach().cpu().clone(),
        "k_layernorm.weight": attention.k_layernorm.weight.detach().cpu().clone(),
    }


def _installed_structural_model(
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[_Model, object]:
    model = _Model(dtype=dtype)
    target = _attention_state(model)
    arguments = {
        **_arguments(model),
        "target_query_weight_template": target["q_proj.weight"],
        "target_key_weight_template": target["k_proj.weight"],
        "target_value_weight_template": target["v_proj.weight"],
        "target_out_weight_template": target["out_proj.weight"],
    }
    structural = _candidate(**arguments).compiled
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        attention.q_proj.weight.copy_(structural.query_weight)
        attention.k_proj.weight.copy_(structural.key_weight)
        attention.v_proj.weight.copy_(structural.value_weight)
        attention.out_proj.weight.copy_(structural.out_weight)
    return model, structural


def _parameter_identity(model: _Model) -> dict[str, object]:
    return {
        "ids": {name: id(value) for name, value in model.named_parameters()},
        "pointers": {
            name: int(value.data_ptr()) for name, value in model.named_parameters()
        },
        "gradient_ids": {
            name: None if value.grad is None else id(value.grad)
            for name, value in model.named_parameters()
        },
        "gradients": {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in model.named_parameters()
        },
    }


def test_installs_only_bounded_source_local_o_and_preserves_qkv_norm_state() -> None:
    model, structural = _installed_structural_model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7.0)
    identity = _parameter_identity(model)
    before = _attention_state(model)
    before_model = fingerprint_lfm2_parent(model).identity_sha256
    fit_case = _source_local_output_fit_case(structural, action_scale=64.0)

    result = compile_install_lfm2_semantic_attention_source_local_output(
        target_model=model,
        layer_index=0,
        structural_candidate=structural,
        fit_cases=(fit_case,),
    )

    after = _attention_state(model)
    receipt = result.receipt
    assert result.verify()
    assert receipt["schema"] == (
        LFM2_SEMANTIC_ATTENTION_SOURCE_LOCAL_OUTPUT_INSTALLATION_SCHEMA
    )
    assert receipt["installed"] is True
    assert receipt["changed_state_keys"] == ["model.layers.0.self_attn.out_proj.weight"]
    assert receipt["only_out_projection_permitted_to_change"] is True
    assert receipt["qkv_bit_identical_to_structural_candidate"] is True
    assert receipt["qk_norms_unchanged"] is True
    assert receipt["parameter_count_changed"] is False
    assert receipt["topology_changed"] is False
    assert receipt["training_steps"] == 0
    assert receipt["backward_calls"] == 0
    assert receipt["optimizer_steps"] == 0
    assert receipt["numeric_metrics_have_semantic_decision_authority"] is False
    assert fingerprint_lfm2_parent(model).identity_sha256 != before_model
    for role in (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    ):
        assert torch.equal(after[role], before[role])
    assert torch.equal(
        after["out_proj.weight"],
        result.compiled.out_weight_source_local_corrected,
    )
    assert not torch.equal(after["out_proj.weight"], before["out_proj.weight"])
    assert {name: id(value) for name, value in model.named_parameters()} == identity[
        "ids"
    ]
    assert {
        name: int(value.data_ptr()) for name, value in model.named_parameters()
    } == identity["pointers"]
    assert {
        name: id(value.grad) for name, value in model.named_parameters()
    } == identity["gradient_ids"]
    for name, value in model.named_parameters():
        assert torch.equal(value.grad, identity["gradients"][name])


def test_rejects_stale_live_structural_state_without_mutation() -> None:
    model, structural = _installed_structural_model()
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.add_(1.0)
    before = fingerprint_lfm2_parent(model).identity_sha256
    fit_case = _source_local_output_fit_case(structural)

    with pytest.raises(
        Lfm2SemanticAttentionSourceLocalOutputError,
        match="Live Q/K/V/O does not equal",
    ):
        compile_install_lfm2_semantic_attention_source_local_output(
            target_model=model,
            layer_index=0,
            structural_candidate=structural,
            fit_cases=(fit_case,),
        )

    assert fingerprint_lfm2_parent(model).identity_sha256 == before


def test_installation_receipt_tampering_fails_result_verification() -> None:
    model, structural = _installed_structural_model()
    fit_case = _source_local_output_fit_case(structural, action_scale=8.0)
    result = compile_install_lfm2_semantic_attention_source_local_output(
        target_model=model,
        layer_index=0,
        structural_candidate=structural,
        fit_cases=(fit_case,),
    )
    body = dict(result.receipt)
    body.pop("result_sha256")
    body["only_out_projection_permitted_to_change"] = False
    with pytest.raises(
        Lfm2SemanticAttentionSourceLocalOutputError,
        match="failed result authentication",
    ):
        replace(
            result,
            receipt={**body, "result_sha256": canonical_sha256(body)},
        )
