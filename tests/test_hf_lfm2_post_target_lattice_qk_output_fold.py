from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_post_target_lattice_qk_output_fold as adapter  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.hf_lfm2_attention_group_qdeim import (  # noqa: E402
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    Lfm2AttentionGroupQdeimSourceLocalOutputFitCase,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_candidate_state_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_cpcf_macrocell_replay import (  # noqa: E402
    replay_lfm2_cpcf_macrocell,
)
from neural_forge.hf_lfm2_saved_body_target_lattice_qk import (  # noqa: E402
    LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (  # noqa: E402
    LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA,
)
from neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter import (  # noqa: E402
    _LayerReplayBank,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


class _Config:
    model_type = "lfm2"

    def to_dict(self) -> dict[str, object]:
        return {"model_type": self.model_type, "hidden_size": 4}


class _Norm(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=dtype))


class _Attention(torch.nn.Module):
    def __init__(self, *, layer_index: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.head_dim = 2
        self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.k_proj = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.v_proj = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
        self.out_proj = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
        self.q_layernorm = _Norm(dtype=dtype)
        self.k_layernorm = _Norm(dtype=dtype)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4, dtype=dtype).mul(2.0 + layer_index))
            self.k_proj.weight.copy_(
                torch.tensor(
                    [[1.5 + layer_index, 0, 0, 0], [0, 1.75 + layer_index, 0, 0]],
                    dtype=dtype,
                )
            )
            self.v_proj.weight.copy_(
                torch.tensor(
                    [[1 + layer_index, 0, 0, 0], [0, 1 + layer_index, 0, 0]],
                    dtype=dtype,
                )
            )
            self.out_proj.weight.copy_(
                torch.eye(4, dtype=dtype).mul(1.0 + 0.25 * layer_index)
            )


class _Layer(torch.nn.Module):
    def __init__(self, *, layer_index: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.self_attn = _Attention(layer_index=layer_index, dtype=dtype)


class _Backbone(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_Layer(layer_index=index, dtype=dtype) for index in range(2)]
        )


class _Model(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _Backbone(dtype=dtype)

    def forward(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("The post-Q/K O adapter must never execute the model.")


def _sealed(body: dict[str, object]) -> MappingProxyType:
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _sealed_provenance(body: dict[str, object]) -> MappingProxyType:
    return MappingProxyType({**body, "receipt_sha256": canonical_sha256(body)})


def _tensor_sha(value: torch.Tensor) -> str:
    return lfm2_authenticated_tensor_sha256(value)


def _structural_candidate(
    model: _Model,
    layer_index: int,
) -> Lfm2CompiledAttentionGroupQdeim:
    attention = model.model.layers[layer_index].self_attn
    dtype = attention.out_proj.weight.dtype
    original_q = torch.eye(4, dtype=dtype).mul(0.5 + layer_index)
    original_k = torch.tensor(
        [[0.5 + layer_index, 0, 0, 0], [0, 0.75 + layer_index, 0, 0]],
        dtype=dtype,
    )
    value = attention.v_proj.weight.detach().cpu().clone()
    out = attention.out_proj.weight.detach().cpu().clone()
    basis = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    interpolation = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    reconstruction = torch.kron(interpolation, torch.eye(4, dtype=torch.float64))
    decoder = torch.eye(4, dtype=torch.float64)
    chart = torch.eye(4, dtype=torch.float64)
    candidate_state = lfm2_attention_group_qdeim_candidate_state_sha256(
        original_q,
        original_k,
        value,
        out,
    )
    body = {
        "schema": LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        "status": "materialized-for-holistic-whole-response-intelligence-evaluation",
        "method": "test-intact-gqa-structural-candidate",
        "authentication": {
            "residual_output_chart_sha256": _tensor_sha(chart),
        },
        "selection": {
            "selected_source_key_value_groups": [0],
            "selected_source_query_heads": [0, 1],
        },
        "interpolation": {
            "group_pod_basis_sha256": _tensor_sha(basis),
            "group_interpolation_sha256": _tensor_sha(interpolation),
            "context_reconstruction_sha256": _tensor_sha(reconstruction),
        },
        "materialization": {
            "candidate_state_sha256": candidate_state,
            "candidate_dtype": str(dtype).removeprefix("torch."),
        },
        "prohibited_operations": {
            "model_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "pruning_used": False,
            "quantization_used": False,
            "sparsity_used": False,
            "parameter_sharing_used": False,
        },
    }
    return Lfm2CompiledAttentionGroupQdeim(
        selected_source_key_value_groups=(0,),
        selected_source_query_heads=(0, 1),
        query_weight=original_q,
        key_weight=original_k,
        value_weight=value,
        out_weight=out,
        group_pod_basis=basis,
        group_interpolation=interpolation,
        context_reconstruction=reconstruction,
        normalized_input_decoder=decoder,
        residual_output_chart=chart,
        report=_sealed(body),
    )


def _qk_treatment_receipt(model: _Model) -> MappingProxyType:
    dtype_name = str(next(model.parameters()).dtype).removeprefix("torch.")
    identity = fingerprint_lfm2_parent(model).identity_sha256
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    installations = []
    steps = []
    changed = []
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        prefix = f"model.layers.{index}.self_attn."
        local_changed = [prefix + "k_proj.weight", prefix + "q_proj.weight"]
        changed.extend(local_changed)
        fixed = {
            role: _tensor_sha(getattr(getattr(attention, module), "weight"))
            for role, module in (
                ("v_proj.weight", "v_proj"),
                ("out_proj.weight", "out_proj"),
                ("q_layernorm.weight", "q_layernorm"),
                ("k_layernorm.weight", "k_layernorm"),
            )
        }
        installation_body = {
            "schema": LFM2_SEMANTIC_TARGET_LATTICE_QK_ONLY_SCHEMA,
            "status": "installed-target-lattice-qk-only-closure",
            "layer_index": index,
            "installed": True,
            "qk_only": True,
            "changed_state_keys": local_changed,
            "actual_changed_state_keys_are_qk_only": True,
            "parameter_dtype": dtype_name,
            "checkpoint_dtype_preserved": True,
            "frozen_query_weight_sha256": _tensor_sha(attention.q_proj.weight),
            "frozen_key_weight_sha256": _tensor_sha(attention.k_proj.weight),
            "fixed_tensor_sha256_before_and_after": fixed,
        }
        installation = _sealed(installation_body)
        installations.append(dict(installation))
        step_body = {
            "schema": "neural-forge-lfm2-saved-body-qk-layer-step-v1",
            "layer_index": index,
            "operator_kind": "full_attention",
            "entry_replay_bank_sha256": str(index + 1) * 64,
            "entry_model_state_sha256": str(index + 2) * 64,
            "qk_installation_sha256": installation["result_sha256"],
            "transport_evidence_sha256": [str(index + 3) * 64],
            "propagation_replay_bank_sha256": str(index + 4) * 64,
            "propagation_model_state_sha256": str(index + 5) * 64,
            "qk_modified": True,
            "non_qk_tensor_modified": False,
        }
        steps.append(dict(_sealed(step_body)))
    body = {
        "schema": LFM2_SAVED_BODY_TARGET_LATTICE_QK_SCHEMA,
        "status": "installed-causal-order-target-lattice-qk-closure",
        "model_state_before_sha256": "a" * 64,
        "model_state_after_sha256": identity,
        "parameter_dtype": dtype_name,
        "checkpoint_dtype_preserved": True,
        "changed_state_keys": sorted(changed),
        "allowed_qk_state_keys": sorted(changed),
        "actual_changed_state_keys_are_qk_only": True,
        "attention_layer_indices": [0, 1],
        "attention_layer_count": 2,
        "attention_install_count": 2,
        "layer_steps": steps,
        "attention_installations": installations,
        "causal_layer_order_proven": True,
        "later_layers_replayed_from_treated_prefix": True,
        "fixed_child_value_output_used_as_endpoint_teacher": True,
        "all_non_qk_tensors_byte_identical": True,
        "parameter_element_count_before": parameter_count,
        "parameter_element_count_after": parameter_count,
        "parameter_count_changed": False,
        "operation_audit": {
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
            "parent_model_generate_calls": 0,
            "parent_weight_tensor_loads": 0,
            "parent_trace_recaptures": 0,
            "child_model_forward_calls": 0,
            "child_model_generate_calls": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    return _sealed(body)


def _typed_replay_case(
    model: _Model,
    *,
    layer_index: int,
    no_effect: bool,
) -> tuple[
    _LayerReplayBank,
    str,
    adapter.Lfm2PostTargetLatticeQkOutputReplayCaseBinding,
    Lfm2AttentionGroupQdeimSourceLocalOutputFitCase,
]:
    dtype = next(model.parameters()).dtype
    config = transformers.Lfm2Config(
        vocab_size=17,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        layer_types=["full_attention"],
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9000 + layer_index)
        replay_model = transformers.Lfm2ForCausalLM(config).to(dtype=dtype)
    replay_model.eval()
    target_attention = model.model.layers[layer_index].self_attn
    replay_attention = replay_model.model.layers[0].self_attn
    with torch.no_grad():
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "q_layernorm",
            "k_layernorm",
        ):
            getattr(replay_attention, name).weight.copy_(
                getattr(target_attention, name).weight
            )
        residual = torch.eye(4, dtype=dtype).unsqueeze(0)
        position_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
        cos, sin = replay_model.model.rotary_emb(
            residual,
            position_ids=position_ids,
        )
        causal_mask = torch.zeros((1, 1, 4, 4), dtype=dtype)
        future = torch.triu(torch.ones((4, 4), dtype=torch.bool), diagonal=1)
        causal_mask.masked_fill_(future, torch.finfo(dtype).min)
    replay = replay_lfm2_cpcf_macrocell(
        replay_model.model.layers[0],
        residual,
        position_embeddings=(cos, sin),
        additive_causal_mask=causal_mask,
    )
    identity = fingerprint_lfm2_parent(model).identity_sha256
    alignment = ("a" if layer_index == 0 else "b") * 64
    bank_body = {
        "schema": "neural-forge-lfm2-semantic-layer-replay-bank-v1",
        "layer_index": layer_index,
        "captured_child_model_state_sha256": identity,
        "batch_replays": [
            {
                "batch_index": 0,
                "alignment_sha256": alignment,
                "replay_sha256": replay.replay_sha256,
            }
        ],
        "model_forward_calls": 0,
        "layer_forward_calls": 0,
        "training_steps": 0,
    }
    bank = _LayerReplayBank(
        captured_child_model_state_sha256=identity,
        layer_index=layer_index,
        pre_residual_by_batch=MappingProxyType({0: residual.detach().cpu().clone()}),
        replay_by_batch=MappingProxyType({0: replay}),
        bank_sha256=canonical_sha256(bank_body),
    )
    case_id = f"layer-{layer_index:03d}.batch-0000"
    binding = adapter.Lfm2PostTargetLatticeQkOutputReplayCaseBinding(
        case_id=case_id,
        batch_index=0,
        alignment_sha256=alignment,
        target_positions=(0, 1, 2, 3),
        donor_positions=(0, 1, 2, 3),
    )
    row_identity = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-source-local-attention-aligned-fit-rows-v1",
            "partition": "fit",
            "layer_index": layer_index,
            "batch_index": 0,
            "alignment_sha256": alignment,
            "target_positions": [0, 1, 2, 3],
            "donor_positions": [0, 1, 2, 3],
            "row_count": 4,
        }
    )
    context = replay.operator_output_input[0]
    prior_action = (
        context.to(torch.float64)
        @ target_attention.out_proj.weight.detach().to(torch.float64).T
    )
    parent_post = prior_action
    if not no_effect:
        parent_post = parent_post + torch.eye(4, dtype=torch.float64).mul(0.25)
    case = Lfm2AttentionGroupQdeimSourceLocalOutputFitCase(
        case_id=case_id,
        parent_pre_operator_residual=torch.zeros(4, 4, dtype=torch.float64),
        parent_post_operator_residual=parent_post,
        live_child_pre_operator_residual=replay.operator_input[0],
        live_child_pre_o_context=context,
        row_sample_weights=(1.0, 1.0, 1.0, 1.0),
        parent_trajectory_receipt_sha256=str(layer_index + 7) * 64,
        live_child_replay_receipt_sha256=replay.replay_sha256,
        aligned_row_identity_sha256=row_identity,
    )
    return bank, alignment, binding, case


def _replay_evidence(
    model: _Model,
    treatment: MappingProxyType,
    *,
    no_effect: bool = False,
) -> dict[int, adapter.Lfm2PostTargetLatticeQkOutputReplayEvidence]:
    result = {}
    for index, step in enumerate(treatment["layer_steps"]):
        bank, alignment, binding, case = _typed_replay_case(
            model,
            layer_index=index,
            no_effect=no_effect,
        )
        result[index] = adapter.Lfm2PostTargetLatticeQkOutputReplayEvidence(
            layer_index=index,
            qk_treatment_result_sha256=treatment["result_sha256"],
            qk_layer_step_result_sha256=step["result_sha256"],
            qk_layer_propagation_replay_bank_sha256=step[
                "propagation_replay_bank_sha256"
            ],
            qk_layer_propagation_model_state_sha256=step[
                "propagation_model_state_sha256"
            ],
            fresh_reloaded_model_identity_sha256=(
                fingerprint_lfm2_parent(model).identity_sha256
            ),
            replay_bank=bank,
            alignment_sha256_by_batch={0: alignment},
            case_bindings=(binding,),
            fit_cases=(case,),
        )
    return result


def _fresh_reload_provenance(
    model: _Model,
    treatment: MappingProxyType,
) -> MappingProxyType:
    identity = fingerprint_lfm2_parent(model).identity_sha256
    state = fingerprint_model_state(model)
    dtype_name = str(next(model.parameters()).dtype).removeprefix("torch.")
    body = {
        "schema": adapter._QK_TREATED_BODY_PROVENANCE_SCHEMA,
        "status": "saved-reloaded-qk-only-treated-body",
        "source_body_authentication_basis": {
            "schema": "test-source-body-authentication-basis-v1",
        },
        "source_body_authentication_basis_sha256": "1" * 64,
        "source_structural_body_receipt_sha256": "2" * 64,
        "source_checkpoint_inventory_sha256": "3" * 64,
        "source_checkpoint_dtype": dtype_name,
        "source_body_model_state_sha256": "4" * 64,
        "source_body_lfm2_identity_sha256": "5" * 64,
        "source_causal_body_inventory_sha256": "6" * 64,
        "source_deferred_final_w2_authentication": {},
        "treated_checkpoint_inventory_sha256": "7" * 64,
        "treated_checkpoint_dtype": dtype_name,
        "checkpoint_dtype_preserved": True,
        "treated_body_model_state_sha256": state,
        "treated_body_lfm2_identity_sha256": identity,
        "treated_causal_body_inventory_sha256": "8" * 64,
        "changed_checkpoint_relative_paths": ["model.safetensors"],
        "changed_state_keys": list(treatment["changed_state_keys"]),
        "target_lattice_qk_treatment_sha256": treatment["result_sha256"],
        "causal_output_head_installed": False,
        "standalone_fresh_reload_verified": True,
        "operation_audit": dict(adapter._QK_FRESH_RELOAD_ZERO_OPERATION_AUDIT),
    }
    return _sealed_provenance(body)


def _arguments(model: _Model) -> dict[str, object]:
    treatment = _qk_treatment_receipt(model)
    provenance = _fresh_reload_provenance(model, treatment)
    replay_evidence = _replay_evidence(model, treatment)
    return {
        "target_model": model,
        "qk_treatment_receipt": treatment,
        "qk_fresh_reload_provenance_receipt": provenance,
        "expected_qk_fresh_reload_provenance_sha256": provenance["receipt_sha256"],
        "structural_candidates": {
            index: _structural_candidate(model, index) for index in range(2)
        },
        "replay_evidence_by_layer": replay_evidence,
        "expected_replay_evidence_sha256_by_layer": {
            index: evidence.evidence_sha256
            for index, evidence in replay_evidence.items()
        },
        "ridge": 1.0e-6,
        "fold_compute_device": "cpu",
    }


def _state(model: _Model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def test_fp32_post_qk_fold_changes_every_o_and_only_o() -> None:
    model = _Model(dtype=torch.float32)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    arguments = _arguments(model)
    before = _state(model)
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    gradient_ids = {name: id(value.grad) for name, value in model.named_parameters()}
    gradients = {
        name: value.grad.detach().clone() for name, value in model.named_parameters()
    }

    result = adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(
        **arguments
    )

    assert result.verify()
    expected_changed = [
        f"model.layers.{index}.self_attn.out_proj.weight" for index in range(2)
    ]
    assert result.receipt["changed_state_keys"] == expected_changed
    assert result.receipt["parameter_dtype"] == "float32"
    assert result.receipt["checkpoint_dtype_preserved"]
    assert result.receipt["qkv_byte_identical_before_and_after"]
    assert result.receipt["all_attention_out_projections_changed"]
    assert result.receipt["operation_audit"] == dict(adapter._ZERO_OPERATIONS)
    assert "child_model_forward_calls" not in result.receipt["operation_audit"]
    assert (
        "no unobserved model-forward counters claimed"
        in result.receipt["operation_audit_scope"]
    )
    after = _state(model)
    for name in before:
        if name in expected_changed:
            assert not torch.equal(before[name], after[name])
        else:
            assert torch.equal(before[name], after[name])
    assert {
        name: id(value) for name, value in model.named_parameters()
    } == parameter_ids
    assert {
        name: value.data_ptr() for name, value in model.named_parameters()
    } == pointers
    assert {
        name: id(value.grad) for name, value in model.named_parameters()
    } == gradient_ids
    assert all(
        torch.equal(value.grad, gradients[name])
        for name, value in model.named_parameters()
    )


def test_live_pre_qk_source_local_o_is_used_as_authenticated_prior() -> None:
    model = _Model(dtype=torch.float32)
    arguments = _arguments(model)
    structural_candidates = arguments["structural_candidates"]
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.out_proj.weight.add_(0.125)
    treatment = _qk_treatment_receipt(model)
    provenance = _fresh_reload_provenance(model, treatment)
    replay_evidence = _replay_evidence(model, treatment)
    arguments.update(
        {
            "qk_treatment_receipt": treatment,
            "qk_fresh_reload_provenance_receipt": provenance,
            "expected_qk_fresh_reload_provenance_sha256": provenance[
                "receipt_sha256"
            ],
            "replay_evidence_by_layer": replay_evidence,
            "expected_replay_evidence_sha256_by_layer": {
                index: evidence.evidence_sha256
                for index, evidence in replay_evidence.items()
            },
        }
    )
    assert any(
        not torch.equal(
            candidate.out_weight,
            model.model.layers[index].self_attn.out_proj.weight,
        )
        for index, candidate in structural_candidates.items()
    )

    result = adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(
        **arguments
    )

    assert result.verify()
    assert result.receipt["all_attention_out_projections_changed"]
    assert result.receipt[
        "live_output_prior_authenticated_by_qk_fixed_tensor_receipts"
    ]
    assert result.receipt["pre_qk_source_local_output_prior_permitted"]
    assert not result.receipt["pristine_structural_o_equality_required"]


def test_bfloat16_post_qk_fold_preserves_checkpoint_dtype() -> None:
    model = _Model(dtype=torch.bfloat16)
    result = adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(
        **_arguments(model)
    )

    assert result.verify()
    assert result.receipt["parameter_dtype"] == "bfloat16"
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert all(
        receipt["effectful_target_dtype_materialization"]
        and receipt["direction_preserved"]
        for receipt in result.receipt["direction_materialization_receipts"].values()
    )


def test_post_install_failure_rolls_back_every_o_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    arguments = _arguments(model)
    before = _state(model)
    identity = fingerprint_lfm2_parent(model).identity_sha256

    def fail_after_copy(**_kwargs: object) -> MappingProxyType:
        raise RuntimeError("forced post-copy authentication failure")

    monkeypatch.setattr(adapter, "_authenticate_installed_o_state", fail_after_copy)
    with pytest.raises(RuntimeError, match="forced post-copy"):
        adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(**arguments)

    assert fingerprint_lfm2_parent(model).identity_sha256 == identity
    assert all(
        torch.equal(value, _state(model)[name]) for name, value in before.items()
    )


def test_stale_treated_model_identity_fails_before_compilation() -> None:
    model = _Model()
    arguments = _arguments(model)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_layernorm.weight.add_(0.25)
    before = _state(model)

    with pytest.raises(
        adapter.Lfm2PostTargetLatticeQkOutputFoldError,
        match="exact dtype-preserving treated Q/K body",
    ):
        adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(**arguments)

    assert all(
        torch.equal(value, _state(model)[name]) for name, value in before.items()
    )


def test_receipted_dtype_mismatch_fails_closed() -> None:
    model = _Model(dtype=torch.float32)
    arguments = _arguments(model)
    body = dict(arguments["qk_treatment_receipt"])
    body.pop("result_sha256")
    body["parameter_dtype"] = "bfloat16"
    tampered = _sealed(body)
    arguments["qk_treatment_receipt"] = tampered
    provenance_body = dict(arguments["qk_fresh_reload_provenance_receipt"])
    provenance_body.pop("receipt_sha256")
    provenance_body["target_lattice_qk_treatment_sha256"] = tampered["result_sha256"]
    provenance = _sealed_provenance(provenance_body)
    arguments["qk_fresh_reload_provenance_receipt"] = provenance
    arguments["expected_qk_fresh_reload_provenance_sha256"] = provenance[
        "receipt_sha256"
    ]
    before = _state(model)

    with pytest.raises(
        adapter.Lfm2PostTargetLatticeQkOutputFoldError,
        match="exact dtype-preserving treated Q/K body",
    ):
        adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(**arguments)

    assert all(
        torch.equal(value, _state(model)[name]) for name, value in before.items()
    )


def test_non_o_compiler_result_fails_without_model_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    arguments = _arguments(model)
    before = _state(model)
    real_compile = (
        adapter.compile_lfm2_attention_group_qdeim_source_local_output_residual
    )

    def changed_query(**kwargs: object):
        compiled = real_compile(**kwargs)
        return replace(compiled, query_weight=compiled.query_weight.add(1.0))

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_attention_group_qdeim_source_local_output_residual",
        changed_query,
    )
    with pytest.raises(
        adapter.Lfm2PostTargetLatticeQkOutputFoldError,
        match="changed Q, K, or V",
    ):
        adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(**arguments)

    assert all(
        torch.equal(value, _state(model)[name]) for name, value in before.items()
    )


def test_arbitrary_o_cannot_hide_behind_valid_compiler_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    arguments = _arguments(model)
    before = _state(model)
    real_compile = (
        adapter.compile_lfm2_attention_group_qdeim_source_local_output_residual
    )

    def forged_output(**kwargs: object):
        compiled = real_compile(**kwargs)
        prior = compiled.out_weight_structural_prior
        forged = prior + prior.new_full(prior.shape, 7.0)
        forged_analytic = (
            forged.to(torch.float64) - prior.to(torch.float64)
        ).T.contiguous()
        return replace(
            compiled,
            out_weight_source_local_corrected=forged,
            analytic_out_correction_fp64=forged_analytic,
        )

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_attention_group_qdeim_source_local_output_residual",
        forged_output,
    )
    with pytest.raises(
        adapter.Lfm2PostTargetLatticeQkOutputFoldError,
        match="compiler tensors do not match",
    ):
        adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(**arguments)

    assert all(
        torch.equal(value, _state(model)[name]) for name, value in before.items()
    )


def test_spoofed_replay_tensor_fails_independent_evidence_pin() -> None:
    model = _Model()
    arguments = _arguments(model)
    before = _state(model)
    evidence = arguments["replay_evidence_by_layer"][0]
    with torch.no_grad():
        evidence.fit_cases[0].live_child_pre_o_context.add_(0.5)

    with pytest.raises(
        adapter.Lfm2PostTargetLatticeQkOutputFoldError,
        match="not independently pinned",
    ):
        adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(**arguments)

    assert all(
        torch.equal(value, _state(model)[name]) for name, value in before.items()
    )


def test_arbitrary_case_with_copied_replay_strings_is_not_typed_replay_evidence() -> (
    None
):
    model = _Model()
    arguments = _arguments(model)
    valid = arguments["replay_evidence_by_layer"][0]
    case = valid.fit_cases[0]
    spoofed_case = replace(
        case,
        live_child_pre_o_context=case.live_child_pre_o_context.add(0.5),
    )

    with pytest.raises(
        adapter.Lfm2PostTargetLatticeQkOutputFoldError,
        match="derived from the typed fresh causal replay",
    ):
        adapter.Lfm2PostTargetLatticeQkOutputReplayEvidence(
            layer_index=valid.layer_index,
            qk_treatment_result_sha256=valid.qk_treatment_result_sha256,
            qk_layer_step_result_sha256=valid.qk_layer_step_result_sha256,
            qk_layer_propagation_replay_bank_sha256=(
                valid.qk_layer_propagation_replay_bank_sha256
            ),
            qk_layer_propagation_model_state_sha256=(
                valid.qk_layer_propagation_model_state_sha256
            ),
            fresh_reloaded_model_identity_sha256=(
                valid.fresh_reloaded_model_identity_sha256
            ),
            replay_bank=valid.replay_bank,
            alignment_sha256_by_batch=valid.alignment_sha256_by_batch,
            case_bindings=valid.case_bindings,
            fit_cases=(spoofed_case,),
        )


def test_fake_fresh_reload_provenance_fails_external_pin() -> None:
    model = _Model()
    arguments = _arguments(model)
    before = _state(model)
    body = dict(arguments["qk_fresh_reload_provenance_receipt"])
    body.pop("receipt_sha256")
    body["target_lattice_qk_treatment_sha256"] = "f" * 64
    arguments["qk_fresh_reload_provenance_receipt"] = _sealed_provenance(body)

    with pytest.raises(
        adapter.Lfm2PostTargetLatticeQkOutputFoldError,
        match="not independently pinned",
    ):
        adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(**arguments)

    assert all(
        torch.equal(value, _state(model)[name]) for name, value in before.items()
    )


def test_target_dtype_no_effect_fails_closed() -> None:
    model = _Model()
    arguments = _arguments(model)
    treatment = arguments["qk_treatment_receipt"]
    replay_evidence = _replay_evidence(model, treatment, no_effect=True)
    arguments["replay_evidence_by_layer"] = replay_evidence
    arguments["expected_replay_evidence_sha256_by_layer"] = {
        index: evidence.evidence_sha256 for index, evidence in replay_evidence.items()
    }
    before = _state(model)

    with pytest.raises(
        adapter.Lfm2PostTargetLatticeQkOutputFoldError,
        match="no target-dtype effect",
    ):
        adapter.compile_install_lfm2_post_target_lattice_qk_output_fold(**arguments)

    assert all(
        torch.equal(value, _state(model)[name]) for name, value in before.items()
    )
