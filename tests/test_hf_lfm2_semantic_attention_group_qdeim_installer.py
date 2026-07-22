from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_attention_group_qdeim_installer as installer  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_attention_group_qdeim import (  # noqa: E402
    LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
    Lfm2CompiledAttentionGroupQdeim,
    lfm2_attention_group_qdeim_source_state_sha256,
    lfm2_attention_group_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (  # noqa: E402
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_semantic_attention_group_qdeim_adapter import (  # noqa: E402
    LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_ADAPTER_SCHEMA,
    Lfm2SemanticAttentionGroupQdeimCandidate,
    Lfm2SemanticAttentionGroupQdeimEvidenceDeficit,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


class _Config:
    model_type = "lfm2"
    norm_eps = 1e-5

    def to_dict(self) -> dict[str, object]:
        return {"model_type": self.model_type, "norm_eps": self.norm_eps}


class _Norm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(width, dtype=torch.bfloat16))


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 2
        self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        self.v_proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        self.out_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.q_layernorm = _Norm(2)
        self.k_layernorm = _Norm(2)
        with torch.no_grad():
            self.q_proj.weight.zero_()
            self.k_proj.weight.zero_()
            self.v_proj.weight.zero_()
            self.out_proj.weight.zero_()


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


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
        raise AssertionError("The installer must never execute the child model.")


def _source() -> dict[str, object]:
    source_q = torch.arange(8 * 4, dtype=torch.bfloat16).reshape(8, 4).add(1)
    source_k = torch.arange(4 * 4, dtype=torch.bfloat16).reshape(4, 4).add(2)
    source_v = torch.arange(4 * 4, dtype=torch.bfloat16).reshape(4, 4).add(3)
    source_o = torch.arange(4 * 8, dtype=torch.bfloat16).reshape(4, 8).add(4)
    source_inventory = seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=source_q,
        source_key_weight=source_k,
        source_value_weight=source_v,
        source_output_weight=source_o,
    )
    query_norm = torch.tensor([1.25, 0.75], dtype=torch.bfloat16)
    key_norm = torch.tensor([0.875, 1.375], dtype=torch.bfloat16)
    norm_inventory = installer.seal_lfm2_semantic_attention_group_qdeim_norm_inventory(
        layer_index=0,
        source_parent_identity_sha256="d" * 64,
        source_inventory=source_inventory,
        source_query_norm_weight=query_norm,
        source_key_norm_weight=key_norm,
    )
    return {
        "source_query_weight": source_q,
        "source_key_weight": source_k,
        "source_value_weight": source_v,
        "source_out_weight": source_o,
        "source_inventory": source_inventory,
        "source_query_norm_weight": query_norm,
        "source_key_norm_weight": key_norm,
        "source_norm_inventory": norm_inventory,
    }


def _arguments(model: _Model) -> dict[str, object]:
    return {
        "target_model": model,
        "session": SimpleNamespace(),
        "layer_index": 0,
        "fit_batch_indices": (0,),
        "fit_batch_sample_weights": {0: 1.0},
        **_source(),
        "normalized_input_decoder": torch.eye(4, dtype=torch.float64),
        "normalized_input_decoder_receipt": {"test": "decoder"},
        "residual_output_chart": torch.eye(4, dtype=torch.float64),
        "residual_chart_receipt": {"test": "chart"},
    }


def _sealed(body: dict[str, object]) -> MappingProxyType:
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _candidate_state_sha256(state: dict[str, torch.Tensor]) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-attention-group-qdeim-candidate-state-v1",
            "tensor_sha256": {
                name: lfm2_authenticated_tensor_sha256(value)
                for name, value in sorted(state.items())
            },
        }
    )


def _candidate(**kwargs: object) -> Lfm2SemanticAttentionGroupQdeimCandidate:
    target_q = kwargs["target_query_weight_template"]
    target_k = kwargs["target_key_weight_template"]
    target_v = kwargs["target_value_weight_template"]
    target_o = kwargs["target_out_weight_template"]
    state = {
        "q_proj.weight": torch.full_like(target_q, 1.0),
        "k_proj.weight": torch.full_like(target_k, 2.0),
        "v_proj.weight": torch.full_like(target_v, 3.0),
        "out_proj.weight": torch.full_like(target_o, 4.0),
    }
    state_sha = _candidate_state_sha256(state)
    source_state_sha = lfm2_attention_group_qdeim_source_state_sha256(
        kwargs["source_query_weight"],
        kwargs["source_key_weight"],
        kwargs["source_value_weight"],
        kwargs["source_out_weight"],
    )
    decoder_sha = lfm2_authenticated_tensor_sha256(kwargs["normalized_input_decoder"])
    chart_sha = lfm2_authenticated_tensor_sha256(kwargs["residual_output_chart"])
    context_sha = "8" * 64
    snapshot_sha = "9" * 64
    basis = torch.tensor([[1.0], [0.5]], dtype=torch.float64)
    interpolation = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    reconstruction = torch.kron(
        interpolation,
        torch.eye(4, dtype=torch.float64),
    )
    target_sha = lfm2_attention_group_qdeim_target_template_sha256(
        target_q,
        target_k,
        target_v,
        target_o,
    )
    compiler_body = {
        "schema": LFM2_ATTENTION_GROUP_QDEIM_SCHEMA,
        "status": "materialized-for-holistic-whole-response-intelligence-evaluation",
        "method": "deterministic-block-constrained-qdeim-intact-gqa-producers",
        "topology": {
            "source_hidden_size": 4,
            "target_hidden_size": 4,
            "source_query_head_count": 4,
            "source_key_value_head_count": 2,
            "target_query_head_count": 2,
            "target_key_value_head_count": 1,
            "queries_per_key_value": 2,
            "head_dim": 2,
            "source_group_block_width": 4,
        },
        "authentication": {
            "source_state_sha256_before": source_state_sha,
            "source_state_sha256_after": source_state_sha,
            "source_state_immutable": True,
            "target_template_sha256": target_sha,
            "normalized_input_decoder_sha256": decoder_sha,
            "residual_output_chart_sha256": chart_sha,
            "fit_context_evidence_sha256": context_sha,
            "functional_snapshot_evidence_sha256": snapshot_sha,
        },
        "evidence": {
            "fit_partition_only": True,
            "heldout_rows_used": False,
        },
        "selection": {
            "selected_source_key_value_groups": [0],
            "selected_source_query_heads": [0, 1],
            "whole_gqa_groups_only": True,
            "qkv_producer_averaging_performed": False,
            "qkv_producer_rows_are_single_source_copies_through_decoder": True,
        },
        "interpolation": {
            "group_pod_basis_sha256": lfm2_authenticated_tensor_sha256(basis),
            "group_interpolation_sha256": lfm2_authenticated_tensor_sha256(
                interpolation
            ),
            "context_reconstruction_sha256": lfm2_authenticated_tensor_sha256(
                reconstruction
            ),
            "group_interpolation_shape": [1, 2],
            "context_reconstruction_shape": [4, 8],
            "numeric_diagnostics_have_semantic_decision_authority": False,
        },
        "materialization": {
            "candidate_state_sha256": state_sha,
            "candidate_dtype": "bfloat16",
            "standalone_dense_tensors": True,
            "dense_all_parent_output_columns_used": True,
            "pre_softmax_qkv_averaging": False,
            "decoder_shape": [4, 4],
            "residual_chart_shape": [4, 4],
        },
        "semantic_policy": {
            "numeric_threshold_auto_accept_permitted": False,
            "numeric_threshold_auto_veto_permitted": False,
            "automatic_semantic_publication_permitted": False,
            "whole_response_intelligence_evaluation_is_primary": True,
            "binary_manual_acceptance_gate_required": False,
            "retention_targets_are_objectives_not_cutoffs": True,
        },
        "prohibited_operations": {
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "pruning_used": False,
            "quantization_used": False,
            "sparsity_used": False,
            "parameter_sharing_used": False,
        },
    }
    compiled_report = _sealed(compiler_body)
    compiled = Lfm2CompiledAttentionGroupQdeim(
        selected_source_key_value_groups=(0,),
        selected_source_query_heads=(0, 1),
        query_weight=state["q_proj.weight"],
        key_weight=state["k_proj.weight"],
        value_weight=state["v_proj.weight"],
        out_weight=state["out_proj.weight"],
        group_pod_basis=basis,
        group_interpolation=interpolation,
        context_reconstruction=reconstruction,
        normalized_input_decoder=kwargs["normalized_input_decoder"].clone(),
        residual_output_chart=kwargs["residual_output_chart"].clone(),
        report=compiled_report,
    )
    tensor_hashes = {
        role: lfm2_authenticated_tensor_sha256(value) for role, value in state.items()
    }
    source_inventory = kwargs["source_inventory"]
    residual_chart_receipt = kwargs.get("residual_chart_receipt")
    residual_chart_receipt_sha = (
        residual_chart_receipt.get("result_sha256")
        if isinstance(residual_chart_receipt, Mapping)
        and isinstance(residual_chart_receipt.get("result_sha256"), str)
        else "4" * 64
    )
    decoder_receipt = kwargs.get("normalized_input_decoder_receipt")
    decoder_receipt_sha = (
        decoder_receipt.get("decoder_sha256")
        if isinstance(decoder_receipt, Mapping)
        and isinstance(decoder_receipt.get("decoder_sha256"), str)
        else "5" * 64
    )
    adapter_body = {
        "schema": LFM2_SEMANTIC_ATTENTION_GROUP_QDEIM_ADAPTER_SCHEMA,
        "status": "dense-candidate-materialized-for-whole-response-review",
        "layer_index": 0,
        "cache_key": "test-cache",
        "cache_parent_fingerprint": "d" * 64,
        "cache_manifest_sha256": "1" * 64,
        "cache_selective_session_sha256": "2" * 64,
        "cache_capability_sha256": "3" * 64,
        "source_inventory_sha256": source_inventory.inventory_sha256,
        "source_state_sha256": source_state_sha,
        "source_attention_contract": {
            "operator": "full_attention",
            "query_heads": 4,
            "key_value_heads": 2,
            "head_dim": 2,
        },
        "target_template_sha256": target_sha,
        "residual_chart_receipt_sha256": residual_chart_receipt_sha,
        "residual_chart_sha256": chart_sha,
        "normalized_input_decoder_receipt_sha256": decoder_receipt_sha,
        "normalized_input_decoder_sha256": decoder_sha,
        "heldout_tensor_payload_loads": 0,
        "fit_batch_indices": list(kwargs.get("fit_batch_indices", (0,))),
        "captured_true_post_softmax_context_only": True,
        "context_bank_receipt_sha256": "6" * 64,
        "standalone_context_evidence_sha256": context_sha,
        "standalone_compiler_result_sha256": compiled_report["result_sha256"],
        "selected_source_key_value_groups": [0],
        "selected_source_query_heads": [0, 1],
        "candidate_tensor_sha256": tensor_hashes,
        "candidate_state_sha256": state_sha,
        "candidate_dense_bfloat16_standalone": True,
        "qkv_producer_averaging_performed": False,
        "dense_all_parent_output_columns_used": True,
        "numeric_metrics_have_semantic_decision_authority": False,
        "binary_manual_threshold_has_iteration_veto_authority": False,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
        "automatic_semantic_acceptance_or_rejection_permitted": False,
    }
    return Lfm2SemanticAttentionGroupQdeimCandidate(
        layer_index=0,
        compiled=compiled,
        report=_sealed(adapter_body),
    )


def _deficit() -> Lfm2SemanticAttentionGroupQdeimEvidenceDeficit:
    body = {
        "schema": "test-attention-qdeim-deficit-v1",
        "status": "no-candidate-materialized",
        "deficit_kind": "insufficient-authenticated-true-context-group-rank",
        "dense_qkvo_candidate_materialized": False,
        "numeric_diagnostics_have_semantic_decision_authority": False,
        "whole_decoded_behavior_is_primary_intelligence_evidence": True,
    }
    return Lfm2SemanticAttentionGroupQdeimEvidenceDeficit(
        layer_index=0,
        deficit_kind="insufficient-authenticated-true-context-group-rank",
        structural_context_rows=None,
        structural_row_sample_weights=(),
        report=_sealed(body),
    )


def _bundle(model: _Model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in installer._transaction._parameter_bundle(model, 0).items()
    }


def test_coupled_candidate_installs_all_six_tensors_and_escapes_zero_norms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 9.0)
    identities = {name: id(value) for name, value in model.named_parameters()}
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    gradient_ids = {name: id(value.grad) for name, value in model.named_parameters()}
    gradient_before = {
        name: value.grad.detach().clone() for name, value in model.named_parameters()
    }
    topology = installer._transaction._topology_sha256(model)
    parameter_count = installer._transaction._parameter_count(model)
    arguments = _arguments(model)
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )

    result = installer.compile_install_lfm2_semantic_attention_group_qdeim(**arguments)

    assert isinstance(
        result,
        installer.Lfm2SemanticAttentionGroupQdeimInstallationResult,
    )
    attention = model.model.layers[0].self_attn
    torch.testing.assert_close(
        attention.q_proj.weight,
        torch.full_like(attention.q_proj.weight, 1.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        attention.k_proj.weight,
        torch.full_like(attention.k_proj.weight, 2.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        attention.v_proj.weight,
        torch.full_like(attention.v_proj.weight, 3.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        attention.out_proj.weight,
        torch.full_like(attention.out_proj.weight, 4.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        attention.q_layernorm.weight,
        arguments["source_query_norm_weight"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        attention.k_layernorm.weight,
        arguments["source_key_norm_weight"],
        rtol=0,
        atol=0,
    )
    report = result.receipt
    prefix = "model.layers.0.self_attn."
    assert report["changed_state_keys"] == sorted(
        prefix + role for role in installer._ALL_ROLES
    )
    assert report["complete_six_parameter_bundle_installed_transactionally"]
    assert report["blank_query_norm_zero_escape_proven"]
    assert report["blank_key_norm_zero_escape_proven"]
    assert report["numeric_metrics_have_semantic_decision_authority"] is False
    assert report["binary_manual_threshold_has_iteration_veto_authority"] is False
    assert report["whole_decoded_behavior_is_primary_intelligence_evidence"]
    assert installer._transaction._topology_sha256(model) == topology
    assert installer._transaction._parameter_count(model) == parameter_count
    assert {name: id(value) for name, value in model.named_parameters()} == identities
    assert {
        name: value.data_ptr() for name, value in model.named_parameters()
    } == pointers
    assert {
        name: id(value.grad) for name, value in model.named_parameters()
    } == gradient_ids
    assert all(
        torch.equal(value.grad, gradient_before[name])
        for name, value in model.named_parameters()
    )


def test_typed_deficit_returns_without_fallback_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before = _bundle(model)
    transaction_called = False

    def forbidden_transaction(*_args: object, **_kwargs: object) -> object:
        nonlocal transaction_called
        transaction_called = True
        raise AssertionError("A typed deficit must never reach installation.")

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        lambda **_kwargs: _deficit(),
    )
    monkeypatch.setattr(
        installer._transaction,
        "_install_qkvo_transaction",
        forbidden_transaction,
    )

    result = installer.compile_install_lfm2_semantic_attention_group_qdeim(
        **_arguments(model)
    )

    assert isinstance(result, Lfm2SemanticAttentionGroupQdeimEvidenceDeficit)
    assert result.candidate_materialized is False
    assert transaction_called is False
    assert fingerprint_lfm2_parent(model).identity_sha256 == before_identity
    assert all(
        torch.equal(value, before[name]) for name, value in _bundle(model).items()
    )


def test_stale_norm_inventory_fails_before_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    arguments = _arguments(model)
    arguments["source_query_norm_weight"] = (
        arguments["source_query_norm_weight"].clone() + 1
    )
    compiler_called = False

    def forbidden_compiler(**_kwargs: object) -> object:
        nonlocal compiler_called
        compiler_called = True
        raise AssertionError("Stale norm lineage must fail before compilation.")

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        forbidden_compiler,
    )
    with pytest.raises(
        installer.Lfm2SemanticAttentionGroupQdeimInstallationError,
        match="stale",
    ):
        installer.prepare_lfm2_semantic_attention_group_qdeim_installation(**arguments)
    assert compiler_called is False
    assert all(
        int(torch.count_nonzero(value).item()) == 0 for value in _bundle(model).values()
    )


def test_prepared_candidate_rejects_stale_target_identity_before_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    arguments = _arguments(model)
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    prepared = installer.prepare_lfm2_semantic_attention_group_qdeim_installation(
        **arguments
    )
    assert isinstance(
        prepared,
        installer.Lfm2PreparedSemanticAttentionGroupQdeimInstallation,
    )
    attention = model.model.layers[0].self_attn
    attention.q_layernorm.weight = torch.nn.Parameter(
        torch.zeros_like(attention.q_layernorm.weight)
    )
    transaction_called = False

    def forbidden_transaction(*_args: object, **_kwargs: object) -> object:
        nonlocal transaction_called
        transaction_called = True
        raise AssertionError("Stale target must fail before installation.")

    monkeypatch.setattr(
        installer._transaction,
        "_install_qkvo_transaction",
        forbidden_transaction,
    )
    with pytest.raises(
        installer.Lfm2SemanticAttentionGroupQdeimInstallationError,
        match="stale",
    ):
        installer.install_lfm2_prepared_semantic_attention_group_qdeim(
            target_model=model,
            prepared=prepared,
        )
    assert transaction_called is False
    assert int(torch.count_nonzero(attention.q_layernorm.weight).item()) == 0


def test_six_tensor_transaction_rolls_back_partial_copy_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 5.0)
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before = _bundle(model)
    identities = {name: id(value) for name, value in model.named_parameters()}
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    gradient_ids = {name: id(value.grad) for name, value in model.named_parameters()}
    gradient_before = {
        name: value.grad.detach().clone() for name, value in model.named_parameters()
    }
    original_copy = installer._transaction._copy_candidate
    copy_calls = 0

    def fail_fourth_copy(parameter: object, candidate: object) -> None:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 4:
            raise RuntimeError("injected-six-tensor-copy-failure")
        original_copy(parameter, candidate)

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    monkeypatch.setattr(
        installer._transaction,
        "_copy_candidate",
        fail_fourth_copy,
    )
    with pytest.raises(RuntimeError, match="injected-six-tensor-copy-failure"):
        installer.compile_install_lfm2_semantic_attention_group_qdeim(
            **_arguments(model)
        )

    assert copy_calls == 4
    assert fingerprint_lfm2_parent(model).identity_sha256 == before_identity
    assert all(
        torch.equal(value, before[name]) for name, value in _bundle(model).items()
    )
    assert {name: id(value) for name, value in model.named_parameters()} == identities
    assert {
        name: value.data_ptr() for name, value in model.named_parameters()
    } == pointers
    assert {
        name: id(value.grad) for name, value in model.named_parameters()
    } == gradient_ids
    assert all(
        torch.equal(value.grad, gradient_before[name])
        for name, value in model.named_parameters()
    )


def test_aliased_six_parameter_topology_fails_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    attention.k_layernorm.weight = attention.q_layernorm.weight
    compiler_called = False

    def forbidden_compiler(**_kwargs: object) -> object:
        nonlocal compiler_called
        compiler_called = True
        raise AssertionError("Aliased topology must fail before compilation.")

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        forbidden_compiler,
    )
    with pytest.raises(
        installer.Lfm2SemanticAttentionGroupQdeimInstallationError,
        match="six-Parameter",
    ):
        installer.prepare_lfm2_semantic_attention_group_qdeim_installation(
            **_arguments(model)
        )
    assert compiler_called is False


def test_distinct_parameters_with_overlapping_storage_fail_before_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    attention.k_layernorm.weight = torch.nn.Parameter(
        attention.q_layernorm.weight.detach()
    )
    assert id(attention.q_layernorm.weight) != id(attention.k_layernorm.weight)
    assert (
        attention.q_layernorm.weight.data_ptr()
        == attention.k_layernorm.weight.data_ptr()
    )
    compiler_called = False

    def forbidden_compiler(**_kwargs: object) -> object:
        nonlocal compiler_called
        compiler_called = True
        raise AssertionError("Overlapping storage must fail before compilation.")

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        forbidden_compiler,
    )
    with pytest.raises(
        installer.Lfm2SemanticAttentionGroupQdeimInstallationError,
        match="independent storage",
    ):
        installer.prepare_lfm2_semantic_attention_group_qdeim_installation(
            **_arguments(model)
        )
    assert compiler_called is False


def test_mutated_snapshot_backup_is_rejected_before_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    prepared = installer.prepare_lfm2_semantic_attention_group_qdeim_installation(
        **_arguments(model)
    )
    assert isinstance(
        prepared,
        installer.Lfm2PreparedSemanticAttentionGroupQdeimInstallation,
    )
    backup_name = "model.layers.0.self_attn.q_proj.weight"
    prepared.target_snapshot.parameter_values[backup_name].fill_(123)
    transaction_called = False

    def forbidden_transaction(*_args: object, **_kwargs: object) -> object:
        nonlocal transaction_called
        transaction_called = True
        raise AssertionError("Mutable backup drift must fail before installation.")

    monkeypatch.setattr(
        installer._transaction,
        "_install_qkvo_transaction",
        forbidden_transaction,
    )
    with pytest.raises(
        installer.Lfm2SemanticAttentionGroupQdeimInstallationError,
        match="unauthenticated",
    ):
        installer.install_lfm2_prepared_semantic_attention_group_qdeim(
            target_model=model,
            prepared=prepared,
        )
    assert transaction_called is False
    assert all(
        int(torch.count_nonzero(value).item()) == 0 for value in _bundle(model).values()
    )


def test_wrong_standalone_compiler_schema_fails_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    before = _bundle(model)

    def wrong_schema_candidate(**kwargs: object) -> object:
        candidate = _candidate(**kwargs)
        compiler_body = dict(candidate.compiled.report)
        compiler_body.pop("result_sha256")
        compiler_body["schema"] = "forged-standalone-compiler-v1"
        compiled = replace(candidate.compiled, report=_sealed(compiler_body))
        adapter_body = dict(candidate.report)
        adapter_body.pop("result_sha256")
        adapter_body["standalone_compiler_result_sha256"] = compiled.report[
            "result_sha256"
        ]
        return replace(candidate, compiled=compiled, report=_sealed(adapter_body))

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        wrong_schema_candidate,
    )
    with pytest.raises(
        installer.Lfm2SemanticAttentionGroupQdeimInstallationError,
        match="authentication",
    ):
        installer.prepare_lfm2_semantic_attention_group_qdeim_installation(
            **_arguments(model)
        )
    assert all(
        torch.equal(value, before[name]) for name, value in _bundle(model).items()
    )


def test_compiled_decoder_tensor_drift_fails_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    before = _bundle(model)

    def stale_decoder_candidate(**kwargs: object) -> object:
        candidate = _candidate(**kwargs)
        candidate.compiled.normalized_input_decoder.add_(1)
        return candidate

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        stale_decoder_candidate,
    )
    with pytest.raises(
        installer.Lfm2SemanticAttentionGroupQdeimInstallationError,
        match="authentication",
    ):
        installer.prepare_lfm2_semantic_attention_group_qdeim_installation(
            **_arguments(model)
        )
    assert all(
        torch.equal(value, before[name]) for name, value in _bundle(model).items()
    )


def test_post_transaction_validation_failure_rolls_back_all_six(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 11.0)
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before = _bundle(model)
    identities = {name: id(value) for name, value in model.named_parameters()}
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    gradient_ids = {name: id(value.grad) for name, value in model.named_parameters()}
    gradient_before = {
        name: value.grad.detach().clone() for name, value in model.named_parameters()
    }
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    prepared = installer.prepare_lfm2_semantic_attention_group_qdeim_installation(
        **_arguments(model)
    )
    assert isinstance(
        prepared,
        installer.Lfm2PreparedSemanticAttentionGroupQdeimInstallation,
    )
    original_transaction = installer._transaction._install_qkvo_transaction

    def wrong_postcondition(*args: object, **kwargs: object) -> object:
        before_sha, after_sha, _changed, count = original_transaction(
            *args,
            **kwargs,
        )
        return before_sha, after_sha, (), count

    monkeypatch.setattr(
        installer._transaction,
        "_install_qkvo_transaction",
        wrong_postcondition,
    )
    with pytest.raises(
        installer.Lfm2SemanticAttentionGroupQdeimInstallationError,
        match="publish exactly all six",
    ):
        installer.install_lfm2_prepared_semantic_attention_group_qdeim(
            target_model=model,
            prepared=prepared,
        )

    assert fingerprint_lfm2_parent(model).identity_sha256 == before_identity
    assert all(
        torch.equal(value, before[name]) for name, value in _bundle(model).items()
    )
    assert {name: id(value) for name, value in model.named_parameters()} == identities
    assert {
        name: value.data_ptr() for name, value in model.named_parameters()
    } == pointers
    assert {
        name: id(value.grad) for name, value in model.named_parameters()
    } == gradient_ids
    assert all(
        torch.equal(value.grad, gradient_before[name])
        for name, value in model.named_parameters()
    )
