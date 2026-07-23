from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_attention_group_qdeim_installer as installer  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from tests.test_hf_lfm2_semantic_attention_group_qdeim_installer import (  # noqa: E402
    _Model,
    _bundle,
    _candidate,
    _deficit,
    _sealed,
    _source,
)


_RECIPE_ID = "nonlinear_qdeim_coupled_attention_structural"


def _receipt(body: dict[str, object], seal_name: str) -> MappingProxyType:
    return MappingProxyType({**body, seal_name: canonical_sha256(body)})


def _reseal(
    value: object,
    *,
    seal_name: str = "result_sha256",
) -> MappingProxyType:
    body = dict(value)
    body.pop(seal_name, None)
    return _receipt(body, seal_name)


def _replace_nested_receipt(
    outer: object,
    *,
    field: str,
    nested: object,
) -> MappingProxyType:
    body = dict(outer)
    body.pop("result_sha256")
    body[field] = dict(nested)
    return _receipt(body, "result_sha256")


def _wrapper_arguments(
    model: _Model,
    *,
    query_norm: torch.Tensor | None = None,
    key_norm: torch.Tensor | None = None,
) -> dict[str, object]:
    source = _source(query_norm=query_norm, key_norm=key_norm)
    norm_inventory = source["source_norm_inventory"]
    decoder_body = {
        "schema": "test-normalized-input-decoder-v1",
        "fit_partition_only": True,
        "numeric_diagnostics_have_semantic_decision_authority": False,
    }
    chart_body = {
        "schema": "test-residual-chart-v1",
        "fit_partition_only": True,
        "numeric_diagnostics_have_semantic_decision_authority": False,
    }
    direct_body = {
        "schema": "neural-forge-lfm2-direct-parent-qkvo-session-load-v1",
        "layer_index": 0,
        "parent_session_open_receipt_sha256": "e" * 64,
        "source_inventory_sha256": source["source_inventory"].inventory_sha256,
        "query_norm_weight_sha256": norm_inventory.query_norm_receipt["tensor_sha256"],
        "key_norm_weight_sha256": norm_inventory.key_norm_receipt["tensor_sha256"],
        "operator_norm_weight_sha256": "f" * 64,
        "qk_norms_loaded_as_atomic_attention_source": True,
        "compiler_lineage_sha256": "9" * 64,
        "direct_local_safetensors": True,
        "parent_model_objects_constructed": 0,
    }
    return {
        "target_model": model,
        "session": SimpleNamespace(),
        "layer_index": 0,
        "fit_batch_indices": (0,),
        "fit_batch_sample_weights": {0: 1.0},
        **source,
        "normalized_input_decoder": torch.eye(4, dtype=torch.float64),
        "normalized_input_decoder_receipt": _receipt(
            decoder_body,
            "decoder_sha256",
        ),
        "residual_output_chart": torch.eye(4, dtype=torch.float64),
        "residual_chart_receipt": _receipt(chart_body, "result_sha256"),
        "direct_attention_source_receipt": _receipt(
            direct_body,
            "result_sha256",
        ),
        "cache_key": "test-cache",
        "bridge_artifact_sha256": "b" * 64,
        "parameter_count": stack._parameter_count(model),
        "body_recipe": _RECIPE_ID,
        "installer": installer.compile_install_lfm2_semantic_attention_group_qdeim,
    }


def _model_transaction_identity(model: _Model) -> dict[str, object]:
    return {
        "fingerprint": fingerprint_lfm2_parent(model).identity_sha256,
        "bundle": _bundle(model),
        "parameter_ids": {name: id(value) for name, value in model.named_parameters()},
        "storage_pointers": {
            name: value.data_ptr() for name, value in model.named_parameters()
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


def _assert_exact_transaction_identity(
    model: _Model,
    expected: dict[str, object],
) -> None:
    assert fingerprint_lfm2_parent(model).identity_sha256 == expected["fingerprint"]
    assert all(
        torch.equal(value, expected["bundle"][name])
        for name, value in _bundle(model).items()
    )
    assert {name: id(value) for name, value in model.named_parameters()} == expected[
        "parameter_ids"
    ]
    assert {
        name: value.data_ptr() for name, value in model.named_parameters()
    } == expected["storage_pointers"]
    assert {
        name: None if value.grad is None else id(value.grad)
        for name, value in model.named_parameters()
    } == expected["gradient_ids"]
    for name, value in model.named_parameters():
        expected_gradient = expected["gradients"][name]
        if expected_gradient is None:
            assert value.grad is None
        else:
            assert torch.equal(value.grad, expected_gradient)


def _advisory_rejection_candidate(**kwargs: object) -> object:
    candidate = _candidate(**kwargs)
    report = dict(candidate.report)
    report.pop("result_sha256")
    report.update(
        {
            "diagnostic_numeric_score": -1.0e300,
            "diagnostic_manual_observation": "apparently-incoherent",
            "diagnostic_numeric_score_is_advisory_only": True,
            "diagnostic_manual_observation_is_advisory_only": True,
        }
    )
    return replace(candidate, report=_sealed(report))


def test_layer_wrapper_installs_and_authenticates_one_atomic_six_tensor_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7.0)
    identities = {name: id(value) for name, value in model.named_parameters()}
    pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    gradient_ids = {name: id(value.grad) for name, value in model.named_parameters()}
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _advisory_rejection_candidate,
    )

    receipt = stack._install_coupled_attention_group_qdeim_operator(
        **_wrapper_arguments(model)
    )
    authenticated = postfold._authenticate_lfm2_semantic_attention_operator_receipt(
        receipt,
        layer_index=0,
        body_recipe=_RECIPE_ID,
    )

    prefix = "model.layers.0.self_attn."
    expected_changed = sorted(
        prefix + role
        for role in (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
            "q_layernorm.weight",
            "k_layernorm.weight",
        )
    )
    assert receipt["changed_state_keys"] == expected_changed
    assert authenticated["changed_state_keys"] == expected_changed
    assert authenticated["changed_projection_state_keys"] == [
        name
        for name in expected_changed
        if name.rsplit(".", 2)[-2] in {"q_proj", "k_proj", "v_proj", "out_proj"}
    ]
    assert authenticated["changed_norm_state_keys"] == [
        name
        for name in expected_changed
        if name.rsplit(".", 2)[-2] in {"q_layernorm", "k_layernorm"}
    ]
    assert set(authenticated["six_tensor_bundle"]) == {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    }
    assert receipt["coupled_qkvo_single_candidate"] is True
    assert receipt["split_qk_or_vo_fallback_used"] is False
    assert receipt["legacy_split_attention_installer_called"] is False
    assert receipt["numeric_metric_used_for_candidate_selection"] is False
    assert receipt["numeric_metric_used_for_publication_or_rollback"] is False
    assert receipt["numeric_metrics_have_final_intelligence_authority"] is False
    assert receipt["binary_manual_threshold_has_iteration_veto_authority"] is False
    assert receipt["binary_manual_acceptance_gate_required"] is False
    assert receipt["intelligence_retention_target_is_objective_not_numeric_cutoff"]
    assert receipt["whole_decoded_behavior_is_primary_intelligence_evidence"]
    assert authenticated["numeric_behavior_gate_used"] is False
    assert authenticated["binary_manual_gate_used"] is False
    assert authenticated["retention_target_is_objective_not_numeric_cutoff"]
    assert {name: id(value) for name, value in model.named_parameters()} == identities
    assert {
        name: value.data_ptr() for name, value in model.named_parameters()
    } == pointers
    assert {
        name: id(value.grad) for name, value in model.named_parameters()
    } == gradient_ids


def test_wrapper_authentication_failure_after_install_restores_exact_entry_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 13.0)
    before = _model_transaction_identity(model)

    def stale_fit_partition_candidate(**kwargs: object) -> object:
        candidate = _candidate(**kwargs)
        report = dict(candidate.report)
        report.pop("result_sha256")
        capability = dict(report["cache_capability_receipt"])
        capability.pop("result_sha256")
        capability["fit_batch_indices"] = [999]
        capability = _sealed(capability)
        context_bank = dict(report["context_bank_receipt"])
        context_bank.pop("result_sha256")
        context_bank["fit_batch_indices"] = [999]
        context_bank = _sealed(context_bank)
        report["fit_batch_indices"] = [999]
        report["cache_capability_receipt"] = dict(capability)
        report["cache_capability_sha256"] = capability["result_sha256"]
        report["context_bank_receipt"] = dict(context_bank)
        report["context_bank_receipt_sha256"] = context_bank["result_sha256"]
        return replace(candidate, report=_sealed(report))

    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        stale_fit_partition_candidate,
    )

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="lost its cache, recipe, six-tensor, or no-gating contract",
    ):
        stack._install_coupled_attention_group_qdeim_operator(
            **_wrapper_arguments(model)
        )

    _assert_exact_transaction_identity(model, before)


def test_typed_deficit_has_no_fallback_and_restores_exact_entry_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 17.0)
    before = _model_transaction_identity(model)
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        lambda **_kwargs: _deficit(),
    )

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="no authenticated dense candidate",
    ):
        stack._install_coupled_attention_group_qdeim_operator(
            **_wrapper_arguments(model)
        )

    _assert_exact_transaction_identity(model, before)


@pytest.mark.parametrize(
    ("field", "unauthorized_value"),
    (
        ("numeric_metrics_have_final_intelligence_authority", True),
        ("binary_manual_threshold_has_iteration_veto_authority", True),
        ("binary_manual_acceptance_gate_required", True),
        ("intelligence_retention_target_is_objective_not_numeric_cutoff", False),
    ),
)
def test_persisted_gate_authority_tamper_is_rejected_without_touching_model(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    unauthorized_value: bool,
) -> None:
    model = _Model()
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    receipt = stack._install_coupled_attention_group_qdeim_operator(
        **_wrapper_arguments(model)
    )
    installed = _model_transaction_identity(model)
    tampered = dict(receipt)
    tampered[field] = unauthorized_value
    tampered.pop("result_sha256")
    resealed = MappingProxyType(
        {**tampered, "result_sha256": canonical_sha256(tampered)}
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="incomplete",
    ):
        postfold._authenticate_lfm2_semantic_attention_operator_receipt(
            resealed,
            layer_index=0,
            body_recipe=_RECIPE_ID,
        )

    _assert_exact_transaction_identity(model, installed)


def test_resealed_norm_inventory_cannot_disagree_with_direct_q_norm_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    receipt = stack._install_coupled_attention_group_qdeim_operator(
        **_wrapper_arguments(model)
    )

    norm_body = dict(receipt["source_norm_inventory"])
    norm_body.pop("inventory_sha256")
    tensor_receipts = {
        name: dict(value) for name, value in norm_body["tensor_receipts"].items()
    }
    tensor_receipts["q_layernorm.weight"]["tensor_sha256"] = "0" * 64
    norm_body["tensor_receipts"] = tensor_receipts
    resealed_norm = _receipt(norm_body, "inventory_sha256")

    raw_body = dict(receipt["coupled_attention_installation_receipt"])
    raw_body.pop("result_sha256")
    raw_body["source_norm_inventory_sha256"] = resealed_norm["inventory_sha256"]
    resealed_raw = _receipt(raw_body, "result_sha256")
    tampered = _replace_nested_receipt(
        receipt,
        field="source_norm_inventory",
        nested=resealed_norm,
    )
    tampered = _replace_nested_receipt(
        tampered,
        field="coupled_attention_installation_receipt",
        nested=resealed_raw,
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="incomplete"):
        postfold._authenticate_lfm2_semantic_attention_operator_receipt(
            tampered,
            layer_index=0,
            body_recipe=_RECIPE_ID,
        )


def test_resealed_swapped_layer_direct_source_receipt_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    receipt = stack._install_coupled_attention_group_qdeim_operator(
        **_wrapper_arguments(model)
    )
    direct_body = dict(receipt["direct_attention_source_receipt"])
    direct_body.pop("result_sha256")
    direct_body["layer_index"] = 2
    tampered = _replace_nested_receipt(
        receipt,
        field="direct_attention_source_receipt",
        nested=_receipt(direct_body, "result_sha256"),
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="incomplete"):
        postfold._authenticate_lfm2_semantic_attention_operator_receipt(
            tampered,
            layer_index=0,
            body_recipe=_RECIPE_ID,
        )


@pytest.mark.parametrize(
    "changed_norm_roles",
    (
        (),
        ("q_layernorm.weight",),
        ("q_layernorm.weight", "k_layernorm.weight"),
    ),
)
def test_authenticator_accepts_required_projections_with_optional_norm_changes(
    monkeypatch: pytest.MonkeyPatch,
    changed_norm_roles: tuple[str, ...],
) -> None:
    model = _Model()
    neutral = torch.ones(2, dtype=torch.bfloat16)
    query_norm = (
        torch.tensor([1.25, 0.75], dtype=torch.bfloat16)
        if "q_layernorm.weight" in changed_norm_roles
        else neutral.clone()
    )
    key_norm = (
        torch.tensor([0.875, 1.375], dtype=torch.bfloat16)
        if "k_layernorm.weight" in changed_norm_roles
        else neutral.clone()
    )
    with torch.no_grad():
        attention = model.model.layers[0].self_attn
        attention.q_layernorm.weight.copy_(neutral)
        attention.k_layernorm.weight.copy_(neutral)
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    receipt = stack._install_coupled_attention_group_qdeim_operator(
        **_wrapper_arguments(
            model,
            query_norm=query_norm,
            key_norm=key_norm,
        )
    )
    prefix = "model.layers.0.self_attn."
    changed = sorted(
        [
            prefix + role
            for role in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "out_proj.weight",
                *changed_norm_roles,
            )
        ]
    )
    authenticated = postfold._authenticate_lfm2_semantic_attention_operator_receipt(
        receipt,
        layer_index=0,
        body_recipe=_RECIPE_ID,
    )

    assert authenticated["changed_state_keys"] == changed
    assert len(authenticated["changed_projection_state_keys"]) == 4
    assert len(authenticated["changed_norm_state_keys"]) == len(changed_norm_roles)


def test_resealed_changed_role_claim_cannot_disagree_with_transaction_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    receipt = stack._install_coupled_attention_group_qdeim_operator(
        **_wrapper_arguments(model)
    )
    prefix = "model.layers.0.self_attn."
    projections_only = sorted(
        prefix + role
        for role in (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
        )
    )
    raw_body = dict(receipt["coupled_attention_installation_receipt"])
    raw_body.pop("result_sha256")
    raw_body["changed_state_keys"] = projections_only
    outer_body = dict(receipt)
    outer_body.pop("result_sha256")
    outer_body["changed_state_keys"] = projections_only
    outer_body["actual_changed_state_key_count"] = len(projections_only)
    outer_body["coupled_attention_installation_receipt"] = dict(
        _receipt(raw_body, "result_sha256")
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="incomplete"):
        postfold._authenticate_lfm2_semantic_attention_operator_receipt(
            _receipt(outer_body, "result_sha256"),
            layer_index=0,
            body_recipe=_RECIPE_ID,
        )


def _coupled_stack_roots(
    authenticated: object,
) -> MappingProxyType:
    operator = dict(authenticated)
    cache_contract = {
        "committed_local_cache_only": True,
        "manifest_completion_and_layout_verified": True,
        "trace_file_sha256_passes_per_session": 1,
        "record_file_sha256_passes_per_session": 1,
        "record_self_hash_verified": True,
        "safetensors_metadata_verified": True,
        "complete_tensor_key_inventory_verified": True,
        "selected_tensor_descriptors_verified_per_load": True,
        "stable_file_identity_checked_before_and_after_load": True,
        "authenticated_trace_guard_held_open": True,
        "safetensors_mapping_held_open": False,
        "safetensors_mapping_opened_per_selection_and_closed": True,
        "unselected_tensor_payloads_loaded": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "toctou_boundary": "test-authenticated-local-file-session",
    }
    cache_session = _receipt(
        {
            "schema": "neural-forge-lfm2-circuit-trace-selective-session-v1",
            "cache_key": operator["cache_key"],
            "manifest_sha256": operator["cache_manifest_sha256"],
            "completion_sha256": "c" * 64,
            "batch_count": 1,
            "ordered_authenticated_batches": [
                {
                    "batch_index": 0,
                    "input_sha256": "1" * 64,
                    "trace_file_sha256": "2" * 64,
                    "record_file_sha256": "3" * 64,
                    "record_sha256": "4" * 64,
                    "tensor_count": 1,
                    "logical_bytes": 2,
                }
            ],
            "authentication_contract": cache_contract,
        },
        "session_sha256",
    )
    reconstructed_parent = {
        "schema": "neural-forge-lfm2-parent-fingerprint-v2",
        "model_class": "test.Lfm2Model",
        "config_sha256": "5" * 64,
        "state_dict_sha256": "6" * 64,
        "state_tensor_count": 1,
        "runtime_buffer_sha256": "7" * 64,
        "runtime_buffer_count": 0,
    }
    parent_identity_sha256 = canonical_sha256(reconstructed_parent)
    operator["cache_parent_fingerprint"] = parent_identity_sha256
    operator["source_parent_identity_sha256"] = parent_identity_sha256
    if isinstance(authenticated, dict):
        authenticated.update(operator)
    parent_identity = _receipt(
        {
            "schema": ("neural-forge-direct-parent-exact-identity-reconstruction-v1"),
            "cache_parent_identity_sha256": parent_identity_sha256,
            "cache_parent_record": dict(reconstructed_parent),
            "checkpoint_manifest_sha256": "8" * 64,
            "config_file_sha256": "9" * 64,
            "normalized_config_sha256": "5" * 64,
            "checkpoint_tensor_inventory_sha256": "a" * 64,
            "checkpoint_tensor_count": 0,
            "checkpoint_tensor_records": [],
            "reconstructed_state_dict_sha256": "6" * 64,
            "reconstructed_state_tensor_count": 1,
            "runtime_buffer_records": [],
            "reconstructed_runtime_buffer_sha256": "7" * 64,
            "reconstructed_runtime_buffer_count": 0,
            "reconstructed_parent_identity": reconstructed_parent,
            "reconstructed_parent_identity_sha256": parent_identity_sha256,
            "exact_parent_identity_authenticated": True,
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "network_access_performed": False,
        },
        "result_sha256",
    )
    parent_session = _receipt(
        {
            "schema": "neural-forge-direct-parent-session-v2",
            "parent_model_identity_sha256": parent_identity_sha256,
            "checkpoint_manifest_sha256": "8" * 64,
            "open_receipt_sha256": operator[
                "direct_parent_session_open_receipt_sha256"
            ],
            "config_file_sha256": "9" * 64,
            "normalized_config_sha256": "5" * 64,
            "norm_epsilon": 1.0e-5,
            "loaded_tensor_sha256": {
                "model.layers.0.self_attn.q_proj.weight": "a" * 64
            },
            "parent_identity_reconstruction_sha256": parent_identity["result_sha256"],
            "parent_identity_reconstruction": dict(parent_identity),
            "exact_parent_identity_authenticated": True,
            "checkpoint_files_rehashed_after_materialization": True,
            "shards_hashed_once": True,
            "shards_opened_once": True,
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "network_access_performed": False,
        },
        "session_sha256",
    )
    body = {
        "cache_key": operator["cache_key"],
        "attention_layer_indices": [2],
        "cache_selective_session_sha256": cache_session["session_sha256"],
        "cache_selective_session": dict(cache_session),
        "bridge_artifact_sha256": operator["bridge_artifact_sha256"],
        "cache_parent_fingerprint": operator["cache_parent_fingerprint"],
        "parent_checkpoint_session_sha256": parent_session["session_sha256"],
        "parent_checkpoint_session": dict(parent_session),
        "direct_attention_source_receipt_sha256": {
            "2": operator["direct_attention_source_receipt_sha256"]
        },
    }
    return _receipt(body, "result_sha256")


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("cache_key", "different-cache-root"),
        ("cache_selective_session_sha256", "a" * 64),
        ("bridge_artifact_sha256", "c" * 64),
        ("cache_parent_fingerprint", "4" * 64),
        ("parent_checkpoint_session_sha256", "a" * 64),
        ("direct_attention_source_receipt_sha256", {"2": "5" * 64}),
    ),
)
def test_resealed_stack_cannot_rebind_coupled_operator_roots(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    model = _Model()
    monkeypatch.setattr(
        installer,
        "compile_lfm2_semantic_attention_group_qdeim_from_cache",
        _candidate,
    )
    receipt = stack._install_coupled_attention_group_qdeim_operator(
        **_wrapper_arguments(model)
    )
    authenticated = dict(
        postfold._authenticate_lfm2_semantic_attention_operator_receipt(
            receipt,
            layer_index=0,
            body_recipe=_RECIPE_ID,
        )
    )
    authenticated["layer_index"] = 2
    roots = _coupled_stack_roots(authenticated)
    authenticated["cache_selective_session_sha256"] = roots[
        "cache_selective_session_sha256"
    ]
    verified_roots = postfold._verify_sealed(
        roots,
        seal_name="result_sha256",
        label="test coupled stack roots",
    )
    bound = postfold._authenticate_lfm2_coupled_attention_stack_root_binding(
        verified_roots,
        authenticated,
    )
    assert bound["layer_index"] == 2

    tampered_body = dict(roots)
    tampered_body.pop("result_sha256")
    tampered_body[field] = replacement
    resealed = _receipt(tampered_body, "result_sha256")
    verified_tampered = postfold._verify_sealed(
        resealed,
        seal_name="result_sha256",
        label="test tampered coupled stack roots",
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="not bound to the persisted layer-stack",
    ):
        postfold._authenticate_lfm2_coupled_attention_stack_root_binding(
            verified_tampered,
            authenticated,
        )
