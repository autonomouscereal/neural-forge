from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import weakref
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
import neural_forge.hf_lfm2_cpcf_terminal_action_residual as terminal_action  # noqa: E402
from neural_forge.analytic_execution_profile import (  # noqa: E402
    resolve_analytic_execution_profile,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_artifact import DynamicFoldArtifactError  # noqa: E402
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    FeatureOrientation,
    ResidualFacingOperator,
    compile_role_balanced_operator_geometry,
)
from neural_forge.hf_lfm2_long_horizon_terminal_refold import (  # noqa: E402
    Lfm2SealedParentFinalRmsNorm,
)


class _TiedModel:
    def __init__(self, weight):
        self.weight = torch.nn.Parameter(weight.clone())
        self.model = SimpleNamespace(
            layers=(),
            rotary_emb=SimpleNamespace(
                inv_freq=torch.tensor([1.0], dtype=torch.float32),
                attention_scaling=1.0,
            ),
        )

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def parameters(self):
        yield self.weight

    def state_dict(self):
        return {"model.embed_tokens.weight": self.weight}


@pytest.mark.parametrize(
    "operator_kinds",
    (
        ("short_conv", "full_attention"),
        (
            "short_conv",
            "full_attention",
            "short_conv",
            "full_attention",
            "short_conv",
        ),
        tuple(
            "full_attention" if index in {2, 4, 6, 8, 10, 12} else "short_conv"
            for index in range(14)
        ),
    ),
    ids=("depth-2", "depth-5", "depth-14"),
)
def test_authenticated_stack_topology_and_live_child_depth_are_dynamic(
    operator_kinds: tuple[str, ...],
) -> None:
    attention_indices = tuple(
        index for index, kind in enumerate(operator_kinds) if kind == "full_attention"
    )
    short_conv_indices = tuple(
        index for index, kind in enumerate(operator_kinds) if kind == "short_conv"
    )
    layer_values = [{"operator_kind": kind} for kind in operator_kinds]
    stack_receipt = {
        "layer_count": len(operator_kinds),
        "operator_layer_kinds": list(operator_kinds),
        "attention_layer_indices": list(attention_indices),
        "short_conv_layer_indices": list(short_conv_indices),
        "terminal_layer_index": len(operator_kinds) - 1,
        "dynamic_layer_count_derived_from_authenticated_cache_and_child_topology": (
            True
        ),
    }
    target = SimpleNamespace(
        model=SimpleNamespace(
            layers=tuple(
                SimpleNamespace(is_attention_layer=kind == "full_attention")
                for kind in operator_kinds
            )
        ),
        config=SimpleNamespace(
            layer_types=tuple(
                "conv" if kind == "short_conv" else kind for kind in operator_kinds
            )
        ),
    )
    cache = SimpleNamespace(
        manifest={
            "layer_topology": {
                "layers": [{"contract": {"kind": kind}} for kind in operator_kinds]
            }
        }
    )

    assert postfold._authenticated_layer_stack_operator_topology(
        stack_receipt,
        layer_values,
    ) == (operator_kinds, attention_indices, short_conv_indices)
    assert postfold._operator_layer_kinds(cache, target) == operator_kinds


def test_fused_terminal_coverage_uses_authenticated_terminal_layer() -> None:
    source = inspect.getsource(
        postfold.compile_install_lfm2_fused_semantic_terminal_refold
    )

    assert source.count("layer_index=terminal_layer_index") == 2
    assert "layer_index=13" not in source


class _GaugeTiedModel(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.weight)


def test_self_dual_tied_gauge_install_preserves_operator_and_runtime(monkeypatch):
    generator = torch.Generator().manual_seed(1701)
    target = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    encoder = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    source_body = {
        "schema": "synthetic-tied-interface-v1",
        "target_weight_sha256": postfold._interface_tensor_sha256(target),
        "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
    }
    interface = postfold.TiedCausalInterfaceFold(
        target_weight=target,
        producer_encoder=encoder,
        report={
            **source_body,
            "report_sha256": canonical_sha256(source_body),
        },
    )
    model = _GaugeTiedModel(target.to(torch.bfloat16))
    model.weight.grad = torch.full_like(model.weight, 0.25)
    parameter_id = id(model.weight)
    storage_pointer = int(model.weight.data_ptr())
    gradient_id = id(model.weight.grad)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda value: SimpleNamespace(
            identity_sha256=canonical_sha256(postfold._state_tensor_hashes(value))
        ),
    )

    self_dual, receipt = postfold.install_lfm2_self_dual_tied_interface_gauge(
        model,
        interface,
    )

    torch.testing.assert_close(
        self_dual.producer_encoder.T @ self_dual.producer_encoder,
        torch.eye(3, dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    assert torch.equal(
        model.weight.detach(),
        self_dual.target_weight.to(torch.bfloat16),
    )
    assert receipt["global_residual_gauge_frozen"] is True
    assert receipt["independent_per_layer_gauge_rotation_allowed"] is False
    assert receipt["materialized_source_to_vocabulary_operator_relative_error"] < (0.01)
    assert receipt["training_steps"] == 0
    assert receipt["optimizer_steps"] == 0
    assert receipt["backward_calls"] == 0
    assert id(model.weight) == parameter_id
    assert int(model.weight.data_ptr()) == storage_pointer
    assert id(model.weight.grad) == gradient_id


def _terminal_evidence(split: str, tag: str):
    return postfold.Lfm2CpcfTerminalActionEvidence(
        split=split,
        row_ids=(f"{tag}-row",),
        mlp_input=torch.ones((1, 2), dtype=torch.bfloat16),
        pre_mlp_residual=torch.zeros((1, 2), dtype=torch.bfloat16),
        parent_final_norm_output=torch.ones((1, 2), dtype=torch.bfloat16),
        sample_weights=torch.ones((1,), dtype=torch.float64),
    )


class _TerminalFeedForward(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.w2 = torch.nn.Linear(
            int(weight.shape[1]),
            int(weight.shape[0]),
            bias=False,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            self.w2.weight.copy_(weight)


class _TerminalLayer(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.feed_forward = _TerminalFeedForward(weight)


class _TerminalBackbone(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TerminalLayer(weight)])


class _TerminalTarget(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.model = _TerminalBackbone(weight)
        self.other = torch.nn.Parameter(torch.tensor([3.0], dtype=weight.dtype))


def _terminal_target_state(model) -> str:
    return canonical_sha256(postfold._state_tensor_hashes(model))


def _terminal_result(
    *,
    incumbent,
    selected,
    published,
    selected_candidate_id: str,
    accepted: bool,
    validation_passed: bool,
    rollback_performed: bool,
):
    policy = terminal_action.Lfm2CpcfTerminalActionPolicy()
    body = {
        "policy_sha256": policy.authentication_sha256,
        "selected_candidate_id": selected_candidate_id,
        "accepted": accepted,
        "selected_nonincumbent": selected_candidate_id != "incumbent",
        "published_is_incumbent": torch.equal(published, incumbent),
        "rollback_performed": rollback_performed,
        "rollback_exact": True,
        "source_seed_writer_unchanged": True,
        "parameter_identity_or_topology_mutation": False,
        "parameter_count_before": int(incumbent.numel()),
        "parameter_count_after": int(published.numel()),
        "validation": {"passed": validation_passed},
        "output_state_sha256": {
            "incumbent": terminal_action._tensor_sha256(incumbent),
            "selected": terminal_action._tensor_sha256(selected),
            "published": terminal_action._tensor_sha256(published),
        },
        "operation_audit": dict(terminal_action._ZERO_OPERATION_AUDIT),
    }
    result_sha256 = terminal_action._canonical_sha256(body)
    return terminal_action.Lfm2CpcfTerminalActionResidualResult(
        policy=policy,
        incumbent_down_weight=incumbent,
        selected_down_weight=selected,
        published_down_weight=published,
        selected_candidate_id=selected_candidate_id,
        accepted=accepted,
        report={**body, "result_sha256": result_sha256},
        result_sha256=result_sha256,
    )


def _compiled_exact_zero_terminal_result(*, policy=None, qdeim_prior=False):
    generator = torch.Generator().manual_seed(1_901)
    hidden = 2
    intermediate = 3
    epsilon = 1e-5
    gate = (torch.randn(intermediate, hidden, generator=generator) * 0.55).to(
        torch.bfloat16
    )
    up = (torch.randn(intermediate, hidden, generator=generator) * 0.55).to(
        torch.bfloat16
    )
    target = torch.tensor(
        [[0.20, -0.16, 0.18], [-0.09, 0.07, -0.08]],
        dtype=torch.bfloat16,
    )
    tied_table = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0]],
        dtype=torch.bfloat16,
    )
    seed = terminal_action.Lfm2CpcfTerminalActionSeed(
        gate_weight=gate,
        up_weight=up,
        down_weight=torch.zeros((hidden, intermediate), dtype=torch.bfloat16),
        final_norm_weight=torch.ones(hidden, dtype=torch.bfloat16),
        producer_encoder=torch.eye(hidden, dtype=torch.float64),
        target_tied_weight=tied_table,
        source_tied_table=tied_table,
        norm_epsilon=epsilon,
    )

    def evidence(split, rows, random_seed):
        split_generator = torch.Generator().manual_seed(random_seed)
        mlp_input = (torch.randn(rows, hidden, generator=split_generator) * 0.75).to(
            torch.bfloat16
        )
        pre_mlp = (torch.randn(rows, hidden, generator=split_generator) * 0.30).to(
            torch.float64
        )
        pre_mlp[:, 0] = torch.where(
            torch.arange(rows) % 2 == 0,
            1.4,
            -1.4,
        )
        features = torch.nn.functional.silu(
            mlp_input.to(torch.float32) @ gate.to(torch.float32).T
        ) * (mlp_input.to(torch.float32) @ up.to(torch.float32).T)
        target_post = pre_mlp + features.to(torch.float64) @ target.to(torch.float64).T
        parent_final_norm = (
            target_post
            / (target_post.square().mean(dim=1, keepdim=True) + epsilon).sqrt()
        )
        return terminal_action.Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=tuple(f"{split}:{index}" for index in range(rows)),
            mlp_input=mlp_input,
            pre_mlp_residual=pre_mlp.to(torch.bfloat16),
            parent_final_norm_output=parent_final_norm.to(torch.bfloat16),
            sample_weights=torch.ones(rows, dtype=torch.float64),
        )

    banks = {
        "fit": evidence("fit", 32, 1),
        "fit_check": evidence("fit_check", 16, 2),
        "selection": evidence("selection", 16, 3),
        "validation": evidence("validation", 16, 4),
    }
    prior = None
    if qdeim_prior:
        recipe = postfold.resolve_lfm2_semantic_body_recipe(
            "nonlinear_qdeim_fit_corrected"
        )
        prior = terminal_action.Lfm2CpcfTerminalActionDownWeightPrior(
            down_weight=target,
            body_recipe_id=recipe.recipe_id,
            body_recipe_sha256=recipe.recipe_sha256,
            mlp_lane=recipe.component_lanes["mlp"],
            selected_role="full-w1-w3-w2",
            selected_candidate_sha256="1" * 64,
            preparation_sha256="2" * 64,
            mlp_installation_core_sha256="3" * 64,
            live_deferred_w2_sha256=(
                terminal_action.lfm2_cpcf_terminal_action_tensor_sha256(
                    seed.down_weight
                )
            ),
        )
    result = terminal_action.compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=banks["fit"],
        fit_check_evidence=banks["fit_check"],
        selection_evidence=banks["selection"],
        validation_evidence=banks["validation"],
        policy=policy,
        down_weight_prior=prior,
    )
    return seed, banks, result


def _compiled_nonzero_terminal_result_with_failed_validation():
    zero_seed, banks, _zero_result = _compiled_exact_zero_terminal_result()
    seed = terminal_action.Lfm2CpcfTerminalActionSeed(
        gate_weight=zero_seed.gate_weight,
        up_weight=zero_seed.up_weight,
        down_weight=torch.tensor(
            [[0.03, -0.02, 0.01], [-0.01, 0.04, 0.02]],
            dtype=torch.bfloat16,
        ),
        final_norm_weight=zero_seed.final_norm_weight,
        producer_encoder=zero_seed.producer_encoder,
        target_tied_weight=zero_seed.target_tied_weight,
        source_tied_table=zero_seed.source_tied_table,
        norm_epsilon=zero_seed.norm_epsilon,
    )
    policy = terminal_action.Lfm2CpcfTerminalActionPolicy(
        minimum_selection_centered_rmse_improvement_fraction=1.0,
        minimum_selection_oracle_top1_gain=1.0,
        minimum_selection_parent_top1_gain=1.0,
        minimum_selection_parent_positive_margin_gain=1.0,
        metric_regression_tolerance=0.0,
        maximum_post_layer_rms_ratio=1e-12,
        maximum_weight_delta_rms_ratio=1e-12,
    )
    result = terminal_action.compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=banks["fit"],
        fit_check_evidence=banks["fit_check"],
        selection_evidence=banks["selection"],
        validation_evidence=banks["validation"],
        policy=policy,
    )
    return seed, banks, result


def _synthetic_terminal_lineages(
    *,
    cache_key,
    cache_parent_fingerprint,
    bridge_artifact_sha256,
    compatibility_sha256,
    payload_binding_sha256,
    session_sha256,
    interface_report_sha256="7" * 64,
    direct_local=True,
    projection_sha256=(),
    projections=(),
):
    def seal(body, name):
        return {**body, name: canonical_sha256(body)}

    parent_source = None
    mapped_source = None
    parent_source_sha256 = None
    mapped_source_sha256 = None
    mapped_reconstruction = None
    mapped_reconstruction_sha256 = None
    if direct_local:
        checkpoint_files = [
            {"name": "model.safetensors", "bytes": 1, "sha256": "6" * 64}
        ]
        parent_identity = {"synthetic_parent_identity": cache_parent_fingerprint}
        parent_body = {
            "schema": "neural-forge-lfm2-local-parent-final-rmsnorm-load-v1",
            "cache_key": cache_key,
            "cache_parent_fingerprint": cache_parent_fingerprint,
            "config_file_sha256": "5" * 64,
            "checkpoint_files": checkpoint_files,
            "reconstructed_parent_identity": parent_identity,
        }
        parent_source = seal(parent_body, "result_sha256")
        parent_source_sha256 = parent_source["result_sha256"]
        mapped_body = {
            "schema": "neural-forge-lfm2-direct-local-mapped-tied-table-source-v1",
            "parent_final_norm_load_sha256": parent_source_sha256,
            "config_file": {
                "name": "config.json",
                "bytes": 1,
                "sha256": parent_body["config_file_sha256"],
            },
            "checkpoint_files": checkpoint_files,
            "independently_authenticated_parent_checkpoint_files_sha256": (
                canonical_sha256(checkpoint_files)
            ),
            "independently_authenticated_parent_identity_sha256": (
                canonical_sha256(parent_identity)
            ),
            "source_checkpoint_exactly_matches_authenticated_parent": True,
            "source_tensor_sha256": "1" * 64,
        }
        mapped_source = seal(mapped_body, "result_sha256")
        mapped_source_sha256 = mapped_source["result_sha256"]
        mapped_descriptor = {
            "dtype": "bfloat16",
            "shape": [2, 2],
            "numel": 4,
            "size_bytes": 8,
            "bfloat16_sha256": "e" * 64,
            "interface_source_sha256": "f" * 64,
        }
        reconstruction_body = {
            "schema": (
                "neural-forge-lfm2-direct-local-mapped-tied-table-reconstruction-v1"
            ),
            "cache_key": cache_key,
            "cache_parent_fingerprint": cache_parent_fingerprint,
            "parent_final_norm_load_sha256": parent_source_sha256,
            "mapped_source_report_sha256": mapped_source_sha256,
            "checkpoint_files": checkpoint_files,
            "checkpoint_files_sha256": canonical_sha256(checkpoint_files),
            "bridge_artifact_sha256": bridge_artifact_sha256,
            "bridge_transport_profile_fingerprint": "c" * 64,
            "target_to_donor_row_indices_sha256": "d" * 64,
            "source_tensor_descriptor": {"sha256": "1" * 64},
            "reconstructed_mapped_table_descriptor": mapped_descriptor,
            "supplied_mapped_table_descriptor": mapped_descriptor,
            "exact_tensor_equality_verified": True,
            "operation_audit": {
                "checkpoint_tensor_loads": 1,
                "parent_model_objects_constructed": 0,
                "parent_model_forward_calls": 0,
                "parent_model_generate_calls": 0,
                "network_access_performed": False,
                "training_steps": 0,
            },
        }
        mapped_reconstruction = seal(reconstruction_body, "result_sha256")
        mapped_reconstruction_sha256 = mapped_reconstruction["result_sha256"]
    source_body = {
        "schema": "neural-forge-lfm2-long-horizon-terminal-source-lineage-v1",
        "cache_key": cache_key,
        "cache_manifest_sha256": "a" * 64,
        "cache_parent_fingerprint": cache_parent_fingerprint,
        "cache_layer_topology_fingerprint": "b" * 64,
        "bridge_artifact_sha256": bridge_artifact_sha256,
        "bridge_transport_profile_fingerprint": "c" * 64,
        "target_to_donor_row_indices_sha256": "d" * 64,
        "parent_final_norm_authentication_sha256": "4" * 64,
        "mapped_table_bfloat16_sha256": "e" * 64,
        "mapped_table_interface_source_sha256": "f" * 64,
        "interface_report_sha256": interface_report_sha256,
        "compiler_source_binding_sha256": "8" * 64,
        "parent_final_norm_direct_local_source_sha256": parent_source_sha256,
        "direct_local_mapped_source_report_sha256": mapped_source_sha256,
        "direct_local_mapped_reconstruction_sha256": (mapped_reconstruction_sha256),
        "parent_final_norm_direct_local_source": parent_source,
        "direct_local_mapped_source_report": mapped_source,
        "direct_local_mapped_reconstruction": mapped_reconstruction,
        "direct_local_checkpoint_source_authenticated": direct_local,
        "behavioral_tolerance_used_for_source_admission": False,
        "exact_lineage_is_source_admission_authority": True,
    }
    source = postfold.Lfm2LongHorizonTerminalSourceLineage(
        seal(source_body, "authentication_sha256")
    )
    compiler_inputs = {
        "source_lineage": source.as_dict(),
        "source_lineage_sha256": source.authentication_sha256,
        "cache_key": cache_key,
        "cache_compatibility_sha256": compatibility_sha256,
        "cache_payload_binding_sha256": payload_binding_sha256,
        "cache_session_sha256": session_sha256,
        "parent_action_projection_sha256": list(projection_sha256),
        "parent_action_projections": list(projections),
    }
    composite_body = {
        "schema": "neural-forge-lfm2-long-horizon-terminal-composite-lineage-v1",
        "compiler_inputs": compiler_inputs,
        "compiler_binding_sha256": canonical_sha256(compiler_inputs),
        "numeric_projection_diagnostics_are_admission_authority": False,
        "exact_lineage_is_compiler_admission_authority": True,
    }
    composite = postfold.Lfm2LongHorizonTerminalCompositeLineage(
        seal(composite_body, "authentication_sha256")
    )
    return source, composite


def _frontier_export_context(terminal, *, resume_mode=False, body_recipe=None):
    session_sha256 = "9" * 64
    prior_bound = (
        terminal.report.get("down_weight_prior_authentication_sha256") is not None
    )
    recipe = postfold.resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_fit_corrected" if prior_bound else "fit_guided"
    )

    def seal(body, name):
        return {**body, name: canonical_sha256(body)}

    compatibility = seal(
        {
            "schema": postfold.LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA,
            "cache_key": "1" * 64,
        },
        "authentication_sha256",
    )
    payload_binding = seal(
        {
            "compatibility_authentication_sha256": compatibility[
                "authentication_sha256"
            ],
            "payload_session_sha256": session_sha256,
        },
        "authentication_sha256",
    )
    interface_authentication = seal(
        {
            "schema": "neural-forge-lfm2-fused-terminal-tied-interface-v1",
            "interface_report_sha256": "7" * 64,
        },
        "result_sha256",
    )
    _source_lineage, composite_lineage = _synthetic_terminal_lineages(
        cache_key="1" * 64,
        cache_parent_fingerprint="2" * 64,
        bridge_artifact_sha256="3" * 64,
        compatibility_sha256=compatibility["authentication_sha256"],
        payload_binding_sha256=payload_binding["authentication_sha256"],
        session_sha256=session_sha256,
        direct_local=False,
    )
    layer_receipt = {
        "result_sha256": "5" * 64,
        "cache_selective_session_sha256": session_sha256,
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
    }
    if prior_bound:
        layer_receipt.update(
            {
                "final_layer_mlp_down_weight_prior_authentication_sha256": (
                    terminal.report["down_weight_prior_authentication_sha256"]
                ),
                "final_layer_mlp_down_weight_prior_tensor_sha256": (
                    terminal.report["down_weight_prior_tensor_sha256"]
                ),
            }
        )
    return postfold._build_terminal_frontier_export_context(
        terminal=terminal,
        cache=SimpleNamespace(cache_key="1" * 64, parent_fingerprint="2" * 64),
        bridge=SimpleNamespace(artifact_sha256="3" * 64),
        parent_final_norm=SimpleNamespace(result_sha256="4" * 64),
        mapped_parent_tied_table=torch.eye(2, dtype=torch.bfloat16),
        layer_receipt=layer_receipt,
        terminal_interface_authentication=interface_authentication,
        terminal_cache_compatibility=compatibility,
        terminal_cache_payload_binding=payload_binding,
        terminal_composite_lineage=composite_lineage,
        terminal_cache_session_sha256=session_sha256,
        same_live_cache_session_as_layer_stack=not resume_mode,
        cache_content_identity_matches_layer_stack=True,
        preterminal_resume_mode=resume_mode,
        body_recipe=body_recipe,
    )


def _installation_publication_decision(weight):
    semantic_sha256 = postfold._tensor_sha256(weight)
    terminal_action_sha256 = terminal_action._tensor_sha256(weight)
    return {
        "terminal_action_result_sha256": "a" * 64,
        "terminal_action_selected_w2_sha256": terminal_action_sha256,
        "terminal_action_published_w2_sha256": terminal_action_sha256,
        "semantic_selected_w2_sha256": semantic_sha256,
        "semantic_published_w2_sha256": semantic_sha256,
        "selected_w2_sha256": semantic_sha256,
        "published_w2_sha256": semantic_sha256,
    }


def test_final_w2_incumbent_retention_is_an_exact_zero_copy_decision(
    monkeypatch,
) -> None:
    target = _TerminalTarget(torch.eye(2, dtype=torch.bfloat16))
    parameter = target.model.layers[-1].feed_forward.w2.weight
    parameter.grad = torch.full_like(parameter, 0.25)
    target.other.grad = torch.full_like(target.other, 0.5)
    before_state = _terminal_target_state(target)
    before_parameter_id = id(parameter)
    before_parameter_ptr = int(parameter.data_ptr())
    before_gradient_id = id(parameter.grad)
    before_gradient_ptr = int(parameter.grad.data_ptr())
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_terminal_target_state(model)),
    )

    receipt = postfold._install_final_w2_once(
        target,
        published_down_weight=parameter.detach().cpu().clone(),
        terminal_publication_decision=_installation_publication_decision(parameter),
    )

    assert receipt["installed"] is False
    assert receipt["install_count"] == 0
    assert receipt["mutation_count"] == 0
    assert receipt["incumbent_retention_count"] == 1
    assert receipt["exact_zero_copy_noop"] is True
    assert receipt["parameter_copy_calls"] == 0
    assert receipt["model_state_before_sha256"] == before_state
    assert receipt["model_state_after_sha256"] == before_state
    assert receipt["changed_state_keys"] == []
    assert receipt["parameter_topology_preserved"] is True
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["published_w2_structurally_nonzero"] is True
    assert receipt["published_w2_structural_coverage"]["structurally_nonzero"] is True
    assert id(parameter) == before_parameter_id
    assert int(parameter.data_ptr()) == before_parameter_ptr
    assert id(parameter.grad) == before_gradient_id
    assert int(parameter.grad.data_ptr()) == before_gradient_ptr


def test_final_w2_changed_publication_mutates_only_the_writer(monkeypatch) -> None:
    target = _TerminalTarget(torch.eye(2, dtype=torch.bfloat16))
    parameter = target.model.layers[-1].feed_forward.w2.weight
    target.other.grad = torch.full_like(target.other, 0.5)
    before_other = target.other.detach().clone()
    published = torch.full_like(parameter.detach().cpu(), 0.75)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_terminal_target_state(model)),
    )

    receipt = postfold._install_final_w2_once(
        target,
        published_down_weight=published,
        terminal_publication_decision=_installation_publication_decision(published),
    )

    assert receipt["installed"] is True
    assert receipt["install_count"] == 1
    assert receipt["mutation_count"] == 1
    assert receipt["incumbent_retention_count"] == 0
    assert receipt["parameter_copy_calls"] == 1
    assert receipt["changed_state_keys"] == ["model.layers.0.feed_forward.w2.weight"]
    assert receipt["model_state_before_sha256"] != receipt["model_state_after_sha256"]
    torch.testing.assert_close(parameter.detach().cpu(), published, rtol=0, atol=0)
    torch.testing.assert_close(target.other.detach(), before_other, rtol=0, atol=0)
    assert receipt["parameter_topology_preserved"] is True
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["published_w2_structurally_nonzero"] is True


def test_exact_zero_structural_frontier_waits_for_whole_response_review(
    monkeypatch,
) -> None:
    seed, evidence, terminal = _compiled_exact_zero_terminal_result()
    target = _TerminalTarget(seed.down_weight)
    parameter = target.model.layers[-1].feed_forward.w2.weight
    before = parameter.detach().clone()
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_terminal_target_state(model)),
    )

    assert torch.count_nonzero(seed.down_weight).item() == 0
    assert terminal.accepted is False
    assert terminal.changed is False
    assert terminal.selected_candidate_id == "incumbent"
    assert terminal.report["final_selection_status"] == (
        "deferred-to-manual-whole-response-review"
    )
    assert terminal.report["diagnostic_order_candidate_count"] == 36
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="review-pending",
    ):
        postfold._terminal_publication_decision(
            terminal,
            expected_seed=seed,
            expected_evidence=evidence,
        )
    torch.testing.assert_close(parameter.detach(), before, rtol=0, atol=0)


def test_exact_zero_proxy_gate_miss_does_not_force_a_publication() -> None:
    policy = terminal_action.Lfm2CpcfTerminalActionPolicy(
        maximum_post_layer_rms_ratio=1e-12,
    )
    seed, evidence, terminal = _compiled_exact_zero_terminal_result(policy=policy)
    report = terminal.as_dict()
    diagnostic_lead = next(
        value
        for value in report["frontier"]
        if value["candidate_id"] == report["diagnostic_lead_candidate_id"]
    )

    assert diagnostic_lead["structurally_eligible"] is True
    assert diagnostic_lead["fit_metric_gates_passed"] is False
    assert diagnostic_lead["fit_gates"]["post_layer_norm"] is False
    assert report["fit_metrics_used_for_diagnostic_ordering"] is True
    assert report["fit_metrics_used_for_pre_review_candidate_selection"] is False
    assert terminal.accepted is False
    assert torch.equal(terminal.published_down_weight, terminal.incumbent_down_weight)
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="review-pending",
    ):
        postfold._terminal_publication_decision(
            terminal,
            expected_seed=seed,
            expected_evidence=evidence,
        )


def test_exact_zero_publication_rejects_authenticated_then_mutated_dense_forgery() -> (
    None
):
    seed, evidence, terminal = _compiled_exact_zero_terminal_result()
    diagnostic_lead_id = terminal.report["diagnostic_lead_candidate_id"]
    forged_weight = terminal.frontier_down_weights[diagnostic_lead_id].clone()
    forged_weight[0, 0] = forged_weight[0, 0] + 1.0
    forged = terminal_action.Lfm2CpcfTerminalActionResidualResult(
        policy=terminal.policy,
        incumbent_down_weight=terminal.incumbent_down_weight,
        selected_down_weight=terminal.selected_down_weight,
        published_down_weight=terminal.published_down_weight,
        frontier_down_weights=terminal.frontier_down_weights,
        selected_candidate_id=terminal.selected_candidate_id,
        accepted=terminal.accepted,
        report=terminal.as_dict(),
        result_sha256=terminal.result_sha256,
        _compiler_input_binding=terminal._compiler_input_binding,
        _compiler_issuer=terminal_action._TERMINAL_RESULT_COMPILER_ISSUER,
        _compiler_lineage_receipt=terminal._compiler_lineage_token,
    )
    assert forged.verify_authentication() is True
    object.__setattr__(
        forged,
        "frontier_down_weights",
        {
            **terminal.frontier_down_weights,
            diagnostic_lead_id: forged_weight,
        },
    )
    assert forged.verify_authentication() is False

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="changed after authentication",
    ):
        postfold._terminal_publication_decision(
            forged,
            expected_seed=seed,
            expected_evidence=evidence,
        )


@pytest.mark.parametrize(
    "published",
    (
        torch.zeros((2, 2), dtype=torch.bfloat16),
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16),
    ),
)
def test_final_w2_refuses_blank_or_structurally_partial_publication(
    monkeypatch,
    published,
) -> None:
    target = _TerminalTarget(torch.eye(2, dtype=torch.bfloat16))
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_terminal_target_state(model)),
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="blank or structurally partial W2",
    ):
        postfold._install_final_w2_once(
            target,
            published_down_weight=published,
            terminal_publication_decision=_installation_publication_decision(published),
        )

    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_legacy_validation_rollback_to_incumbent_is_rejected() -> None:
    incumbent = torch.eye(2, dtype=torch.bfloat16)
    selected = torch.ones((2, 2), dtype=torch.bfloat16)
    terminal = _terminal_result(
        incumbent=incumbent,
        selected=selected,
        published=incumbent.clone(),
        selected_candidate_id="gn-00-ridge-00-alpha-01",
        accepted=False,
        validation_passed=False,
        rollback_performed=True,
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="published neither",
    ):
        postfold._terminal_publication_decision(terminal)


def test_nonzero_failed_validation_remains_diagnostic_without_publication() -> None:
    _seed, _evidence, terminal = (
        _compiled_nonzero_terminal_result_with_failed_validation()
    )
    report = terminal.as_dict()

    assert report["validation"]["passed"] is False
    assert report["validation"]["heldout_diagnostic_only"] is True
    assert report["validation"]["publication_gate_applied"] is False
    assert report["validation"]["used_for_final_selection"] is False
    assert report["validation"]["used_for_candidate_acceptance"] is False
    assert report["validation_publication_veto_applied"] is False
    assert report["rollback_authorized"] is False
    assert report["rollback_performed"] is False
    assert terminal.accepted is False
    assert terminal.changed is False
    assert torch.equal(
        terminal.published_down_weight,
        terminal.incumbent_down_weight,
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="review-pending",
    ):
        postfold._terminal_publication_decision(terminal)


def test_exact_zero_writer_cannot_use_legacy_validation_rollback() -> None:
    incumbent = torch.zeros((2, 2), dtype=torch.bfloat16)
    selected = torch.ones((2, 2), dtype=torch.bfloat16)
    terminal = _terminal_result(
        incumbent=incumbent,
        selected=selected,
        published=incumbent.clone(),
        selected_candidate_id="gn-00-ridge-00-alpha-01",
        accepted=False,
        validation_passed=False,
        rollback_performed=True,
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="published neither",
    ):
        postfold._terminal_publication_decision(terminal)

    forged_body = terminal.as_dict()
    forged_body.pop("result_sha256")
    forged_body["exact_zero_cold_start"] = True
    forged_sha256 = terminal_action._canonical_sha256(forged_body)
    forged = terminal_action.Lfm2CpcfTerminalActionResidualResult(
        policy=terminal.policy,
        incumbent_down_weight=terminal.incumbent_down_weight,
        selected_down_weight=terminal.selected_down_weight,
        published_down_weight=terminal.published_down_weight,
        selected_candidate_id=terminal.selected_candidate_id,
        accepted=terminal.accepted,
        report={**forged_body, "result_sha256": forged_sha256},
        result_sha256=forged_sha256,
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="published neither",
    ):
        postfold._terminal_publication_decision(forged)


def test_fused_terminal_frontier_path_defers_installation_until_manual_review() -> None:
    source = inspect.getsource(
        postfold.compile_install_lfm2_fused_semantic_terminal_refold
    )

    assert (
        '"final_selection_status": "deferred-to-manual-whole-response-review"' in source
    )
    assert '"terminal_writer_install_count": 0' in source
    assert '"candidate_published_before_manual_review": False' in source
    assert "published_down_weight=published_down_weight" in source


@pytest.mark.parametrize("collision", ("same-output", "nested-source"))
def test_preterminal_capsule_path_guard_rejects_overlap_before_loading(
    tmp_path: Path,
    collision: str,
) -> None:
    sources = {
        name: tmp_path / name
        for name in ("input", "cache", "bridge", "parent", "statistics")
    }
    for path in sources.values():
        path.mkdir()
    diagnostic = tmp_path / "compiler.json"
    diagnostic.write_text("{}", encoding="ascii")
    output = tmp_path / "output"
    capsule = output if collision == "same-output" else sources["input"] / "capsule"

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="disjoint"):
        postfold._validate_preterminal_capsule_path_disjointness(
            output_checkpoint_directory=output,
            preterminal_capsule_directory=capsule,
            input_checkpoint_directory=sources["input"],
            cache_directory=sources["cache"],
            bridge_directory=sources["bridge"],
            parent_model_directory=sources["parent"],
            compiler_diagnostic_json=diagnostic,
            interface_statistics_directory=sources["statistics"],
            interface_frontier_companion_json=None,
            donor_tokenizer=SimpleNamespace(),
            target_tokenizer=SimpleNamespace(),
        )


def test_preterminal_capsule_path_guard_ignores_blank_tokenizer_path_sentinels(
    tmp_path: Path,
) -> None:
    sources = {
        name: tmp_path / name
        for name in ("input", "cache", "bridge", "parent", "statistics")
    }
    for path in sources.values():
        path.mkdir()
    diagnostic = tmp_path / "compiler.json"
    diagnostic.write_text("{}", encoding="ascii")

    output, capsule = postfold._validate_preterminal_capsule_path_disjointness(
        output_checkpoint_directory=tmp_path / "run" / "checkpoint",
        preterminal_capsule_directory=tmp_path / "run" / "preterminal-capsule",
        input_checkpoint_directory=sources["input"],
        cache_directory=sources["cache"],
        bridge_directory=sources["bridge"],
        parent_model_directory=sources["parent"],
        compiler_diagnostic_json=diagnostic,
        interface_statistics_directory=sources["statistics"],
        interface_frontier_companion_json=None,
        donor_tokenizer=SimpleNamespace(
            name_or_path="",
            tokenizer_file="   ",
            init_kwargs={"vocab_file": ""},
        ),
        target_tokenizer=SimpleNamespace(
            name_or_path="",
            init_kwargs={"name_or_path": "   "},
        ),
    )

    assert output == (tmp_path / "run" / "checkpoint").resolve()
    assert capsule == (tmp_path / "run" / "preterminal-capsule").resolve()


def test_preterminal_capsule_path_guard_still_protects_real_tokenizer_source(
    tmp_path: Path,
) -> None:
    sources = {
        name: tmp_path / name
        for name in ("input", "cache", "bridge", "parent", "statistics", "tokenizer")
    }
    for path in sources.values():
        path.mkdir()
    diagnostic = tmp_path / "compiler.json"
    diagnostic.write_text("{}", encoding="ascii")

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="local tokenizer source",
    ):
        postfold._validate_preterminal_capsule_path_disjointness(
            output_checkpoint_directory=tmp_path / "output",
            preterminal_capsule_directory=sources["tokenizer"] / "capsule",
            input_checkpoint_directory=sources["input"],
            cache_directory=sources["cache"],
            bridge_directory=sources["bridge"],
            parent_model_directory=sources["parent"],
            compiler_diagnostic_json=diagnostic,
            interface_statistics_directory=sources["statistics"],
            interface_frontier_companion_json=None,
            donor_tokenizer=SimpleNamespace(
                name_or_path=str(sources["tokenizer"]),
                init_kwargs={"name_or_path": ""},
            ),
            target_tokenizer=SimpleNamespace(name_or_path=""),
        )


def test_terminal_frontier_export_seam_receives_full_bank_once_before_w2_install() -> (
    None
):
    _seed, _evidence, terminal = _compiled_exact_zero_terminal_result()
    live_final_w2 = torch.zeros_like(terminal.incumbent_down_weight)
    context = _frontier_export_context(terminal)
    calls = []

    def exporter(result, export_context):
        calls.append(result.result_sha256)
        assert result is terminal
        assert export_context == context
        assert export_context["terminal_seed_sha256"] == result.report["seed_sha256"]
        assert (
            export_context["terminal_incumbent_down_weight_terminal_action_sha256"]
            == result.report["incumbent_seed_writer_sha256"]
        )
        assert result.verify_authentication() is True
        assert len(result.frontier_down_weights) == 36
        assert int(torch.count_nonzero(live_final_w2).item()) == 0
        frontier_records = {
            row["candidate_id"]: row
            for row in result.report["frontier"]
            if row["candidate_id"] != "incumbent"
        }
        assert set(frontier_records) == set(result.frontier_down_weights)
        for candidate_id, weight in result.frontier_down_weights.items():
            assert (
                terminal_action._tensor_sha256(weight)
                == frontier_records[candidate_id]["state_sha256"]
            )
        return {"receipt_sha256": "a" * 64, "bank_count": 36}

    reference = postfold._export_authenticated_terminal_frontier(
        terminal,
        context,
        exporter,
    )
    fields = postfold._terminal_frontier_export_receipt_fields(reference, context)

    assert calls == [terminal.result_sha256]
    assert fields["terminal_frontier_export"] == reference
    assert fields["terminal_frontier_export_sha256"] == canonical_sha256(reference)
    assert fields["terminal_frontier_export_context"] == dict(context)
    assert (
        fields["terminal_frontier_export_context_sha256"] == context["context_sha256"]
    )
    assert fields["terminal_frontier_exported_before_w2_installation"] is True
    source = inspect.getsource(
        postfold.compile_install_lfm2_fused_semantic_terminal_refold
    )
    assert source.index("_export_authenticated_terminal_frontier") < source.index(
        "_install_final_w2_once"
    )


def test_qdeim_prior_is_bound_into_compiler_issued_frontier_export_context() -> None:
    seed, _evidence, terminal = _compiled_exact_zero_terminal_result(qdeim_prior=True)
    context = _frontier_export_context(terminal)

    assert torch.count_nonzero(seed.down_weight).item() == 0
    assert terminal.report["frontier_count"] == 38
    assert "qdeim-prior-alpha-00" in terminal.frontier_down_weights
    assert context["body_recipe_id"] == "nonlinear_qdeim_fit_corrected"
    assert (
        context["down_weight_prior_authentication_sha256"]
        == terminal.report["down_weight_prior_authentication_sha256"]
    )
    assert (
        context["down_weight_prior_tensor_sha256"]
        == terminal.report["down_weight_prior_tensor_sha256"]
    )
    assert context["down_weight_prior_bound_from_layer_stack_into_terminal"] is True
    assert context["pure_down_weight_prior_frontier_candidate_id"] == (
        "qdeim-prior-alpha-00"
    )
    assert (
        postfold._authenticate_terminal_frontier_export_context(terminal, context)
        is context
    )
    source = inspect.getsource(
        postfold.compile_install_lfm2_fused_semantic_terminal_refold
    )
    assert "down_weight_prior=layer_stack.final_layer_mlp_down_weight_prior" in source


def test_terminal_export_context_rejects_explicit_wrong_body_recipe() -> None:
    _seed, _evidence, terminal = _compiled_exact_zero_terminal_result()

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="did not bind the authenticated semantic body recipe",
    ):
        _frontier_export_context(
            terminal,
            resume_mode=True,
            body_recipe="residual_writer",
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "mapped_parent_tied_table_sha256",
        "tied_interface_authentication_sha256",
        "terminal_cache_compatibility_sha256",
        "terminal_cache_payload_binding_sha256",
        "body_recipe_sha256",
    ),
)
def test_terminal_frontier_export_context_rejects_resealed_live_fact_relabel(
    field_name,
) -> None:
    _seed, _evidence, terminal = _compiled_exact_zero_terminal_result()
    issued = _frontier_export_context(terminal, resume_mode=True)
    relabeled = dict(issued)
    relabeled[field_name] = hashlib.sha256(field_name.encode("ascii")).hexdigest()
    relabeled.pop("context_sha256")
    relabeled["context_sha256"] = canonical_sha256(relabeled)

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="not compiler-issued",
    ):
        postfold._authenticate_terminal_frontier_export_context(
            terminal,
            relabeled,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "mapped_parent_tied_table_sha256",
        "tied_interface_authentication_sha256",
        "terminal_cache_compatibility_sha256",
        "terminal_cache_payload_binding_sha256",
        "body_recipe_sha256",
    ),
)
def test_terminal_frontier_exporter_cannot_reassign_issued_context_state(
    field_name,
) -> None:
    _seed, _evidence, terminal = _compiled_exact_zero_terminal_result()
    issued = _frontier_export_context(terminal, resume_mode=True)
    relabeled = dict(issued)
    relabeled[field_name] = hashlib.sha256(
        f"exporter:{field_name}".encode("ascii")
    ).hexdigest()
    relabeled.pop("context_sha256")
    relabeled["context_sha256"] = canonical_sha256(relabeled)

    def mutate_context(_result, export_context):
        export_context._values = MappingProxyType(relabeled)
        pytest.fail("frozen compiler-issued context accepted reassignment")

    with pytest.raises(AttributeError, match="frozen"):
        postfold._export_authenticated_terminal_frontier(
            terminal,
            issued,
            mutate_context,
        )
    with pytest.raises(AttributeError, match="frozen"):
        issued._issued_fact_tuple = tuple(
            (name, relabeled[name])
            for name in postfold._TERMINAL_FRONTIER_LIVE_FACT_NAMES
        )
    with pytest.raises(AttributeError, match="frozen"):
        issued._issued_facts = MappingProxyType(
            {
                name: relabeled[name]
                for name in postfold._TERMINAL_FRONTIER_LIVE_FACT_NAMES
            }
        )
    assert (
        postfold._authenticate_terminal_frontier_export_context(
            terminal,
            issued,
        )
        is issued
    )


def test_terminal_frontier_exporter_exception_or_mutation_precedes_w2_install() -> None:
    _seed, _evidence, terminal = _compiled_exact_zero_terminal_result()
    live_final_w2 = torch.zeros_like(terminal.incumbent_down_weight)
    context = _frontier_export_context(terminal)

    def fail_export(_result, _context):
        assert int(torch.count_nonzero(live_final_w2).item()) == 0
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        postfold._export_authenticated_terminal_frontier(
            terminal,
            context,
            fail_export,
        )
    assert int(torch.count_nonzero(live_final_w2).item()) == 0

    _seed, _evidence, mutation_terminal = _compiled_exact_zero_terminal_result()
    mutation_context = _frontier_export_context(mutation_terminal, resume_mode=True)

    def mutate_export(result, export_context):
        assert export_context["preterminal_resume_mode"] is True
        assert export_context["same_live_cache_session_as_layer_stack"] is False
        candidate_id = next(iter(result.frontier_down_weights))
        result.frontier_down_weights[candidate_id].reshape(-1)[0] += torch.tensor(
            1.0,
            dtype=torch.bfloat16,
        )
        return {"receipt_sha256": "b" * 64}

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="mutated",
    ):
        postfold._export_authenticated_terminal_frontier(
            mutation_terminal,
            mutation_context,
            mutate_export,
        )
    assert int(torch.count_nonzero(live_final_w2).item()) == 0


def test_nonzero_heldout_regression_is_a_sealed_diagnostic_only(
    monkeypatch,
) -> None:
    class Metrics:
        def __init__(self, name: str) -> None:
            self.name = name

        def as_dict(self) -> dict[str, object]:
            return {"name": self.name, "finite": True}

    monkeypatch.setattr(
        postfold,
        "_metric_gates",
        lambda *_args, **_kwargs: {"heldout_not_regressed": False},
    )
    report = postfold._terminal_heldout_diagnostic(
        Metrics("published"),
        Metrics("nonzero-incumbent"),
        terminal_action.Lfm2CpcfTerminalActionPolicy(),
    )

    assert report["passed"] is False
    assert report["role"] == (
        "diagnostic-only-after-frozen-terminal-publication-decision"
    )
    assert report["publication_gate_applied"] is False
    body = dict(report)
    supplied = body.pop("diagnostic_sha256")
    assert supplied == canonical_sha256(body)
    source = inspect.getsource(
        postfold.compile_install_lfm2_fused_semantic_terminal_refold
    )
    assert "separate-heldout-lane-regression" not in source


def test_heldout_selected_incumbent_is_not_a_terminal_publication() -> None:
    incumbent = torch.eye(2, dtype=torch.bfloat16)
    terminal = _terminal_result(
        incumbent=incumbent,
        selected=incumbent.clone(),
        published=incumbent.clone(),
        selected_candidate_id="incumbent",
        accepted=False,
        validation_passed=True,
        rollback_performed=False,
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="published neither",
    ):
        postfold._terminal_publication_decision(terminal)


def test_reconstructs_saved_mean_preserving_interface_without_parent_model() -> None:
    source = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, -1.0, 1.0],
            [0.5, 4.0, -2.0],
            [-1.0, 0.25, 2.0],
        ],
        dtype=torch.bfloat16,
    )
    target = torch.tensor(
        [
            [0.5, 1.0],
            [0.5, -1.0],
            [0.5, 0.5],
            [0.5, -0.5],
        ],
        dtype=torch.bfloat16,
    )
    model = _TiedModel(target)

    interface = postfold.reconstruct_lfm2_saved_tied_interface(model, source)

    assert interface.target_width == 2
    assert interface.source_width == 3
    assert interface.vocabulary_size == 4
    assert interface.report["checkpoint_target_bytes_are_authoritative"] is True
    assert interface.report["parent_model_calls"] == 0
    body = dict(interface.report)
    seal = body.pop("report_sha256")
    assert seal == canonical_sha256(body)
    torch.testing.assert_close(
        interface.target_weight.to(torch.bfloat16),
        model.weight.detach(),
        rtol=0,
        atol=0,
    )


def _seal(value, field):
    return {**value, field: canonical_sha256(value)}


def test_compiler_lineage_accepts_distinct_semantic_cache_key(monkeypatch) -> None:
    compiler_key = "1" * 64
    semantic_key = "2" * 64
    parent = "3" * 64
    bridge_sha = "4" * 64
    target_state = "5" * 64
    artifact = {
        "query_groups": [[0]],
        "query_barycentric_weights": [[1.0]],
    }
    direct = _seal(
        {
            "candidate_id": "direct_monolithic",
            "operator_artifact": artifact,
            "operator_artifact_sha256": canonical_sha256(artifact),
            "selection_context_metrics": {
                "query_groups_sha256": canonical_sha256(
                    {"groups": [[0]], "weights": [[1.0]]}
                )
            },
        },
        "report_sha256",
    )
    endpoint = _seal(
        {
            "source_inventory": {"fixture": True},
            "source_inventory_sha256": "6" * 64,
        },
        "artifact_sha256",
    )
    layer = _seal(
        {
            "layer_index": 2,
            "operator_kind": "full_attention",
            "solver_evidence_policy": {
                "attention_candidate_evaluations": {
                    "direct_monolithic": direct,
                },
                "endpoint_residual_attention_artifact": endpoint,
                "endpoint_residual_attention_artifact_sha256": endpoint[
                    "artifact_sha256"
                ],
            },
        },
        "result_sha256",
    )
    compiled = _seal({"fixture": "compiled"}, "result_sha256")
    diagnostic = {
        "receipt_sha256": "7" * 64,
        "inputs": {
            "cache_key": compiler_key,
            "parent_fingerprint": parent,
            "bridge_artifact_sha256": bridge_sha,
        },
        "compiled_candidate_consumer_seam": {
            "cache_sha256": compiler_key,
            "parent_sha256": parent,
            "bridge_sha256": bridge_sha,
            "target_state_sha256": target_state,
        },
        "compiled_model_fold": compiled,
        "layer_results": [layer],
    }
    inventory = SimpleNamespace(inventory_sha256="6" * 64)
    monkeypatch.setattr(
        postfold,
        "_authenticated_c61_diagnostic",
        lambda *_a, **_k: (MappingProxyType(diagnostic), "8" * 64),
    )
    monkeypatch.setattr(postfold, "_typed_qkvo_inventory", lambda _value: inventory)
    monkeypatch.setattr(
        postfold,
        "Lfm2SemanticCompilerLayerLineage",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    result = postfold.load_lfm2_semantic_compiler_lineage(
        "unused.json",
        expected_diagnostic_sha256=diagnostic["receipt_sha256"],
        cache=SimpleNamespace(cache_key=semantic_key, parent_fingerprint=parent),
        bridge=SimpleNamespace(artifact_sha256=bridge_sha),
        input_metadata={"model_state_sha256": target_state},
    )

    assert result[2].report["compiler_cache_key"] == compiler_key
    assert result[2].report["semantic_evidence_cache_key"] == semantic_key
    assert (
        result[2].report["compiler_and_semantic_cache_identities_bound_separately"]
        is True
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="not the compiler ancestry",
    ):
        postfold.load_lfm2_semantic_compiler_lineage(
            "unused.json",
            expected_diagnostic_sha256=diagnostic["receipt_sha256"],
            cache=SimpleNamespace(
                cache_key=semantic_key,
                parent_fingerprint="f" * 64,
            ),
            bridge=SimpleNamespace(artifact_sha256=bridge_sha),
            input_metadata={"model_state_sha256": target_state},
        )


def test_anchor_active_interface_rehydrates_exact_producer_encoder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compiler_key = "1" * 64
    semantic_key = "2" * 64
    parent = "3" * 64
    bridge_sha = "4" * 64
    final_norm_sha = "5" * 64
    source = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, -1.0, 1.0], [0.5, 4.0, -2.0]],
        dtype=torch.bfloat16,
    )
    source64 = source.to(torch.float64)
    encoder = torch.tensor(
        [[0.2, 0.1], [0.3, -0.4], [0.5, 0.2]],
        dtype=torch.float64,
    )
    base_target = torch.tensor(
        [[0.5, 1.0], [0.5, -1.0], [0.5, 0.25]],
        dtype=torch.float64,
    )
    # Anchor repair intentionally destroys the constant first column.  An
    # inverse of the saved table is therefore invalid, while the producer map
    # remains exactly the base compiler's map.
    active_target = base_target + torch.tensor(
        [[0.125, 0.0], [0.0, 0.25], [-0.125, 0.0]],
        dtype=torch.float64,
    )
    base_report = _seal(
        {
            "source_table_sha256": postfold._interface_tensor_sha256(source64),
            "target_weight_sha256": postfold._interface_tensor_sha256(base_target),
            "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
        },
        "report_sha256",
    )
    anchor_report = _seal(
        {
            "base_interface_report_sha256": base_report["report_sha256"],
            "source_tied_table_sha256": postfold._interface_tensor_sha256(source64),
            "target_weight_sha256": postfold._interface_tensor_sha256(active_target),
            "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
            "selected_candidate_is_base_control": False,
            "active_anchor_residual_selected": True,
        },
        "report_sha256",
    )
    interface = postfold.TiedCausalInterfaceFold(
        target_weight=active_target,
        producer_encoder=encoder,
        report=anchor_report,
    )
    base_interface = postfold.TiedCausalInterfaceFold(
        target_weight=base_target,
        producer_encoder=encoder,
        report=base_report,
    )
    model = _TiedModel(active_target.to(torch.bfloat16))
    model.config = SimpleNamespace(hidden_size=2)

    store = tmp_path / "catalog-store"
    artifact = store / "artifact"
    artifact.mkdir(parents=True)
    compiler_cache = tmp_path / "compiler-cache"
    compiler_cache.mkdir()
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text("[]", encoding="ascii")
    trajectory_sha = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    preparation_path = tmp_path / "catalog-receipt.json"
    preparation_body = {
        "catalog_receipt_json": str(preparation_path.resolve()),
        "catalog_store_directory": str(store.resolve()),
        "trace_cache_directory": str(compiler_cache.resolve()),
        "artifact_path": str(artifact.resolve()),
        "cache_key": compiler_key,
        "binding_sha256": "6" * 64,
        "artifact_manifest_sha256": "7" * 64,
        "artifact_catalog_sha256": "8" * 64,
        "selective_catalog_sha256": "9" * 64,
        "selective_source_sha256": "a" * 64,
        "selective_provider_sha256": "b" * 64,
    }
    preparation = _seal(preparation_body, "preparation_receipt_sha256")
    preparation_path.write_bytes(
        (
            json.dumps(
                preparation,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )
    lineage = {
        "base_interface_report": base_report,
        "causal_anchor_report": anchor_report,
        "base_interface_report_sha256": base_report["report_sha256"],
        "causal_anchor_report_sha256": anchor_report["report_sha256"],
        "active_interface_report_sha256": anchor_report["report_sha256"],
        "active_interface_is_anchor_repaired": True,
        "active_interface_is_base_control": False,
        "producer_encoder_fixed_by_anchor_selection": True,
    }
    bundle = _seal(
        {
            "interface_report_sha256": anchor_report["report_sha256"],
            "base_interface_report_sha256": base_report["report_sha256"],
            "causal_anchor_report_sha256": anchor_report["report_sha256"],
            "interface_lineage": lineage,
        },
        "bundle_sha256",
    )
    compiled = _seal(
        {
            "selective_catalog_sha256": preparation["selective_catalog_sha256"],
            "selective_source_sha256": preparation["selective_source_sha256"],
            "selective_provider_sha256": preparation["selective_provider_sha256"],
            "final_norm_authentication_sha256": final_norm_sha,
            "boundary_bundle_sha256": bundle["bundle_sha256"],
        },
        "result_sha256",
    )
    install = _seal(
        {
            "interface_report_sha256": anchor_report["report_sha256"],
            "target_weight_sha256": postfold.cpcf_model_compiler._tensor_sha256(
                model.weight.detach()
            ),
        },
        "report_sha256",
    )
    inputs = {
        "cache_key": compiler_key,
        "parent_fingerprint": parent,
        "bridge_artifact_sha256": bridge_sha,
        "trace_cache_directory": str(compiler_cache.resolve()),
        "trajectory_corpus": str(trajectory.resolve()),
        "trajectory_corpus_sha256": trajectory_sha,
        "trace_adapter_sha256": "c" * 64,
        "selective_catalog_preparation_receipt_sha256": preparation[
            "preparation_receipt_sha256"
        ],
        "selective_catalog_binding_sha256": preparation["binding_sha256"],
        "selective_catalog_artifact_manifest_sha256": preparation[
            "artifact_manifest_sha256"
        ],
        "selective_catalog_artifact_sha256": preparation["artifact_catalog_sha256"],
    }
    diagnostic = {
        "receipt_sha256": "d" * 64,
        "inputs": inputs,
        "request": {
            "selective_catalog_store_directory": str(store.resolve()),
        },
        "selective_catalog_artifact": preparation,
        "compiled_model_fold": compiled,
        "boundary_bundle": bundle,
        "interface_install": install,
        "compiled_candidate_consumer_seam": {
            "cache_sha256": compiler_key,
            "parent_sha256": parent,
            "bridge_sha256": bridge_sha,
        },
    }

    class Provider:
        catalog = SimpleNamespace(
            catalog_sha256=preparation["selective_catalog_sha256"],
            source_sha256=preparation["selective_source_sha256"],
            provider_sha256=preparation["selective_provider_sha256"],
        )

        def load_final_norm_payload(self):
            return object()

        def authenticate_final_norm_payload(self, _payload):
            return final_norm_sha

        def release_final_norm_payload(self, _payload):
            return None

        def close(self):
            return None

    class BoundaryCompiler:
        def __init__(self, *_args, **_kwargs):
            self.base_interface = base_interface
            self.anchor_fold = SimpleNamespace(report=anchor_report)

        def fit_interface(self, *_args, **_kwargs):
            return interface

    monkeypatch.setattr(
        postfold,
        "_authenticated_c61_diagnostic",
        lambda *_a, **_k: (MappingProxyType(diagnostic), "e" * 64),
    )
    monkeypatch.setattr(
        postfold,
        "open_lfm2_circuit_trace_cache",
        lambda *_a, **_k: SimpleNamespace(
            cache_key=compiler_key,
            parent_fingerprint=parent,
        ),
    )
    monkeypatch.setattr(postfold.cpcf_resources, "_read_trajectory_rows", lambda _p: ())
    monkeypatch.setattr(
        postfold.cpcf_resources,
        "_build_trace_adapter",
        lambda **_k: SimpleNamespace(evidence_sha256="c" * 64),
    )
    monkeypatch.setattr(
        postfold,
        "open_lfm2_cpcf_selective_evidence_provider_from_catalog",
        lambda *_a, **_k: Provider(),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "build_lfm2_cpcf_v5_target_phase_shells",
        lambda *_a, **_k: (),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "materialize_lfm2_cpcf_v5_final_norm_phases",
        lambda *_a, **_k: (),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "_SelectiveBoundaryCompiler",
        BoundaryCompiler,
    )
    monkeypatch.setattr(
        postfold.cpcf_model_fold, "_final_norm_epsilon", lambda _m: 1e-5
    )

    rehydrated, receipt = postfold.rehydrate_lfm2_c61_tied_interface(
        model,
        source,
        semantic_cache=SimpleNamespace(
            cache_key=semantic_key,
            parent_fingerprint=parent,
        ),
        bridge=SimpleNamespace(artifact_sha256=bridge_sha),
        target_tokenizer=object(),
        compiler_diagnostic_json="unused.json",
        expected_compiler_diagnostic_sha256=diagnostic["receipt_sha256"],
    )

    torch.testing.assert_close(rehydrated.producer_encoder, encoder, rtol=0, atol=0)
    assert receipt["active_interface_is_anchor_repaired"] is True
    assert receipt["producer_encoder_recovered_from_repaired_target_table"] is False
    assert receipt["compiler_cache_key"] == compiler_key
    assert receipt["semantic_evidence_cache_key"] == semantic_key


def test_attention_installation_receipt_requires_a_real_layer_local_change() -> None:
    receipt = postfold.seal_lfm2_semantic_attention_installation_receipt(
        layer_index=2,
        transport_evidence_sha256=("1" * 64, "2" * 64),
        model_state_before_sha256="3" * 64,
        model_state_after_sha256="4" * 64,
        changed_state_keys=("model.layers.2.self_attn.q_proj.weight",),
        parameter_element_count=57_405_368,
    )

    assert receipt["installed"] is True
    assert receipt["parameter_count_changed"] is False
    assert receipt["operation_audit"]["training_steps"] == 0
    body = dict(receipt)
    seal = body.pop("result_sha256")
    assert seal == canonical_sha256(body)


def _normalized_decoder_fixture(
    parent_rows: tuple[torch.Tensor, ...],
    chart: torch.Tensor,
    *,
    parent_epsilon: float,
    target_epsilon: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    parent_gamma = torch.tensor(
        [0.75, 1.25, 1.5, 0.5],
        dtype=torch.float64,
    )
    target_gamma = torch.tensor([1.125, 0.625], dtype=torch.float64)
    child_rows = tuple(
        (
            (row @ chart)
            * torch.rsqrt(
                (row @ chart).square().mean(dim=-1, keepdim=True) + target_epsilon
            )
        )
        * target_gamma
        for row in parent_rows
    )
    decoder, receipt = postfold._fit_normalized_operator_input_decoder(
        parent_raw_rows=parent_rows,
        child_normalized_rows=child_rows,
        parent_operator_norm_weight=parent_gamma,
        target_operator_norm_weight=target_gamma,
        parent_norm_epsilon=parent_epsilon,
        target_norm_epsilon=target_epsilon,
        residual_chart=chart,
        residual_chart_sha256=postfold._tensor_sha256(chart),
        fit_trajectory_count=len(parent_rows) - 1,
        ridge=1e-12,
    )
    return decoder, dict(receipt)


def _coordinate_completion_receipt(
    raw_structural_decoder: torch.Tensor,
    *,
    generic_body_global: bool = False,
) -> dict[str, object]:
    structural_rank = min(map(int, raw_structural_decoder.shape))
    body = {
        "schema": (
            "neural-forge-lfm2-body-global-coordinate-completion-receipt-v1"
            if generic_body_global
            else "neural-forge-analytic-coordinate-completion-candidate-v1"
        ),
        (
            "requested_recipe" if generic_body_global else "recipe"
        ): "bridge-procrustes-isometric" if generic_body_global else "joint_gram",
        ("input_evidence_sha256" if generic_body_global else "evidence_sha256"): "d"
        * 64,
        "coordinate_map_shape": [int(value) for value in raw_structural_decoder.shape],
        "coordinate_map_sha256": (
            postfold._analytic_coordinate_completion_tensor_sha256(
                raw_structural_decoder
            )
        ),
        (
            "required_full_rank" if generic_body_global else "structural_rank"
        ): structural_rank,
        "materialized_rank": structural_rank,
        "full_rank_materialized": True,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _bounded_attention_decoder_fixture(
    *,
    rho: float,
    heldout_parent: torch.Tensor | None = None,
    heldout_child: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    chart = torch.eye(2, dtype=torch.float64)
    parent_rows = (
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        (
            torch.tensor([[1.0, 1.0]], dtype=torch.float64)
            if heldout_parent is None
            else heldout_parent
        ),
    )
    child_rows = (
        torch.tensor([[100.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 100.0]], dtype=torch.float64),
        (
            torch.tensor([[1.0, 1.0]], dtype=torch.float64)
            if heldout_child is None
            else heldout_child
        ),
    )
    decoder, receipt = postfold._fit_normalized_operator_input_decoder(
        parent_raw_rows=parent_rows,
        child_normalized_rows=child_rows,
        parent_operator_norm_weight=torch.ones(2, dtype=torch.float64),
        target_operator_norm_weight=torch.ones(2, dtype=torch.float64),
        parent_norm_epsilon=1e-5,
        target_norm_epsilon=1e-5,
        residual_chart=chart,
        residual_chart_sha256=postfold._tensor_sha256(chart),
        fit_trajectory_count=2,
        normalized_input_correction_rho=rho,
        ridge=1e-12,
    )
    return decoder, dict(receipt)


def test_attention_normalized_decoder_caps_fit_correction_relative_to_prior() -> None:
    decoder, receipt = _bounded_attention_decoder_fixture(rho=0.1)
    prior = torch.eye(2, dtype=torch.float64)
    correction_norm = float(torch.linalg.vector_norm(decoder - prior).item())

    assert receipt["structural_prior_relative_trust_bound"] is True
    assert receipt["normalized_input_correction_rho"] == 0.1
    assert receipt["correction_norm_cap_scale"] < 1.0
    assert receipt["correction_bound_satisfied"] is True
    assert correction_norm == pytest.approx(
        receipt["published_fit_rowspace_correction_frobenius_norm"],
        rel=1e-12,
        abs=1e-12,
    )
    assert correction_norm == pytest.approx(
        receipt["allowed_fit_rowspace_correction_frobenius_norm"],
        rel=1e-12,
        abs=1e-12,
    )
    assert receipt["allowed_fit_rowspace_correction_frobenius_norm"] == (
        pytest.approx(
            0.1 * receipt["fit_supported_structural_prior_frobenius_norm"],
            rel=1e-12,
            abs=1e-12,
        )
    )


def test_attention_normalized_decoder_rho_zero_is_exact_structural_prior() -> None:
    decoder, receipt = _bounded_attention_decoder_fixture(rho=0.0)
    prior = torch.eye(2, dtype=torch.float64)

    assert torch.equal(decoder, prior)
    assert receipt["rho_zero_structural_prior_exact"] is True
    assert receipt["correction_norm_cap_scale"] == 0.0
    assert receipt["published_fit_rowspace_correction_frobenius_norm"] == 0.0
    assert receipt["published_decoder_sha256"] == receipt["structural_prior_sha256"]
    assert receipt["uncapped_fit_rowspace_correction_frobenius_norm"] > 0.0


def test_attention_normalized_decoder_is_deterministic_and_heldout_is_diagnostic() -> (
    None
):
    first_decoder, first_receipt = _bounded_attention_decoder_fixture(rho=0.25)
    repeat_decoder, repeat_receipt = _bounded_attention_decoder_fixture(rho=0.25)
    changed_decoder, changed_receipt = _bounded_attention_decoder_fixture(
        rho=0.25,
        heldout_parent=torch.tensor([[50.0, -25.0]], dtype=torch.float64),
        heldout_child=torch.tensor([[-100.0, 30.0]], dtype=torch.float64),
    )

    assert torch.equal(first_decoder, repeat_decoder)
    assert first_receipt == repeat_receipt
    assert torch.equal(first_decoder, changed_decoder)
    assert (
        first_receipt["published_decoder_sha256"]
        == changed_receipt["published_decoder_sha256"]
    )
    assert (
        first_receipt["residual_correction_sha256"]
        == changed_receipt["residual_correction_sha256"]
    )
    assert first_receipt["heldout_metrics"] != changed_receipt["heldout_metrics"]
    assert changed_receipt["heldout_metrics_are_diagnostic_only"] is True
    assert changed_receipt["heldout_metrics_used_for_acceptance_or_veto"] is False
    assert changed_receipt["heldout_rows_used_for_fit"] == 0


def test_normalized_decoder_width_change_uses_fit_covariance_scale_and_epsilons() -> (
    None
):
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    parent_rows = (
        torch.tensor([[1.0, 2.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0, -1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[-1.0, -2.0, 0.0, 0.0]], dtype=torch.float64),
    )
    parent_epsilon = 1e-5
    target_epsilon = 3e-5
    decoder, receipt = _normalized_decoder_fixture(
        parent_rows,
        chart,
        parent_epsilon=parent_epsilon,
        target_epsilon=target_epsilon,
    )
    source_energy = parent_rows[0].square().mean()
    projected_energy = (parent_rows[0] @ chart).square().mean()
    expected_scale = torch.sqrt(
        (projected_energy + target_epsilon) / (source_energy + parent_epsilon)
    )
    scale = receipt["structural_covariance_scale"]

    assert scale["trajectory_balanced"] is True
    assert scale["actual_child_rows_used_for_scale"] == 0
    assert scale["heldout_rows_used_for_scale"] == 0
    assert scale["target_norm_epsilon"] == target_epsilon
    assert scale["covariance_scale"] == pytest.approx(
        float(expected_scale.item()),
        rel=1e-12,
    )
    assert scale["covariance_scale"] == pytest.approx(2.0**0.5, rel=2e-5)
    parent_gamma = torch.tensor([0.75, 1.25, 1.5, 0.5], dtype=torch.float64)
    target_gamma = torch.tensor([1.125, 0.625], dtype=torch.float64)
    expected_decoder = (
        expected_scale
        * torch.diag(target_gamma.reciprocal())
        @ torch.linalg.pinv(chart)
        @ torch.diag(parent_gamma)
    )
    torch.testing.assert_close(decoder, expected_decoder, rtol=1e-10, atol=1e-10)
    assert (
        receipt["heldout_rowwise_denominator_ratio_diagnostic"]["diagnostic_only"]
        is True
    )
    assert receipt["raw_structural_decoder_provenance"] == {
        "source": "residual_chart_moore_penrose_pseudoinverse",
        "raw_structural_decoder_shape": [2, 4],
        "raw_structural_decoder_sha256": postfold._tensor_sha256(
            torch.linalg.pinv(chart)
        ),
        "coordinate_completion_coordinate_map_sha256": None,
        "authenticated_coordinate_completion_used": False,
        "coordinate_completion_schema": None,
        "coordinate_completion_recipe": None,
        "coordinate_completion_evidence_sha256": None,
        "coordinate_completion_input_evidence_sha256": None,
        "coordinate_completion_receipt_sha256": None,
        "coordinate_completion_rank_claim_fields": [],
        "observed_materialized_rank": None,
        "materialized_rank_threshold": None,
    }


@pytest.mark.parametrize(
    "generic_body_global",
    (False, True),
    ids=("analytic-candidate", "body-global-generic"),
)
def test_normalized_decoder_accepts_authenticated_structural_completion(
    generic_body_global: bool,
) -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    parent_rows = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    parent_gamma = torch.tensor([0.75, 1.25, 1.5, 0.5], dtype=torch.float64)
    target_gamma = torch.tensor([1.125, 0.625], dtype=torch.float64)
    target_epsilon = 3e-5
    child_rows = tuple(
        (
            (row @ chart)
            * torch.rsqrt(
                (row @ chart).square().mean(dim=-1, keepdim=True) + target_epsilon
            )
        )
        * target_gamma
        for row in parent_rows
    )
    raw_decoder = torch.tensor(
        [[1.0, 0.0, 0.25, -0.5], [0.0, 1.0, 0.75, 0.125]],
        dtype=torch.float64,
    )
    completion = _coordinate_completion_receipt(
        raw_decoder,
        generic_body_global=generic_body_global,
    )
    assert postfold._tensor_sha256(raw_decoder) != completion["coordinate_map_sha256"]

    decoder, receipt_value = postfold._fit_normalized_operator_input_decoder(
        parent_raw_rows=parent_rows,
        child_normalized_rows=child_rows,
        parent_operator_norm_weight=parent_gamma,
        target_operator_norm_weight=target_gamma,
        parent_norm_epsilon=1e-5,
        target_norm_epsilon=target_epsilon,
        residual_chart=chart,
        residual_chart_sha256=postfold._tensor_sha256(chart),
        fit_trajectory_count=2,
        raw_structural_decoder=raw_decoder,
        raw_structural_decoder_sha256=postfold._tensor_sha256(raw_decoder),
        coordinate_completion_receipt=completion,
        coordinate_completion_receipt_sha256=completion["result_sha256"],
        ridge=1e-12,
    )
    receipt = dict(receipt_value)
    covariance_scale = receipt["structural_covariance_scale"]["covariance_scale"]
    structural_prior = (
        covariance_scale
        * torch.diag(target_gamma.reciprocal())
        @ raw_decoder
        @ torch.diag(parent_gamma)
    )

    # The fit rows occupy target coordinate zero only.  Their residual can
    # correct that supported row, while the completed row stays structural.
    torch.testing.assert_close(decoder[1], structural_prior[1], rtol=1e-12, atol=1e-12)
    assert receipt["fit_design_rank"] == 1
    assert receipt["fit_nullspace_correction_maximum_absolute"] < 1e-12
    assert receipt["structural_prior"].endswith(
        "raw_structural_decoder@diag(parent_gamma)"
    )
    expected_rank_threshold = float(
        torch.linalg.svdvals(raw_decoder).max().item()
    ) * max(
        1e-12,
        torch.finfo(torch.float64).eps * max(map(int, raw_decoder.shape)),
    )
    assert receipt["raw_structural_decoder_provenance"] == {
        "source": "authenticated_coordinate_completion",
        "raw_structural_decoder_shape": [2, 4],
        "raw_structural_decoder_sha256": postfold._tensor_sha256(raw_decoder),
        "coordinate_completion_coordinate_map_sha256": (
            postfold._analytic_coordinate_completion_tensor_sha256(raw_decoder)
        ),
        "authenticated_coordinate_completion_used": True,
        "coordinate_completion_schema": completion["schema"],
        "coordinate_completion_recipe": completion.get(
            "recipe",
            completion.get("requested_recipe"),
        ),
        "coordinate_completion_evidence_sha256": completion.get(
            "evidence_sha256",
            completion.get("input_evidence_sha256"),
        ),
        "coordinate_completion_input_evidence_sha256": completion.get(
            "input_evidence_sha256"
        ),
        "coordinate_completion_receipt_sha256": completion["result_sha256"],
        "coordinate_completion_rank_claim_fields": [
            "required_full_rank" if generic_body_global else "structural_rank"
        ],
        "observed_materialized_rank": 2,
        "materialized_rank_threshold": expected_rank_threshold,
    }


def test_normalized_decoder_rejects_partial_structural_completion_inputs() -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    parent_rows = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    parent_gamma = torch.ones(4, dtype=torch.float64)
    target_gamma = torch.ones(2, dtype=torch.float64)
    child_rows = tuple(row @ chart for row in parent_rows)

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="requires its tensor SHA-256",
    ):
        postfold._fit_normalized_operator_input_decoder(
            parent_raw_rows=parent_rows,
            child_normalized_rows=child_rows,
            parent_operator_norm_weight=parent_gamma,
            target_operator_norm_weight=target_gamma,
            parent_norm_epsilon=1e-5,
            target_norm_epsilon=1e-5,
            residual_chart=chart,
            residual_chart_sha256=postfold._tensor_sha256(chart),
            fit_trajectory_count=2,
            raw_structural_decoder=torch.linalg.pinv(chart),
        )


@pytest.mark.parametrize("invalid_kind", ("shape", "nonfinite", "tensor_hash"))
def test_normalized_decoder_rejects_invalid_raw_structural_decoder(
    invalid_kind: str,
) -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    parent_rows = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    child_rows = tuple(row @ chart for row in parent_rows)
    valid_decoder = torch.linalg.pinv(chart)
    raw_decoder = valid_decoder.clone()
    if invalid_kind == "shape":
        raw_decoder = raw_decoder[:, :3]
    elif invalid_kind == "nonfinite":
        raw_decoder[0, 0] = torch.inf
    completion = _coordinate_completion_receipt(valid_decoder)
    raw_sha256 = (
        "a" * 64
        if invalid_kind == "tensor_hash"
        else postfold._tensor_sha256(raw_decoder)
    )
    expected_error = (
        "incompatible shape or nonfinite"
        if invalid_kind in {"shape", "nonfinite"}
        else "tensor authentication failed"
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match=expected_error):
        postfold._fit_normalized_operator_input_decoder(
            parent_raw_rows=parent_rows,
            child_normalized_rows=child_rows,
            parent_operator_norm_weight=torch.ones(4, dtype=torch.float64),
            target_operator_norm_weight=torch.ones(2, dtype=torch.float64),
            parent_norm_epsilon=1e-5,
            target_norm_epsilon=1e-5,
            residual_chart=chart,
            residual_chart_sha256=postfold._tensor_sha256(chart),
            fit_trajectory_count=2,
            raw_structural_decoder=raw_decoder,
            raw_structural_decoder_sha256=raw_sha256,
            coordinate_completion_receipt=completion,
            coordinate_completion_receipt_sha256=completion["result_sha256"],
        )


@pytest.mark.parametrize("invalid_kind", ("self_seal", "expected_hash", "map"))
def test_normalized_decoder_rejects_inconsistent_completion_receipt(
    invalid_kind: str,
) -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    parent_rows = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    child_rows = tuple(row @ chart for row in parent_rows)
    raw_decoder = torch.linalg.pinv(chart)
    completion = _coordinate_completion_receipt(raw_decoder)
    expected_receipt_sha256 = completion["result_sha256"]
    if invalid_kind == "self_seal":
        completion["recipe"] = "tampered"
    elif invalid_kind == "expected_hash":
        expected_receipt_sha256 = "a" * 64
    else:
        completion_body = dict(completion)
        completion_body.pop("result_sha256")
        completion_body["coordinate_map_sha256"] = "b" * 64
        completion = {
            **completion_body,
            "result_sha256": canonical_sha256(completion_body),
        }
        expected_receipt_sha256 = completion["result_sha256"]
    expected_error = {
        "self_seal": "self-seal is invalid",
        "expected_hash": "receipt authentication failed",
        "map": "receipt is inconsistent",
    }[invalid_kind]

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match=expected_error):
        postfold._fit_normalized_operator_input_decoder(
            parent_raw_rows=parent_rows,
            child_normalized_rows=child_rows,
            parent_operator_norm_weight=torch.ones(4, dtype=torch.float64),
            target_operator_norm_weight=torch.ones(2, dtype=torch.float64),
            parent_norm_epsilon=1e-5,
            target_norm_epsilon=1e-5,
            residual_chart=chart,
            residual_chart_sha256=postfold._tensor_sha256(chart),
            fit_trajectory_count=2,
            raw_structural_decoder=raw_decoder,
            raw_structural_decoder_sha256=postfold._tensor_sha256(raw_decoder),
            coordinate_completion_receipt=completion,
            coordinate_completion_receipt_sha256=expected_receipt_sha256,
        )


@pytest.mark.parametrize(
    "invalid_kind",
    ("blank_schema", "missing_evidence", "wrong_required_rank", "rank_deficient"),
)
def test_normalized_decoder_generic_completion_contract_fails_closed(
    invalid_kind: str,
) -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    parent_rows = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    child_rows = tuple(row @ chart for row in parent_rows)
    raw_decoder = torch.linalg.pinv(chart)
    if invalid_kind == "rank_deficient":
        raw_decoder[1].zero_()
    completion = _coordinate_completion_receipt(
        raw_decoder,
        generic_body_global=True,
    )
    completion_body = dict(completion)
    completion_body.pop("result_sha256")
    if invalid_kind == "blank_schema":
        completion_body["schema"] = " "
    elif invalid_kind == "missing_evidence":
        completion_body.pop("input_evidence_sha256")
    elif invalid_kind == "wrong_required_rank":
        completion_body["required_full_rank"] = 1
    completion = {
        **completion_body,
        "result_sha256": canonical_sha256(completion_body),
    }
    expected_error = (
        "no authenticated evidence"
        if invalid_kind == "missing_evidence"
        else "receipt is inconsistent"
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match=expected_error):
        postfold._fit_normalized_operator_input_decoder(
            parent_raw_rows=parent_rows,
            child_normalized_rows=child_rows,
            parent_operator_norm_weight=torch.ones(4, dtype=torch.float64),
            target_operator_norm_weight=torch.ones(2, dtype=torch.float64),
            parent_norm_epsilon=1e-5,
            target_norm_epsilon=1e-5,
            residual_chart=chart,
            residual_chart_sha256=postfold._tensor_sha256(chart),
            fit_trajectory_count=2,
            raw_structural_decoder=raw_decoder,
            raw_structural_decoder_sha256=postfold._tensor_sha256(raw_decoder),
            coordinate_completion_receipt=completion,
            coordinate_completion_receipt_sha256=completion["result_sha256"],
        )


def test_normalized_decoder_isotropic_projection_scale_is_one() -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    basis = tuple(
        torch.eye(4, dtype=torch.float64)[index : index + 1] for index in range(4)
    )
    _decoder, receipt = _normalized_decoder_fixture(
        (*basis, torch.ones((1, 4), dtype=torch.float64)),
        chart,
        parent_epsilon=2e-5,
        target_epsilon=2e-5,
    )

    assert receipt["structural_covariance_scale"]["covariance_scale"] == pytest.approx(
        1.0, rel=1e-12, abs=1e-12
    )


def test_normalized_decoder_nonorthogonal_chart_uses_pseudoinverse_prior() -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [0.25, -0.5], [0.0, 0.75]],
        dtype=torch.float64,
    )
    coefficients = (
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        torch.tensor([[-1.0, -2.0]], dtype=torch.float64),
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    )
    parent_rows = tuple(value @ chart.T for value in coefficients)
    decoder, receipt = _normalized_decoder_fixture(
        parent_rows,
        chart,
        parent_epsilon=1e-5,
        target_epsilon=1e-5,
    )
    scale = receipt["structural_covariance_scale"]["covariance_scale"]
    parent_gamma = torch.tensor([0.75, 1.25, 1.5, 0.5], dtype=torch.float64)
    target_gamma = torch.tensor([1.125, 0.625], dtype=torch.float64)
    expected = (
        scale
        * torch.diag(target_gamma.reciprocal())
        @ torch.linalg.pinv(chart)
        @ torch.diag(parent_gamma)
    )

    torch.testing.assert_close(decoder, expected, rtol=1e-9, atol=1e-9)
    assert receipt["fit_metrics"]["maximum_absolute_error"] < 1e-9

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="must change model state",
    ):
        postfold.seal_lfm2_semantic_attention_installation_receipt(
            layer_index=2,
            transport_evidence_sha256=("1" * 64,),
            model_state_before_sha256="3" * 64,
            model_state_after_sha256="3" * 64,
            changed_state_keys=("model.layers.2.self_attn.q_proj.weight",),
            parameter_element_count=10,
        )


def test_layer_stack_defers_final_w2_and_fuses_three_equal_mass_lanes() -> None:
    evidence = {
        split: _terminal_evidence(split, "final-mlp")
        for split in ("fit", "fit_check", "selection", "validation")
    }
    evidence_sha = {
        split: evidence[split].authentication_sha256 for split in sorted(evidence)
    }
    body = {
        "final_layer_mlp_w2_deferred_to_fused_terminal": True,
        "final_layer_mlp_w2_install_count": 0,
        "final_layer_mlp_evidence_sha256": evidence_sha,
    }
    installation = postfold.Lfm2SemanticLayerStackInstallation(
        receipt={**body, "result_sha256": canonical_sha256(body)},
        final_layer_mlp_evidence=evidence,
    )

    assert installation.receipt["final_layer_mlp_w2_install_count"] == 0
    merged = postfold._merge_terminal_evidence(
        {
            "final_layer_mlp": evidence["fit"],
            "lexical": _terminal_evidence("fit", "lexical"),
            "parent_endpoint": _terminal_evidence("fit", "parent"),
        }
    )
    assert merged.rows == 3
    assert float(merged.sample_weights.sum().item()) == pytest.approx(1.0)
    assert tuple(merged.sample_weights.tolist()) == pytest.approx((1.0 / 3.0,) * 3)


def test_lexical_coordinates_cover_all_87_inserted_rows() -> None:
    multiplicities = (1,) * 273 + (2,) * 66 + (3,) * 9 + (4,)
    phases = [
        SimpleNamespace(
            phase="prefill",
            target_full_prefix_length=10,
        )
    ]
    length = 10
    token = 100
    for index, multiplicity in enumerate(multiplicities):
        replacement = tuple(range(token, token + multiplicity))
        phases.append(
            SimpleNamespace(
                phase=f"decode_{index:03d}",
                target_replacement_token_ids=replacement,
                target_replaced_suffix_token_count=0,
                target_rewrite_start=length,
                target_full_prefix_length=length + multiplicity,
            )
        )
        length += multiplicity
        token += multiplicity
    semantic = SimpleNamespace(source_batch_index=7, phases=tuple(phases))

    rows, desired, row_ids, observed = (
        postfold._append_only_lexical_supervision_coordinates(semantic)
    )

    assert len(observed) == 349
    assert len(rows) == len(desired) == len(row_ids) == 87
    assert length - 10 == 436
    assert max(observed) == 4
    assert {value: observed.count(value) for value in set(observed)} == {
        1: 273,
        2: 66,
        3: 9,
        4: 1,
    }
    assert len(set(row_ids)) == 87


def test_lexical_teacher_uses_best_finite_witness_when_exact_top1_is_impossible() -> (
    None
):
    tied = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.bfloat16,
    )

    teacher, report = postfold._lexical_teacher_rows(tied, (1, 2, 1))

    winners = (teacher.to(torch.float64) @ tied.to(torch.float64).T).argmax(dim=1)
    assert winners.tolist() == [0, 2, 0]
    assert torch.isfinite(teacher).all()
    assert report["all_teacher_rows_exact_target_top1"] is False
    assert report["exact_unique_desired_token_count"] == 1
    assert report["approximate_unique_desired_token_count"] == 1
    assert report["exact_desired_row_count"] == 1
    assert report["approximate_desired_row_count"] == 2
    assert report["missing_exact_top1_token_ids"] == [1]
    diagnostics = {
        value["target_token_id"]: value
        for value in report["selected_unique_token_diagnostics"]
    }
    assert diagnostics[1]["exact_target_top1"] is False
    assert diagnostics[1]["winner_token_id"] == 0
    assert diagnostics[1]["target_rank"] == 1
    assert diagnostics[1]["target_margin_over_best_competitor"] == 0.0
    body = dict(report)
    supplied = body.pop("teacher_sha256")
    assert supplied == canonical_sha256(body)


def test_lexical_teacher_exact_frontier_reports_exactness_truthfully() -> None:
    tied = torch.eye(3, dtype=torch.bfloat16)

    teacher, report = postfold._lexical_teacher_rows(tied, (2, 0, 1, 2))

    winners = (teacher.to(torch.float64) @ tied.to(torch.float64).T).argmax(dim=1)
    assert winners.tolist() == [2, 0, 1, 2]
    assert report["all_teacher_rows_exact_target_top1"] is True
    assert report["exact_unique_desired_token_count"] == 3
    assert report["approximate_unique_desired_token_count"] == 0
    assert report["exact_desired_row_count"] == 4
    assert report["approximate_desired_row_count"] == 0
    assert report["missing_exact_top1_token_ids"] == []


def test_semantic_alignment_wires_authenticated_selected_input_contract(
    monkeypatch,
) -> None:
    reference = SimpleNamespace(
        index=0,
        input_sha256="1" * 64,
        ordered_tensor_names=(
            "decode.000.forced_token_ids",
            "input.attention_mask",
            "input.forced_decode_ids",
            "input.input_ids",
        ),
        record={"capture": {"decode_steps": 1}},
    )
    payload = object()
    session_receipt = {"session_sha256": "2" * 64}
    released = []

    class Session:
        batch_references = (reference,)

        def as_dict(self):
            return session_receipt

        def load_selected_tensors(self, index, names):
            assert index == 0
            assert names == reference.ordered_tensor_names
            return payload

        def release_selected_tensors(self, value):
            released.append(value)

    cache = SimpleNamespace(
        cache_key="3" * 64,
        batch_count=1,
        manifest={"manifest_sha256": "4" * 64},
    )
    selected_input = object()
    semantic = SimpleNamespace(
        source_prompt_token_ids=(1, 2),
        phase_order=("prefill", "decode.000"),
    )
    alignment = object()
    calls = []
    tokenizer_receipt = {"receipt_sha256": "7" * 64}

    class TokenizerSession:
        donor_tokenizer_semantics_sha256 = "5" * 64
        target_tokenizer_semantics_sha256 = "6" * 64

        def __enter__(self):
            calls.append("tokenizer-open")
            return self

        def __exit__(self, *_args):
            calls.append("tokenizer-close")
            return None

        def as_dict(self):
            return tokenizer_receipt

    tokenizer_session = TokenizerSession()

    class Bridge:
        def authenticated_live_session(
            self,
            donor,
            target,
            *,
            selective_session_sha256,
        ):
            assert donor == "donor" and target == "target"
            assert selective_session_sha256 == session_receipt["session_sha256"]
            return tokenizer_session

    def build_selected(observed_reference, observed_payload, receipt, **kwargs):
        calls.append("selected")
        assert observed_reference is reference
        assert observed_payload is payload
        assert receipt is session_receipt
        assert kwargs == {"expected_cache_key": cache.cache_key}
        return selected_input

    def build_semantic(source, manifest, donor, target, **kwargs):
        calls.append("semantic")
        assert source is selected_input
        assert manifest is cache.manifest
        assert donor == "donor" and target == "target"
        assert kwargs["expected_source_input_sha256"] == reference.input_sha256
        assert kwargs["tokenizer_session"] is tokenizer_session
        return semantic

    def build_alignment(observed, donor, target, **kwargs):
        calls.append("alignment")
        assert observed is semantic
        assert donor == "donor" and target == "target"
        assert kwargs["parent_sampled_suffix_tokens"] == {
            "prefill": 2,
            "decode.000": 1,
        }
        assert kwargs["tokenizer_session"] is tokenizer_session
        return alignment

    monkeypatch.setattr(
        postfold,
        "build_lfm2_semantic_span_selected_input_evidence",
        build_selected,
    )
    monkeypatch.setattr(
        postfold,
        "build_lfm2_semantic_span_phase_bridge",
        build_semantic,
    )
    monkeypatch.setattr(
        postfold,
        "build_lfm2_semantic_span_cpcf_alignment",
        build_alignment,
    )

    observed_alignments, observed_receipt = postfold._semantic_alignments_from_session(
        cache=cache,
        session=Session(),
        bridge=Bridge(),
        donor_tokenizer="donor",
        target_tokenizer="target",
    )
    assert observed_alignments == (alignment,)
    assert dict(observed_receipt) == tokenizer_receipt
    assert calls == [
        "tokenizer-open",
        "selected",
        "semantic",
        "alignment",
        "tokenizer-close",
    ]
    assert released == [payload]


def test_attention_stage_fails_closed_with_exact_missing_compiler(
    monkeypatch,
) -> None:
    class FakeCache:
        cache_key = "a" * 64
        parent_fingerprint = "b" * 64
        batch_count = 1

        def selective_authenticated_session(self):
            class Session:
                batch_references = ()

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def verify(self):
                    return self

                def as_dict(self):
                    return {"session_sha256": "c" * 64}

            return Session()

    class FakeBridge:
        artifact_sha256 = "d" * 64

        def verify_live_tokenizers(self, _donor, _target):
            return self

    alignment = SimpleNamespace(
        alignment_sha256="e" * 64,
        to_evidence_dict=lambda: {
            "alignment_sha256": "e" * 64,
        },
    )
    target = SimpleNamespace(_neural_forge_quarantine_tokenizer=object())
    monkeypatch.setattr(postfold, "Lfm2CircuitTraceCache", FakeCache)
    monkeypatch.setattr(postfold, "StrictTokenizerBridgeArtifact", FakeBridge)
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _value: "f" * 64,
    )
    monkeypatch.setattr(
        postfold,
        "_attention_layer_indices",
        lambda _cache, _model: (2,),
    )
    monkeypatch.setattr(
        postfold,
        "_target_rope",
        lambda _model: (torch.tensor([1.0], dtype=torch.float64), 1.0),
    )
    monkeypatch.setattr(
        postfold,
        "_semantic_alignments_from_session",
        lambda **_kwargs: ((alignment,), {"receipt_sha256": "8" * 64}),
    )
    monkeypatch.setattr(
        postfold,
        "audit_lfm2_semantic_span_cpcf_refinement",
        lambda *_args, **_kwargs: {
            "probability_preserving_attention_transport_go": True,
            "blocking_reasons": [],
            "report_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(postfold, "_resolve_attention_installer", lambda _value: None)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="9" * 64),
    )
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 57_405_368)
    monkeypatch.setattr(
        postfold,
        "load_lfm2_parent_qkvo_inventory_from_local_safetensors",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldBlockedError) as captured:
        postfold.compile_install_lfm2_semantic_attention_stage(
            cache=FakeCache(),
            bridge=FakeBridge(),
            target_model=target,
            interface=object(),
            compiler_lineage={
                2: SimpleNamespace(source_inventory=object()),
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
        )

    report = dict(captured.value.report)
    assert report["status"] == "blocked-no-checkpoint-published"
    assert report["terminal_refold_called"] is False
    assert report["checkpoint_save_called"] is False
    assert report["missing_callable"] == (
        "neural_forge.hf_lfm2_semantic_attention_refold_adapter."
        "compile_install_lfm2_semantic_attention_postfold"
    )
    assert report["operation_audit"]["training_steps"] == 0
    body = dict(report)
    seal = body.pop("failure_sha256")
    assert seal == canonical_sha256(body)


def _write_minimal_postfold_artifact(
    directory: Path,
    *,
    extra_name: str | None = None,
) -> None:
    config = directory / "config.json"
    weight = directory / "model.safetensors"
    tokenizer = directory / "tokenizer.json"
    config.write_text("{}", encoding="utf-8")
    weight.write_bytes(b"test")
    tokenizer.write_text("{}", encoding="utf-8")
    body = {
        "schema": postfold.LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
        "dense_safetensor_files": [
            {
                "path": weight.name,
                "size_bytes": weight.stat().st_size,
                "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            }
        ],
        "tokenizer_package_files": [
            {
                "path": tokenizer.name,
                "bytes": tokenizer.stat().st_size,
                "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            }
        ],
        "dense_safetensors_only": True,
        "target_tokenizer_saved_with_child": True,
        "local_reload_only": True,
        "network_access": False,
        "hub_loader_used": False,
    }
    value = {**body, "artifact_sha256": canonical_sha256(body)}
    (directory / postfold.LFM2_CPCF_V5_ARTIFACT_METADATA).write_text(
        json.dumps(value),
        encoding="utf-8",
    )
    if extra_name is not None:
        (directory / extra_name).write_bytes(b"unlisted runtime material")


def test_checkpoint_publication_is_staged_then_reloaded_after_atomic_move(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _TiedModel(torch.full((4, 2), 0.5, dtype=torch.bfloat16))
    tokenizer = object()
    events = []
    model_state = "1" * 64
    inventory_sha = "2" * 64
    tokenizer_sha = "3" * 64

    class Saver:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _model, directory):
            events.append(("save", directory.name))
            _write_minimal_postfold_artifact(directory)

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, directory):
            events.append(("load", directory.name))
            return model

    monkeypatch.setattr(postfold, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(postfold, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(postfold, "fingerprint_model_state", lambda _model: model_state)
    monkeypatch.setattr(
        postfold,
        "independent_parameter_inventory",
        lambda _model: {"inventory_sha256": inventory_sha},
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: tokenizer_sha,
    )
    monkeypatch.setattr(postfold, "_ffn_width", lambda _model: 7)
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 100)
    output = tmp_path / "standalone-postfold"

    reloaded, receipt = postfold.publish_lfm2_semantic_postfold_checkpoint(
        target_model=model,
        target_tokenizer=tokenizer,
        output_directory=output,
        context_binding_sha256="4" * 64,
        device="cpu",
    )

    assert reloaded is model
    assert output.is_dir()
    assert events[0][0] == "save"
    assert events[1][0] == "load"
    assert events[1][1].startswith(".semantic-postfold-")
    assert events[2] == ("load", "standalone-postfold")
    assert receipt["staged_reload_verified"] is True
    assert receipt["published_reload_verified"] is True
    assert receipt["complete_artifact_tree_hashed"] is True
    assert receipt["forbidden_named_or_referenced_sidecars_present"] is False
    assert receipt["standalone_directory_purity_authenticated"] is True
    assert receipt["artifact_file_count"] == 4
    assert receipt["artifact_tree_sha256"] == canonical_sha256(receipt["artifact_tree"])
    assert receipt["canonical_typed_artifact_inventory_authenticated"] is True
    assert receipt["staged_artifact_tree_purity_verified_before_publication"] is True
    assert receipt["unallowlisted_artifact_files_present"] is False
    assert receipt["artifact_inventory_sha256"] == canonical_sha256(
        receipt["artifact_inventory"]
    )
    assert {value["artifact_role"] for value in receipt["artifact_inventory"]} == {
        "artifact_metadata",
        "model_config",
        "model_tensor",
        "tokenizer_resource",
    }
    assert receipt["operation_audit"]["optimizer_steps"] == 0


@pytest.mark.parametrize("extra_name", ("auxiliary.dat", "payload.pkl"))
def test_checkpoint_publication_rejects_unallowlisted_sidecar_and_removes_output(
    tmp_path: Path,
    monkeypatch,
    extra_name: str,
) -> None:
    model = _TiedModel(torch.full((4, 2), 0.5, dtype=torch.bfloat16))

    class Saver:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _model, directory):
            _write_minimal_postfold_artifact(directory, extra_name=extra_name)

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _directory):
            return model

    monkeypatch.setattr(postfold, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(postfold, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(postfold, "fingerprint_model_state", lambda _model: "1" * 64)
    monkeypatch.setattr(
        postfold,
        "independent_parameter_inventory",
        lambda _model: {"inventory_sha256": "2" * 64},
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: "3" * 64,
    )
    monkeypatch.setattr(postfold, "_ffn_width", lambda _model: 7)
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 100)
    output = tmp_path / "forbidden-sidecar-publication"

    with pytest.raises(
        DynamicFoldArtifactError,
        match="canonical typed inventory",
    ):
        postfold.publish_lfm2_semantic_postfold_checkpoint(
            target_model=model,
            target_tokenizer=object(),
            output_directory=output,
            context_binding_sha256="4" * 64,
            device="cpu",
        )

    assert not output.exists()


def test_failed_post_move_reload_removes_only_new_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _TiedModel(torch.full((4, 2), 0.5, dtype=torch.bfloat16))

    class Saver:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _model, directory):
            _write_minimal_postfold_artifact(directory)

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, directory):
            if directory.name == "failed-publication":
                raise RuntimeError("synthetic final reload failure")
            return model

    monkeypatch.setattr(postfold, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(postfold, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(postfold, "fingerprint_model_state", lambda _model: "1" * 64)
    monkeypatch.setattr(
        postfold,
        "independent_parameter_inventory",
        lambda _model: {"inventory_sha256": "2" * 64},
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: "3" * 64,
    )
    monkeypatch.setattr(postfold, "_ffn_width", lambda _model: 7)
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 100)
    output = tmp_path / "failed-publication"

    with pytest.raises(RuntimeError, match="synthetic final reload failure"):
        postfold.publish_lfm2_semantic_postfold_checkpoint(
            target_model=model,
            target_tokenizer=object(),
            output_directory=output,
            context_binding_sha256="4" * 64,
            device="cpu",
        )

    assert not output.exists()


def test_complete_layer_stack_fails_closed_on_conv_and_mlp_gaps(
    monkeypatch,
) -> None:
    topology = [
        {
            "contract": {
                "kind": "full_attention" if index in {2, 5, 8, 11} else "short_conv"
            }
        }
        for index in range(14)
    ]
    cache = SimpleNamespace(manifest={"layer_topology": {"layers": topology}})
    monkeypatch.setattr(
        postfold,
        "_resolve_layer_stack_installer",
        lambda _value: None,
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldBlockedError) as captured:
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=cache,
            bridge=object(),
            target_model=object(),
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
        )

    report = dict(captured.value.report)
    assert report["required_causal_layer_count"] == 14
    assert report["required_attention_operator_count"] == 4
    assert report["required_short_conv_operator_count"] == 10
    assert "long-horizon-short-conv-refold-missing" in report["blocking_reasons"]
    assert "replay-fresh-per-layer-mlp-refold-missing" in report["blocking_reasons"]
    assert report["fused_terminal_refold_called"] is False
    assert report["checkpoint_save_called"] is False
    body = dict(report)
    seal = body.pop("failure_sha256")
    assert seal == canonical_sha256(body)


@pytest.mark.parametrize(
    "recipe_id",
    postfold.LFM2_SEMANTIC_BODY_RECIPE_IDS,
)
def test_layer_stack_stage_forwards_exact_authenticated_body_recipe(
    monkeypatch,
    recipe_id: str,
) -> None:
    captured = {}
    target = object()

    def compiler(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-body-recipe-forwarding")

    monkeypatch.setattr(
        postfold, "_resolve_layer_stack_installer", lambda _value: compiler
    )
    monkeypatch.setattr(postfold, "_resolve_attention_installer", lambda _value: None)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="a" * 64),
    )
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 1)
    monkeypatch.setattr(
        postfold,
        "_final_w2_parameter",
        lambda _model: torch.zeros((1, 1), dtype=torch.bfloat16),
    )

    with pytest.raises(RuntimeError, match="stop-after-body-recipe-forwarding"):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=target,
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            body_recipe=recipe_id,
            global_residual_basis_rank=776,
        )

    recipe = postfold.resolve_lfm2_semantic_body_recipe(recipe_id)
    assert captured["body_recipe"] == recipe
    assert captured["body_recipe"].recipe_sha256 == recipe.recipe_sha256
    assert captured["global_residual_basis_rank"] == 776


def test_layer_stack_stage_forwards_exact_fhvr_short_conv_construction(
    monkeypatch,
) -> None:
    captured = {}
    target = object()
    construction = postfold._resolve_short_conv_construction(
        "fhvr-mf-cap10k-sweeps1-full-history-full-coverage-v1"
    )

    def compiler(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-fhvr-construction-forwarding")

    monkeypatch.setattr(
        postfold, "_resolve_layer_stack_installer", lambda _value: compiler
    )
    monkeypatch.setattr(postfold, "_resolve_attention_installer", lambda _value: None)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="a" * 64),
    )
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 1)
    monkeypatch.setattr(
        postfold,
        "_final_w2_parameter",
        lambda _model: torch.zeros((1, 1), dtype=torch.bfloat16),
    )

    with pytest.raises(RuntimeError, match="stop-after-fhvr-construction-forwarding"):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=target,
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            short_conv_construction=construction,
        )

    assert captured["short_conv_construction"] == construction
    assert captured["short_conv_construction"].verify() is True


@pytest.mark.parametrize("partial_field", ("operators", "recipe"))
def test_layer_stack_stage_requires_all_coordinate_completion_fields(
    monkeypatch,
    partial_field: str,
) -> None:
    monkeypatch.setattr(
        postfold,
        "_resolve_layer_stack_installer",
        lambda _value: pytest.fail("partial completion reached the stack compiler"),
    )
    completion_kwargs = (
        {"coordinate_completion_parent_operators": (object(),)}
        if partial_field == "operators"
        else {"coordinate_completion_recipe": "bridge-procrustes-isometric"}
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="must be supplied together",
    ):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=object(),
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            **completion_kwargs,
        )


def test_layer_stack_stage_forwards_exact_coordinate_completion_configuration(
    monkeypatch,
) -> None:
    captured = {}
    target = object()
    operators = (object(), object())
    recipe = "bridge-procrustes-isometric"

    def compiler(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-coordinate-completion-forwarding")

    monkeypatch.setattr(
        postfold, "_resolve_layer_stack_installer", lambda _value: compiler
    )
    monkeypatch.setattr(postfold, "_resolve_attention_installer", lambda _value: None)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="a" * 64),
    )
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 1)
    monkeypatch.setattr(
        postfold,
        "_final_w2_parameter",
        lambda _model: torch.zeros((1, 1), dtype=torch.bfloat16),
    )

    with pytest.raises(
        RuntimeError,
        match="stop-after-coordinate-completion-forwarding",
    ):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=target,
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            coordinate_completion_parent_operators=operators,
            coordinate_completion_recipe=recipe,
        )

    assert captured["coordinate_completion_parent_operators"] is operators
    assert captured["coordinate_completion_recipe"] == recipe
    assert "coordinate_completion_parent_geometry" not in captured
    assert "coordinate_completion_least_squares_execution_profile" not in captured


def _coordinate_completion_geometry_fixture():
    operators = (
        ResidualFacingOperator(
            role="consumer",
            name="consumer",
            tensor=torch.tensor(
                [[1.0, 2.0], [0.5, 3.0]],
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.INPUT,
        ),
        ResidualFacingOperator(
            role="producer",
            name="producer",
            tensor=torch.tensor(
                [[2.0, 0.25], [1.0, 4.0]],
                dtype=torch.float64,
            ),
            orientation=FeatureOrientation.OUTPUT,
        ),
    )
    return (
        compile_role_balanced_operator_geometry(operators),
        resolve_analytic_execution_profile("cpu"),
    )


def _patch_layer_stack_forwarding_boundary(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, object],
) -> object:
    target = object()

    def compiler(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-geometry-forwarding")

    monkeypatch.setattr(
        postfold,
        "_resolve_layer_stack_installer",
        lambda _value: compiler,
    )
    monkeypatch.setattr(postfold, "_resolve_attention_installer", lambda _value: None)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="a" * 64),
    )
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 1)
    monkeypatch.setattr(
        postfold,
        "_final_w2_parameter",
        lambda _model: torch.zeros((1, 1), dtype=torch.bfloat16),
    )
    return target


def test_layer_stack_stage_forwards_authenticated_preaggregated_geometry_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    target = _patch_layer_stack_forwarding_boundary(monkeypatch, captured)
    geometry, profile = _coordinate_completion_geometry_fixture()

    with pytest.raises(RuntimeError, match="stop-after-geometry-forwarding"):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=target,
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            coordinate_completion_parent_geometry=geometry,
            coordinate_completion_least_squares_execution_profile=profile,
            coordinate_completion_recipe="bridge-procrustes-isometric",
        )

    assert captured["coordinate_completion_parent_geometry"] is geometry
    assert captured["coordinate_completion_least_squares_execution_profile"] is profile
    assert captured["coordinate_completion_recipe"] == "bridge-procrustes-isometric"
    assert "coordinate_completion_parent_operators" not in captured


@pytest.mark.parametrize(
    ("configuration", "message"),
    (
        ("both-sources", "exactly one"),
        ("geometry-without-profile", "requires an authenticated"),
        ("operators-with-profile", "valid only with"),
        ("unsupported-recipe", "recipe is unsupported"),
    ),
)
def test_layer_stack_stage_rejects_invalid_geometry_source_xor(
    monkeypatch: pytest.MonkeyPatch,
    configuration: str,
    message: str,
) -> None:
    geometry, profile = _coordinate_completion_geometry_fixture()
    kwargs: dict[str, object] = {
        "coordinate_completion_recipe": "bridge-procrustes-isometric",
    }
    if configuration == "both-sources":
        kwargs.update(
            {
                "coordinate_completion_parent_operators": (object(),),
                "coordinate_completion_parent_geometry": geometry,
                "coordinate_completion_least_squares_execution_profile": profile,
            }
        )
    elif configuration == "geometry-without-profile":
        kwargs["coordinate_completion_parent_geometry"] = geometry
    elif configuration == "operators-with-profile":
        kwargs.update(
            {
                "coordinate_completion_parent_operators": (object(),),
                "coordinate_completion_least_squares_execution_profile": profile,
            }
        )
    else:
        kwargs.update(
            {
                "coordinate_completion_parent_geometry": geometry,
                "coordinate_completion_least_squares_execution_profile": profile,
                "coordinate_completion_recipe": "metric-auto-winner",
            }
        )
    monkeypatch.setattr(
        postfold,
        "_resolve_layer_stack_installer",
        lambda _value: pytest.fail("invalid geometry reached stack compiler"),
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match=message):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=object(),
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            **kwargs,
        )


@pytest.mark.parametrize("tamper_target", ("geometry", "profile"))
def test_layer_stack_stage_rejects_tampered_geometry_or_profile(
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    geometry, profile = _coordinate_completion_geometry_fixture()
    if tamper_target == "geometry":
        geometry.raw_gram[0, 0] += 1.0
        message = "geometry failed reauthentication"
    else:
        profile_receipt = profile.as_dict()
        profile_receipt["result_sha256"] = "0" * 64
        object.__setattr__(
            profile,
            "receipt",
            MappingProxyType(profile_receipt),
        )
        message = "profile failed reauthentication"
    monkeypatch.setattr(
        postfold,
        "_resolve_layer_stack_installer",
        lambda _value: pytest.fail("tampered geometry reached stack compiler"),
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match=message):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=object(),
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            coordinate_completion_parent_geometry=geometry,
            coordinate_completion_least_squares_execution_profile=profile,
            coordinate_completion_recipe="bridge-procrustes-isometric",
        )


def _returned_geometry_binding_fixture():
    geometry, profile = _coordinate_completion_geometry_fixture()
    hashes = geometry.report["hashes"]
    operator_sha256s = list(geometry.ordered_operator_sha256s)
    operator_manifest_sha256 = canonical_sha256(operator_sha256s)
    wrapper = {
        "schema": (
            postfold.LFM2_BODY_GLOBAL_COORDINATE_COMPLETION_GEOMETRY_RECEIPT_SCHEMA
        ),
        "parent_geometry_sha256": geometry.geometry_sha256,
        "parent_geometry_result_sha256": geometry.geometry_sha256,
        "parent_raw_gram_sha256": hashes["raw_gram_sha256"],
        "parent_balanced_gram_sha256": hashes["balanced_gram_sha256"],
        "parent_operator_sha256s": operator_sha256s,
        "parent_operator_manifest_sha256": operator_manifest_sha256,
        "least_squares_execution_profile": profile.as_dict(),
        "least_squares_execution_profile_sha256": profile.result_sha256,
        "preaggregated_parent_geometry": True,
        "parent_operator_reconstruction_calls": 0,
    }
    receipt = {
        "body_global_parent_operator_count": geometry.report["operator_count"],
        "body_global_parent_operator_sha256s": operator_sha256s,
        "body_global_parent_operator_manifest_sha256": operator_manifest_sha256,
        "body_global_parent_geometry_sha256": geometry.geometry_sha256,
        "body_global_least_squares_execution_profile": profile.as_dict(),
        "body_global_least_squares_execution_profile_sha256": (profile.result_sha256),
        "body_global_parent_raw_gram_sha256": hashes["raw_gram_sha256"],
        "body_global_parent_balanced_gram_sha256": (hashes["balanced_gram_sha256"]),
        "body_global_preaggregated_parent_geometry": True,
        "body_global_parent_operator_reconstruction_calls": 0,
    }
    return geometry, profile, wrapper, receipt


def test_returned_geometry_binding_accepts_complete_authenticated_identity() -> None:
    geometry, profile, wrapper, receipt = _returned_geometry_binding_fixture()

    assert postfold._returned_parent_geometry_completion_binding_valid(
        layer_stack_receipt=receipt,
        completion_wrapper_body=wrapper,
        parent_geometry=geometry,
        least_squares_execution_profile=profile,
    )


@pytest.mark.parametrize(
    ("location", "field"),
    (
        ("wrapper", "schema"),
        ("wrapper", "parent_operator_manifest_sha256"),
        ("wrapper", "preaggregated_parent_geometry"),
        ("wrapper", "parent_operator_reconstruction_calls"),
        ("receipt", "body_global_parent_geometry_sha256"),
        ("receipt", "body_global_parent_raw_gram_sha256"),
        ("receipt", "body_global_parent_operator_sha256s"),
        ("receipt", "body_global_parent_operator_manifest_sha256"),
        ("receipt", "body_global_least_squares_execution_profile"),
        ("receipt", "body_global_preaggregated_parent_geometry"),
        ("receipt", "body_global_parent_operator_reconstruction_calls"),
        ("wrapper", "parent_balanced_gram_sha256"),
        ("wrapper", "least_squares_execution_profile"),
    ),
)
def test_returned_geometry_binding_rejects_dropped_fields(
    location: str,
    field: str,
) -> None:
    geometry, profile, wrapper, receipt = _returned_geometry_binding_fixture()
    target = receipt if location == "receipt" else wrapper
    target.pop(field)

    assert not postfold._returned_parent_geometry_completion_binding_valid(
        layer_stack_receipt=receipt,
        completion_wrapper_body=wrapper,
        parent_geometry=geometry,
        least_squares_execution_profile=profile,
    )


@pytest.mark.parametrize(
    ("location", "field"),
    (
        ("wrapper", "schema"),
        ("wrapper", "parent_operator_manifest_sha256"),
        ("wrapper", "parent_operator_reconstruction_calls"),
        ("receipt", "body_global_parent_operator_count"),
        ("receipt", "body_global_parent_operator_sha256s"),
        ("receipt", "body_global_parent_operator_manifest_sha256"),
        ("receipt", "body_global_least_squares_execution_profile"),
        ("receipt", "body_global_least_squares_execution_profile_sha256"),
        ("receipt", "body_global_parent_balanced_gram_sha256"),
        ("receipt", "body_global_preaggregated_parent_geometry"),
        ("receipt", "body_global_parent_operator_reconstruction_calls"),
        ("wrapper", "parent_geometry_result_sha256"),
        ("wrapper", "parent_operator_sha256s"),
        ("wrapper", "least_squares_execution_profile_sha256"),
    ),
)
def test_returned_geometry_binding_rejects_substituted_identity(
    location: str,
    field: str,
) -> None:
    geometry, profile, wrapper, receipt = _returned_geometry_binding_fixture()
    target = receipt if location == "receipt" else wrapper
    if field == "body_global_parent_operator_count":
        target[field] += 1
    elif field in {
        "parent_operator_sha256s",
        "body_global_parent_operator_sha256s",
    }:
        target[field] = list(reversed(target[field]))
    elif field in {
        "parent_operator_reconstruction_calls",
        "body_global_parent_operator_reconstruction_calls",
    }:
        target[field] = 1
    elif field in {
        "preaggregated_parent_geometry",
        "body_global_preaggregated_parent_geometry",
    }:
        target[field] = False
    elif field in {
        "least_squares_execution_profile",
        "body_global_least_squares_execution_profile",
    }:
        target[field] = {**target[field], "requested_device": "cuda:7"}
    elif field == "schema":
        target[field] = "neural-forge-lfm2-body-global-coordinate-completion-receipt-v1"
    else:
        target[field] = "0" * 64

    assert not postfold._returned_parent_geometry_completion_binding_valid(
        layer_stack_receipt=receipt,
        completion_wrapper_body=wrapper,
        parent_geometry=geometry,
        least_squares_execution_profile=profile,
    )


@pytest.mark.parametrize(
    ("field", "substitute"),
    (
        ("body_global_parent_operator_manifest_sha256", "0" * 64),
        ("body_global_preaggregated_parent_geometry", False),
        ("body_global_parent_operator_reconstruction_calls", 1),
    ),
)
def test_returned_geometry_binding_rejects_resealed_outer_contradictions(
    field: str,
    substitute: object,
) -> None:
    geometry, profile, wrapper, receipt = _returned_geometry_binding_fixture()
    receipt[field] = substitute
    sealed_receipt = {
        **receipt,
        "result_sha256": canonical_sha256(receipt),
    }
    verified_receipt = postfold._verify_sealed(
        sealed_receipt,
        seal_name="result_sha256",
        label="contradictory layer stack",
    )

    assert not postfold._returned_parent_geometry_completion_binding_valid(
        layer_stack_receipt=verified_receipt,
        completion_wrapper_body=wrapper,
        parent_geometry=geometry,
        least_squares_execution_profile=profile,
    )


@pytest.mark.parametrize(
    ("field", "substitute"),
    (
        (
            "schema",
            "neural-forge-lfm2-body-global-coordinate-completion-receipt-v1",
        ),
        ("parent_operator_manifest_sha256", "0" * 64),
        ("parent_operator_reconstruction_calls", 1),
    ),
)
def test_returned_geometry_binding_rejects_resealed_wrapper_contradictions(
    field: str,
    substitute: object,
) -> None:
    geometry, profile, wrapper, receipt = _returned_geometry_binding_fixture()
    wrapper[field] = substitute
    sealed_wrapper = {
        **wrapper,
        "result_sha256": canonical_sha256(wrapper),
    }
    verified_wrapper = postfold._verify_sealed(
        sealed_wrapper,
        seal_name="result_sha256",
        label="contradictory completion wrapper",
    )
    verified_wrapper.pop("result_sha256")

    assert not postfold._returned_parent_geometry_completion_binding_valid(
        layer_stack_receipt=receipt,
        completion_wrapper_body=verified_wrapper,
        parent_geometry=geometry,
        least_squares_execution_profile=profile,
    )


def test_layer_stack_stage_legacy_path_omits_coordinate_completion_keywords(
    monkeypatch,
) -> None:
    captured = {}
    target = object()

    def compiler(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-legacy-forwarding")

    monkeypatch.setattr(
        postfold, "_resolve_layer_stack_installer", lambda _value: compiler
    )
    monkeypatch.setattr(postfold, "_resolve_attention_installer", lambda _value: None)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="a" * 64),
    )
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 1)
    monkeypatch.setattr(
        postfold,
        "_final_w2_parameter",
        lambda _model: torch.zeros((1, 1), dtype=torch.bfloat16),
    )

    with pytest.raises(RuntimeError, match="stop-after-legacy-forwarding"):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=target,
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
        )

    assert "coordinate_completion_parent_operators" not in captured
    assert "coordinate_completion_recipe" not in captured
    assert "coordinate_completion_parent_geometry" not in captured
    assert "coordinate_completion_least_squares_execution_profile" not in captured


def test_body_recipe_public_api_defaults_remain_fit_guided() -> None:
    assert (
        inspect.signature(postfold.compile_install_lfm2_semantic_layer_stack_stage)
        .parameters["body_recipe"]
        .default
        == "fit_guided"
    )
    assert (
        inspect.signature(postfold.run_lfm2_semantic_postfold)
        .parameters["body_recipe"]
        .default
        == "fit_guided"
    )


def test_postfold_public_entrypoints_reject_explicit_rank_for_v1(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="does not support an explicit",
    ):
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=object(),
            bridge=object(),
            target_model=object(),
            interface=object(),
            compiler_lineage={},
            parent_model_directory=tmp_path / "parent",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            global_residual_closure_version="output_writer_v1_diagnostic",
            global_residual_basis_rank=776,
        )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="does not support an explicit",
    ):
        postfold.run_lfm2_semantic_postfold(
            input_checkpoint_directory=tmp_path / "input",
            expected_input_artifact_sha256="1" * 64,
            output_checkpoint_directory=tmp_path / "output",
            cache_directory=tmp_path / "cache",
            expected_cache_key="2" * 64,
            bridge_directory=tmp_path / "bridge",
            expected_bridge_artifact_sha256="3" * 64,
            donor_tokenizer=object(),
            target_tokenizer=object(),
            parent_model_directory=tmp_path / "parent",
            compiler_diagnostic_json=tmp_path / "compiler.json",
            expected_compiler_diagnostic_sha256="4" * 64,
            device="cpu",
            global_residual_closure_version="output_writer_v1_diagnostic",
            global_residual_basis_rank=776,
        )


def _minimal_postfold_run_arguments(tmp_path: Path) -> dict[str, object]:
    return {
        "input_checkpoint_directory": tmp_path / "input",
        "expected_input_artifact_sha256": "1" * 64,
        "output_checkpoint_directory": tmp_path / "output",
        "cache_directory": tmp_path / "cache",
        "expected_cache_key": "2" * 64,
        "bridge_directory": tmp_path / "bridge",
        "expected_bridge_artifact_sha256": "3" * 64,
        "donor_tokenizer": object(),
        "target_tokenizer": object(),
        "parent_model_directory": tmp_path / "parent",
        "compiler_diagnostic_json": tmp_path / "compiler.json",
        "expected_compiler_diagnostic_sha256": "4" * 64,
        "device": "cpu",
    }


def _cached_fused_terminal_evidence(
    *,
    numerically_close: bool = True,
    disagreement_envelope_verified: bool = True,
) -> dict[str, object]:
    cache_key = "1" * 64
    manifest_sha256 = "2" * 64
    completion_sha256 = "3" * 64
    record_sha256 = "4" * 64
    record_file_sha256 = "5" * 64

    def auth(body: dict[str, object], seal: str) -> dict[str, object]:
        return {**body, seal: canonical_sha256(body)}

    phases = []
    for phase, rows, disagreements, forced_sha in (
        ("cached_prefill", 2, 0, None),
        ("decode.000", 1, 1, "a" * 64),
    ):
        phase_envelope_verified = (
            True if disagreements == 0 else disagreement_envelope_verified
        )
        phase_body = {
            "phase": phase,
            "cached_logits_sha256": "6" * 64,
            "full_prefix_logits_sha256": "7" * 64,
            "row_count": rows,
            "maximum_absolute_error": 0.125,
            "numerically_close": numerically_close,
            "top1_exact_agreement": disagreements == 0,
            "top1_disagreement_count": disagreements,
            "all_top1_disagreements_within_numeric_error_envelope": (
                phase_envelope_verified
            ),
            "forced_token_sha256": forced_sha,
        }
        phases.append(auth(phase_body, "authentication_sha256"))
    raw_body = {
        "batch_index": 0,
        "decode_trace_mode": "cached_verified",
        "raw_cached_full_prefix_pair_count": 2,
        "raw_logit_rows_verified": 3,
        "phase_verification": phases,
        "phase_authentication_sha256": [
            value["authentication_sha256"] for value in phases
        ],
        "top1_disagreement_count": 1,
        "all_logits_numerically_close": numerically_close,
        "all_top1_disagreements_within_numeric_error_envelope": (
            disagreement_envelope_verified
        ),
        "top1_exactness_is_diagnostic_only": True,
    }
    raw = auth(raw_body, "authentication_sha256")
    batch_compatibility = {
        "batch_index": 0,
        "decode_trace_mode": "cached_verified",
        "verification_phase_count": 2,
        "verification_phase_order": ["cached_prefill", "decode.000"],
        "top1_disagreement_count": 1,
        "cached_logits_numerically_verified": numerically_close,
        "all_top1_disagreements_numeric_envelope_verified": (
            disagreement_envelope_verified
        ),
    }
    compatibility_body = {
        "schema": postfold.LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA,
        "cache_key": cache_key,
        "manifest_sha256": manifest_sha256,
        "completion_sha256": completion_sha256,
        "batch_count": 1,
        "decode_trace_mode": "cached_verified",
        "ordered_batch_record_sha256": [record_sha256],
        "ordered_batch_record_file_sha256": [record_file_sha256],
        "batch_compatibility": [batch_compatibility],
        "all_cached_logits_numerically_verified": numerically_close,
        "cached_top1_disagreement_count": 1,
        "terminal_trajectory_compatibility_verified": True,
        "forced_decode_tokens_are_caller_supplied": True,
        "top1_exactness_is_diagnostic_only": True,
        "top1_exactness_is_publication_gate": False,
        "all_top1_disagreements_numeric_envelope_verified": (
            disagreement_envelope_verified
        ),
        "full_prefix_recache_required": False,
        "preflight_io_contract": {
            "trace_file_sha256_passes": 0,
            "trace_payload_bytes_read": 0,
            "trace_tensor_payloads_materialized": 0,
        },
    }
    compatibility = auth(compatibility_body, "authentication_sha256")
    session_body = {
        "cache_key": cache_key,
        "manifest_sha256": manifest_sha256,
        "completion_sha256": completion_sha256,
        "ordered_authenticated_batches": [
            {
                "batch_index": 0,
                "record_sha256": record_sha256,
                "record_file_sha256": record_file_sha256,
                "tensor_count": 2,
                "logical_bytes": 128,
            }
        ],
    }
    session = auth(session_body, "session_sha256")
    cache_authentication = {
        "session": session,
        "selected_payloads": [
            {
                "batch_index": 0,
                "authenticated_tensor_count": 2,
                "authenticated_logical_bytes": 128,
                "selective_payload": {
                    "selected_tensor_count": 2,
                    "selected_logical_bytes": 128,
                },
            }
        ],
    }
    payload_load_audit = postfold._summarize_terminal_cache_payload_loads(
        cache_authentication
    )
    binding_body = {
        "compatibility_authentication_sha256": compatibility["authentication_sha256"],
        "payload_session_sha256": session["session_sha256"],
        "manifest_completion_and_ordered_records_exact": True,
        "large_trace_file_sha256_authenticated_once_by_consuming_session": True,
    }
    binding = auth(binding_body, "authentication_sha256")
    cache_parent_fingerprint = "8" * 64
    bridge_artifact_sha256 = "9" * 64
    source_lineage, composite_lineage = _synthetic_terminal_lineages(
        cache_key=cache_key,
        cache_parent_fingerprint=cache_parent_fingerprint,
        bridge_artifact_sha256=bridge_artifact_sha256,
        compatibility_sha256=compatibility["authentication_sha256"],
        payload_binding_sha256=binding["authentication_sha256"],
        session_sha256=session["session_sha256"],
    )
    io_body = {
        "metadata_preflight_trace_file_sha256_passes": 0,
        "metadata_preflight_trace_payload_bytes_read": 0,
        "consuming_trace_file_sha256_passes_per_batch": 1,
        "consuming_trace_file_sha256_passes_total": 1,
        "same_consuming_session_reused_by_semantic_collector": True,
        "same_consuming_session_reused_from_layer_stack": True,
        "same_live_cache_session_as_layer_stack": True,
        "cache_content_identity_matches_layer_stack": True,
        "preterminal_resume_mode": False,
        "fused_terminal_selective_session_constructions": 0,
        "end_to_end_selective_session_constructions": 1,
        "additional_full_trace_file_sha256_passes_performed_by_fused_terminal": 0,
        "selected_tensor_payload_loads_total": 1,
        "selected_tensor_payload_logical_bytes_total": 128,
        "full_tensor_inventory_payload_loads_per_batch": 1,
        "full_tensor_inventory_payload_loads_total": 1,
        "full_tensor_inventory_payload_logical_bytes_total": 128,
    }
    io_audit = auth(io_body, "result_sha256")
    body = {
        "cache_key": cache_key,
        "cache_parent_fingerprint": cache_parent_fingerprint,
        "bridge_artifact_sha256": bridge_artifact_sha256,
        "terminal_cache_compatibility": compatibility,
        "terminal_cache_compatibility_sha256": compatibility["authentication_sha256"],
        "terminal_cache_payload_binding": binding,
        "terminal_source_lineage": source_lineage.as_dict(),
        "terminal_source_lineage_sha256": source_lineage.authentication_sha256,
        "terminal_composite_lineage": composite_lineage.as_dict(),
        "terminal_composite_lineage_sha256": composite_lineage.authentication_sha256,
        "terminal_compiler_binding_sha256": composite_lineage.compiler_binding_sha256,
        "parent_action_projection_sha256": [],
        "parent_action_projection_diagnostics": [],
        "parent_action_projection_numeric_threshold_applied": False,
        "parent_action_projection_metrics_diagnostic_only": True,
        "parent_cache_authentication": cache_authentication,
        "terminal_cache_tensor_payload_load_audit": payload_load_audit,
        "cached_full_prefix_payload_verification": [raw],
        "cached_full_prefix_payload_verification_sha256": [
            raw["authentication_sha256"]
        ],
        "terminal_cache_io_audit": io_audit,
    }
    return auth(body, "result_sha256")


def _reseal(value: dict[str, object], seal: str) -> None:
    value.pop(seal, None)
    value[seal] = canonical_sha256(value)


def test_fused_cache_evidence_rejects_fully_resealed_session_and_phase_drift():
    valid = _cached_fused_terminal_evidence()
    verified = postfold._verify_sealed(
        valid,
        seal_name="result_sha256",
        label="synthetic fused terminal",
    )
    authentication = postfold._authenticate_fused_terminal_cache_evidence(verified)
    assert authentication["every_raw_cached_full_prefix_row_rederived"] is True
    assert authentication["raw_logit_rows_verified"] == 3
    assert authentication["top1_disagreement_count"] == 1

    diagnostic_only = _cached_fused_terminal_evidence(
        numerically_close=False,
        disagreement_envelope_verified=False,
    )
    diagnostic_authentication = postfold._authenticate_fused_terminal_cache_evidence(
        postfold._verify_sealed(
            diagnostic_only,
            seal_name="result_sha256",
            label="numeric-diagnostic-only fused terminal",
        )
    )
    assert diagnostic_authentication["all_logits_numerically_close"] is False
    assert (
        diagnostic_authentication[
            "all_top1_disagreements_within_numeric_error_envelope"
        ]
        is False
    )
    assert (
        diagnostic_authentication["numeric_diagnostics_are_admission_authority"]
        is False
    )

    session_drift = copy.deepcopy(valid)
    binding = session_drift["terminal_cache_payload_binding"]
    assert isinstance(binding, dict)
    binding["payload_session_sha256"] = "f" * 64
    _reseal(binding, "authentication_sha256")
    _reseal(session_drift, "result_sha256")
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="compatibility/session binding is invalid",
    ):
        postfold._authenticate_fused_terminal_cache_evidence(
            postfold._verify_sealed(
                session_drift,
                seal_name="result_sha256",
                label="resealed session drift",
            )
        )

    phase_drift = copy.deepcopy(valid)
    raw = phase_drift["cached_full_prefix_payload_verification"][0]
    phase = raw["phase_verification"][1]
    phase["numerically_close"] = False
    _reseal(phase, "authentication_sha256")
    raw["phase_authentication_sha256"][1] = phase["authentication_sha256"]
    _reseal(raw, "authentication_sha256")
    phase_drift["cached_full_prefix_payload_verification_sha256"][0] = raw[
        "authentication_sha256"
    ]
    _reseal(phase_drift, "result_sha256")
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="phase receipt is inconsistent",
    ):
        postfold._authenticate_fused_terminal_cache_evidence(
            postfold._verify_sealed(
                phase_drift,
                seal_name="result_sha256",
                label="resealed phase drift",
            )
        )


@pytest.mark.parametrize(
    ("statistics_directory", "statistics_sha256"),
    (
        (Path("unused-statistics"), None),
        (None, "a" * 64),
    ),
)
def test_top_level_rejects_unpaired_interface_statistics_before_dispatch(
    tmp_path: Path,
    monkeypatch,
    statistics_directory: Path | None,
    statistics_sha256: str | None,
) -> None:
    def unexpected_dispatch(*_args, **_kwargs):
        pytest.fail("unpaired statistics reached ancestry or asset dispatch")

    monkeypatch.setattr(
        postfold,
        "_claims_semantic_foundation_schema",
        unexpected_dispatch,
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="must be supplied together",
    ):
        postfold.run_lfm2_semantic_postfold(
            **_minimal_postfold_run_arguments(tmp_path),
            interface_statistics_directory=statistics_directory,
            expected_interface_statistics_sha256=statistics_sha256,
        )


def test_top_level_exporter_factory_requires_capsule_before_asset_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        postfold,
        "_claims_semantic_foundation_schema",
        lambda *_args, **_kwargs: pytest.fail(
            "missing capsule reached ancestry or asset dispatch"
        ),
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="requires a preterminal capsule directory",
    ):
        postfold.run_lfm2_semantic_postfold(
            **_minimal_postfold_run_arguments(tmp_path),
            terminal_frontier_exporter_factory=(lambda _reference: lambda *_args: {}),
        )


def test_top_level_rejects_direct_exporter_and_factory_before_asset_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        postfold,
        "_claims_semantic_foundation_schema",
        lambda *_args, **_kwargs: pytest.fail(
            "exporter conflict reached ancestry or asset dispatch"
        ),
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="mutually exclusive",
    ):
        postfold.run_lfm2_semantic_postfold(
            **_minimal_postfold_run_arguments(tmp_path),
            preterminal_capsule_directory=tmp_path / "capsule",
            terminal_frontier_exporter=lambda *_args: {},
            terminal_frontier_exporter_factory=(lambda _reference: lambda *_args: {}),
        )


def test_top_level_rejects_statistics_for_nonfoundation_before_output_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    diagnostic = tmp_path / "compiler.json"
    diagnostic.write_text("{}", encoding="ascii")

    def unexpected_output_dispatch(*_args, **_kwargs):
        pytest.fail("nonfoundation statistics reached output or asset dispatch")

    monkeypatch.setattr(postfold, "_new_output_path", unexpected_output_dispatch)

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="only for semantic-foundation ancestry",
    ):
        postfold.run_lfm2_semantic_postfold(
            **_minimal_postfold_run_arguments(tmp_path),
            interface_statistics_directory=tmp_path / "statistics",
            expected_interface_statistics_sha256="a" * 64,
        )


@pytest.mark.parametrize("use_foundation_statistics", (False, True))
def test_top_level_real_fused_terminal_exports_frontier_then_stops_for_review(
    tmp_path: Path,
    monkeypatch,
    use_foundation_statistics: bool,
) -> None:
    events: list[str] = []
    recipe = postfold.resolve_lfm2_semantic_body_recipe(
        "structural_sequence_residual_mlp_writer"
    )
    seed, evidence, _fixture_result = _compiled_exact_zero_terminal_result()
    cache_key = "1" * 64
    cache_parent = "2" * 64
    cache_topology = "3" * 64
    bridge_sha256 = "4" * 64
    session_sha256 = "5" * 64
    layer_state_before = "6" * 64
    layer_state_after = "7" * 64
    target_model = _TerminalTarget(seed.down_weight)
    target_state_before = _terminal_target_state(target_model)
    target_parameter_identity_before = tuple(
        (
            name,
            id(parameter),
            int(parameter.data_ptr()),
            int(parameter.untyped_storage().data_ptr()),
            id(parameter.grad),
        )
        for name, parameter in target_model.named_parameters()
    )
    target_tokenizer = SimpleNamespace(name="target-tokenizer")
    saved_tokenizer = SimpleNamespace(name="saved-tokenizer")
    bridge = SimpleNamespace(artifact_sha256=bridge_sha256)
    source_table = seed.source_tied_table.detach().clone()
    interface_report = {
        "report_sha256": "8" * 64,
    }
    interface = postfold.TiedCausalInterfaceFold(
        target_weight=seed.target_tied_weight.to(torch.float64),
        producer_encoder=torch.eye(seed.target_hidden_size, dtype=torch.float64),
        report=interface_report,
    )
    compatibility_body = {
        "schema": postfold.LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA,
        "cache_key": cache_key,
    }
    compatibility = {
        **compatibility_body,
        "authentication_sha256": canonical_sha256(compatibility_body),
    }
    payload_binding_body = {
        "compatibility_authentication_sha256": compatibility["authentication_sha256"],
        "payload_session_sha256": session_sha256,
    }
    payload_binding = {
        **payload_binding_body,
        "authentication_sha256": canonical_sha256(payload_binding_body),
    }
    source_lineage, composite_lineage = _synthetic_terminal_lineages(
        cache_key=cache_key,
        cache_parent_fingerprint=cache_parent,
        bridge_artifact_sha256=bridge_sha256,
        compatibility_sha256=compatibility["authentication_sha256"],
        payload_binding_sha256=payload_binding["authentication_sha256"],
        session_sha256=session_sha256,
        interface_report_sha256=interface_report["report_sha256"],
    )
    parent_final_norm = Lfm2SealedParentFinalRmsNorm(
        cache_key=cache_key,
        parent_fingerprint=cache_parent,
        layer_topology_fingerprint=cache_topology,
        weight=seed.final_norm_weight,
        epsilon=seed.norm_epsilon,
    )
    parent_load_body = {
        "cache_key": cache_key,
        "cache_parent_fingerprint": cache_parent,
        "final_norm_authentication_sha256": (parent_final_norm.authentication_sha256),
        "operation_audit": {
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
            "parent_model_generate_calls": 0,
            "network_access_performed": False,
            "direct_local_json_and_safetensors_only": True,
        },
    }
    parent_load_sha256 = canonical_sha256(parent_load_body)
    parent_norm = postfold.Lfm2LocalParentFinalRmsNormLoad(
        final_norm=parent_final_norm,
        report={**parent_load_body, "result_sha256": parent_load_sha256},
        result_sha256=parent_load_sha256,
    )
    coverage_body = {
        "schema": "neural-forge-lfm2-dense-producer-chain-coverage-v1",
        "layer_index": 0,
        "producer_kind": "mlp",
        "deferred_roles": ["feed_forward.w2.weight"],
        "all_required_roles_structurally_nonzero": True,
        "blank_or_partially_blank_required_role_count": 0,
    }
    coverage = {
        **coverage_body,
        "result_sha256": canonical_sha256(coverage_body),
    }
    evidence_sha256 = {
        split: evidence[split].authentication_sha256 for split in sorted(evidence)
    }
    deferred_w2_sha256 = postfold._tensor_sha256(
        target_model.model.layers[-1].feed_forward.w2.weight
    )
    layer_body = {
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "interface_anchor_selection_sha256": None,
        "final_layer_mlp_w2_deferred_to_fused_terminal": True,
        "final_layer_mlp_w2_install_count": 0,
        "final_layer_mlp_evidence_sha256": evidence_sha256,
        "cache_selective_session_sha256": session_sha256,
        "model_state_before_sha256": layer_state_before,
        "model_state_after_sha256": layer_state_after,
        "deferred_final_w2_sha256": deferred_w2_sha256,
        "complete_causal_layer_stack_adjudicated": True,
        "complete_causal_layer_stack_installed": False,
        "whole_stage_contains_meaningful_parameter_changes": True,
        "changed_state_keys": ["model.layers.0.synthetic-folded-writer"],
        "complete_nonzero_structural_producer_coverage": True,
        "blank_or_partially_blank_producer_chain_count": 0,
        "attention_operator_decision_count": 6,
        "attention_operator_install_count": 5,
        "short_conv_operator_decision_count": 8,
        "short_conv_operator_install_count": 8,
        "layer_count": 1,
        "operator_layer_kinds": ["full_attention"],
        "attention_layer_indices": [0],
        "short_conv_layer_indices": [],
        "terminal_layer_index": 0,
        "dynamic_layer_count_derived_from_authenticated_cache_and_child_topology": (
            True
        ),
        "layer_installations": [
            {
                "operator_kind": "full_attention",
                "mlp_producer_coverage_decision": {
                    "coverage_after_sha256": coverage["result_sha256"]
                },
            }
        ],
    }
    layer_receipt = {
        **layer_body,
        "result_sha256": canonical_sha256(layer_body),
    }
    layer_stack = postfold.Lfm2SemanticLayerStackInstallation(
        receipt=layer_receipt,
        final_layer_mlp_evidence=evidence,
    )
    attention_body = {
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "attention_receipt_count": 6,
        "complete_six_tensor_bundle_count": 6,
    }
    attention_authentication = {
        **attention_body,
        "result_sha256": canonical_sha256(attention_body),
    }
    gauge_body = {
        "schema": "neural-forge-lfm2-self-dual-tied-gauge-install-v1",
        "global_residual_gauge_frozen": True,
    }
    gauge_receipt = {
        **gauge_body,
        "result_sha256": canonical_sha256(gauge_body),
    }
    tied_authentication_body = {
        "schema": "neural-forge-lfm2-fused-terminal-tied-interface-v1",
        "interface_report_sha256": interface_report["report_sha256"],
    }
    tied_authentication = {
        **tied_authentication_body,
        "result_sha256": canonical_sha256(tied_authentication_body),
    }

    session_receipt = {
        "session_sha256": session_sha256,
        "cache_key": cache_key,
    }

    class PipelineCacheSession:
        def __enter__(self):
            events.append("cache-session-open")
            return self

        def __exit__(self, *_args):
            events.append("cache-session-close")
            return None

        def verify(self):
            events.append("cache-session-verify")
            return self

        def as_dict(self):
            return session_receipt

    pipeline_cache_session = PipelineCacheSession()
    cache = SimpleNamespace(
        cache_key=cache_key,
        parent_fingerprint=cache_parent,
        layer_topology_fingerprint=cache_topology,
        batch_count=4,
        manifest={"manifest_sha256": "9" * 64},
        selective_authenticated_session=lambda: pipeline_cache_session,
    )
    plan = SimpleNamespace(
        assignments={
            0: "fit",
            1: "fit_check",
            2: "selection",
            3: "validation",
        },
        verify_for=lambda observed_cache: (
            None
            if observed_cache is cache
            else pytest.fail("split plan received the wrong cache")
        ),
    )

    cached = []
    semantic_bridges = []
    split_order = ("fit", "fit_check", "selection", "validation")
    for batch_index, split in enumerate(split_order):
        raw_body = {
            "batch_index": batch_index,
            "split": split,
        }
        raw = {
            **raw_body,
            "authentication_sha256": canonical_sha256(raw_body),
        }
        projection_body = {
            "batch_index": batch_index,
            "diagnostic_only": True,
        }
        projection = {
            **projection_body,
            "authentication_sha256": canonical_sha256(projection_body),
        }
        cached.append(
            SimpleNamespace(
                parent_final_norm_endpoint={
                    "runtime_replay_to_cached_output_verified": True,
                    "authenticated_cached_output_used_as_compiler_target": True,
                    "cached_output_endpoint_available": False,
                },
                cached_full_prefix_payload=raw,
                parent_action_projection=projection,
            )
        )
        semantic_bridges.append(
            SimpleNamespace(
                source_batch_index=batch_index,
                target_position_origin=0,
                phases=(
                    SimpleNamespace(target_full_prefix_length=1),
                    SimpleNamespace(
                        target_full_prefix_length=2,
                        target_full_prefix_token_ids=(batch_index,),
                    ),
                ),
            )
        )

    cache_receipt = {
        "session": session_receipt,
        "selected_payloads": [],
    }
    payload_audit = {
        "every_batch_loaded_full_tensor_inventory_once": True,
        "full_tensor_inventory_payload_load_count": cache.batch_count,
        "full_tensor_inventory_payload_logical_bytes_total": 512,
        "selected_tensor_payload_load_count": cache.batch_count,
        "selected_tensor_payload_logical_bytes_total": 512,
    }
    replay_splits = iter(split_order)

    def replay_lexical_rows(*_args, **_kwargs):
        split = next(replay_splits)
        bank = evidence[split]
        return (
            bank.mlp_input[:1].clone(),
            bank.pre_mlp_residual[:1].clone(),
            {"replay_sha256": canonical_sha256({"split": split})},
        )

    compiler_calls = 0
    real_terminal_compiler = postfold.compile_lfm2_cpcf_terminal_action_residual

    def compile_terminal(**kwargs):
        nonlocal compiler_calls
        compiler_calls += 1
        events.append("closed-form-terminal-compiler")
        return real_terminal_compiler(**kwargs)

    exported: list[tuple[object, object]] = []
    issued_terminal_receipts: list[object] = []

    def export_frontier(result, export_context):
        events.append("frontier-export")
        assert (
            int(
                torch.count_nonzero(
                    target_model.model.layers[-1].feed_forward.w2.weight
                ).item()
            )
            == 0
        )
        assert result.accepted is False
        assert result.selected_candidate_id == "incumbent"
        assert len(result.frontier_down_weights) == 36
        exported.append((result, export_context))
        reference_body = {
            "schema": "fixture-authenticated-terminal-frontier-reference-v1",
            "terminal_result_sha256": result.result_sha256,
            "complete_candidate_count": len(result.frontier_down_weights),
        }
        return {
            **reference_body,
            "receipt_sha256": canonical_sha256(reference_body),
        }

    monkeypatch.setattr(
        postfold,
        "open_lfm2_circuit_trace_cache",
        lambda *_args, **_kwargs: cache,
    )

    def preflight(observed_cache):
        assert observed_cache is cache
        events.append("metadata-preflight")
        return compatibility

    monkeypatch.setattr(
        postfold,
        "preflight_lfm2_long_horizon_terminal_cache",
        preflight,
    )
    monkeypatch.setattr(
        postfold,
        "load_strict_tokenizer_bridge_artifact",
        lambda *_args, **_kwargs: bridge,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_cpcf_v5_standalone_child",
        lambda *_args, **_kwargs: (
            target_model,
            saved_tokenizer,
            {
                "artifact_sha256": "a" * 64,
                "model_state_sha256": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: "c" * 64,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_parent_final_rmsnorm_from_local_safetensors",
        lambda *_args, **_kwargs: parent_norm,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
        lambda *_args, **_kwargs: (
            source_table,
            {"result_sha256": "d" * 64},
        ),
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_semantic_compiler_lineage",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        postfold,
        "rehydrate_lfm2_c61_tied_interface",
        lambda *_args, **_kwargs: (
            interface,
            {
                "compiler_cache_key": "e" * 64,
                "semantic_evidence_cache_key": cache_key,
                "result_sha256": "f" * 64,
            },
        ),
    )
    statistics_sha256 = "0" * 64
    statistics_manifest_sha256 = "1" * 64
    statistics_replay_sha256 = "2" * 64
    foundation_rehydration_body = {
        "schema": "neural-forge-lfm2-semantic-foundation-interface-rehydration-v2",
        "compiler_cache_key": "e" * 64,
        "semantic_evidence_cache_key": cache_key,
        "interface_statistics_artifact_sha256": statistics_sha256,
        "interface_statistics_manifest_file_sha256": statistics_manifest_sha256,
        "interface_statistics_replay_sha256": statistics_replay_sha256,
        "compiler_cache_open_calls": 0,
        "trace_safetensor_open_calls": 0,
        "selective_provider_open_calls": 0,
    }
    foundation_rehydration = {
        **foundation_rehydration_body,
        "result_sha256": canonical_sha256(foundation_rehydration_body),
    }
    foundation_calls: list[dict[str, object]] = []

    def rehydrate_foundation(*_args, **kwargs):
        foundation_calls.append(kwargs)
        return interface, foundation_rehydration

    monkeypatch.setattr(
        postfold,
        "rehydrate_lfm2_semantic_foundation_interface",
        rehydrate_foundation,
    )
    monkeypatch.setattr(
        postfold,
        "install_lfm2_tied_residual_congruence_gauge",
        lambda *_args, **_kwargs: (
            events.append("tied-residual-gauge") or (interface, gauge_receipt)
        ),
    )
    monkeypatch.setattr(
        postfold,
        "build_lfm2_long_horizon_terminal_source_lineage",
        lambda **_kwargs: source_lineage,
    )
    monkeypatch.setattr(
        postfold.Lfm2LongHorizonTerminalSourceLineage,
        "verify_for",
        lambda self, **_kwargs: self,
    )

    def compile_layer_stack(**kwargs):
        events.append("layer-stack")
        assert kwargs["body_recipe"] == recipe
        assert kwargs["authenticated_cache_session"] is pipeline_cache_session
        assert kwargs["fold_compute_device"] == "cuda:1"
        return layer_stack

    monkeypatch.setattr(
        postfold,
        "compile_install_lfm2_semantic_layer_stack_stage",
        compile_layer_stack,
    )
    monkeypatch.setattr(
        postfold,
        "authenticate_lfm2_semantic_layer_stack_attention_receipts",
        lambda _receipt, *, body_recipe: (
            attention_authentication
            if body_recipe == recipe
            else pytest.fail("top-level changed the body recipe")
        ),
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256=layer_state_after),
    )
    monkeypatch.setattr(
        postfold,
        "_lfm2_dense_producer_chain_coverage",
        lambda *_args, **_kwargs: coverage,
    )
    monkeypatch.setattr(
        postfold,
        "_authenticate_fused_terminal_tied_interface",
        lambda **_kwargs: tied_authentication,
    )
    monkeypatch.setattr(
        postfold,
        "_target_runtime_seed",
        lambda *_args, **_kwargs: (seed, seed.down_weight, (), seed.norm_epsilon),
    )
    monkeypatch.setattr(
        postfold,
        "_collect_cached_semantic_trajectories",
        lambda *_args, **_kwargs: (
            cached,
            semantic_bridges,
            cache_receipt,
        ),
    )
    monkeypatch.setattr(
        postfold,
        "_bind_terminal_cache_compatibility_to_payload_session",
        lambda *_args, **_kwargs: payload_binding,
    )
    monkeypatch.setattr(
        postfold,
        "_build_terminal_composite_lineage",
        lambda **_kwargs: composite_lineage,
    )
    monkeypatch.setattr(
        postfold,
        "_summarize_terminal_cache_payload_loads",
        lambda _receipt: payload_audit,
    )
    monkeypatch.setattr(
        postfold,
        "_build_split_evidence",
        lambda *_args, **_kwargs: (
            tuple(SimpleNamespace(split=split) for split in split_order),
            (),
            (),
            0,
        ),
    )
    monkeypatch.setattr(
        postfold,
        "_convert_parent_terminal_evidence",
        lambda value, _encoder: evidence[value.split],
    )
    monkeypatch.setattr(
        postfold,
        "_append_only_lexical_supervision_coordinates",
        lambda semantic: (
            (0,),
            (semantic.source_batch_index % 2,),
            (f"lexical:{semantic.source_batch_index}",),
            (1,),
        ),
    )
    monkeypatch.setattr(postfold, "_replay_lexical_rows", replay_lexical_rows)
    monkeypatch.setattr(
        postfold,
        "compile_lfm2_cpcf_terminal_action_residual",
        compile_terminal,
    )
    monkeypatch.setattr(
        postfold,
        "_authenticate_fused_terminal_cache_evidence",
        lambda _receipt: {
            "terminal_cache_compatibility_sha256": compatibility[
                "authentication_sha256"
            ],
            "terminal_cache_payload_binding_sha256": payload_binding[
                "authentication_sha256"
            ],
            "terminal_cache_session_sha256": session_sha256,
        },
    )
    real_receipt_validator = postfold._validate_fused_semantic_terminal_receipt

    def validate_issued_terminal_receipt(receipt, **kwargs):
        issued_terminal_receipts.append(receipt)
        return real_receipt_validator(receipt, **kwargs)

    monkeypatch.setattr(
        postfold,
        "_validate_fused_semantic_terminal_receipt",
        validate_issued_terminal_receipt,
    )
    monkeypatch.setattr(
        postfold,
        "_install_final_w2_once",
        lambda *_args, **_kwargs: pytest.fail(
            "review-pending orchestration installed final W2"
        ),
    )
    monkeypatch.setattr(
        postfold,
        "publish_lfm2_semantic_postfold_checkpoint",
        lambda **_kwargs: pytest.fail(
            "review-pending orchestration published a checkpoint"
        ),
    )

    for name in ("input", "cache", "bridge", "parent"):
        (tmp_path / name).mkdir()
    compiler_diagnostic = tmp_path / "compiler.json"
    compiler_diagnostic.write_text(
        (
            json.dumps({"schema": postfold.LFM2_SEMANTIC_FOUNDATION_SCHEMA})
            if use_foundation_statistics
            else "{}"
        ),
        encoding="utf-8",
    )
    statistics_arguments = {}
    if use_foundation_statistics:
        (tmp_path / "statistics").mkdir()
        statistics_arguments = {
            "interface_statistics_directory": tmp_path / "statistics",
            "expected_interface_statistics_sha256": statistics_sha256,
        }

    result = postfold.run_lfm2_semantic_postfold(
        input_checkpoint_directory=tmp_path / "input",
        expected_input_artifact_sha256="a" * 64,
        output_checkpoint_directory=tmp_path / "output",
        cache_directory=tmp_path / "cache",
        expected_cache_key=cache_key,
        bridge_directory=tmp_path / "bridge",
        expected_bridge_artifact_sha256=bridge_sha256,
        donor_tokenizer=SimpleNamespace(name="donor-tokenizer"),
        target_tokenizer=target_tokenizer,
        parent_model_directory=tmp_path / "parent",
        compiler_diagnostic_json=compiler_diagnostic,
        expected_compiler_diagnostic_sha256="3" * 64,
        device="cpu",
        fold_compute_device="cuda:1",
        body_recipe=recipe,
        fused_terminal_compiler=(
            postfold.compile_install_lfm2_fused_semantic_terminal_refold
        ),
        terminal_frontier_exporter=export_frontier,
        split_plan=plan,
        **statistics_arguments,
    )

    assert compiler_calls == 1
    assert len(exported) == 1
    assert len(issued_terminal_receipts) == 1
    issued_receipt = issued_terminal_receipts[0]
    assert type(issued_receipt) is postfold._CompilerIssuedFusedSemanticTerminalReceipt
    assert not hasattr(issued_receipt, "__dict__")
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="not compiler-issued"):
        real_receipt_validator(
            dict(issued_receipt),
            layer_receipt=layer_stack.receipt,
            current_model_state_sha256=layer_state_after,
        )
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="unregistered"):
        real_receipt_validator(
            copy.copy(issued_receipt),
            layer_receipt=layer_stack.receipt,
            current_model_state_sha256=layer_state_after,
        )
    with pytest.raises(postfold.Lfm2SemanticPostfoldError, match="real compiler"):
        postfold._CompilerIssuedFusedSemanticTerminalReceipt()

    gc.collect()
    registry_size_before = postfold._fused_semantic_terminal_receipt_registry_size()
    temporary_receipt = postfold._issue_fused_semantic_terminal_receipt(
        dict(issued_receipt)
    )
    temporary_reference = weakref.ref(temporary_receipt)
    assert temporary_reference() is temporary_receipt
    assert (
        postfold._fused_semantic_terminal_receipt_registry_size()
        == registry_size_before + 1
    )
    del temporary_receipt
    gc.collect()
    assert temporary_reference() is None
    assert (
        postfold._fused_semantic_terminal_receipt_registry_size()
        == registry_size_before
    )

    foreign_recipe_id = next(
        value
        for value in postfold.LFM2_SEMANTIC_BODY_RECIPE_IDS
        if value != recipe.recipe_id
    )
    foreign_recipe = postfold.resolve_lfm2_semantic_body_recipe(foreign_recipe_id)
    foreign_receipt_body = copy.deepcopy(dict(issued_receipt))
    foreign_receipt_body.pop("result_sha256")
    foreign_context = foreign_receipt_body["terminal_frontier_export_context"]
    foreign_context.pop("context_sha256")
    foreign_context.update(
        {
            "body_recipe": foreign_recipe.as_dict(),
            "body_recipe_sha256": foreign_recipe.recipe_sha256,
            "body_recipe_id": foreign_recipe.recipe_id,
        }
    )
    foreign_context["context_sha256"] = canonical_sha256(foreign_context)
    foreign_receipt_body["terminal_frontier_export_context_sha256"] = foreign_context[
        "context_sha256"
    ]
    foreign_receipt = postfold._issue_fused_semantic_terminal_receipt(
        {
            **foreign_receipt_body,
            "result_sha256": canonical_sha256(foreign_receipt_body),
        }
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="did not bind the authenticated semantic body recipe",
    ):
        real_receipt_validator(
            foreign_receipt,
            layer_receipt=layer_stack.receipt,
            current_model_state_sha256=layer_state_after,
        )

    def issue_coherently_resealed_terminal_tamper(tamper):
        forged = copy.deepcopy(dict(issued_receipt))
        forged.pop("result_sha256")
        tamper(forged)
        terminal_result = forged.get("terminal_result")
        if isinstance(terminal_result, dict):
            terminal_result.pop("result_sha256", None)
            terminal_result["result_sha256"] = canonical_sha256(terminal_result)
            forged["terminal_action_result_sha256"] = terminal_result["result_sha256"]
            context = forged["terminal_frontier_export_context"]
            context.pop("context_sha256", None)
            context["terminal_result_sha256"] = terminal_result["result_sha256"]
            context["context_sha256"] = canonical_sha256(context)
            forged["terminal_frontier_export_context_sha256"] = context[
                "context_sha256"
            ]
            export = forged["terminal_frontier_export"]
            export.pop("receipt_sha256", None)
            export["terminal_result_sha256"] = terminal_result["result_sha256"]
            export["receipt_sha256"] = canonical_sha256(export)
            forged["terminal_frontier_export_sha256"] = canonical_sha256(export)
        return postfold._issue_fused_semantic_terminal_receipt(
            {**forged, "result_sha256": canonical_sha256(forged)}
        )

    def replace_terminal_selected_and_published_state(value):
        result = value["terminal_result"]
        selected_record = next(
            record
            for record in result["frontier"][1:]
            if record["structurally_eligible"] is True
        )
        replacement_id = selected_record["candidate_id"]
        replacement_sha256 = selected_record["state_sha256"]
        result.update(
            {
                "selected_candidate_id": replacement_id,
                "selected_state_sha256": replacement_sha256,
                "selected_nonincumbent": True,
                "accepted": True,
                "published_is_incumbent": False,
                "final_selected_candidate_id": replacement_id,
                "final_selection_performed": True,
                "candidate_published_before_manual_review": True,
                "published_writer_is_incumbent_pending_manual_review": False,
                "published_writer_is_structurally_safe_effectful_nonincumbent": True,
            }
        )
        result["output_state_sha256"]["selected"] = replacement_sha256
        result["output_state_sha256"]["published"] = replacement_sha256
        value["terminal_action_selected_w2_sha256"] = replacement_sha256
        value["terminal_action_published_w2_sha256"] = replacement_sha256
        value["final_selected_candidate_id"] = replacement_id
        value["final_selection_performed"] = True
        value["candidate_published_before_manual_review"] = True
        value["accepted"] = True
        value["terminal_publication_decision_accepted"] = True
        value["terminal_writer_install_count"] = 1
        value["terminal_writer_mutation_count"] = 1

    tamper_cases = {
        "missing-terminal-result": lambda value: value.pop("terminal_result"),
        "result-accepted": lambda value: value["terminal_result"].__setitem__(
            "accepted", True
        ),
        "result-selected-nonincumbent": lambda value: value[
            "terminal_result"
        ].__setitem__("selected_nonincumbent", True),
        "result-published-nonincumbent": lambda value: value[
            "terminal_result"
        ].__setitem__("published_is_incumbent", False),
        "result-rollback": lambda value: value["terminal_result"].__setitem__(
            "rollback_performed", True
        ),
        "result-wrong-structural-schema": lambda value: value[
            "terminal_result"
        ].__setitem__("structural_nonveto_schema", "caller-resealed"),
        "validation-not-diagnostic": lambda value: value["terminal_result"][
            "validation"
        ].__setitem__("heldout_diagnostic_only", False),
        "validation-publication-gate": lambda value: value["terminal_result"][
            "validation"
        ].__setitem__("publication_gate_applied", True),
        "validation-candidate-acceptance": lambda value: value["terminal_result"][
            "validation"
        ].__setitem__("used_for_candidate_acceptance", True),
        "receipt-writer-install": lambda value: value.__setitem__(
            "terminal_writer_install_count", 1
        ),
        "receipt-publication-accepted": lambda value: value.__setitem__(
            "terminal_publication_decision_accepted", True
        ),
        "coherent-terminal-state-replacement": (
            replace_terminal_selected_and_published_state
        ),
    }
    accepted_tampers = []
    for tamper_name, tamper in tamper_cases.items():
        forged_receipt = issue_coherently_resealed_terminal_tamper(tamper)
        try:
            real_receipt_validator(
                forged_receipt,
                layer_receipt=layer_stack.receipt,
                current_model_state_sha256=layer_state_after,
            )
        except postfold.Lfm2SemanticPostfoldError:
            continue
        accepted_tampers.append(tamper_name)
    assert not accepted_tampers, (
        "Coherently resealed terminal tamper(s) were accepted: "
        + ", ".join(accepted_tampers)
    )
    assert events == [
        "metadata-preflight",
        "cache-session-open",
        "tied-residual-gauge",
        "layer-stack",
        "metadata-preflight",
        "cache-session-verify",
        "closed-form-terminal-compiler",
        "frontier-export",
        "cache-session-verify",
        "cache-session-close",
    ]
    assert result["status"] == "terminal-frontier-exported-review-pending"
    assert result["final_selection_status"] == (
        "deferred-to-manual-whole-response-review"
    )
    assert result["final_selected_candidate_id"] is None
    assert result["terminal_publication_decision_accepted"] is False
    assert result["terminal_nonincumbent_candidate_accepted"] is False
    assert result["terminal_incumbent_retained"] is True
    assert result["terminal_writer_install_count"] == 0
    assert result["terminal_writer_mutation_count"] == 0
    assert result["checkpoint_published"] is False
    assert result["body_recipe_id"] == recipe.recipe_id
    assert result["body_recipe_sha256"] == recipe.recipe_sha256
    assert result["compiler_cache_key"] == "e" * 64
    assert result["semantic_evidence_cache_key"] == cache_key
    assert result["context_binding"]["fold_compute_device"] == "cuda:1"
    assert result["context_binding"]["terminal_review_pending"] is True
    assert (
        result["pipeline_cache_session_reuse"][
            "same_pipeline_session_reused_by_layer_stack_and_terminal"
        ]
        is True
    )
    assert result["claims"]["training_performed"] is False
    assert result["claims"]["optimizer_steps"] == 0
    assert result["claims"]["backward_calls"] == 0
    assert result["claims"]["parameters_pruned"] == 0
    assert result["claims"]["parameters_quantized"] == 0
    assert result["claims"]["parameters_sparsified"] == 0
    assert (
        result["claims"]["numeric_metrics_have_final_intelligence_authority"] is False
    )
    assert result["claims"]["manual_whole_response_review_required"] is True
    assert not (tmp_path / "output").exists()
    assert _terminal_target_state(target_model) == target_state_before
    assert (
        tuple(
            (
                name,
                id(parameter),
                int(parameter.data_ptr()),
                int(parameter.untyped_storage().data_ptr()),
                id(parameter.grad),
            )
            for name, parameter in target_model.named_parameters()
        )
        == target_parameter_identity_before
    )
    assert (
        int(
            torch.count_nonzero(
                target_model.model.layers[-1].feed_forward.w2.weight
            ).item()
        )
        == 0
    )
    if use_foundation_statistics:
        assert len(foundation_calls) == 1
        assert result["interface_statistics_replay"]["artifact_sha256"] == (
            statistics_sha256
        )
        assert result["interface_statistics_replay"]["replay_sha256"] == (
            statistics_replay_sha256
        )
    else:
        assert foundation_calls == []
        assert "interface_statistics_replay" not in result


@pytest.mark.parametrize("exact_zero", (False, True), ids=("nonzero", "exact-zero"))
def test_terminal_publication_stops_at_manual_whole_response_review(
    exact_zero,
) -> None:
    if exact_zero:
        seed, evidence, terminal = _compiled_exact_zero_terminal_result()
    else:
        seed, evidence, terminal = (
            _compiled_nonzero_terminal_result_with_failed_validation()
        )
    report = terminal.as_dict()

    assert report["review_pending_schema"] == (
        terminal_action.LFM2_CPCF_TERMINAL_ACTION_REVIEW_PENDING_SCHEMA
    )
    assert report["final_selection_status"] == (
        "deferred-to-manual-whole-response-review"
    )
    assert report["final_selected_candidate_id"] is None
    assert report["final_selection_performed"] is False
    assert report["automatic_candidate_selection_performed"] is False
    assert report["automatic_candidate_acceptance_performed"] is False
    assert report["candidate_published_before_manual_review"] is False
    assert report["diagnostic_order_is_provisional"] is True
    assert report["diagnostic_order_has_final_selection_authority"] is False
    assert report["diagnostic_order_candidate_count"] == 36
    assert terminal.selected_candidate_id == "incumbent"
    assert terminal.accepted is False
    assert terminal.changed is False
    assert torch.equal(
        terminal.published_down_weight,
        terminal.incumbent_down_weight,
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="review-pending",
    ):
        postfold._terminal_publication_decision(
            terminal,
            expected_seed=seed,
            expected_evidence=evidence,
        )
