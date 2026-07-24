from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionDownWeightPrior,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_global_linearization import (  # noqa: E402
    Lfm2AttentionParameters,
    Lfm2ShortConvParameters,
    capture_lfm2_macrocell_linearization,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (  # noqa: E402
    resolve_lfm2_semantic_body_recipe,
)


def _alignment(
    batch_index: int,
    boundary: tuple[int | None, ...],
) -> SimpleNamespace:
    donor_count = sum(value is not None for value in boundary)
    final = SimpleNamespace(
        target_to_donor_boundary_positions=boundary,
        donor_full_prefix_tokens=donor_count,
    )
    return SimpleNamespace(
        source_batch_index=batch_index,
        alignment_sha256=f"{batch_index + 1:064x}",
        phases=(final,),
    )


def test_global_row_maps_and_parent_boundaries_are_exact_and_compact() -> None:
    alignments = (
        _alignment(0, (0, None, 1)),
        _alignment(1, (None, 0)),
    )
    accumulators, receipt = stack._prepare_global_residual_accumulators(alignments)

    assert receipt["exact_target_donor_canonical_maps"] is True
    assert receipt["canonical_boundary_row_count"] == 3
    assert accumulators[0].selected_target_positions.tolist() == [0, 2]
    assert accumulators[0].source_donor_positions.tolist() == [0, 1]
    assert accumulators[0].compact_donor_positions.tolist() == [0, 1]
    assert accumulators[0].canonical_boundary_row_indices.tolist() == [0, 1]
    assert accumulators[1].selected_target_positions.tolist() == [1]
    assert accumulators[1].canonical_boundary_row_indices.tolist() == [2]

    chart = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    encoder = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    parent = {
        0: SimpleNamespace(
            final_histories={
                "residual.post_operator": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
                "residual.post_layer": torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
                "output.final_norm_output": torch.tensor([[[9.0, 10.0], [11.0, 12.0]]]),
            }
        ),
        1: SimpleNamespace(
            final_histories={
                "residual.post_operator": torch.tensor([[[2.0, 3.0]]]),
                "residual.post_layer": torch.tensor([[[4.0, 5.0]]]),
                "output.final_norm_output": torch.tensor([[[6.0, 7.0]]]),
            }
        ),
    }
    stack._capture_global_parent_layer_boundaries(
        accumulators,
        parent=parent,
        residual_chart=chart,
        producer_encoder=encoder,
        layer_index=0,
        terminal_layer_index=0,
    )

    torch.testing.assert_close(
        accumulators[0].mapped_parent_post_operator_residuals[0],
        parent[0].final_histories["residual.post_operator"].to(torch.float64) @ chart,
    )
    torch.testing.assert_close(
        accumulators[0].mapped_parent_terminal_final_norm,
        parent[0].final_histories["output.final_norm_output"].to(torch.float64)
        @ encoder,
    )


def test_global_macrocell_capture_uses_live_layers_and_deferred_terminal_w2() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Lfm2Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention"],
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
        dtype="bfloat16",
    )
    model = transformers.Lfm2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    residual = torch.randn((1, 3, 4), dtype=torch.bfloat16)
    phase = SimpleNamespace(
        canonical_target_position_ids=torch.arange(3, dtype=torch.int64)[None],
        canonical_target_attention_mask=torch.ones((1, 3), dtype=torch.int64),
    )
    alignment = SimpleNamespace(phases=(phase,))

    short_parameters, short_mlp = stack._global_macrocell_parameters(
        model,
        layer_index=0,
        residual=residual,
        alignment=alignment,
        deferred_final_w2_prior=None,
    )
    assert isinstance(short_parameters, Lfm2ShortConvParameters)
    assert capture_lfm2_macrocell_linearization(
        residual,
        short_parameters,
        short_mlp,
    ).verify_authentication()

    terminal_layer = model.model.layers[-1]
    live_w2 = terminal_layer.feed_forward.w2.weight
    recipe = resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_structural"
    )
    deferred = (live_w2.detach().clone() + 1).to(torch.bfloat16)
    prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=deferred,
        body_recipe_id=recipe.recipe_id,
        body_recipe_sha256=recipe.recipe_sha256,
        mlp_lane=recipe.component_lanes["mlp"],
        selected_role="full-w1-w3-w2",
        selected_candidate_sha256="1" * 64,
        preparation_sha256="2" * 64,
        mlp_installation_core_sha256="3" * 64,
        live_deferred_w2_sha256=lfm2_cpcf_terminal_action_tensor_sha256(live_w2),
    )
    attention_parameters, terminal_mlp = stack._global_macrocell_parameters(
        model,
        layer_index=1,
        residual=residual,
        alignment=alignment,
        deferred_final_w2_prior=prior,
    )
    assert isinstance(attention_parameters, Lfm2AttentionParameters)
    torch.testing.assert_close(
        terminal_mlp.down_weight,
        deferred.to(torch.float64),
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        terminal_mlp.down_weight,
        live_w2.detach().to(torch.float64),
    )


def test_global_evidence_final_norm_uses_embedding_norm_not_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_gain = torch.tensor([2.0, 3.0])
    poisoned_gain = torch.tensor([99.0, 99.0])
    embedding_norm = SimpleNamespace(weight=embedding_gain)
    target = SimpleNamespace(
        model=SimpleNamespace(
            layers=(object(),),
            embedding_norm=embedding_norm,
            norm=SimpleNamespace(weight=poisoned_gain),
        )
    )
    accumulator = stack._GlobalResidualTrajectoryAccumulator(
        batch_index=0,
        alignment_sha256="1" * 64,
        selected_target_positions=torch.tensor([0]),
        source_donor_positions=torch.tensor([0]),
        compact_donor_positions=torch.tensor([0]),
        canonical_boundary_row_indices=torch.tensor([0]),
        macrocells=[SimpleNamespace(mlp=SimpleNamespace(output=torch.ones(1, 1, 2)))],
        mapped_parent_post_operator_residuals=[torch.ones(1, 1, 2)],
        mapped_parent_post_layer_residuals=[torch.ones(1, 1, 2)],
        mapped_parent_terminal_final_norm=torch.ones(1, 1, 2),
    )
    captured: dict[str, object] = {}

    def terminal_parameters(*, gain: object, epsilon: float) -> object:
        captured["gain"] = gain
        captured["epsilon"] = epsilon
        return object()

    def norm_epsilon(module: object, label: str) -> float:
        assert module is embedding_norm
        assert label == "model.embedding_norm"
        return 0.125

    monkeypatch.setattr(stack, "Lfm2TerminalNormParameters", terminal_parameters)
    monkeypatch.setattr(stack, "_positive_norm_epsilon", norm_epsilon)
    monkeypatch.setattr(
        stack,
        "capture_lfm2_terminal_norm_primal",
        lambda _residual, _parameters: object(),
    )
    monkeypatch.setattr(
        stack,
        "Lfm2StackPrimalBatch",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        stack,
        "Lfm2SemanticGlobalResidualTrajectoryEvidence",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    package = SimpleNamespace(evidence_sha256="e" * 64)
    monkeypatch.setattr(
        stack,
        "Lfm2SemanticGlobalResidualEvidencePackage",
        lambda **_kwargs: package,
    )
    monkeypatch.setattr(
        stack,
        "_global_live_writer_state",
        lambda _model: (MappingProxyType({}), MappingProxyType({}), 2),
    )
    installation = object()
    monkeypatch.setattr(
        stack,
        "compile_install_lfm2_semantic_global_residual_closure",
        lambda *_args, **_kwargs: installation,
    )

    observed, evidence = (
        stack._compile_install_global_residual_v1_diagnostic_from_accumulators(
            MappingProxyType({0: accumulator}),
            target_model=target,
            mapped_parent_tied_table=torch.eye(2),
            deferred_final_w2_prior=object(),
            deferred_final_writer_parameter=torch.nn.Parameter(torch.eye(2)),
        )
    )

    assert observed is installation
    assert evidence is package
    assert captured["gain"] is embedding_gain
    assert captured["epsilon"] == 0.125


def _valid_global_transition_document() -> tuple[dict[str, object], object]:
    recipe = resolve_lfm2_semantic_body_recipe(
        "nonlinear_qdeim_coupled_attention_structural"
    )
    site_ids = [f"model.dynamic.writer.{index}" for index in range(6)]
    installed_ids = site_ids[:-1]
    before = {name: "1" * 64 for name in installed_ids}
    after = dict(before)
    after[installed_ids[0]] = "2" * 64
    prior_sha = "3" * 64
    prior_tensor_sha = "4" * 64
    lineage_sha = "5" * 64
    report_body = {
        "evidence_package_sha256": "6" * 64,
        "closure_result_sha256": "7" * 64,
        "closure_scope": "6_output_writers_only_baseline_v1",
        "writer_site_count": len(site_ids),
        "site_specifications": [{"site_id": name} for name in site_ids],
        "installed_writer_count": len(installed_ids),
        "installed_writer_site_ids": installed_ids,
        "dense_live_writer_sha256_before": before,
        "dense_live_writer_sha256_after": after,
        "final_w2_site_id": site_ids[-1],
        "final_w2_unchanged": True,
        "final_w2_installation_count": 0,
        "quality_gate": False,
        "numeric_diagnostics_used_for_auto_veto": False,
        "verdict_required_for_continuation": False,
        "corrected_final_w2_prior_sha256": prior_sha,
        "final_layer_mlp_down_weight_prior_authentication_sha256": prior_sha,
        "final_layer_mlp_down_weight_prior_tensor_sha256": prior_tensor_sha,
        "final_layer_mlp_down_weight_prior_install_count": 0,
        "final_layer_mlp_down_weight_prior_closure_lineage_sha256": lineage_sha,
        "corrected_final_w2_closure_lineage_sha256": lineage_sha,
        "v2_input_transport_extension_roadmap": {
            "implemented_in_this_result": False,
        },
    }
    report = {**report_body, "report_sha256": canonical_sha256(report_body)}
    row_body = {
        "schema": stack.LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
        "kind": "global_residual_semantic_row_maps",
        "trajectory_count": 2,
        "canonical_boundary_row_count": 3,
        "trajectories": [
            {
                "batch_index": 0,
                "alignment_sha256": "8" * 64,
                "selected_target_positions": [0, 2],
                "source_donor_positions": [0, 1],
                "compact_donor_positions": [0, 1],
                "canonical_boundary_row_indices": [0, 1],
            },
            {
                "batch_index": 1,
                "alignment_sha256": "9" * 64,
                "selected_target_positions": [1],
                "source_donor_positions": [0],
                "compact_donor_positions": [0],
                "canonical_boundary_row_indices": [2],
            },
        ],
        "exact_target_donor_canonical_maps": True,
        "parent_histories_compacted_to_selected_donor_rows": True,
        "numeric_or_manual_gate_used": False,
    }
    row_map = {**row_body, "result_sha256": canonical_sha256(row_body)}
    local_changed = ["model.local.writer"]
    closure_changed = [installed_ids[0]]
    composed = sorted(local_changed + closure_changed)
    stack_document: dict[str, object] = {
        "global_residual_closure_required_by_body_recipe": True,
        "global_residual_closure_applied": True,
        "global_residual_closure_version": "output_writer_v1_diagnostic",
        "global_residual_backend": "auto",
        "global_residual_requested_basis_rank": None,
        "global_residual_correction_scale_frontier": [1.0],
        "global_residual_correction_scale": 1.0,
        "global_residual_correction_scale_policy_sha256": (
            stack.Lfm2ProducerAwareCorrectionScalePolicy().authentication_sha256
        ),
        "global_residual_closure_policy_sha256": (
            stack.Lfm2GlobalResidualClosurePolicy().authentication_sha256
        ),
        "global_residual_closure_scope": report_body["closure_scope"],
        "global_residual_closure_evidence_sha256": report_body[
            "evidence_package_sha256"
        ],
        "global_residual_semantic_row_map_sha256": row_map["result_sha256"],
        "global_residual_semantic_row_maps": row_map,
        "global_residual_closure_receipt_sha256": report["report_sha256"],
        "global_residual_closure_receipt": report,
        "global_residual_closure_result_sha256": report_body["closure_result_sha256"],
        "global_residual_corrected_final_w2_prior_sha256": prior_sha,
        "global_residual_corrected_final_w2_closure_lineage_sha256": lineage_sha,
        "final_layer_mlp_down_weight_prior_authentication_sha256": prior_sha,
        "final_layer_mlp_down_weight_prior_tensor_sha256": prior_tensor_sha,
        "final_layer_mlp_down_weight_prior_closure_lineage_sha256": lineage_sha,
        "global_residual_tied_table_source": (
            "authenticated_live_interface_target_weight_child_coordinates"
        ),
        "causal_layer_stack_model_state_after_sha256": "a" * 64,
        "causal_layer_stack_changed_state_keys": local_changed,
        "global_residual_closure_model_state_before_sha256": "a" * 64,
        "global_residual_closure_model_state_after_sha256": "b" * 64,
        "model_state_after_sha256": "b" * 64,
        "global_residual_closure_changed_state_keys": closure_changed,
        "composed_transition_changed_state_keys": composed,
        "changed_state_keys": composed,
        "global_residual_closure_live_final_w2_unchanged": True,
        "global_residual_closure_parent_histories_compacted": True,
        "global_residual_v1_shared_attention_head_dimension": True,
        "global_residual_general_per_site_head_dimensions_claimed": False,
        "global_residual_producer_aware_v2a": False,
        "global_residual_v1_diagnostic_baseline": True,
        "global_residual_closure_numeric_quality_gate": False,
        "global_residual_closure_manual_quality_gate": False,
        "decoded_behavior_review_is_evidence_for_understanding": True,
        "decoded_behavior_verdict_required_for_continuation": False,
    }
    return stack_document, recipe


def test_global_transition_authenticates_two_states_without_quality_gate() -> None:
    document, recipe = _valid_global_transition_document()

    result = postfold._authenticate_semantic_global_residual_transition(
        document,
        recipe,
    )

    assert result is not None
    assert result["two_transition_chain_authenticated"] is True
    assert result["numeric_or_manual_quality_gate_used"] is False


def _valid_v2_global_transition_document() -> tuple[dict[str, object], object]:
    document, recipe = _valid_global_transition_document()
    layer_count = 2
    site_specifications = [
        {
            "site_id": f"layer.{index // 4}.site.{index % 4}",
            "layer_index": index // 4,
            "causal_order": 6 * (index // 4) + (0, 1, 3, 4)[index % 4],
            "writer_kind": "test",
            "input_width": 2,
        }
        for index in range(4 * layer_count)
    ]
    topology_body = {
        "schema": "test-writer-topology-v1",
        "kind": "writer_topology",
        "hidden_size": 4,
        "head_dimension": 2,
        "requested_basis_rank": None,
        "topology_basis_rank_formula": "2*head_dimension",
        "topology_basis_rank": 4,
        "effective_basis_rank": 4,
        "sites": site_specifications,
    }
    topology = {
        **topology_body,
        "topology_sha256": canonical_sha256(topology_body),
    }
    parameter_ids = [
        "model.layers.0.conv.in_proj.weight",
        "model.layers.0.conv.out_proj.weight",
        "model.layers.0.feed_forward.w1.weight",
        "model.layers.0.feed_forward.w3.weight",
        "model.layers.0.feed_forward.w2.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
        "model.layers.1.self_attn.out_proj.weight",
        "model.layers.1.feed_forward.w1.weight",
        "model.layers.1.feed_forward.w3.weight",
        "model.layers.1.feed_forward.w2.weight",
    ]
    roles = [
        "operator_input_weight",
        "operator_output_weight",
        "mlp_gate_weight",
        "mlp_up_weight",
        "mlp_down_weight",
        "query_weight",
        "key_weight",
        "value_weight",
        "operator_output_weight",
        "mlp_gate_weight",
        "mlp_up_weight",
        "mlp_down_weight",
    ]
    layer_indices = [0] * 5 + [1] * 7
    final_id = parameter_ids[-1]
    installed_ids = parameter_ids[:-1]
    before = {name: "1" * 64 for name in installed_ids}
    after = {
        name: (f"{index + 2:x}" * 64)[:64] for index, name in enumerate(installed_ids)
    }
    compute_body = {
        "schema": "neural-forge-lfm2-producer-aware-dense-absorption-compute-v1",
        "kind": "producer_aware_dense_absorption_compute_execution",
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "backend": "cpu",
        "compute_dtype": "float64",
        "archive_devices": {
            "layer_seed_weights": "cpu",
            "output_basis": "cpu",
            "coefficient_blocks": "cpu",
        },
        "output_devices": {
            "transaction_seed_weights": "cpu",
            "dense_corrected_weights": "cpu",
        },
        "output_basis_staged_to_resolved_device": True,
        "coefficient_blocks_staged_to_resolved_device": True,
        "active_layer_seed_weights_staged_to_resolved_device": True,
        "active_layers_staged_one_at_a_time": True,
        "gpu_is_ephemeral_dense_algebra_workspace": False,
        "system_memory_is_immutable_archive": True,
        "silent_device_fallback_performed": False,
        "cuda_identity": None,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    compute_receipt = {
        **compute_body,
        "report_sha256": canonical_sha256(compute_body),
    }
    representability_frontier = [
        {
            "level": "first_effectful",
            "ray_multiplier": 1.0,
            "selected_for_this_transaction": True,
        }
    ]
    representability_body = {
        "schema": "neural-forge-lfm2-producer-aware-global-linearization-v2a",
        "kind": "target_dtype_global_ray_representability_projection",
        "selected_level": "first_effectful",
        "selected_ray_multiplier": 1.0,
        "frontier": representability_frontier,
        "requested_analytic_nonzero_coordinate_count": 1,
        "required_analytic_nonzero_coordinate_count": 1,
        "selected_changed_coordinate_count": 1,
        "selected_changed_required_coordinate_count": 1,
        "continuous_analytic_ray_preserved_exactly": True,
        "nextafter_values_installed_as_weights": False,
        "numeric_or_manual_quality_gate": False,
    }
    representability_receipt = {
        **representability_body,
        "report_sha256": canonical_sha256(representability_body),
    }
    dense_body = {
        "parameter_ids": parameter_ids,
        "roles": roles,
        "layer_indices": layer_indices,
        "compute_execution_receipt_sha256": compute_receipt["report_sha256"],
        "compute_execution_receipt": compute_receipt,
        "target_dtype_representability_sha256": representability_receipt[
            "report_sha256"
        ],
        "target_dtype_representability": representability_receipt,
        "target_dtype_representability_level": "first_effectful",
        "target_dtype_representability_ray_multiplier": 1.0,
        "target_dtype_representability_frontier": representability_frontier,
        "target_dtype_changed_parameter_ids": parameter_ids,
    }
    dense_receipt = {
        **dense_body,
        "report_sha256": canonical_sha256(dense_body),
    }
    installation_body = {
        "transaction_report_sha256": dense_receipt["report_sha256"],
        "installed_parameter_ids": installed_ids,
        "deferred_parameter_ids": [final_id],
        "post_install_sha256": after,
        "all_nondeferred_physical_parameters_installed": True,
    }
    installation_receipt = {
        **installation_body,
        "report_sha256": canonical_sha256(installation_body),
    }
    scale_policy = stack.Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=(0.5, 1.0),
        materialization_scale=0.5,
    )
    closure_policy = stack.Lfm2GlobalResidualClosurePolicy(backend="auto")
    linearized_body = {
        "schema": "neural-forge-lfm2-producer-aware-semantic-integration-v2a",
        "kind": "target_dtype_representability_linearized_residual_diagnostics",
        "diagnostics_are_advisory_only": True,
        "numeric_or_manual_quality_gate": False,
        "behavioral_winner_claimed": False,
    }
    linearized_receipt = {
        **linearized_body,
        "report_sha256": canonical_sha256(linearized_body),
    }
    physical_specs = [
        {
            "parameter_id": parameter_id,
            "layer_index": layer_index,
            "role": role,
            "seed_sha256": "1" * 64,
            "corrected_sha256": (
                "f" * 64 if parameter_id == final_id else after[parameter_id]
            ),
            "installed": parameter_id != final_id,
            "deferred": parameter_id == final_id,
        }
        for parameter_id, layer_index, role in zip(
            parameter_ids,
            layer_indices,
            roles,
            strict=True,
        )
    ]
    prior_sha = document["global_residual_corrected_final_w2_prior_sha256"]
    prior_tensor_sha = document["final_layer_mlp_down_weight_prior_tensor_sha256"]
    lineage_sha = document["global_residual_corrected_final_w2_closure_lineage_sha256"]
    report_body = {
        "schema": "neural-forge-lfm2-producer-aware-semantic-integration-v2a",
        "evidence_package_sha256": "6" * 64,
        "closure_result_sha256": "7" * 64,
        "closure_scope": "8_producer_aware_input_output_sites_v2a",
        "closure_backend": "producer_aware_v2a",
        "global_residual_backend": closure_policy.backend,
        "closure_policy": closure_policy.as_dict(),
        "closure_policy_sha256": closure_policy.authentication_sha256,
        "correction_scale_policy": {
            **scale_policy.authentication_body(),
            "authentication_sha256": scale_policy.authentication_sha256,
        },
        "correction_scale_policy_sha256": scale_policy.authentication_sha256,
        "materialization_frontier": [0.5, 1.0],
        "materialization_scale": 0.5,
        "materialization_scale_is_execution_variant_not_winner": True,
        "frontier_member_selected_as_winner": False,
        "separate_fold_decode_retain_required_for_each_frontier_scale": True,
        "layer_count": layer_count,
        "writer_site_count": 4 * layer_count,
        "four_sites_per_layer": True,
        "dynamic_layer_count_derived_from_authenticated_evidence": True,
        "site_specifications": site_specifications,
        "topology_receipt": topology,
        "topology_sha256": topology["topology_sha256"],
        "requested_basis_rank": None,
        "topology_basis_rank_formula": "2*head_dimension",
        "topology_basis_rank": 4,
        "effective_basis_rank": 4,
        "basis_rank": 4,
        "dense_transaction_receipt": dense_receipt,
        "dense_transaction_sha256": dense_receipt["report_sha256"],
        "target_dtype_representability_sha256": representability_receipt[
            "report_sha256"
        ],
        "target_dtype_representability": representability_receipt,
        "target_dtype_representability_level": "first_effectful",
        "target_dtype_representability_ray_multiplier": 1.0,
        "target_dtype_representability_frontier": representability_frontier,
        "linearized_residual_diagnostics_sha256": linearized_receipt[
            "report_sha256"
        ],
        "linearized_residual_diagnostics": linearized_receipt,
        "physical_installation_receipt": installation_receipt,
        "installation_receipt_sha256": installation_receipt["report_sha256"],
        "physical_parameter_specifications": physical_specs,
        "installed_parameter_ids": installed_ids,
        "installed_parameter_count": len(installed_ids),
        "all_nondeferred_physical_parameters_installed": True,
        "dense_live_physical_sha256_before": before,
        "dense_live_physical_sha256_after": after,
        "changed_physical_parameter_ids": sorted(installed_ids),
        "materialization_compute_device": "cpu",
        "materialization_compute_execution_receipt_sha256": compute_receipt[
            "report_sha256"
        ],
        "materialization_compute_execution_receipt": compute_receipt,
        "final_w2_parameter_id": final_id,
        "final_w2_unchanged": True,
        "final_w2_installation_count": 0,
        "quality_gate": False,
        "numeric_diagnostics_used_for_auto_veto": False,
        "verdict_required_for_continuation": False,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "corrected_final_w2_prior_sha256": prior_sha,
        "final_layer_mlp_down_weight_prior_authentication_sha256": prior_sha,
        "final_layer_mlp_down_weight_prior_tensor_sha256": prior_tensor_sha,
        "final_layer_mlp_down_weight_prior_install_count": 0,
        "final_layer_mlp_down_weight_prior_closure_lineage_sha256": lineage_sha,
        "corrected_final_w2_closure_lineage_sha256": lineage_sha,
    }
    report = {**report_body, "report_sha256": canonical_sha256(report_body)}
    local_changed = document["causal_layer_stack_changed_state_keys"]
    closure_changed = sorted(installed_ids)
    document.update(
        {
            "global_residual_closure_version": "producer_aware_v2a",
            "global_residual_backend": closure_policy.backend,
            "global_residual_closure_policy_sha256": (
                closure_policy.authentication_sha256
            ),
            "global_residual_correction_scale_frontier": [0.5, 1.0],
            "global_residual_correction_scale": 0.5,
            "global_residual_correction_scale_policy_sha256": (
                scale_policy.authentication_sha256
            ),
            "global_residual_target_dtype_representability_level": (
                "first_effectful"
            ),
            "global_residual_target_dtype_representability_ray_multiplier": 1.0,
            "global_residual_target_dtype_representability_frontier": (
                representability_frontier
            ),
            "global_residual_closure_scope": report_body["closure_scope"],
            "global_residual_closure_receipt": report,
            "global_residual_closure_receipt_sha256": report["report_sha256"],
            "global_residual_closure_changed_state_keys": closure_changed,
            "composed_transition_changed_state_keys": sorted(
                set(local_changed).union(closure_changed)
            ),
            "changed_state_keys": sorted(set(local_changed).union(closure_changed)),
            "global_residual_producer_aware_v2a": True,
            "global_residual_v1_diagnostic_baseline": False,
            "global_residual_dynamic_layer_count": layer_count,
            "global_residual_producer_aware_site_count": 4 * layer_count,
            "global_residual_full_physical_parameter_transaction": True,
            "global_residual_input_and_output_producer_roles_folded": True,
            "global_residual_materialization_scale_is_execution_variant_not_winner": True,
            "global_residual_frontier_member_selected_as_winner": False,
            "global_residual_each_frontier_scale_requires_separate_fold_decode_retain": True,
            "fold_compute_device": "cpu",
            "global_residual_materialization_compute_device": "cpu",
            "global_residual_materialization_compute_execution_receipt_sha256": (
                compute_receipt["report_sha256"]
            ),
            "global_residual_materialization_compute_execution_receipt": (
                compute_receipt
            ),
        }
    )
    document.pop("global_residual_v1_shared_attention_head_dimension")
    document.pop("global_residual_general_per_site_head_dimensions_claimed")
    return document, recipe


def test_producer_aware_v2_transition_authenticates_full_physical_update() -> None:
    document, recipe = _valid_v2_global_transition_document()

    result = postfold._authenticate_semantic_global_residual_transition(
        document,
        recipe,
    )

    assert result is not None
    assert result["global_residual_closure_version"] == "producer_aware_v2a"
    assert result["global_residual_requested_basis_rank"] is None
    assert result["correction_scale_frontier"] == [0.5, 1.0]
    assert result["correction_scale"] == 0.5
    assert result["materialization_compute_device"] == "cpu"
    assert result["frontier_member_selected_as_winner"] is False
    closure = document["global_residual_closure_receipt"]
    assert result["physical_parameter_sha256_before"] == closure[
        "dense_live_physical_sha256_before"
    ]
    assert result["physical_parameter_sha256_after"] == closure[
        "dense_live_physical_sha256_after"
    ]
    assert result["installed_parameter_ids"] == closure[
        "installed_parameter_ids"
    ]


def _producer_coverage(
    role_hashes: dict[str, tuple[str, str]],
    *,
    all_required_roles_structurally_nonzero: bool = True,
) -> dict[str, object]:
    role_receipts = {
        role: {
            "parameter_name": parameter_name,
            "shape": [4, 4],
            "dtype": "bfloat16",
            "element_count": 16,
            "tensor_sha256": tensor_sha256,
        }
        for role, (parameter_name, tensor_sha256) in role_hashes.items()
    }
    body = {
        "layer_index": 0,
        "producer_kind": "short_conv",
        "required_roles": list(role_hashes),
        "deferred_roles": [],
        "role_receipts": role_receipts,
        "all_required_roles_structurally_nonzero": (
            all_required_roles_structurally_nonzero
        ),
        "blank_or_partially_blank_required_role_count": int(
            not all_required_roles_structurally_nonzero
        ),
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def test_live_coverage_authenticates_local_fold_then_effectful_closure() -> None:
    in_name = "model.layers.0.conv.in_proj.weight"
    kernel_name = "model.layers.0.conv.conv.weight"
    out_name = "model.layers.0.conv.out_proj.weight"
    recorded = _producer_coverage(
        {
            "conv.in_proj.weight": (in_name, "1" * 64),
            "conv.conv.weight": (kernel_name, "2" * 64),
            "conv.out_proj.weight": (out_name, "3" * 64),
        }
    )
    live = _producer_coverage(
        {
            "conv.in_proj.weight": (in_name, "4" * 64),
            "conv.conv.weight": (kernel_name, "2" * 64),
            "conv.out_proj.weight": (out_name, "5" * 64),
        }
    )
    transition = {
        "global_residual_closure_version": "producer_aware_v2a",
        "installed_parameter_ids": [in_name, out_name],
        "physical_parameter_sha256_before": {
            in_name: "1" * 64,
            out_name: "3" * 64,
        },
        "physical_parameter_sha256_after": {
            in_name: "4" * 64,
            out_name: "5" * 64,
        },
    }

    assert postfold._closure_adjusted_producer_coverage_matches(
        live,
        recorded,
        transition,
    )

    tampered = dict(transition)
    tampered["physical_parameter_sha256_before"] = {
        in_name: "9" * 64,
        out_name: "3" * 64,
    }
    assert not postfold._closure_adjusted_producer_coverage_matches(
        live,
        recorded,
        tampered,
    )
    structurally_blank = _producer_coverage(
        {
            "conv.in_proj.weight": (in_name, "4" * 64),
            "conv.conv.weight": (kernel_name, "2" * 64),
            "conv.out_proj.weight": (out_name, "5" * 64),
        },
        all_required_roles_structurally_nonzero=False,
    )
    assert not postfold._closure_adjusted_producer_coverage_matches(
        structurally_blank,
        recorded,
        transition,
    )


def test_v1_transition_rejects_an_explicit_residual_basis_rank_claim() -> None:
    document, recipe = _valid_global_transition_document()
    document["global_residual_requested_basis_rank"] = 776

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="cannot claim an explicit",
    ):
        postfold._authenticate_semantic_global_residual_transition(
            document,
            recipe,
        )


def test_v2_transition_rejects_rank_claim_not_bound_to_topology() -> None:
    document, recipe = _valid_v2_global_transition_document()
    document["global_residual_requested_basis_rank"] = 3

    with pytest.raises(postfold.Lfm2SemanticPostfoldError):
        postfold._authenticate_semantic_global_residual_transition(
            document,
            recipe,
        )


def test_v2_transition_authenticates_explicit_rank_through_topology() -> None:
    document, recipe = _valid_v2_global_transition_document()
    report_body = dict(document["global_residual_closure_receipt"])
    report_body.pop("report_sha256")
    topology_body = dict(report_body["topology_receipt"])
    topology_body.pop("topology_sha256")
    topology_body.update(
        {
            "requested_basis_rank": 3,
            "topology_basis_rank_formula": "explicit_requested_basis_rank",
            "topology_basis_rank": 3,
            "effective_basis_rank": 3,
        }
    )
    topology = {
        **topology_body,
        "topology_sha256": canonical_sha256(topology_body),
    }
    report_body.update(
        {
            "topology_receipt": topology,
            "topology_sha256": topology["topology_sha256"],
            "requested_basis_rank": 3,
            "topology_basis_rank_formula": "explicit_requested_basis_rank",
            "topology_basis_rank": 3,
            "effective_basis_rank": 3,
            "basis_rank": 3,
        }
    )
    report = {**report_body, "report_sha256": canonical_sha256(report_body)}
    document["global_residual_requested_basis_rank"] = 3
    document["global_residual_closure_receipt"] = report
    document["global_residual_closure_receipt_sha256"] = report["report_sha256"]

    result = postfold._authenticate_semantic_global_residual_transition(
        document,
        recipe,
    )

    assert result is not None
    assert result["global_residual_requested_basis_rank"] == 3


@pytest.mark.parametrize(
    "tamper",
    ("state_chain", "writer_delta", "row_partition", "quality_gate", "lineage"),
)
def test_global_transition_rejects_each_resealed_tamper_seam(tamper: str) -> None:
    document, recipe = _valid_global_transition_document()
    if tamper == "state_chain":
        document["global_residual_closure_model_state_before_sha256"] = "c" * 64
    elif tamper == "writer_delta":
        document["global_residual_closure_changed_state_keys"] = []
    elif tamper == "row_partition":
        row_map = dict(document["global_residual_semantic_row_maps"])
        row_map.pop("result_sha256")
        rows = [dict(value) for value in row_map["trajectories"]]
        rows[1]["canonical_boundary_row_indices"] = [1]
        row_map["trajectories"] = rows
        resealed = {**row_map, "result_sha256": canonical_sha256(row_map)}
        document["global_residual_semantic_row_maps"] = resealed
        document["global_residual_semantic_row_map_sha256"] = resealed["result_sha256"]
    elif tamper == "quality_gate":
        report = dict(document["global_residual_closure_receipt"])
        report.pop("report_sha256")
        report["quality_gate"] = True
        resealed = {**report, "report_sha256": canonical_sha256(report)}
        document["global_residual_closure_receipt"] = resealed
        document["global_residual_closure_receipt_sha256"] = resealed["report_sha256"]
    else:
        document["final_layer_mlp_down_weight_prior_closure_lineage_sha256"] = "d" * 64

    with pytest.raises(postfold.Lfm2SemanticPostfoldError):
        postfold._authenticate_semantic_global_residual_transition(
            document,
            recipe,
        )


def test_partial_global_transition_is_not_treated_as_legacy() -> None:
    document, recipe = _valid_global_transition_document()
    document.pop("global_residual_closure_applied")

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="partial global residual closure",
    ):
        postfold._authenticate_semantic_global_residual_transition(
            document,
            recipe,
        )
