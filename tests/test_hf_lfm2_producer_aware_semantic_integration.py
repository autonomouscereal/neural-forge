from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")

from neural_forge import (  # noqa: E402
    hf_lfm2_producer_aware_global_linearization as producer_module,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionDownWeightPrior,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_global_linearization import (  # noqa: E402
    Lfm2AttentionParameters,
    Lfm2MlpParameters,
    Lfm2ShortConvParameters,
    Lfm2StackPrimalBatch,
    Lfm2TerminalNormParameters,
    capture_lfm2_macrocell_linearization,
    capture_lfm2_terminal_norm_primal,
)
from neural_forge.hf_lfm2_global_residual_closure import (  # noqa: E402
    Lfm2GlobalResidualClosurePolicy,
    fingerprint_lfm2_global_residual_tensor,
)
from neural_forge.hf_lfm2_producer_aware_global_linearization import (  # noqa: E402
    Lfm2ProducerAwareLinearizationError,
)
from neural_forge.hf_lfm2_producer_aware_semantic_integration import (  # noqa: E402
    Lfm2ProducerAwareCorrectionScalePolicy,
    Lfm2ProducerAwareSemanticAuthenticationError,
    Lfm2ProducerAwareSemanticInstallation,
    Lfm2ProducerAwareSemanticIntegrationError,
    build_lfm2_producer_aware_semantic_basis,
    build_lfm2_producer_aware_semantic_boundaries,
    build_lfm2_producer_aware_semantic_topology,
    compile_install_lfm2_producer_aware_semantic_closure,
    integrate_lfm2_producer_aware_global_closure,
)
from neural_forge.hf_lfm2_semantic_global_residual_integration import (  # noqa: E402
    Lfm2SemanticGlobalResidualEvidencePackage,
    Lfm2SemanticGlobalResidualTrajectoryEvidence,
    build_lfm2_semantic_global_residual_basis,
    build_lfm2_semantic_global_residual_boundaries,
    build_lfm2_semantic_global_residual_topology,
)


DTYPE = torch.float64
ATTENTION_LAYERS = {2, 4, 6, 8, 10, 12}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _bf16(*shape: int, scale: float = 0.1) -> torch.Tensor:
    return (torch.randn(*shape, dtype=DTYPE) * scale).to(torch.bfloat16)


def _physical_ids(layer: int, attention: bool) -> dict[str, str]:
    mlp = f"model.layers.{layer}.feed_forward"
    if attention:
        operator = f"model.layers.{layer}.self_attn"
        result = {
            "query_weight": f"{operator}.q_proj.weight",
            "key_weight": f"{operator}.k_proj.weight",
            "value_weight": f"{operator}.v_proj.weight",
            "operator_output_weight": f"{operator}.out_proj.weight",
        }
    else:
        operator = f"model.layers.{layer}.conv"
        result = {
            "operator_input_weight": f"{operator}.in_proj.weight",
            "operator_output_weight": f"{operator}.out_proj.weight",
        }
    result.update(
        {
            "mlp_gate_weight": f"{mlp}.w1.weight",
            "mlp_up_weight": f"{mlp}.w3.weight",
            "mlp_down_weight": f"{mlp}.w2.weight",
        }
    )
    return result


def _attention(
    weights: dict[str, torch.Tensor],
    *,
    sequence: int,
    hidden: int,
) -> Lfm2AttentionParameters:
    head_dimension = 2
    angles = torch.randn((1, sequence, 1), dtype=DTYPE) * 0.1
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    return Lfm2AttentionParameters(
        operator_norm_gain=torch.ones(hidden, dtype=DTYPE),
        operator_norm_epsilon=1e-3,
        query_weight=weights["query_weight"],
        key_weight=weights["key_weight"],
        value_weight=weights["value_weight"],
        output_weight=weights["operator_output_weight"],
        query_norm_gain=torch.ones(head_dimension, dtype=DTYPE),
        key_norm_gain=torch.ones(head_dimension, dtype=DTYPE),
        query_key_norm_epsilon=1e-3,
        rope_cosine=cosine,
        rope_sine=sine,
        causal_mask=torch.tril(
            torch.ones((sequence, sequence), dtype=torch.bool)
        ).reshape(1, 1, sequence, sequence),
        query_heads=hidden // head_dimension,
        key_value_heads=1,
        head_dim=head_dimension,
    )


def _conv(
    weights: dict[str, torch.Tensor],
    *,
    sequence: int,
    hidden: int,
) -> Lfm2ShortConvParameters:
    return Lfm2ShortConvParameters(
        operator_norm_gain=torch.ones(hidden, dtype=DTYPE),
        operator_norm_epsilon=1e-3,
        input_weight=weights["operator_input_weight"],
        stored_fir_kernel=_bf16(hidden, 1, 2),
        output_weight=weights["operator_output_weight"],
        input_bias=torch.zeros(3 * hidden, dtype=DTYPE),
        fir_bias=torch.zeros(hidden, dtype=DTYPE),
        output_bias=torch.zeros(hidden, dtype=DTYPE),
        padding_mask=torch.ones((1, sequence), dtype=DTYPE),
    )


def _mlp(
    weights: dict[str, torch.Tensor],
    *,
    hidden: int,
) -> Lfm2MlpParameters:
    return Lfm2MlpParameters(
        norm_gain=torch.ones(hidden, dtype=DTYPE),
        norm_epsilon=1e-3,
        gate_weight=weights["mlp_gate_weight"],
        up_weight=weights["mlp_up_weight"],
        down_weight=weights["mlp_down_weight"],
    )


def _problem(
    *,
    trajectory_count: int = 2,
    defect_scale: float = 0.08,
    layer_count: int = 14,
):
    torch.manual_seed(1207)
    hidden, sequence, intermediate = 4, 2, 5
    final_prior_seed = _bf16(hidden, intermediate)
    final_live = torch.nn.Parameter(
        _bf16(hidden, intermediate),
        requires_grad=False,
    )
    final_layer_index = layer_count - 1
    final_id = f"model.layers.{final_layer_index}.feed_forward.w2.weight"
    residual = torch.randn((1, sequence, hidden), dtype=DTYPE) * 0.1
    macrocells = []
    live_physical = {}
    writer_seeds = {}
    for layer in range(layer_count):
        attention = layer in ATTENTION_LAYERS
        ids = _physical_ids(layer, attention)
        if attention:
            weights = {
                "query_weight": _bf16(hidden, hidden),
                "key_weight": _bf16(2, hidden),
                "value_weight": _bf16(2, hidden),
                "operator_output_weight": _bf16(hidden, hidden),
            }
        else:
            weights = {
                "operator_input_weight": _bf16(3 * hidden, hidden),
                "operator_output_weight": _bf16(hidden, hidden),
            }
        weights.update(
            {
                "mlp_gate_weight": _bf16(intermediate, hidden),
                "mlp_up_weight": _bf16(intermediate, hidden),
                "mlp_down_weight": (
                    final_prior_seed
                    if layer == final_layer_index
                    else _bf16(hidden, intermediate)
                ),
            }
        )
        operator = (
            _attention(weights, sequence=sequence, hidden=hidden)
            if attention
            else _conv(weights, sequence=sequence, hidden=hidden)
        )
        macrocell = capture_lfm2_macrocell_linearization(
            residual,
            operator,
            _mlp(weights, hidden=hidden),
        )
        macrocells.append(macrocell)
        for role, parameter_id in ids.items():
            if parameter_id != final_id:
                live_physical[parameter_id] = torch.nn.Parameter(
                    weights[role].clone(),
                    requires_grad=False,
                )
        writer_seeds[ids["operator_output_weight"]] = weights[
            "operator_output_weight"
        ].clone()
        if layer != final_layer_index:
            writer_seeds[ids["mlp_down_weight"]] = weights["mlp_down_weight"].clone()
        residual = macrocell.mlp.output
    terminal = capture_lfm2_terminal_norm_primal(
        residual,
        Lfm2TerminalNormParameters(
            gain=torch.ones(hidden, dtype=DTYPE),
            epsilon=1e-3,
        ),
    )
    prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=final_prior_seed,
        body_recipe_id="nonlinear_qdeim_structural",
        body_recipe_sha256=_digest("recipe"),
        mlp_lane="nonlinear_qdeim_structural_priority",
        selected_role="full-w1-w3-w2",
        selected_candidate_sha256=_digest("candidate"),
        preparation_sha256=_digest("preparation"),
        mlp_installation_core_sha256=_digest("installation"),
        live_deferred_w2_sha256=lfm2_cpcf_terminal_action_tensor_sha256(final_live),
    )
    trajectories = []
    for batch_index in range(trajectory_count):
        primal = Lfm2StackPrimalBatch(
            macrocells=tuple(macrocells),
            selected_target_positions=torch.tensor([1]),
            canonical_boundary_row_indices=torch.tensor([batch_index]),
            terminal=terminal,
        )
        offset = defect_scale * (batch_index + 1)
        parent_operator = []
        parent_layer = []
        for macrocell in macrocells:
            post_operator = macrocell.operator.output.clone()
            post_operator[:, 0] = macrocell.operator.output[:, 1] + offset
            post_layer = macrocell.mlp.output.clone()
            post_layer[:, 0] = macrocell.mlp.output[:, 1] + 2.0 * offset
            parent_operator.append(post_operator)
            parent_layer.append(post_layer)
        parent_terminal = terminal.output.clone()
        parent_terminal[:, 0] = terminal.output[:, 1] + 3.0 * offset
        trajectories.append(
            Lfm2SemanticGlobalResidualTrajectoryEvidence(
                batch_index=batch_index,
                primal_batch=primal,
                selected_donor_positions=torch.tensor([0]),
                mapped_parent_post_operator_residuals=tuple(parent_operator),
                mapped_parent_post_layer_residuals=tuple(parent_layer),
                mapped_parent_terminal_final_norm=parent_terminal,
            )
        )
    evidence = Lfm2SemanticGlobalResidualEvidencePackage(
        trajectories=tuple(trajectories),
        live_writer_seeds=writer_seeds,
        deferred_final_w2_prior=prior,
        mapped_parent_tied_table=torch.randn((9, hidden), dtype=DTYPE),
        head_dimension=2,
    )
    return {
        "evidence": evidence,
        "live": live_physical,
        "final_live": final_live,
        "final_id": final_id,
    }


def _solve(
    problem,
    *,
    materialization_compute_device="cpu",
    requested_basis_rank=None,
):
    return compile_install_lfm2_producer_aware_semantic_closure(
        problem["evidence"],
        live_physical_parameters=problem["live"],
        deferred_final_writer_parameter=problem["final_live"],
        closure_policy=Lfm2GlobalResidualClosurePolicy(
            backend="cpu",
            pcg_iterations=1,
            boundary_chunk_rows=4,
        ),
        materialization_compute_device=materialization_compute_device,
        requested_basis_rank=requested_basis_rank,
    )


def test_builds_authenticated_producer_aware_semantic_problem_without_gate() -> None:
    problem = _problem()
    evidence = problem["evidence"]
    topology = build_lfm2_producer_aware_semantic_topology(evidence)
    boundaries = build_lfm2_producer_aware_semantic_boundaries(evidence)
    basis = build_lfm2_producer_aware_semantic_basis(
        evidence,
        topology,
        boundaries,
    )
    policy = Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=(0.5, 1.0),
        materialization_scale=0.5,
    )

    assert evidence.verify_authentication()
    assert topology.verify_authentication()
    assert len(topology.sites) == 56
    assert len(boundaries.boundaries) == 29
    assert boundaries.boundaries[-1].causal_order == 84
    assert basis.verify_authentication()
    assert policy.verify_authentication()
    assert policy.authentication_body()["numeric_diagnostic_threshold"] is False
    assert policy.authentication_body()["manual_review_gate"] is False
    assert (
        policy.authentication_body()[
            "all_candidates_eligible_for_decoded_behavioral_review"
        ]
        is True
    )


def test_explicit_residual_basis_rank_reaches_the_physical_closure() -> None:
    result = _solve(_problem(layer_count=2), requested_basis_rank=3)

    assert result.verify_authentication()
    assert result.topology.requested_basis_rank == 3
    assert result.topology.effective_basis_rank == 3
    assert result.basis.rank == 3
    assert result.report["topology_receipt"]["topology_basis_rank_formula"] == (
        "explicit_requested_basis_rank"
    )


@pytest.mark.parametrize("layer_count", (2, 5, 14))
def test_semantic_evidence_and_v2_installation_derive_from_layer_count(
    layer_count: int,
) -> None:
    problem = _problem(layer_count=layer_count, defect_scale=32.0)
    evidence = problem["evidence"]
    baseline_topology = build_lfm2_semantic_global_residual_topology(evidence)
    baseline_boundaries = build_lfm2_semantic_global_residual_boundaries(evidence)
    baseline_basis = build_lfm2_semantic_global_residual_basis(
        evidence,
        baseline_topology,
        baseline_boundaries,
    )
    topology = build_lfm2_producer_aware_semantic_topology(evidence)
    boundaries = build_lfm2_producer_aware_semantic_boundaries(evidence)
    basis = build_lfm2_producer_aware_semantic_basis(
        evidence,
        topology,
        boundaries,
    )

    assert evidence.snapshot.layer_count == layer_count
    assert len(evidence.writer_seed_mapping()) == 2 * layer_count
    assert len(baseline_topology.sites) == 2 * layer_count
    assert len(baseline_boundaries.boundaries) == 2 * layer_count + 1
    assert baseline_boundaries.boundaries[-1].layer_index == layer_count - 1
    assert baseline_basis.verify_authentication()
    assert len(topology.sites) == 4 * layer_count
    assert len(boundaries.boundaries) == 2 * layer_count + 1
    assert boundaries.boundaries[-1].causal_order == 6 * layer_count
    assert basis.verify_authentication()

    result = _solve(problem)
    assert result.verify_authentication()
    assert result.report["layer_count"] == layer_count
    assert result.report["writer_site_count"] == 4 * layer_count
    assert result.report["installed_parameter_count"] == len(problem["live"])
    assert result.report["final_w2_unchanged"] is True
    changed = set(result.report["changed_physical_parameter_ids"])
    for layer in range(layer_count):
        ids = _physical_ids(layer, layer in ATTENTION_LAYERS)
        if layer in ATTENTION_LAYERS:
            assert all(
                ids[role] in changed
                for role in ("query_weight", "key_weight", "value_weight")
            )
        else:
            assert ids["operator_input_weight"] in changed
        assert ids["operator_output_weight"] in changed
        assert ids["mlp_gate_weight"] in changed
        assert ids["mlp_up_weight"] in changed
        if layer < layer_count - 1:
            assert ids["mlp_down_weight"] in changed
    assert result.report["autograd_grad_calls"] == 0
    assert result.report["backward_calls"] == 0
    assert result.report["optimizer_steps"] == 0
    assert result.report["training_steps"] == 0


def test_solves_installs_all_nonfinal_roles_and_returns_final_prior() -> None:
    problem = _problem(defect_scale=0.16)
    final_before = fingerprint_lfm2_global_residual_tensor(problem["final_live"])
    result = _solve(problem)

    assert isinstance(result, Lfm2ProducerAwareSemanticInstallation)
    assert result.verify_authentication()
    assert result.report["kind"] == (
        "producer_aware_semantic_global_residual_installation"
    )
    assert result.report["writer_site_count"] == 56
    assert result.report["installed_parameter_count"] == 81
    assert result.report["final_w2_installation_count"] == 0
    assert result.report["final_w2_unchanged"] is True
    assert result.report["numeric_or_manual_quality_gate"] is False
    assert (
        result.report["all_frontier_candidates_eligible_for_decoded_behavioral_review"]
        is True
    )
    assert result.report["training_steps"] == 0
    assert result.report["parameters_pruned"] == 0
    assert result.report["parameters_quantized"] == 0
    compute_receipt = result.report["materialization_compute_execution_receipt"]
    assert result.report["materialization_compute_device"] == "cpu"
    assert compute_receipt["requested_device"] == "cpu"
    assert compute_receipt["resolved_device"] == "cpu"
    assert (
        result.report["materialization_compute_execution_receipt_sha256"]
        == compute_receipt["report_sha256"]
    )
    assert (
        compute_receipt == result.dense_transaction.report["compute_execution_receipt"]
    )
    assert fingerprint_lfm2_global_residual_tensor(problem["final_live"]) == (
        final_before
    )
    assert isinstance(
        result.corrected_final_w2_prior,
        Lfm2CpcfTerminalActionDownWeightPrior,
    )
    assert result.corrected_final_w2_prior.verify_authentication()
    corrected = dict(
        zip(
            result.dense_transaction.parameter_ids,
            result.dense_transaction.corrected_weights,
            strict=True,
        )
    )
    torch.testing.assert_close(
        result.corrected_final_w2_prior.down_weight,
        corrected[problem["final_id"]],
    )
    for parameter_id, parameter in problem["live"].items():
        torch.testing.assert_close(parameter, corrected[parameter_id])


def test_named_alias_threads_materialization_compute_device() -> None:
    problem = _problem(layer_count=2)
    result = integrate_lfm2_producer_aware_global_closure(
        problem["evidence"],
        live_physical_parameters=problem["live"],
        deferred_final_writer_parameter=problem["final_live"],
        closure_policy=Lfm2GlobalResidualClosurePolicy(
            backend="cpu",
            pcg_iterations=1,
            boundary_chunk_rows=4,
        ),
        materialization_compute_device="cpu",
    )

    assert result.verify_authentication()
    assert result.report["materialization_compute_device"] == "cpu"
    assert (
        result.dense_transaction.report["compute_execution_receipt"]["requested_device"]
        == "cpu"
    )


def test_install_failure_rolls_back_every_changed_physical_role(monkeypatch) -> None:
    problem = _problem()
    before = {
        name: parameter.detach().clone() for name, parameter in problem["live"].items()
    }
    final_before = problem["final_live"].detach().clone()
    original_copy = producer_module._copy_live_parameter
    calls = 0

    def fail_second_copy(parameter, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected semantic publication failure")
        original_copy(parameter, value)

    monkeypatch.setattr(producer_module, "_copy_live_parameter", fail_second_copy)
    with pytest.raises(
        Lfm2ProducerAwareLinearizationError,
        match="every changed parameter was restored",
    ):
        _solve(problem)
    for parameter_id, expected in before.items():
        torch.testing.assert_close(problem["live"][parameter_id], expected)
    torch.testing.assert_close(problem["final_live"], final_before)


def test_rejects_non_parameter_target_before_solve() -> None:
    problem = _problem()
    parameter_id = next(iter(problem["live"]))
    problem["live"][parameter_id] = problem["live"][parameter_id].detach().clone()

    with pytest.raises(
        Lfm2ProducerAwareSemanticIntegrationError,
        match="ordinary dense Parameter",
    ):
        _solve(problem)


def test_rejects_mutated_evidence_before_solve() -> None:
    problem = _problem()
    evidence = problem["evidence"]
    evidence.trajectories[0].primal_batch.macrocells[0].operator.output[0, 0, 0] += 1

    with pytest.raises(
        Lfm2ProducerAwareSemanticAuthenticationError,
        match="mutated evidence",
    ):
        build_lfm2_producer_aware_semantic_topology(evidence)
    with pytest.raises(
        Lfm2ProducerAwareSemanticAuthenticationError,
        match="authenticated semantic residual package",
    ):
        _solve(problem)
