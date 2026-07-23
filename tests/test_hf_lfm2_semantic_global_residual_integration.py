from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")

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
from neural_forge.hf_lfm2_semantic_global_residual_integration import (  # noqa: E402
    Lfm2SemanticGlobalResidualEvidencePackage,
    Lfm2SemanticGlobalResidualIntegrationAuthenticationError,
    Lfm2SemanticGlobalResidualIntegrationError,
    Lfm2SemanticGlobalResidualTrajectoryEvidence,
    build_lfm2_semantic_global_residual_basis,
    build_lfm2_semantic_global_residual_boundaries,
    build_lfm2_semantic_global_residual_topology,
    compile_install_lfm2_semantic_global_residual_closure,
)


DTYPE = torch.float64
ATTENTION_LAYERS = {2, 4, 6, 8, 10, 12}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _bf16(*shape: int, scale: float = 0.1) -> torch.Tensor:
    return (torch.randn(*shape, dtype=DTYPE) * scale).to(torch.bfloat16)


def _attention(
    output_weight: torch.Tensor,
    *,
    sequence: int,
    hidden: int,
) -> Lfm2AttentionParameters:
    head_dim = 2
    query_heads = hidden // head_dim
    angles = torch.randn((1, sequence, 1), dtype=DTYPE) * 0.1
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    return Lfm2AttentionParameters(
        operator_norm_gain=torch.ones(hidden, dtype=DTYPE),
        operator_norm_epsilon=1e-3,
        query_weight=torch.randn(hidden, hidden, dtype=DTYPE) * 0.1,
        key_weight=torch.randn(head_dim, hidden, dtype=DTYPE) * 0.1,
        value_weight=torch.randn(head_dim, hidden, dtype=DTYPE) * 0.1,
        output_weight=output_weight.to(DTYPE),
        query_norm_gain=torch.ones(head_dim, dtype=DTYPE),
        key_norm_gain=torch.ones(head_dim, dtype=DTYPE),
        query_key_norm_epsilon=1e-3,
        rope_cosine=cosine,
        rope_sine=sine,
        causal_mask=torch.tril(
            torch.ones((sequence, sequence), dtype=torch.bool)
        ).reshape(1, 1, sequence, sequence),
        query_heads=query_heads,
        key_value_heads=1,
        head_dim=head_dim,
    )


def _conv(
    output_weight: torch.Tensor,
    *,
    sequence: int,
    hidden: int,
) -> Lfm2ShortConvParameters:
    return Lfm2ShortConvParameters(
        operator_norm_gain=torch.ones(hidden, dtype=DTYPE),
        operator_norm_epsilon=1e-3,
        input_weight=torch.randn(3 * hidden, hidden, dtype=DTYPE) * 0.1,
        stored_fir_kernel=torch.randn(hidden, 1, 2, dtype=DTYPE) * 0.1,
        output_weight=output_weight.to(DTYPE),
        input_bias=torch.zeros(3 * hidden, dtype=DTYPE),
        fir_bias=torch.zeros(hidden, dtype=DTYPE),
        output_bias=torch.zeros(hidden, dtype=DTYPE),
        padding_mask=torch.ones((1, sequence), dtype=DTYPE),
    )


def _mlp(down_weight: torch.Tensor, *, hidden: int) -> Lfm2MlpParameters:
    intermediate = int(down_weight.shape[1])
    return Lfm2MlpParameters(
        norm_gain=torch.ones(hidden, dtype=DTYPE),
        norm_epsilon=1e-3,
        gate_weight=torch.randn(intermediate, hidden, dtype=DTYPE) * 0.1,
        up_weight=torch.randn(intermediate, hidden, dtype=DTYPE) * 0.1,
        down_weight=down_weight.to(DTYPE),
    )


def _site_id(layer: int, branch: int) -> str:
    if branch:
        return f"model.layers.{layer}.feed_forward.w2.weight"
    component = "self_attn" if layer in ATTENTION_LAYERS else "conv"
    return f"model.layers.{layer}.{component}.out_proj.weight"


def _problem(*, defect_scale: float = 0.02):
    torch.manual_seed(947)
    hidden, sequence, intermediate = 4, 2, 5
    final_prior_seed = _bf16(hidden, intermediate)
    final_live = torch.nn.Parameter(_bf16(hidden, intermediate))
    residual = torch.randn((1, sequence, hidden), dtype=DTYPE) * 0.1
    macrocells = []
    live_seeds = {}
    for layer in range(14):
        operator_seed = _bf16(hidden, hidden)
        mlp_seed = final_prior_seed if layer == 13 else _bf16(hidden, intermediate)
        operator = (
            _attention(operator_seed, sequence=sequence, hidden=hidden)
            if layer in ATTENTION_LAYERS
            else _conv(operator_seed, sequence=sequence, hidden=hidden)
        )
        macrocell = capture_lfm2_macrocell_linearization(
            residual,
            operator,
            _mlp(mlp_seed, hidden=hidden),
        )
        macrocells.append(macrocell)
        live_seeds[_site_id(layer, 0)] = operator_seed.clone()
        if layer != 13:
            live_seeds[_site_id(layer, 1)] = mlp_seed.clone()
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
    for batch_index in range(16):
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
        live_writer_seeds=live_seeds,
        deferred_final_w2_prior=prior,
        mapped_parent_tied_table=torch.randn((9, hidden), dtype=DTYPE),
        head_dimension=2,
    )
    live = {
        name: torch.nn.Parameter(value.clone()) for name, value in live_seeds.items()
    }
    return {
        "evidence": evidence,
        "live": live,
        "final_live": final_live,
        "defect_scale": defect_scale,
    }


def test_builds_exact_28_site_29_boundary_semantic_problem() -> None:
    problem = _problem()
    evidence = problem["evidence"]
    topology = build_lfm2_semantic_global_residual_topology(evidence)
    boundaries = build_lfm2_semantic_global_residual_boundaries(evidence)
    basis = build_lfm2_semantic_global_residual_basis(
        evidence,
        topology,
        boundaries,
    )

    assert topology.verify_authentication()
    assert len(topology.sites) == 28
    assert [site.writer_kind for site in topology.sites[::2]].count(
        "attention_output"
    ) == 6
    assert [site.writer_kind for site in topology.sites[::2]].count(
        "short_conv_output"
    ) == 8
    assert topology.sites[-1].site_id == ("model.layers.13.feed_forward.w2.weight")
    assert len(boundaries.boundaries) == 29
    assert boundaries.boundaries[-1].boundary_kind == "terminal_observable"
    expected = torch.tensor(
        [problem["defect_scale"] * (index + 1) for index in range(16)],
        dtype=DTYPE,
    )
    torch.testing.assert_close(boundaries.boundaries[0].defect[:, 0], expected)
    torch.testing.assert_close(
        boundaries.boundaries[1].defect[:, 0],
        2.0 * expected,
    )
    torch.testing.assert_close(
        boundaries.boundaries[-1].defect[:, 0],
        3.0 * expected,
    )
    torch.testing.assert_close(
        boundaries.boundaries[0].row_mass,
        torch.full((16,), 1.0 / 16.0, dtype=DTYPE),
    )
    assert basis.verify_authentication()
    assert basis.report["all_three_source_groups_present"] is True
    assert basis.report["basis_rank"] == 4
    assert evidence.verify_authentication()


def test_reusable_evidence_accepts_non_sixteen_trajectory_bank() -> None:
    problem = _problem()
    original = problem["evidence"]
    evidence = Lfm2SemanticGlobalResidualEvidencePackage(
        trajectories=original.trajectories[:3],
        live_writer_seeds=original.live_writer_seeds,
        deferred_final_w2_prior=original.deferred_final_w2_prior,
        mapped_parent_tied_table=original.mapped_parent_tied_table,
        head_dimension=original.head_dimension,
    )
    boundaries = build_lfm2_semantic_global_residual_boundaries(evidence)

    assert evidence.verify_authentication()
    assert len(evidence.trajectories) == 3
    assert evidence.snapshot.row_count == 3
    torch.testing.assert_close(
        boundaries.boundaries[0].row_mass,
        torch.full((3,), 1.0 / 3.0, dtype=DTYPE),
    )


def test_dense_transaction_installs_27_and_returns_uninstalled_final_prior() -> None:
    problem = _problem(defect_scale=0.04)
    before_final = fingerprint_lfm2_global_residual_tensor(problem["final_live"])
    result = compile_install_lfm2_semantic_global_residual_closure(
        problem["evidence"],
        live_writer_parameters=problem["live"],
        deferred_final_writer_parameter=problem["final_live"],
        policy=Lfm2GlobalResidualClosurePolicy(
            backend="cpu",
            pcg_iterations=2,
            boundary_chunk_rows=16,
        ),
    )

    assert result.verify_authentication()
    assert result.report["installed_writer_count"] == 27
    assert result.report["final_w2_installation_count"] == 0
    assert result.report["final_w2_unchanged"] is True
    assert result.report["quality_gate"] is False
    assert result.report["numeric_diagnostics_used_for_auto_veto"] is False
    assert (
        result.report["decoded_behavior_review_is_evidence_for_understanding"] is True
    )
    assert result.report["verdict_required_for_continuation"] is False
    assert result.report["accept_veto_publish_delete_stop_authority"] == {
        "decoded_behavior_review": False,
        "numeric_metrics": False,
    }
    assert (
        result.report["v2_input_transport_extension_roadmap"][
            "implemented_in_this_result"
        ]
        is False
    )
    assert result.report["runtime_sidecar_required"] is False
    assert result.report["execution"]["training_steps"] == 0
    assert result.report["execution"]["backward_calls"] == 0
    assert result.report["execution"]["pruning_operations"] == 0
    assert result.report["execution"]["quantization_operations"] == 0
    assert fingerprint_lfm2_global_residual_tensor(problem["final_live"]) == (
        before_final
    )
    assert isinstance(
        result.corrected_final_w2_prior,
        Lfm2CpcfTerminalActionDownWeightPrior,
    )
    assert result.corrected_final_w2_prior.verify_authentication()
    base_prior = problem["evidence"].deferred_final_w2_prior
    assert result.corrected_final_w2_prior.body_recipe_id == base_prior.body_recipe_id
    assert result.corrected_final_w2_prior.body_recipe_sha256 == (
        base_prior.body_recipe_sha256
    )
    assert result.corrected_final_w2_prior.mlp_lane == base_prior.mlp_lane
    assert result.corrected_final_w2_prior.selected_role == base_prior.selected_role
    assert result.corrected_final_w2_prior.live_deferred_w2_sha256 == (
        base_prior.live_deferred_w2_sha256
    )
    assert (
        result.corrected_final_w2_prior.as_dict()["installed_before_terminal"] is False
    )
    assert (
        result.report["final_layer_mlp_down_weight_prior_authentication_sha256"]
        == result.corrected_final_w2_prior.authentication_sha256
    )
    assert result.report["final_layer_mlp_down_weight_prior_install_count"] == 0
    for site_index, site in enumerate(result.topology.sites[:-1]):
        seed = problem["evidence"].live_writer_seeds[site.site_id]
        expected = (
            seed.to(DTYPE) + result.closure_result.dense_writer_corrections[site_index]
        ).to(seed.dtype)
        assert torch.equal(problem["live"][site.site_id].detach(), expected)


def test_mutated_cached_trajectory_fails_before_problem_construction() -> None:
    problem = _problem()
    evidence = problem["evidence"]
    evidence.trajectories[3].mapped_parent_post_layer_residuals[7].add_(1.0)

    assert evidence.verify_authentication() is False
    with pytest.raises(
        Lfm2SemanticGlobalResidualIntegrationAuthenticationError,
        match="mutated evidence",
    ):
        build_lfm2_semantic_global_residual_topology(evidence)


class _FailingParameter(torch.nn.Parameter):
    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value, value.requires_grad)

    def copy_(self, _source, *, non_blocking=False):
        del non_blocking
        raise RuntimeError("synthetic publication failure")


def test_failed_dense_install_rolls_back_every_changed_writer() -> None:
    problem = _problem(defect_scale=0.04)
    site_ids = tuple(problem["live"])
    failing_id = site_ids[1]
    problem["live"][failing_id] = _FailingParameter(
        problem["live"][failing_id].detach().clone()
    )
    before = {
        name: fingerprint_lfm2_global_residual_tensor(value)
        for name, value in problem["live"].items()
    }
    final_before = fingerprint_lfm2_global_residual_tensor(problem["final_live"])

    with pytest.raises(
        Lfm2SemanticGlobalResidualIntegrationError,
        match="all changed writers were restored",
    ):
        compile_install_lfm2_semantic_global_residual_closure(
            problem["evidence"],
            live_writer_parameters=problem["live"],
            deferred_final_writer_parameter=problem["final_live"],
            policy=Lfm2GlobalResidualClosurePolicy(
                backend="cpu",
                pcg_iterations=1,
                boundary_chunk_rows=16,
            ),
        )

    after = {
        name: fingerprint_lfm2_global_residual_tensor(value)
        for name, value in problem["live"].items()
    }
    assert after == before
    assert fingerprint_lfm2_global_residual_tensor(problem["final_live"]) == (
        final_before
    )
