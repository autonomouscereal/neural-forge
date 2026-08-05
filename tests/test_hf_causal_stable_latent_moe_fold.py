"""Chimera-omni Stable LatentMoE folding -- stage 0 and stage 1 acceptance.

Every parent here is a **real** ``chimera_omni.modeling.layers``
``StableLatentMoEFeedForward`` built through chimera-omni's own configuration
API.  Nothing is vendored; when the checkout is unavailable the suite skips and
says so.

Stage 0/1 of ``docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md`` section 2:
the routing recovered from the captured router boundary *is* the parent's, the
routed/shared split is exact, an identity fold reproduces the mixer, the
compiled child is installable into the real module, a contracted fold beats the
no-branch baseline under both activations, and every fail-closed condition
fires -- including the sparse-routing row budget that decides whether an expert
has enough evidence to be folded at all.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_causal_stable_latent_moe_fold import (  # noqa: E402
    ChimeraStableLatentMoEFoldPolicy,
    ChimeraStableLatentMoEGeometry,
    ChimeraStableLatentMoEParameters,
    HFCausalStableLatentMoEFoldError,
    capture_chimera_stable_latent_moe_layer_evidence,
    chimera_stable_latent_moe_geometry,
    chimera_stable_latent_moe_parameters,
    compile_hf_causal_stable_latent_moe_fold,
    moe_latent_basis,
    stable_latent_moe_consistency,
    stable_latent_moe_routing,
)


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_chimera_omni_toy_fold.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_chimera_omni_toy_fold_stable_latent_moe", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

try:
    _REPO = runner.resolve_chimera_omni_repo()
    _CHIMERA = runner.import_chimera_omni(_REPO)
except runner.ChimeraOmniFoldError as exc:  # pragma: no cover - environment gate
    pytest.skip(f"chimera-omni checkout unavailable: {exc}", allow_module_level=True)

chimera_config = _CHIMERA["config"]
chimera_layers = sys.modules["chimera_omni.modeling.layers"]


HIDDEN = 32
BATCH = 8
SEQUENCE = 24
EXPERTS = 4
EXPERTS_PER_TOKEN = 2
SHARED = 1
LATENT = 16
ROUTED_INTERMEDIATE = 12
INTERMEDIATE = 24


def _config(**changes):
    body = {
        "vocab_size": 32,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_sequence_length": 64,
        "block_pattern": ("local",),
        "residual_mode": "none",
        "attention_head_dim": 8,
        "feed_forward_kind": "stable_latent_moe",
        "activation_kind": "silu",
        "moe_num_experts": EXPERTS,
        "moe_experts_per_token": EXPERTS_PER_TOKEN,
        "moe_shared_experts": SHARED,
        "moe_latent_size": LATENT,
        "moe_routed_intermediate_size": ROUTED_INTERMEDIATE,
        "moe_first_dense_layers": 0,
    }
    body.update(changes)
    return chimera_config.ChimeraConfig(**body)


def _mixer(seed: int, **changes):
    torch.manual_seed(seed)
    module = chimera_layers.StableLatentMoEFeedForward(_config(**changes))
    module.eval()
    return module.double()


def _rows(seed: int, *, hidden: int = HIDDEN):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        BATCH, SEQUENCE, hidden, generator=generator, dtype=torch.float64
    )


def _masks():
    fit = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    fit[:, : SEQUENCE - 8] = True
    return fit, ~fit


def _evidence(seed: int, **changes):
    module = _mixer(seed, **changes)
    evidence = capture_chimera_stable_latent_moe_layer_evidence(
        module, _rows(seed + 1), layer_index=1
    )
    return module, evidence


def _relative_rmse(prediction, target) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _compile(evidence, parameters, geometry, **changes):
    fit, validation = _masks()
    body = {"fit_token_mask": fit, "validation_token_mask": validation}
    body.update(changes)
    return compile_hf_causal_stable_latent_moe_fold(
        evidence, parameters, geometry, **body
    )


# ---------------------------------------------------------------------------
# Capture, routing recovery, and the two evidence-consistency gates
# ---------------------------------------------------------------------------


def test_the_capture_is_a_trace_of_the_parent() -> None:
    module, evidence = _evidence(101)

    assert evidence.metadata["parent_forward_calls"] == 1
    assert evidence.geometry == chimera_stable_latent_moe_geometry(module)
    assert evidence.geometry.expert_count == EXPERTS
    assert evidence.geometry.routed_intermediate_size == ROUTED_INTERMEDIATE
    assert evidence.rows == BATCH * SEQUENCE
    assert len(evidence.expert_output_rows) == EXPERTS
    assert sum(
        int(value.shape[0]) for value in evidence.expert_output_rows
    ) == evidence.rows * EXPERTS_PER_TOKEN


def test_the_routing_recovered_from_the_router_boundary_is_the_parents() -> None:
    """The routing identity is exact linear algebra on a float64 parent.

    ``StableLatentMoEFeedForward.forward`` never downcasts, so recovering the
    assignment from the captured router output and rebuilding the accumulator
    from the captured expert outputs reproduces the captured accumulator to
    machine epsilon.  The row counts are compared as integers first, so a
    mis-recovered assignment fails before any tolerance is consulted.
    """

    module, evidence = _evidence(103)
    parameters = chimera_stable_latent_moe_parameters(module)
    gates = stable_latent_moe_consistency(evidence, parameters)

    assert gates.routing_relative_rmse < 1e-12
    assert gates.expert_input_relative_rmse < 1e-12
    assert gates.split_relative_rmse < 1e-12
    assert gates.diagnostics["expert_row_counts"] == [
        int(value.shape[0]) for value in evidence.expert_output_rows
    ]
    assert gates.diagnostics["parent_forward_calls"] == 0
    assert 0.0 < gates.diagnostics["routed_energy_fraction"]
    assert 0.0 < gates.diagnostics["shared_energy_fraction"]


def test_the_router_bias_enters_selection_only() -> None:
    module, evidence = _evidence(105)
    zero = torch.zeros(EXPERTS, dtype=torch.float64)
    unbiased = stable_latent_moe_routing(
        evidence.router_logit_rows, zero, experts_per_token=EXPERTS_PER_TOKEN
    )
    biased_values = torch.zeros(EXPERTS, dtype=torch.float64)
    biased_values[0] = 10.0
    biased = stable_latent_moe_routing(
        evidence.router_logit_rows,
        biased_values,
        experts_per_token=EXPERTS_PER_TOKEN,
    )

    assert torch.equal(unbiased.raw_scores, biased.raw_scores)
    assert int(biased.expert_rows[0].numel()) == evidence.rows
    assert not torch.equal(unbiased.expert_indices, biased.expert_indices)
    assert torch.allclose(
        biased.weights.sum(dim=-1),
        torch.ones(evidence.rows, dtype=torch.float64),
        atol=1e-12,
    )
    del module


def test_a_wrong_carried_bias_fails_the_routing_identity_closed() -> None:
    module, evidence = _evidence(107)
    wrong = torch.zeros(EXPERTS, dtype=torch.float64)
    wrong[0] = 10.0
    parameters = ChimeraStableLatentMoEParameters(
        router_bias=wrong, geometry=evidence.geometry
    )

    with pytest.raises(
        HFCausalStableLatentMoEFoldError, match="not the parent's"
    ):
        stable_latent_moe_consistency(evidence, parameters)


def test_an_unchanged_latent_width_takes_the_identity_basis() -> None:
    _module, evidence = _evidence(109)
    basis = moe_latent_basis(evidence.latent_rows, LATENT)

    assert basis.identity is True
    assert torch.equal(basis.basis, torch.eye(LATENT, dtype=torch.float64))
    truncated = moe_latent_basis(evidence.latent_rows, LATENT // 2)
    assert truncated.identity is False
    assert 0.0 < truncated.retained_energy_fraction < 1.0


# ---------------------------------------------------------------------------
# Stage 1: the identity fold and the installable child
# ---------------------------------------------------------------------------


def test_an_identity_fold_reproduces_the_channel_mixer() -> None:
    module, evidence = _evidence(201)
    parameters = chimera_stable_latent_moe_parameters(module)
    fold = _compile(evidence, parameters, evidence.geometry)
    report = dict(fold.report)

    assert report["validation_beats_no_branch_baseline"] is True
    assert report["validation_residual_relative_rmse"] < 1e-2
    assert report["latent_basis"] == "identity"
    assert report["router_top_k_agreement"] > 0.9
    assert report["expert_rows_are_the_parents_own_assignment"] is True
    assert report["router_bias_carried_not_fitted"] is True
    assert report["parent_forward_calls"] == 0
    assert report["training_steps"] == 0
    assert report["backward_calls"] == 0
    assert report["alternation_objective_monotone"] is True


def test_the_compiled_child_is_installable_into_the_real_module() -> None:
    """The kernel's replay is the module's forward, not an approximation.

    Nothing in this module downcasts, so a float64 child loaded with the
    compiled tensors must reproduce the kernel's FP64 replay to machine
    epsilon -- including the top-k selection, which is a discrete decision that
    a slightly different router would change.
    """

    module, evidence = _evidence(203)
    parameters = chimera_stable_latent_moe_parameters(module)
    fold = _compile(evidence, parameters, evidence.geometry)
    child = _mixer(999)
    state = fold.child_state_dict()
    live = {
        name: value
        for name, value in child.state_dict().items()
        if name != "router_bias"
    }

    assert set(state) == set(live)
    with torch.no_grad():
        for name, value in state.items():
            child.state_dict()[name].copy_(value)
        child.router_bias.copy_(fold.router_bias)
        installed = child(evidence.feed_forward_norm_output)
    replayed, _routed, _shared = fold.replay(evidence.feed_forward_norm_output)

    assert _relative_rmse(
        installed.reshape(-1, HIDDEN).to(dtype=torch.float64), replayed
    ) < 1e-12


def test_a_contracted_fold_beats_the_no_branch_baseline() -> None:
    module, evidence = _evidence(205)
    parameters = chimera_stable_latent_moe_parameters(module)
    geometry = ChimeraStableLatentMoEGeometry(
        hidden_size=HIDDEN,
        expert_count=EXPERTS,
        experts_per_token=EXPERTS_PER_TOKEN,
        shared_expert_count=SHARED,
        latent_size=LATENT // 2,
        routed_intermediate_size=ROUTED_INTERMEDIATE // 2,
        intermediate_size=INTERMEDIATE // 2,
    )
    fold = _compile(evidence, parameters, geometry)
    report = dict(fold.report)

    assert report["validation_beats_no_branch_baseline"] is True
    assert report["latent_basis"] == "observed_row_pod"
    assert 0.0 < report["latent_retained_energy_fraction"] < 1.0
    assert all(
        len(value) == ROUTED_INTERMEDIATE // 2
        for value in report["selected_routed_neurons"]
    )
    assert all(
        len(value) == INTERMEDIATE // 2
        for value in report["selected_shared_neurons"]
    )
    assert tuple(fold.latent_up_weight.shape) == (HIDDEN, LATENT // 2)


def test_the_situ_activation_folds_through_the_same_executable() -> None:
    """One executable, two activations; the activation is a contract constant."""

    module, evidence = _evidence(207, activation_kind="situ")
    parameters = chimera_stable_latent_moe_parameters(module)

    assert evidence.geometry.activation == "situ"
    fold = _compile(evidence, parameters, evidence.geometry)
    assert dict(fold.report)["validation_beats_no_branch_baseline"] is True
    assert dict(fold.report)["target_geometry"]["activation"] == "situ"

    silu = ChimeraStableLatentMoEGeometry(
        **{**evidence.geometry.as_dict(), "activation": "silu"}
    )
    with pytest.raises(
        HFCausalStableLatentMoEFoldError, match="contract constants"
    ):
        _compile(evidence, parameters, silu)


def test_the_sub_branch_endpoints_are_reported_separately() -> None:
    module, evidence = _evidence(209)
    parameters = chimera_stable_latent_moe_parameters(module)
    report = dict(_compile(evidence, parameters, evidence.geometry).report)

    assert report["routed_sub_branch_validation_relative_rmse"] > 0.0
    assert report["shared_sub_branch_validation_relative_rmse"] > 0.0
    assert report["split_reconstruction_relative_rmse"] < 1e-6
    assert len(report["expert_fit_row_counts"]) == EXPERTS
    assert report["expected_expert_rows"] == pytest.approx(
        BATCH * SEQUENCE * EXPERTS_PER_TOKEN / EXPERTS
    )


# ---------------------------------------------------------------------------
# Fail-closed conditions
# ---------------------------------------------------------------------------


def test_an_expert_count_change_is_refused_as_a_kind_translation() -> None:
    module, evidence = _evidence(301)
    parameters = chimera_stable_latent_moe_parameters(module)
    geometry = ChimeraStableLatentMoEGeometry(
        **{**evidence.geometry.as_dict(), "expert_count": EXPERTS // 2}
    )

    with pytest.raises(
        HFCausalStableLatentMoEFoldError, match="contract constants"
    ):
        _compile(evidence, parameters, geometry)


def test_the_sparse_row_budget_refuses_an_underfed_expert() -> None:
    """Sparse routing decides how much evidence each expert's solve consumes.

    With ``experts_per_token`` of ``expert_count`` selected, each expert sees
    about ``rows * K / E`` rows and roughly half of those land in the fit
    partition.  A policy floor above that count refuses the whole channel
    mixer, names the expert and prints the two numbers, rather than fitting a
    dictionary on rows the endpoint gate will never exercise.
    """

    module, evidence = _evidence(303)
    parameters = chimera_stable_latent_moe_parameters(module)

    with pytest.raises(HFCausalStableLatentMoEFoldError, match="fit rows") as error:
        _compile(
            evidence,
            parameters,
            evidence.geometry,
            policy=ChimeraStableLatentMoEFoldPolicy(
                minimum_expert_fit_rows=BATCH * SEQUENCE
            ),
        )
    assert "Routed expert" in str(error.value)
    assert "sealed floor" in str(error.value)


def test_the_relative_row_floor_is_measured_against_the_expected_share() -> None:
    module, evidence = _evidence(305)
    parameters = chimera_stable_latent_moe_parameters(module)

    with pytest.raises(HFCausalStableLatentMoEFoldError, match="expected"):
        _compile(
            evidence,
            parameters,
            evidence.geometry,
            policy=ChimeraStableLatentMoEFoldPolicy(
                minimum_expert_row_fraction=0.99
            ),
        )


def test_a_router_agreement_floor_refuses_a_collapsed_router() -> None:
    """A router that abandons the parent's selection is refused, not measured.

    An identity fold recovers the parent's routing exactly, so the case is
    driven where it can be driven: a ridge that crushes the router to nothing
    sends every row to the same two experts, and the sealed agreement floor
    refuses before a single expert dictionary is installed.
    """

    module, evidence = _evidence(307)
    parameters = chimera_stable_latent_moe_parameters(module)

    with pytest.raises(
        HFCausalStableLatentMoEFoldError, match="top-k expert selection"
    ):
        _compile(
            evidence,
            parameters,
            evidence.geometry,
            policy=ChimeraStableLatentMoEFoldPolicy(
                ridge=1.0e14, minimum_router_top_k_agreement=0.9
            ),
        )


def test_a_branch_that_loses_to_writing_nothing_is_refused() -> None:
    """The design's 1.0 no-branch floor, exercised through the ridge axis."""

    module, evidence = _evidence(309)
    parameters = chimera_stable_latent_moe_parameters(module)

    with pytest.raises(
        HFCausalStableLatentMoEFoldError, match="no-branch baseline"
    ):
        _compile(
            evidence,
            parameters,
            evidence.geometry,
            policy=ChimeraStableLatentMoEFoldPolicy(
                ridge=1.0e14, minimum_router_top_k_agreement=0.0
            ),
        )


def test_latent_expansion_is_refused() -> None:
    module, evidence = _evidence(311)
    parameters = chimera_stable_latent_moe_parameters(module)
    geometry = ChimeraStableLatentMoEGeometry(
        **{**evidence.geometry.as_dict(), "latent_size": LATENT * 2}
    )

    with pytest.raises(HFCausalStableLatentMoEFoldError, match="expansion"):
        _compile(evidence, parameters, geometry)
