"""Planner contracts for chimera-omni: what folds, and what is still named.

``NOTES-W10A.md`` recorded the refusal that motivated this file: the promoted
causal branch kernels required *both* graphs to declare
``inspector == "hf-causal-decoder"`` and ``position_policy == "rope"``, and a
chimera-omni graph declares neither, so a real 4x fold ran as a driver-local
transaction with no ledger record at all.

The relaxation here is deliberately not "let chimera through the RoPE gate".
It is a second, separately named lane -- windowed NoPE attention and a fused
gate/up SwiGLU MLP -- whose contracts admit only what the registered operators
can honestly execute.  Every test below is either a case that must now plan, or
a case that must still fail closed, and the failures name the missing kernel
family rather than collapsing into one generic gap.

Every model is a real ``chimera_omni.modeling.ChimeraOmniModel`` built through
chimera-omni's own configuration API.  Nothing is vendored; when the checkout is
unavailable the suite skips and says so.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_planner import (
    CAUSAL_ATTENTION_BRANCH_METHOD,
    CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
    CAUSAL_GATED_MLP_BRANCH_METHOD,
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
    NATIVE_CAUSAL_LAYER_BRANCH_METHODS,
    NATIVE_RESIDUAL_CHART_INTERFACE_METHOD,
    DynamicFoldPlanner,
)
from neural_forge.dynamic_fold_registry import MethodStatus, default_method_registry
from neural_forge.dynamic_model_inspector import inspect_model_circuits


torch = pytest.importorskip("torch")

RUNNER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_chimera_omni_toy_fold.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_chimera_omni_toy_fold_contract", RUNNER
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
except Exception as exc:  # pragma: no cover - environment boundary
    pytest.skip(f"chimera-omni checkout unavailable: {exc}", allow_module_level=True)

chimera_config = _CHIMERA["config"]
chimera_modeling = _CHIMERA["modeling"]

TOKENIZER = canonical_sha256({"tokenizer": "shared-chimera-native-fold"})
PARENT_WIDTH = 128
CHILD_WIDTH = 32


def coverable_config(hidden_size: int, **changes):
    """One chimera configuration whose every block family folds today.

    Every learned tensor here is exactly linear in ``hidden_size``: the
    attention projections span ``num_heads * attention_head_dim`` on their
    non-residual side and the feed-forward spans ``intermediate_size`` on
    theirs, and both are pinned.  Halving ``hidden_size`` twice therefore
    divides the unique parameter count by exactly four with nothing left over.
    """

    settings = {
        "vocab_size": 1024,
        "hidden_size": int(hidden_size),
        "intermediate_size": 320,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_sequence_length": 64,
        "block_pattern": ("local",),
        "residual_mode": "block_attnres",
        "attnres_block_size": 2,
        "attention_head_dim": 32,
        "feed_forward_kind": "dense",
        "activation_kind": "silu",
        "native_feature_dim": 64,
        "native_position_dim": 4,
        "native_type_count": 8,
        "mtp_heads": 1,
        "aux_heads": ("flow", "grounding", "sync"),
        "local_window": 32,
    }
    settings.update(changes)
    config = chimera_config.ChimeraConfig(**settings)
    config.validate()
    return config


def _graph(config, name: str, seed: int = 20260804):
    torch.manual_seed(seed)
    model = chimera_modeling.ChimeraOmniModel(config)
    model.eval()
    return inspect_model_circuits(
        model, model_name=name, tokenizer_fingerprint=TOKENIZER
    )


def _plan(parent_graph, child_graph, *, enabled=NATIVE_CAUSAL_LAYER_BRANCH_METHODS):
    return DynamicFoldPlanner(enabled_experimental_method_ids=enabled).plan(
        (parent_graph,),
        child_graph,
        child_state_fingerprint=canonical_sha256({"child": child_graph.fingerprint}),
        evidence_fingerprint=canonical_sha256({"evidence": "contract-only"}),
        parent_state_fingerprints={
            parent_graph.model_name: canonical_sha256(
                {"parent": parent_graph.fingerprint}
            )
        },
    )


@pytest.fixture(scope="module")
def coverable_graphs():
    return (
        _graph(coverable_config(PARENT_WIDTH), "parent"),
        _graph(coverable_config(CHILD_WIDTH), "child", seed=4711),
    )


@pytest.fixture(scope="module")
def coverable_plan(coverable_graphs):
    return _plan(*coverable_graphs)


def test_the_coverable_crossing_is_ready_with_no_gaps(coverable_graphs, coverable_plan):
    parent_graph, child_graph = coverable_graphs

    assert parent_graph.unique_parameter_count == 4 * child_graph.unique_parameter_count
    assert coverable_plan.ready
    assert not coverable_plan.gaps

    methods = [action.method_id for action in coverable_plan.actions]
    assert methods.count(CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD) == 4
    assert methods.count(CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD) == 4
    # The promoted RoPE/split-SwiGLU pair is never selected for this crossing.
    assert CAUSAL_ATTENTION_BRANCH_METHOD not in methods
    assert CAUSAL_GATED_MLP_BRANCH_METHOD not in methods
    assert methods.count(NATIVE_RESIDUAL_CHART_INTERFACE_METHOD) == 14
    assert methods.count("copy_exact") == 2

    owned = [
        path for action in coverable_plan.actions for path in action.target_parameter_paths
    ]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(child_graph.parameter_paths)


def test_every_crossing_action_declares_the_boundary_chart(coverable_plan):
    for action in coverable_plan.actions:
        configuration = dict(action.configuration)
        if action.target_parameter_paths or action.method_id == "copy_exact":
            assert configuration["boundary_chart_required"] is True


def test_the_branch_actions_name_the_windowed_nope_objective(coverable_plan):
    attention = next(
        action
        for action in coverable_plan.actions
        if action.method_id == CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD
    )
    mlp = next(
        action
        for action in coverable_plan.actions
        if action.method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
    )
    assert dict(attention.configuration)["fit_objective"] == (
        "cached_parent_endpoint_coupled_windowed_nope_gqa_qk_softmax_vo"
    )
    assert dict(mlp.configuration)["fit_objective"] == (
        "cached_parent_endpoint_coupled_fused_gate_up_swiglu_residual_write"
    )
    # The depth selector travels with the branch whose input it chooses.
    assert any(
        path.endswith("attention_residual.query")
        for path in attention.target_parameter_paths
    )
    assert any(
        path.endswith("feed_forward_residual.query")
        for path in mlp.target_parameter_paths
    )


def test_the_interface_actions_name_their_transport_class(coverable_plan):
    objectives = {
        tuple(sorted(action.target_parameter_paths)): dict(action.configuration)[
            "fit_objective"
        ]
        for action in coverable_plan.actions
        if action.method_id == NATIVE_RESIDUAL_CHART_INTERFACE_METHOD
    }
    assert objectives[("embed.weight",)] == "chart_projected_residual_row_table"
    assert objectives[("lm_head.weight",)] == "chart_projected_residual_consumer_head"
    assert objectives[("norm.weight",)] == "chart_projected_residual_norm_gain"
    assert objectives[("native_media_bridge.feature_proj.weight",)] == (
        "chart_projected_residual_producer_projection"
    )
    assert objectives[("native_media_head.out.weight",)] == (
        "chart_projected_residual_consumer_head"
    )
    assert objectives[
        (
            "output_attention_residual.norm.weight",
            "output_attention_residual.query",
        )
    ] == "chart_projected_depth_source_selector"


def test_the_same_crossing_stays_gapped_without_the_experimental_lane(
    coverable_graphs,
):
    """The lane is opt-in; nothing about chimera is plannable by default."""

    plan = DynamicFoldPlanner().plan(
        (coverable_graphs[0],),
        coverable_graphs[1],
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"evidence": "contract-only"}),
        parent_state_fingerprints={"parent": canonical_sha256({"parent": "state"})},
    )

    assert not plan.ready
    capabilities = {gap.required_capability for gap in plan.gaps}
    assert "cross_family_closed_loop_layer_operator" in capabilities


def test_half_the_lane_is_not_a_lane(coverable_graphs):
    """A layer whose attention half has no owning transaction stays gapped."""

    plan = _plan(
        *coverable_graphs,
        enabled=(CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,),
    )

    assert not plan.ready
    assert CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD not in {
        action.method_id for action in plan.actions
    }


def test_a_window_change_fails_closed(coverable_graphs):
    """A window is a contract constant, not a fold coordinate."""

    child = _graph(
        coverable_config(CHILD_WIDTH, local_window=8), "child-narrow-window", seed=4711
    )
    plan = _plan(coverable_graphs[0], child)

    assert not plan.ready
    assert CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD not in {
        action.method_id for action in plan.actions
    }
    assert "cross_family_closed_loop_layer_operator" in {
        gap.required_capability for gap in plan.gaps
    }


def test_a_non_silu_gate_fails_closed(coverable_graphs):
    """``situ`` is a tanh-capped gate; the SwiGLU solve cannot represent it."""

    child = _graph(
        coverable_config(CHILD_WIDTH, activation_kind="situ"), "child-situ", seed=4711
    )
    plan = _plan(coverable_graphs[0], child)

    assert not plan.ready
    assert CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD not in {
        action.method_id for action in plan.actions
    }


def _gap_capabilities(parent_config, child_config, parent_name, child_name):
    plan = _plan(
        _graph(parent_config, parent_name),
        _graph(child_config, child_name, seed=4711),
    )
    assert not plan.ready
    return {gap.required_capability for gap in plan.gaps}, plan


def test_a_kda_layer_gaps_by_the_name_of_the_recurrence_kernel():
    parent = dataclasses.replace(
        chimera_config.small_test_config(), hidden_size=128, attention_head_dim=32
    )
    parent.validate()
    child = dataclasses.replace(parent, hidden_size=64)
    child.validate()
    capabilities, plan = _gap_capabilities(parent, child, "kda-parent", "kda-child")

    assert "bounded_gated_delta_recurrence_layer_operator" in capabilities
    assert "compressed_summary_attention_layer_operator" in capabilities
    reasons = " ".join(gap.reason for gap in plan.gaps)
    assert "bounded_gated_delta_recurrence" in reasons
    assert "compressed_summary_attention" in reasons
    # The one coverable family in that pattern still folds.
    assert CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD in {
        action.method_id for action in plan.actions
    }


def test_gated_mla_and_stable_latent_moe_gap_by_name():
    parent = chimera_config.k3_aligned_test_config()
    # The MoE latent width is pinned at 64 and must stay below the residual
    # width, so this crossing contracts 128 -> 96 rather than 128 -> 64.
    child = dataclasses.replace(parent, hidden_size=96)
    child.validate()
    capabilities, _plan = _gap_capabilities(parent, child, "k3-parent", "k3-child")

    assert "compressed_latent_kv_attention_layer_operator" in capabilities
    assert "bounded_gated_delta_recurrence_layer_operator" in capabilities
    assert "stable_latent_moe_channel_mixer_operator" in capabilities


def test_the_new_methods_are_registered_experimental():
    registry = default_method_registry()
    for method_id in (
        CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
        CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
        NATIVE_RESIDUAL_CHART_INTERFACE_METHOD,
    ):
        method = registry.get(method_id)
        assert method.status == MethodStatus.EXPERIMENTAL.value
        assert method.zero_gradient is True
        assert method.standalone is True
        assert method.evidence_channels
        assert method.coupled_roles
    # Distinct lanes: a closure recorded against the RoPE branch must not
    # silently close the NoPE variant.
    assert registry.get(CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD).family_id != (
        registry.get(CAUSAL_ATTENTION_BRANCH_METHOD).family_id
    )
    assert registry.get(CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD).family_id != (
        registry.get(CAUSAL_GATED_MLP_BRANCH_METHOD).family_id
    )
