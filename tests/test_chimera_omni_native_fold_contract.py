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
    CAUSAL_KDA_RECURRENT_BRANCH_METHOD,
    CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
    CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
    CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
    CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
    NATIVE_CAUSAL_LAYER_BRANCH_METHODS,
    NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
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


# ---------------------------------------------------------------------------
# The two chimera sequence-mixer families that gained a kernel: the bounded
# gated delta recurrence and the composite compressed-summary attention.
# ---------------------------------------------------------------------------

FAMILY_PARENT_WIDTH = 128
FAMILY_CHILD_WIDTH = 34


def family_config(hidden_size: int, **changes):
    """A trunk running every sequence mixer that has a registered kernel.

    ``csa``/``hca`` own four ``[hidden, hidden]`` summary projections, which
    shrink quadratically while everything else shrinks linearly, so a
    quarter-width child of this trunk is never exactly 4x.  128 -> 34 with
    every other geometry identical is the exact integer solution, and the
    execution suite asserts the two parameter counts.
    """

    return coverable_config(
        hidden_size,
        **{
            "vocab_size": 1534,
            "intermediate_size": 448,
            "block_pattern": ("local", "kda", "csa", "hca"),
            "kda_head_dim": 9,
            "kda_value_head_dim": 8,
            "kda_decay_rank": 4,
            "kda_short_conv_kernel_size": 4,
            "local_window": 8,
            "csa_compress_stride": 4,
            "hca_compress_stride": 8,
            **changes,
        },
    )


@pytest.fixture(scope="module")
def family_graphs():
    return (
        _graph(family_config(FAMILY_PARENT_WIDTH), "family-parent"),
        _graph(family_config(FAMILY_CHILD_WIDTH), "family-child", seed=4711),
    )


def test_every_kernelled_sequence_mixer_family_plans_with_zero_gaps(family_graphs):
    parent_graph, child_graph = family_graphs
    plan = _plan(parent_graph, child_graph, enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS)

    assert plan.ready
    assert not plan.gaps
    assert parent_graph.unique_parameter_count == 4 * child_graph.unique_parameter_count

    methods = [action.method_id for action in plan.actions]
    assert methods.count(CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD) == 1
    assert methods.count(CAUSAL_KDA_RECURRENT_BRANCH_METHOD) == 1
    # csa and hca share one method id: one stride-parameterized executable.
    assert methods.count(CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD) == 2
    assert methods.count(CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD) == 4

    owned = [path for action in plan.actions for path in action.target_parameter_paths]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(child_graph.parameter_paths)


def test_the_new_branch_actions_name_their_objectives_and_constants(family_graphs):
    plan = _plan(*family_graphs, enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS)
    recurrence = next(
        action
        for action in plan.actions
        if action.method_id == CAUSAL_KDA_RECURRENT_BRANCH_METHOD
    )
    summaries = [
        action
        for action in plan.actions
        if action.method_id == CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD
    ]

    assert dict(recurrence.configuration)["fit_objective"] == (
        "cached_parent_endpoint_coupled_bounded_gated_delta_recurrence_write"
    )
    for action in summaries:
        assert dict(action.configuration)["fit_objective"] == (
            "cached_parent_endpoint_coupled_local_window_plus_compressed_"
            "summary_sub_branch_write"
        )
    # The composite is one transaction over one node: every tensor of the
    # nested local attention *and* the summary projections travels together.
    paths = set(summaries[0].target_parameter_paths)
    assert any(path.endswith("mixer.local.q_proj.weight") for path in paths)
    assert any(path.endswith("mixer.summary_q.weight") for path in paths)
    assert any(path.endswith("mixer.out.weight") for path in paths)
    assert any(path.endswith("mixer.local.out.weight") for path in paths)
    # The recurrence carries its three internal short convs and its headwise
    # output norm in the same transaction, because they are internal structure
    # of one operator rather than circuits of their own.
    recurrent_paths = set(recurrence.target_parameter_paths)
    assert any(path.endswith("mixer.q_conv.conv.weight") for path in recurrent_paths)
    assert any(path.endswith("mixer.output_norm.weight") for path in recurrent_paths)
    assert any(path.endswith("mixer.decay_log_scale") for path in recurrent_paths)


@pytest.mark.parametrize(
    ("changes", "blocked", "survivors", "capability"),
    (
        (
            {"csa_compress_stride": 2},
            CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
            1,
            "compressed_summary_attention_layer_operator",
        ),
        (
            {"hca_compress_stride": 16},
            CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
            1,
            "compressed_summary_attention_layer_operator",
        ),
        (
            {"kda_short_conv_kernel_size": 3},
            CAUSAL_KDA_RECURRENT_BRANCH_METHOD,
            0,
            "bounded_gated_delta_recurrence_layer_operator",
        ),
    ),
)
def test_a_contract_constant_change_fails_closed(
    family_graphs, changes, blocked, survivors, capability
):
    """A stride and a short-conv kernel are contract constants, not coordinates.

    Changing either changes which inputs the operator can see or what its own
    arithmetic is.  No evidence anywhere says what the parent would have
    computed under a different one, so *that layer* keeps its gap -- by the name
    of the kernel family it needed -- while every layer whose constants are
    unchanged still folds through the same method id.
    """

    child = _graph(
        family_config(FAMILY_CHILD_WIDTH, **changes),
        f"family-child-{sorted(changes)[0]}",
        seed=4711,
    )
    plan = _plan(
        family_graphs[0], child, enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS
    )

    assert not plan.ready
    methods = [action.method_id for action in plan.actions]
    assert methods.count(blocked) == survivors
    assert capability in {gap.required_capability for gap in plan.gaps}


def test_the_two_new_families_stay_gapped_without_their_lane(family_graphs):
    """The lane is opt-in per family, exactly as the local/dense pair is."""

    plan = _plan(*family_graphs, enabled=NATIVE_CAUSAL_LAYER_BRANCH_METHODS)

    assert not plan.ready
    capabilities = {gap.required_capability for gap in plan.gaps}
    assert "bounded_gated_delta_recurrence_layer_operator" in capabilities
    assert "compressed_summary_attention_layer_operator" in capabilities
    # The one family whose lane is enabled still folds.
    assert CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD in {
        action.method_id for action in plan.actions
    }


def test_the_real_k3_reference_plans_with_zero_gaps():
    """The reference configuration chimera-omni targets plans completely.

    ``NOTES-W12F.md`` closed the last two chimera *families* and left one
    layer gapped on this very config: layer 0 is the ``moe_first_dense_layers``
    layer, so its channel mixer is a dense ``FeedForward`` under
    ``activation_kind = "situ"``, and the fused gate/up lane admitted SiLU
    only.  That lane's activation is now a sealed contract constant over
    ``{silu, situ}`` rather than a hardcoded SiLU, so every layer of the real
    ``k3_aligned_test_config`` has an owning transaction and the plan is ready
    with no gap of any kind.

    Nothing here is approximated: the dense layer is planned by the *same*
    method id the SiLU dense lane uses, with the SiTU constant carried in the
    action's own fit objective.
    """

    parent = chimera_config.k3_aligned_test_config()
    child = dataclasses.replace(parent, hidden_size=96)
    child.validate()
    parent_graph = _graph(parent, "k3-parent-covered")
    child_graph = _graph(child, "k3-child-covered", seed=4711)
    plan = _plan(
        parent_graph,
        child_graph,
        enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
    )

    assert dict(parent_graph.metadata)["kernel_gap_block_kinds"] == ""
    assert plan.ready
    assert not plan.gaps

    methods = [action.method_id for action in plan.actions]
    # Layers 0-2 are kda, layers 3 and 4 are gated_mla; layer 0's channel mixer
    # is the dense SiTU one and layers 1-4 all run the Stable LatentMoE.
    assert methods.count(CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD) == 2
    assert methods.count(CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD) == 4
    assert methods.count(CAUSAL_KDA_RECURRENT_BRANCH_METHOD) == 3
    assert methods.count(CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD) == 1

    # Every child tensor is owned by exactly one transaction.
    owned = [
        path for action in plan.actions for path in action.target_parameter_paths
    ]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(child_graph.parameter_paths)

    # The dense SiTU action names its own activation in the sealed objective,
    # so a SiTU plan can never be read back as a SwiGLU one.
    dense = next(
        action
        for action in plan.actions
        if action.method_id == CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD
    )
    assert dict(dense.configuration)["fit_objective"] == (
        "cached_parent_endpoint_coupled_fused_gate_up_situ_residual_write"
    )


def test_a_gated_activation_kind_translation_keeps_its_gap():
    """SiLU and SiTU share a lane; a crossing between them is not a fold.

    The two activations are the same executable under different sealed
    constants, which is exactly why the constant has to be checked: no cached
    evidence anywhere says what a SiLU parent would have computed under SiTU's
    caps, so the planner keeps its explicit operator gap rather than fitting
    one nonlinearity's rows through the other's dictionary.
    """

    parent = chimera_config.k3_aligned_test_config()
    child = dataclasses.replace(parent, hidden_size=96, activation_kind="silu")
    child.validate()
    plan = _plan(
        _graph(parent, "k3-situ-parent"),
        _graph(child, "k3-silu-child", seed=4711),
        enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
    )

    assert not plan.ready
    assert CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD not in {
        action.method_id for action in plan.actions
    }


def test_a_situ_cap_the_child_does_not_share_keeps_its_gap():
    """The betas are the constant too, not decoration on the name.

    ``situ_gate_beta`` and ``situ_up_beta`` are what SiTU caps each half with,
    so a child that caps somewhere else computes a different function under the
    same name.  Both channel-mixer families read them -- the MoE lane has
    checked them since ``NOTES-W12F.md`` -- so a rebeta'd child gaps
    everywhere, and the dense lane is one of the places it gaps rather than the
    one place it does not.
    """

    parent = chimera_config.k3_aligned_test_config()
    child = dataclasses.replace(parent, hidden_size=96, situ_gate_beta=8.0)
    child.validate()
    plan = _plan(
        _graph(parent, "k3-beta4-parent"),
        _graph(child, "k3-beta8-child", seed=4711),
        enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS,
    )

    assert not plan.ready
    assert CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD not in {
        action.method_id for action in plan.actions
    }


def test_the_two_new_methods_are_registered_on_their_own_experimental_lanes():
    registry = default_method_registry()
    families = set()
    for method_id in (
        CAUSAL_KDA_RECURRENT_BRANCH_METHOD,
        CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
    ):
        method = registry.get(method_id)
        assert method.status == MethodStatus.EXPERIMENTAL.value
        assert method.zero_gradient is True
        assert method.standalone is True
        assert method.evidence_channels
        assert method.coupled_roles
        families.add(method.family_id)
    # Four distinct lanes: a closure recorded against one branch operator must
    # not silently close any of the others.
    families.update(
        registry.get(method_id).family_id
        for method_id in (
            CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
            CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
        )
    )
    assert len(families) == 4


# ---------------------------------------------------------------------------
# The last two chimera families: gated MLA and the Stable LatentMoE channel
# mixer.
# ---------------------------------------------------------------------------

LATENT_PARENT_WIDTH = 128
LATENT_CHILD_WIDTH = 31


def latent_config(hidden_size: int, **changes):
    """A k3-aligned-style trunk: local + gated_mla + kda, MoE everywhere.

    Nothing here is quadratic in ``hidden_size`` and every latent width is a
    pinned constant, so the unique parameter count is exactly affine in the
    residual width and ``128 -> 31`` is exactly 4x.  The execution suite states
    the arithmetic and asserts both counts.
    """

    return coverable_config(
        hidden_size,
        **{
            "vocab_size": 1459,
            "intermediate_size": 320,
            "block_pattern": ("local", "gated_mla", "kda", "gated_mla"),
            "activation_kind": "situ",
            "situ_gate_beta": 4.0,
            "situ_up_beta": 25.0,
            "kda_head_dim": 8,
            "kda_value_head_dim": 8,
            "kda_decay_rank": 4,
            "kda_short_conv_kernel_size": 4,
            "mla_latent_size": 24,
            "mla_query_latent_size": 24,
            "mla_key_head_dim": 16,
            "mla_extra_head_dim": 8,
            "mla_value_head_dim": 16,
            "feed_forward_kind": "stable_latent_moe",
            "moe_num_experts": 4,
            "moe_experts_per_token": 2,
            "moe_shared_experts": 2,
            "moe_latent_size": 16,
            "moe_routed_intermediate_size": 8,
            "moe_first_dense_layers": 0,
            "local_window": 8,
            **changes,
        },
    )


@pytest.fixture(scope="module")
def latent_graphs():
    return (
        _graph(latent_config(LATENT_PARENT_WIDTH), "latent-parent"),
        _graph(latent_config(LATENT_CHILD_WIDTH), "latent-child", seed=4711),
    )


def test_the_last_two_chimera_families_plan_with_zero_gaps(latent_graphs):
    parent_graph, child_graph = latent_graphs
    plan = _plan(
        parent_graph, child_graph, enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS
    )

    assert plan.ready
    assert not plan.gaps
    assert parent_graph.unique_parameter_count == 4 * child_graph.unique_parameter_count

    methods = [action.method_id for action in plan.actions]
    assert methods.count(CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD) == 1
    assert methods.count(CAUSAL_KDA_RECURRENT_BRANCH_METHOD) == 1
    # One method id serves both gated-MLA layers; one serves every MoE mixer.
    assert methods.count(CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD) == 2
    assert methods.count(CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD) == 4

    owned = [path for action in plan.actions for path in action.target_parameter_paths]
    assert len(owned) == len(set(owned))
    assert set(owned) == set(child_graph.parameter_paths)


def test_the_last_two_branch_actions_name_their_objectives(latent_graphs):
    plan = _plan(*latent_graphs, enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS)
    latent = next(
        action
        for action in plan.actions
        if action.method_id == CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD
    )
    channel = next(
        action
        for action in plan.actions
        if action.method_id == CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD
    )

    assert dict(latent.configuration)["fit_objective"] == (
        "cached_parent_endpoint_coupled_compressed_latent_kv_softmax_gated_write"
    )
    assert dict(channel.configuration)["fit_objective"] == (
        "cached_parent_endpoint_coupled_top_k_routed_latent_experts_plus_"
        "shared_expert_write"
    )
    # The gated-MLA transaction carries the whole compressed query and
    # key/value path, and the channel mixer carries every expert, because each
    # is one operator.
    latent_paths = set(latent.target_parameter_paths)
    for suffix in ("mixer.q_down.weight", "mixer.kv_up.weight", "mixer.gate.weight"):
        assert any(path.endswith(suffix) for path in latent_paths)
    channel_paths = set(channel.target_parameter_paths)
    for suffix in (
        "ffn.router.weight",
        "ffn.routed_experts.0.up.weight",
        "ffn.routed_norm.weight",
        "ffn.shared_experts.1.down.weight",
    ):
        assert any(path.endswith(suffix) for path in channel_paths)


@pytest.mark.parametrize(
    ("changes", "blocked", "survivors", "capability"),
    (
        (
            {"mla_extra_head_dim": 4},
            CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
            0,
            "compressed_latent_kv_attention_layer_operator",
        ),
        (
            {"moe_num_experts": 2},
            CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
            0,
            "stable_latent_moe_channel_mixer_operator",
        ),
        (
            {"moe_experts_per_token": 1},
            CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
            0,
            "stable_latent_moe_channel_mixer_operator",
        ),
        (
            {"activation_kind": "silu"},
            CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
            0,
            "stable_latent_moe_channel_mixer_operator",
        ),
    ),
)
def test_a_new_contract_constant_change_fails_closed(
    latent_graphs, changes, blocked, survivors, capability
):
    """A shared key block, an expert count, a top-k or an activation is a
    contract constant, not a coordinate.

    Each one changes which inputs the operator sees or what its own arithmetic
    is, so no evidence anywhere says what the parent would have computed under
    a different one.  Every layer that needs the changed family keeps its gap,
    by the name of the kernel family it needed.
    """

    child = _graph(
        latent_config(LATENT_CHILD_WIDTH, **changes),
        f"latent-child-{sorted(changes)[0]}",
        seed=4711,
    )
    plan = _plan(
        latent_graphs[0], child, enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS
    )

    assert not plan.ready
    methods = [action.method_id for action in plan.actions]
    assert methods.count(blocked) == survivors
    assert capability in {gap.required_capability for gap in plan.gaps}


def test_a_dense_channel_mixer_child_of_a_moe_parent_fails_closed(latent_graphs):
    """Crossing between two channel-mixer classes is a kind translation."""

    child = _graph(
        latent_config(
            LATENT_CHILD_WIDTH,
            feed_forward_kind="dense",
            moe_num_experts=0,
            moe_experts_per_token=0,
            moe_shared_experts=0,
            moe_latent_size=0,
            moe_routed_intermediate_size=0,
        ),
        "latent-child-dense",
        seed=4711,
    )
    plan = _plan(
        latent_graphs[0], child, enabled=NATIVE_CHIMERA_LAYER_BRANCH_METHODS
    )

    assert not plan.ready
    assert CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD not in {
        action.method_id for action in plan.actions
    }
    assert "stable_latent_moe_channel_mixer_operator" in {
        gap.required_capability for gap in plan.gaps
    }


def test_the_last_two_families_stay_gapped_without_their_lane(latent_graphs):
    """The lane is opt-in per family, exactly as every other one is."""

    plan = _plan(*latent_graphs, enabled=NATIVE_CAUSAL_LAYER_BRANCH_METHODS)

    assert not plan.ready
    capabilities = {gap.required_capability for gap in plan.gaps}
    assert "compressed_latent_kv_attention_layer_operator" in capabilities
    assert "stable_latent_moe_channel_mixer_operator" in capabilities


def test_the_last_two_methods_are_registered_on_their_own_lanes():
    registry = default_method_registry()
    families = set()
    for method_id in (
        CAUSAL_LATENT_KV_ATTENTION_BRANCH_METHOD,
        CAUSAL_STABLE_LATENT_MOE_BRANCH_METHOD,
    ):
        method = registry.get(method_id)
        assert method.status == MethodStatus.EXPERIMENTAL.value
        assert method.zero_gradient is True
        assert method.standalone is True
        assert method.evidence_channels
        assert method.coupled_roles
        families.add(method.family_id)
    # Six distinct lanes now: a closure recorded against one branch operator
    # must not silently close any of the others.
    families.update(
        registry.get(method_id).family_id
        for method_id in (
            CAUSAL_WINDOWED_NOPE_ATTENTION_BRANCH_METHOD,
            CAUSAL_FUSED_GATE_UP_MLP_BRANCH_METHOD,
            CAUSAL_KDA_RECURRENT_BRANCH_METHOD,
            CAUSAL_SUMMARY_ATTENTION_BRANCH_METHOD,
        )
    )
    assert len(families) == 6
