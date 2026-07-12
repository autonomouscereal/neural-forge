from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_attention_codebook import (  # noqa: E402
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_functional_decode_trace import (  # noqa: E402
    FunctionalDecodeLimits,
    collect_functional_decode_trace,
    trace_mapping,
)
import neural_forge.hf_residual_attention_behavior_tuner as behavior_tuner  # noqa: E402
from neural_forge.hf_residual_attention_behavior_tuner import (  # noqa: E402
    EXACT_AFFECTED_SITES,
    BehaviorVOTuningControls,
    ResidualAttentionBehaviorTuningError,
    tune_exact_p35_residual_attention_vo,
)
from neural_forge.hf_residual_attention_role_scale import (  # noqa: E402
    AttentionRoleScales,
)


class _Cache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


def _exact_registry() -> RaggedAttentionCodebookRegistry:
    layers = 36
    groups = 8
    prototype_count = 279
    sources = [value for value in range(288) if value not in {156, *range(280, 288)}]
    assert len(sources) == prototype_count
    survivor_slot = sources.index(174)
    target_positions = {2 * groups + 1, 19 * groups + 4, 21 * groups + 6}
    other_slots = [
        value for value in range(prototype_count) if value != survivor_slot
    ] + [0] * 7
    assert len(other_slots) == layers * groups - len(target_positions)
    assignments = []
    other_index = 0
    for flat_index in range(layers * groups):
        if flat_index in target_positions:
            assignments.append(survivor_slot)
        else:
            assignments.append(other_slots[other_index])
            other_index += 1
    assignments_tensor = torch.tensor(assignments, dtype=torch.int64).reshape(
        layers, groups
    )
    assert set(assignments_tensor.reshape(-1).tolist()) == set(range(prototype_count))
    role_scales = torch.ones(layers, groups, 4, dtype=torch.float32)
    role_scales[19, 4, 2:] = 0.0
    ones = torch.ones(prototype_count, 1, 1, dtype=torch.float32)
    return RaggedAttentionCodebookRegistry(
        ones,
        ones.clone(),
        ones.clone(),
        ones.clone(),
        assignments_tensor,
        torch.tensor(sources, dtype=torch.int64),
        torch.arange(groups, dtype=torch.int64).repeat(layers, 1),
        role_scales=role_scales,
    )


class _BehaviorChild(torch.nn.Module):
    def __init__(self, registry: RaggedAttentionCodebookRegistry) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(3, 2)
        self.registry = registry
        self.output_bias = torch.nn.Parameter(torch.zeros(3))
        self.config = SimpleNamespace(
            vocab_size=3,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
        )

    def get_input_embeddings(self):
        return self.embedding

    def _action_row(self) -> torch.Tensor:
        value_weight, _ = self.registry.materialize_projection(19, "v")
        output_weight, _ = self.registry.materialize_projection(19, "o")
        signal = value_weight[4, 0] + output_weight[0, 4]
        return torch.stack((signal * 0.0, signal, -signal)) + self.output_bias

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **_kwargs,
    ):
        del attention_mask, return_dict
        row = self._action_row()
        logits = row.reshape(1, 1, 3).expand(1, int(input_ids.shape[1]), 3)
        previous = 0 if past_key_values is None else int(past_key_values.length)
        cache = _Cache(previous + int(input_ids.shape[1])) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


class _TamperingBehaviorChild(_BehaviorChild):
    def forward(self, *args, use_cache=True, **kwargs):
        if torch.is_grad_enabled() and not use_cache:
            assert self.registry.role_scales is not None
            with torch.no_grad():
                self.registry.role_scales[0, 0, 0].add_(0.25)
        return super().forward(*args, use_cache=use_cache, **kwargs)


class _ConstantParent(torch.nn.Module):
    def __init__(self, target_token: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(3, 2)
        self.target_token = target_token
        self.config = SimpleNamespace(
            vocab_size=3,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
        )

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **_kwargs,
    ):
        del attention_mask, return_dict
        row = torch.full(
            (3,),
            -3.0,
            dtype=self.embedding.weight.dtype,
            device=input_ids.device,
        )
        row[self.target_token] = 4.0
        logits = row.reshape(1, 1, 3).expand(1, int(input_ids.shape[1]), 3)
        previous = 0 if past_key_values is None else int(past_key_values.length)
        cache = _Cache(previous + int(input_ids.shape[1])) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _limits() -> FunctionalDecodeLimits:
    return FunctionalDecodeLimits(
        max_horizon=4,
        max_prompt_tokens=8,
        max_vocabulary=8,
        max_logit_bytes=100_000,
        max_model_devices=1,
    )


def _controls(**changes) -> BehaviorVOTuningControls:
    values = {
        "steps": 8,
        "learning_rate": 0.2,
        "action_ce_weight": 1.0,
        "action_margin_weight": 0.25,
        "topk_distill_weight": 0.1,
        "topk_distill_tokens": 2,
        "functional_top_k": 2,
        "max_prompts_per_split": 4,
        "max_horizon": 3,
        "max_total_fit_actions": 32,
    }
    values.update(changes)
    return BehaviorVOTuningControls(**values)


def _parent_traces(
    parent: torch.nn.Module,
    prompts: tuple[tuple[int, ...], ...],
    *,
    horizon: int = 2,
):
    traces = []
    for index, prompt in enumerate(prompts):
        sequence_id = f"prompt-{index}"
        greedy = collect_functional_decode_trace(
            parent,
            prompt,
            horizon=horizon,
            sequence_id=sequence_id,
            lane="greedy",
            limits=_limits(),
        )
        forced = collect_functional_decode_trace(
            parent,
            prompt,
            horizon=horizon,
            sequence_id=sequence_id,
            lane="forced",
            forced_token_ids=greedy.selected_token_ids,
            limits=_limits(),
        )
        traces.extend((forced, greedy))
    return trace_mapping(traces)


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }


def test_behavior_tuner_learns_output_actions_and_rolls_back_exactly() -> None:
    registry = _exact_registry()
    child = _BehaviorChild(registry).train()
    child.embedding.weight.grad = torch.ones_like(child.embedding.weight)
    gradient = child.embedding.weight.grad
    before = _state(child)
    parameter_ids = tuple(id(value) for value in child.parameters())
    fit = _parent_traces(_ConstantParent(1), ((0,),))
    calibration = _parent_traces(_ConstantParent(1), ((2,),))
    heldout = _parent_traces(_ConstantParent(1), ((1, 0),))

    result = tune_exact_p35_residual_attention_vo(
        child,
        registry,
        fit,
        calibration,
        heldout,
        controls=_controls(),
        trace_limits=_limits(),
    )

    assert result.selected_scales.v > 0.0
    assert result.selected_scales.o > 0.0
    assert child.training is True
    assert child.embedding.weight.grad is gradient
    assert tuple(id(value) for value in child.parameters()) == parameter_ids
    assert all(
        torch.equal(child.state_dict()[name], value) for name, value in before.items()
    )
    report = result.as_dict()
    assert report["exact_fold_lineage"]["parent_prototype_count"] == 280
    assert report["exact_fold_lineage"]["child_prototype_count"] == 279
    assert report["exact_fold_lineage"]["affected_sites"] == [
        list(value) for value in EXACT_AFFECTED_SITES
    ]
    assert report["objective_contract"][
        "numeric_logits_are_functional_outputs_not_weight_targets"
    ]
    assert report["objective_contract"]["weight_similarity_used"] is False
    assert report["objective_contract"]["attention_similarity_used"] is False
    assert report["differentiable_training"]["optimizer_object_constructed"] is False
    assert report["differentiable_training"]["optimizer_state_saved"] is False
    assert report["differentiable_training"]["training_split"] == "fit-only"
    assert report["prompt_split_audit"][
        "all_three_token_row_sets_pairwise_disjoint"
    ]
    assert report["calibration_selection"][
        "fit_used_for_selection_only_via_fitted_candidate"
    ]
    assert report["accounting"]["this_report_covers_one_redirect_only"] is True
    assert report["accounting"]["full_p35_to_p34_requires_redirect_count"] == 8
    assert report["accounting"]["prototype_count_removed"] == 1
    assert report["heldout_evaluation"]["winner_named"] is False
    assert report["heldout_evaluation"]["used_for_selection"] is False
    assert report["checkpoint_saved"] is False
    assert report["fold_committed"] is False
    allowed = {(19, 4, 2), (19, 4, 3)}
    for step in report["differentiable_training"]["history"]:
        assert {
            tuple(value) for value in step["changed_coordinates"]
        }.issubset(allowed)


def test_adversarial_non_target_scale_mutation_fails_and_restores_all_scales() -> None:
    registry = _exact_registry()
    child = _TamperingBehaviorChild(registry)
    assert registry.role_scales is not None
    before = registry.role_scales.detach().clone()
    fit = _parent_traces(_ConstantParent(1), ((0,),))
    calibration = _parent_traces(_ConstantParent(1), ((2,),))
    heldout = _parent_traces(_ConstantParent(1), ((1,),))

    with pytest.raises(
        ResidualAttentionBehaviorTuningError,
        match="outside the target L19/G4 V/O",
    ):
        tune_exact_p35_residual_attention_vo(
            child,
            registry,
            fit,
            calibration,
            heldout,
            controls=_controls(steps=1),
            trace_limits=_limits(),
        )

    assert torch.equal(registry.role_scales, before)
    assert all(value.requires_grad for value in child.parameters())


def test_adversarial_heldout_outputs_cannot_change_fit_or_selection() -> None:
    fit = _parent_traces(_ConstantParent(1), ((0,),))
    calibration = _parent_traces(_ConstantParent(1), ((2,),))
    heldout_agrees = _parent_traces(_ConstantParent(1), ((1, 0),))
    heldout_opposes = _parent_traces(_ConstantParent(2), ((1, 0),))

    first_registry = _exact_registry()
    first = tune_exact_p35_residual_attention_vo(
        _BehaviorChild(first_registry),
        first_registry,
        fit,
        calibration,
        heldout_agrees,
        controls=_controls(),
        trace_limits=_limits(),
    )
    second_registry = _exact_registry()
    second = tune_exact_p35_residual_attention_vo(
        _BehaviorChild(second_registry),
        second_registry,
        fit,
        calibration,
        heldout_opposes,
        controls=_controls(),
        trace_limits=_limits(),
    )

    first_report = first.as_dict()
    second_report = second.as_dict()
    assert first.selected_scales == second.selected_scales
    assert first.selected_candidate_id == second.selected_candidate_id
    assert (
        first_report["differentiable_training"]["history"]
        == second_report["differentiable_training"]["history"]
    )
    assert (
        first_report["calibration_selection"]["selection_lock_sha256"]
        == second_report["calibration_selection"]["selection_lock_sha256"]
    )
    assert first_report["heldout_evaluation"]["used_for_selection"] is False
    assert second_report["heldout_evaluation"]["used_for_selection"] is False
    assert (
        first_report["heldout_evaluation"]["ordering"]
        != second_report["heldout_evaluation"]["ordering"]
        or first_report["functional_comparison"]["variants"]
        != second_report["functional_comparison"]["variants"]
    )


def test_fit_outputs_reach_selector_only_through_fitted_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_one = _parent_traces(_ConstantParent(1), ((0,),))
    fit_two = _parent_traces(_ConstantParent(2), ((0,),))
    calibration = _parent_traces(_ConstantParent(1), ((2,),))
    heldout = _parent_traces(_ConstantParent(1), ((1,),))

    def fixed_fit(
        _model,
        _registry,
        _targets,
        _controls,
        *,
        canonical,
        incumbent,
    ):
        del canonical
        fitted = AttentionRoleScales(
            q=incumbent.q,
            k=incumbent.k,
            v=0.5,
            o=0.5,
        )
        return fitted, [{"step": 0, "fixed_candidate": True}], {
            "test_transaction": True
        }

    monkeypatch.setattr(behavior_tuner, "_fit_two_scalars", fixed_fit)
    first_registry = _exact_registry()
    first = tune_exact_p35_residual_attention_vo(
        _BehaviorChild(first_registry),
        first_registry,
        fit_one,
        calibration,
        heldout,
        controls=_controls(),
        trace_limits=_limits(),
    )
    second_registry = _exact_registry()
    second = tune_exact_p35_residual_attention_vo(
        _BehaviorChild(second_registry),
        second_registry,
        fit_two,
        calibration,
        heldout,
        controls=_controls(),
        trace_limits=_limits(),
    )

    first_report = first.as_dict()
    second_report = second.as_dict()
    assert first_report["fit_targets"]["target_sha256"] != second_report[
        "fit_targets"
    ]["target_sha256"]
    assert first_report["differentiable_training"][
        "fitted_candidate_scales"
    ] == second_report["differentiable_training"]["fitted_candidate_scales"]
    assert (
        first_report["calibration_selection"]["selection_lock_sha256"]
        == second_report["calibration_selection"]["selection_lock_sha256"]
    )
    assert first.selected_candidate_id == second.selected_candidate_id


def test_calibration_outputs_select_but_cannot_change_fit_gradients() -> None:
    fit = _parent_traces(_ConstantParent(1), ((0,),))
    calibration_agrees = _parent_traces(_ConstantParent(1), ((2,),))
    calibration_opposes = _parent_traces(_ConstantParent(2), ((2,),))
    heldout = _parent_traces(_ConstantParent(1), ((1,),))

    first_registry = _exact_registry()
    first = tune_exact_p35_residual_attention_vo(
        _BehaviorChild(first_registry),
        first_registry,
        fit,
        calibration_agrees,
        heldout,
        controls=_controls(),
        trace_limits=_limits(),
    )
    second_registry = _exact_registry()
    second = tune_exact_p35_residual_attention_vo(
        _BehaviorChild(second_registry),
        second_registry,
        fit,
        calibration_opposes,
        heldout,
        controls=_controls(),
        trace_limits=_limits(),
    )

    first_report = first.as_dict()
    second_report = second.as_dict()
    assert (
        first_report["differentiable_training"]["history"]
        == second_report["differentiable_training"]["history"]
    )
    assert (
        first_report["differentiable_training"]["fitted_candidate_scales"]
        == second_report["differentiable_training"]["fitted_candidate_scales"]
    )
    assert first.selected_candidate_id != second.selected_candidate_id
    assert first_report["differentiable_training"][
        "calibration_logits_or_actions_used_for_gradients"
    ] is False


@pytest.mark.parametrize(
    "overlap_pair",
    ("fit-calibration", "fit-heldout", "calibration-heldout"),
)
def test_every_prompt_split_pair_must_be_disjoint(overlap_pair: str) -> None:
    rows = {
        "fit": ((0,),),
        "calibration": ((2,),),
        "heldout": ((1,),),
    }
    left, right = overlap_pair.split("-")
    rows[right] = rows[left]
    fit = _parent_traces(_ConstantParent(1), rows["fit"])
    calibration = _parent_traces(_ConstantParent(1), rows["calibration"])
    heldout = _parent_traces(_ConstantParent(1), rows["heldout"])
    registry = _exact_registry()
    with pytest.raises(
        ResidualAttentionBehaviorTuningError, match="pairwise disjoint"
    ):
        tune_exact_p35_residual_attention_vo(
            _BehaviorChild(registry),
            registry,
            fit,
            calibration,
            heldout,
            controls=_controls(steps=1),
            trace_limits=_limits(),
        )


def test_prompt_overlap_and_non_exact_lineage_fail_before_training() -> None:
    fit = _parent_traces(_ConstantParent(1), ((0,),))
    calibration = _parent_traces(_ConstantParent(1), ((0,),))
    heldout = _parent_traces(_ConstantParent(1), ((2,),))
    registry = _exact_registry()
    child = _BehaviorChild(registry)
    before = _state(child)
    with pytest.raises(
        ResidualAttentionBehaviorTuningError, match="pairwise disjoint"
    ):
        tune_exact_p35_residual_attention_vo(
            child,
            registry,
            fit,
            calibration,
            heldout,
            controls=_controls(steps=1),
            trace_limits=_limits(),
        )
    assert all(
        torch.equal(child.state_dict()[name], value) for name, value in before.items()
    )

    bad_registry = _exact_registry()
    bad_registry.prototype_source_indices[0] = 156
    bad_child = _BehaviorChild(bad_registry)
    calibration = _parent_traces(_ConstantParent(1), ((2,),))
    heldout = _parent_traces(_ConstantParent(1), ((1,),))
    with pytest.raises(
        ResidualAttentionBehaviorTuningError, match="source lineage"
    ):
        tune_exact_p35_residual_attention_vo(
            bad_child,
            bad_registry,
            fit,
            calibration,
            heldout,
            controls=_controls(steps=1),
            trace_limits=_limits(),
        )
