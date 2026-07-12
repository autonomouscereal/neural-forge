from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
Tensor = torch.Tensor
nn = torch.nn

from neural_forge.hf_behavior_residual_swiglu import (  # noqa: E402
    BehaviorResidualSwiGLUAccounting,
    BehaviorResidualSwiGLUFitRefresh,
    BehaviorResidualSwiGLUTopologyError,
    BehaviorResidualSwiGLUTrainingError,
    ResidualSwiGLUBranch,
    audit_mlp_codebook_redirects,
    bake_behavior_residual_swiglu,
    calibration_rollout_candidate_can_promote,
    calibration_rollout_selection_key,
    install_behavior_residual_swiglu,
    train_behavior_residual_swiglu_disjoint,
)
from neural_forge.hf_functional_residual_adapter import (  # noqa: E402
    FunctionalActionBatch,
    FunctionalActionEvaluation,
)
from neural_forge.hf_neuron_codebook import NeuronCodebookRegistry  # noqa: E402


class _TinyMLP(nn.Module):
    def __init__(self, hidden: int = 6, intermediate: int = 9) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, values: Tensor) -> Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(values)) * self.up_proj(values)
        )


class _TinyLayer(nn.Module):
    def __init__(self, hidden: int = 6) -> None:
        super().__init__()
        self.mlp = _TinyMLP(hidden)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.mlp(values)


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 11, hidden: int = 6) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer(hidden)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        del attention_mask
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _batch(
    *,
    sequence_id: str,
    split: str,
    prefix_source: str,
    tokens: tuple[int, ...],
    action: int,
    vocab: int = 11,
) -> FunctionalActionBatch:
    logits = torch.zeros((1, vocab), dtype=torch.float32)
    logits[0, action] = 5.0
    return FunctionalActionBatch(
        sequence_id=sequence_id,
        split=split,  # type: ignore[arg-type]
        prefix_source=prefix_source,  # type: ignore[arg-type]
        input_ids=torch.tensor([tokens], dtype=torch.long),
        attention_mask=torch.ones((1, len(tokens)), dtype=torch.long),
        parent_action_ids=torch.tensor([action], dtype=torch.long),
        parent_action_logits=logits,
    )


def _synthetic_codebook() -> NeuronCodebookRegistry:
    layers = 3
    intermediate = 4
    hidden = 2
    omitted = (0, 1, 1, 2)
    sources: list[int] = []
    assignments = torch.empty((layers, intermediate), dtype=torch.int32)
    for neuron, missing in enumerate(omitted):
        kept = [layer for layer in range(layers) if layer != missing]
        start = len(sources)
        sources.extend(layer * intermediate + neuron for layer in kept)
        for layer in range(layers):
            chosen = layer if layer in kept else kept[0]
            assignments[layer, neuron] = start + kept.index(chosen)
    generator = torch.Generator().manual_seed(4)
    prototype_count = len(sources)
    return NeuronCodebookRegistry(
        torch.randn((prototype_count, hidden), generator=generator),
        torch.randn((prototype_count, hidden), generator=generator),
        torch.randn((hidden, prototype_count), generator=generator),
        assignments,
        torch.tensor(sources, dtype=torch.int32),
        activation="silu",
    )


def test_qwen_p35_accounting_keeps_a_large_real_saving() -> None:
    accounting = BehaviorResidualSwiGLUAccounting.build(
        hidden_size=2560,
        residual_width=64,
        site_count=1,
        removed_prototype_count=9728,
        selected_redirected_assignments=2122,
    )
    assert accounting.removed_prototype_parameters == 74_711_040
    assert accounting.residual_parameters == 491_520
    assert accounting.net_parameters_saved_vs_unfolded_codebook == 74_219_520
    assert accounting.selected_redirect_parameter_budget == 16_296_960
    assert accounting.selected_budget_after_residual == 15_805_440


def test_codebook_audit_decodes_each_missing_prototype_redirect_once() -> None:
    registry = _synthetic_codebook()
    audit = audit_mlp_codebook_redirects(registry, selected_layers=(1,))
    assert audit.prototypes_per_neuron == 2
    assert audit.redirected_assignments_by_layer == (1, 2, 1)
    assert audit.total_redirected_assignments == 4
    assert audit.selected_redirected_assignments == 2
    assert audit.as_dict()["each_missing_prototype_redirected_once"] is True


def test_zero_initialization_bake_and_rollback_are_exact() -> None:
    torch.manual_seed(7)
    model = _TinyCausalLM()
    layer = model.model.layers[0]
    original = layer.mlp
    inputs = torch.randn(2, 3, 6)
    before = model(input_ids=torch.tensor([[1, 2]])).logits.detach().clone()
    accounting = BehaviorResidualSwiGLUAccounting.build(
        hidden_size=6,
        residual_width=2,
        site_count=1,
        removed_prototype_count=4,
        selected_redirected_assignments=3,
        checkpoint_element_size=4,
    )
    handle = install_behavior_residual_swiglu(
        model,
        layer_indices=(0,),
        residual_width=2,
        accounting=accounting,
        initialization_seed=11,
    )
    wrapper = handle.installed[0]
    assert wrapper.residual.starts_as_exact_zero_correction
    torch.testing.assert_close(wrapper(inputs), original(inputs), rtol=0, atol=0)
    torch.testing.assert_close(
        model(input_ids=torch.tensor([[1, 2]])).logits, before, rtol=0, atol=0
    )
    with torch.no_grad():
        wrapper.residual.down_weight.normal_()
    baked = bake_behavior_residual_swiglu(wrapper)
    torch.testing.assert_close(baked(inputs), wrapper(inputs), rtol=1e-5, atol=1e-6)
    handle.rollback()
    assert layer.mlp is original
    torch.testing.assert_close(
        model(input_ids=torch.tensor([[1, 2]])).logits, before, rtol=0, atol=0
    )


def test_output_training_is_disjoint_bounded_and_restores_base() -> None:
    torch.manual_seed(9)
    model = _TinyCausalLM()
    original = model.model.layers[0].mlp
    base_versions = {id(value): int(value._version) for value in model.parameters()}
    accounting = BehaviorResidualSwiGLUAccounting.build(
        hidden_size=6,
        residual_width=2,
        site_count=1,
        removed_prototype_count=4,
        selected_redirected_assignments=3,
        checkpoint_element_size=4,
    )
    handle = install_behavior_residual_swiglu(
        model,
        layer_indices=(0,),
        residual_width=2,
        accounting=accounting,
    )
    fit = (
        _batch(
            sequence_id="fit-parent",
            split="fit",
            prefix_source="parent-greedy",
            tokens=(1, 2),
            action=3,
        ),
        _batch(
            sequence_id="fit-student",
            split="fit",
            prefix_source="student-greedy",
            tokens=(1, 4),
            action=3,
        ),
    )
    calibration = (
        _batch(
            sequence_id="cal-parent",
            split="calibration-selection",
            prefix_source="parent-greedy",
            tokens=(5, 6),
            action=2,
        ),
        _batch(
            sequence_id="cal-student",
            split="calibration-selection",
            prefix_source="student-greedy",
            tokens=(5, 7),
            action=2,
        ),
    )
    report = train_behavior_residual_swiglu_disjoint(
        handle,
        fit,
        calibration,
        steps=4,
        calibration_steps=(0, 2, 4),
    )
    assert report.fit_gradient_calls == 4
    assert report.calibration_evaluations == 4
    assert report.best_step in {0, 2, 4}
    assert report.as_dict()["excluded_objectives"] == [
        "weight-similarity",
        "activation-similarity",
        "hidden-state-similarity",
        "mlp-output-similarity",
        "residual-stream-similarity",
    ]
    handle.assert_base_unchanged()
    handle.rollback()
    assert model.model.layers[0].mlp is original
    assert all(
        int(value._version) == base_versions[id(value)] for value in model.parameters()
    )


def _evaluation(
    *, top1: float, rank: float, total_loss: float = 1.0
) -> FunctionalActionEvaluation:
    return FunctionalActionEvaluation(
        split="calibration-selection",
        example_count=8,
        total_loss=total_loss,
        hard_action_loss=1.0,
        distribution_loss=1.0,
        parent_action_top1_rate=top1,
        mean_parent_action_log_probability=-1.0,
        mean_parent_action_rank=rank,
        used_for_selection=True,
        prefix_sources=("parent-greedy", "student-greedy"),
    )


def _rollout(*, exact: int, tokens: int, mean_lcp: float) -> dict[str, object]:
    return {
        "prompt_count": 2,
        "exact_rollout_matches": exact,
        "token_matches": tokens,
        "token_count": 8,
        "mean_longest_common_prefix": mean_lcp,
        "output_tokens_only": True,
    }


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _rollout_provenance() -> dict[str, object]:
    prompt_hashes = [hashlib.sha256(name.encode()).hexdigest() for name in ("a", "b")]
    value: dict[str, object] = {
        "selection_split": "calibration-selection",
        "prompt_count": 2,
        "prompt_token_row_sha256s": prompt_hashes,
        "prompt_token_domain_sha256": _sha256(prompt_hashes),
        "heldout_prompt_token_domain_sha256": hashlib.sha256(b"heldout").hexdigest(),
        "calibration_heldout_token_overlap_count": 0,
    }
    value["provenance_sha256"] = _sha256(value)
    return value


def _rollout_record(*, regressed: bool) -> dict[str, object]:
    rows = [
        {
            "sequence_id": "calibration-selection-prompt-0",
            "exact_rollout_match": True,
            "token_matches": 4,
            "token_count": 4,
            "longest_common_prefix": 4,
        },
        {
            "sequence_id": "calibration-selection-prompt-1",
            "exact_rollout_match": not regressed,
            "token_matches": 3 if regressed else 4,
            "token_count": 4,
            "longest_common_prefix": 3 if regressed else 4,
        },
    ]
    exact = 1 if regressed else 2
    tokens = 7 if regressed else 8
    lcp = 3.5 if regressed else 4.0
    comparison: dict[str, object] = {
        "prompt_count": 2,
        "exact_rollout_matches": exact,
        "exact_rollout_rate": exact / 2,
        "token_matches": tokens,
        "token_count": 8,
        "token_match_rate": tokens / 8,
        "mean_longest_common_prefix": lcp,
        "rows": rows,
        "output_tokens_only": True,
    }
    trace_hashes = [
        {
            "sequence_id": row["sequence_id"],
            "prompt_token_ids_sha256": hashlib.sha256(
                f"prompt-{index}".encode()
            ).hexdigest(),
            "parent_selected_token_ids_sha256": hashlib.sha256(
                f"parent-{index}".encode()
            ).hexdigest(),
            "child_selected_token_ids_sha256": hashlib.sha256(
                f"child-{index}-{regressed}".encode()
            ).hexdigest(),
        }
        for index, row in enumerate(rows)
    ]
    record = {
        **comparison,
        "comparison_sha256": _sha256(comparison),
        "trace_hashes": trace_hashes,
        "selection_provenance": _rollout_provenance(),
    }
    record["rollout_evaluation_sha256"] = _sha256(record)
    return record


def test_rollout_first_key_rejects_higher_top1_with_worse_rollouts() -> None:
    incumbent = calibration_rollout_selection_key(
        _evaluation(top1=0.50, rank=5.0),
        _rollout(exact=2, tokens=8, mean_lcp=4.0),
    )
    superficially_better_action = calibration_rollout_selection_key(
        _evaluation(top1=0.99, rank=1.0),
        _rollout(exact=1, tokens=7, mean_lcp=3.5),
    )
    assert incumbent < superficially_better_action


def test_rollout_promotion_requires_pareto_nonregression() -> None:
    incumbent_action = _evaluation(top1=0.50, rank=5.0)
    candidate_action = _evaluation(top1=0.99, rank=1.0)
    assert not calibration_rollout_candidate_can_promote(
        incumbent_action,
        _rollout(exact=0, tokens=7, mean_lcp=3.5),
        candidate_action,
        _rollout(exact=1, tokens=6, mean_lcp=3.0),
    )
    assert not calibration_rollout_candidate_can_promote(
        incumbent_action,
        _rollout(exact=1, tokens=7, mean_lcp=3.5),
        candidate_action,
        _rollout(exact=1, tokens=8, mean_lcp=3.0),
    )


def _training_case():
    model = _TinyCausalLM()
    accounting = BehaviorResidualSwiGLUAccounting.build(
        hidden_size=6,
        residual_width=1,
        site_count=1,
        removed_prototype_count=3,
        selected_redirected_assignments=2,
        checkpoint_element_size=4,
    )
    handle = install_behavior_residual_swiglu(
        model,
        layer_indices=(0,),
        residual_width=1,
        accounting=accounting,
    )
    fit = (
        _batch(
            sequence_id="fit-a",
            split="fit",
            prefix_source="parent-greedy",
            tokens=(1, 2),
            action=3,
        ),
        _batch(
            sequence_id="fit-b",
            split="fit",
            prefix_source="student-greedy",
            tokens=(1, 3),
            action=3,
        ),
    )
    calibration = (
        _batch(
            sequence_id="cal-a",
            split="calibration-selection",
            prefix_source="parent-greedy",
            tokens=(5, 6),
            action=2,
        ),
        _batch(
            sequence_id="cal-b",
            split="calibration-selection",
            prefix_source="student-greedy",
            tokens=(5, 7),
            action=2,
        ),
    )
    return handle, fit, calibration


def _fit_phase_provenance(
    batches: tuple[FunctionalActionBatch, ...], *, refresh_after_step: int
) -> dict[str, object]:
    prefix_hashes: set[str] = set()
    for batch in batches:
        for row_index in range(batch.batch_size):
            length = int(batch.attention_mask[row_index].sum().item())
            prefix_hashes.add(_sha256(batch.input_ids[row_index, :length].tolist()))
    value: dict[str, object] = {
        "schema": "neural-forge-behavior-residual-on-policy-fit-phase-v1",
        "refresh_after_step": refresh_after_step,
        "fit_split": "fit",
        "fit_prompt_token_domain_sha256": _sha256("fit-prompts"),
        "student_driver_trace_sha256": _sha256(f"drivers-{refresh_after_step}"),
        "parent_target_trace_sha256": _sha256(f"targets-{refresh_after_step}"),
        "fit_prefix_domain_sha256": _sha256(sorted(prefix_hashes)),
        "batch_manifest_sha256": _sha256([batch.as_dict() for batch in batches]),
        "refreshed_from_current_student_rollout": True,
        "calibration_used_for_training": False,
        "calibration_model_forwards": 0,
        "heldout_model_forwards": 0,
        "parent_output_targets_only": True,
    }
    value["phase_provenance_sha256"] = _sha256(value)
    return value


def test_trainer_cannot_lock_higher_top1_state_that_regresses_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle, fit, calibration = _training_case()
    action_evaluations = iter(
        (
            _evaluation(top1=0.50, rank=5.0),
            _evaluation(top1=0.99, rank=1.0),
            _evaluation(top1=0.50, rank=5.0),
        )
    )
    rollout_evaluations = iter(
        (
            _rollout_record(regressed=False),
            _rollout_record(regressed=True),
            _rollout_record(regressed=False),
        )
    )

    monkeypatch.setattr(
        "neural_forge.hf_behavior_residual_swiglu.evaluate_functional_action_batches",
        lambda *_args, **_kwargs: next(action_evaluations),
    )
    report = train_behavior_residual_swiglu_disjoint(
        handle,
        fit,
        calibration,
        steps=1,
        calibration_steps=(0, 1),
        calibration_rollout_evaluator=lambda: next(rollout_evaluations),
        calibration_rollout_provenance=_rollout_provenance(),
    )
    assert report.best_step == 0
    assert report.best_calibration.parent_action_top1_rate == 0.50
    assert report.calibration_rollout_evaluations == 3
    assert report.calibration_history[1]["promoted"] is False
    handle.rollback()


def test_fit_refresh_preserves_adam_moments_and_never_uses_calibration() -> None:
    handle, fit, calibration = _training_case()
    refreshed = (
        _batch(
            sequence_id="refresh-parent",
            split="fit",
            prefix_source="parent-greedy",
            tokens=(8, 9),
            action=3,
        ),
        _batch(
            sequence_id="refresh-student",
            split="fit",
            prefix_source="student-greedy",
            tokens=(8, 10),
            action=3,
        ),
    )
    refresh_calls: list[int] = []

    def refresh(step: int) -> BehaviorResidualSwiGLUFitRefresh:
        refresh_calls.append(step)
        return BehaviorResidualSwiGLUFitRefresh(
            refresh_after_step=step,
            batches=refreshed,
            provenance=_fit_phase_provenance(refreshed, refresh_after_step=step),
        )

    report = train_behavior_residual_swiglu_disjoint(
        handle,
        fit,
        calibration,
        steps=2,
        calibration_steps=(0, 1, 2),
        fit_refresh_steps=(1,),
        fit_batch_refresher=refresh,
        initial_fit_provenance=_fit_phase_provenance(fit, refresh_after_step=0),
    )
    assert refresh_calls == [1]
    assert report.fit_refresh_evaluations == 1
    assert report.manual_adam_moment_reset_count == 0
    assert [row["fit_phase_index"] for row in report.fit_history] == [0, 1]
    assert report.fit_history[1]["sequence_id"] == "refresh-student"
    assert report.fit_phase_history[1]["manual_adam_moments_reset"] is False
    assert report.fit_phase_history[1]["batch_cursor_start"] == 1
    assert (
        report.fit_phase_history[1]["manual_adam_moment_sha256_before"]
        == report.fit_phase_history[1]["manual_adam_moment_sha256_after"]
    )
    assert (
        report.fit_phase_history[1]["manual_adam_moment_sha256_before"]
        != report.fit_phase_history[0]["manual_adam_moment_sha256_before"]
    )
    assert report.as_dict()[
        "generic_trainer_does_not_claim_unobserved_heldout_call_count"
    ]
    handle.rollback()


def test_exact_32_update_refresh_cursor_covers_every_schedule_position_once() -> None:
    handle, _fit, calibration = _training_case()

    def phase_batches(phase: int) -> tuple[FunctionalActionBatch, ...]:
        return tuple(
            _batch(
                sequence_id=f"phase-{phase}-position-{position}",
                split="fit",
                prefix_source=(
                    "parent-greedy" if position % 2 == 0 else "student-greedy"
                ),
                tokens=(1, 2 + ((position // 2) % 3)),
                action=3,
            )
            for position in range(32)
        )

    phases = {index: phase_batches(index) for index in range(4)}
    phase_by_step = {8: 1, 16: 2, 24: 3}

    def refresh(step: int) -> BehaviorResidualSwiGLUFitRefresh:
        phase = phase_by_step[step]
        batches = phases[phase]
        return BehaviorResidualSwiGLUFitRefresh(
            refresh_after_step=step,
            batches=batches,
            provenance=_fit_phase_provenance(batches, refresh_after_step=step),
        )

    report = train_behavior_residual_swiglu_disjoint(
        handle,
        phases[0],
        calibration,
        steps=32,
        calibration_steps=(0, 8, 16, 32),
        fit_refresh_steps=(8, 16, 24),
        fit_batch_refresher=refresh,
        initial_fit_provenance=_fit_phase_provenance(phases[0], refresh_after_step=0),
    )
    assert [row["fit_batch_schedule_position"] for row in report.fit_history] == list(
        range(32)
    )
    assert [row["batch_cursor_start"] for row in report.fit_phase_history] == [
        0,
        8,
        16,
        24,
    ]
    assert report.fit_history[8]["sequence_id"] == "phase-1-position-8"
    assert report.fit_history[16]["sequence_id"] == "phase-2-position-16"
    assert report.fit_history[24]["sequence_id"] == "phase-3-position-24"
    handle.rollback()


def test_locked_winner_requires_identical_per_id_rollout_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle, fit, calibration = _training_case()
    changed_reproduction = _rollout_record(regressed=False)
    changed_reproduction["trace_hashes"][0][  # type: ignore[index]
        "child_selected_token_ids_sha256"
    ] = hashlib.sha256(b"different-same-aggregate-trace").hexdigest()
    changed_reproduction["rollout_evaluation_sha256"] = _sha256(
        {
            key: value
            for key, value in changed_reproduction.items()
            if key != "rollout_evaluation_sha256"
        }
    )
    action_evaluations = iter(_evaluation(top1=0.50, rank=5.0) for _index in range(3))
    rollout_evaluations = iter(
        (
            _rollout_record(regressed=False),
            _rollout_record(regressed=False),
            changed_reproduction,
        )
    )
    monkeypatch.setattr(
        "neural_forge.hf_behavior_residual_swiglu.evaluate_functional_action_batches",
        lambda *_args, **_kwargs: next(action_evaluations),
    )
    with pytest.raises(
        BehaviorResidualSwiGLUTrainingError,
        match="full action and per-prompt rollout",
    ):
        train_behavior_residual_swiglu_disjoint(
            handle,
            fit,
            calibration,
            steps=1,
            calibration_steps=(0, 1),
            calibration_rollout_evaluator=lambda: next(rollout_evaluations),
            calibration_rollout_provenance=_rollout_provenance(),
        )
    handle.rollback()


def test_locked_winner_requires_all_action_metrics_not_only_selection_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle, fit, calibration = _training_case()
    action_evaluations = iter(
        (
            _evaluation(top1=0.50, rank=5.0),
            _evaluation(top1=0.50, rank=5.0),
            _evaluation(top1=0.50, rank=5.0, total_loss=2.0),
        )
    )
    rollout_evaluations = iter(_rollout_record(regressed=False) for _index in range(3))
    monkeypatch.setattr(
        "neural_forge.hf_behavior_residual_swiglu.evaluate_functional_action_batches",
        lambda *_args, **_kwargs: next(action_evaluations),
    )
    with pytest.raises(
        BehaviorResidualSwiGLUTrainingError,
        match="full action and per-prompt rollout",
    ):
        train_behavior_residual_swiglu_disjoint(
            handle,
            fit,
            calibration,
            steps=1,
            calibration_steps=(0, 1),
            calibration_rollout_evaluator=lambda: next(rollout_evaluations),
            calibration_rollout_provenance=_rollout_provenance(),
        )
    handle.rollback()


def test_calibration_rollout_callback_must_be_state_pure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle, fit, calibration = _training_case()
    monkeypatch.setattr(
        "neural_forge.hf_behavior_residual_swiglu.evaluate_functional_action_batches",
        lambda *_args, **_kwargs: _evaluation(top1=0.50, rank=5.0),
    )

    def mutate_candidate():
        with torch.no_grad():
            handle.output_trainable_parameters()[0].add_(1.0)
        return _rollout_record(regressed=False)

    with pytest.raises(
        BehaviorResidualSwiGLUTrainingError,
        match="mutated the candidate residual state",
    ):
        train_behavior_residual_swiglu_disjoint(
            handle,
            fit,
            calibration,
            steps=1,
            calibration_steps=(0, 1),
            calibration_rollout_evaluator=mutate_candidate,
            calibration_rollout_provenance=_rollout_provenance(),
        )
    handle.rollback()


def test_training_rejects_fit_calibration_prefix_overlap() -> None:
    model = _TinyCausalLM()
    accounting = BehaviorResidualSwiGLUAccounting.build(
        hidden_size=6,
        residual_width=1,
        site_count=1,
        removed_prototype_count=3,
        selected_redirected_assignments=2,
        checkpoint_element_size=4,
    )
    handle = install_behavior_residual_swiglu(
        model,
        layer_indices=(0,),
        residual_width=1,
        accounting=accounting,
    )
    fit = (
        _batch(
            sequence_id="fit-a",
            split="fit",
            prefix_source="parent-greedy",
            tokens=(1, 2),
            action=3,
        ),
        _batch(
            sequence_id="fit-b",
            split="fit",
            prefix_source="student-greedy",
            tokens=(1, 3),
            action=3,
        ),
    )
    calibration = (
        _batch(
            sequence_id="cal-a",
            split="calibration-selection",
            prefix_source="parent-greedy",
            tokens=(1, 2),
            action=3,
        ),
        _batch(
            sequence_id="cal-b",
            split="calibration-selection",
            prefix_source="student-greedy",
            tokens=(4, 5),
            action=3,
        ),
    )
    with pytest.raises(BehaviorResidualSwiGLUTrainingError, match="overlapping"):
        train_behavior_residual_swiglu_disjoint(
            handle,
            fit,
            calibration,
            steps=2,
            calibration_steps=(0, 2),
        )
    handle.rollback()


def test_residual_random_features_are_seeded_but_not_blank() -> None:
    first = ResidualSwiGLUBranch(
        5, 3, dtype=torch.float32, device="cpu", initialization_seed=13
    )
    second = ResidualSwiGLUBranch(
        5, 3, dtype=torch.float32, device="cpu", initialization_seed=13
    )
    assert torch.equal(first.gate_weight, second.gate_weight)
    assert torch.equal(first.up_weight, second.up_weight)
    assert int(torch.count_nonzero(first.gate_weight)) == 15
    assert int(torch.count_nonzero(first.up_weight)) == 15
    assert first.starts_as_exact_zero_correction


def test_nonfinite_disposable_factor_cannot_block_exact_rollback() -> None:
    model = _TinyCausalLM()
    original = model.model.layers[0].mlp
    accounting = BehaviorResidualSwiGLUAccounting.build(
        hidden_size=6,
        residual_width=1,
        site_count=1,
        removed_prototype_count=3,
        selected_redirected_assignments=2,
        checkpoint_element_size=4,
    )
    handle = install_behavior_residual_swiglu(
        model,
        layer_indices=(0,),
        residual_width=1,
        accounting=accounting,
    )
    with torch.no_grad():
        handle.installed[0].residual.down_weight[0, 0] = torch.nan
    handle.rollback()
    assert model.model.layers[0].mlp is original


def test_multisite_installation_failure_is_atomic() -> None:
    model = _TinyCausalLM()

    class BadLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Identity()

    model.model.layers.append(BadLayer())
    original = model.model.layers[0].mlp
    accounting = BehaviorResidualSwiGLUAccounting.build(
        hidden_size=6,
        residual_width=1,
        site_count=2,
        removed_prototype_count=8,
        selected_redirected_assignments=6,
        checkpoint_element_size=4,
    )
    with pytest.raises(BehaviorResidualSwiGLUTopologyError):
        install_behavior_residual_swiglu(
            model,
            layer_indices=(0, 1),
            residual_width=1,
            accounting=accounting,
        )
    assert model.model.layers[0].mlp is original
