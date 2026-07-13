from __future__ import annotations

# ruff: noqa: E402

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import prepare_qwen_exact_target_native_continuation_splits as split_tools
import prepare_qwen_exact_target_native_guided_candidate_tournament as prepare
import run_qwen_exact_target_native_guided_candidate_tournament as tournament
import run_qwen_exact_target_native_stage as stage
from neural_forge.hf_exact_candidate_frontier import ExactCandidateFrontierError
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BehaviorCheckpoint,
    CausalBehaviorMetrics,
    DurableBehaviorCheckpoint,
    DurableFactorStateSnapshot,
    DurableFactorTensorSnapshot,
    GuidedFactorSourceBinding,
)
import neural_forge.hf_target_native_behavior_distillation as behavior


QKVO_GUD_NAMES = {
    *(f"qkvo.{role}_{side}" for role in ("q", "k", "v", "o") for side in ("left", "right")),
    *(f"gud.{role}_{side}" for role in ("gate", "up", "down") for side in ("left", "right")),
}
LEXICAL_NAMES = {
    f"lexical.{role}_{side}"
    for role in ("embedding", "lm_head")
    for side in ("left", "right")
}
GUIDED_NAMES = {
    f"guided_qkvo.{role}_{side}"
    for role in ("q", "k", "v", "o")
    for side in ("left", "right")
}
V3_NAMES = tuple(sorted(QKVO_GUD_NAMES | LEXICAL_NAMES | GUIDED_NAMES))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source_binding() -> GuidedFactorSourceBinding:
    return GuidedFactorSourceBinding(
        source_v2_factor_state_sha256=_digest("source-v2"),
        source_v2_tensor_sha256s=tuple(
            (name, _digest(f"v2:{name}"))
            for name in sorted(QKVO_GUD_NAMES | LEXICAL_NAMES)
        ),
        audit_report_sha256=_digest("audit"),
        basis_envelope_file_sha256=_digest("envelope"),
        basis_payload_sha256=_digest("basis-payload"),
        basis_manifest_sha256=_digest("basis-manifest"),
        source_basis_tensor_sha256s=tuple(
            (role, _digest(f"basis:{role}")) for role in ("q", "k", "v", "o")
        ),
        inherited_installation_sha256=_digest("installation"),
        inherited_factor_state_sha256_at_install=_digest("inherited-state"),
        adapter_site_indices_sha256=_digest("adapter-sites"),
        audit_site_indices_sha256=_digest("audit-sites"),
    )


def _factor_state(seed: int) -> DurableFactorStateSnapshot:
    binding = _source_binding()
    snapshots = []
    rows = []
    for index, name in enumerate(V3_NAMES):
        tensor = torch.tensor([float(seed), float(index + 1)], dtype=torch.float32)
        digest = behavior._tensor_sha256(tensor)
        snapshots.append(
            DurableFactorTensorSnapshot(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                tensor_sha256=digest,
                raw_bytes=tensor.view(torch.uint8).numpy().tobytes(),
            )
        )
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor_sha256": digest,
            }
        )
    factor_sha = behavior._factor_state_sha256(
        rows,
        schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=4,
        guided_source_binding=binding,
    )
    return DurableFactorStateSnapshot(
        schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        tensors=tuple(snapshots),
        factor_state_sha256=factor_sha,
        lexical_rank=4,
        guided_source_binding=binding,
    )


def _metrics(score: int) -> CausalBehaviorMetrics:
    exact = min(24, score // 8)
    lcp4 = min(24, score // 4)
    lcp = max(4 * lcp4, min(192, 16 + score))
    tokens = max(lcp, min(192, 24 + score))
    actions = min(192, 40 + score)
    return CausalBehaviorMetrics(
        split="calibration-selection",
        action_count=192,
        action_agreement_count=actions,
        action_accuracy=actions / 192,
        parent_action_mean_logprob=-4.0,
        parent_action_topk_fraction=0.2,
        parent_action_mean_rank=100.0,
        parent_action_worst_rank=1000,
        rollout_count=24,
        exact_rollout_count=exact,
        exact_rollout_fraction=exact / 24,
        rollout_token_count=192,
        token_match_count=tokens,
        token_match_fraction=tokens / 192,
        lcp_token_count=lcp,
        lcp_fraction=lcp / 192,
        lcp_at_least_4_count=lcp4,
        repeat_fraction_excess=max(0.0, 0.5 - score / 1000),
        adjacent_repeat_fraction_excess=max(0.0, 0.6 - score / 1000),
        longest_run_excess=max(0, 8 - score // 16),
        report_sha256=_digest(f"metrics:{score}"),
    )


def _event(arm_index: int, step: int) -> DurableBehaviorCheckpoint:
    score = arm_index * 20 + step
    return DurableBehaviorCheckpoint(
        step=step,
        metrics=_metrics(score),
        factor_state=_factor_state(arm_index * 1000 + step),
        pareto_safe_vs_step0=step > 0,
    )


def _prompt(base: int) -> dict[str, Any]:
    tokens = tuple([base] * 8 + [base + 100])
    digest = split_tools._token_sequence_sha256(tokens)
    return {
        "token_ids": list(tokens),
        "prompt_id": f"prompt-{digest}",
        "token_ids_sha256": digest,
    }


def _split(name: str, prompts: list[dict[str, Any]], path: Path) -> stage.SplitBinding:
    return stage.SplitBinding(
        name=name,
        path=path,
        sha256=_digest(f"{name}:{len(prompts)}"),
        expected_count=len(prompts),
        payload={"prompts": prompts},
        row_ids=tuple(value["prompt_id"] for value in prompts),
        prompt_sha256s=tuple(value["token_ids_sha256"] for value in prompts),
    )


def test_schedule_is_exact_candidate_only_contract() -> None:
    schedule = tournament.candidate_tournament_schedule()
    assert schedule.normalized_checkpoint_steps == tournament.CHECKPOINT_STEPS
    assert schedule.steps == 64
    assert schedule.learning_rate == 5e-5
    assert schedule.fit_horizon == 16
    assert schedule.evaluation_horizon == 8
    assert schedule.fit_prompts_per_step == 8
    assert schedule.parent_only_steps == 4
    assert schedule.warmup_fit_horizon == 8
    assert schedule.parent_lane_loss_weight == 2.0
    assert schedule.child_lane_loss_weight == 2.0
    assert schedule.child_lane_starts_at_divergence is True
    assert schedule.max_child_lane_prefixes == 8
    assert schedule.child_repeat_stop_run_length == 0
    assert schedule.late_position_loss_multiplier == 2.0
    assert schedule.divergence_loss_multiplier == 6.0
    assert schedule.post_divergence_loss_multiplier == 3.0
    assert schedule.post_divergence_positions == 4
    assert schedule.repetition_excess_loss_weight == 0.4
    assert schedule.guided_qk_learning_rate_scale == 0.5
    assert schedule.guided_vo_learning_rate_scale == 1.0
    assert schedule.guided_right_factor_learning_rate_scale == 0.125
    assert schedule.guided_joint_inherited_learning_rate_scale == 0.25
    assert schedule.guided_gud_unlock_step == 0
    assert schedule.guided_lexical_embedding_unlock_step == 0
    assert schedule.guided_lexical_head_unlock_step == 0
    assert schedule.selection_policy == "rollout-first-acquisition"
    assert schedule.sealed_evaluation_policy == "disabled"


def test_rank_is_rollout_first_and_rejects_internal_metrics() -> None:
    weak = _metrics(8).as_dict()
    strong = _metrics(64).as_dict()
    assert tournament.tournament_behavior_rank(strong) < (
        tournament.tournament_behavior_rank(weak)
    )
    with pytest.raises(ExactCandidateFrontierError):
        tournament.tournament_behavior_rank({**strong, "weight_cosine": 1.0})
    with pytest.raises(ExactCandidateFrontierError):
        tournament.tournament_behavior_rank({**strong, "activation_mse": 0.0})


def test_split_evolution_promotes_candidate_and_keeps_fresh_candidate_disjoint(
    tmp_path: Path,
) -> None:
    old_fit = [_prompt(index) for index in range(1, 145)]
    old_candidate = [_prompt(index) for index in range(201, 225)]
    old_sealed = [_prompt(index) for index in range(301, 333)]
    sealed_path = tmp_path / "sealed.json"
    old = {
        "fit": _split("fit", old_fit, tmp_path / "old-fit.json"),
        "candidate": _split(
            "candidate", old_candidate, tmp_path / "old-candidate.json"
        ),
        "sealed": _split("sealed", old_sealed, sealed_path),
    }
    new_candidate = [_prompt(index) for index in range(401, 425)]
    active = {
        "fit": _split("fit", [*old_fit, *old_candidate], tmp_path / "fit.json"),
        "candidate": _split("candidate", new_candidate, tmp_path / "candidate.json"),
        "sealed": old["sealed"],
    }
    source = SimpleNamespace(execution_contract=SimpleNamespace(splits=old))
    tournament._authenticate_split_evolution(active, source)
    collided = dict(active)
    collided["candidate"] = _split(
        "candidate", [old_fit[0], *new_candidate[1:]], tmp_path / "bad.json"
    )
    with pytest.raises(tournament.CandidateTournamentError):
        tournament._authenticate_split_evolution(collided, source)


def test_model_free_builder_writes_fit168_fresh24_and_reuses_sealed32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_fit = [_prompt(index) for index in range(1, 145)]
    old_candidate = [_prompt(index) for index in range(201, 225)]
    old_sealed = [_prompt(index) for index in range(301, 333)]
    sealed_path = tmp_path / "sealed.json"
    old = {
        "fit": _split("fit", old_fit, tmp_path / "old-fit.json"),
        "candidate": _split(
            "candidate", old_candidate, tmp_path / "old-candidate.json"
        ),
        "sealed": _split("sealed", old_sealed, sealed_path),
    }
    source = SimpleNamespace(execution_contract=SimpleNamespace(splits=old))
    fresh = [
        split_tools._prompt_record(_prompt(index), f"fresh[{offset}]")
        for offset, index in enumerate(range(401, 425))
    ]
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    tokenizer_json = tokenizer_root / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        split_tools,
        "PINNED_TOKENIZER_JSON_SHA256",
        stage._file_sha256(tokenizer_json),
    )
    monkeypatch.setattr(split_tools, "_load_corpus", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        split_tools,
        "_select_fresh_candidate",
        lambda *_args, **_kwargs: (fresh, {"accepted": 24}),
    )
    tokenizer_loads = 0

    def load_tokenizer(_path: Path) -> object:
        nonlocal tokenizer_loads
        tokenizer_loads += 1
        return object()

    staging = tmp_path / "staging"
    staging.mkdir()
    bindings, evidence = prepare.prepare_tournament_splits(
        staging=staging,
        source=source,
        corpus_path=corpus,
        tokenizer_path=tokenizer_root,
        tokenizer_loader=load_tokenizer,
    )
    fit_payload = stage._strict_json(staging / "splits" / "fit.json")
    candidate_payload = stage._strict_json(staging / "splits" / "candidate.json")
    assert fit_payload["prompts"] == [*old_fit, *old_candidate]
    assert candidate_payload["prompts"] == [value.payload for value in fresh]
    assert bindings["sealed"] == {
        "path": str(sealed_path),
        "sha256": old["sealed"].sha256,
        "expected_count": 32,
    }
    assert evidence["active_counts"] == tournament.SPLIT_COUNTS
    assert evidence["model_weights_loaded"] is False
    assert evidence["model_forward_calls"] == 0
    assert tokenizer_loads == 1


class _FakeContract:
    def __init__(self) -> None:
        self.evaluation_contract_sha256 = _digest("evaluation")
        self.arms = (
            tournament.TournamentArm(
                tournament.ARM_IDS[0], "v3", _digest("initial-40"), False
            ),
            tournament.TournamentArm(
                tournament.ARM_IDS[1], "v3", _digest("initial-64"), False
            ),
            tournament.TournamentArm(
                tournament.ARM_IDS[2], "v2", _digest("initial-v2"), True
            ),
        )

    def preflight_report(self) -> dict[str, Any]:
        return {
            "schema": tournament.RUN_SCHEMA,
            "status": "preflight-passed",
            "sealed_suite_supplied_to_trainer": False,
        }


class _FakeRuntime:
    def __init__(self, *, deduplicate: bool = False) -> None:
        self.compiled_sha = _digest("compiled")
        self.calls: list[tuple[str, str, bool]] = []
        self.closed = False
        self.deduplicate = deduplicate
        self.duplicate_state = _factor_state(9999) if deduplicate else None

    def prepare(self, _contract: Any) -> dict[str, Any]:
        return {
            "compiled_state_sha256": self.compiled_sha,
            "models_loaded": False,
            "gpu_work_launched": False,
        }

    def run_arm(self, _contract: Any, arm: Any, callback: Any) -> Any:
        arm_index = tournament.ARM_IDS.index(arm.arm_id)
        checkpoints = []
        for step in tournament.CHECKPOINT_STEPS:
            event = _event(arm_index, step)
            if self.duplicate_state is not None and step > 0:
                event = DurableBehaviorCheckpoint(
                    step=step,
                    metrics=event.metrics,
                    factor_state=self.duplicate_state,
                    pareto_safe_vs_step0=True,
                )
            callback(event)
            checkpoints.append(
                BehaviorCheckpoint(
                    step=step,
                    metrics=event.metrics,
                    factor_state_sha256=event.factor_state.factor_state_sha256,
                    pareto_safe_vs_step0=event.pareto_safe_vs_step0,
                )
            )
        self.calls.append(
            (arm.arm_id, arm.initial_factor_state_sha256, arm.site_mask_enabled)
        )
        result = SimpleNamespace(
            checkpoints=tuple(checkpoints),
            sealed_metrics=None,
            sealed_evaluation_unlocked=False,
            sealed_evaluation_policy="disabled",
            evidence_sha256=_digest(f"evidence:{arm.arm_id}"),
        )
        return tournament.ArmExecution(
            result=result,
            reset_state_sha256=self.compiled_sha,
            rollback_state_sha256=self.compiled_sha,
            initial_factor_state_sha256=arm.initial_factor_state_sha256,
            site_mask_enabled=arm.site_mask_enabled,
            post_optimizer_mask_verification_count=64 if arm.site_mask_enabled else 0,
            sealed_forward_count=0,
        )

    def close(self) -> None:
        self.closed = True


def test_tournament_retains_top_three_selects_one_winner_and_never_uses_sealed(
    tmp_path: Path,
) -> None:
    contract = _FakeContract()
    runtime = _FakeRuntime()
    output = tmp_path / "tournament"
    report = tournament.execute_candidate_tournament(
        contract, output, runtime=runtime
    )
    assert report["status"] == "candidate-winner-selected"
    assert report["sealed_forward_count"] == 0
    assert report["sealed_evaluation_accessed"] is False
    assert report["model_checkpoint_written"] is False
    assert report["publication_attempted"] is False
    assert len(report["top_three_factor_state_sha256s"]) == 3
    assert len(set(report["top_three_factor_state_sha256s"])) == 3
    assert (output / "winner_selection.json").is_file()
    assert len(list((output / "retained").glob("*/factor_pack.pt"))) == 3
    assert [value[0] for value in runtime.calls] == list(tournament.ARM_IDS)
    assert [value[1] for value in runtime.calls] == [
        value.initial_factor_state_sha256 for value in contract.arms
    ]
    assert [value[2] for value in runtime.calls] == [False, False, True]
    assert runtime.closed is True
    assert all(row["sealed_forward_count"] == 0 for row in report["arm_executions"])
    assert report["arm_executions"][-1][
        "post_optimizer_mask_verification_count"
    ] == 64


def test_tournament_fails_closed_when_three_distinct_states_do_not_exist(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(deduplicate=True)
    output = tmp_path / "no-winner"
    with pytest.raises(tournament.CandidateTournamentError):
        tournament.execute_candidate_tournament(
            _FakeContract(), output, runtime=runtime
        )
    assert not (output / "winner_selection.json").exists()
    assert (output / "run_report.json").is_file()
    assert runtime.closed is True
