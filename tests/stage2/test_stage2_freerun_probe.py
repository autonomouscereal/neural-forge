from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_forge.stage2_freerun_probe import (
    LFM2_ON_POLICY_FIT_PROMPTS_CORPUS_SHA256,
    STAGE2_FREERUN_PROBE_INTERVAL_TOKENS,
    STAGE2_FREERUN_PROBE_NEW_TOKENS,
    STAGE2_FREERUN_PROBE_PROMPT_COUNT,
    Stage2FreeRunProbeError,
    frozen_review_prompt_digests,
    greedy_donor_rollout,
    load_stage2_freerun_probe_prompts,
    run_stage2_freerun_probe,
    score_free_run_rollouts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIT_PROMPTS = REPOSITORY_ROOT / "experiments" / "lfm2_on_policy_fit_prompts_v1.json"
REVIEW_PROMPTS = (
    REPOSITORY_ROOT / "research" / "lfm2_semantic_whole_response_prompts_v1.json"
)


def test_design_constants_match_section_three_point_two() -> None:
    assert STAGE2_FREERUN_PROBE_PROMPT_COUNT == 64
    assert STAGE2_FREERUN_PROBE_NEW_TOKENS == 256
    assert STAGE2_FREERUN_PROBE_INTERVAL_TOKENS == 1_000_000
    assert LFM2_ON_POLICY_FIT_PROMPTS_CORPUS_SHA256.startswith("4bf7263d")


def test_the_probe_draw_is_seeded_sealed_and_review_disjoint() -> None:
    probe = load_stage2_freerun_probe_prompts(
        FIT_PROMPTS,
        review_prompts_json=REVIEW_PROMPTS,
        seed=17,
    )
    repeat = load_stage2_freerun_probe_prompts(
        FIT_PROMPTS,
        review_prompts_json=REVIEW_PROMPTS,
        seed=17,
    )
    other = load_stage2_freerun_probe_prompts(
        FIT_PROMPTS,
        review_prompts_json=REVIEW_PROMPTS,
        seed=18,
    )
    assert len(probe.prompts) == STAGE2_FREERUN_PROBE_PROMPT_COUNT
    assert probe.probe_sha256 == repeat.probe_sha256
    assert probe.probe_sha256 != other.probe_sha256
    assert probe.source_prompt_count == 120
    payload = probe.as_dict()
    assert payload["frozen_review_surface_screened"] is True
    assert payload["frozen_review_screen_digest_count"] > 0


def test_the_frozen_review_screen_catches_a_planted_overlap(
    tmp_path: Path,
) -> None:
    """The screen is re-derived from the V5 file, not trusted from a field."""

    review_text = json.loads(REVIEW_PROMPTS.read_text(encoding="utf-8"))
    leaked = str(review_text["prompts"][0]["messages"][0]["content"])
    pack = json.loads(FIT_PROMPTS.read_text(encoding="utf-8"))
    pack["prompts"][0]["prompt"] = leaked
    poisoned = tmp_path / "poisoned.json"
    poisoned.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(Stage2FreeRunProbeError, match="frozen review surface"):
        load_stage2_freerun_probe_prompts(
            poisoned,
            review_prompts_json=REVIEW_PROMPTS,
            seed=0,
            prompt_count=120,
        )


def test_a_pack_that_is_not_the_sealed_corpus_is_refused(tmp_path: Path) -> None:
    pack = json.loads(FIT_PROMPTS.read_text(encoding="utf-8"))
    pack["corpus_sha256"] = "0" * 64
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(Stage2FreeRunProbeError, match="not the sealed corpus"):
        load_stage2_freerun_probe_prompts(
            altered,
            review_prompts_json=REVIEW_PROMPTS,
            seed=1,
        )


def test_a_pack_without_a_verified_disjointness_claim_is_refused(
    tmp_path: Path,
) -> None:
    pack = json.loads(FIT_PROMPTS.read_text(encoding="utf-8"))
    pack["reserved_manual_review_disjointness"]["verified_disjoint"] = False
    altered = tmp_path / "undeclared.json"
    altered.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(Stage2FreeRunProbeError, match="verified review disjointness"):
        load_stage2_freerun_probe_prompts(
            altered,
            review_prompts_json=REVIEW_PROMPTS,
            seed=1,
        )


def test_the_review_screen_covers_every_v5_message() -> None:
    digests = frozen_review_prompt_digests(REVIEW_PROMPTS)
    payload = json.loads(REVIEW_PROMPTS.read_text(encoding="utf-8"))
    assert len(digests) >= len(payload["prompts"])


def test_greedy_rollout_is_deterministic_and_stops_at_end_of_sequence() -> None:
    def logits(donor_ids: torch.Tensor) -> torch.Tensor:
        length = int(donor_ids.shape[1])
        scores = torch.zeros(1, length, 5)
        for position in range(length):
            scores[0, position, (int(donor_ids[0, position]) + 1) % 5] = 1.0
        return scores

    first = greedy_donor_rollout(logits, [0], maximum_new_tokens=4)
    second = greedy_donor_rollout(logits, [0], maximum_new_tokens=4)
    assert first == second == (1, 2, 3, 4)
    stopped = greedy_donor_rollout(
        logits,
        [0],
        maximum_new_tokens=4,
        end_of_sequence_ids=(2,),
    )
    assert stopped == (1, 2)


def test_scoring_keeps_the_estate_rollout_invariants() -> None:
    scored = score_free_run_rollouts(
        [
            {
                "prompt_id": "a",
                "student_rollout": (1, 2, 3, 9),
                "teacher_rollout": (1, 2, 3, 4),
            },
            {
                "prompt_id": "b",
                "student_rollout": (5, 5),
                "teacher_rollout": (6, 5),
            },
        ]
    )
    assert scored["rollout_token_count"] == 6
    assert scored["token_match_count"] == 4
    assert scored["lcp_token_count"] == 3
    assert scored["lcp_at_least_4_count"] == 0
    assert scored["exact_rollout_count"] == 0
    assert scored["token_match_count"] >= scored["lcp_token_count"]
    assert scored["token_match_fraction"] == pytest.approx(4 / 6)
    assert scored["lcp_fraction"] == pytest.approx(3 / 6)
    assert scored["mean_longest_common_prefix"] == pytest.approx(1.5)
    assert scored["terminal_reconstruction_used"] is False
    assert scored["teacher_forced_top1_used"] is False
    assert scored["acceptance_authority"].startswith("none")


def test_an_identical_student_scores_a_perfect_exact_rollout() -> None:
    scored = score_free_run_rollouts(
        [
            {
                "prompt_id": "a",
                "student_rollout": (1, 2, 3, 4, 5),
                "teacher_rollout": (1, 2, 3, 4, 5),
            }
        ]
    )
    assert scored["exact_rollout_count"] == 1
    assert scored["lcp_at_least_4_count"] == 1
    assert scored["token_match_fraction"] == 1.0


def test_the_probe_runs_both_models_and_caches_the_frozen_teacher() -> None:
    probe = load_stage2_freerun_probe_prompts(
        FIT_PROMPTS,
        review_prompts_json=REVIEW_PROMPTS,
        seed=3,
        prompt_count=2,
    )
    calls = {"teacher": 0, "student": 0}

    def _constant(donor_ids: torch.Tensor, row: int) -> torch.Tensor:
        return torch.zeros(1, int(donor_ids.shape[1]), 4).index_fill(
            2,
            torch.tensor([row]),
            1.0,
        )

    def teacher(donor_ids: torch.Tensor) -> torch.Tensor:
        calls["teacher"] += 1
        return _constant(donor_ids, 1)

    def student(donor_ids: torch.Tensor) -> torch.Tensor:
        calls["student"] += 1
        return _constant(donor_ids, 2)

    cache: dict[str, tuple[int, ...]] = {}
    first = run_stage2_freerun_probe(
        student_donor_logits=student,
        teacher_donor_logits=teacher,
        probe=probe,
        tokenize=lambda text: [len(text) % 4],
        maximum_new_tokens=3,
        teacher_rollout_cache=cache,
    )
    teacher_calls_after_first = calls["teacher"]
    second = run_stage2_freerun_probe(
        student_donor_logits=student,
        teacher_donor_logits=teacher,
        probe=probe,
        tokenize=lambda text: [len(text) % 4],
        maximum_new_tokens=3,
        teacher_rollout_cache=cache,
    )
    assert first["rollout_count"] == 2
    assert first["token_match_count"] == 0
    assert first["probe_sha256"] == probe.probe_sha256
    assert calls["teacher"] == teacher_calls_after_first
    assert second["token_match_fraction"] == first["token_match_fraction"]
