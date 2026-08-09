"""The stage-2 curve signal: free-run token match and lcp against the teacher.

``docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`` section 3.2 fixes the
only numeric signal the training curve may carry: greedy free-run rollouts of
256 donor tokens on a fixed 64-prompt probe drawn (seeded) from
``experiments/lfm2_on_policy_fit_prompts_v1.json``, scored as token match and
mean longest-common-prefix against the teacher's rollouts.  Both cheap
alternatives are formally disqualified -- terminal reconstruction swung 0.507
with no behavioural consequence, and teacher-forced top-1 moved *against*
free-run across four children (ceiling report sections 2.3 and 4.3) -- so
neither appears here, and ``tests/test_stage2_distillation_trainer.py`` pins
their absence from the whole stage-2 trainer surface.

Two screens run before any prompt is used.  The pack's own corpus digest and
disjointness declaration must verify, and every selected prompt is compared
directly against the 16 frozen V5 review prompts by the same
utf8-sha256-exact-and-stripped rule the pack used.  The review surface staying
out of training and out of the curve is a risk the design names explicitly
(section 12); it is cheaper to re-derive it here than to trust a field.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from .dynamic_circuit_ir import canonical_sha256


STAGE2_FREERUN_PROBE_SCHEMA = "neural-forge-stage2-freerun-probe-v1"
# Design section 3.2: the probe is drawn from the fit-prompt pack whose corpus
# digest and 552-digest disjointness check are already sealed.
LFM2_ON_POLICY_FIT_PROMPTS_SCHEMA = "neural-forge-lfm2-on-policy-fit-prompts-v1"
LFM2_ON_POLICY_FIT_PROMPTS_CORPUS_SHA256 = (
    "4bf7263dc5a6158c9a0ed4c2692ffd98b7ae4f56d5694dac6ddbd23e3071c452"
)
STAGE2_FREERUN_PROBE_PROMPT_COUNT = 64
STAGE2_FREERUN_PROBE_NEW_TOKENS = 256
# Design section 3.2: "Logged every 1M tokens."
STAGE2_FREERUN_PROBE_INTERVAL_TOKENS = 1_000_000
FROZEN_REVIEW_PROMPTS_SCHEMA = "neural-forge-whole-response-review-prompts-v1"


class Stage2FreeRunProbeError(RuntimeError):
    """A probe pack, screen, rollout, or metric set is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage2FreeRunProbeError(f"{name} must be a positive integer.")
    return value


def _strict_json(path: Path, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage2FreeRunProbeError(f"{label} repeats key {key!r}.")
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise Stage2FreeRunProbeError(f"{label} is missing or not a real file.")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except Stage2FreeRunProbeError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2FreeRunProbeError(f"{label} is not strict UTF-8 JSON.") from exc


def _text_digests(value: str) -> tuple[str, str]:
    """The pack's ``utf8-sha256-exact-and-stripped`` comparison, both forms."""

    exact = hashlib.sha256(value.encode("utf-8")).hexdigest()
    stripped = hashlib.sha256(value.strip().encode("utf-8")).hexdigest()
    return exact, stripped


def frozen_review_prompt_digests(review_prompts_json: Path) -> frozenset[str]:
    """Every renderable text of the 16-prompt frozen V5 surface, both forms."""

    payload = _strict_json(Path(review_prompts_json), "frozen review prompt pack")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != FROZEN_REVIEW_PROMPTS_SCHEMA
        or not isinstance(payload.get("prompts"), list)
    ):
        raise Stage2FreeRunProbeError(
            "Frozen review prompt pack is not the sealed V5 surface."
        )
    digests: set[str] = set()
    for row in payload["prompts"]:
        if not isinstance(row, Mapping):
            raise Stage2FreeRunProbeError("Frozen review row must be an object.")
        texts: list[str] = []
        messages = row.get("messages")
        if isinstance(messages, list):
            texts.extend(
                str(message.get("content", ""))
                for message in messages
                if isinstance(message, Mapping)
            )
        if isinstance(row.get("prompt"), str):
            texts.append(str(row["prompt"]))
        if not texts:
            raise Stage2FreeRunProbeError(
                "Frozen review row carries no comparable prompt text."
            )
        for text in texts:
            digests.update(_text_digests(text))
    return frozenset(digests)


@dataclass(frozen=True, slots=True)
class Stage2FreeRunProbePrompts:
    """The sealed 64-prompt probe surface for one measurement."""

    prompts: tuple[Mapping[str, Any], ...]
    seed: int
    source_corpus_sha256: str
    source_prompt_count: int
    frozen_review_screen_digest_count: int

    @property
    def probe_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": STAGE2_FREERUN_PROBE_SCHEMA,
                "seed": self.seed,
                "source_corpus_sha256": self.source_corpus_sha256,
                "prompt_ids": [str(row["id"]) for row in self.prompts],
                "prompt_texts": [str(row["prompt"]) for row in self.prompts],
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE2_FREERUN_PROBE_SCHEMA,
            "seed": self.seed,
            "probe_sha256": self.probe_sha256,
            "prompt_count": len(self.prompts),
            "prompt_ids": [str(row["id"]) for row in self.prompts],
            "source_corpus_sha256": self.source_corpus_sha256,
            "source_prompt_count": self.source_prompt_count,
            "frozen_review_surface_screened": True,
            "frozen_review_screen_digest_count": (
                self.frozen_review_screen_digest_count
            ),
        }


def load_stage2_freerun_probe_prompts(
    fit_prompts_json: Path,
    *,
    review_prompts_json: Path,
    seed: int,
    prompt_count: int = STAGE2_FREERUN_PROBE_PROMPT_COUNT,
) -> Stage2FreeRunProbePrompts:
    """Draw the seeded probe subset, refusing anything the design forbids."""

    payload = _strict_json(Path(fit_prompts_json), "on-policy fit prompt pack")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != LFM2_ON_POLICY_FIT_PROMPTS_SCHEMA
        or payload.get("corpus_sha256") != LFM2_ON_POLICY_FIT_PROMPTS_CORPUS_SHA256
        or not isinstance(payload.get("prompts"), list)
    ):
        raise Stage2FreeRunProbeError(
            "On-policy fit prompt pack is not the sealed corpus "
            f"{LFM2_ON_POLICY_FIT_PROMPTS_CORPUS_SHA256}."
        )
    disjointness = payload.get("reserved_manual_review_disjointness")
    if (
        not isinstance(disjointness, Mapping)
        or disjointness.get("verified_disjoint") is not True
        or disjointness.get("overlapping_prompt_count") != 0
    ):
        raise Stage2FreeRunProbeError(
            "Fit prompt pack does not declare verified review disjointness."
        )
    prompts = payload["prompts"]
    if payload.get("prompt_count") != len(prompts):
        raise Stage2FreeRunProbeError(
            "Fit prompt pack count disagrees with its own prompt list."
        )
    wanted = _positive_int(prompt_count, "prompt_count")
    if wanted > len(prompts):
        raise Stage2FreeRunProbeError(
            "Probe cannot draw more prompts than the pack contains."
        )
    ordered = sorted(
        (dict(row) for row in prompts if isinstance(row, Mapping)),
        key=lambda row: str(row.get("id", "")),
    )
    if len(ordered) != len(prompts) or any(
        not isinstance(row.get("id"), str) or not isinstance(row.get("prompt"), str)
        for row in ordered
    ):
        raise Stage2FreeRunProbeError(
            "Fit prompt rows must carry a string id and a string prompt."
        )
    selection = sorted(
        random.Random(int(seed)).sample(range(len(ordered)), wanted)
    )
    selected = tuple(ordered[index] for index in selection)
    screen = frozen_review_prompt_digests(Path(review_prompts_json))
    for row in selected:
        exact, stripped = _text_digests(str(row["prompt"]))
        if exact in screen or stripped in screen:
            raise Stage2FreeRunProbeError(
                "Probe selection touches the frozen review surface: "
                f"{row['id']!r}."
            )
    return Stage2FreeRunProbePrompts(
        prompts=selected,
        seed=int(seed),
        source_corpus_sha256=str(payload["corpus_sha256"]),
        source_prompt_count=len(ordered),
        frozen_review_screen_digest_count=len(screen),
    )


def greedy_donor_rollout(
    donor_logits: Callable[[Tensor], Tensor],
    prompt_ids: Sequence[int],
    *,
    maximum_new_tokens: int,
    end_of_sequence_ids: Sequence[int] = (),
    device: Any = None,
) -> tuple[int, ...]:
    """Free-run greedy decode in donor space, identical for both models.

    The teacher and the routed student expose the same callable contract --
    ``[batch, length] -> [batch, length, donor_rows]`` -- so both rollouts come
    from one decode loop.  A separate loop per model is exactly the kind of
    surface difference the design refuses to hand a comparison (section 5.4
    item 4).
    """

    limit = _positive_int(maximum_new_tokens, "maximum_new_tokens")
    if not prompt_ids:
        raise Stage2FreeRunProbeError("Free-run decode needs a nonempty prompt.")
    terminal = {int(value) for value in end_of_sequence_ids}
    context = [int(value) for value in prompt_ids]
    generated: list[int] = []
    with torch.no_grad():
        for _step in range(limit):
            batch = torch.tensor([context], dtype=torch.int64, device=device)
            logits = donor_logits(batch)
            if (
                not isinstance(logits, Tensor)
                or logits.ndim != 3
                or int(logits.shape[0]) != 1
                or int(logits.shape[1]) != len(context)
            ):
                raise Stage2FreeRunProbeError(
                    "Free-run decode expects [batch, length, donor_rows] logits."
                )
            selected = int(torch.argmax(logits[0, -1, :]).item())
            generated.append(selected)
            if selected in terminal:
                break
            context.append(selected)
    return tuple(generated)


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


def score_free_run_rollouts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate the sanctioned free-run metrics from per-prompt rollouts.

    Names follow the estate's existing rollout vocabulary
    (``hf_exact_candidate_frontier.rollout_first_rank``) so a stage-2 receipt
    and a stage-1 candidate record can be read side by side without a
    translation table.
    """

    if not rows:
        raise Stage2FreeRunProbeError("Free-run scoring needs at least one row.")
    records: list[dict[str, Any]] = []
    token_match_count = 0
    rollout_token_count = 0
    lcp_token_count = 0
    lcp_at_least_4_count = 0
    exact_rollout_count = 0
    for row in rows:
        student = tuple(int(value) for value in row["student_rollout"])
        teacher = tuple(int(value) for value in row["teacher_rollout"])
        if not teacher:
            raise Stage2FreeRunProbeError(
                "A teacher rollout must contain at least one token."
            )
        compared = min(len(student), len(teacher))
        matches = sum(
            1 for index in range(compared) if student[index] == teacher[index]
        )
        lcp = _longest_common_prefix(student, teacher)
        token_match_count += matches
        rollout_token_count += len(teacher)
        lcp_token_count += lcp
        lcp_at_least_4_count += 1 if lcp >= 4 else 0
        exact_rollout_count += 1 if student == teacher else 0
        records.append(
            {
                "prompt_id": str(row["prompt_id"]),
                "student_rollout_token_count": len(student),
                "teacher_rollout_token_count": len(teacher),
                "compared_token_count": compared,
                "token_match_count": matches,
                "longest_common_prefix": lcp,
            }
        )
    return {
        "schema": STAGE2_FREERUN_PROBE_SCHEMA,
        "rollout_count": len(records),
        "rollout_token_count": rollout_token_count,
        "token_match_count": token_match_count,
        "token_match_fraction": (
            0.0
            if rollout_token_count == 0
            else token_match_count / rollout_token_count
        ),
        "lcp_token_count": lcp_token_count,
        "lcp_fraction": (
            0.0 if rollout_token_count == 0 else lcp_token_count / rollout_token_count
        ),
        "mean_longest_common_prefix": lcp_token_count / len(records),
        "lcp_at_least_4_count": lcp_at_least_4_count,
        "exact_rollout_count": exact_rollout_count,
        "rows": records,
        "decode": "greedy-free-run-both-models",
        "terminal_reconstruction_used": False,
        "teacher_forced_top1_used": False,
        "acceptance_authority": "none-frozen-v5-manual-review-only",
    }


def run_stage2_freerun_probe(
    *,
    student_donor_logits: Callable[[Tensor], Tensor],
    teacher_donor_logits: Callable[[Tensor], Tensor],
    probe: Stage2FreeRunProbePrompts,
    tokenize: Callable[[str], Sequence[int]],
    maximum_new_tokens: int = STAGE2_FREERUN_PROBE_NEW_TOKENS,
    end_of_sequence_ids: Sequence[int] = (),
    device: Any = None,
    tokens_consumed: int = 0,
    optimizer_steps: int = 0,
    teacher_rollout_cache: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Decode both models freely on the probe and score the two rollout sets.

    The teacher is frozen, so its greedy rollouts are constant for the whole
    run; passing a cache turns every probe after the first into student-only
    work, which is what makes a 1M-token cadence affordable.
    """

    if not isinstance(probe, Stage2FreeRunProbePrompts):
        raise Stage2FreeRunProbeError("probe must be Stage2FreeRunProbePrompts.")
    cache = teacher_rollout_cache if teacher_rollout_cache is not None else {}
    rows: list[dict[str, Any]] = []
    for row in probe.prompts:
        prompt_id = str(row["id"])
        prompt_ids = tuple(int(value) for value in tokenize(str(row["prompt"])))
        if not prompt_ids:
            raise Stage2FreeRunProbeError(
                f"Probe prompt {prompt_id!r} tokenized to nothing."
            )
        teacher_rollout = cache.get(prompt_id)
        if teacher_rollout is None:
            teacher_rollout = greedy_donor_rollout(
                teacher_donor_logits,
                prompt_ids,
                maximum_new_tokens=maximum_new_tokens,
                end_of_sequence_ids=end_of_sequence_ids,
                device=device,
            )
            cache[prompt_id] = teacher_rollout
        student_rollout = greedy_donor_rollout(
            student_donor_logits,
            prompt_ids,
            maximum_new_tokens=maximum_new_tokens,
            end_of_sequence_ids=end_of_sequence_ids,
            device=device,
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "student_rollout": student_rollout,
                "teacher_rollout": teacher_rollout,
            }
        )
    scored = score_free_run_rollouts(rows)
    scored.update(
        {
            "probe_sha256": probe.probe_sha256,
            "maximum_new_tokens": int(maximum_new_tokens),
            "tokens_consumed": int(tokens_consumed),
            "optimizer_steps": int(optimizer_steps),
        }
    )
    return scored


__all__ = [
    "FROZEN_REVIEW_PROMPTS_SCHEMA",
    "LFM2_ON_POLICY_FIT_PROMPTS_CORPUS_SHA256",
    "LFM2_ON_POLICY_FIT_PROMPTS_SCHEMA",
    "STAGE2_FREERUN_PROBE_INTERVAL_TOKENS",
    "STAGE2_FREERUN_PROBE_NEW_TOKENS",
    "STAGE2_FREERUN_PROBE_PROMPT_COUNT",
    "STAGE2_FREERUN_PROBE_SCHEMA",
    "Stage2FreeRunProbeError",
    "Stage2FreeRunProbePrompts",
    "frozen_review_prompt_digests",
    "greedy_donor_rollout",
    "load_stage2_freerun_probe_prompts",
    "run_stage2_freerun_probe",
    "score_free_run_rollouts",
]
