"""Zero-gradient Qwen pair refolding on child-generated token trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


REPORT_SCHEMA = "neural-forge-qwen-rollout-refold-v2"
PAIR_LINEAR_NAMES = ("q", "k", "v", "o", "gate", "up", "down")

__all__ = [
    "REPORT_SCHEMA",
    "refold_qwen_pair_composition_on_rollouts_models",
    "validate_exact_rollout_token_contract",
]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _special_token_ids(tokenizer) -> dict[str, Any]:
    result = {}
    for name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        value = getattr(tokenizer, name, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value = [int(item) for item in value]
        elif value is not None:
            value = int(value)
        result[name] = value
    return result


def validate_exact_rollout_token_contract(
    source_tokenizer,
    target_tokenizer,
    donor,
    target,
) -> dict[str, Any]:
    """Require one exact runtime token-ID contract for zero-copy rollout traces."""

    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    if not isinstance(source_vocab, Mapping) or not isinstance(target_vocab, Mapping):
        raise ValueError("Exact rollout refolding requires tokenizer vocabulary maps.")
    normalized_source = sorted(
        (str(token), int(index)) for token, index in source_vocab.items()
    )
    normalized_target = sorted(
        (str(token), int(index)) for token, index in target_vocab.items()
    )
    if normalized_source != normalized_target:
        raise ValueError(
            "Exact rollout refolding requires identical source and target token-ID maps."
        )
    source_special = _special_token_ids(source_tokenizer)
    target_special = _special_token_ids(target_tokenizer)
    if source_special != target_special:
        raise ValueError(
            "Exact rollout refolding requires identical source and target special token IDs."
        )
    vocab_rows = max((index for _token, index in normalized_source), default=-1) + 1
    donor_input_rows = int(donor.get_input_embeddings().weight.shape[0])
    donor_output_rows = int(donor.get_output_embeddings().weight.shape[0])
    target_input_rows = int(target.get_input_embeddings().weight.shape[0])
    target_output_rows = int(target.get_output_embeddings().weight.shape[0])
    row_counts = {
        "vocabulary_rows": vocab_rows,
        "donor_input_rows": donor_input_rows,
        "donor_output_rows": donor_output_rows,
        "target_input_rows": target_input_rows,
        "target_output_rows": target_output_rows,
    }
    if any(
        value < vocab_rows
        for key, value in row_counts.items()
        if key != "vocabulary_rows"
    ):
        raise ValueError(
            "A model embedding or output head cannot address the tokenizer vocabulary."
        )
    return {
        "exact_token_ids": True,
        "runtime_bridge": False,
        "vocabulary_sha256": _json_sha256(normalized_source),
        "special_token_ids": source_special,
        **row_counts,
    }


def _normalize_alphas(values: Sequence[float]) -> tuple[float, ...]:
    normalized = []
    for value in values:
        alpha = float(value)
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError(
                "rollout refold candidate alphas must be finite in (0, 1]."
            )
        if alpha not in normalized:
            normalized.append(alpha)
    if not normalized:
        raise ValueError("rollout refolding needs at least one candidate alpha.")
    return tuple(sorted(normalized, reverse=True))


def _snapshot_pair(compose, spec: Mapping[str, Any]) -> dict[str, Any]:
    return {name: compose._snapshot_linear(spec[name]) for name in PAIR_LINEAR_NAMES}


def _restore_pair(compose, tracker, spec, snapshot, reason: str) -> None:
    for name in PAIR_LINEAR_NAMES:
        compose._restore_linear(tracker, spec[name], snapshot[name], f"{reason}-{name}")


def _blend_linear(
    compose, tracker, module, baseline, fitted, alpha: float, reason: str
) -> None:
    weight = baseline["weight"] + float(alpha) * (fitted["weight"] - baseline["weight"])
    tracker.write(module.weight, weight, reason)
    bias = getattr(module, "bias", None)
    if bias is not None:
        baseline_bias = baseline["bias"]
        fitted_bias = fitted["bias"]
        tracker.write(
            bias,
            baseline_bias + float(alpha) * (fitted_bias - baseline_bias),
            f"{reason}-bias",
        )


def _blend_pair(
    compose, tracker, spec, baseline, fitted, alpha: float, reason: str
) -> None:
    for name in PAIR_LINEAR_NAMES:
        _blend_linear(
            compose,
            tracker,
            spec[name],
            baseline[name],
            fitted[name],
            alpha,
            f"{reason}-{name}",
        )


def _snapshot_final(compose, target) -> dict[str, Any]:
    return {
        "norm": compose._snapshot_linear(compose._find_final_norm(target)),
        "head": compose._snapshot_linear(target.get_output_embeddings()),
    }


def _restore_final(compose, tracker, target, snapshot, reason: str) -> None:
    compose._restore_linear(
        tracker, compose._find_final_norm(target), snapshot["norm"], f"{reason}-norm"
    )
    compose._restore_linear(
        tracker,
        target.get_output_embeddings(),
        snapshot["head"],
        f"{reason}-head",
    )


def _blend_final(
    compose, tracker, target, baseline, fitted, alpha: float, reason: str
) -> None:
    _blend_linear(
        compose,
        tracker,
        compose._find_final_norm(target),
        baseline["norm"],
        fitted["norm"],
        alpha,
        f"{reason}-norm",
    )
    _blend_linear(
        compose,
        tracker,
        target.get_output_embeddings(),
        baseline["head"],
        fitted["head"],
        alpha,
        f"{reason}-head",
    )


def _pad_token_id(tokenizer) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        raise ValueError("Rollout refolding requires a pad or EOS token ID.")
    return int(value)


def _trace_state_from_rollout(
    torch,
    rollout: Mapping[str, Any],
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> dict[str, Any]:
    from neural_forge.hf_qwen_rollout import build_exact_trace_batches

    prompt_lengths = [int(item["prompt_length"]) for item in rollout["rollout_records"]]
    batches = build_exact_trace_batches(
        rollout["token_id_sequences"],
        batch_size=batch_size,
        pad_token_id=_pad_token_id(tokenizer),
        device=device,
        prompt_lengths=prompt_lengths,
        max_length=max_length,
        vocab_size=len(tokenizer),
        torch_module=torch,
    )
    return {
        "rollout": rollout,
        "batches": batches,
        "prompt_count": len(rollout["rollout_records"]),
        "prompt_lengths": prompt_lengths,
    }


def _rollout_state(
    torch,
    donor,
    target,
    tokenizer,
    seed_texts: Sequence[str],
    *,
    max_prompts: int,
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    device: str,
    top_k: int,
) -> dict[str, Any]:
    from neural_forge.hf_qwen_rollout import generate_greedy_child_rollouts
    from neural_forge.hf_rollout_gate import evaluate_parent_rollouts

    rollout = generate_greedy_child_rollouts(
        target,
        tokenizer,
        seed_texts,
        max_prompt_count=max_prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        device=device,
        torch_module=torch,
    )
    state = _trace_state_from_rollout(
        torch,
        rollout,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    score = evaluate_parent_rollouts(
        donor,
        rollout,
        batch_size=batch_size,
        top_k=top_k,
        pad_token_id=_pad_token_id(tokenizer),
        device=device,
        torch_module=torch,
    )
    state["score"] = score
    return state


def _fixed_parent_anchor_state(
    torch,
    donor,
    tokenizer,
    seed_texts: Sequence[str],
    *,
    max_prompts: int,
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    from neural_forge.hf_qwen_rollout import generate_greedy_child_rollouts

    rollout = generate_greedy_child_rollouts(
        donor,
        tokenizer,
        seed_texts,
        max_prompt_count=max_prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        device=device,
        torch_module=torch,
    )
    return _trace_state_from_rollout(
        torch,
        rollout,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )


def _score_fixed_parent_anchors(
    torch,
    target,
    fixed_parent_anchors: Mapping[str, Any],
    tokenizer,
    *,
    batch_size: int,
    top_k: int,
    device: str,
) -> dict[str, Any]:
    from neural_forge.hf_rollout_gate import evaluate_parent_rollouts

    return evaluate_parent_rollouts(
        target,
        fixed_parent_anchors["rollout"],
        batch_size=batch_size,
        top_k=top_k,
        pad_token_id=_pad_token_id(tokenizer),
        device=device,
        torch_module=torch,
    )


def _rollout_hash_summary(rollout: Mapping[str, Any]) -> dict[str, Any]:
    sequences = [list(row) for row in rollout["token_id_sequences"]]
    prompt_rows = [
        list(record["token_ids"][: int(record["prompt_length"])])
        for record in rollout["rollout_records"]
    ]
    provenance = dict(rollout.get("provenance", {}))
    return {
        "prompt_count": len(sequences),
        "prompt_token_ids_sha256": _json_sha256(prompt_rows),
        "sequence_token_ids_sha256": rollout.get(
            "sequence_token_ids_sha256", _json_sha256(sequences)
        ),
        "rollout_records_sha256": _json_sha256(rollout["rollout_records"]),
        "provenance_sha256": provenance.get(
            "provenance_sha256", _json_sha256(provenance)
        ),
    }


def _score_hash_summary(score: Mapping[str, Any]) -> dict[str, Any]:
    hashes = dict(score.get("hashes", {}))
    return {
        "parent_top1_agreement": float(score["parent_top1_agreement"]),
        "mean_parent_action_logprob": float(score["mean_parent_action_logprob"]),
        "generated_repetition_rate": float(score["generated_repetition_rate"]),
        "generated_longest_run": int(score["generated_longest_run"]),
        "generated_action_count": int(score["generated_action_count"]),
        "report_sha256": hashes.get("report_sha256", _json_sha256(score)),
        "sequence_token_ids_sha256": hashes.get("sequence_token_ids_sha256"),
    }


def _build_mixed_trajectory_state(
    torch,
    current_child: Mapping[str, Any],
    fixed_parent_anchors: Mapping[str, Any],
    tokenizer,
    prompt_split: Mapping[str, Any],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> dict[str, Any]:
    """Interleave child and fixed-parent trajectories without prompt leakage."""

    from neural_forge.hf_qwen_rollout import build_exact_trace_batches

    child_rollout = current_child["rollout"]
    anchor_rollout = fixed_parent_anchors["rollout"]
    child_prompts = [str(item) for item in child_rollout["prompts"]]
    anchor_prompts = [str(item) for item in anchor_rollout["prompts"]]
    if child_prompts != anchor_prompts:
        raise ValueError(
            "Child and fixed-parent trajectories must use identical prompts."
        )
    prompt_count = len(child_prompts)
    if prompt_count == 0:
        raise ValueError("Mixed rollout fitting requires at least one prompt.")

    child_records = list(child_rollout["rollout_records"])
    anchor_records = list(anchor_rollout["rollout_records"])
    if len(child_records) != prompt_count or len(anchor_records) != prompt_count:
        raise ValueError("Mixed rollout records do not align with their prompt count.")

    train_groups = [int(index) for index in prompt_split["train_indices"]]
    validation_groups = [int(index) for index in prompt_split["validation_indices"]]
    train_set = set(train_groups)
    validation_set = set(validation_groups)
    expected_groups = set(range(prompt_count))
    if train_set & validation_set or train_set | validation_set != expected_groups:
        raise ValueError("Prompt split must cover each prompt exactly once.")

    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    trajectory_records: list[dict[str, Any]] = []
    group_to_trajectory_indices: list[list[int]] = []
    for prompt_index, (child_record, anchor_record) in enumerate(
        zip(child_records, anchor_records, strict=True)
    ):
        child_prompt_ids = list(
            child_record["token_ids"][: int(child_record["prompt_length"])]
        )
        anchor_prompt_ids = list(
            anchor_record["token_ids"][: int(anchor_record["prompt_length"])]
        )
        if child_prompt_ids != anchor_prompt_ids:
            raise ValueError(
                "Child and fixed-parent trajectories must share exact prompt token IDs."
            )
        indices = []
        for role, record in (
            ("current-child", child_record),
            ("fixed-parent-anchor", anchor_record),
        ):
            trajectory_index = len(sequences)
            token_ids = [int(item) for item in record["token_ids"]]
            prompt_length = int(record["prompt_length"])
            sequences.append(token_ids)
            prompt_lengths.append(prompt_length)
            indices.append(trajectory_index)
            trajectory_records.append(
                {
                    "trajectory_index": trajectory_index,
                    "prompt_group_index": prompt_index,
                    "role": role,
                    "source_index": int(record.get("source_index", prompt_index)),
                    "prompt_length": prompt_length,
                    "generated_length": len(token_ids) - prompt_length,
                    "prompt_token_ids_sha256": _json_sha256(token_ids[:prompt_length]),
                    "generated_token_ids_sha256": _json_sha256(
                        token_ids[prompt_length:]
                    ),
                    "sequence_token_ids_sha256": _json_sha256(token_ids),
                }
            )
        group_to_trajectory_indices.append(indices)

    train_indices = [
        index
        for prompt_index in train_groups
        for index in group_to_trajectory_indices[prompt_index]
    ]
    validation_indices = [
        index
        for prompt_index in validation_groups
        for index in group_to_trajectory_indices[prompt_index]
    ]
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("Mixed trajectory split leaked a prompt across partitions.")
    if len(train_indices) + len(validation_indices) != len(sequences):
        raise RuntimeError("Mixed trajectory split omitted one or more trajectories.")

    batches = build_exact_trace_batches(
        sequences,
        batch_size=batch_size,
        pad_token_id=_pad_token_id(tokenizer),
        device=device,
        prompt_lengths=prompt_lengths,
        max_length=max_length,
        vocab_size=len(tokenizer),
        torch_module=torch,
    )
    provenance = {
        "schema": "neural-forge-mixed-rollout-trajectories-v1",
        "ordering": "prompt-major-current-child-then-fixed-parent-anchor",
        "prompt_group_count": prompt_count,
        "trajectory_count": len(sequences),
        "trajectories_per_prompt": 2,
        "roles": ["current-child", "fixed-parent-anchor"],
        "fixed_parent_anchor_generated_once": True,
        "exact_prompt_token_identity": True,
        "prompt_group_split": {
            "train_prompt_indices": train_groups,
            "validation_prompt_indices": validation_groups,
            "train_trajectory_indices": train_indices,
            "validation_trajectory_indices": validation_indices,
            "group_to_trajectory_indices": group_to_trajectory_indices,
            "no_prompt_leakage": True,
            "seed": int(prompt_split.get("seed", 0)),
        },
        "trajectories": trajectory_records,
        "hashes": {
            "ordered_prompt_texts_sha256": _json_sha256(child_prompts),
            "ordered_mixed_token_ids_sha256": _json_sha256(sequences),
            "ordered_trajectory_records_sha256": _json_sha256(trajectory_records),
            "child_rollout_sha256": _rollout_hash_summary(child_rollout)[
                "sequence_token_ids_sha256"
            ],
            "fixed_parent_anchor_sha256": _rollout_hash_summary(anchor_rollout)[
                "sequence_token_ids_sha256"
            ],
        },
    }
    provenance["hashes"]["provenance_sha256"] = _json_sha256(provenance)
    return {
        "batches": batches,
        "prompt_count": len(sequences),
        "prompt_lengths": prompt_lengths,
        "trajectory_group_ids": [
            record["prompt_group_index"] for record in trajectory_records
        ],
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "provenance": provenance,
    }


def _promotion_decision(
    incumbent,
    candidate,
    *,
    min_top1_gain: float,
    min_action_logprob_gain: float,
    max_repetition_regression: float,
    max_longest_run_regression: int,
    min_action_count_ratio: float,
) -> dict[str, Any]:
    from neural_forge.hf_rollout_gate import compare_rollout_gate_reports

    decision = compare_rollout_gate_reports(
        incumbent,
        candidate,
        min_top1_gain=min_top1_gain,
        min_action_logprob_gain=min_action_logprob_gain,
        max_repetition_regression=max_repetition_regression,
        max_longest_run_regression=max_longest_run_regression,
    )
    incumbent_actions = int(incumbent["generated_action_count"])
    candidate_actions = int(candidate["generated_action_count"])
    required_actions = math.ceil(float(min_action_count_ratio) * incumbent_actions)
    decision["action_count_gate"] = {
        "incumbent": incumbent_actions,
        "candidate": candidate_actions,
        "minimum_ratio": float(min_action_count_ratio),
        "minimum_candidate_actions": required_actions,
        "accepted": candidate_actions >= required_actions,
    }
    if candidate_actions < required_actions:
        decision["accepted"] = False
        decision["reverted"] = True
        decision["decision"] = "reverted"
        decision["reason"] = "generated_action_count_below_threshold"
        decision["reasons"] = ["generated_action_count_below_threshold"]
        decision["accepted_reason"] = None
        decision["reverted_reason"] = "generated_action_count_below_threshold"
    return decision


def _anchor_promotion_decision(
    incumbent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    min_top1_gain: float,
    min_action_logprob_gain: float,
) -> dict[str, Any]:
    """Require the child not to regress on fixed parent-generated actions."""

    incumbent_top1 = float(incumbent["parent_top1_agreement"])
    candidate_top1 = float(candidate["parent_top1_agreement"])
    incumbent_logprob = float(incumbent["mean_parent_action_logprob"])
    candidate_logprob = float(candidate["mean_parent_action_logprob"])
    top1_delta = candidate_top1 - incumbent_top1
    logprob_delta = candidate_logprob - incumbent_logprob
    top1_required = float(min_top1_gain)
    logprob_required = float(min_action_logprob_gain)
    top1_accepted = top1_delta >= top1_required
    logprob_accepted = logprob_delta >= logprob_required
    accepted = top1_accepted and logprob_accepted
    reasons = []
    if not top1_accepted:
        reasons.append("fixed_parent_anchor_top1_below_required_gain")
    if not logprob_accepted:
        reasons.append("fixed_parent_anchor_logprob_below_required_gain")
    if accepted:
        reasons.append("fixed_parent_anchor_metrics_satisfied")
    decision = {
        "schema": "neural-forge-fixed-parent-anchor-gate-v1",
        "accepted": accepted,
        "reverted": not accepted,
        "decision": "accepted" if accepted else "reverted",
        "reason": reasons[0],
        "reasons": reasons,
        "metric_policy": "both-metrics-must-meet-independent-required-gains",
        "thresholds": {
            "min_child_top1_gain_on_fixed_parent_actions": top1_required,
            "min_child_action_logprob_gain_on_fixed_parent_actions": logprob_required,
        },
        "deltas": {
            "child_top1_on_fixed_parent_actions": top1_delta,
            "child_action_logprob_on_fixed_parent_actions": logprob_delta,
        },
        "metric_gates": {
            "top1": top1_accepted,
            "action_logprob": logprob_accepted,
        },
        "incumbent": _score_hash_summary(incumbent),
        "candidate": _score_hash_summary(candidate),
        "hashes": {
            "incumbent_report_sha256": _json_sha256(incumbent),
            "candidate_report_sha256": _json_sha256(candidate),
        },
    }
    decision["hashes"]["decision_sha256"] = _json_sha256(decision)
    return decision


def _dual_promotion_decision(
    incumbent_anchor: Mapping[str, Any],
    candidate_anchor: Mapping[str, Any],
    incumbent_child: Mapping[str, Any],
    candidate_child: Mapping[str, Any],
    *,
    min_top1_gain: float,
    min_action_logprob_gain: float,
    max_repetition_regression: float,
    max_longest_run_regression: int,
    min_action_count_ratio: float,
) -> dict[str, Any]:
    anchor_gate = _anchor_promotion_decision(
        incumbent_anchor,
        candidate_anchor,
        min_top1_gain=min_top1_gain,
        min_action_logprob_gain=min_action_logprob_gain,
    )
    child_rollout_gate = _promotion_decision(
        incumbent_child,
        candidate_child,
        min_top1_gain=min_top1_gain,
        min_action_logprob_gain=min_action_logprob_gain,
        max_repetition_regression=max_repetition_regression,
        max_longest_run_regression=max_longest_run_regression,
        min_action_count_ratio=min_action_count_ratio,
    )
    accepted = bool(anchor_gate["accepted"]) and bool(child_rollout_gate["accepted"])
    failed_gates = []
    if not bool(anchor_gate["accepted"]):
        failed_gates.append("fixed_parent_anchor_gate")
    if not bool(child_rollout_gate["accepted"]):
        failed_gates.append("regenerated_child_rollout_gate")
    reason = "both_global_gates_satisfied" if accepted else failed_gates[0]
    decision = {
        "schema": "neural-forge-dual-rollout-promotion-v1",
        "accepted": accepted,
        "reverted": not accepted,
        "decision": "accepted" if accepted else "reverted",
        "reason": reason,
        "reasons": [reason] if accepted else failed_gates,
        "required_gates": [
            "fixed_parent_anchor_gate",
            "regenerated_child_rollout_gate",
        ],
        "all_required_gates_must_pass": True,
        "fixed_parent_anchor_gate": anchor_gate,
        "regenerated_child_rollout_gate": child_rollout_gate,
    }
    decision["decision_sha256"] = _json_sha256(decision)
    return decision


def _attempt_dual_gated_candidates(
    *,
    alphas: Sequence[float],
    apply_alpha: Callable[[float], None],
    restore: Callable[[], None],
    build_child_state: Callable[[], dict[str, Any]],
    score_fixed_anchors: Callable[[], dict[str, Any]],
    incumbent_child_state: Mapping[str, Any],
    incumbent_anchor_score: Mapping[str, Any],
    min_top1_gain: float,
    min_action_logprob_gain: float,
    max_repetition_regression: float,
    max_longest_run_regression: int,
    min_action_count_ratio: float,
) -> dict[str, Any]:
    """Try closed-form candidate blends and restore unless both gates pass."""

    attempts = []
    committed = False
    try:
        for alpha in alphas:
            apply_alpha(float(alpha))
            candidate_child_state = build_child_state()
            candidate_anchor_score = score_fixed_anchors()
            dual_decision = _dual_promotion_decision(
                incumbent_anchor_score,
                candidate_anchor_score,
                incumbent_child_state["score"],
                candidate_child_state["score"],
                min_top1_gain=min_top1_gain,
                min_action_logprob_gain=min_action_logprob_gain,
                max_repetition_regression=max_repetition_regression,
                max_longest_run_regression=max_longest_run_regression,
                min_action_count_ratio=min_action_count_ratio,
            )
            attempt = {
                "alpha": float(alpha),
                "candidate_child_rollout": _rollout_hash_summary(
                    candidate_child_state["rollout"]
                ),
                "candidate_child_rollout_score": _score_hash_summary(
                    candidate_child_state["score"]
                ),
                "candidate_fixed_anchor_score": _score_hash_summary(
                    candidate_anchor_score
                ),
                "decision": dual_decision,
                "dual_decision": dual_decision,
                "post_commit_anchor_recheck": None,
                "accepted": False,
            }
            attempts.append(attempt)
            if not bool(dual_decision["accepted"]):
                continue

            refreshed_anchor_score = score_fixed_anchors()
            anchor_recheck = _anchor_promotion_decision(
                incumbent_anchor_score,
                refreshed_anchor_score,
                min_top1_gain=min_top1_gain,
                min_action_logprob_gain=min_action_logprob_gain,
            )
            attempt["post_commit_anchor_recheck"] = anchor_recheck
            if not bool(anchor_recheck["accepted"]):
                continue
            attempt["accepted"] = True
            committed = True
            return {
                "accepted": True,
                "accepted_alpha": float(alpha),
                "child_state": candidate_child_state,
                "anchor_score": refreshed_anchor_score,
                "attempts": attempts,
                "restored": False,
            }
        return {
            "accepted": False,
            "accepted_alpha": None,
            "child_state": None,
            "anchor_score": None,
            "attempts": attempts,
            "restored": True,
        }
    finally:
        if not committed:
            restore()


def _fit_final_interface(
    torch,
    compose,
    donor,
    target,
    donor_traces,
    structural,
    state,
    train_indices,
    validation_indices,
    *,
    ridge: float,
    validation_fraction: float,
    min_train_prompts: int,
    min_validation_prompts: int,
    min_relative_improvement: float,
    max_cosine_drop: float,
    max_fitted_relative_rmse: float | None,
    max_fit_rows: int,
    min_validation_rows: int,
    head_validation_vocab_rows: int,
    split_seed: int,
    tracker,
) -> dict[str, Any]:
    batches = state["batches"]
    prompt_count = int(state["prompt_count"])
    child_before = compose._trace_child_final_interface(
        torch, target, batches, prompt_count
    )
    norm_report = compose._fit_final_rms_norm(
        torch,
        target,
        child_before,
        donor_traces["final_projected"],
        train_indices,
        validation_indices,
        ridge,
        min_relative_improvement,
        max_cosine_drop,
        max_fitted_relative_rmse,
        max_fit_rows,
        min_validation_rows,
        tracker,
    )
    child_after_norm = compose._trace_child_final_interface(
        torch, target, batches, prompt_count
    )
    projected_head_report = compose._fit_final_head(
        torch,
        target,
        child_after_norm["norm_output"],
        donor_traces["final_projected"],
        structural["head_token_mapping"],
        train_indices,
        validation_indices,
        ridge,
        min_relative_improvement,
        max_cosine_drop,
        max_fitted_relative_rmse,
        max_fit_rows,
        min_validation_rows,
        head_validation_vocab_rows,
        tracker,
    )
    child_rows, child_prompt_ids = compose._cat_prompt_rows_with_ids(
        torch, child_after_norm["norm_output"], max_fit_rows
    )
    donor_rows, donor_prompt_ids = compose._cat_prompt_rows_with_ids(
        torch, donor_traces["final_hidden"], max_fit_rows
    )
    if child_prompt_ids != donor_prompt_ids:
        raise RuntimeError(
            "Rollout final-state rows do not align between parent and child."
        )
    trajectory_group_ids = state.get("trajectory_group_ids")
    if trajectory_group_ids is not None:
        child_prompt_ids = [
            f"prompt-group-{int(trajectory_group_ids[int(prompt_id)])}"
            for prompt_id in child_prompt_ids
        ]
    from neural_forge.hf_qwen_logit_head import fit_qwen_logit_head

    result = fit_qwen_logit_head(
        child_rows,
        donor_rows,
        donor.get_output_embeddings().weight,
        target.get_output_embeddings().weight,
        prompt_ids=child_prompt_ids,
        ridge=ridge,
        validation_fraction=validation_fraction,
        min_train_prompts=min_train_prompts,
        min_validation_prompts=min_validation_prompts,
        split_seed=split_seed,
        vocab_chunk_size=head_validation_vocab_rows,
    )
    tracker.write(
        target.get_output_embeddings().weight,
        result["weight"],
        "rollout-full-vocabulary-parent-logit-head-fit",
    )
    report = {
        "norm": norm_report,
        "projected_head": projected_head_report,
        "full_head": result["report"],
    }
    del result
    return report


def _gradient_references(model) -> tuple[tuple[int, Any], ...]:
    return tuple((id(parameter), parameter.grad) for parameter in model.parameters())


def _gradients_preserved(
    before: Sequence[tuple[int, Any]], after: Sequence[tuple[int, Any]]
) -> bool:
    return len(before) == len(after) and all(
        before_id == after_id and before_gradient is after_gradient
        for (before_id, before_gradient), (after_id, after_gradient) in zip(
            before, after, strict=True
        )
    )


def _linear_snapshot_equal(
    torch, left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if not bool(torch.equal(left["weight"], right["weight"])):
        return False
    left_bias = left.get("bias")
    right_bias = right.get("bias")
    if left_bias is None or right_bias is None:
        return left_bias is None and right_bias is None
    return bool(torch.equal(left_bias, right_bias))


def _snapshot_mapping_equal(
    torch, left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return set(left) == set(right) and all(
        _linear_snapshot_equal(torch, left[name], right[name]) for name in left
    )


def refold_qwen_pair_composition_on_rollouts_models(
    donor,
    target,
    source_tokenizer,
    target_tokenizer,
    *,
    pair_plan,
    structural: Mapping[str, Any],
    basis,
    pair_bases,
    calibration_texts: Sequence[str],
    tracker,
    rounds: int,
    max_prompts: int,
    max_new_tokens: int,
    candidate_alphas: Sequence[float] = (1.0, 0.5, 0.25),
    rollout_batch_size: int = 4,
    top_k: int = 5,
    min_top1_gain: float = 0.0,
    min_action_logprob_gain: float = 0.0,
    max_repetition_regression: float = 0.0,
    max_longest_run_regression: int = 0,
    min_action_count_ratio: float = 0.9,
    ridge: float = 1e-3,
    validation_fraction: float = 0.25,
    min_validation_prompts: int = 1,
    min_train_prompts: int = 2,
    min_relative_improvement: float = 0.02,
    max_cosine_drop: float = 0.0,
    max_fitted_relative_rmse: float | None = 0.75,
    max_length: int = 160,
    max_fit_rows: int = 4096,
    min_validation_rows: int = 128,
    head_validation_vocab_rows: int = 4096,
    split_seed: int = 0,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Refit child pairs on mixed child and fixed parent-generated trajectories."""

    if int(rounds) <= 0:
        return {"schema": REPORT_SCHEMA, "enabled": False, "rounds": []}
    if int(max_prompts) < int(min_train_prompts) + int(min_validation_prompts):
        raise ValueError("rollout max_prompts cannot satisfy the prompt split.")
    if not 0.0 < float(min_action_count_ratio) <= 1.0:
        raise ValueError("min_action_count_ratio must be in (0, 1].")
    alphas = _normalize_alphas(candidate_alphas)

    import torch
    from neural_forge import hf_qwen_compose as compose
    from neural_forge.hf_qwen_rollout import derive_prompt_only_strings

    token_contract = validate_exact_rollout_token_contract(
        source_tokenizer, target_tokenizer, donor, target
    )
    seed_texts = derive_prompt_only_strings(
        calibration_texts, max_prompt_count=int(max_prompts)
    )
    seed_texts = [text for text in seed_texts if text.strip()]
    if len(seed_texts) < int(min_train_prompts) + int(min_validation_prompts):
        raise ValueError("Too few non-empty rollout prompts after response stripping.")

    donor.to(device)
    target.to(device)
    donor.eval()
    target.eval()
    donor_gradients_before = _gradient_references(donor)
    target_gradients_before = _gradient_references(target)

    fixed_parent_anchors = _fixed_parent_anchor_state(
        torch,
        donor,
        source_tokenizer,
        seed_texts,
        max_prompts=int(max_prompts),
        max_length=int(max_length),
        max_new_tokens=int(max_new_tokens),
        batch_size=int(rollout_batch_size),
        device=device,
    )
    current = _rollout_state(
        torch,
        donor,
        target,
        target_tokenizer,
        seed_texts,
        max_prompts=int(max_prompts),
        max_length=int(max_length),
        max_new_tokens=int(max_new_tokens),
        batch_size=int(rollout_batch_size),
        device=device,
        top_k=int(top_k),
    )
    current_anchor_score = _score_fixed_parent_anchors(
        torch,
        target,
        fixed_parent_anchors,
        target_tokenizer,
        batch_size=int(rollout_batch_size),
        top_k=int(top_k),
        device=device,
    )
    initial_score = current["score"]
    initial_anchor_score = current_anchor_score
    target_layers = list(compose._decoder_layer_container(target)[2])
    round_reports = []

    def build_child_state() -> dict[str, Any]:
        return _rollout_state(
            torch,
            donor,
            target,
            target_tokenizer,
            seed_texts,
            max_prompts=int(max_prompts),
            max_length=int(max_length),
            max_new_tokens=int(max_new_tokens),
            batch_size=int(rollout_batch_size),
            device=device,
            top_k=int(top_k),
        )

    def score_fixed_anchors() -> dict[str, Any]:
        return _score_fixed_parent_anchors(
            torch,
            target,
            fixed_parent_anchors,
            target_tokenizer,
            batch_size=int(rollout_batch_size),
            top_k=int(top_k),
            device=device,
        )

    for round_index in range(int(rounds)):
        split = compose.prompt_level_split(
            current["rollout"]["prompts"],
            validation_fraction=validation_fraction,
            min_validation_prompts=min_validation_prompts,
            min_train_prompts=min_train_prompts,
            seed=int(split_seed) + round_index,
        )
        mixed = _build_mixed_trajectory_state(
            torch,
            current,
            fixed_parent_anchors,
            target_tokenizer,
            split,
            batch_size=int(rollout_batch_size),
            max_length=int(max_length),
            device=device,
        )
        mixed_history = [
            {
                "refresh_reason": "round-start",
                "provenance": mixed["provenance"],
            }
        ]
        donor_traces = None
        pair_reports = []
        for pair in pair_plan:
            if donor_traces is None:
                donor_traces = compose._trace_donor_pair_targets(
                    torch,
                    donor,
                    pair_plan,
                    structural["pair_plans"],
                    mixed["batches"],
                    basis,
                    mixed["prompt_count"],
                    pair_bases=pair_bases,
                )
            target_index = int(pair.target_layer)
            spec = compose._qwen_layer_spec(target_layers[target_index])
            baseline_snapshot = _snapshot_pair(compose, spec)
            fit_provenance_sha256 = mixed["provenance"]["hashes"]["provenance_sha256"]
            try:
                local_report = compose._fit_ordered_child_pair(
                    torch,
                    target,
                    pair,
                    donor_traces["pairs"][target_index],
                    mixed["batches"],
                    mixed["prompt_count"],
                    mixed["train_indices"],
                    mixed["validation_indices"],
                    ridge,
                    min_relative_improvement,
                    max_cosine_drop,
                    max_fitted_relative_rmse,
                    max_fit_rows,
                    min_validation_rows,
                    tracker,
                )
            except Exception:
                _restore_pair(
                    compose,
                    tracker,
                    spec,
                    baseline_snapshot,
                    f"rollout-v2-exception-revert-pair-{target_index}",
                )
                raise

            if bool(local_report["pair_validation"]["accepted"]):
                fitted_snapshot = _snapshot_pair(compose, spec)

                def apply_pair_alpha(alpha: float) -> None:
                    _blend_pair(
                        compose,
                        tracker,
                        spec,
                        baseline_snapshot,
                        fitted_snapshot,
                        alpha,
                        f"rollout-v2-pair-{target_index}-alpha-{alpha:g}",
                    )

                def restore_pair() -> None:
                    _restore_pair(
                        compose,
                        tracker,
                        spec,
                        baseline_snapshot,
                        f"rollout-v2-dual-gate-revert-pair-{target_index}",
                    )

                transaction = _attempt_dual_gated_candidates(
                    alphas=alphas,
                    apply_alpha=apply_pair_alpha,
                    restore=restore_pair,
                    build_child_state=build_child_state,
                    score_fixed_anchors=score_fixed_anchors,
                    incumbent_child_state=current,
                    incumbent_anchor_score=current_anchor_score,
                    min_top1_gain=min_top1_gain,
                    min_action_logprob_gain=min_action_logprob_gain,
                    max_repetition_regression=max_repetition_regression,
                    max_longest_run_regression=max_longest_run_regression,
                    min_action_count_ratio=min_action_count_ratio,
                )
            else:
                _restore_pair(
                    compose,
                    tracker,
                    spec,
                    baseline_snapshot,
                    f"rollout-v2-local-gate-revert-pair-{target_index}",
                )
                transaction = {
                    "accepted": False,
                    "accepted_alpha": None,
                    "child_state": None,
                    "anchor_score": None,
                    "attempts": [],
                    "restored": True,
                }

            rollback_exact = None
            if not bool(transaction["accepted"]):
                rollback_exact = _snapshot_mapping_equal(
                    torch,
                    baseline_snapshot,
                    _snapshot_pair(compose, spec),
                )
                if not rollback_exact:
                    raise RuntimeError(
                        f"Rejected rollout refold pair {target_index} was not fully restored."
                    )
            else:
                current = transaction["child_state"]
                current_anchor_score = transaction["anchor_score"]
                donor_traces = None
                mixed = _build_mixed_trajectory_state(
                    torch,
                    current,
                    fixed_parent_anchors,
                    target_tokenizer,
                    split,
                    batch_size=int(rollout_batch_size),
                    max_length=int(max_length),
                    device=device,
                )
                mixed_history.append(
                    {
                        "refresh_reason": f"accepted-pair-{target_index}",
                        "provenance": mixed["provenance"],
                    }
                )
            pair_reports.append(
                {
                    "target_layer": target_index,
                    "donor_layers": [int(item) for item in pair.donor_layers],
                    "local_fit": local_report,
                    "fit_trajectory_provenance_sha256": fit_provenance_sha256,
                    "fit_trajectory_roles": [
                        "current-child",
                        "fixed-parent-anchor",
                    ],
                    "candidate_alphas": list(alphas),
                    "attempts": transaction["attempts"],
                    "accepted": bool(transaction["accepted"]),
                    "accepted_alpha": transaction["accepted_alpha"],
                    "reverted": not bool(transaction["accepted"]),
                    "rollback_exact": rollback_exact,
                }
            )

        if donor_traces is None:
            donor_traces = compose._trace_donor_pair_targets(
                torch,
                donor,
                pair_plan,
                structural["pair_plans"],
                mixed["batches"],
                basis,
                mixed["prompt_count"],
                pair_bases=pair_bases,
            )
        final_baseline = _snapshot_final(compose, target)
        final_fit_provenance_sha256 = mixed["provenance"]["hashes"]["provenance_sha256"]
        try:
            final_local = _fit_final_interface(
                torch,
                compose,
                donor,
                target,
                donor_traces,
                structural,
                mixed,
                mixed["train_indices"],
                mixed["validation_indices"],
                ridge=ridge,
                validation_fraction=validation_fraction,
                min_train_prompts=min_train_prompts,
                min_validation_prompts=min_validation_prompts,
                min_relative_improvement=min_relative_improvement,
                max_cosine_drop=max_cosine_drop,
                max_fitted_relative_rmse=max_fitted_relative_rmse,
                max_fit_rows=max_fit_rows,
                min_validation_rows=min_validation_rows,
                head_validation_vocab_rows=head_validation_vocab_rows,
                split_seed=int(split_seed) + 10000 + round_index,
                tracker=tracker,
            )
        except Exception:
            _restore_final(
                compose,
                tracker,
                target,
                final_baseline,
                "rollout-v2-exception-revert-final-interface",
            )
            raise
        final_fitted = _snapshot_final(compose, target)

        def apply_final_alpha(alpha: float) -> None:
            _blend_final(
                compose,
                tracker,
                target,
                final_baseline,
                final_fitted,
                alpha,
                f"rollout-v2-final-alpha-{alpha:g}",
            )

        def restore_final() -> None:
            _restore_final(
                compose,
                tracker,
                target,
                final_baseline,
                "rollout-v2-dual-gate-revert-final-interface",
            )

        final_transaction = _attempt_dual_gated_candidates(
            alphas=alphas,
            apply_alpha=apply_final_alpha,
            restore=restore_final,
            build_child_state=build_child_state,
            score_fixed_anchors=score_fixed_anchors,
            incumbent_child_state=current,
            incumbent_anchor_score=current_anchor_score,
            min_top1_gain=min_top1_gain,
            min_action_logprob_gain=min_action_logprob_gain,
            max_repetition_regression=max_repetition_regression,
            max_longest_run_regression=max_longest_run_regression,
            min_action_count_ratio=min_action_count_ratio,
        )
        final_rollback_exact = None
        if not bool(final_transaction["accepted"]):
            final_rollback_exact = _snapshot_mapping_equal(
                torch,
                final_baseline,
                _snapshot_final(compose, target),
            )
            if not final_rollback_exact:
                raise RuntimeError(
                    "Rejected rollout refold final interface was not fully restored."
                )
        else:
            current = final_transaction["child_state"]
            current_anchor_score = final_transaction["anchor_score"]
            mixed = _build_mixed_trajectory_state(
                torch,
                current,
                fixed_parent_anchors,
                target_tokenizer,
                split,
                batch_size=int(rollout_batch_size),
                max_length=int(max_length),
                device=device,
            )
            mixed_history.append(
                {
                    "refresh_reason": "accepted-final-interface",
                    "provenance": mixed["provenance"],
                }
            )
        round_reports.append(
            {
                "round": round_index,
                "prompt_split": split,
                "mixed_trajectory_provenance": mixed_history,
                "pairs": pair_reports,
                "accepted_pairs": sum(bool(item["accepted"]) for item in pair_reports),
                "final_interface": {
                    "local_fit": final_local,
                    "fit_trajectory_provenance_sha256": (final_fit_provenance_sha256),
                    "attempts": final_transaction["attempts"],
                    "accepted": bool(final_transaction["accepted"]),
                    "accepted_alpha": final_transaction["accepted_alpha"],
                    "reverted": not bool(final_transaction["accepted"]),
                    "rollback_exact": final_rollback_exact,
                },
                "ending_rollout_score": current["score"],
                "ending_fixed_parent_anchor_score": current_anchor_score,
                "ending_rollout_provenance": current["rollout"]["provenance"],
            }
        )

    final_state = build_child_state()
    final_anchor_score = score_fixed_anchors()
    donor_gradients_after = _gradient_references(donor)
    target_gradients_after = _gradient_references(target)
    donor_gradients_preserved = _gradients_preserved(
        donor_gradients_before, donor_gradients_after
    )
    target_gradients_preserved = _gradients_preserved(
        target_gradients_before, target_gradients_after
    )
    if not donor_gradients_preserved or not target_gradients_preserved:
        raise RuntimeError("Rollout refolding changed a parameter gradient reference.")

    report = {
        "schema": REPORT_SCHEMA,
        "enabled": True,
        "trace_distribution": "current-child-greedy-rollout",
        "trace_distribution_scope": "regenerated-child-global-gate",
        "fit_trace_distribution": (
            "deterministic-union-of-current-child-and-fixed-parent-anchor-trajectories"
        ),
        "rollout_refresh_scope": "after-each-dual-gate-accepted-transaction",
        "fixed_parent_anchor_generation_scope": "once-before-all-rounds",
        "fixed_parent_anchor_rescore_scope": "after-each-accepted-transaction",
        "donor_prefix_identity": "exact-token-id",
        "zero_gradient": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_steps": 0,
        "parameter_gradients_written": False,
        "gradient_guarantees": {
            "donor_gradient_references_preserved": donor_gradients_preserved,
            "target_gradient_references_preserved": target_gradients_preserved,
            "donor_non_null_gradients_before": sum(
                gradient is not None for _, gradient in donor_gradients_before
            ),
            "donor_non_null_gradients_after": sum(
                gradient is not None for _, gradient in donor_gradients_after
            ),
            "target_non_null_gradients_before": sum(
                gradient is not None for _, gradient in target_gradients_before
            ),
            "target_non_null_gradients_after": sum(
                gradient is not None for _, gradient in target_gradients_after
            ),
        },
        "token_contract": token_contract,
        "fixed_parent_anchor": {
            "generated_once": True,
            "generation_count": 1,
            "generator": "donor-parent-greedy",
            "score_direction": "child-model-on-exact-fixed-parent-actions",
            "rollout": _rollout_hash_summary(fixed_parent_anchors["rollout"]),
        },
        "settings": {
            "rounds": int(rounds),
            "max_prompts": int(max_prompts),
            "max_new_tokens": int(max_new_tokens),
            "candidate_alphas": list(alphas),
            "batch_size": int(rollout_batch_size),
            "top_k": int(top_k),
            "min_top1_gain": float(min_top1_gain),
            "min_action_logprob_gain": float(min_action_logprob_gain),
            "anchor_metric_policy": (
                "both-top1-and-logprob-meet-independent-required-gains"
            ),
            "max_repetition_regression": float(max_repetition_regression),
            "max_longest_run_regression": int(max_longest_run_regression),
            "min_action_count_ratio": float(min_action_count_ratio),
        },
        "initial_rollout_score": initial_score,
        "initial_fixed_parent_anchor_score": initial_anchor_score,
        "rounds": round_reports,
        "final_rollout_score": final_state["score"],
        "final_rollout_provenance": final_state["rollout"]["provenance"],
        "final_fixed_parent_anchor_score": final_anchor_score,
    }
    report["report_sha256"] = _json_sha256(report)
    return report
