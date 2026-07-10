"""Zero-gradient Qwen pair refolding on child-generated token trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


REPORT_SCHEMA = "neural-forge-qwen-rollout-refold-v1"
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
    normalized_source = sorted((str(token), int(index)) for token, index in source_vocab.items())
    normalized_target = sorted((str(token), int(index)) for token, index in target_vocab.items())
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
    if any(value < vocab_rows for key, value in row_counts.items() if key != "vocabulary_rows"):
        raise ValueError("A model embedding or output head cannot address the tokenizer vocabulary.")
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
            raise ValueError("rollout refold candidate alphas must be finite in (0, 1].")
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


def _blend_linear(compose, tracker, module, baseline, fitted, alpha: float, reason: str) -> None:
    weight = baseline["weight"] + float(alpha) * (
        fitted["weight"] - baseline["weight"]
    )
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


def _blend_pair(compose, tracker, spec, baseline, fitted, alpha: float, reason: str) -> None:
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


def _blend_final(compose, tracker, target, baseline, fitted, alpha: float, reason: str) -> None:
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
    from neural_forge.hf_qwen_rollout import (
        build_exact_trace_batches,
        generate_greedy_child_rollouts,
    )
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
    prompt_lengths = [int(item["prompt_length"]) for item in rollout["rollout_records"]]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("Rollout refolding requires a pad or EOS token ID.")
    batches = build_exact_trace_batches(
        rollout["token_id_sequences"],
        batch_size=batch_size,
        pad_token_id=int(pad_token_id),
        device=device,
        prompt_lengths=prompt_lengths,
        max_length=max_length,
        vocab_size=len(tokenizer),
        torch_module=torch,
    )
    score = evaluate_parent_rollouts(
        donor,
        rollout,
        batch_size=batch_size,
        top_k=top_k,
        pad_token_id=int(pad_token_id),
        device=device,
        torch_module=torch,
    )
    return {
        "rollout": rollout,
        "batches": batches,
        "score": score,
        "prompt_count": len(rollout["rollout_records"]),
        "prompt_lengths": prompt_lengths,
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
        raise RuntimeError("Rollout final-state rows do not align between parent and child.")
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
    """Refit complete child pairs on prefixes generated by the current child."""

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
    initial_score = current["score"]
    target_layers = list(compose._decoder_layer_container(target)[2])
    round_reports = []

    for round_index in range(int(rounds)):
        split = compose.prompt_level_split(
            current["rollout"]["prompts"],
            validation_fraction=validation_fraction,
            min_validation_prompts=min_validation_prompts,
            min_train_prompts=min_train_prompts,
            seed=int(split_seed) + round_index,
        )
        train_indices = split["train_indices"]
        validation_indices = split["validation_indices"]
        donor_traces = None
        pair_reports = []
        for pair in pair_plan:
            if donor_traces is None:
                donor_traces = compose._trace_donor_pair_targets(
                    torch,
                    donor,
                    pair_plan,
                    structural["pair_plans"],
                    current["batches"],
                    basis,
                    current["prompt_count"],
                    pair_bases=pair_bases,
                )
            target_index = int(pair.target_layer)
            spec = compose._qwen_layer_spec(target_layers[target_index])
            baseline_snapshot = _snapshot_pair(compose, spec)
            local_report = compose._fit_ordered_child_pair(
                torch,
                target,
                pair,
                donor_traces["pairs"][target_index],
                current["batches"],
                current["prompt_count"],
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
            attempts = []
            accepted_alpha = None
            accepted_state = None
            if bool(local_report["pair_validation"]["accepted"]):
                fitted_snapshot = _snapshot_pair(compose, spec)
                for alpha in alphas:
                    _blend_pair(
                        compose,
                        tracker,
                        spec,
                        baseline_snapshot,
                        fitted_snapshot,
                        alpha,
                        f"rollout-pair-{target_index}-alpha-{alpha:g}",
                    )
                    candidate_state = _rollout_state(
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
                    decision = _promotion_decision(
                        current["score"],
                        candidate_state["score"],
                        min_top1_gain=min_top1_gain,
                        min_action_logprob_gain=min_action_logprob_gain,
                        max_repetition_regression=max_repetition_regression,
                        max_longest_run_regression=max_longest_run_regression,
                        min_action_count_ratio=min_action_count_ratio,
                    )
                    attempts.append({"alpha": alpha, "decision": decision})
                    if bool(decision["accepted"]):
                        accepted_alpha = alpha
                        accepted_state = candidate_state
                        break
                if accepted_state is None:
                    _restore_pair(
                        compose,
                        tracker,
                        spec,
                        baseline_snapshot,
                        f"rollout-global-gate-revert-pair-{target_index}",
                    )
                else:
                    current = accepted_state
                    donor_traces = None
            pair_reports.append(
                {
                    "target_layer": target_index,
                    "donor_layers": [int(item) for item in pair.donor_layers],
                    "local_fit": local_report,
                    "candidate_alphas": list(alphas),
                    "attempts": attempts,
                    "accepted": accepted_state is not None,
                    "accepted_alpha": accepted_alpha,
                    "reverted": accepted_state is None,
                }
            )

        if donor_traces is None:
            donor_traces = compose._trace_donor_pair_targets(
                torch,
                donor,
                pair_plan,
                structural["pair_plans"],
                current["batches"],
                basis,
                current["prompt_count"],
                pair_bases=pair_bases,
            )
        final_baseline = _snapshot_final(compose, target)
        final_local = _fit_final_interface(
            torch,
            compose,
            donor,
            target,
            donor_traces,
            structural,
            current,
            train_indices,
            validation_indices,
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
        final_fitted = _snapshot_final(compose, target)
        final_attempts = []
        final_accepted_alpha = None
        final_accepted_state = None
        for alpha in alphas:
            _blend_final(
                compose,
                tracker,
                target,
                final_baseline,
                final_fitted,
                alpha,
                f"rollout-final-alpha-{alpha:g}",
            )
            candidate_state = _rollout_state(
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
            decision = _promotion_decision(
                current["score"],
                candidate_state["score"],
                min_top1_gain=min_top1_gain,
                min_action_logprob_gain=min_action_logprob_gain,
                max_repetition_regression=max_repetition_regression,
                max_longest_run_regression=max_longest_run_regression,
                min_action_count_ratio=min_action_count_ratio,
            )
            final_attempts.append({"alpha": alpha, "decision": decision})
            if bool(decision["accepted"]):
                final_accepted_alpha = alpha
                final_accepted_state = candidate_state
                break
        if final_accepted_state is None:
            _restore_final(
                compose,
                tracker,
                target,
                final_baseline,
                "rollout-global-gate-revert-final-interface",
            )
        else:
            current = final_accepted_state
        round_reports.append(
            {
                "round": round_index,
                "prompt_split": split,
                "pairs": pair_reports,
                "accepted_pairs": sum(bool(item["accepted"]) for item in pair_reports),
                "final_interface": {
                    "local_fit": final_local,
                    "attempts": final_attempts,
                    "accepted": final_accepted_state is not None,
                    "accepted_alpha": final_accepted_alpha,
                    "reverted": final_accepted_state is None,
                },
                "ending_rollout_score": current["score"],
                "ending_rollout_provenance": current["rollout"]["provenance"],
            }
        )

    for parameter in target.parameters():
        parameter.grad = None
    report = {
        "schema": REPORT_SCHEMA,
        "enabled": True,
        "trace_distribution": "current-child-greedy-rollout",
        "rollout_refresh_scope": "after-each-globally-accepted-pair",
        "donor_prefix_identity": "exact-token-id",
        "zero_gradient": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "token_contract": token_contract,
        "settings": {
            "rounds": int(rounds),
            "max_prompts": int(max_prompts),
            "max_new_tokens": int(max_new_tokens),
            "candidate_alphas": list(alphas),
            "batch_size": int(rollout_batch_size),
            "top_k": int(top_k),
            "min_top1_gain": float(min_top1_gain),
            "min_action_logprob_gain": float(min_action_logprob_gain),
            "max_repetition_regression": float(max_repetition_regression),
            "max_longest_run_regression": int(max_longest_run_regression),
            "min_action_count_ratio": float(min_action_count_ratio),
        },
        "initial_rollout_score": initial_score,
        "rounds": round_reports,
        "final_rollout_score": current["score"],
        "final_rollout_provenance": current["rollout"]["provenance"],
    }
    report["report_sha256"] = _json_sha256(report)
    return report
