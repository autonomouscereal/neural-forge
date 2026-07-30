"""Shared LFM2 decoded-response collection without intelligence gating."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class Lfm2DecodedReviewError(RuntimeError):
    """A prompt or decoded-response collection invariant failed."""


def response_termination_evidence(
    *,
    ended_with_eos: bool,
    generation_limit_reached: bool,
    loop_observed: bool | None = None,
    runtime_error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe how an observed response ended without grading its quality.

    A generation limit is an observation boundary, not evidence that folding
    failed.  Looping is likewise kept as a separate observation and is never
    inferred merely because a response consumed its token allowance.  Runtime
    errors remain distinct from both behavioral quality and censoring.
    """

    for value, name in (
        (ended_with_eos, "ended_with_eos"),
        (generation_limit_reached, "generation_limit_reached"),
    ):
        if not isinstance(value, bool):
            raise Lfm2DecodedReviewError(f"{name} must be boolean.")
    if loop_observed is not None and not isinstance(loop_observed, bool):
        raise Lfm2DecodedReviewError("loop_observed must be boolean or None.")
    if runtime_error is not None and not isinstance(runtime_error, Mapping):
        raise Lfm2DecodedReviewError("runtime_error must be a mapping or None.")
    if ended_with_eos and generation_limit_reached:
        raise Lfm2DecodedReviewError(
            "A response cannot both end with EOS and hit the generation limit."
        )
    if runtime_error is not None:
        finish_reason = "error"
        completion_status = "runtime-error"
    elif generation_limit_reached:
        finish_reason = "length"
        completion_status = "censored"
    elif ended_with_eos:
        finish_reason = "eos"
        completion_status = "complete"
    else:
        finish_reason = "stop"
        completion_status = "complete"
    return {
        "finish_reason": finish_reason,
        "completion_status": completion_status,
        "ended_with_eos": ended_with_eos,
        "generation_limit_reached": generation_limit_reached,
        "output_cap_hit": generation_limit_reached,
        "censored_by_output_cap": generation_limit_reached,
        "output_cap_hit_is_automatic_folding_failure": False,
        "loop_observed": loop_observed,
        "loop_observation_source": (
            "not-reviewed" if loop_observed is None else "manual-review"
        ),
        "runtime_error": (
            None if runtime_error is None else dict(runtime_error)
        ),
        "automatic_folding_quality_decision": None,
    }


def _normalize_messages(value: Any, *, prompt_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise Lfm2DecodedReviewError(
            f"Prompt {prompt_id!r} messages must be a nonempty list."
        )
    messages = []
    for message in value:
        if (
            not isinstance(message, Mapping)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise Lfm2DecodedReviewError(
                f"Prompt {prompt_id!r} contains an invalid message."
            )
        messages.append(
            {"role": str(message["role"]), "content": str(message["content"])}
        )
    return messages


def load_review_prompts(path: Path) -> tuple[dict[str, Any], ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("prompts") if isinstance(document, Mapping) else document
    if not isinstance(rows, list) or not rows:
        raise Lfm2DecodedReviewError(
            "Whole-response prompt file must contain a nonempty prompt list."
        )
    normalized = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Lfm2DecodedReviewError("Every prompt must be an object.")
        prompt_id = str(row.get("id", f"review-{index:04d}")).strip()
        if not prompt_id or prompt_id in seen:
            raise Lfm2DecodedReviewError(
                "Whole-response prompt IDs must be unique and nonempty."
            )
        seen.add(prompt_id)
        prompt = row.get("prompt")
        messages = row.get("messages")
        if isinstance(prompt, str) and prompt.strip() and messages is None:
            normalized_input: dict[str, Any] = {"prompt": prompt}
        elif prompt is None and messages is not None:
            normalized_input = {
                "messages": _normalize_messages(messages, prompt_id=prompt_id)
            }
        else:
            raise Lfm2DecodedReviewError(
                f"Prompt {prompt_id!r} must supply exactly one of prompt or messages."
            )
        normalized.append(
            {
                "id": prompt_id,
                **normalized_input,
                "category": str(row.get("category", "general")),
                "context": row.get("context"),
                "parent_response": row.get("parent_response"),
            }
        )
    return tuple(normalized)


def _prompt_tensor(tokenizer: Any, row: Mapping[str, Any], torch: Any) -> Any:
    if "messages" in row:
        template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(template):
            raise Lfm2DecodedReviewError(
                "Tokenizer cannot render chat-message review prompts."
            )
        rendered = template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(rendered, Mapping):
            input_ids = rendered.get("input_ids")
            attention_mask = rendered.get("attention_mask")
        else:
            input_ids = rendered
            attention_mask = None
    else:
        rendered = tokenizer(
            row["prompt"],
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = rendered["input_ids"]
        attention_mask = rendered.get("attention_mask")
    if input_ids is None or getattr(input_ids, "ndim", None) != 2:
        raise Lfm2DecodedReviewError("Tokenizer returned invalid input IDs.")
    return input_ids.to(dtype=torch.long), attention_mask


def response_diagnostics(
    token_ids: Sequence[int],
    text: str,
    *,
    maximum_new_tokens: int,
    eos_token_ids: frozenset[int],
) -> dict[str, Any]:
    tokens = tuple(int(value) for value in token_ids)
    bigrams = tuple(zip(tokens, tokens[1:]))
    trigrams = tuple(zip(tokens, tokens[1:], tokens[2:]))

    def maximum_share(values: Sequence[tuple[int, ...]]) -> float:
        if not values:
            return 0.0
        counts: dict[tuple[int, ...], int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return max(counts.values()) / len(values)

    ended_with_eos = bool(tokens and tokens[-1] in eos_token_ids)
    termination = response_termination_evidence(
        ended_with_eos=ended_with_eos,
        generation_limit_reached=(
            len(tokens) >= maximum_new_tokens and not ended_with_eos
        ),
    )
    return {
        "characters": len(text),
        "tokens": len(tokens),
        "empty": not bool(text.strip()),
        "fragment_like": len(text.strip()) < 12,
        "unique_token_ratio": 1.0 if not tokens else len(set(tokens)) / len(tokens),
        "maximum_repeated_bigram_share": maximum_share(bigrams),
        "maximum_repeated_trigram_share": maximum_share(trigrams),
        **termination,
        "interpretation": "diagnostic-only; decoded response controls review",
    }


def generate_review_rows(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required for response review.") from exc
    device = next(model.parameters()).device
    raw_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(raw_eos, int):
        eos_token_ids = frozenset({raw_eos})
    elif isinstance(raw_eos, (list, tuple)):
        eos_token_ids = frozenset(int(value) for value in raw_eos)
    else:
        eos_token_ids = frozenset()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    rows = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for prompt in prompts:
                input_ids, attention_mask = _prompt_tensor(tokenizer, prompt, torch)
                if int(input_ids.shape[1]) > maximum_prompt_tokens:
                    raise Lfm2DecodedReviewError(
                        f"Prompt {prompt['id']!r} exceeds maximum_prompt_tokens."
                    )
                input_ids = input_ids.to(device=device, dtype=torch.long)
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "max_new_tokens": maximum_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask.to(
                        device=device,
                        dtype=torch.long,
                    )
                if pad_token_id is not None:
                    kwargs["pad_token_id"] = int(pad_token_id)
                generated = model.generate(**kwargs)
                continuation = generated[:, int(input_ids.shape[1]) :]
                token_ids = [
                    int(value) for value in continuation[0].detach().cpu().tolist()
                ]
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                rows.append(
                    {
                        **dict(prompt),
                        "response": text,
                        "response_token_ids": token_ids,
                        "diagnostics": response_diagnostics(
                            token_ids,
                            text,
                            maximum_new_tokens=maximum_new_tokens,
                            eos_token_ids=eos_token_ids,
                        ),
                    }
                )
    finally:
        model.train(was_training)
    return rows


__all__ = [
    "Lfm2DecodedReviewError",
    "generate_review_rows",
    "load_review_prompts",
    "response_diagnostics",
    "response_termination_evidence",
]
