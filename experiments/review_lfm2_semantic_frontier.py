#!/usr/bin/env python3
"""Generate long-form response dossiers for standalone LFM2 frontier children.

The output deliberately contains no automatic acceptance or rejection field.
Heuristics such as repetition, empty output, and generation-limit saturation are
diagnostics for navigating the candidate bank.  Complete decoded responses are
the primary behavioral evidence.  Optional human or agent observations describe
coherence, competence, regressions, and failure modes across dimensions; neither
one scalar nor one binary manual verdict controls continuation or retention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments import run_lfm2_residual_attention_fold as residual  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)


DOSSIER_SCHEMA = "neural-forge-lfm2-semantic-frontier-manual-review-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-semantic-frontier-review-progress-v1"
PROMPTS_SCHEMA = "neural-forge-whole-response-review-prompts-v1"
PARENT_ENVELOPE_SCHEMA = "neural-forge-lfm2-whole-response-parent-envelope-v1"


class Lfm2SemanticFrontierReviewError(RuntimeError):
    """A standalone frontier response dossier could not be produced."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_inventory(checkpoint: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise Lfm2SemanticFrontierReviewError(
                "Candidate checkpoint inventory contains a symlink."
            )
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(checkpoint).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _file_sha256(path),
                }
            )
    if not rows:
        raise Lfm2SemanticFrontierReviewError(
            "Candidate checkpoint inventory is empty."
        )
    return rows


def _validate_materialization_checkpoint_inventory(
    materialization: Mapping[str, Any],
    checkpoint: Path,
) -> list[dict[str, Any]] | None:
    expected_inventory = materialization.get("checkpoint_inventory")
    expected_sha256 = materialization.get("checkpoint_inventory_sha256")
    if expected_inventory is None and expected_sha256 is None:
        # Legacy frontier materializations predate full checkpoint inventories.
        return None
    if not isinstance(expected_inventory, list) or expected_sha256 is None:
        raise Lfm2SemanticFrontierReviewError(
            "Checkpoint inventory materialization fields must be supplied together."
        )
    expected_digest = _digest(
        str(expected_sha256),
        label="materialized checkpoint inventory SHA-256",
    )
    if canonical_sha256(expected_inventory) != expected_digest:
        raise Lfm2SemanticFrontierReviewError(
            "Materialized checkpoint inventory does not match its SHA-256."
        )
    actual = _checkpoint_inventory(checkpoint)
    if actual != expected_inventory or canonical_sha256(actual) != expected_digest:
        raise Lfm2SemanticFrontierReviewError(
            "Candidate checkpoint files differ from the materialized inventory."
        )
    return actual


def _digest(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Lfm2SemanticFrontierReviewError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return normalized


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _new_file(path: Path, *, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        raise Lfm2SemanticFrontierReviewError(
            f"Refusing to replace existing {label}: {lexical}."
        )
    parent = lexical.parent.resolve(strict=True)
    if parent != lexical.parent or not parent.is_dir() or parent.is_symlink():
        raise Lfm2SemanticFrontierReviewError(
            f"{label} parent must be one canonical real directory."
        )
    return lexical


def _load_frontier_run(
    path: Path,
    *,
    expected_run_sha256: str,
    expected_file_sha256: str | None,
) -> dict[str, Any]:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise Lfm2SemanticFrontierReviewError(
            "Frontier run receipt must be one real local file."
        )
    if expected_file_sha256 is not None and _file_sha256(lexical) != _digest(
        expected_file_sha256,
        label="expected frontier run file SHA-256",
    ):
        raise Lfm2SemanticFrontierReviewError("Frontier run receipt bytes changed.")
    document = json.loads(lexical.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise Lfm2SemanticFrontierReviewError(
            "Frontier run receipt must be a JSON object."
        )
    body = dict(document)
    supplied = body.pop("run_sha256", None)
    if (
        document.get("schema") != "neural-forge-lfm2-semantic-terminal-frontier-run-v1"
        or document.get("status") != "complete"
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_run_sha256, label="expected frontier run SHA-256")
        or document.get("all_retained_candidates_materialized") is not True
        or document.get("candidate_metric_threshold_used_for_materialization")
        is not False
    ):
        raise Lfm2SemanticFrontierReviewError(
            "Frontier run receipt is incomplete or differs from its external pin."
        )
    candidate_ids = document.get("candidate_ids")
    materializations = document.get("materializations")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
        or not isinstance(materializations, list)
        or len(materializations) != len(candidate_ids)
        or [row.get("candidate_id") for row in materializations] != candidate_ids
    ):
        raise Lfm2SemanticFrontierReviewError(
            "Frontier run candidate inventory is incomplete."
        )
    return document


def _load_parent_envelope(
    path: Path,
    *,
    expected_envelope_sha256: str,
    expected_file_sha256: str | None,
    expected_prompt_set_sha256: str,
    expected_prompt_ids: Sequence[str],
) -> dict[str, Any]:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise Lfm2SemanticFrontierReviewError(
            "Parent response envelope must be one real local file."
        )
    if expected_file_sha256 is not None and _file_sha256(lexical) != _digest(
        expected_file_sha256,
        label="expected parent envelope file SHA-256",
    ):
        raise Lfm2SemanticFrontierReviewError("Parent response envelope bytes changed.")
    document = json.loads(lexical.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise Lfm2SemanticFrontierReviewError(
            "Parent response envelope must be a JSON object."
        )
    body = dict(document)
    supplied = body.pop("envelope_sha256", None)
    responses = document.get("responses")
    if (
        document.get("schema") != PARENT_ENVELOPE_SCHEMA
        or document.get("status") != "complete-one-time-parent-response-cache"
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(
            expected_envelope_sha256,
            label="expected parent envelope SHA-256",
        )
        or document.get("prompt_set_sha256") != expected_prompt_set_sha256
        or not isinstance(responses, list)
        or [row.get("id") for row in responses] != list(expected_prompt_ids)
        or document.get("prompt_count") != len(expected_prompt_ids)
        or document.get("response_bank_sha256") != canonical_sha256(responses)
    ):
        raise Lfm2SemanticFrontierReviewError(
            "Parent response envelope is incomplete or differs from its pins."
        )
    return document


def _normalize_messages(value: Any, *, prompt_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise Lfm2SemanticFrontierReviewError(
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
            raise Lfm2SemanticFrontierReviewError(
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
        raise Lfm2SemanticFrontierReviewError(
            "Whole-response prompt file must contain a nonempty prompt list."
        )
    normalized = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Lfm2SemanticFrontierReviewError("Every prompt must be an object.")
        prompt_id = str(row.get("id", f"review-{index:04d}")).strip()
        if not prompt_id or prompt_id in seen:
            raise Lfm2SemanticFrontierReviewError(
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
            raise Lfm2SemanticFrontierReviewError(
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
            raise Lfm2SemanticFrontierReviewError(
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
        raise Lfm2SemanticFrontierReviewError("Tokenizer returned invalid input IDs.")
    return input_ids.to(dtype=torch.long), attention_mask


def _response_diagnostics(
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
    return {
        "characters": len(text),
        "tokens": len(tokens),
        "empty": not bool(text.strip()),
        "fragment_like": len(text.strip()) < 12,
        "unique_token_ratio": 1.0 if not tokens else len(set(tokens)) / len(tokens),
        "maximum_repeated_bigram_share": maximum_share(bigrams),
        "maximum_repeated_trigram_share": maximum_share(trigrams),
        "ended_with_eos": ended_with_eos,
        "generation_limit_reached": (
            len(tokens) >= maximum_new_tokens and not ended_with_eos
        ),
        "interpretation": "diagnostic-only; decoded response controls review",
    }


def _generate_rows(
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
                    raise Lfm2SemanticFrontierReviewError(
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
                        "diagnostics": _response_diagnostics(
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


def _review_one_child(
    materialization: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    *,
    device: str,
    dtype: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required for response review.") from exc
    checkpoint = Path(str(materialization["checkpoint_path"])).absolute()
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise Lfm2SemanticFrontierReviewError(
            f"Candidate checkpoint is missing: {checkpoint}."
        )
    checkpoint_inventory = _validate_materialization_checkpoint_inventory(
        materialization,
        checkpoint,
    )
    expected_lfm2_identity = materialization.get("lfm2_identity_sha256")
    expected_lfm2_config = materialization.get("lfm2_config_sha256")
    if (expected_lfm2_identity is None) != (expected_lfm2_config is None):
        raise Lfm2SemanticFrontierReviewError(
            "LFM2 identity and config materialization fields must be supplied together."
        )
    if expected_lfm2_identity is not None:
        expected_lfm2_identity = _digest(
            str(expected_lfm2_identity),
            label="materialized LFM2 identity SHA-256",
        )
        expected_lfm2_config = _digest(
            str(expected_lfm2_config),
            label="materialized LFM2 config SHA-256",
        )
    torch_dtype = residual._torch_dtype(dtype)
    tokenizer = residual.load_local_tokenizer(checkpoint)
    model = residual.load_local_lfm2_model(
        checkpoint,
        device=torch.device(device),
        dtype=torch_dtype,
    )
    try:
        # Materialization receipts use the canonical dynamic-fold state
        # fingerprint.  The residual-attention runner's similarly named helper
        # hashes a different document and cannot authenticate this receipt.
        state_sha256 = fingerprint_model_state(model)
        if state_sha256 != materialization.get("model_state_sha256"):
            raise Lfm2SemanticFrontierReviewError(
                "Freshly loaded candidate differs from its materialization receipt."
            )
        lfm2_before = (
            None if expected_lfm2_identity is None else fingerprint_lfm2_parent(model)
        )
        if lfm2_before is not None and (
            lfm2_before.identity_sha256 != expected_lfm2_identity
            or lfm2_before.config_sha256 != expected_lfm2_config
        ):
            raise Lfm2SemanticFrontierReviewError(
                "Freshly loaded candidate LFM2 identity/config differs from its "
                "materialization receipt."
            )
        rows = _generate_rows(
            model,
            tokenizer,
            prompts,
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=maximum_new_tokens,
        )
        if checkpoint_inventory is not None:
            _validate_materialization_checkpoint_inventory(
                materialization,
                checkpoint,
            )
        if lfm2_before is not None:
            lfm2_after = fingerprint_lfm2_parent(model)
            if lfm2_after.as_dict() != lfm2_before.as_dict():
                raise Lfm2SemanticFrontierReviewError(
                    "Candidate LFM2 identity/config changed during generation."
                )
        result = {
            "candidate_id": materialization["candidate_id"],
            "candidate_state_sha256": materialization["candidate_state_sha256"],
            "model_state_sha256": state_sha256,
            "checkpoint_path": str(checkpoint),
            "unique_parameters": residual.unique_parameter_count(model),
            "responses": rows,
        }
        if checkpoint_inventory is not None:
            result.update(
                {
                    "checkpoint_inventory": checkpoint_inventory,
                    "checkpoint_inventory_sha256": materialization[
                        "checkpoint_inventory_sha256"
                    ],
                }
            )
        if lfm2_before is not None:
            result.update(
                {
                    "lfm2_identity_sha256": lfm2_before.identity_sha256,
                    "lfm2_config_sha256": lfm2_before.config_sha256,
                }
            )
        return result
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def review_frontier(
    *,
    frontier_run_receipt: Path,
    expected_frontier_run_sha256: str,
    expected_frontier_run_file_sha256: str | None,
    prompts_json: Path,
    output_json: Path,
    candidate_ids: Sequence[str],
    device: str,
    dtype: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    parent_envelope: Path | None = None,
    expected_parent_envelope_sha256: str | None = None,
    expected_parent_envelope_file_sha256: str | None = None,
) -> dict[str, Any]:
    if not 1 <= maximum_prompt_tokens <= 8192:
        raise Lfm2SemanticFrontierReviewError(
            "maximum_prompt_tokens must be between 1 and 8192."
        )
    if not 16 <= maximum_new_tokens <= 2048:
        raise Lfm2SemanticFrontierReviewError(
            "maximum_new_tokens must be between 16 and 2048."
        )
    output = _new_file(output_json, label="response dossier")
    progress = _new_file(
        output.with_name(f"{output.stem}.progress.json"),
        label="response dossier progress",
    )
    run = _load_frontier_run(
        frontier_run_receipt,
        expected_run_sha256=expected_frontier_run_sha256,
        expected_file_sha256=expected_frontier_run_file_sha256,
    )
    prompts = load_review_prompts(prompts_json)
    prompt_rows = [dict(row) for row in prompts]
    prompt_sha256 = canonical_sha256(prompt_rows)
    if (parent_envelope is None) != (expected_parent_envelope_sha256 is None):
        raise Lfm2SemanticFrontierReviewError(
            "Parent envelope path and expected envelope SHA-256 must be paired."
        )
    parent = (
        None
        if parent_envelope is None
        else _load_parent_envelope(
            parent_envelope,
            expected_envelope_sha256=str(expected_parent_envelope_sha256),
            expected_file_sha256=expected_parent_envelope_file_sha256,
            expected_prompt_set_sha256=prompt_sha256,
            expected_prompt_ids=[row["id"] for row in prompts],
        )
    )
    parent_responses = (
        {} if parent is None else {str(row["id"]): row for row in parent["responses"]}
    )
    materializations = {
        str(row["candidate_id"]): row for row in run["materializations"]
    }
    selected = tuple(candidate_ids) if candidate_ids else tuple(run["candidate_ids"])
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(candidate_id not in materializations for candidate_id in selected)
    ):
        raise Lfm2SemanticFrontierReviewError(
            "Requested review candidates are missing or duplicated."
        )
    completed = []
    for candidate_id in selected:
        child = _review_one_child(
            materializations[candidate_id],
            prompts,
            device=device,
            dtype=dtype,
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=maximum_new_tokens,
        )
        if parent is not None:
            for response in child["responses"]:
                parent_response = parent_responses[str(response["id"])]
                response["parent_response"] = parent_response["response"]
                response["parent_response_token_ids"] = parent_response[
                    "response_token_ids"
                ]
                response["parent_diagnostics"] = parent_response["diagnostics"]
        completed.append(child)
        progress_body = {
            "schema": PROGRESS_SCHEMA,
            "status": "running" if len(completed) < len(selected) else "complete",
            "frontier_run_sha256": run["run_sha256"],
            "prompt_set_sha256": prompt_sha256,
            "requested_candidate_ids": list(selected),
            "completed_candidate_ids": [row["candidate_id"] for row in completed],
            "remaining_candidate_ids": list(selected[len(completed) :]),
            "maximum_prompt_tokens": maximum_prompt_tokens,
            "maximum_new_tokens": maximum_new_tokens,
            "automatic_acceptance_or_rejection_performed": False,
            "complete_decoded_responses_preserved": True,
            "multidimensional_observations_optional": True,
            "binary_manual_verdict_required": False,
        }
        _atomic_json(
            progress,
            {**progress_body, "progress_sha256": canonical_sha256(progress_body)},
        )
    body = {
        "schema": DOSSIER_SCHEMA,
        "status": "decoded-whole-response-evidence-complete",
        "frontier_run_receipt": str(frontier_run_receipt.absolute()),
        "frontier_run_sha256": run["run_sha256"],
        "prompt_source": str(prompts_json.absolute()),
        "prompt_set_schema": PROMPTS_SCHEMA,
        "prompt_set_sha256": prompt_sha256,
        "prompt_count": len(prompts),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "parent_response_envelope": (
            None
            if parent is None
            else {
                "path": str(parent_envelope.absolute()),
                "envelope_sha256": parent["envelope_sha256"],
                "parent_state_sha256": parent["parent_state_sha256"],
                "parent_identity_sha256": parent["parent_identity_sha256"],
                "parent_unique_parameters": parent["parent_unique_parameters"],
                "tokenizer_sha256": parent["tokenizer_sha256"],
                "maximum_new_tokens": parent["maximum_new_tokens"],
                "inference_device": parent.get("inference_device"),
                "inference_dtype": parent.get("inference_dtype"),
                "response_bank_sha256": parent["response_bank_sha256"],
            }
        ),
        "requested_candidate_ids": list(selected),
        "reviewed_candidate_count": len(completed),
        "candidate_reviews": completed,
        "evaluation_policy": {
            "decoded_whole_responses_are_primary_evidence": True,
            "complete_decoded_responses_preserved": True,
            "diagnostic_numbers_may_guide_navigation": True,
            "diagnostic_numbers_have_decision_authority": False,
            "automatic_acceptance_or_rejection_performed": False,
            "exact_parent_token_match_required": False,
            "multidimensional_behavior_observations_encouraged": True,
            "manual_observations_required_for_continuation": False,
            "single_binary_manual_verdict_has_decision_authority": False,
            "retention_target_is_objective_not_cutoff": True,
            "parent_comparison_uses_cached_text_only": parent is not None,
        },
        "operation_audit": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_forward_passes": 0,
        },
        "progress_json": str(progress),
    }
    document = {**body, "dossier_sha256": canonical_sha256(body)}
    _atomic_json(output, document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate long-form outputs from standalone folded children for "
            "holistic behavioral review; diagnostics and binary verdicts never "
            "veto a child."
        )
    )
    parser.add_argument("--frontier-run-receipt", required=True, type=Path)
    parser.add_argument("--expected-frontier-run-sha256", required=True)
    parser.add_argument("--expected-frontier-run-file-sha256")
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Review this retained candidate; omitted reviews the complete bank.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--parent-envelope", type=Path)
    parser.add_argument("--expected-parent-envelope-sha256")
    parser.add_argument("--expected-parent-envelope-file-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = review_frontier(
        frontier_run_receipt=args.frontier_run_receipt,
        expected_frontier_run_sha256=args.expected_frontier_run_sha256,
        expected_frontier_run_file_sha256=(args.expected_frontier_run_file_sha256),
        prompts_json=args.prompts_json,
        output_json=args.output_json,
        candidate_ids=args.candidate_id,
        device=args.device,
        dtype=args.dtype,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
        maximum_new_tokens=args.maximum_new_tokens,
        parent_envelope=args.parent_envelope,
        expected_parent_envelope_sha256=args.expected_parent_envelope_sha256,
        expected_parent_envelope_file_sha256=(
            args.expected_parent_envelope_file_sha256
        ),
    )
    print(
        json.dumps(
            {
                "status": document["status"],
                "reviewed_candidate_count": document["reviewed_candidate_count"],
                "prompt_count": document["prompt_count"],
                "dossier_sha256": document["dossier_sha256"],
                "output": str(args.output_json),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
