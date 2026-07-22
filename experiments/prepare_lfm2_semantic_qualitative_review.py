#!/usr/bin/env python3
"""Authenticate LFM2 response dossiers and prepare qualitative review files.

This utility is deliberately standard-library-only.  It does not load a model,
run inference, calculate an intelligence score, or turn numeric diagnostics into
an acceptance decision.  It authenticates exactly three complete response
dossiers, merges their review evidence, renders a compact side-by-side Markdown
view, and writes a sealed *blank* qualitative-verdict template for a human or
agent reviewer to complete.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DOSSIER_SCHEMA = "neural-forge-lfm2-semantic-frontier-manual-review-v1"
MERGED_SCHEMA = "neural-forge-lfm2-semantic-qualitative-review-merge-v1"
VERDICT_TEMPLATE_SCHEMA = (
    "neural-forge-lfm2-semantic-qualitative-verdict-template-v1"
)
PROMPT_STATUSES = ("successful", "partial", "failed", "unsafe")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_LANE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class Lfm2SemanticQualitativeReviewError(RuntimeError):
    """The qualitative review preparation transaction failed closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticQualitativeReviewError(
            "Review evidence is not canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _DIGEST.fullmatch(normalized):
        raise Lfm2SemanticQualitativeReviewError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return normalized


def _ordinary_file(value: Path, *, label: str) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2SemanticQualitativeReviewError(f"{label} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticQualitativeReviewError(f"{label} is missing.") from exc
    if resolved != lexical or not resolved.is_file():
        raise Lfm2SemanticQualitativeReviewError(
            f"{label} must be one canonical real file."
        )
    return resolved


def _new_file(value: Path, *, label: str) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        raise Lfm2SemanticQualitativeReviewError(
            f"Refusing to replace existing {label}: {lexical}."
        )
    try:
        parent = lexical.parent.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticQualitativeReviewError(
            f"{label} parent is missing."
        ) from exc
    if parent != lexical.parent or not parent.is_dir() or parent.is_symlink():
        raise Lfm2SemanticQualitativeReviewError(
            f"{label} parent must be one canonical real directory."
        )
    return lexical


def _write_new_bytes(path: Path, payload: bytes) -> None:
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
        if path.exists() or path.is_symlink():
            raise Lfm2SemanticQualitativeReviewError(
                f"Refusing to replace an output that appeared concurrently: {path}."
            )
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticQualitativeReviewError(f"{label} must be an object.")
    return dict(value)


def _text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise Lfm2SemanticQualitativeReviewError(f"{label} must be {qualifier}.")
    return value


def _load_authenticated_dossier(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_dossier_sha256: str,
) -> tuple[dict[str, Any], str]:
    source = _ordinary_file(path, label="response dossier")
    expected_file = _digest(
        expected_file_sha256,
        label="expected dossier file SHA-256",
    )
    actual_file = _file_sha256(source)
    if actual_file != expected_file:
        raise Lfm2SemanticQualitativeReviewError(
            f"Response dossier bytes changed: {source}."
        )
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticQualitativeReviewError(
            f"Response dossier is not valid UTF-8 JSON: {source}."
        ) from exc
    dossier = _json_object(document, label="response dossier")
    body = dict(dossier)
    supplied_sha256 = body.pop("dossier_sha256", None)
    expected_dossier = _digest(
        expected_dossier_sha256,
        label="expected canonical dossier SHA-256",
    )
    if (
        dossier.get("schema") != DOSSIER_SCHEMA
        or dossier.get("status") != "ready-for-manual-intelligence-review"
        or supplied_sha256 != expected_dossier
        or supplied_sha256 != _canonical_sha256(body)
    ):
        raise Lfm2SemanticQualitativeReviewError(
            f"Response dossier schema, status, or canonical seal is invalid: {source}."
        )

    policy = _json_object(dossier.get("evaluation_policy"), label="evaluation policy")
    required_policy = {
        "decoded_whole_responses_are_authoritative": True,
        "diagnostic_numbers_may_guide_navigation": True,
        "diagnostic_numbers_have_decision_authority": False,
        "automatic_acceptance_or_rejection_performed": False,
        "exact_parent_token_match_required": False,
        "manual_intelligence_judgment_required": True,
        "parent_comparison_uses_cached_text_only": True,
    }
    if any(policy.get(key) is not expected for key, expected in required_policy.items()):
        raise Lfm2SemanticQualitativeReviewError(
            f"Response dossier evaluation policy is incompatible: {source}."
        )
    operation_audit = _json_object(
        dossier.get("operation_audit"),
        label="operation audit",
    )
    for key in ("training_steps", "optimizer_steps", "backward_calls"):
        if operation_audit.get(key) != 0:
            raise Lfm2SemanticQualitativeReviewError(
                f"Response dossier reports a forbidden operation: {key}."
            )
    if operation_audit.get("parent_forward_passes") != 0:
        raise Lfm2SemanticQualitativeReviewError(
            "Response dossier reran the parent instead of using cached text."
        )

    prompt_count = dossier.get("prompt_count")
    prompt_set_sha256 = dossier.get("prompt_set_sha256")
    requested = dossier.get("requested_candidate_ids")
    candidates = dossier.get("candidate_reviews")
    if (
        not isinstance(prompt_count, int)
        or isinstance(prompt_count, bool)
        or prompt_count <= 0
        or prompt_set_sha256
        != _digest(prompt_set_sha256, label="prompt set SHA-256")
        or not isinstance(requested, list)
        or not requested
        or len(requested) != len(set(requested))
        or any(not isinstance(value, str) or not value for value in requested)
        or not isinstance(candidates, list)
        or len(candidates) != len(requested)
        or dossier.get("reviewed_candidate_count") != len(candidates)
        or [row.get("candidate_id") for row in candidates] != requested
    ):
        raise Lfm2SemanticQualitativeReviewError(
            f"Response dossier candidate or prompt inventory is invalid: {source}."
        )
    _digest(dossier.get("frontier_run_sha256"), label="frontier run SHA-256")
    parent = _json_object(
        dossier.get("parent_response_envelope"),
        label="parent response envelope",
    )
    for key in (
        "envelope_sha256",
        "parent_state_sha256",
        "parent_identity_sha256",
        "tokenizer_sha256",
        "response_bank_sha256",
    ):
        _digest(parent.get(key), label=f"parent envelope {key}")

    expected_prompt_ids: list[str] | None = None
    expected_review_contexts: list[dict[str, Any]] | None = None
    for candidate in candidates:
        candidate_row = _json_object(candidate, label="candidate review")
        for key in ("candidate_state_sha256", "model_state_sha256"):
            _digest(candidate_row.get(key), label=f"candidate {key}")
        _text(candidate_row.get("candidate_id"), label="candidate ID")
        _text(candidate_row.get("checkpoint_path"), label="checkpoint path")
        unique_parameters = candidate_row.get("unique_parameters")
        if (
            not isinstance(unique_parameters, int)
            or isinstance(unique_parameters, bool)
            or unique_parameters <= 0
        ):
            raise Lfm2SemanticQualitativeReviewError(
                "Candidate unique parameter count must be a positive integer."
            )
        responses = candidate_row.get("responses")
        if not isinstance(responses, list) or len(responses) != prompt_count:
            raise Lfm2SemanticQualitativeReviewError(
                "Every candidate must contain the complete response prompt set."
            )
        prompt_ids = []
        review_contexts = []
        for response in responses:
            response_row = _json_object(response, label="candidate response")
            prompt_id = _text(response_row.get("id"), label="response prompt ID")
            prompt_ids.append(prompt_id)
            _text(
                response_row.get("response"),
                label="decoded child response",
                allow_empty=True,
            )
            _text(
                response_row.get("parent_response"),
                label="cached parent response",
                allow_empty=True,
            )
            _json_object(response_row.get("diagnostics"), label="child diagnostics")
            _json_object(
                response_row.get("parent_diagnostics"),
                label="parent diagnostics",
            )
            review_contexts.append(
                {
                    "prompt_id": prompt_id,
                    "category": _text(
                        response_row.get("category"),
                        label="prompt category",
                    ),
                    **_prompt_input(response_row),
                    "context": response_row.get("context"),
                    "parent_response": response_row["parent_response"],
                    "parent_response_token_ids": response_row.get(
                        "parent_response_token_ids"
                    ),
                    "parent_diagnostics": dict(response_row["parent_diagnostics"]),
                }
            )
        if len(prompt_ids) != len(set(prompt_ids)):
            raise Lfm2SemanticQualitativeReviewError(
                "Candidate response prompt IDs are duplicated."
            )
        if expected_prompt_ids is None:
            expected_prompt_ids = prompt_ids
            expected_review_contexts = review_contexts
        elif prompt_ids != expected_prompt_ids:
            raise Lfm2SemanticQualitativeReviewError(
                "Candidate response prompt order changed within a dossier."
            )
        elif review_contexts != expected_review_contexts:
            raise Lfm2SemanticQualitativeReviewError(
                "Candidate prompt or cached-parent review context changed within "
                "a dossier."
            )
    return dossier, actual_file


def _prompt_input(response: Mapping[str, Any]) -> dict[str, Any]:
    prompt = response.get("prompt")
    messages = response.get("messages")
    if isinstance(prompt, str) and messages is None:
        return {"prompt": prompt}
    if prompt is None and isinstance(messages, list) and messages:
        normalized = []
        for message in messages:
            row = _json_object(message, label="prompt message")
            if row.get("role") not in {"system", "user", "assistant"}:
                raise Lfm2SemanticQualitativeReviewError(
                    "Prompt message role is invalid."
                )
            normalized.append(
                {
                    "role": str(row["role"]),
                    "content": _text(row.get("content"), label="prompt content"),
                }
            )
        return {"messages": normalized}
    raise Lfm2SemanticQualitativeReviewError(
        "Every response must retain exactly one prompt representation."
    )


def _prompt_catalog(dossier: Mapping[str, Any]) -> list[dict[str, Any]]:
    first_candidate = dossier["candidate_reviews"][0]
    catalog = []
    for response in first_candidate["responses"]:
        catalog.append(
            {
                "prompt_id": response["id"],
                "category": _text(response.get("category"), label="prompt category"),
                **_prompt_input(response),
                "context": response.get("context"),
                "parent_response": response["parent_response"],
                "parent_diagnostics": dict(response["parent_diagnostics"]),
            }
        )
    return catalog


def _parent_binding(dossier: Mapping[str, Any]) -> dict[str, Any]:
    parent = dossier["parent_response_envelope"]
    keys = (
        "envelope_sha256",
        "parent_state_sha256",
        "parent_identity_sha256",
        "parent_unique_parameters",
        "tokenizer_sha256",
        "maximum_new_tokens",
        "response_bank_sha256",
    )
    return {key: parent[key] for key in keys}


def _response_evidence(response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": response["id"],
        "response": response["response"],
        "diagnostics": dict(response["diagnostics"]),
    }


def _diagnostic_summary(value: Mapping[str, Any]) -> str:
    ordered_keys = (
        "tokens",
        "characters",
        "empty",
        "fragment_like",
        "ended_with_eos",
        "generation_limit_reached",
        "unique_token_ratio",
        "maximum_repeated_bigram_share",
        "maximum_repeated_trigram_share",
    )
    parts = [f"{key}={value[key]}" for key in ordered_keys if key in value]
    return ", ".join(parts) if parts else "no navigation diagnostics"


def _prompt_markdown(prompt: Mapping[str, Any]) -> str:
    if "prompt" in prompt:
        return str(prompt["prompt"])
    return "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in prompt["messages"]
    )


def render_markdown(merged: Mapping[str, Any]) -> str:
    """Render authenticated parent/child text without deriving a verdict."""

    prompt_map = {row["prompt_id"]: row for row in merged["prompts"]}
    lines = [
        "# LFM2 folded-child qualitative review",
        "",
        f"Merged evidence SHA-256: `{merged['merged_sha256']}`",
        "",
        (
            "Decoded whole responses are authoritative. Numeric diagnostics are "
            "shown only to help navigate the evidence; they never accept, reject, "
            "promote, rank, or estimate intelligence retention."
        ),
        "",
    ]
    for candidate in merged["candidates"]:
        lines.extend(
            [
                f"## {candidate['candidate_key']}",
                "",
                f"Checkpoint: `{candidate['checkpoint_path']}`  ",
                f"Model state: `{candidate['model_state_sha256']}`  ",
                f"Dense unique parameters: `{candidate['unique_parameters']}`",
                "",
            ]
        )
        for response in candidate["responses"]:
            prompt = prompt_map[response["prompt_id"]]
            context = "" if prompt.get("context") is None else str(prompt["context"])
            lines.extend(
                [
                    f"### {prompt['prompt_id']} ({prompt['category']})",
                    "",
                    "Prompt:",
                    "",
                    f"<pre>{html.escape(_prompt_markdown(prompt))}</pre>",
                    "",
                    "Manual review context:",
                    "",
                    f"> {context}" if context else "> No additional context supplied.",
                    "",
                    "<table>",
                    "<thead><tr><th>Cached parent response</th>"
                    "<th>Folded child response</th></tr></thead>",
                    "<tbody><tr>",
                    f"<td><pre>{html.escape(str(prompt['parent_response']))}</pre></td>",
                    f"<td><pre>{html.escape(str(response['response']))}</pre></td>",
                    "</tr></tbody>",
                    "</table>",
                    "",
                    (
                        "Navigation diagnostics only: "
                        f"{_diagnostic_summary(response['diagnostics'])}"
                    ),
                    "",
                    (
                        "Manual status: choose `successful`, `partial`, `failed`, "
                        "or `unsafe` in the bound verdict template."
                    ),
                    "",
                    "Manual reason: explain the semantic judgment in free text.",
                    "",
                ]
            )
        lines.extend(
            [
                "Candidate conclusion (manual):",
                "",
                "Candidate conclusion reason (manual):",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-candidate conclusion",
            "",
            "Selected candidate keys (manual):",
            "",
            "Conclusion (manual):",
            "",
            "Reason (manual):",
            "",
        ]
    )
    return "\n".join(lines)


def _verdict_template(
    merged: Mapping[str, Any],
    *,
    markdown_sha256: str,
) -> dict[str, Any]:
    candidate_reviews = []
    prompt_ids = [row["prompt_id"] for row in merged["prompts"]]
    for candidate in merged["candidates"]:
        candidate_reviews.append(
            {
                "candidate_key": candidate["candidate_key"],
                "source_dossier_sha256": candidate["source_dossier_sha256"],
                "candidate_id": candidate["candidate_id"],
                "prompt_reviews": [
                    {"prompt_id": prompt_id, "status": None, "reason": ""}
                    for prompt_id in prompt_ids
                ],
                "candidate_conclusion": "",
                "candidate_conclusion_reason": "",
            }
        )
    body = {
        "schema": VERDICT_TEMPLATE_SCHEMA,
        "status": "blank-manual-qualitative-review-template",
        "source_merged_evidence_sha256": merged["merged_sha256"],
        "source_dossiers": list(merged["source_dossiers"]),
        "rendered_markdown_sha256": markdown_sha256,
        "allowed_prompt_statuses": list(PROMPT_STATUSES),
        "instructions": {
            "every_prompt_status_must_be_selected_manually": True,
            "every_prompt_reason_must_be_nonempty_free_text": True,
            "candidate_conclusions_are_manual_free_text": True,
            "cross_candidate_conclusion_is_manual": True,
            "exact_parent_wording_is_not_a_criterion": True,
            "numeric_diagnostics_may_guide_navigation": True,
            "numeric_diagnostics_have_decision_authority": False,
            "no_formula_computes_acceptance_rejection_promotion_retention_or_verdict": (
                True
            ),
        },
        "candidate_reviews": candidate_reviews,
        "cross_candidate_conclusion": {
            "selected_candidate_keys": [],
            "conclusion": "",
            "reason": "",
        },
        "completion_requirements": {
            "replace_every_null_status_with_one_allowed_prompt_status": True,
            "write_a_concrete_reason_for_every_prompt": True,
            "write_candidate_and_cross_candidate_conclusions_manually": True,
            "automatic_completion_or_decision_performed": False,
        },
    }
    return {**body, "template_sha256": _canonical_sha256(body)}


def prepare_qualitative_review(
    *,
    dossier_paths: Sequence[Path],
    expected_dossier_file_sha256s: Sequence[str],
    expected_dossier_sha256s: Sequence[str],
    lane_labels: Sequence[str],
    merged_json: Path,
    markdown_path: Path,
    verdict_template_json: Path,
) -> dict[str, Any]:
    """Authenticate three dossiers and publish review-only derived artifacts."""

    cardinalities = {
        len(dossier_paths),
        len(expected_dossier_file_sha256s),
        len(expected_dossier_sha256s),
        len(lane_labels),
    }
    if cardinalities != {3}:
        raise Lfm2SemanticQualitativeReviewError(
            "Exactly three dossier paths, file pins, canonical pins, and lane "
            "labels are required."
        )
    normalized_labels = tuple(str(value).strip() for value in lane_labels)
    if (
        len(set(normalized_labels)) != 3
        or any(not _LANE_LABEL.fullmatch(value) for value in normalized_labels)
    ):
        raise Lfm2SemanticQualitativeReviewError(
            "Lane labels must be unique stable identifiers."
        )
    outputs = (
        _new_file(merged_json, label="merged review JSON"),
        _new_file(markdown_path, label="review Markdown"),
        _new_file(verdict_template_json, label="verdict template JSON"),
    )
    if len(set(outputs)) != len(outputs):
        raise Lfm2SemanticQualitativeReviewError("Review output paths must be unique.")

    loaded = []
    for path, expected_file, expected_dossier in zip(
        dossier_paths,
        expected_dossier_file_sha256s,
        expected_dossier_sha256s,
        strict=True,
    ):
        dossier, file_sha256 = _load_authenticated_dossier(
            path,
            expected_file_sha256=expected_file,
            expected_dossier_sha256=expected_dossier,
        )
        loaded.append((path.expanduser().absolute(), dossier, file_sha256))

    reference_prompt_set = loaded[0][1]["prompt_set_sha256"]
    reference_prompt_count = loaded[0][1]["prompt_count"]
    reference_prompts = _prompt_catalog(loaded[0][1])
    reference_parent = _parent_binding(loaded[0][1])
    reference_new_tokens = loaded[0][1]["maximum_new_tokens"]
    if any(
        dossier["prompt_set_sha256"] != reference_prompt_set
        or dossier["prompt_count"] != reference_prompt_count
        or dossier["maximum_new_tokens"] != reference_new_tokens
        or _prompt_catalog(dossier) != reference_prompts
        or _parent_binding(dossier) != reference_parent
        for _, dossier, _ in loaded[1:]
    ):
        raise Lfm2SemanticQualitativeReviewError(
            "Dossiers do not share the exact prompt set, cached parent evidence, "
            "and generation horizon."
        )

    source_dossiers = []
    merged_candidates = []
    candidate_keys = set()
    for lane_label, (path, dossier, file_sha256) in zip(
        normalized_labels,
        loaded,
        strict=True,
    ):
        source_dossiers.append(
            {
                "lane_label": lane_label,
                "path": str(path),
                "file_sha256": file_sha256,
                "dossier_sha256": dossier["dossier_sha256"],
                "frontier_run_sha256": dossier["frontier_run_sha256"],
                "candidate_count": dossier["reviewed_candidate_count"],
            }
        )
        for candidate in dossier["candidate_reviews"]:
            candidate_key = f"{lane_label}::{candidate['candidate_id']}"
            if candidate_key in candidate_keys:
                raise Lfm2SemanticQualitativeReviewError(
                    f"Merged candidate key is duplicated: {candidate_key}."
                )
            candidate_keys.add(candidate_key)
            merged_candidates.append(
                {
                    "candidate_key": candidate_key,
                    "lane_label": lane_label,
                    "source_dossier_sha256": dossier["dossier_sha256"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_state_sha256": candidate["candidate_state_sha256"],
                    "model_state_sha256": candidate["model_state_sha256"],
                    "checkpoint_path": candidate["checkpoint_path"],
                    "unique_parameters": candidate["unique_parameters"],
                    "responses": [
                        _response_evidence(response)
                        for response in candidate["responses"]
                    ],
                }
            )

    body = {
        "schema": MERGED_SCHEMA,
        "status": "ready-for-manual-qualitative-review",
        "source_dossiers": source_dossiers,
        "source_dossier_count": len(source_dossiers),
        "prompt_set_sha256": reference_prompt_set,
        "prompt_count": reference_prompt_count,
        "maximum_new_tokens": reference_new_tokens,
        "parent_response_envelope": reference_parent,
        "prompts": reference_prompts,
        "candidate_count": len(merged_candidates),
        "candidates": merged_candidates,
        "evaluation_policy": {
            "decoded_whole_responses_are_authoritative": True,
            "exact_parent_text_or_token_match_required": False,
            "numeric_diagnostics_may_guide_navigation": True,
            "numeric_diagnostics_have_decision_authority": False,
            "automatic_acceptance_rejection_promotion_or_retention_computed": False,
            "candidate_conclusions_are_manual": True,
            "cross_candidate_conclusion_is_manual": True,
        },
    }
    merged = {**body, "merged_sha256": _canonical_sha256(body)}
    markdown = render_markdown(merged)
    markdown_bytes = markdown.encode("utf-8")
    template = _verdict_template(
        merged,
        markdown_sha256=hashlib.sha256(markdown_bytes).hexdigest(),
    )
    merged_bytes = _canonical_json_bytes(merged) + b"\n"
    template_bytes = _canonical_json_bytes(template) + b"\n"
    _write_new_bytes(outputs[0], merged_bytes)
    _write_new_bytes(outputs[1], markdown_bytes)
    _write_new_bytes(outputs[2], template_bytes)
    return {
        "status": "qualitative-review-artifacts-prepared",
        "merged_json": str(outputs[0]),
        "merged_file_sha256": hashlib.sha256(merged_bytes).hexdigest(),
        "merged_sha256": merged["merged_sha256"],
        "markdown": str(outputs[1]),
        "markdown_file_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
        "verdict_template_json": str(outputs[2]),
        "verdict_template_file_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "verdict_template_sha256": template["template_sha256"],
        "source_dossier_count": len(source_dossiers),
        "candidate_count": len(merged_candidates),
        "prompt_count": reference_prompt_count,
        "automatic_decision_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate and merge exactly three LFM2 whole-response dossiers, "
            "render side-by-side Markdown, and emit a sealed blank qualitative "
            "verdict template without numeric gating."
        )
    )
    parser.add_argument("--dossier", action="append", required=True, type=Path)
    parser.add_argument(
        "--expected-dossier-file-sha256",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--expected-dossier-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--lane-label", action="append", required=True)
    parser.add_argument("--merged-json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--verdict-template-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare_qualitative_review(
        dossier_paths=args.dossier,
        expected_dossier_file_sha256s=args.expected_dossier_file_sha256,
        expected_dossier_sha256s=args.expected_dossier_sha256,
        lane_labels=args.lane_label,
        merged_json=args.merged_json,
        markdown_path=args.markdown,
        verdict_template_json=args.verdict_template_json,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
