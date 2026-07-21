#!/usr/bin/env python3
"""Build and score the exact-50% cached-causal LFM2 value fold.

The parent is loaded locally only to pack its immutable state.  Candidate
allocation consumes an already sealed cached-causal frontier; it performs no
parent forward or response generation.  The independent 64-case parent
responses are loaded from their existing sealed envelope and the freshly
reloaded candidate is the only model that generates during scoring.

Open-ended cases use paired parent/candidate ratings.  Abilities absent in the
parent are neutralized and every scalar score remains diagnostic-only.  The
reviewer's sealed qualitative case verdicts and holistic suite conclusion are
the only intelligence judgment allowed to authorize promotion.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from experiments import evaluate_lfm2_intelligence_holdout as holdout_eval
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
    from experiments import run_lfm2_residual_attention_fold as direct_lfm2
    from experiments import run_lfm2_value_fold_control as control_runner
except ImportError:  # Direct execution from the experiments directory.
    import evaluate_lfm2_intelligence_holdout as holdout_eval
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_residual_attention_fold as direct_lfm2
    import run_lfm2_value_fold_control as control_runner

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_causal_value_allocator import (
    export_lfm2_causal_value_fold,
    inspect_lfm2_causal_value_fold,
    load_lfm2_causal_value_fold,
    load_lfm2_causal_value_tokenizer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_intelligence_holdout import (
    baseline_from_lfm2_intelligence_holdout_envelope,
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_lfm2_parent_relative_manual_gate import (
    MINIMUM_RETENTION,
    PACKET_SCHEMA as PARENT_RELATIVE_PACKET_SCHEMA,
    PROMOTION_ELIGIBLE_SUITE_CONCLUSION,
    QUALITATIVE_CRITERIA,
    build_all_case_parent_relative_manual_packet,
    evaluate_parent_relative_manual_retention,
    normalize_parent_relative_completion_metadata,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LEGACY_RUNNER_SCHEMA = "neural-forge-lfm2-causal-value-allocator-run-v1"
RUNNER_SCHEMA = "neural-forge-lfm2-causal-value-allocator-run-v2"
LEGACY_TARGET_REPORT_SCHEMA = "neural-forge-lfm2-causal-value-target-report-v1"
TARGET_REPORT_SCHEMA = "neural-forge-lfm2-causal-value-target-report-v2"
EXPECTED_HOLDOUT_CASE_COUNT = 64


class Lfm2CausalValueRunnerError(RuntimeError):
    """A local identity, sealed input, or parent-relative gate failed."""


def _build_exact_all_case_parent_relative_packet(
    standard_packet: Mapping[str, Any],
    envelope: Any,
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind whole parent/child responses for every authenticated holdout case."""

    manifest = envelope.manifest
    manifest_cases = tuple(manifest.cases)
    baseline = baseline_from_lfm2_intelligence_holdout_envelope(envelope)
    parent_cases = tuple(baseline.cases)
    rows = tuple(candidate_rows)
    if (
        len(manifest_cases) != EXPECTED_HOLDOUT_CASE_COUNT
        or len(parent_cases) != EXPECTED_HOLDOUT_CASE_COUNT
        or len(rows) != EXPECTED_HOLDOUT_CASE_COUNT
    ):
        raise Lfm2CausalValueRunnerError(
            "Parent-relative promotion requires the exact 64-case holdout."
        )
    manifest_ids = tuple(str(value.case_id) for value in manifest_cases)
    parent_by_id = {str(value.case_id): value for value in parent_cases}
    candidate_by_id = {
        str(value.get("case_id")): value for value in rows if isinstance(value, Mapping)
    }
    if (
        len(set(manifest_ids)) != EXPECTED_HOLDOUT_CASE_COUNT
        or set(parent_by_id) != set(manifest_ids)
        or set(candidate_by_id) != set(manifest_ids)
    ):
        raise Lfm2CausalValueRunnerError(
            "Parent-relative review identities differ from the exact 64-case holdout."
        )
    eos_token_ids = frozenset(int(value) for value in envelope.eos_token_ids)
    all_case_rows = []
    for spec in manifest_cases:
        case_id = str(spec.case_id)
        parent = parent_by_id[case_id]
        candidate = candidate_by_id[case_id]
        if (
            parent.prompt != spec.prompt
            or candidate.get("generation_limit") != spec.generation_limit
            or not isinstance(candidate.get("output"), str)
            or not isinstance(candidate.get("output_token_ids"), list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in candidate.get("output_token_ids", ())
            )
            or len(candidate.get("output_token_ids", ())) > spec.generation_limit
        ):
            raise Lfm2CausalValueRunnerError(
                f"Parent-relative review evidence differs for {case_id}."
            )
        candidate_token_ids = tuple(candidate["output_token_ids"])
        candidate_eos_observed = bool(
            candidate_token_ids and candidate_token_ids[-1] in eos_token_ids
        )
        all_case_rows.append(
            {
                "case_id": case_id,
                "prompt": spec.prompt,
                "parent_response": parent.parent_output,
                "candidate_response": candidate["output"],
                "generation_limit": spec.generation_limit,
                "parent_completion": {
                    "generated_token_count": parent.generation.generated_token_count,
                    "eos_observed": parent.generation.eos_observed,
                    "unfinished_at_generation_limit": (
                        parent.generation.unfinished_at_generation_limit
                    ),
                },
                "candidate_completion": {
                    "generated_token_count": len(candidate_token_ids),
                    "eos_observed": candidate_eos_observed,
                    "unfinished_at_generation_limit": (
                        len(candidate_token_ids) == spec.generation_limit
                        and not candidate_eos_observed
                    ),
                },
            }
        )
    return build_all_case_parent_relative_manual_packet(
        standard_packet,
        all_case_rows,
        manifest_sha256=manifest.manifest_sha256,
        manifest_document_sha256=manifest.document_sha256,
    )


def _exact_all_case_review_contract(
    intelligence: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
) -> bool:
    """Return whether one packet is the sealed exact-manifest all-case review."""

    if not isinstance(packet, Mapping):
        return False
    body = dict(packet)
    packet_sha256 = body.pop("packet_sha256", None)
    inputs = intelligence.get("inputs")
    cases = body.get("cases")
    rubric = body.get("rubric")
    if (
        not isinstance(packet_sha256, str)
        or packet_sha256 != canonical_sha256(body)
        or body.get("schema") != PARENT_RELATIVE_PACKET_SCHEMA
        or body.get("review_scope") != "exact_authenticated_manifest_all_cases"
        or body.get("all_case_manual_review_mode") is not True
        or body.get("case_count") != EXPECTED_HOLDOUT_CASE_COUNT
        or not isinstance(cases, list)
        or len(cases) != EXPECTED_HOLDOUT_CASE_COUNT
        or not isinstance(inputs, Mapping)
        or inputs.get("case_count") != EXPECTED_HOLDOUT_CASE_COUNT
        or body.get("manifest_sha256") != inputs.get("manifest_sha256")
        or body.get("manifest_document_sha256")
        != inputs.get("manifest_document_sha256")
        or not isinstance(rubric, Mapping)
        or rubric.get("review_every_manifest_case") is not True
        or rubric.get("full_parent_and_candidate_response_review_required") is not True
        or rubric.get("per_case_generation_limit_bound") is not True
        or rubric.get("qualitative_case_verdicts_are_authoritative") is not True
        or rubric.get("holistic_suite_conclusion_is_authoritative") is not True
        or rubric.get("automatic_accept_reject_permitted") is not False
        or body.get("legacy_generic_manual_subset_is_diagnostic_only") is not True
    ):
        return False
    case_ids = [
        str(value.get("case_id")) for value in cases if isinstance(value, Mapping)
    ]
    limit_rows = [
        {
            "case_id": str(value.get("case_id")),
            "generation_limit": value.get("generation_limit"),
        }
        for value in cases
        if isinstance(value, Mapping)
    ]
    completion_rows = [
        {
            "case_id": str(value.get("case_id")),
            "parent_completion": value.get("parent_completion"),
            "candidate_completion": value.get("candidate_completion"),
        }
        for value in cases
        if isinstance(value, Mapping)
    ]
    completion_metadata_valid = len(completion_rows) == EXPECTED_HOLDOUT_CASE_COUNT
    if completion_metadata_valid:
        try:
            for case, completion in zip(cases, completion_rows):
                limit = case.get("generation_limit")
                if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                    completion_metadata_valid = False
                    break
                normalize_parent_relative_completion_metadata(
                    completion["parent_completion"],
                    generation_limit=limit,
                    name=f"{completion['case_id']}.parent",
                )
                normalize_parent_relative_completion_metadata(
                    completion["candidate_completion"],
                    generation_limit=limit,
                    name=f"{completion['case_id']}.candidate",
                )
        except (TypeError, ValueError):
            completion_metadata_valid = False
    return (
        len(case_ids) == EXPECTED_HOLDOUT_CASE_COUNT
        and len(set(case_ids)) == EXPECTED_HOLDOUT_CASE_COUNT
        and len(limit_rows) == EXPECTED_HOLDOUT_CASE_COUNT
        and completion_metadata_valid
        and all(
            isinstance(value["generation_limit"], int)
            and not isinstance(value["generation_limit"], bool)
            and value["generation_limit"] > 0
            for value in limit_rows
        )
        and body.get("case_ids_sha256") == canonical_sha256(case_ids)
        and body.get("generation_limit_rows_sha256") == canonical_sha256(limit_rows)
        and body.get("completion_metadata_rows_sha256")
        == canonical_sha256(completion_rows)
        and rubric.get("qualitative_criteria") == list(QUALITATIVE_CRITERIA)
        and rubric.get("parent_and_candidate_completion_metadata_bound") is True
        and rubric.get("completion_metadata_is_diagnostic_context_only") is True
        and rubric.get("completion_metadata_automatic_accept_reject_permitted") is False
        and rubric.get("explicit_truncation_consideration_required") is True
        and rubric.get("criterion_specific_observations_required") is True
        and rubric.get("duplicate_boilerplate_observations_permitted") is False
    )


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CausalValueRunnerError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _authenticate_intelligence_report(document: Mapping[str, Any]) -> str:
    """Recompute the cached holdout report seal and return its bound digest."""

    if not isinstance(document, Mapping):
        raise Lfm2CausalValueRunnerError(
            "Intelligence report must contain a JSON object."
        )
    body = dict(document)
    report_sha256 = body.pop("report_sha256", None)
    if body.get("schema") != holdout_eval.RUNNER_SCHEMA or _digest(
        report_sha256, "intelligence_report_sha256"
    ) != canonical_sha256(body):
        raise Lfm2CausalValueRunnerError(
            "Intelligence report schema or content seal is invalid."
        )
    return report_sha256


def _read_json(
    path: Path,
    *,
    expected_file_sha256: str,
    name: str,
) -> dict[str, Any]:
    if control_runner._file_sha256(path) != _digest(
        expected_file_sha256,
        f"{name}_file_sha256",
    ):
        raise Lfm2CausalValueRunnerError(f"{name} file SHA-256 changed.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2CausalValueRunnerError(f"{name} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise Lfm2CausalValueRunnerError(f"{name} must contain an object.")
    return value


def _probe(
    checkpoint: Path,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
) -> tuple[torch.Tensor, str]:
    model = load_lfm2_causal_value_fold(checkpoint, device=device)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    runtime_device = control_runner._runtime_device(model)
    input_ids = encoded["input_ids"].to(runtime_device, dtype=torch.long)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(runtime_device, dtype=torch.long)
    with torch.inference_mode():
        logits = (
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            .logits.detach()
            .to("cpu")
        )
    manifest_sha = model.neural_forge_load_report["manifest_sha256"]
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits, manifest_sha


def _parent_relative_decision(
    intelligence: Mapping[str, Any],
    manual_report: Mapping[str, Any] | None,
    manual_packet: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gate = intelligence.get("core_intelligence_gate")
    policy = intelligence.get("policy")
    if not isinstance(gate, Mapping) or not isinstance(policy, Mapping):
        raise Lfm2CausalValueRunnerError(
            "Automatic intelligence report lacks its sealed gate/policy."
        )
    components = gate.get("components")
    required = policy.get("required_components")
    if (
        not isinstance(components, list)
        or not isinstance(required, list)
        or not required
    ):
        raise Lfm2CausalValueRunnerError(
            "Automatic intelligence component coverage is invalid."
        )
    by_name = {
        str(value.get("name")): value
        for value in components
        if isinstance(value, Mapping)
    }
    if not set(required).issubset(by_name):
        raise Lfm2CausalValueRunnerError(
            "Automatic intelligence report omits a required component."
        )
    headline_values = {}
    for name in required:
        component = by_name[name]
        retention = component.get("retention")
        if (
            not isinstance(retention, bool)
            and isinstance(retention, (int, float))
            and 0.0 <= float(retention) <= 1.0
        ):
            headline_values[name] = float(retention)
    task_absolute = intelligence.get("parent_relative_task_retention", {}).get(
        "absolute_accuracy_retention"
    )
    if isinstance(task_absolute, (int, float)) and not isinstance(task_absolute, bool):
        headline_values["task_absolute_accuracy_retention"] = float(task_absolute)
    exact_all_case_review = _exact_all_case_review_contract(
        intelligence,
        manual_packet,
    )
    manual_pending = manual_report is None
    holistic_conclusion = None
    if manual_report is not None:
        retention = manual_report.get("retention")
        if (
            not isinstance(retention, bool)
            and isinstance(retention, (int, float))
            and 0.0 <= float(retention) <= 1.0
        ):
            headline_values["parent_relative_manual"] = float(retention)
        holistic = manual_report.get("holistic_review")
        suite = (
            holistic.get("suite_conclusion") if isinstance(holistic, Mapping) else None
        )
        if (
            exact_all_case_review
            and isinstance(holistic, Mapping)
            and holistic.get("status") == "complete"
            and holistic.get("qualitative_judgment_is_authoritative") is True
            and holistic.get("numeric_scores_and_thresholds_are_diagnostic_only")
            is True
            and holistic.get("automatic_scalar_accept_reject_permitted") is False
            and isinstance(suite, Mapping)
            and manual_report.get("packet_sha256")
            == (
                manual_packet.get("packet_sha256")
                if isinstance(manual_packet, Mapping)
                else None
            )
            and isinstance(manual_report.get("case_reports"), list)
            and len(manual_report["case_reports"]) == EXPECTED_HOLDOUT_CASE_COUNT
            and isinstance(holistic.get("qualitative_case_reviews"), list)
            and len(holistic["qualitative_case_reviews"]) == EXPECTED_HOLDOUT_CASE_COUNT
            and tuple(
                str(value.get("case_id"))
                for value in holistic["qualitative_case_reviews"]
                if isinstance(value, Mapping)
            )
            == tuple(
                str(value.get("case_id"))
                for value in manual_packet["cases"]
                if isinstance(value, Mapping)
            )
        ):
            holistic_conclusion = suite.get("conclusion")
    reasons = []
    if not exact_all_case_review:
        reasons.append("authoritative_holistic_review:exact_64_case_packet_required")
    elif manual_pending:
        reasons.append("authoritative_holistic_review:pending")
    elif holistic_conclusion is None:
        reasons.append("authoritative_holistic_review:missing")
    elif holistic_conclusion != PROMOTION_ELIGIBLE_SUITE_CONCLUSION:
        reasons.append(f"authoritative_holistic_review:{holistic_conclusion}")
    promoted = (
        exact_all_case_review
        and holistic_conclusion == PROMOTION_ELIGIBLE_SUITE_CONCLUSION
    )
    observed_diagnostic_floor = (
        min(headline_values.values()) if headline_values else None
    )
    return {
        "promoted": promoted,
        "status": (
            "promoted"
            if promoted
            else (
                "promotion_ineligible_incomplete_review_scope"
                if not exact_all_case_review
                else (
                    "pending_parent_relative_manual"
                    if manual_pending
                    else "holistic_review_complete_not_promoted"
                )
            )
        ),
        "reasons": reasons,
        "certified_parent_relative_lower_bound": None,
        "observed_parent_relative_diagnostic_floor": observed_diagnostic_floor,
        "headline_inputs": headline_values,
        "automatic_components_passed": all(
            by_name[name].get("passed") is True for name in required
        ),
        "automatic_component_pass_flags_are_diagnostic_only": True,
        "parent_relative_manual_status": (
            "ineligible_review_scope"
            if not exact_all_case_review
            else ("pending" if manual_pending else "holistic_review_complete")
        ),
        "exact_64_case_manual_review_contract_authenticated": exact_all_case_review,
        "holistic_suite_conclusion": holistic_conclusion,
        "required_holistic_suite_conclusion": (PROMOTION_ELIGIBLE_SUITE_CONCLUSION),
        "minimum_component_retention_diagnostic_reference": MINIMUM_RETENTION,
        "numeric_scores_and_thresholds_are_diagnostic_only": True,
        "absolute_manual_scores_are_diagnostics_only": True,
        "abilities_absent_in_parent_are_neutralized": True,
        "qualitative_judgment_is_authoritative": True,
    }


@dataclass(frozen=True, slots=True)
class CausalValueRunConfig:
    parent_model_directory: Path
    tokenizer_directory: Path
    output_root: Path
    cached_causal_frontier: Path
    expected_cached_causal_frontier_file_sha256: str
    parent_envelope: Path
    expected_parent_envelope_file_sha256: str
    expected_parent_envelope_sha256: str
    expected_manifest_sha256: str
    expected_parent_state_fingerprint: str
    expected_parent_model_fingerprint: str
    expected_tokenizer_fingerprint: str
    calibration_token_selection: Path | None = None
    expected_calibration_token_selection_file_sha256: str | None = None
    parent_relative_manual_ratings: Path | None = None
    expected_parent_relative_manual_ratings_file_sha256: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    group_size: int = 256
    analysis_chunk_groups: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_model_directory",
            control_runner._local_directory(
                self.parent_model_directory,
                "parent_model_directory",
            ),
        )
        object.__setattr__(
            self,
            "tokenizer_directory",
            control_runner._local_directory(
                self.tokenizer_directory,
                "tokenizer_directory",
            ),
        )
        for field in ("cached_causal_frontier", "parent_envelope"):
            object.__setattr__(
                self,
                field,
                control_runner._local_file(getattr(self, field), field),
            )
        output = Path(self.output_root).expanduser().absolute()
        if output.exists():
            raise Lfm2CausalValueRunnerError(
                f"Refusing to replace existing output root {output}."
            )
        object.__setattr__(self, "output_root", output)
        for field in (
            "expected_cached_causal_frontier_file_sha256",
            "expected_parent_envelope_file_sha256",
            "expected_parent_envelope_sha256",
            "expected_manifest_sha256",
            "expected_parent_state_fingerprint",
            "expected_parent_model_fingerprint",
            "expected_tokenizer_fingerprint",
        ):
            _digest(getattr(self, field), field)
        for path_field, digest_field in (
            (
                "calibration_token_selection",
                "expected_calibration_token_selection_file_sha256",
            ),
            (
                "parent_relative_manual_ratings",
                "expected_parent_relative_manual_ratings_file_sha256",
            ),
        ):
            path = getattr(self, path_field)
            digest = getattr(self, digest_field)
            if (path is None) != (digest is None):
                raise Lfm2CausalValueRunnerError(
                    f"{path_field} and {digest_field} must be supplied together."
                )
            if path is not None:
                object.__setattr__(
                    self,
                    path_field,
                    control_runner._local_file(path, path_field),
                )
                _digest(digest, digest_field)
        if self.dtype not in {"float16", "bfloat16"}:
            raise Lfm2CausalValueRunnerError("dtype must be float16 or bfloat16.")


def run_lfm2_causal_value_allocator(
    config: CausalValueRunConfig,
) -> dict[str, Any]:
    if not isinstance(config, CausalValueRunConfig):
        raise TypeError("config must be CausalValueRunConfig.")
    frontier_document = _read_json(
        config.cached_causal_frontier,
        expected_file_sha256=(config.expected_cached_causal_frontier_file_sha256),
        name="cached_causal_frontier",
    )
    parent_document = _read_json(
        config.parent_envelope,
        expected_file_sha256=config.expected_parent_envelope_file_sha256,
        name="parent_envelope",
    )
    envelope = load_lfm2_intelligence_holdout_parent_envelope(
        parent_document,
        expected_envelope_sha256=config.expected_parent_envelope_sha256,
        expected_manifest_sha256=config.expected_manifest_sha256,
        expected_parent_state_fingerprint=(config.expected_parent_state_fingerprint),
        expected_parent_model_fingerprint=(config.expected_parent_model_fingerprint),
        expected_tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
    )
    tokenizer_fingerprint, _files = posthoc.fingerprint_bound_local_tokenizer(
        config.tokenizer_directory
    )
    if tokenizer_fingerprint != config.expected_tokenizer_fingerprint:
        raise Lfm2CausalValueRunnerError(
            "Local tokenizer differs from the sealed parent tokenizer."
        )
    special = posthoc.load_bound_special_token_ids(
        config.tokenizer_directory,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    eos_token_ids = tuple(special["special_token_ids"]["eos"])
    if eos_token_ids != envelope.eos_token_ids:
        raise Lfm2CausalValueRunnerError(
            "Tokenizer EOS IDs differ from the parent envelope."
        )
    calibration_document = None
    if config.calibration_token_selection is not None:
        calibration_document = _read_json(
            config.calibration_token_selection,
            expected_file_sha256=(
                config.expected_calibration_token_selection_file_sha256
            ),
            name="calibration_token_selection",
        )
    ratings_document = None
    if config.parent_relative_manual_ratings is not None:
        ratings_document = _read_json(
            config.parent_relative_manual_ratings,
            expected_file_sha256=(
                config.expected_parent_relative_manual_ratings_file_sha256
            ),
            name="parent_relative_manual_ratings",
        )

    config.output_root.mkdir(parents=True)
    checkpoint = config.output_root / "checkpoint"
    parent = direct_lfm2.load_local_lfm2_model(
        config.parent_model_directory,
        device=config.device,
        dtype=direct_lfm2._torch_dtype(config.dtype),
    )
    parent.eval()
    parent_state_before = direct_lfm2.model_state_sha256(parent)
    parent_identity = fingerprint_lfm2_parent(parent).identity_sha256
    if (
        parent_state_before != config.expected_parent_state_fingerprint
        or parent_identity != config.expected_parent_model_fingerprint
        or any(parameter.grad is not None for parameter in parent.parameters())
    ):
        raise Lfm2CausalValueRunnerError(
            "Loaded local parent differs from its sealed identity."
        )
    manifest = export_lfm2_causal_value_fold(
        parent,
        checkpoint,
        tokenizer_directory=config.tokenizer_directory,
        frontier_document=frontier_document,
        eos_token_ids=eos_token_ids,
        calibration_token_selection=calibration_document,
        group_size=config.group_size,
        analysis_chunk_groups=config.analysis_chunk_groups,
    )
    if direct_lfm2.model_state_sha256(parent) != parent_state_before or any(
        parameter.grad is not None for parameter in parent.parameters()
    ):
        raise Lfm2CausalValueRunnerError(
            "Packing mutated the parent state or created gradients."
        )
    del parent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = load_lfm2_causal_value_tokenizer(checkpoint)
    if (
        tokenizer.neural_forge_tokenizer_fingerprint
        != config.expected_tokenizer_fingerprint
    ):
        raise Lfm2CausalValueRunnerError(
            "Fresh checkpoint tokenizer differs from the sealed tokenizer."
        )
    probe_prompt = envelope.manifest.cases[0].prompt
    first_logits, first_manifest = _probe(
        checkpoint,
        tokenizer,
        probe_prompt,
        device=config.device,
    )
    candidate = load_lfm2_causal_value_fold(
        checkpoint,
        device=config.device,
    )
    encoded = tokenizer(
        probe_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    runtime_device = control_runner._runtime_device(candidate)
    input_ids = encoded["input_ids"].to(runtime_device, dtype=torch.long)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(runtime_device, dtype=torch.long)
    with torch.inference_mode():
        second_logits = (
            candidate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            .logits.detach()
            .to("cpu")
        )
    if first_manifest != manifest["manifest_sha256"] or not torch.equal(
        first_logits,
        second_logits,
    ):
        raise Lfm2CausalValueRunnerError(
            "Two fresh causal-value reloads are not logit-identical."
        )
    candidate_rows = control_runner._generate_candidate_rows(
        candidate,
        tokenizer,
        envelope.manifest,
    )
    candidate_document = seal_lfm2_intelligence_holdout_candidate_outputs(
        envelope.manifest,
        candidate_rows,
        candidate_state_fingerprint=manifest["manifest_sha256"],
        tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
        eos_token_ids=eos_token_ids,
    )
    candidate_path = config.output_root / "candidate_outputs.json"
    control_runner._write_new_json(candidate_path, candidate_document)
    candidate_file_sha = control_runner._file_sha256(candidate_path)
    del candidate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    intelligence_path = config.output_root / "intelligence_report.json"
    standard_packet_path = config.output_root / "manual_review_packet.json"
    intelligence = holdout_eval.evaluate_lfm2_intelligence_holdout(
        holdout_eval.HoldoutEvaluationConfig(
            parent_envelope=config.parent_envelope,
            expected_parent_envelope_file_sha256=(
                config.expected_parent_envelope_file_sha256
            ),
            expected_parent_envelope_sha256=(config.expected_parent_envelope_sha256),
            expected_manifest_sha256=config.expected_manifest_sha256,
            expected_parent_state_fingerprint=(
                config.expected_parent_state_fingerprint
            ),
            expected_parent_model_fingerprint=(
                config.expected_parent_model_fingerprint
            ),
            tokenizer_directory=checkpoint,
            expected_tokenizer_fingerprint=(config.expected_tokenizer_fingerprint),
            candidate_report=candidate_path,
            expected_candidate_report_sha256=candidate_file_sha,
            report_path=intelligence_path,
            manual_review_packet_path=standard_packet_path,
            expected_candidate_outputs_sha256=candidate_document[
                "candidate_outputs_sha256"
            ],
            expected_candidate_state_fingerprint=manifest["manifest_sha256"],
        )
    )
    standard_packet = json.loads(standard_packet_path.read_text(encoding="utf-8"))
    parent_relative_packet = _build_exact_all_case_parent_relative_packet(
        standard_packet,
        envelope,
        candidate_rows,
    )
    parent_relative_packet_path = (
        config.output_root / "parent_relative_manual_packet.json"
    )
    control_runner._write_new_json(
        parent_relative_packet_path,
        parent_relative_packet,
    )
    manual_report = None
    if ratings_document is not None:
        manual_report = evaluate_parent_relative_manual_retention(
            parent_relative_packet,
            ratings_document,
        )
        control_runner._write_new_json(
            config.output_root / "parent_relative_manual_report.json",
            manual_report,
        )
    decision = _parent_relative_decision(
        intelligence,
        manual_report,
        parent_relative_packet,
    )
    accounting = inspect_lfm2_causal_value_fold(checkpoint)
    core = {
        "schema": TARGET_REPORT_SCHEMA,
        "status": decision["status"],
        "promoted": decision["promoted"],
        "decision": decision,
        "checkpoint": accounting,
        "fresh_reload_equivalence": {
            "reload_count": 2,
            "probe_logits_bitwise_equal": True,
            "manifest_sha256": manifest["manifest_sha256"],
            "exact_source_dtype_overlays_active": True,
        },
        "holdout": {
            "suite": "sealed-independent-64-case-lfm2-holdout",
            "case_count": len(candidate_rows),
            "candidate_outputs_sha256": candidate_document["candidate_outputs_sha256"],
            "candidate_output_file_sha256": candidate_file_sha,
            "automatic_intelligence_report_sha256": intelligence["report_sha256"],
            "standard_absolute_manual_status": intelligence["manual_review"]["status"],
            "standard_absolute_manual_promotion_eligible": False,
            "parent_relative_manual_packet_sha256": parent_relative_packet[
                "packet_sha256"
            ],
            "parent_relative_manual_packet_case_count": parent_relative_packet[
                "case_count"
            ],
            "parent_relative_manual_review_scope": parent_relative_packet[
                "review_scope"
            ],
            "parent_relative_manual_report_sha256": (
                None if manual_report is None else manual_report["report_sha256"]
            ),
            "parent_model_loaded_during_scoring": False,
            "new_parent_responses": 0,
        },
        "parent_binding": {
            "parent_state_fingerprint": (config.expected_parent_state_fingerprint),
            "parent_model_fingerprint": (config.expected_parent_model_fingerprint),
            "parent_envelope_sha256": config.expected_parent_envelope_sha256,
            "cached_causal_frontier_file_sha256": (
                config.expected_cached_causal_frontier_file_sha256
            ),
            "frontier_sha256": accounting["frontier_sha256"],
            "tokenizer_fingerprint": config.expected_tokenizer_fingerprint,
            "parent_state_unchanged_by_export": True,
        },
        "execution_audit": {
            "network_access": False,
            "hub_loader_used": False,
            "training_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_forward_calls_during_allocation": 0,
            "parent_generate_invocations": 0,
            "candidate_generate_invocations": len(candidate_rows),
            "independent_holdout_used_for_allocation": False,
        },
    }
    report = {**core, "report_sha256": canonical_sha256(core)}
    control_runner._write_new_json(config.output_root / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and score the exact-50% cached-causal mixed-precision LFM2 "
            "checkpoint from local models and sealed caches only."
        )
    )
    parser.add_argument("--parent-model-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cached-causal-frontier", type=Path, required=True)
    parser.add_argument(
        "--expected-cached-causal-frontier-file-sha256",
        required=True,
    )
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-state-fingerprint", required=True)
    parser.add_argument("--expected-parent-model-fingerprint", required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--calibration-token-selection", type=Path)
    parser.add_argument(
        "--expected-calibration-token-selection-file-sha256",
    )
    parser.add_argument("--parent-relative-manual-ratings", type=Path)
    parser.add_argument(
        "--expected-parent-relative-manual-ratings-file-sha256",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--analysis-chunk-groups", type=int, default=4096)
    return parser


def config_from_args(args: argparse.Namespace) -> CausalValueRunConfig:
    return CausalValueRunConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    report = run_lfm2_causal_value_allocator(
        config_from_args(build_parser().parse_args(argv))
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "promoted": report["promoted"],
                "report_sha256": report["report_sha256"],
                "compressed_checkpoint_bytes": report["checkpoint"][
                    "compressed_checkpoint_bytes"
                ],
                "parent_relative_lower_bound": report["decision"][
                    "certified_parent_relative_lower_bound"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
