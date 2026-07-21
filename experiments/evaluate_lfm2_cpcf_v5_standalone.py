#!/usr/bin/env python3
"""Capture and finalize the retained CPCF-v5 child's 64-case intelligence test.

``capture`` performs one direct-local reload of an already-retained v5
``candidate`` directory, generates each built-in holdout case exactly once at
that case's own generation limit, and stores the decoded responses and
automatic behavioral diagnostics outside the child artifact.  It also authenticates
the dense child's unique-parameter count and records a parent/child ratio from
an explicitly pinned expected parent count, without using checkpoint bytes,
quantization, or pruning adjustments.  For a semantic-postfold child, optional
receipt arguments bind the external run receipt, nested checkpoint receipt,
artifact metadata, model state, and tokenizer semantics before generation.

``finalize`` never loads either model.  It authenticates an independently
pinned capture report, the stored candidate responses, cached parent envelope,
paired parent/child semantic-review packet, and packet-bound ratings, then
writes the final pinned-suite behavioral-retention report.

Neither mode trains, prunes, quantizes, downloads, or executes the parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
for _path in (_REPOSITORY_ROOT, _SOURCE_ROOT):
    if str(_path) not in os.sys.path:
        os.sys.path.insert(0, str(_path))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from experiments.capture_lfm2_cpcf_holdout_candidate import (  # noqa: E402
    generate_candidate_rows_once,
)
from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_artifact import (  # noqa: E402
    DynamicFoldArtifactError,
    STANDALONE_FOLD_ARTIFACT_INVENTORY_SCHEMA,
    canonical_standalone_fold_artifact_inventory,
    scan_standalone_fold_artifact_tree,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (  # noqa: E402
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_cpcf_v5_quarantine import (  # noqa: E402
    LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
    Lfm2CpcfV5StrictLocalLoader,
    _read_artifact_metadata,
)
from neural_forge.hf_lfm2_intelligence_gate import (  # noqa: E402
    IntelligenceGatePolicy,
    Lfm2IntelligenceGateError,
    bind_manual_ratings,
    build_manual_review_packet,
    evaluate_cached_intelligence,
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (  # noqa: E402
    baseline_from_lfm2_intelligence_holdout_envelope,
    load_lfm2_intelligence_holdout_candidate_outputs,
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_parent_relative_manual_gate import (  # noqa: E402
    Lfm2ParentRelativeManualGateError,
    QUALITATIVE_CRITERIA,
    build_all_case_parent_relative_manual_packet,
    evaluate_parent_relative_manual_retention,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    Lfm2SemanticPostfoldError,
    authenticate_lfm2_semantic_layer_stack_attention_receipts,
)


CAPTURE_REPORT_SCHEMA = "neural-forge-lfm2-cpcf-v5-standalone-intelligence-capture-v3"
FINAL_REPORT_SCHEMA = "neural-forge-lfm2-cpcf-v5-standalone-intelligence-final-v3"
CHILD_OUTPUT_FILENAME = "standalone-child-outputs-64.json"
AUTOMATIC_REPORT_FILENAME = "automatic-intelligence-report-64.json"
MANUAL_PACKET_FILENAME = "manual-review-packet-64.json"
FINAL_REPORT_FILENAME = "final-intelligence-report-64.json"
EXPECTED_CASE_COUNT = 64
LFM2_SEMANTIC_POSTFOLD_SCHEMA = "neural-forge-lfm2-semantic-postfold-v1"
LFM2_SEMANTIC_POSTFOLD_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-semantic-postfold-checkpoint-v1"
)
SEMANTIC_POSTFOLD_EVALUATION_PROVENANCE_SCHEMA = (
    "neural-forge-lfm2-semantic-postfold-evaluation-provenance-v1"
)
SEMANTIC_POSTFOLD_ZERO_OPERATION_AUDIT = {
    "parent_model_objects_constructed": 0,
    "parent_model_forward_calls": 0,
    "parent_model_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "child_model_forward_calls": 0,
    "child_model_generate_calls": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "training_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class Lfm2CpcfV5StandaloneEvaluationError(RuntimeError):
    """Standalone capture or finalization failed closed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfV5StandaloneEvaluationError(f"{name} must be a positive integer.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_directory(value: Path, name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} must be an explicit local directory."
        )
    unresolved = Path(value).expanduser().absolute()
    if unresolved.is_symlink():
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} is not an ordinary local directory."
        )
    path = unresolved.resolve(strict=True)
    if not path.is_dir():
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} is not an ordinary local directory."
        )
    return path


def _local_file(value: Path, name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} must be an explicit local file."
        )
    unresolved = Path(value).expanduser().absolute()
    if unresolved.is_symlink():
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} is not an ordinary local file."
        )
    path = unresolved.resolve(strict=True)
    if not path.is_file():
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} is not an ordinary local file."
        )
    return path


def _output_directory(value: Path) -> Path:
    path = Path(value).expanduser().absolute()
    if path.exists():
        if path.is_symlink():
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "output_directory must be an ordinary directory."
            )
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "output_directory must be an ordinary directory."
            )
        return resolved
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve(strict=True)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _read_canonical_json(
    path: Path,
    *,
    name: str,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_file_sha256 is not None:
        expected = _digest(
            expected_file_sha256,
            f"expected {name} file SHA-256",
        )
        if _file_sha256(path) != expected:
            raise Lfm2CpcfV5StandaloneEvaluationError(
                f"{name} differs from its independent file SHA-256."
            )
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} is not readable canonical JSON."
        ) from error
    if not isinstance(document, dict):
        raise Lfm2CpcfV5StandaloneEvaluationError(f"{name} must be a JSON object.")
    canonical = canonical_json_bytes(document)
    if raw not in (canonical, canonical + b"\n"):
        raise Lfm2CpcfV5StandaloneEvaluationError(f"{name} is not canonically encoded.")
    return document


def _sealed_report(
    document: Mapping[str, Any],
    *,
    schema: str,
    seal_field: str,
    name: str,
) -> dict[str, Any]:
    normalized = dict(document)
    claimed = normalized.pop(seal_field, None)
    if normalized.get("schema") != schema or _digest(
        claimed, f"{name} seal"
    ) != canonical_sha256(normalized):
        raise Lfm2CpcfV5StandaloneEvaluationError(f"{name} schema or seal is invalid.")
    return {**normalized, seal_field: claimed}


def _atomic_publish_documents(
    documents: Sequence[tuple[Path, Mapping[str, Any]]],
) -> None:
    destinations = [destination for destination, _document in documents]
    if len(set(destinations)) != len(destinations):
        raise Lfm2CpcfV5StandaloneEvaluationError("Output destinations must be unique.")
    if any(path.exists() or path.is_symlink() for path in destinations):
        occupied = [
            str(path) for path in destinations if path.exists() or path.is_symlink()
        ]
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"Refusing to replace existing outputs: {occupied}."
        )
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, document in documents:
            payload = canonical_json_bytes(document) + b"\n"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            staged.append((destination, temporary))
        for destination, temporary in staged:
            os.link(temporary, destination)
            published.append(destination)
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for _destination, temporary in staged:
            temporary.unlink(missing_ok=True)


def _normalized_eos_ids(tokenizer: Any) -> tuple[int, ...]:
    value = getattr(tokenizer, "eos_token_id", None)
    values = (
        list(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else [value]
    )
    normalized = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Retained target tokenizer has no valid EOS identity."
            )
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Retained target tokenizer has no EOS identity."
        )
    return tuple(normalized)


def _load_parent_envelope(
    path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[Any, Any]:
    document = _read_canonical_json(
        path,
        name="parent envelope",
        expected_file_sha256=expected_file_sha256,
    )
    manifest = load_lfm2_intelligence_holdout_manifest()
    required = (
        "envelope_sha256",
        "parent_state_fingerprint",
        "parent_model_fingerprint",
        "tokenizer_fingerprint",
    )
    if any(not isinstance(document.get(field), str) for field in required):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Parent envelope identity fields are incomplete."
        )
    envelope = load_lfm2_intelligence_holdout_parent_envelope(
        document,
        expected_envelope_sha256=document["envelope_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_parent_state_fingerprint=document["parent_state_fingerprint"],
        expected_parent_model_fingerprint=document["parent_model_fingerprint"],
        expected_tokenizer_fingerprint=document["tokenizer_fingerprint"],
    )
    if len(manifest.cases) != EXPECTED_CASE_COUNT:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "The authenticated holdout is not the exact 64-case suite."
        )
    return manifest, envelope


def _parameter_inventory(model: Any) -> dict[str, Any]:
    return dict(independent_parameter_inventory(model))


def _gradients_are_absent(model: Any) -> bool:
    return all(parameter.grad is None for parameter in model.parameters())


def _build_all_case_parent_relative_manual_packet(
    *,
    automatic_gate_manual_packet: Mapping[str, Any],
    baseline: Any,
    candidates: Sequence[Any],
    manifest: Any,
) -> dict[str, Any]:
    """Build the standalone evaluator's explicit all-case review packet."""

    manifest_cases = tuple(getattr(manifest, "cases", ()))
    parent_cases = tuple(getattr(baseline, "cases", ()))
    candidate_cases = tuple(candidates)
    if (
        len(manifest_cases) != EXPECTED_CASE_COUNT
        or len(parent_cases) != EXPECTED_CASE_COUNT
        or len(candidate_cases) != EXPECTED_CASE_COUNT
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Whole-response review requires the exact 64-case manifest, parent, "
            "and child outputs."
        )

    manifest_ids = tuple(str(value.case_id) for value in manifest_cases)
    parent_map = {str(value.case_id): value for value in parent_cases}
    candidate_map = {str(value.case_id): value for value in candidate_cases}
    if (
        len(set(manifest_ids)) != EXPECTED_CASE_COUNT
        or len(parent_map) != EXPECTED_CASE_COUNT
        or len(candidate_map) != EXPECTED_CASE_COUNT
        or set(parent_map) != set(manifest_ids)
        or set(candidate_map) != set(manifest_ids)
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Whole-response review case identities differ from the exact manifest."
        )

    rows = []
    for spec in manifest_cases:
        case_id = str(spec.case_id)
        parent = parent_map[case_id]
        candidate = candidate_map[case_id]
        generation_limit = spec.generation_limit
        if (
            parent.prompt != spec.prompt
            or parent.generation.generation_limit != generation_limit
            or candidate.generation.generation_limit != generation_limit
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                f"Whole-response review evidence differs for {case_id}."
            )
        rows.append(
            {
                "case_id": case_id,
                "prompt": spec.prompt,
                "parent_response": parent.parent_output,
                "candidate_response": candidate.output,
                "generation_limit": generation_limit,
                "parent_completion": {
                    "generated_token_count": parent.generation.generated_token_count,
                    "eos_observed": parent.generation.eos_observed,
                    "unfinished_at_generation_limit": (
                        parent.generation.unfinished_at_generation_limit
                    ),
                },
                "candidate_completion": {
                    "generated_token_count": (
                        candidate.generation.generated_token_count
                    ),
                    "eos_observed": candidate.generation.eos_observed,
                    "unfinished_at_generation_limit": (
                        candidate.generation.unfinished_at_generation_limit
                    ),
                },
            }
        )
    return build_all_case_parent_relative_manual_packet(
        automatic_gate_manual_packet,
        rows,
        manifest_sha256=manifest.manifest_sha256,
        manifest_document_sha256=manifest.document_sha256,
    )


def _require_exact_all_case_manual_ratings(
    ratings_document: Mapping[str, Any],
    manifest: Any,
) -> None:
    """Fail closed unless ratings review every manifest response exactly once."""

    expected_ids = tuple(str(value.case_id) for value in manifest.cases)
    rows = ratings_document.get("ratings")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASE_COUNT:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Manual ratings must review all 64 whole responses exactly once."
        )
    if any(not isinstance(row, Mapping) for row in rows):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Manual ratings contain a malformed whole-response review."
        )
    actual_ids = tuple(str(row.get("case_id")) for row in rows)
    if actual_ids != expected_ids or len(set(actual_ids)) != EXPECTED_CASE_COUNT:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Manual ratings do not cover the exact 64-case manifest in order."
        )
    if any(
        not isinstance(row.get("reason"), str) or not row["reason"].strip()
        for row in rows
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Every whole-response review requires a concrete reason."
        )
    qualitative = ratings_document.get("qualitative_case_reviews")
    suite = ratings_document.get("holistic_suite_conclusion")
    if (
        not isinstance(qualitative, list)
        or len(qualitative) != EXPECTED_CASE_COUNT
        or any(not isinstance(row, Mapping) for row in qualitative)
        or tuple(str(row.get("case_id")) for row in qualitative) != expected_ids
        or not isinstance(suite, Mapping)
        or ratings_document.get("qualitative_judgment_is_authoritative") is not True
        or ratings_document.get("numeric_scores_and_thresholds_are_diagnostic_only")
        is not True
        or ratings_document.get("automatic_scalar_accept_reject_permitted") is not False
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Manual ratings require authoritative qualitative verdicts for all "
            "64 cases plus one explicit holistic suite conclusion."
        )


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    candidate_directory: Path
    parent_envelope: Path
    expected_parent_envelope_file_sha256: str
    expected_parent_unique_parameters: int
    expected_candidate_state_sha256: str
    expected_target_tokenizer_semantics_sha256: str
    output_directory: Path
    semantic_postfold_receipt: Path | None = None
    expected_semantic_postfold_receipt_file_sha256: str | None = None
    expected_semantic_postfold_receipt_sha256: str | None = None
    expected_candidate_artifact_sha256: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        candidate = _local_directory(
            self.candidate_directory,
            "candidate_directory",
        )
        envelope = _local_file(self.parent_envelope, "parent_envelope")
        for name in (
            "expected_parent_envelope_file_sha256",
            "expected_candidate_state_sha256",
            "expected_target_tokenizer_semantics_sha256",
        ):
            _digest(getattr(self, name), name)
        semantic_bindings = (
            self.semantic_postfold_receipt,
            self.expected_semantic_postfold_receipt_file_sha256,
            self.expected_semantic_postfold_receipt_sha256,
            self.expected_candidate_artifact_sha256,
        )
        if any(value is not None for value in semantic_bindings) and not all(
            value is not None for value in semantic_bindings
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold receipt path, file SHA-256, receipt seal, "
                "and candidate artifact SHA-256 must be supplied together."
            )
        semantic_receipt = None
        if self.semantic_postfold_receipt is not None:
            semantic_receipt = _local_file(
                self.semantic_postfold_receipt,
                "semantic_postfold_receipt",
            )
            if _is_within(semantic_receipt, candidate):
                raise Lfm2CpcfV5StandaloneEvaluationError(
                    "Semantic-postfold provenance must remain outside the "
                    "standalone child artifact."
                )
            for name in (
                "expected_semantic_postfold_receipt_file_sha256",
                "expected_semantic_postfold_receipt_sha256",
                "expected_candidate_artifact_sha256",
            ):
                _digest(str(getattr(self, name)), name)
        _positive_int(
            self.expected_parent_unique_parameters,
            "expected_parent_unique_parameters",
        )
        if not isinstance(self.device, str) or not self.device.strip():
            raise Lfm2CpcfV5StandaloneEvaluationError("device must be nonempty.")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise Lfm2CpcfV5StandaloneEvaluationError("dtype must be nonempty.")
        output = Path(self.output_directory).expanduser().absolute()
        object.__setattr__(self, "candidate_directory", candidate)
        object.__setattr__(self, "parent_envelope", envelope)
        object.__setattr__(self, "output_directory", output)
        object.__setattr__(self, "semantic_postfold_receipt", semantic_receipt)


@dataclass(frozen=True, slots=True)
class FinalizeConfig:
    capture_directory: Path
    expected_capture_report_file_sha256: str
    expected_capture_report_sha256: str
    parent_envelope: Path
    expected_parent_envelope_file_sha256: str
    manual_ratings: Path
    expected_manual_ratings_file_sha256: str
    output_directory: Path
    expected_semantic_postfold_receipt_sha256: str | None = None
    expected_candidate_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        capture = _local_directory(
            self.capture_directory,
            "capture_directory",
        )
        envelope = _local_file(self.parent_envelope, "parent_envelope")
        ratings = _local_file(self.manual_ratings, "manual_ratings")
        for name in (
            "expected_capture_report_file_sha256",
            "expected_capture_report_sha256",
            "expected_parent_envelope_file_sha256",
            "expected_manual_ratings_file_sha256",
        ):
            _digest(getattr(self, name), name)
        semantic_bindings = (
            self.expected_semantic_postfold_receipt_sha256,
            self.expected_candidate_artifact_sha256,
        )
        if any(value is not None for value in semantic_bindings) and not all(
            value is not None for value in semantic_bindings
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Final semantic receipt and candidate artifact SHA-256 pins "
                "must be supplied together."
            )
        for name, value in (
            (
                "expected_semantic_postfold_receipt_sha256",
                self.expected_semantic_postfold_receipt_sha256,
            ),
            (
                "expected_candidate_artifact_sha256",
                self.expected_candidate_artifact_sha256,
            ),
        ):
            if value is not None:
                _digest(value, name)
        output = Path(self.output_directory).expanduser().absolute()
        object.__setattr__(self, "capture_directory", capture)
        object.__setattr__(self, "parent_envelope", envelope)
        object.__setattr__(self, "manual_ratings", ratings)
        object.__setattr__(self, "output_directory", output)


def _authenticate_semantic_postfold_provenance(
    config: CaptureConfig,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind a standalone artifact to its independently pinned fold receipt."""

    receipt_path = config.semantic_postfold_receipt
    if receipt_path is None:
        return None
    receipt_document = _read_canonical_json(
        receipt_path,
        name="semantic-postfold receipt",
        expected_file_sha256=str(config.expected_semantic_postfold_receipt_file_sha256),
    )
    receipt = _sealed_report(
        receipt_document,
        schema=LFM2_SEMANTIC_POSTFOLD_SCHEMA,
        seal_field="result_sha256",
        name="semantic-postfold receipt",
    )
    expected_receipt_sha256 = _digest(
        config.expected_semantic_postfold_receipt_sha256,
        "expected_semantic_postfold_receipt_sha256",
    )
    if receipt["result_sha256"] != expected_receipt_sha256:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold receipt differs from its independent seal."
        )
    checkpoint_value = receipt.get("checkpoint")
    if not isinstance(checkpoint_value, Mapping):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold receipt has no checkpoint publication receipt."
        )
    checkpoint = _sealed_report(
        checkpoint_value,
        schema=LFM2_SEMANTIC_POSTFOLD_CHECKPOINT_SCHEMA,
        seal_field="result_sha256",
        name="semantic-postfold checkpoint receipt",
    )
    expected_artifact_sha256 = _digest(
        config.expected_candidate_artifact_sha256,
        "expected_candidate_artifact_sha256",
    )
    expected_state_sha256 = config.expected_candidate_state_sha256
    expected_semantics_sha256 = config.expected_target_tokenizer_semantics_sha256
    context = receipt.get("context_binding")
    claims = receipt.get("claims")
    operation_audit = receipt.get("operation_audit")
    checkpoint_audit = checkpoint.get("operation_audit")
    parameter_inventory = metadata.get("parameter_inventory")
    attention_layer_stack = receipt.get("semantic_layer_stack")
    attention_authentication = receipt.get("attention_receipt_authentication")
    context_attention_sha256 = (
        context.get("attention_receipt_authentication_sha256")
        if isinstance(context, Mapping)
        else None
    )
    attention_contract_values = (
        attention_layer_stack,
        attention_authentication,
        context_attention_sha256,
    )
    attention_contract_present = any(
        value is not None for value in attention_contract_values
    )
    if attention_contract_present and not all(
        value is not None for value in attention_contract_values
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold attention persistence contract is partial."
        )
    authenticated_attention = None
    if attention_contract_present:
        if not isinstance(attention_layer_stack, Mapping) or not isinstance(
            attention_authentication,
            Mapping,
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold attention persistence contract is malformed."
            )
        try:
            authenticated_attention = (
                authenticate_lfm2_semantic_layer_stack_attention_receipts(
                    attention_layer_stack
                )
            )
        except (Lfm2SemanticPostfoldError, TypeError, ValueError) as exc:
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold attention receipts failed authentication."
            ) from exc
        if (
            dict(authenticated_attention) != dict(attention_authentication)
            or authenticated_attention.get("result_sha256") != context_attention_sha256
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold attention receipt persistence changed."
            )
    artifact_tree_fields = (
        "artifact_tree",
        "artifact_tree_sha256",
        "artifact_file_count",
        "artifact_total_bytes",
        "complete_artifact_tree_hashed",
        "forbidden_named_or_referenced_sidecars_present",
        "standalone_directory_purity_authenticated",
    )
    artifact_tree_contract_present = any(
        checkpoint.get(name) is not None for name in artifact_tree_fields
    )
    if artifact_tree_contract_present and any(
        checkpoint.get(name) is None for name in artifact_tree_fields
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold artifact-tree purity contract is partial."
        )
    enhanced_artifact_inventory_fields = (
        "artifact_inventory_schema",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "canonical_typed_artifact_inventory_authenticated",
        "staged_artifact_tree_purity_verified_before_publication",
        "unallowlisted_artifact_files_present",
    )
    enhanced_artifact_inventory_present = any(
        checkpoint.get(name) is not None for name in enhanced_artifact_inventory_fields
    )
    if enhanced_artifact_inventory_present and (
        not artifact_tree_contract_present
        or any(
            checkpoint.get(name) is None for name in enhanced_artifact_inventory_fields
        )
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold typed artifact inventory contract is partial."
        )
    try:
        authenticated_artifact_inventory = canonical_standalone_fold_artifact_inventory(
            config.candidate_directory
        )
        artifact_file_roles = {
            value["path"]: value["artifact_role"]
            for value in authenticated_artifact_inventory
        }
        authenticated_artifact_tree = scan_standalone_fold_artifact_tree(
            config.candidate_directory,
            expected_file_roles=artifact_file_roles,
        )
    except DynamicFoldArtifactError as exc:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold standalone artifact tree failed purity scanning."
        ) from exc
    if artifact_tree_contract_present:
        if (
            checkpoint.get("artifact_tree")
            != [dict(value) for value in authenticated_artifact_tree]
            or checkpoint.get("artifact_tree_sha256")
            != canonical_sha256(authenticated_artifact_tree)
            or checkpoint.get("artifact_file_count") != len(authenticated_artifact_tree)
            or checkpoint.get("artifact_total_bytes")
            != sum(int(value["size_bytes"]) for value in authenticated_artifact_tree)
            or checkpoint.get("complete_artifact_tree_hashed") is not True
            or checkpoint.get("forbidden_named_or_referenced_sidecars_present")
            is not False
            or checkpoint.get("standalone_directory_purity_authenticated") is not True
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold standalone artifact-tree identity changed."
            )
        if enhanced_artifact_inventory_present and (
            checkpoint.get("artifact_inventory_schema")
            != STANDALONE_FOLD_ARTIFACT_INVENTORY_SCHEMA
            or checkpoint.get("artifact_inventory")
            != [dict(value) for value in authenticated_artifact_inventory]
            or checkpoint.get("artifact_inventory_sha256")
            != canonical_sha256(authenticated_artifact_inventory)
            or checkpoint.get("canonical_typed_artifact_inventory_authenticated")
            is not True
            or checkpoint.get("staged_artifact_tree_purity_verified_before_publication")
            is not True
            or checkpoint.get("unallowlisted_artifact_files_present") is not False
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold typed artifact inventory identity changed."
            )
    artifact_purity_promotion_eligible = bool(
        artifact_tree_contract_present and enhanced_artifact_inventory_present
    )
    if not isinstance(context, Mapping):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold receipt has no context binding."
        )
    context_body = dict(context)
    context_sha256 = context_body.pop("context_sha256", None)
    if _digest(context_sha256, "semantic-postfold context binding") != canonical_sha256(
        context_body
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold context binding self-hash is invalid."
        )
    fresh_reload_parent_free_state_sha256 = _digest(
        receipt.get("fresh_reload_parent_free_state_sha256"),
        "semantic-postfold fresh-reload parent-free state",
    )
    if (
        receipt.get("status") != "standalone-semantic-postfold-saved-and-reloaded"
        or context_sha256 != checkpoint.get("context_binding_sha256")
        or checkpoint.get("context_binding_sha256")
        != metadata.get("context_binding_sha256")
        or checkpoint.get("artifact_sha256") != expected_artifact_sha256
        or metadata.get("artifact_sha256") != expected_artifact_sha256
        or checkpoint.get("model_state_sha256") != expected_state_sha256
        or metadata.get("model_state_sha256") != expected_state_sha256
        or checkpoint.get("parameter_inventory_sha256")
        != metadata.get("parameter_inventory_sha256")
        or checkpoint.get("tokenizer_semantics_sha256") != expected_semantics_sha256
        or metadata.get("tokenizer_semantics_sha256") != expected_semantics_sha256
        or checkpoint.get("materialized_ffn_width")
        != metadata.get("materialized_ffn_width")
        or not isinstance(parameter_inventory, Mapping)
        or checkpoint.get("parameter_element_count")
        != parameter_inventory.get("unique_parameters")
        or checkpoint.get("staged_reload_verified") is not True
        or checkpoint.get("published_reload_verified") is not True
        or checkpoint.get("standalone_dense_safetensors") is not True
        or checkpoint.get("parent_model_required_for_reload") is not False
        or checkpoint.get("cache_required_for_reload") is not False
        or checkpoint.get("network_access_performed") is not False
        or not isinstance(claims, Mapping)
        or claims.get("standalone_child") is not True
        or claims.get("training_performed") is not False
        or claims.get("pruning_performed") is not False
        or claims.get("quantization_performed") is not False
        or not isinstance(operation_audit, Mapping)
        or not isinstance(checkpoint_audit, Mapping)
        or dict(operation_audit) != SEMANTIC_POSTFOLD_ZERO_OPERATION_AUDIT
        or dict(checkpoint_audit) != SEMANTIC_POSTFOLD_ZERO_OPERATION_AUDIT
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold receipt does not authenticate this exact "
            "standalone zero-training child."
        )
    producer_directory = checkpoint.get("output_directory")
    if not isinstance(producer_directory, str) or not producer_directory:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Semantic-postfold checkpoint receipt lost its publication path."
        )
    return {
        "schema": SEMANTIC_POSTFOLD_EVALUATION_PROVENANCE_SCHEMA,
        "semantic_postfold_receipt_file_sha256": (
            config.expected_semantic_postfold_receipt_file_sha256
        ),
        "semantic_postfold_receipt_sha256": receipt["result_sha256"],
        "semantic_postfold_checkpoint_receipt_sha256": checkpoint["result_sha256"],
        "candidate_artifact_sha256": expected_artifact_sha256,
        "candidate_model_state_sha256": expected_state_sha256,
        "candidate_parameter_inventory_sha256": metadata["parameter_inventory_sha256"],
        "candidate_tokenizer_semantics_sha256": expected_semantics_sha256,
        "context_binding_sha256": metadata["context_binding_sha256"],
        "fresh_reload_parent_free_state_sha256": (
            fresh_reload_parent_free_state_sha256
        ),
        "producer_output_directory": producer_directory,
        "evaluated_candidate_directory": str(config.candidate_directory),
        "directory_binding_mode": "content_identity_not_absolute_path",
        "receipt_and_checkpoint_seals_authenticated": True,
        "artifact_metadata_matches_checkpoint_receipt": True,
        "attention_receipt_persistence_authenticated": (
            authenticated_attention is not None
        ),
        "attention_receipt_authentication_sha256": (
            None
            if authenticated_attention is None
            else authenticated_attention["result_sha256"]
        ),
        "complete_artifact_tree_purity_authenticated": (artifact_tree_contract_present),
        "artifact_tree_sha256": canonical_sha256(authenticated_artifact_tree),
        "canonical_typed_artifact_inventory_authenticated": (
            enhanced_artifact_inventory_present
        ),
        "artifact_inventory_sha256": canonical_sha256(authenticated_artifact_inventory),
        "live_complete_artifact_tree_scan_authenticated": True,
        "live_canonical_typed_artifact_inventory_scan_authenticated": True,
        "artifact_purity_promotion_eligible": artifact_purity_promotion_eligible,
        "legacy_same_schema_artifact_purity_receipt_quarantined": (
            not artifact_purity_promotion_eligible
        ),
        "artifact_purity_review_status": (
            "enhanced_publisher_tree_and_typed_inventory_authenticated"
            if artifact_purity_promotion_eligible
            else "legacy_receipt_live_scan_only_quarantined"
        ),
        "standalone_parent_free_reload_authenticated": False,
        "zero_training_pruning_quantization_authenticated": True,
    }


def capture_standalone_intelligence(
    config: CaptureConfig,
) -> dict[str, Any]:
    """Reload one retained child, generate 64 responses once, and persist them."""

    if not isinstance(config, CaptureConfig):
        raise TypeError("config must be CaptureConfig.")
    if _is_within(config.parent_envelope, config.candidate_directory):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Cached parent evidence must remain outside the standalone child artifact."
        )
    prospective_output = config.output_directory.resolve(strict=False)
    if _paths_overlap(prospective_output, config.candidate_directory):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Evaluation artifacts must use a directory disjoint from the "
            "child artifact."
        )
    manifest, envelope = _load_parent_envelope(
        config.parent_envelope,
        expected_file_sha256=(config.expected_parent_envelope_file_sha256),
    )
    metadata = _read_artifact_metadata(config.candidate_directory)
    zero_artifact_fields = (
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
    )
    if (
        metadata.get("schema") != LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA
        or metadata.get("model_state_sha256") != config.expected_candidate_state_sha256
        or metadata.get("tokenizer_semantics_sha256")
        != config.expected_target_tokenizer_semantics_sha256
        or any(metadata.get(name) != 0 for name in zero_artifact_fields)
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Retained v5 artifact differs from the independently bound "
            "zero-training child."
        )
    semantic_postfold_provenance = _authenticate_semantic_postfold_provenance(
        config,
        metadata,
    )
    output_directory = _output_directory(config.output_directory)
    destinations = (
        output_directory / CHILD_OUTPUT_FILENAME,
        output_directory / AUTOMATIC_REPORT_FILENAME,
        output_directory / MANUAL_PACKET_FILENAME,
    )
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Capture outputs already exist; refusing to regenerate responses."
        )
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=metadata["context_binding_sha256"],
        device=config.device,
        dtype=config.dtype,
        expected_ffn_width=metadata["materialized_ffn_width"],
        expected_model_state_sha256=config.expected_candidate_state_sha256,
        expected_parameter_inventory_sha256=metadata["parameter_inventory_sha256"],
        expected_tokenizer_semantics_sha256=(
            config.expected_target_tokenizer_semantics_sha256
        ),
    )
    model = loader(config.candidate_directory)
    tokenizer = getattr(
        model,
        "_neural_forge_quarantine_tokenizer",
        None,
    )
    load_audit = getattr(
        model,
        "_neural_forge_quarantine_load_audit",
        None,
    )
    if tokenizer is None or not isinstance(load_audit, Mapping):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Strict loader did not return its retained target tokenizer audit."
        )
    if (
        load_audit.get("direct_local_json_and_safetensors") is not True
        or load_audit.get("from_pretrained_used") is not False
        or load_audit.get("hub_loader_used") is not False
        or load_audit.get("network_access") is not False
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Child reload was not strict direct-local JSON+safetensors."
        )

    state_before = fingerprint_model_state(model)
    inventory_before = _parameter_inventory(model)
    child_unique_parameters = _positive_int(
        inventory_before.get("unique_parameters"),
        "child unique parameter count",
    )
    parent_unique_parameters = _positive_int(
        config.expected_parent_unique_parameters,
        "expected_parent_unique_parameters",
    )
    metadata_inventory = metadata.get("parameter_inventory")
    expected_metadata_inventory = dict(inventory_before)
    expected_metadata_inventory.pop("checkpoint_byte_size_used", None)
    training_before = bool(model.training)
    if (
        state_before != config.expected_candidate_state_sha256
        or inventory_before.get("inventory_sha256")
        != metadata["parameter_inventory_sha256"]
        or not isinstance(metadata_inventory, Mapping)
        or dict(metadata_inventory) != expected_metadata_inventory
        or parent_unique_parameters <= child_unique_parameters
        or training_before
        or not _gradients_are_absent(model)
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Fresh child is not the expected compact frozen inference state."
        )
    if semantic_postfold_provenance is not None:
        fresh_reload_parent_free_state_sha256 = fingerprint_lfm2_parent(
            model
        ).identity_sha256
        if fresh_reload_parent_free_state_sha256 != semantic_postfold_provenance.get(
            "fresh_reload_parent_free_state_sha256"
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Fresh strict-local child differs from the semantic-postfold "
                "parent-free reload identity."
            )
        semantic_postfold_provenance = {
            **semantic_postfold_provenance,
            "evaluated_fresh_reload_parent_free_state_sha256": (
                fresh_reload_parent_free_state_sha256
            ),
            "standalone_parent_free_reload_authenticated": True,
        }
    eos_token_ids = _normalized_eos_ids(tokenizer)
    rows = tuple(generate_candidate_rows_once(model, tokenizer, manifest))
    state_after = fingerprint_model_state(model)
    inventory_after = _parameter_inventory(model)
    if (
        len(rows) != EXPECTED_CASE_COUNT
        or state_after != state_before
        or inventory_after != inventory_before
        or bool(model.training) != training_before
        or not _gradients_are_absent(model)
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "One-shot generation changed the child or lost holdout coverage."
        )

    limit_rows = [
        {
            "case_id": spec.case_id,
            "generation_limit": spec.generation_limit,
        }
        for spec in manifest.cases
    ]
    observed_limits = {
        str(row.get("case_id")): row.get("generation_limit") for row in rows
    }
    if any(
        observed_limits.get(spec.case_id) != spec.generation_limit
        for spec in manifest.cases
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Candidate rows did not preserve every per-case generation limit."
        )
    limit_distribution = dict(
        sorted(Counter(spec.generation_limit for spec in manifest.cases).items())
    )
    if not any(limit > 64 for limit in limit_distribution):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "The holdout unexpectedly lost all generation limits above 64."
        )

    child_outputs = seal_lfm2_intelligence_holdout_candidate_outputs(
        manifest,
        rows,
        candidate_state_fingerprint=state_before,
        tokenizer_fingerprint=metadata["tokenizer_package_sha256"],
        eos_token_ids=eos_token_ids,
    )
    typed_outputs = load_lfm2_intelligence_holdout_candidate_outputs(
        child_outputs,
        expected_candidate_outputs_sha256=child_outputs["candidate_outputs_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_candidate_state_fingerprint=state_before,
        expected_tokenizer_fingerprint=metadata["tokenizer_package_sha256"],
        eos_token_ids=eos_token_ids,
    )
    baseline = baseline_from_lfm2_intelligence_holdout_envelope(envelope)
    policy = IntelligenceGatePolicy()
    automatic_gate = evaluate_cached_intelligence(
        baseline,
        typed_outputs.candidates,
        policy=policy,
        manual_ratings=None,
    )
    automatic_gate_document = automatic_gate.as_dict()
    automatic_gate_manual_packet = build_manual_review_packet(
        baseline,
        typed_outputs.candidates,
        policy,
    )
    manual_packet = _build_all_case_parent_relative_manual_packet(
        automatic_gate_manual_packet=automatic_gate_manual_packet,
        baseline=baseline,
        candidates=typed_outputs.candidates,
        manifest=manifest,
    )
    if (
        automatic_gate_document.get("manual_review", {}).get("status") != "pending"
        or automatic_gate_document.get("manual_review", {}).get("packet_sha256")
        != automatic_gate_manual_packet["packet_sha256"]
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Automatic gate did not produce the exact pending manual packet."
        )

    prompt_rows = [
        {"case_id": spec.case_id, "prompt": spec.prompt} for spec in manifest.cases
    ]
    if any(
        parent.case_id != spec.case_id or parent.prompt != spec.prompt
        for parent, spec in zip(baseline.cases, manifest.cases)
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Cached parent prompts differ from the child holdout prompts."
        )
    realized_compaction_ratio = parent_unique_parameters / child_unique_parameters
    semantic_artifact_purity_finalization_eligible = (
        None
        if semantic_postfold_provenance is None
        else semantic_postfold_provenance["artifact_purity_promotion_eligible"]
    )
    capture_status = (
        "automatic_diagnostics_complete_artifact_purity_quarantined"
        if semantic_artifact_purity_finalization_eligible is False
        else "automatic_diagnostics_complete_manual_review_pending"
    )

    report_core = {
        "schema": CAPTURE_REPORT_SCHEMA,
        "status": capture_status,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_document_sha256": manifest.document_sha256,
        "case_count": len(manifest.cases),
        "parent_envelope_file_sha256": (config.expected_parent_envelope_file_sha256),
        "parent_envelope_sha256": envelope.envelope_sha256,
        "parent_envelope_document_sha256": envelope.document_sha256,
        "candidate_state_sha256": state_before,
        "candidate_artifact_sha256": metadata["artifact_sha256"],
        "candidate_parameter_inventory_sha256": metadata["parameter_inventory_sha256"],
        "candidate_tokenizer_package_sha256": metadata["tokenizer_package_sha256"],
        "target_tokenizer_semantics_sha256": metadata["tokenizer_semantics_sha256"],
        "target_eos_token_ids": list(eos_token_ids),
        "candidate_outputs_sha256": typed_outputs.candidate_outputs_sha256,
        "candidate_outputs_document_sha256": typed_outputs.document_sha256,
        "automatic_gate_report": automatic_gate_document,
        "automatic_gate_report_sha256": automatic_gate.report_sha256,
        "automatic_gate_report_is_diagnostic_only": True,
        "automatic_gate_manual_packet_sha256": (
            automatic_gate_manual_packet["packet_sha256"]
        ),
        "manual_review_packet_sha256": manual_packet["packet_sha256"],
        "manual_review_packet_case_count": len(manual_packet["cases"]),
        "manual_review_scope": manual_packet["review_scope"],
        "manual_review_generation_limit_rows_sha256": manual_packet[
            "generation_limit_rows_sha256"
        ],
        "manual_review_completion_metadata_rows_sha256": manual_packet[
            "completion_metadata_rows_sha256"
        ],
        "semantic_postfold_artifact_purity_finalization_eligible": (
            semantic_artifact_purity_finalization_eligible
        ),
        **(
            {"semantic_postfold_provenance": semantic_postfold_provenance}
            if semantic_postfold_provenance is not None
            else {}
        ),
        "parameter_compaction": {
            "counting_method": inventory_before["method"],
            "parent_state_fingerprint": envelope.parent_state_fingerprint,
            "externally_pinned_expected_parent_unique_parameters": (
                parent_unique_parameters
            ),
            "child_unique_parameters": child_unique_parameters,
            "parameters_removed_from_dense_topology": (
                parent_unique_parameters - child_unique_parameters
            ),
            "recorded_parameter_compaction_ratio": realized_compaction_ratio,
            "child_count_authenticated_from_reloaded_artifact": True,
            "parent_count_independently_recomputed_by_evaluator": False,
            "ratio_certification_status": "externally_pinned_parent_count",
            "checkpoint_byte_ratio_used": False,
            "quantization_adjustment_used": False,
            "pruning_mask_adjustment_used": False,
        },
        "holdout_prompt_binding": {
            "same_manifest_prompts_used_for_parent_and_child": True,
            "prompt_count": len(prompt_rows),
            "prompt_rows_sha256": canonical_sha256(prompt_rows),
            "parent_outputs_are_cached_only": True,
            "parent_outputs_are_not_child_inference_inputs": True,
        },
        "generation_protocol": {
            "mode": "greedy_whole_response",
            "do_sample": False,
            "use_cache": True,
            "child_generate_invocations": len(rows),
            "one_generate_invocation_per_case": True,
            "per_case_manifest_generation_limits_used": True,
            "global_64_token_cap_used": False,
            "generation_limit_distribution": {
                str(limit): count for limit, count in limit_distribution.items()
            },
            "generation_limit_rows_sha256": canonical_sha256(limit_rows),
            "limits_above_64_preserved": True,
            "full_decoded_text_persisted": True,
        },
        "direct_local_reload": {
            **dict(load_audit),
            "strict_v5_loader_used": True,
            "candidate_artifact_directory": str(config.candidate_directory),
        },
        "operation_audit": {
            "child_model_loads": 1,
            "child_generate_invocations": len(rows),
            "parent_model_loaded": False,
            "parent_forward_passes": 0,
            "parent_generate_invocations": 0,
            "parent_cache_regeneration_calls": 0,
            "training_used": False,
            "training_steps": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "pruning_used": False,
            "parameters_pruned": 0,
            "quantization_used": False,
            "parameters_quantized": 0,
            "network_access": False,
            "hub_loader_used": False,
            "candidate_artifact_zero_operation_fields_authenticated": True,
            "semantic_postfold_receipt_authenticated": (
                semantic_postfold_provenance is not None
            ),
        },
        "diagnostic_review_semantics": {
            "whole_response_intelligence": True,
            "exact_token_matching_used": False,
            "exact_wording_matching_used": False,
            "manual_review_required": True,
            "manual_review_is_parent_relative": True,
            "manual_review_case_count": len(manual_packet["cases"]),
            "manual_review_exact_manifest_coverage": True,
            "per_case_generation_limits_bound_to_manual_packet": True,
            "parent_and_child_completion_metadata_bound": True,
            "completion_and_truncation_flags_are_diagnostic_only": True,
            "qualitative_review_criteria": list(QUALITATIVE_CRITERIA),
            "criterion_specific_case_observations_required": True,
            "duplicate_boilerplate_case_observations_permitted": False,
            "diagnostic_parent_relative_retention_reference": 0.90,
            "automatic_metrics_are_behavioral_or_structural": True,
            "lexical_similarity_is_not_an_intelligence_metric": True,
            "retention_result_terminology": "pinned_suite_behavioral_diagnostics",
            "retention_scope": "pinned_64_case_holdout_only",
            "numeric_thresholds_are_diagnostic_only": True,
            "automatic_accept_reject_authority": False,
            "publication_blocked_by_metrics": False,
            "holistic_whole_response_judgment_required": True,
            "statistical_confidence_bound_claimed": False,
            "general_intelligence_retention_claimed": False,
            "semantic_postfold_checkpoint_and_receipt_pinned": (
                semantic_postfold_provenance is not None
            ),
            "semantic_postfold_artifact_purity_finalization_eligible": (
                semantic_artifact_purity_finalization_eligible
            ),
            "legacy_semantic_artifact_purity_is_diagnostic_only": (
                semantic_artifact_purity_finalization_eligible is False
            ),
        },
    }
    automatic_report = {
        **report_core,
        "capture_report_sha256": canonical_sha256(report_core),
    }
    _atomic_publish_documents(
        (
            (destinations[0], child_outputs),
            (destinations[1], automatic_report),
            (destinations[2], manual_packet),
        )
    )
    return automatic_report


def finalize_standalone_intelligence(
    config: FinalizeConfig,
) -> dict[str, Any]:
    """Bind manual ratings and write a final report without loading any model."""

    if not isinstance(config, FinalizeConfig):
        raise TypeError("config must be FinalizeConfig.")
    child_path = config.capture_directory / CHILD_OUTPUT_FILENAME
    automatic_path = config.capture_directory / AUTOMATIC_REPORT_FILENAME
    packet_path = config.capture_directory / MANUAL_PACKET_FILENAME
    for path, name in (
        (child_path, "standalone child outputs"),
        (automatic_path, "automatic intelligence report"),
        (packet_path, "manual review packet"),
    ):
        if path.is_symlink() or not path.is_file():
            raise Lfm2CpcfV5StandaloneEvaluationError(
                f"{name} is missing from the capture directory."
            )
    automatic_document = _read_canonical_json(
        automatic_path,
        name="automatic intelligence report",
        expected_file_sha256=config.expected_capture_report_file_sha256,
    )
    automatic = _sealed_report(
        automatic_document,
        schema=CAPTURE_REPORT_SCHEMA,
        seal_field="capture_report_sha256",
        name="automatic intelligence report",
    )
    if automatic["capture_report_sha256"] != config.expected_capture_report_sha256:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Automatic intelligence report differs from its independent seal."
        )
    semantic_provenance = automatic.get("semantic_postfold_provenance")
    if "semantic_postfold_provenance" in automatic:
        if (
            not isinstance(semantic_provenance, Mapping)
            or semantic_provenance.get("complete_artifact_tree_purity_authenticated")
            is not True
            or semantic_provenance.get(
                "canonical_typed_artifact_inventory_authenticated"
            )
            is not True
            or semantic_provenance.get("live_complete_artifact_tree_scan_authenticated")
            is not True
            or semantic_provenance.get(
                "live_canonical_typed_artifact_inventory_scan_authenticated"
            )
            is not True
            or semantic_provenance.get("artifact_purity_promotion_eligible") is not True
            or semantic_provenance.get(
                "legacy_same_schema_artifact_purity_receipt_quarantined"
            )
            is not False
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold artifact purity is legacy or unauthenticated; "
                "final manual review publication is quarantined."
            )
    if config.expected_semantic_postfold_receipt_sha256 is not None:
        if (
            not isinstance(semantic_provenance, Mapping)
            or semantic_provenance.get("semantic_postfold_receipt_sha256")
            != config.expected_semantic_postfold_receipt_sha256
            or semantic_provenance.get("candidate_artifact_sha256")
            != config.expected_candidate_artifact_sha256
            or semantic_provenance.get("receipt_and_checkpoint_seals_authenticated")
            is not True
            or semantic_provenance.get("artifact_metadata_matches_checkpoint_receipt")
            is not True
            or semantic_provenance.get("standalone_parent_free_reload_authenticated")
            is not True
            or semantic_provenance.get(
                "zero_training_pruning_quantization_authenticated"
            )
            is not True
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Automatic report differs from the independently pinned "
                "semantic child provenance."
            )
    reload_record = automatic.get("direct_local_reload")
    candidate_path_value = (
        reload_record.get("candidate_artifact_directory")
        if isinstance(reload_record, Mapping)
        else None
    )
    if not isinstance(candidate_path_value, str) or not candidate_path_value:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Automatic report lost its standalone child directory binding."
        )
    candidate_directory = Path(candidate_path_value).resolve(strict=False)
    if isinstance(semantic_provenance, Mapping):
        try:
            live_artifact_inventory = canonical_standalone_fold_artifact_inventory(
                candidate_directory
            )
            live_artifact_tree = scan_standalone_fold_artifact_tree(
                candidate_directory,
                expected_file_roles={
                    value["path"]: value["artifact_role"]
                    for value in live_artifact_inventory
                },
            )
        except DynamicFoldArtifactError as exc:
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold standalone artifact tree failed final purity "
                "scanning."
            ) from exc
        if canonical_sha256(live_artifact_tree) != semantic_provenance.get(
            "artifact_tree_sha256"
        ) or canonical_sha256(live_artifact_inventory) != semantic_provenance.get(
            "artifact_inventory_sha256"
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Semantic-postfold standalone artifact purity changed after capture."
            )
    prospective_output = config.output_directory.resolve(strict=False)
    if _paths_overlap(prospective_output, candidate_directory):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Final evaluation artifacts must use a directory disjoint from "
            "the child artifact."
        )
    output_directory = _output_directory(config.output_directory)
    destination = output_directory / FINAL_REPORT_FILENAME
    if destination.exists() or destination.is_symlink():
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Final intelligence report already exists; refusing overwrite."
        )
    manifest, envelope = _load_parent_envelope(
        config.parent_envelope,
        expected_file_sha256=(config.expected_parent_envelope_file_sha256),
    )
    if (
        automatic.get("case_count") != EXPECTED_CASE_COUNT
        or automatic.get("manifest_sha256") != manifest.manifest_sha256
        or automatic.get("parent_envelope_sha256") != envelope.envelope_sha256
        or automatic.get("parent_envelope_file_sha256")
        != config.expected_parent_envelope_file_sha256
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Automatic report belongs to different parent or holdout evidence."
        )

    child_document = _read_canonical_json(
        child_path,
        name="standalone child outputs",
    )
    typed_outputs = load_lfm2_intelligence_holdout_candidate_outputs(
        child_document,
        expected_candidate_outputs_sha256=automatic["candidate_outputs_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_candidate_state_fingerprint=automatic["candidate_state_sha256"],
        expected_tokenizer_fingerprint=automatic["candidate_tokenizer_package_sha256"],
        eos_token_ids=automatic["target_eos_token_ids"],
    )
    if typed_outputs.document_sha256 != automatic["candidate_outputs_document_sha256"]:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Stored child output document differs from the capture report."
        )

    baseline = baseline_from_lfm2_intelligence_holdout_envelope(envelope)
    policy = IntelligenceGatePolicy()
    packet_document = _read_canonical_json(
        packet_path,
        name="manual review packet",
    )
    automatic_gate_manual_packet = build_manual_review_packet(
        baseline,
        typed_outputs.candidates,
        policy,
    )
    rebuilt_packet = _build_all_case_parent_relative_manual_packet(
        automatic_gate_manual_packet=automatic_gate_manual_packet,
        baseline=baseline,
        candidates=typed_outputs.candidates,
        manifest=manifest,
    )
    if (
        automatic_gate_manual_packet["packet_sha256"]
        != automatic["automatic_gate_manual_packet_sha256"]
        or rebuilt_packet["packet_sha256"] != automatic["manual_review_packet_sha256"]
        or rebuilt_packet["completion_metadata_rows_sha256"]
        != automatic["manual_review_completion_metadata_rows_sha256"]
        or canonical_json_bytes(packet_document) != canonical_json_bytes(rebuilt_packet)
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Manual packet does not match the authenticated parent and child."
        )
    stored_packet = rebuilt_packet

    automatic_gate = evaluate_cached_intelligence(
        baseline,
        typed_outputs.candidates,
        policy=policy,
        manual_ratings=None,
    )
    if (
        canonical_json_bytes(automatic_gate.as_dict())
        != canonical_json_bytes(automatic["automatic_gate_report"])
        or automatic_gate.report_sha256 != automatic["automatic_gate_report_sha256"]
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Automatic gate cannot be reproduced from stored evidence."
        )

    ratings_document = _read_canonical_json(
        config.manual_ratings,
        name="manual ratings",
        expected_file_sha256=config.expected_manual_ratings_file_sha256,
    )
    ratings_sha256 = _digest(
        ratings_document.get("ratings_sha256"),
        "manual ratings seal",
    )
    _require_exact_all_case_manual_ratings(ratings_document, manifest)
    try:
        parent_relative_report = evaluate_parent_relative_manual_retention(
            stored_packet,
            ratings_document,
        )
        holistic_review = parent_relative_report.get("holistic_review")
        if (
            not isinstance(holistic_review, Mapping)
            or holistic_review.get("status") != "complete"
            or holistic_review.get("qualitative_judgment_is_authoritative") is not True
            or holistic_review.get("numeric_scores_and_thresholds_are_diagnostic_only")
            is not True
            or holistic_review.get("automatic_scalar_accept_reject_permitted")
            is not False
        ):
            raise Lfm2CpcfV5StandaloneEvaluationError(
                "Manual ratings lack an authoritative holistic review."
            )
        ratings_by_id = {
            str(row["case_id"]): row for row in ratings_document["ratings"]
        }
        candidate_ratings = [
            {
                "case_id": str(case["case_id"]),
                **dict(ratings_by_id[str(case["case_id"])]["candidate"]),
            }
            for case in automatic_gate_manual_packet["cases"]
        ]
        gate_ratings = bind_manual_ratings(
            automatic_gate_manual_packet,
            candidate_ratings,
            reviewer_id=str(ratings_document["reviewer_id"]),
        )
        final_gate = evaluate_cached_intelligence(
            baseline,
            typed_outputs.candidates,
            policy=policy,
            manual_ratings=gate_ratings,
        )
    except (Lfm2IntelligenceGateError, Lfm2ParentRelativeManualGateError) as error:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Manual ratings are not bound to the exact stored review packet."
        ) from error
    final_gate_document = final_gate.as_dict()
    manual_status = final_gate_document.get("manual_review", {}).get("status")
    if manual_status not in {"accepted", "rejected"}:
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Bound manual ratings did not resolve manual review."
        )
    retention_values = (
        final_gate.headline_lower_bound,
        parent_relative_report["retention"],
    )
    observed_behavioral_floor = (
        min(float(value) for value in retention_values if value is not None)
        if all(value is not None for value in retention_values)
        else None
    )
    diagnostic_reference = 0.90
    diagnostic_reference_met = (
        None
        if observed_behavioral_floor is None
        else observed_behavioral_floor >= diagnostic_reference
    )

    final_core = {
        "schema": FINAL_REPORT_SCHEMA,
        "status": "manual_intelligence_review_complete",
        "manual_intelligence_review_complete": True,
        "automatic_accept_reject_decision": None,
        "publication_status": "authenticated_diagnostics_published",
        "behavioral_retention_diagnostics": {
            "status": "diagnostic_complete",
            "observed_pinned_suite_behavioral_retention_floor": (
                observed_behavioral_floor
            ),
            "parent_relative_manual_retention": (parent_relative_report["retention"]),
            "diagnostic_retention_reference": diagnostic_reference,
            "diagnostic_retention_reference_met": diagnostic_reference_met,
            "parent_relative_diagnostic_reference_met": parent_relative_report[
                "diagnostic_retention_reference_met"
            ],
            "severe_regression_flags": parent_relative_report["severe_regressions"],
            "generic_automatic_policy_thresholds_met": final_gate.promoted,
            "generic_automatic_report_is_diagnostic_only": True,
            "numeric_thresholds_are_diagnostic_only": True,
            "automatic_accept_reject_authority": False,
            "publication_blocked_by_metrics": False,
            "scope": "pinned_64_case_holdout_only",
            "statistical_confidence_bound": False,
            "general_intelligence_retention_claimed": False,
            "intelligence_certified": False,
            "exact_output_matching_used": False,
        },
        "holistic_whole_response_judgment": {
            "required_for_any_intelligence_conclusion": True,
            "status": "complete",
            "reviewer_id": ratings_document["reviewer_id"],
            "paired_parent_and_child_responses_reviewed": True,
            "reviewed_case_count": EXPECTED_CASE_COUNT,
            "exact_manifest_case_coverage": True,
            "per_case_generation_limits_bound": True,
            "case_level_reasons_authenticated": True,
            "qualitative_case_verdicts_authenticated": True,
            "qualitative_case_verdicts_sha256": canonical_sha256(
                holistic_review["qualitative_case_reviews"]
            ),
            "authoritative_suite_conclusion": holistic_review["suite_conclusion"],
            "qualitative_judgment_is_authoritative": True,
            "scalar_scores_and_thresholds_are_diagnostic_only": True,
            "automatic_accept_reject_decision_recorded": False,
            "instruction": (
                "A human or agent must judge whole-response correctness, coherence, "
                "instruction following, and usefulness from the authenticated "
                "responses and reasons; scalar metrics cannot decide the result."
            ),
        },
        "manifest_sha256": manifest.manifest_sha256,
        "case_count": len(manifest.cases),
        "parent_envelope_file_sha256": (config.expected_parent_envelope_file_sha256),
        "parent_envelope_sha256": envelope.envelope_sha256,
        "candidate_state_sha256": typed_outputs.candidate_state_fingerprint,
        "candidate_outputs_sha256": (typed_outputs.candidate_outputs_sha256),
        "automatic_capture_report_sha256": automatic["capture_report_sha256"],
        "automatic_capture_report_file_sha256": (
            config.expected_capture_report_file_sha256
        ),
        **(
            {"semantic_postfold_provenance": automatic["semantic_postfold_provenance"]}
            if "semantic_postfold_provenance" in automatic
            else {}
        ),
        "parameter_compaction": automatic["parameter_compaction"],
        "holdout_prompt_binding": automatic["holdout_prompt_binding"],
        "manual_review_packet_sha256": stored_packet["packet_sha256"],
        "manual_review_packet_case_count": len(stored_packet["cases"]),
        "manual_review_scope": stored_packet["review_scope"],
        "manual_review_generation_limit_rows_sha256": stored_packet[
            "generation_limit_rows_sha256"
        ],
        "manual_review_completion_metadata_rows_sha256": stored_packet[
            "completion_metadata_rows_sha256"
        ],
        "manual_ratings_file_sha256": (config.expected_manual_ratings_file_sha256),
        "manual_ratings_sha256": ratings_sha256,
        "reviewer_id": ratings_document["reviewer_id"],
        "parent_relative_manual_report": parent_relative_report,
        "parent_relative_manual_report_sha256": (
            parent_relative_report["report_sha256"]
        ),
        "automatic_diagnostic_bound_manual_ratings_sha256": (
            gate_ratings.ratings_sha256
        ),
        "generic_automatic_diagnostic_report": final_gate_document,
        "generic_automatic_diagnostic_report_sha256": final_gate.report_sha256,
        "operation_audit": {
            "child_model_loaded": False,
            "child_forward_passes": 0,
            "child_generate_invocations": 0,
            "parent_model_loaded": False,
            "parent_forward_passes": 0,
            "parent_generate_invocations": 0,
            "training_used": False,
            "training_steps": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "pruning_used": False,
            "parameters_pruned": 0,
            "quantization_used": False,
            "parameters_quantized": 0,
            "network_access": False,
            "hub_loader_used": False,
            "cached_evidence_only": True,
            "candidate_artifact_zero_operation_fields_authenticated": True,
        },
        "diagnostic_review_semantics": {
            "whole_response_intelligence": True,
            "exact_token_matching_used": False,
            "exact_wording_matching_used": False,
            "manual_ratings_packet_bound": True,
            "manual_review_case_count": len(stored_packet["cases"]),
            "manual_review_exact_manifest_coverage": True,
            "per_case_generation_limits_bound_to_manual_packet": True,
            "parent_and_child_completion_metadata_bound": True,
            "completion_and_truncation_flags_are_diagnostic_only": True,
            "qualitative_review_criteria": list(QUALITATIVE_CRITERIA),
            "criterion_specific_case_observations_required": True,
            "duplicate_boilerplate_case_observations_permitted": False,
            "parent_and_child_scored_on_same_semantic_rubric": True,
            "diagnostic_parent_relative_retention_reference": 0.90,
            "lexical_similarity_is_not_an_intelligence_metric": True,
            "retention_result_terminology": "pinned_suite_behavioral_diagnostics",
            "retention_scope": "pinned_64_case_holdout_only",
            "numeric_thresholds_are_diagnostic_only": True,
            "automatic_accept_reject_authority": False,
            "publication_blocked_by_metrics": False,
            "holistic_whole_response_judgment_required": True,
            "statistical_confidence_bound_claimed": False,
            "general_intelligence_retention_claimed": False,
            "semantic_postfold_checkpoint_and_receipt_pinned": (
                "semantic_postfold_provenance" in automatic
            ),
        },
    }
    final_report = {
        **final_core,
        "final_report_sha256": canonical_sha256(final_core),
    }
    _atomic_publish_documents(((destination, final_report),))
    return final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    capture = modes.add_parser(
        "capture",
        help=(
            "Reload the retained child and persist one 64-case evaluation "
            "with per-case limits up to 96 tokens."
        ),
    )
    capture.add_argument(
        "--candidate-directory",
        type=Path,
        required=True,
    )
    capture.add_argument("--parent-envelope", type=Path, required=True)
    capture.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    capture.add_argument(
        "--expected-parent-unique-parameters",
        type=int,
        required=True,
    )
    capture.add_argument(
        "--expected-candidate-state-sha256",
        required=True,
    )
    capture.add_argument(
        "--expected-target-tokenizer-semantics-sha256",
        required=True,
    )
    capture.add_argument(
        "--semantic-postfold-receipt",
        type=Path,
        help=(
            "External semantic-postfold run receipt. Supply with all three "
            "semantic receipt/artifact SHA-256 pins to authenticate the exact "
            "published semantic child before generation."
        ),
    )
    capture.add_argument(
        "--expected-semantic-postfold-receipt-file-sha256",
    )
    capture.add_argument(
        "--expected-semantic-postfold-receipt-sha256",
    )
    capture.add_argument(
        "--expected-candidate-artifact-sha256",
    )
    capture.add_argument("--output-directory", type=Path, required=True)
    capture.add_argument("--device", default="cuda:0")
    capture.add_argument("--dtype", default="bfloat16")

    finalize = modes.add_parser(
        "finalize",
        help="Bind manual ratings and write the final cached-evidence report.",
    )
    finalize.add_argument("--capture-directory", type=Path, required=True)
    finalize.add_argument("--parent-envelope", type=Path, required=True)
    finalize.add_argument(
        "--expected-capture-report-file-sha256",
        required=True,
    )
    finalize.add_argument(
        "--expected-capture-report-sha256",
        required=True,
    )
    finalize.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    finalize.add_argument("--manual-ratings", type=Path, required=True)
    finalize.add_argument(
        "--expected-manual-ratings-file-sha256",
        required=True,
    )
    finalize.add_argument(
        "--expected-semantic-postfold-receipt-sha256",
    )
    finalize.add_argument(
        "--expected-candidate-artifact-sha256",
    )
    finalize.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "capture":
        values = vars(args).copy()
        values.pop("mode")
        report = capture_standalone_intelligence(CaptureConfig(**values))
        summary = {
            "status": report["status"],
            "case_count": report["case_count"],
            "candidate_outputs_sha256": report["candidate_outputs_sha256"],
            "automatic_gate_report_sha256": report["automatic_gate_report_sha256"],
            "manual_review_packet_sha256": report["manual_review_packet_sha256"],
            "recorded_parameter_compaction_ratio": report["parameter_compaction"][
                "recorded_parameter_compaction_ratio"
            ],
            "capture_report_sha256": report["capture_report_sha256"],
        }
    else:
        values = vars(args).copy()
        values.pop("mode")
        report = finalize_standalone_intelligence(FinalizeConfig(**values))
        summary = {
            "status": report["status"],
            "manual_intelligence_review_complete": report[
                "manual_intelligence_review_complete"
            ],
            "observed_pinned_suite_behavioral_retention_floor": report[
                "behavioral_retention_diagnostics"
            ]["observed_pinned_suite_behavioral_retention_floor"],
            "diagnostic_retention_reference_met": report[
                "behavioral_retention_diagnostics"
            ]["diagnostic_retention_reference_met"],
            "generic_automatic_diagnostic_report_sha256": report[
                "generic_automatic_diagnostic_report_sha256"
            ],
            "final_report_sha256": report["final_report_sha256"],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
