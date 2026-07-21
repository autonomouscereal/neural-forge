#!/usr/bin/env python3
"""Capture and finalize the retained CPCF-v5 child's 64-case intelligence test.

``capture`` performs one direct-local reload of an already-retained v5
``candidate`` directory, generates each built-in holdout case exactly once at
that case's own generation limit, and stores the decoded responses and
automatic semantic gate outside the child artifact.  It also authenticates
the dense child's unique-parameter count and records a parent/child ratio from
an explicitly pinned expected parent count, without using checkpoint bytes,
quantization, or pruning adjustments.

``finalize`` never loads either model.  It authenticates the stored candidate
responses, cached parent envelope, paired parent/child semantic-review packet,
and packet-bound ratings, then writes the final intelligence report.

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
from neural_forge.hf_lfm2_parent_relative_manual_gate import (  # noqa: E402
    Lfm2ParentRelativeManualGateError,
    build_parent_relative_manual_packet,
    evaluate_parent_relative_manual_retention,
)


CAPTURE_REPORT_SCHEMA = "neural-forge-lfm2-cpcf-v5-standalone-intelligence-capture-v2"
FINAL_REPORT_SCHEMA = "neural-forge-lfm2-cpcf-v5-standalone-intelligence-final-v2"
CHILD_OUTPUT_FILENAME = "standalone-child-outputs-64.json"
AUTOMATIC_REPORT_FILENAME = "automatic-intelligence-report-64.json"
MANUAL_PACKET_FILENAME = "manual-review-packet-64.json"
FINAL_REPORT_FILENAME = "final-intelligence-report-64.json"
EXPECTED_CASE_COUNT = 64
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
        raise Lfm2CpcfV5StandaloneEvaluationError(
            f"{name} must be a positive integer."
        )
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


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    candidate_directory: Path
    parent_envelope: Path
    expected_parent_envelope_file_sha256: str
    expected_parent_unique_parameters: int
    expected_candidate_state_sha256: str
    expected_target_tokenizer_semantics_sha256: str
    output_directory: Path
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


@dataclass(frozen=True, slots=True)
class FinalizeConfig:
    capture_directory: Path
    parent_envelope: Path
    expected_parent_envelope_file_sha256: str
    manual_ratings: Path
    expected_manual_ratings_file_sha256: str
    output_directory: Path

    def __post_init__(self) -> None:
        capture = _local_directory(
            self.capture_directory,
            "capture_directory",
        )
        envelope = _local_file(self.parent_envelope, "parent_envelope")
        ratings = _local_file(self.manual_ratings, "manual_ratings")
        for name in (
            "expected_parent_envelope_file_sha256",
            "expected_manual_ratings_file_sha256",
        ):
            _digest(getattr(self, name), name)
        output = Path(self.output_directory).expanduser().absolute()
        object.__setattr__(self, "capture_directory", capture)
        object.__setattr__(self, "parent_envelope", envelope)
        object.__setattr__(self, "manual_ratings", ratings)
        object.__setattr__(self, "output_directory", output)


def capture_standalone_intelligence(
    config: CaptureConfig,
) -> dict[str, Any]:
    """Reload one retained child, generate 64 responses once, and persist them."""

    if not isinstance(config, CaptureConfig):
        raise TypeError("config must be CaptureConfig.")
    if _is_within(config.parent_envelope, config.candidate_directory):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Cached parent evidence must remain outside the standalone child "
            "artifact."
        )
    prospective_output = config.output_directory.resolve(strict=False)
    if _paths_overlap(prospective_output, config.candidate_directory):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Evaluation artifacts must use a directory disjoint from the "
            "child artifact."
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
    manual_packet = build_parent_relative_manual_packet(
        automatic_gate_manual_packet
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
        {"case_id": spec.case_id, "prompt": spec.prompt}
        for spec in manifest.cases
    ]
    if any(
        parent.case_id != spec.case_id or parent.prompt != spec.prompt
        for parent, spec in zip(baseline.cases, manifest.cases)
    ):
        raise Lfm2CpcfV5StandaloneEvaluationError(
            "Cached parent prompts differ from the child holdout prompts."
        )
    realized_compaction_ratio = parent_unique_parameters / child_unique_parameters

    report_core = {
        "schema": CAPTURE_REPORT_SCHEMA,
        "status": "automatic_evaluation_complete_manual_review_pending",
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
        "automatic_gate_manual_packet_sha256": (
            automatic_gate_manual_packet["packet_sha256"]
        ),
        "manual_review_packet_sha256": manual_packet["packet_sha256"],
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
        },
        "acceptance_semantics": {
            "whole_response_intelligence": True,
            "exact_token_matching_used": False,
            "exact_wording_matching_used": False,
            "manual_review_required": True,
            "manual_review_is_parent_relative": True,
            "minimum_parent_relative_manual_retention": 0.90,
            "automatic_metrics_are_behavioral_or_structural": True,
            "lexical_similarity_is_not_an_intelligence_metric": True,
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
    )
    automatic = _sealed_report(
        automatic_document,
        schema=CAPTURE_REPORT_SCHEMA,
        seal_field="capture_report_sha256",
        name="automatic intelligence report",
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
    rebuilt_packet = build_parent_relative_manual_packet(
        automatic_gate_manual_packet
    )
    if (
        automatic_gate_manual_packet["packet_sha256"]
        != automatic["automatic_gate_manual_packet_sha256"]
        or rebuilt_packet["packet_sha256"]
        != automatic["manual_review_packet_sha256"]
        or canonical_json_bytes(packet_document)
        != canonical_json_bytes(rebuilt_packet)
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
    try:
        parent_relative_report = evaluate_parent_relative_manual_retention(
            stored_packet,
            ratings_document,
        )
        candidate_ratings = [
            {
                "case_id": row["case_id"],
                **dict(row["candidate"]),
            }
            for row in ratings_document["ratings"]
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
    promoted = bool(final_gate.promoted and parent_relative_report["passed"])
    retention_values = (
        final_gate.headline_lower_bound,
        parent_relative_report["retention"],
    )
    certified_lower_bound = (
        min(float(value) for value in retention_values if value is not None)
        if promoted
        else None
    )

    final_core = {
        "schema": FINAL_REPORT_SCHEMA,
        "status": "promoted" if promoted else "rejected",
        "promoted": promoted,
        "intelligence_retention_claim": {
            "status": (
                "certified_conservative_lower_bound"
                if promoted
                else "rejected"
            ),
            "certified_lower_bound": certified_lower_bound,
            "parent_relative_manual_retention": (
                parent_relative_report["retention"]
            ),
            "minimum_required_retention": 0.90,
            "exact_output_matching_used": False,
        },
        "manifest_sha256": manifest.manifest_sha256,
        "case_count": len(manifest.cases),
        "parent_envelope_file_sha256": (config.expected_parent_envelope_file_sha256),
        "parent_envelope_sha256": envelope.envelope_sha256,
        "candidate_state_sha256": typed_outputs.candidate_state_fingerprint,
        "candidate_outputs_sha256": (typed_outputs.candidate_outputs_sha256),
        "automatic_capture_report_sha256": automatic["capture_report_sha256"],
        "parameter_compaction": automatic["parameter_compaction"],
        "holdout_prompt_binding": automatic["holdout_prompt_binding"],
        "manual_review_packet_sha256": stored_packet["packet_sha256"],
        "manual_ratings_file_sha256": (config.expected_manual_ratings_file_sha256),
        "manual_ratings_sha256": ratings_sha256,
        "reviewer_id": ratings_document["reviewer_id"],
        "parent_relative_manual_report": parent_relative_report,
        "parent_relative_manual_report_sha256": (
            parent_relative_report["report_sha256"]
        ),
        "automatic_gate_bound_manual_ratings_sha256": (
            gate_ratings.ratings_sha256
        ),
        "final_gate_report": final_gate_document,
        "final_gate_report_sha256": final_gate.report_sha256,
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
        "acceptance_semantics": {
            "whole_response_intelligence": True,
            "exact_token_matching_used": False,
            "exact_wording_matching_used": False,
            "manual_ratings_packet_bound": True,
            "parent_and_child_scored_on_same_semantic_rubric": True,
            "minimum_parent_relative_manual_retention": 0.90,
            "lexical_similarity_is_not_an_intelligence_metric": True,
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
        help="Reload the retained child and persist one 64-case evaluation.",
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
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    finalize.add_argument("--manual-ratings", type=Path, required=True)
    finalize.add_argument(
        "--expected-manual-ratings-file-sha256",
        required=True,
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
            "recorded_parameter_compaction_ratio": report[
                "parameter_compaction"
            ]["recorded_parameter_compaction_ratio"],
            "capture_report_sha256": report["capture_report_sha256"],
        }
    else:
        values = vars(args).copy()
        values.pop("mode")
        report = finalize_standalone_intelligence(FinalizeConfig(**values))
        summary = {
            "status": report["status"],
            "promoted": report["promoted"],
            "certified_lower_bound": report["intelligence_retention_claim"][
                "certified_lower_bound"
            ],
            "final_gate_report_sha256": report["final_gate_report_sha256"],
            "final_report_sha256": report["final_report_sha256"],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
