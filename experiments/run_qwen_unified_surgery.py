#!/usr/bin/env python3
"""Run bounded, zero-recovery mixed Qwen3 neuron/attention surgery.

The production path starts from an executable full-depth Qwen3 codebook
identity, collects bounded sensitivity evidence, independently measures
candidate-marginal damage, and commits only batches that pass an auditable
cached-parent-relative behavioral gate.  Every rejected candidate or batch is
rolled back by :class:`Qwen3UnifiedSurgeryAllocator`; evidence is refreshed only
inside an accepted transaction.

``--candidate-validation-mode provisional`` is intentionally explicit.  It
uses discovery risk in place of independent candidate measurements and labels
the resulting artifact provisional; it never upgrades that evidence into a
candidate-marginal claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from neural_forge.adaptive_surgery_graph import EvaluationDecision
from neural_forge.hf_evaluation_provenance import (
    LoadedTextEvaluationProvenance,
    TokenPathDomain,
    assert_evaluation_report_count_contract,
    assert_strict_evaluation_path_disjointness,
    bind_loaded_text_evaluation_provenance,
    bind_prompt_path_domain,
)
from neural_forge.hf_qwen_codebook_model import (
    load_qwen3_codebook_checkpoint,
    plan_qwen3_codebook_artifact,
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)
from neural_forge.hf_qwen_sensitivity import (
    Qwen3AttentionGroupRef,
    Qwen3ModelFingerprintManifest,
    Qwen3NeuronRef,
    Qwen3SensitivityEvidence,
    Qwen3SensitivityLimits,
    collect_qwen3_sensitivity_evidence,
)
import neural_forge.hf_qwen_surgery_allocator as surgery_allocator
from neural_forge.hf_qwen_surgery_allocator import (
    Qwen3CandidateValidationDamage,
    Qwen3MixedSurgeryBatch,
    Qwen3SurgeryAllocatorLimits,
    Qwen3UnifiedSurgeryAccounting,
    Qwen3UnifiedSurgeryAllocator,
    Qwen3UnifiedSurgeryEvaluationContext,
    Qwen3UnifiedSurgeryProposal,
    Qwen3UnifiedSurgeryRefreshContext,
)
from neural_forge.hf_tied_depth import exact_token_row_mapping
from neural_forge.hf_tools import _load_eval_suite, evaluate_loaded_text_model


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import run_qwen_codebook_in_memory as codebook_runner  # noqa: E402
import run_qwen_compensated_neuron_fold as production_runner  # noqa: E402


RUN_SCHEMA = "neural-forge-qwen3-unified-surgery-run-v2"
SPLIT_SCHEMA = "neural-forge-qwen3-unified-surgery-four-split-v2"
FRONTIER_SCHEMA = "neural-forge-qwen3-unified-surgery-frontier-v2"
RUN_CONTRACT_SCHEMA = "neural-forge-qwen3-unified-surgery-contract-v3"
MODEL_CONTENT_SCHEMA = "neural-forge-qwen3-model-content-v1"
INITIAL_INCUMBENT_SCHEMA = "neural-forge-qwen3-initial-incumbent-v1"
ACCEPTED_COMMIT_SCHEMA = "neural-forge-qwen3-accepted-commit-v1"
ACCEPTED_COMMIT_JOURNAL_SCHEMA = (
    "neural-forge-qwen3-accepted-commit-journal-v1"
)
CandidateValidationMode = Literal["independent", "provisional"]


class QwenUnifiedSurgeryRunnerError(RuntimeError):
    """The unified production workflow cannot prove a required invariant."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QwenUnifiedSurgeryRunnerError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QwenUnifiedSurgeryRunnerError(
            f"{name} must be a positive integer."
        ) from exc
    if result <= 0 or result != value:
        raise QwenUnifiedSurgeryRunnerError(f"{name} must be a positive integer.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QwenUnifiedSurgeryRunnerError(
            f"{name} must be a non-negative integer."
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QwenUnifiedSurgeryRunnerError(
            f"{name} must be a non-negative integer."
        ) from exc
    if result < 0 or result != value:
        raise QwenUnifiedSurgeryRunnerError(
            f"{name} must be a non-negative integer."
        )
    return result


def _canonical_sha256(value: Any) -> str:
    return production_runner._canonical_sha256(value)


def _record_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _canonical_sha256(payload)


def _assert_run_contract_integrity(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    binding = payload.pop("binding_sha256", None)
    contract = payload.pop("run_contract_sha256", None)
    expected = _canonical_sha256(payload)
    if (
        value.get("schema") != RUN_CONTRACT_SCHEMA
        or binding != expected
        or contract != expected
    ):
        raise QwenUnifiedSurgeryRunnerError(
            "Immutable run contract failed canonical integrity verification."
        )


def _write_json_exclusive_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete immutable JSON record without an overwrite window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise QwenUnifiedSurgeryRunnerError(
                f"Immutable record already exists: {path}."
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenUnifiedSurgeryRunnerError(f"{description} is unreadable.") from exc
    if not isinstance(value, Mapping):
        raise QwenUnifiedSurgeryRunnerError(f"{description} must be a JSON object.")
    return dict(value)


def _model_content_identity(runtime: nn.Module) -> dict[str, Any]:
    """Hash executable tensor content without allocator counters or evidence state."""

    tensors: list[dict[str, Any]] = []
    for name, value in sorted(runtime.state_dict().items()):
        if not isinstance(value, Tensor) or value.device.type == "meta":
            raise QwenUnifiedSurgeryRunnerError(
                f"Cannot content-hash model state entry: {name}."
            )
        tensor = value.detach().contiguous().reshape(-1)
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {"dtype": str(value.dtype), "shape": list(value.shape)},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        )
        elements = max(1, (8 * 1024**2) // max(int(value.element_size()), 1))
        for start in range(0, int(tensor.numel()), elements):
            piece = tensor[start : start + elements].to(device="cpu").contiguous()
            digest.update(piece.view(torch.uint8).numpy().tobytes())
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": digest.hexdigest(),
            }
        )
    report = qwen3_codebook_parameter_report(runtime).as_dict()
    config = dict(runtime.config.to_dict())
    for name in ("_name_or_path", "transformers_version"):
        config.pop(name, None)
    parent_rows = getattr(runtime, "parent_token_rows", None)
    payload = {
        "schema": MODEL_CONTENT_SCHEMA,
        "state_tensor_count": len(tensors),
        "state_tensor_manifest_sha256": _canonical_sha256(tensors),
        "parameter_report_sha256": _canonical_sha256(report),
        "executable_config_sha256": _canonical_sha256(config),
        "parent_token_rows_sha256": (
            None
            if not isinstance(parent_rows, Tensor)
            else hashlib.sha256(_tensor_bytes(parent_rows)).hexdigest()
        ),
    }
    payload["model_content_sha256"] = _canonical_sha256(payload)
    return payload


def _evidence_lineage(evidence: Qwen3SensitivityEvidence) -> dict[str, Any]:
    manifest = evidence.fingerprints.model_manifest
    payload = {
        "model_root": manifest.root,
        "mode": manifest.mode,
        "parent_root": manifest.parent_root,
        "accepted_transaction_sha256": manifest.accepted_transaction_sha256,
        "changed_state_prefixes": list(manifest.changed_state_prefixes),
        "tensors_hashed": manifest.tensors_hashed,
        "tensors_reused": manifest.tensors_reused,
        "calibration_sha256": evidence.fingerprints.calibration,
        "gate_sha256": evidence.fingerprints.gate,
        "runtime_sha256": evidence.fingerprints.runtime,
        "measurement_sha256": evidence.fingerprints.measurement,
        "cache_sha256": evidence.fingerprints.cache,
    }
    payload["lineage_sha256"] = _canonical_sha256(payload)
    return payload


def _verify_evidence_lineage(
    evidence: Qwen3SensitivityEvidence,
    expected: Mapping[str, Any],
) -> None:
    current = _evidence_lineage(evidence)
    for name in (
        "model_root",
        "calibration_sha256",
        "gate_sha256",
        "runtime_sha256",
    ):
        if current.get(name) != expected.get(name):
            raise QwenUnifiedSurgeryRunnerError(
                f"Resumed sensitivity evidence differs from accepted lineage: {name}."
            )


def _prepare_run_output(output: Path, *, resume: bool) -> None:
    """Refuse stale or destructive output-directory reuse."""

    if not output.exists():
        output.mkdir(parents=True)
        return
    if not output.is_dir():
        raise QwenUnifiedSurgeryRunnerError("Run output path is not a directory.")
    entries = tuple(output.iterdir())
    if not entries:
        return
    if not resume:
        raise QwenUnifiedSurgeryRunnerError(
            "--no-resume refuses a nonempty output directory."
        )
    durable_markers = (
        output / "run_report.json",
        output / "initial_incumbent.json",
        output / "accepted_commit_journal",
    )
    if not any(path.exists() for path in durable_markers):
        raise QwenUnifiedSurgeryRunnerError(
            "Nonempty output has no durable unified-surgery resume contract."
        )


def _tensor_bytes(value: Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous().reshape(-1)
    return memoryview(cpu.view(torch.uint8).numpy()).tobytes()


def _positioned_token_hash(token_ids: Tensor) -> tuple[str, str]:
    ids = token_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if not int(ids.numel()):
        raise QwenUnifiedSurgeryRunnerError("A split row cannot tokenize to empty.")
    positions = torch.arange(int(ids.numel()), dtype=torch.long)
    content = hashlib.sha256(_tensor_bytes(ids)).hexdigest()
    positioned = hashlib.sha256(_tensor_bytes(ids) + _tensor_bytes(positions)).hexdigest()
    return positioned, content


def _tokenize_one(tokenizer: Any, text: str, *, max_length: int) -> Tensor:
    try:
        encoded = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            max_length=_positive_int(max_length, "max_length"),
            add_special_tokens=True,
        )
    except Exception as exc:
        raise QwenUnifiedSurgeryRunnerError("Tokenizer failed on a split row.") from exc
    if not isinstance(encoded, Mapping) or not isinstance(
        encoded.get("input_ids"), Tensor
    ):
        raise QwenUnifiedSurgeryRunnerError("Tokenizer must return tensor input_ids.")
    ids = encoded["input_ids"].detach().to(device="cpu", dtype=torch.long)
    if ids.ndim != 2 or int(ids.shape[0]) != 1 or int(ids.shape[1]) == 0:
        raise QwenUnifiedSurgeryRunnerError(
            "Every split row must tokenize to one non-empty rank-2 row."
        )
    mask = encoded.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids, dtype=torch.bool)
    if not isinstance(mask, Tensor) or tuple(mask.shape) != tuple(ids.shape):
        raise QwenUnifiedSurgeryRunnerError("Tokenizer attention-mask shape changed.")
    active = ids[0][mask.detach().to(device="cpu", dtype=torch.bool)[0]]
    if not int(active.numel()):
        raise QwenUnifiedSurgeryRunnerError("A split row has no active tokens.")
    return active


@dataclass(frozen=True, slots=True)
class EncodedPromptRows:
    """One left-padded collector batch plus exact active-token provenance."""

    input_ids: Tensor
    attention_mask: Tensor
    row_fingerprints: tuple[str, ...]
    content_fingerprints: tuple[str, ...]
    prompts_sha256: str
    path_domain: TokenPathDomain

    @property
    def row_count(self) -> int:
        return int(self.input_ids.shape[0])


def encode_prompt_rows(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
    name: str = "collector",
) -> EncodedPromptRows:
    """Tokenize unique prompts into a last-position-safe collector batch."""

    values = tuple(str(item) for item in prompts)
    if not values or any(not item.strip() for item in values):
        raise QwenUnifiedSurgeryRunnerError(
            "Prompt rows must be non-empty strings."
        )
    active_rows = tuple(
        _tokenize_one(tokenizer, value, max_length=max_length) for value in values
    )
    positioned: list[str] = []
    content: list[str] = []
    for row in active_rows:
        positioned_hash, content_hash = _positioned_token_hash(row)
        positioned.append(positioned_hash)
        content.append(content_hash)
    if len(set(positioned)) != len(positioned) or len(set(content)) != len(content):
        raise QwenUnifiedSurgeryRunnerError(
            "Prompt split contains duplicate active token paths."
        )
    width = max(int(row.numel()) for row in active_rows)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise QwenUnifiedSurgeryRunnerError(
            "Tokenizer needs pad_token_id or eos_token_id for batching."
        )
    input_ids = torch.full((len(active_rows), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(active_rows), width), dtype=torch.bool)
    for index, row in enumerate(active_rows):
        length = int(row.numel())
        # Left padding keeps score_position=-1 active in every row.
        input_ids[index, width - length :] = row
        attention_mask[index, width - length :] = True
    return EncodedPromptRows(
        input_ids=input_ids,
        attention_mask=attention_mask,
        row_fingerprints=tuple(positioned),
        content_fingerprints=tuple(content),
        prompts_sha256=production_runner._canonical_sha256(list(values)),
        path_domain=bind_prompt_path_domain(
            tokenizer,
            values,
            name=str(name),
            max_length=max_length,
        ),
    )


def _suite_rows(suite: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for index, text in enumerate(suite["text_evals"]):
        result.append((f"text:{index}", str(text)))
    for collection, prefix in (
        ("choice_probes", "choice"),
        ("generation_prompts", "generation"),
        ("behavior_probes", "behavior"),
    ):
        for index, item in enumerate(suite[collection]):
            result.append(
                (f"{prefix}:{item.get('id', index)}", str(item["prompt"]))
            )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class BoundEvaluationSuite:
    """Normalized evaluation suite bound to exact token-path provenance."""

    name: str
    suite: Mapping[str, Any]
    suite_sha256: str
    validation_fingerprint: str
    row_fingerprints: tuple[str, ...]
    content_fingerprints: tuple[str, ...]
    behavior_ids: tuple[str, ...]
    choice_ids: tuple[str, ...]
    generation_ids: tuple[str, ...]
    evaluation_provenance: LoadedTextEvaluationProvenance

    @property
    def behavioral_sample_count(self) -> int:
        return len(self.behavior_ids) + len(self.choice_ids) + len(self.generation_ids)

    @property
    def behavioral_item_fingerprints(self) -> tuple[str, ...]:
        values = tuple(
            production_runner._canonical_sha256(item.as_dict())
            for item in self.evaluation_provenance.scored_items
            if item.domain in {"behavior", "choice", "generation"}
        )
        if len(values) != self.behavioral_sample_count or len(set(values)) != len(values):
            raise QwenUnifiedSurgeryRunnerError(
                "Behavioral item provenance is incomplete or non-unique."
            )
        return values


def bind_evaluation_suite(
    tokenizer: Any,
    suite: str | Path | Mapping[str, Any],
    *,
    name: str,
    max_new_tokens: int,
) -> BoundEvaluationSuite:
    """Normalize a suite and bind the evaluator's literal scoring paths."""

    normalized = _load_eval_suite(dict(suite) if isinstance(suite, Mapping) else suite)
    if not any(normalized[key] for key in (
        "text_evals",
        "choice_probes",
        "generation_prompts",
        "behavior_probes",
    )):
        raise QwenUnifiedSurgeryRunnerError(f"{name} evaluation suite is empty.")
    evaluation_provenance = bind_loaded_text_evaluation_provenance(
        tokenizer,
        normalized,
        max_new_tokens=max_new_tokens,
    )
    path_domain = evaluation_provenance.to_path_domain(str(name))
    row_fingerprints = path_domain.exact_path_sha256s
    content_fingerprints = path_domain.token_content_sha256s
    behavior_ids = tuple(str(item["id"]) for item in normalized["behavior_probes"])
    choice_ids = tuple(str(item["id"]) for item in normalized["choice_probes"])
    generation_ids = tuple(
        str(item["id"]) for item in normalized["generation_prompts"]
    )
    if not behavior_ids or not choice_ids or not generation_ids:
        raise QwenUnifiedSurgeryRunnerError(
            f"{name} must contain behavior, choice, and generation rows."
        )
    for label, values in (
        ("behavior", behavior_ids),
        ("choice", choice_ids),
        ("generation", generation_ids),
    ):
        if len(values) != len(set(values)):
            raise QwenUnifiedSurgeryRunnerError(
                f"{name} contains duplicate {label} IDs."
            )
    suite_payload = {
        key: normalized[key]
        for key in (
            "text_evals",
            "choice_probes",
            "generation_prompts",
            "behavior_probes",
        )
    }
    suite_sha256 = production_runner._canonical_sha256(suite_payload)
    validation_fingerprint = production_runner._canonical_sha256(
        {
            "name": name,
            "suite_sha256": suite_sha256,
            "evaluator_path_provenance_sha256": (
                evaluation_provenance.aggregate_sha256
            ),
        }
    )
    return BoundEvaluationSuite(
        name=str(name),
        suite=suite_payload,
        suite_sha256=suite_sha256,
        validation_fingerprint=validation_fingerprint,
        row_fingerprints=row_fingerprints,
        content_fingerprints=content_fingerprints,
        behavior_ids=behavior_ids,
        choice_ids=choice_ids,
        generation_ids=generation_ids,
        evaluation_provenance=evaluation_provenance,
    )


def build_split_provenance(
    calibration: EncodedPromptRows,
    sensitivity_gate: EncodedPromptRows,
    transaction_gate: BoundEvaluationSuite,
    candidate_validation: BoundEvaluationSuite | None,
    *,
    mode: CandidateValidationMode,
) -> dict[str, Any]:
    """Fail closed unless the literal collector/evaluator paths are disjoint."""

    domain_objects: dict[str, TokenPathDomain] = {
        "sensitivity_calibration": calibration.path_domain,
        "sensitivity_gate": sensitivity_gate.path_domain,
        "transaction_gate": transaction_gate.evaluation_provenance.to_path_domain(
            "transaction_gate"
        ),
    }
    domains: dict[str, tuple[set[str], set[str], set[str], set[str]]] = {
        "sensitivity_calibration": (
            set(calibration.path_domain.exact_path_sha256s),
            set(calibration.path_domain.token_content_sha256s),
            set(calibration.path_domain.raw_source_text_sha256s),
            set(calibration.path_domain.token_prefix_sha256s),
        ),
        "sensitivity_gate": (
            set(sensitivity_gate.path_domain.exact_path_sha256s),
            set(sensitivity_gate.path_domain.token_content_sha256s),
            set(sensitivity_gate.path_domain.raw_source_text_sha256s),
            set(sensitivity_gate.path_domain.token_prefix_sha256s),
        ),
        "transaction_gate": (
            set(domain_objects["transaction_gate"].exact_path_sha256s),
            set(domain_objects["transaction_gate"].token_content_sha256s),
            set(domain_objects["transaction_gate"].raw_source_text_sha256s),
            set(domain_objects["transaction_gate"].token_prefix_sha256s),
        ),
    }
    strict_report = None
    if mode == "independent":
        if candidate_validation is None:
            raise QwenUnifiedSurgeryRunnerError(
                "Independent mode requires a candidate-validation suite."
            )
        candidate_domain = candidate_validation.evaluation_provenance.to_path_domain(
            "candidate_validation"
        )
        domain_objects["candidate_validation"] = candidate_domain
        domains["candidate_validation"] = (
            set(candidate_domain.exact_path_sha256s),
            set(candidate_domain.token_content_sha256s),
            set(candidate_domain.raw_source_text_sha256s),
            set(candidate_domain.token_prefix_sha256s),
        )
        strict_report = assert_strict_evaluation_path_disjointness(
            calibration=calibration.path_domain,
            sensitivity=sensitivity_gate.path_domain,
            candidate=candidate_validation.evaluation_provenance,
            transaction=transaction_gate.evaluation_provenance,
        )
    overlaps: dict[str, dict[str, int]] = {}
    names = tuple(domains)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            positioned = len(domains[left][0] & domains[right][0])
            content = len(domains[left][1] & domains[right][1])
            raw_source = len(domains[left][2] & domains[right][2])
            left_full_in_right_prefix = len(domains[left][1] & domains[right][3])
            right_full_in_left_prefix = len(domains[right][1] & domains[left][3])
            overlaps[f"{left}__{right}"] = {
                "positioned": positioned,
                "content": content,
                "raw_source_text": raw_source,
                "left_full_content_in_right_prefix": left_full_in_right_prefix,
                "right_full_content_in_left_prefix": right_full_in_left_prefix,
                "token_prefix": (
                    left_full_in_right_prefix + right_full_in_left_prefix
                ),
            }
    if any(
        count
        for item in overlaps.values()
        for count in (
            item["positioned"],
            item["content"],
            item["raw_source_text"],
            item["token_prefix"],
        )
    ):
        raise QwenUnifiedSurgeryRunnerError(
            f"Unified surgery split token overlap is nonzero: {overlaps}."
        )
    payload: dict[str, Any] = {
        "schema": SPLIT_SCHEMA,
        "candidate_validation_mode": mode,
        "domains": {
            name: {
                "positioned_row_count": len(values[0]),
                "content_row_count": len(values[1]),
                "positioned_sha256": production_runner._canonical_sha256(
                    sorted(values[0])
                ),
                "content_sha256": production_runner._canonical_sha256(
                    sorted(values[1])
                ),
                "raw_source_text_count": len(values[2]),
                "raw_source_text_sha256": production_runner._canonical_sha256(
                    sorted(values[2])
                ),
                "token_prefix_count": len(values[3]),
                "token_prefix_sha256": production_runner._canonical_sha256(
                    sorted(values[3])
                ),
            }
            for name, values in domains.items()
        },
        "pairwise_overlaps": overlaps,
        "all_token_paths_disjoint": True,
        "literal_evaluator_path_contract": True,
        "strict_four_domain_report": (
            None if strict_report is None else strict_report.as_dict()
        ),
        "candidate_marginal_measurement_independent": mode == "independent",
        "provisional_unvalidated_candidate_risk": mode == "provisional",
    }
    payload["provenance_sha256"] = production_runner._canonical_sha256(payload)
    return payload


def _bounded_refs(
    layers: int,
    width: int,
    count: int,
    ref_type: type[Qwen3NeuronRef] | type[Qwen3AttentionGroupRef],
) -> tuple[Qwen3NeuronRef | Qwen3AttentionGroupRef, ...]:
    total = _positive_int(layers, "layers") * _positive_int(width, "width")
    requested = min(_positive_int(count, "candidate count"), total)
    indices = [((2 * index + 1) * total) // (2 * requested) for index in range(requested)]
    return tuple(ref_type(index // width, index % width) for index in indices)


def prepare_qwen3_identity_codebook(
    model: nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    device: str,
    max_length: int,
    expected_layers: int = 36,
) -> tuple[nn.Module, dict[str, Any]]:
    """Convert a dense Qwen3 parent into an executable N/N identity runtime."""

    layers = codebook_runner.build_qwen3_bundles(model)
    expected = _positive_int(expected_layers, "expected_layers")
    if layers.logical_layers != expected:
        raise QwenUnifiedSurgeryRunnerError(
            f"Expected a {expected}-layer Qwen3 parent; found {layers.logical_layers}."
        )
    parent_rows = exact_token_row_mapping(tokenizer, tokenizer)
    reference = codebook_runner._capture_noop_reference(
        model,
        tokenizer,
        str(prompt),
        parent_rows,
        max_length=max_length,
        device=device,
    )
    converted = prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=expected,
        neuron_prototypes_per_neuron=expected,
        attention_use_scales=False,
        neuron_use_scales=False,
    )
    if converted is not model:
        raise QwenUnifiedSurgeryRunnerError("Qwen3 identity conversion was not in place.")
    projection_report = codebook_runner._verify_full_depth_projections(model, layers)
    logit_report = codebook_runner._verify_noop_logits(
        model, reference, device=device
    )
    report = qwen3_codebook_parameter_report(model)
    if (
        report.logical_layers != expected
        or report.attention_prototypes_per_group != expected
        or report.neuron_prototypes_per_neuron != expected
        or not report.full_logical_depth
        or not report.exact_parent_token_rows
    ):
        raise QwenUnifiedSurgeryRunnerError(
            "Prepared Qwen3 runtime is not an exact full-depth codebook identity."
        )
    codebook_runner._assert_zero_gradients(model, "prepared identity")
    codebook_runner._assert_model_finite(model, "prepared identity")
    return model, {
        "logical_layers": expected,
        "attention_prototypes_per_group": expected,
        "neuron_prototypes_per_neuron": expected,
        "implicit_permutations": "identity",
        "attention_use_scales": False,
        "neuron_use_scales": False,
        "projection_verification": projection_report,
        "execution_verification": logit_report,
        "parameter_report": report.as_dict(),
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "recovery_steps": 0,
    }


class _GuardedEvaluationAdapter:
    """Supply harmless module-like methods around allocator-guarded inference."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def to(self, _device: Any) -> _GuardedEvaluationAdapter:
        return self

    def eval(self) -> _GuardedEvaluationAdapter:
        return self

    def parameters(self) -> tuple[()]:
        return ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._runtime.generate(*args, **kwargs)


SuiteEvaluator = Callable[..., dict[str, Any]]


class SuiteExecutor:
    """Reuse the production evaluator for live and allocator-guarded runtimes."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        device: str,
        max_new_tokens: int,
        evaluator: SuiteEvaluator = evaluate_loaded_text_model,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = str(device)
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.evaluator = evaluator

    def evaluate(
        self,
        runtime: Any,
        suite: BoundEvaluationSuite,
        output_path: Path,
        *,
        label: str,
        child_unique_parameters: int | None = None,
    ) -> dict[str, Any]:
        model = runtime if isinstance(runtime, nn.Module) else _GuardedEvaluationAdapter(runtime)
        report = self.evaluator(
            model,
            self.tokenizer,
            output_path,
            model_label=str(label),
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            eval_suite={
                "source": suite.evaluation_provenance.suite_source,
                **dict(suite.suite),
            },
            max_text_evals=len(suite.suite["text_evals"]),
            max_choice_probes=len(suite.suite["choice_probes"]),
            max_generation_prompts=len(suite.suite["generation_prompts"]),
            max_behavior_probes=len(suite.suite["behavior_probes"]),
            min_continuation_chars=production_runner.GENERATION_HEALTH_CONTRACT[
                "min_continuation_chars"
            ],
            min_unique_token_ratio=production_runner.GENERATION_HEALTH_CONTRACT[
                "min_unique_token_ratio"
            ],
            identity_binding=(
                suite.evaluation_provenance.evaluation_identity_binding()
            ),
        )
        if not isinstance(report, Mapping):
            raise QwenUnifiedSurgeryRunnerError("Suite evaluator returned no report.")
        result = dict(report)
        if child_unique_parameters is not None:
            result["param_count"] = _nonnegative_int(
                child_unique_parameters, "child_unique_parameters"
            )
        contract = (result.get("generation_health") or {}).get("contract")
        if contract != production_runner.GENERATION_HEALTH_CONTRACT:
            raise QwenUnifiedSurgeryRunnerError(
                "Evaluation report lacks the strict generation-health contract."
            )
        verification = assert_evaluation_report_count_contract(
            suite.evaluation_provenance,
            result,
        )
        result["evaluation_path_provenance"] = {
            "aggregate_sha256": suite.evaluation_provenance.aggregate_sha256,
            "scored_items_sha256": suite.evaluation_provenance.scored_items_sha256,
            "expected_total_metric_items": verification.expected_total,
            "observed_total_metric_items": verification.observed_total,
            "ordered_ids_match": verification.ordered_ids_match,
            "identity_binding_match": verification.identity_binding_match,
            "suite_content_match": verification.suite_content_match,
            "choice_scoring_sequences": verification.choice_scoring_sequences,
        }
        production_runner._write_json(output_path, result)
        return result


def _behavior_outputs(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "perplexity": report.get("perplexity"),
        "choice_accuracy": report.get("choice_accuracy"),
        "generation_diagnostics": report.get("generation_diagnostics"),
        "generation_health": report.get("generation_health"),
        "behavior_suite": report.get("behavior_suite"),
    }


def _behavior_score(report: Mapping[str, Any]) -> tuple[float, int]:
    counts = (
        (report.get("choice_accuracy") or {}, "correct"),
        (report.get("generation_health") or {}, "correct"),
        (report.get("behavior_suite") or {}, "correct"),
    )
    correct = 0
    total = 0
    for payload, key in counts:
        value = payload.get(key)
        size = payload.get("total")
        if isinstance(value, bool) or not isinstance(value, int):
            raise QwenUnifiedSurgeryRunnerError("Evaluation correct count is invalid.")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise QwenUnifiedSurgeryRunnerError("Evaluation sample count is invalid.")
        if not 0 <= value <= size:
            raise QwenUnifiedSurgeryRunnerError("Evaluation correct count exceeds total.")
        correct += value
        total += size
    if total <= 0:
        raise QwenUnifiedSurgeryRunnerError(
            "Behavioral evaluation needs at least one scored item."
        )
    return correct / total, total


def _assert_nonvacuous_baseline(report: Mapping[str, Any]) -> None:
    """Require every retained-parent domain to contain a real passing anchor."""

    domains = (
        ("behavior", report.get("behavior_suite") or {}),
        ("choice", report.get("choice_accuracy") or {}),
        ("generation", report.get("generation_health") or {}),
    )
    for name, payload in domains:
        if not isinstance(payload, Mapping):
            raise QwenUnifiedSurgeryRunnerError(
                f"Absolute retention baseline has no {name} domain."
            )
        correct = payload.get("correct")
        total = payload.get("total")
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
            or correct <= 0
            or correct > total
        ):
            raise QwenUnifiedSurgeryRunnerError(
                f"Absolute retention baseline is vacuous or unhealthy for {name}."
            )


class CachedParentSuite:
    """Evaluate each exact parent state once and bind its cached outputs."""

    def __init__(
        self,
        *,
        suite: BoundEvaluationSuite,
        executor: SuiteExecutor,
        output_directory: Path,
    ) -> None:
        self.suite = suite
        self.executor = executor
        self.output_directory = output_directory
        self.parent_state_sha256: str | None = None
        self.parent_report: dict[str, Any] | None = None
        self.parent_outputs_sha256: str | None = None
        self.parent_evaluation_fingerprint: str | None = None

    def _path(self, parent_state_sha256: str) -> Path:
        return self.output_directory / f"parent-{parent_state_sha256}.json"

    def store_parent(
        self,
        parent_state_sha256: str,
        report: Mapping[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        outputs_sha256 = production_runner._canonical_sha256(
            _behavior_outputs(report)
        )
        fingerprint = production_runner._canonical_sha256(
            {
                "role": f"{self.suite.name}-parent",
                "parent_state_sha256": parent_state_sha256,
                "suite_sha256": self.suite.suite_sha256,
                "outputs_sha256": outputs_sha256,
            }
        )
        envelope = {
            "schema": "neural-forge-cached-parent-suite-v1",
            "suite_name": self.suite.name,
            "suite_sha256": self.suite.suite_sha256,
            "parent_state_sha256": parent_state_sha256,
            "parent_outputs_sha256": outputs_sha256,
            "parent_evaluation_fingerprint": fingerprint,
            "source": str(source),
            "report": dict(report),
        }
        production_runner._write_json(self._path(parent_state_sha256), envelope)
        self.parent_state_sha256 = parent_state_sha256
        self.parent_report = dict(report)
        self.parent_outputs_sha256 = outputs_sha256
        self.parent_evaluation_fingerprint = fingerprint
        return envelope

    def bind_parent(
        self,
        runtime: nn.Module,
        parent_state_sha256: str,
        *,
        child_unique_parameters: int,
    ) -> dict[str, Any]:
        if self.parent_state_sha256 == parent_state_sha256:
            assert self.parent_report is not None
            return self.parent_report
        path = self._path(parent_state_sha256)
        if path.is_file():
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise QwenUnifiedSurgeryRunnerError(
                    "Cached parent evaluation is unreadable."
                ) from exc
            if (
                envelope.get("suite_sha256") != self.suite.suite_sha256
                or envelope.get("parent_state_sha256") != parent_state_sha256
                or not isinstance(envelope.get("report"), Mapping)
            ):
                raise QwenUnifiedSurgeryRunnerError(
                    "Cached parent evaluation binding does not match this run."
                )
            report = dict(envelope["report"])
            if envelope.get("parent_outputs_sha256") != production_runner._canonical_sha256(
                _behavior_outputs(report)
            ):
                raise QwenUnifiedSurgeryRunnerError(
                    "Cached parent outputs hash is invalid."
                )
            self.parent_state_sha256 = parent_state_sha256
            self.parent_report = report
            self.parent_outputs_sha256 = str(envelope["parent_outputs_sha256"])
            self.parent_evaluation_fingerprint = str(
                envelope["parent_evaluation_fingerprint"]
            )
            return report
        raw_path = self.output_directory / f".parent-{parent_state_sha256}.raw.json"
        report = self.executor.evaluate(
            runtime,
            self.suite,
            raw_path,
            label=f"{self.suite.name}:parent:{parent_state_sha256}",
            child_unique_parameters=child_unique_parameters,
        )
        self.store_parent(
            parent_state_sha256,
            report,
            source="evaluated-current-parent-once",
        )
        raw_path.unlink(missing_ok=True)
        return report


@dataclass(frozen=True, slots=True)
class RetainedParentFloors:
    """Minimum retained-parent pass ratios for the three behavioral domains."""

    behavior: float = 0.90
    choice: float = 0.90
    generation: float = 0.90

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise QwenUnifiedSurgeryRunnerError(
                    f"{name} retained-parent floor must be in [0, 1]."
                )
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class ParentRelativeBehavioralGate:
    """Cached-parent-relative transaction gate compatible with allocator audits."""

    def __init__(
        self,
        *,
        parent_cache: CachedParentSuite,
        retention_floors: RetainedParentFloors,
        hard_loss_limits: tuple[int | None, int | None, int | None],
        output_directory: Path,
    ) -> None:
        self.parent_cache = parent_cache
        self.retention_floors = retention_floors
        self.hard_loss_limits = hard_loss_limits
        self.output_directory = output_directory
        self.last_candidate_report: dict[str, Any] | None = None
        self.last_metrics: dict[str, Any] | None = None
        self.last_artifact: Path | None = None
        self.baseline_path = self.output_directory / "absolute_retention_baseline.json"
        self.baseline_state_sha256: str | None = None
        self.baseline_report: dict[str, Any] | None = None
        if self.baseline_path.is_file():
            payload = _read_json_mapping(
                self.baseline_path, "Absolute retention baseline"
            )
            if (
                payload.get("suite_sha256") != self.parent_cache.suite.suite_sha256
                or not isinstance(payload.get("report"), Mapping)
                or payload.get("outputs_sha256")
                != _canonical_sha256(_behavior_outputs(payload.get("report") or {}))
            ):
                raise QwenUnifiedSurgeryRunnerError(
                    "Absolute retention baseline does not match the gate suite."
                )
            self.baseline_state_sha256 = str(payload["state_sha256"])
            self.baseline_report = dict(payload["report"])
            _assert_nonvacuous_baseline(self.baseline_report)

    def bind_parent(
        self,
        runtime: nn.Module,
        parent_state_sha256: str,
        accounting: Qwen3UnifiedSurgeryAccounting,
    ) -> dict[str, Any]:
        report = self.parent_cache.bind_parent(
            runtime,
            parent_state_sha256,
            child_unique_parameters=accounting.child_unique_parameters,
        )
        if self.baseline_report is None:
            _assert_nonvacuous_baseline(report)
            self.baseline_state_sha256 = parent_state_sha256
            self.baseline_report = dict(report)
            production_runner._write_json(
                self.baseline_path,
                {
                    "schema": "neural-forge-absolute-retention-baseline-v1",
                    "suite_sha256": self.parent_cache.suite.suite_sha256,
                    "state_sha256": parent_state_sha256,
                    "outputs_sha256": production_runner._canonical_sha256(
                        _behavior_outputs(report)
                    ),
                    "report": report,
                },
            )
        return report

    def __call__(
        self, context: Qwen3UnifiedSurgeryEvaluationContext
    ) -> EvaluationDecision:
        if self.parent_cache.parent_state_sha256 != context.parent_state_sha256:
            raise QwenUnifiedSurgeryRunnerError(
                "Transaction gate is not bound to the exact current parent."
            )
        parent = self.parent_cache.parent_report
        parent_outputs = self.parent_cache.parent_outputs_sha256
        if parent is None or parent_outputs is None:
            raise QwenUnifiedSurgeryRunnerError("Transaction parent cache is empty.")
        if self.baseline_report is None or self.baseline_state_sha256 is None:
            raise QwenUnifiedSurgeryRunnerError(
                "Transaction gate absolute retention baseline is empty."
            )
        candidate_path = self.output_directory / (
            f"candidate-{context.transaction_index:06d}-"
            f"{context.proposed_state_sha256}.json"
        )
        candidate = self.parent_cache.executor.evaluate(
            context.runtime,
            self.parent_cache.suite,
            candidate_path,
            label=f"unified-surgery:candidate:{context.proposed_state_sha256}",
            child_unique_parameters=context.proposed_accounting.child_unique_parameters,
        )
        suite = self.parent_cache.suite
        baseline_domain_counts = (
            int((self.baseline_report.get("behavior_suite") or {}).get("correct", 0)),
            int((self.baseline_report.get("choice_accuracy") or {}).get("correct", 0)),
            int((self.baseline_report.get("generation_health") or {}).get("correct", 0)),
        )
        floor_values = (
            self.retention_floors.behavior,
            self.retention_floors.choice,
            self.retention_floors.generation,
        )
        floor_loss_limits = tuple(
            int(math.floor(count * (1.0 - floor) + 1e-12))
            for count, floor in zip(baseline_domain_counts, floor_values, strict=True)
        )
        effective_limits = tuple(
            floor_limit if hard is None else min(floor_limit, hard)
            for floor_limit, hard in zip(
                floor_loss_limits, self.hard_loss_limits, strict=True
            )
        )
        thresholds = production_runner.GateThresholds(
            max_behavior_losses=effective_limits[0],
            max_choice_losses=effective_limits[1],
            max_generation_losses=effective_limits[2],
        )
        current_parent_verdict = production_runner.score_cached_parent_gate(
            parent,
            candidate,
            behavior_ids=suite.behavior_ids,
            choice_ids=suite.choice_ids,
            generation_ids=suite.generation_ids,
            thresholds=thresholds,
        )
        verdict = production_runner.score_cached_parent_gate(
            self.baseline_report,
            candidate,
            behavior_ids=suite.behavior_ids,
            choice_ids=suite.choice_ids,
            generation_ids=suite.generation_ids,
            thresholds=thresholds,
        )
        retained_parent_ratios = {
            name: (
                1.0
                if int(verdict["domains"][name]["parent_correct"]) == 0
                else float(verdict["domains"][name]["retained_parent_passes"])
                / int(verdict["domains"][name]["parent_correct"])
            )
            for name in ("behavior", "choice", "generation")
        }
        floor_map = self.retention_floors.as_dict()
        floor_passed = all(
            retained_parent_ratios[name] + 1e-12 >= floor_map[name]
            for name in retained_parent_ratios
        )
        verdict["retained_parent_ratios"] = retained_parent_ratios
        verdict["absolute_baseline_state_sha256"] = self.baseline_state_sha256
        verdict["current_parent_comparison"] = current_parent_verdict
        verdict["retained_parent_floors"] = floor_map
        verdict["floor_loss_limits"] = {
            name: value
            for name, value in zip(
                ("behavior", "choice", "generation"),
                floor_loss_limits,
                strict=True,
            )
        }
        verdict["effective_loss_limits"] = thresholds.as_dict()
        verdict["retained_parent_floors_passed"] = floor_passed
        verdict["perplexity_used_for_acceptance"] = False
        current_score_for_pareto, _ = _behavior_score(parent)
        candidate_score_for_pareto, _ = _behavior_score(candidate)
        pareto_nonworsening = candidate_score_for_pareto + 1e-12 >= current_score_for_pareto
        verdict["current_parent_pareto_nonworsening"] = pareto_nonworsening
        verdict["pareto_tradeoff_allowed_above_absolute_floors"] = True
        verdict["passed"] = bool(verdict["passed"] and floor_passed)
        parent_score, parent_count = _behavior_score(parent)
        candidate_score, candidate_count = _behavior_score(candidate)
        if parent_count != candidate_count or parent_count != suite.behavioral_sample_count:
            raise QwenUnifiedSurgeryRunnerError(
                "Transaction gate sample cardinality changed."
            )
        candidate_outputs = production_runner._canonical_sha256(
            _behavior_outputs(candidate)
        )
        metrics: dict[str, Any] = {
            "parent_relative": True,
            "evaluation_kind": surgery_allocator._BEHAVIORAL_GATE_KIND,
            "parent_state_sha256": context.parent_state_sha256,
            "candidate_state_sha256": context.proposed_state_sha256,
            "evaluation_suite_sha256": suite.suite_sha256,
            "parent_outputs_sha256": parent_outputs,
            "candidate_outputs_sha256": candidate_outputs,
            "behavioral_sample_count": parent_count,
            "parent_behavior_score": parent_score,
            "candidate_behavior_score": candidate_score,
        }
        metrics["evaluation_fingerprint"] = (
            surgery_allocator._behavioral_evaluation_fingerprint(metrics)
        )
        audit_path = candidate_path.with_name(f"{candidate_path.stem}.gate.json")
        production_runner._write_json(
            audit_path,
            {
                "context": {
                    "transaction_index": context.transaction_index,
                    "parent_state_sha256": context.parent_state_sha256,
                    "candidate_state_sha256": context.proposed_state_sha256,
                    "batch_sha256": context.batch.batch_sha256,
                },
                "metrics": metrics,
                "verdict": verdict,
                "candidate_report": str(candidate_path),
                "candidate_report_sha256": production_runner._file_sha256(
                    candidate_path
                ),
            },
        )
        self.last_candidate_report = candidate
        self.last_metrics = metrics
        self.last_artifact = audit_path
        if verdict["passed"]:
            return EvaluationDecision.accept("cached parent-relative gate passed", metrics)
        return EvaluationDecision.reject("cached parent-relative gate rejected", metrics)

    def promote_accepted_candidate(self, committed_state_sha256: str) -> None:
        if self.last_candidate_report is None:
            raise QwenUnifiedSurgeryRunnerError(
                "Cannot promote a missing accepted candidate report."
            )
        self.parent_cache.store_parent(
            committed_state_sha256,
            self.last_candidate_report,
            source="promoted-accepted-candidate-no-parent-recompute",
        )


def _single_proposal_batch(
    allocator: Qwen3UnifiedSurgeryAllocator,
    proposal: Qwen3UnifiedSurgeryProposal,
) -> Qwen3MixedSurgeryBatch:
    return Qwen3MixedSurgeryBatch(
        proposals=(proposal,),
        parent_state_sha256=allocator._state_sha256(),
        requested_physical_bytes=proposal.individual_physical_bytes_saved,
        prototype_bytes_saved=proposal.prototype_bytes_saved,
        scale_setup_bytes=proposal.scale_setup_bytes,
        physical_bytes_saved=proposal.individual_physical_bytes_saved,
        total_risk=proposal.risk,
        proposal_work=1,
        knapsack_states_considered=1,
    )


class IndependentCandidateValidator:
    """Measure each candidate on an independent suite, then force rollback."""

    def __init__(
        self,
        *,
        allocator: Qwen3UnifiedSurgeryAllocator,
        evidence: Callable[[], Qwen3SensitivityEvidence],
        parent_cache: CachedParentSuite,
        output_directory: Path,
        max_candidates: int,
        discovery_row_fingerprints: Sequence[str],
    ) -> None:
        self.allocator = allocator
        self.evidence = evidence
        self.parent_cache = parent_cache
        self.output_directory = output_directory
        self.max_candidates = _positive_int(max_candidates, "max_candidates")
        self.discovery_row_fingerprints = tuple(
            str(value) for value in discovery_row_fingerprints
        )
        if (
            not self.discovery_row_fingerprints
            or len(set(self.discovery_row_fingerprints))
            != len(self.discovery_row_fingerprints)
            or any(len(value) != 64 for value in self.discovery_row_fingerprints)
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Candidate validation requires unique discovery path SHA-256 values."
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.round_index = max(
            (
                int(path.name.removeprefix("round-"))
                for path in self.output_directory.glob("round-[0-9][0-9][0-9][0-9][0-9][0-9]")
                if path.is_dir() and path.name.removeprefix("round-").isdigit()
            ),
            default=0,
        )

    def _reserve_round(self) -> tuple[int, Path]:
        index = self.round_index
        while True:
            index += 1
            round_dir = self.output_directory / f"round-{index:06d}"
            try:
                round_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            claim = {
                "schema": "neural-forge-independent-candidate-round-v1",
                "round_index": index,
                "directory": str(round_dir.resolve()),
            }
            claim["round_id"] = _canonical_sha256(claim)
            _write_json_exclusive_atomic(round_dir / "round_claim.json", claim)
            self.round_index = index
            return index, round_dir

    def validate(
        self,
        proposals: Sequence[Qwen3UnifiedSurgeryProposal],
    ) -> dict[str, Any]:
        values = tuple(proposals[: self.max_candidates])
        if not values:
            raise QwenUnifiedSurgeryRunnerError(
                "No proposals remain for independent candidate validation."
            )
        round_index, round_dir = self._reserve_round()
        parent_state = self.allocator._state_sha256()
        accounting = self.allocator.accounting()
        parent = self.parent_cache.bind_parent(
            self.allocator.runtime,
            parent_state,
            child_unique_parameters=accounting.child_unique_parameters,
        )
        parent_metric, parent_items = _behavior_score(parent)
        parent_error = 1.0 - parent_metric
        evidence = self.evidence()
        discovery_rows = self.discovery_row_fingerprints
        suite = self.parent_cache.suite
        if parent_items != suite.behavioral_sample_count:
            raise QwenUnifiedSurgeryRunnerError(
                "Candidate parent report count differs from its bound provenance."
            )
        if set(discovery_rows) & set(suite.content_fingerprints):
            raise QwenUnifiedSurgeryRunnerError(
                "Candidate validation rows overlap sensitivity evidence rows."
            )
        records: list[Qwen3CandidateValidationDamage] = []
        audits: list[dict[str, Any]] = []
        for index, proposal in enumerate(values, start=1):
            captured: dict[str, Any] = {}

            def measure(context: Qwen3UnifiedSurgeryEvaluationContext) -> EvaluationDecision:
                candidate_path = round_dir / (
                    f"candidate-{index:04d}-{proposal.kind}-"
                    f"{proposal.candidate.candidate_sha256}.json"
                )
                candidate = self.parent_cache.executor.evaluate(
                    context.runtime,
                    suite,
                    candidate_path,
                    label=(
                        "unified-surgery:candidate-marginal:"
                        f"{proposal.candidate.candidate_sha256}"
                    ),
                    child_unique_parameters=(
                        context.proposed_accounting.child_unique_parameters
                    ),
                )
                candidate_metric, candidate_items = _behavior_score(candidate)
                if candidate_items != suite.behavioral_sample_count:
                    raise QwenUnifiedSurgeryRunnerError(
                        "Candidate report count differs from its bound provenance."
                    )
                candidate_error = 1.0 - candidate_metric
                outputs_sha256 = production_runner._canonical_sha256(
                    _behavior_outputs(candidate)
                )
                evaluation_fingerprint = production_runner._canonical_sha256(
                    {
                        "role": "independent-candidate-marginal",
                        "candidate_sha256": proposal.candidate.candidate_sha256,
                        "parent_state_sha256": parent_state,
                        "suite_sha256": suite.suite_sha256,
                        "outputs_sha256": outputs_sha256,
                    }
                )
                captured.update(
                    {
                        "candidate_error": candidate_error,
                        "candidate_behavioral_sample_count": candidate_items,
                        "candidate_outputs_sha256": outputs_sha256,
                        "candidate_evaluation_fingerprint": evaluation_fingerprint,
                        "artifact": str(candidate_path),
                        "artifact_sha256": production_runner._file_sha256(candidate_path),
                    }
                )
                return EvaluationDecision.reject(
                    "measurement-only candidate probe; force exact rollback",
                    {
                        "candidate_marginal_measurement": True,
                        "parent_state_sha256": parent_state,
                        "candidate_sha256": proposal.candidate.candidate_sha256,
                        "validation_suite_sha256": suite.suite_sha256,
                    },
                )

            transaction = self.allocator.apply_batch(
                _single_proposal_batch(self.allocator, proposal),
                evaluator=measure,
                unsafe_allow_unvalidated_candidates=True,
            )
            if transaction.accepted or transaction.rollback_exact is not True:
                raise QwenUnifiedSurgeryRunnerError(
                    "Candidate-marginal probe was not exactly rolled back."
                )
            if self.allocator._state_sha256() != parent_state:
                raise QwenUnifiedSurgeryRunnerError(
                    "Candidate-marginal probe changed the current parent state."
                )
            candidate_error = float(captured["candidate_error"])
            marginal_damage = max(0.0, candidate_error - parent_error)
            parent_fingerprint = production_runner._canonical_sha256(
                {
                    "role": "independent-candidate-marginal-parent",
                    "candidate_sha256": proposal.candidate.candidate_sha256,
                    "parent_state_sha256": parent_state,
                    "suite_sha256": suite.suite_sha256,
                    "outputs_sha256": self.parent_cache.parent_outputs_sha256,
                }
            )
            record = Qwen3CandidateValidationDamage(
                kind=proposal.kind,
                candidate_sha256=proposal.candidate.candidate_sha256,
                parent_state_sha256=parent_state,
                discovery_fingerprint=evidence.fingerprints.cache,
                validation_fingerprint=suite.validation_fingerprint,
                discovery_row_fingerprints=discovery_rows,
                validation_row_fingerprints=suite.behavioral_item_fingerprints,
                marginal_damage=marginal_damage,
                metric_name="independent_behavioral_failure_rate",
                parent_metric_value=parent_error,
                candidate_metric_value=candidate_error,
                metric_direction="lower_is_better",
                parent_evaluation_fingerprint=parent_fingerprint,
                candidate_evaluation_fingerprint=str(
                    captured["candidate_evaluation_fingerprint"]
                ),
                validation_sample_count=int(
                    captured["candidate_behavioral_sample_count"]
                ),
            )
            records.append(record)
            audits.append(
                {
                    "proposal_sha256": proposal.proposal_sha256,
                    "candidate_sha256": proposal.candidate.candidate_sha256,
                    "kind": proposal.kind,
                    "marginal_damage": marginal_damage,
                    "rollback_exact": True,
                    **captured,
                }
            )
        self.allocator.install_candidate_validation(records)
        summary = {
            "round_index": round_index,
            "round_id": _read_json_mapping(
                round_dir / "round_claim.json", "Candidate-validation round claim"
            )["round_id"],
            "parent_state_sha256": parent_state,
            "discovery_fingerprint": evidence.fingerprints.cache,
            "validation_fingerprint": suite.validation_fingerprint,
            "validated_candidate_count": len(records),
            "records": [item.as_dict() for item in records],
            "audits": audits,
            "every_probe_rolled_back_exactly": True,
            "parent_forward_evaluations_for_round": 1,
        }
        summary_path = round_dir / "validation_summary.json"
        production_runner._write_json(summary_path, summary)
        return {
            **summary,
            "summary_artifact": str(summary_path.resolve()),
            "summary_artifact_sha256": production_runner._file_sha256(summary_path),
        }


@dataclass(frozen=True, slots=True)
class JournalReplay:
    initial: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    artifact_verified_records: tuple[dict[str, Any], ...]
    selected: dict[str, Any] | None
    fallback_reason: str | None

    @property
    def accepted_transaction_total(self) -> int:
        if self.selected is None:
            return 0
        return int(self.selected["accepted_transaction_total"])

    @property
    def model_content_sha256(self) -> str:
        source = self.initial if self.selected is None else self.selected
        return str(source["model_content_sha256"])

    @property
    def commit_seq(self) -> int:
        return 0 if self.selected is None else int(self.selected["seq"])


class AcceptedCommitJournal:
    """Immutable accepted-commit records with verified branch replay."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory.resolve()
        self.root = self.output_directory / "accepted_commit_journal"
        self.root.mkdir(parents=True, exist_ok=True)
        self.initial_path = self.output_directory / "initial_incumbent.json"

    @staticmethod
    def _commit_path(root: Path, seq: int) -> Path:
        return root / f"commit-{seq:06d}.json"

    def record_or_verify_initial(
        self,
        *,
        model_content: Mapping[str, Any],
        accounting: Qwen3UnifiedSurgeryAccounting,
        run_contract: Mapping[str, Any],
        source_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        contract = dict(run_contract)
        _assert_run_contract_integrity(contract)
        contract_sha256 = str(contract.get("run_contract_sha256") or "")
        if not contract_sha256:
            raise QwenUnifiedSurgeryRunnerError(
                "Initial incumbent requires a complete run contract."
            )
        payload: dict[str, Any] = {
            "schema": INITIAL_INCUMBENT_SCHEMA,
            "model_content_sha256": str(model_content["model_content_sha256"]),
            "model_content": dict(model_content),
            "accounting": accounting.as_dict(),
            "run_contract_sha256": contract_sha256,
            "run_contract": contract,
            "source_identity": dict(source_identity),
        }
        payload["record_sha256"] = _record_sha256(payload, "record_sha256")
        if self.initial_path.is_file():
            existing = self.load_initial()
            if existing != payload:
                raise QwenUnifiedSurgeryRunnerError(
                    "Initial incumbent differs from the immutable run record."
                )
            return existing
        _write_json_exclusive_atomic(self.initial_path, payload)
        return payload

    def load_initial(self) -> dict[str, Any]:
        if not self.initial_path.is_file():
            raise QwenUnifiedSurgeryRunnerError(
                "Resume requires an immutable initial-incumbent record."
            )
        payload = _read_json_mapping(
            self.initial_path, "Initial incumbent record"
        )
        if (
            payload.get("schema") != INITIAL_INCUMBENT_SCHEMA
            or payload.get("record_sha256")
            != _record_sha256(payload, "record_sha256")
            or not isinstance(payload.get("model_content"), Mapping)
            or payload.get("model_content_sha256")
            != payload["model_content"].get("model_content_sha256")
            or not isinstance(payload.get("accounting"), Mapping)
            or not isinstance(payload.get("run_contract"), Mapping)
            or payload.get("run_contract_sha256")
            != payload["run_contract"].get("run_contract_sha256")
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Initial incumbent record failed integrity verification."
            )
        _assert_run_contract_integrity(payload["run_contract"])
        return payload

    def _entry_paths(self) -> tuple[Path, ...]:
        values = sorted(self.root.glob("commit-[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        expected = list(range(1, len(values) + 1))
        actual = [int(path.stem.removeprefix("commit-")) for path in values]
        if actual != expected:
            raise QwenUnifiedSurgeryRunnerError(
                "Accepted-commit journal sequence is not contiguous."
            )
        return tuple(values)

    def _bound_artifact_is_verified(
        self,
        path_value: Any,
        sha256_value: Any,
    ) -> bool:
        try:
            path = Path(str(path_value)).resolve()
            path.relative_to(self.output_directory)
        except (OSError, ValueError):
            return False
        return (
            path.is_file()
            and isinstance(sha256_value, str)
            and production_runner._file_sha256(path) == sha256_value
        )

    def _artifact_is_verified(
        self,
        record: Mapping[str, Any],
        *,
        candidate_validation_mode: CandidateValidationMode,
    ) -> bool:
        checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
        if not checkpoint.is_dir():
            return False
        try:
            tree = production_runner.artifact_tree_report(checkpoint)
        except Exception:
            return False
        if not (
            tree.get("tree_sha256") == record.get("artifact_tree_sha256")
            and int(tree.get("total_bytes", -1))
            == int(record.get("artifact_total_bytes", -2))
        ):
            return False
        if not self._bound_artifact_is_verified(
            record.get("gate_artifact"), record.get("gate_artifact_sha256")
        ):
            return False
        transaction_payload = record.get("transaction_payload")
        if (
            not isinstance(transaction_payload, Mapping)
            or transaction_payload.get("accepted") is not True
            or record.get("transaction_sha256")
            != _canonical_sha256(dict(transaction_payload))
        ):
            return False
        validation_path = record.get("candidate_validation_artifact")
        validation_sha256 = record.get("candidate_validation_artifact_sha256")
        validation_round_id = record.get("candidate_validation_round_id")
        if candidate_validation_mode == "provisional":
            return (
                validation_path is None
                and validation_sha256 is None
                and validation_round_id is None
            )
        if not self._bound_artifact_is_verified(validation_path, validation_sha256):
            return False
        try:
            validation = _read_json_mapping(
                Path(str(validation_path)).resolve(),
                "Candidate-validation summary",
            )
        except QwenUnifiedSurgeryRunnerError:
            return False
        if (
            validation.get("round_id") != validation_round_id
        ):
            return False
        return True

    def replay(
        self,
        *,
        expected_run_contract_sha256: str | None = None,
    ) -> JournalReplay:
        initial = self.load_initial()
        candidate_validation_mode = initial["run_contract"].get(
            "candidate_validation_mode"
        )
        if candidate_validation_mode not in {"independent", "provisional"}:
            raise QwenUnifiedSurgeryRunnerError(
                "Run contract lacks a supported candidate-validation mode."
            )
        if (
            expected_run_contract_sha256 is not None
            and initial.get("run_contract_sha256")
            != expected_run_contract_sha256
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Run policy differs from the immutable initial contract."
            )
        by_seq: dict[int, Mapping[str, Any]] = {0: initial}
        records: list[dict[str, Any]] = []
        verified: list[dict[str, Any]] = []
        for path in self._entry_paths():
            record = _read_json_mapping(path, "Accepted-commit journal record")
            seq = int(path.stem.removeprefix("commit-"))
            parent_seq = record.get("parent_commit_seq")
            if (
                record.get("schema") != ACCEPTED_COMMIT_SCHEMA
                or record.get("seq") != seq
                or record.get("record_sha256")
                != _record_sha256(record, "record_sha256")
                or isinstance(parent_seq, bool)
                or not isinstance(parent_seq, int)
                or parent_seq < 0
                or parent_seq >= seq
                or parent_seq not in by_seq
            ):
                raise QwenUnifiedSurgeryRunnerError(
                    f"Accepted-commit journal record {seq} failed integrity."
                )
            parent = by_seq[parent_seq]
            parent_total = int(parent.get("accepted_transaction_total", 0))
            if (
                record.get("parent_model_content_sha256")
                != parent.get("model_content_sha256")
                or record.get("accepted_transaction_total") != parent_total + 1
                or record.get("run_contract_sha256")
                != initial.get("run_contract_sha256")
                or record.get("candidate_validation_mode")
                != candidate_validation_mode
                or (
                    expected_run_contract_sha256 is not None
                    and record.get("run_contract_sha256")
                    != expected_run_contract_sha256
                )
                or not isinstance(record.get("accounting"), Mapping)
                or record.get("artifact_total_bytes")
                != record["accounting"].get("deployable_bytes")
                or not isinstance(record.get("evidence_lineage"), Mapping)
                or record["evidence_lineage"].get("lineage_sha256")
                != _record_sha256(
                    record["evidence_lineage"], "lineage_sha256"
                )
            ):
                raise QwenUnifiedSurgeryRunnerError(
                    f"Accepted-commit journal chain {seq} is invalid."
                )
            by_seq[seq] = record
            records.append(record)
            if self._artifact_is_verified(
                record,
                candidate_validation_mode=candidate_validation_mode,
            ):
                verified.append(record)
        selected = max(verified, key=lambda item: int(item["seq"]), default=None)
        fallback_reason = None
        if records and (selected is None or selected["seq"] != records[-1]["seq"]):
            fallback_reason = "newer journal checkpoint missing or artifact-corrupt"
        return JournalReplay(
            initial=initial,
            records=tuple(records),
            artifact_verified_records=tuple(verified),
            selected=selected,
            fallback_reason=fallback_reason,
        )

    def append(
        self,
        *,
        parent_commit_seq: int,
        parent_model_content_sha256: str,
        accepted_transaction_total: int,
        model_content_sha256: str,
        checkpoint: Path,
        artifact_tree: Mapping[str, Any],
        accounting: Qwen3UnifiedSurgeryAccounting,
        run_contract_sha256: str,
        transaction_payload: Mapping[str, Any],
        gate_artifact: Path,
        candidate_validation_artifact: Path | None,
        candidate_validation_round_id: str | None,
        evidence_lineage: Mapping[str, Any],
        behavior_score: float,
    ) -> dict[str, Any]:
        initial = self.load_initial()
        if initial.get("run_contract_sha256") != run_contract_sha256:
            raise QwenUnifiedSurgeryRunnerError(
                "Cannot append a commit under a different run contract."
            )
        replay = self.replay(
            expected_run_contract_sha256=run_contract_sha256
        )
        if (
            parent_commit_seq != replay.commit_seq
            or parent_model_content_sha256 != replay.model_content_sha256
            or accepted_transaction_total
            != replay.accepted_transaction_total + 1
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Accepted commit does not extend the highest verified journal state."
            )
        existing = [
            int(path.stem.removeprefix("commit-"))
            for path in self._entry_paths()
        ]
        seq = max(existing, default=0) + 1
        lineage = dict(evidence_lineage)
        if lineage.get("lineage_sha256") != _record_sha256(
            lineage, "lineage_sha256"
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Accepted evidence lineage hash is invalid."
            )
        if int(artifact_tree["total_bytes"]) != accounting.deployable_bytes:
            raise QwenUnifiedSurgeryRunnerError(
                "Accepted checkpoint tree differs from canonical artifact accounting."
            )
        candidate_validation_mode = initial["run_contract"].get(
            "candidate_validation_mode"
        )
        if candidate_validation_mode not in {"independent", "provisional"}:
            raise QwenUnifiedSurgeryRunnerError(
                "Run contract lacks a supported candidate-validation mode."
            )
        transaction = dict(transaction_payload)
        if transaction.get("accepted") is not True:
            raise QwenUnifiedSurgeryRunnerError(
                "Accepted journal commits require an accepted transaction payload."
            )
        transaction_sha256 = _canonical_sha256(transaction)
        resolved_gate_artifact = gate_artifact.resolve()
        try:
            resolved_gate_artifact.relative_to(self.output_directory)
        except ValueError as exc:
            raise QwenUnifiedSurgeryRunnerError(
                "Transaction gate artifact escapes the run output."
            ) from exc
        if not resolved_gate_artifact.is_file():
            raise QwenUnifiedSurgeryRunnerError(
                "Accepted transaction gate artifact is missing."
            )
        validation_path: Path | None = None
        validation_sha256: str | None = None
        validation_round: str | None = None
        if candidate_validation_mode == "independent":
            if candidate_validation_artifact is None or not candidate_validation_round_id:
                raise QwenUnifiedSurgeryRunnerError(
                    "Independent commits require a candidate-validation summary."
                )
            validation_path = candidate_validation_artifact.resolve()
            if not validation_path.is_file():
                raise QwenUnifiedSurgeryRunnerError(
                    "Candidate-validation summary is missing."
                )
            try:
                validation_path.relative_to(self.output_directory)
            except ValueError as exc:
                raise QwenUnifiedSurgeryRunnerError(
                    "Candidate-validation summary escapes the run output."
                ) from exc
            validation = _read_json_mapping(
                validation_path, "Candidate-validation summary"
            )
            if validation.get("round_id") != candidate_validation_round_id:
                raise QwenUnifiedSurgeryRunnerError(
                    "Candidate-validation round binding changed before commit."
                )
            validation_sha256 = production_runner._file_sha256(validation_path)
            validation_round = str(candidate_validation_round_id)
        elif (
            candidate_validation_artifact is not None
            or candidate_validation_round_id is not None
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Provisional commits cannot attach independent-validation evidence."
            )
        payload: dict[str, Any] = {
            "schema": ACCEPTED_COMMIT_SCHEMA,
            "journal_schema": ACCEPTED_COMMIT_JOURNAL_SCHEMA,
            "seq": seq,
            "parent_commit_seq": _nonnegative_int(
                parent_commit_seq, "parent_commit_seq"
            ),
            "parent_model_content_sha256": str(parent_model_content_sha256),
            "accepted_transaction_total": _positive_int(
                accepted_transaction_total, "accepted_transaction_total"
            ),
            "model_content_sha256": str(model_content_sha256),
            "checkpoint": str(checkpoint.resolve()),
            "artifact_tree_sha256": str(artifact_tree["tree_sha256"]),
            "artifact_total_bytes": int(artifact_tree["total_bytes"]),
            "accounting": accounting.as_dict(),
            "run_contract_sha256": str(run_contract_sha256),
            "candidate_validation_mode": candidate_validation_mode,
            "transaction_payload": transaction,
            "transaction_sha256": transaction_sha256,
            "gate_artifact": str(resolved_gate_artifact),
            "gate_artifact_sha256": production_runner._file_sha256(
                resolved_gate_artifact
            ),
            "candidate_validation_artifact": (
                None if validation_path is None else str(validation_path)
            ),
            "candidate_validation_artifact_sha256": validation_sha256,
            "candidate_validation_round_id": validation_round,
            "evidence_lineage": lineage,
            "behavior_score": float(behavior_score),
        }
        payload["record_sha256"] = _record_sha256(payload, "record_sha256")
        path = self._commit_path(self.root, seq)
        _write_json_exclusive_atomic(path, payload)
        return payload


@dataclass(frozen=True, slots=True)
class FrontierRecord:
    accepted_transaction_total: int
    state_sha256: str
    model_content_sha256: str
    deployable_bytes: int
    behavior_score: float
    checkpoint: str
    artifact_tree_sha256: str
    artifact_total_bytes: int
    transaction_sha256: str | None
    reload_verified: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


class ParetoCheckpointManager:
    """Keep a bounded, hashed frontier of standalone compact checkpoints."""

    def __init__(
        self,
        *,
        output_directory: Path,
        tokenizer: Any,
        tokenizer_name: str,
        max_checkpoints: int,
        verify_reload: bool = True,
    ) -> None:
        self.output_directory = output_directory.resolve()
        self.root = (self.output_directory / "checkpoints").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.tokenizer_name = str(tokenizer_name)
        self.max_checkpoints = _positive_int(max_checkpoints, "max_checkpoints")
        self.verify_reload = bool(verify_reload)
        self.manifest_path = self.output_directory / "frontier_manifest.json"
        self.records: list[FrontierRecord] = []
        self.current_checkpoint: str | None = None
        self.evicted_total = 0
        self.manifest_validation_error: str | None = None
        self._pending_removals: list[FrontierRecord] = []
        if self.manifest_path.is_file():
            try:
                payload = _read_json_mapping(
                    self.manifest_path, "Existing frontier manifest"
                )
                declared_hash = payload.get("manifest_sha256")
                if (
                    payload.get("schema") != FRONTIER_SCHEMA
                    or declared_hash != _record_sha256(payload, "manifest_sha256")
                    or payload.get("max_checkpoints") != self.max_checkpoints
                    or not isinstance(payload.get("records"), list)
                ):
                    raise QwenUnifiedSurgeryRunnerError(
                        "Existing frontier manifest failed its immutable contract."
                    )
                records = [FrontierRecord(**item) for item in payload["records"]]
                for record in records:
                    if record.state_sha256 != record.model_content_sha256:
                        raise QwenUnifiedSurgeryRunnerError(
                            "Frontier artifact ID is not its stable model content."
                        )
                    checkpoint = Path(record.checkpoint).resolve()
                    try:
                        checkpoint.relative_to(self.root)
                    except ValueError as exc:
                        raise QwenUnifiedSurgeryRunnerError(
                            "Frontier checkpoint escapes the frontier root."
                        ) from exc
                    tree = production_runner.artifact_tree_report(checkpoint)
                    if (
                        tree["tree_sha256"] != record.artifact_tree_sha256
                        or int(tree["total_bytes"]) != record.artifact_total_bytes
                    ):
                        raise QwenUnifiedSurgeryRunnerError(
                            "Frontier checkpoint artifact tree is corrupt."
                        )
                current = payload.get("current_checkpoint")
                if current is not None and current not in {
                    item.checkpoint for item in records
                }:
                    raise QwenUnifiedSurgeryRunnerError(
                        "Frontier current checkpoint is not retained."
                    )
                self.records = records
                self.current_checkpoint = current
                self.evicted_total = int(payload.get("evicted_total", 0))
            except Exception as exc:
                self.records = []
                self.current_checkpoint = None
                self.evicted_total = 0
                self.manifest_validation_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _dominates(left: FrontierRecord, right: FrontierRecord) -> bool:
        return (
            left.deployable_bytes <= right.deployable_bytes
            and left.behavior_score >= right.behavior_score
            and (
                left.deployable_bytes < right.deployable_bytes
                or left.behavior_score > right.behavior_score
            )
        )

    def _remove_checkpoint(self, path: str) -> None:
        candidate = Path(path).resolve()
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise QwenUnifiedSurgeryRunnerError(
                "Refusing to remove a checkpoint outside the frontier root."
            ) from exc
        if not relative.parts or candidate == self.root:
            raise QwenUnifiedSurgeryRunnerError(
                "Refusing to remove the checkpoint frontier root."
            )
        if candidate.exists():
            shutil.rmtree(candidate)

    def _retain(self, newest: FrontierRecord) -> tuple[list[FrontierRecord], list[FrontierRecord]]:
        prior_current = self.current_record
        values = [
            item
            for item in [*self.records, newest]
            if not any(
                other != item and self._dominates(other, item)
                for other in [*self.records, newest]
            )
        ]
        values.sort(
            key=lambda item: (
                item.deployable_bytes,
                -item.behavior_score,
                item.accepted_transaction_total,
            )
        )
        if len(values) > self.max_checkpoints:
            priority = sorted(
                values,
                key=lambda item: (
                    item.state_sha256 != newest.state_sha256,
                    -item.behavior_score,
                    item.deployable_bytes,
                    -item.accepted_transaction_total,
                ),
            )
            kept_states = {item.state_sha256 for item in priority[: self.max_checkpoints]}
            kept = [item for item in values if item.state_sha256 in kept_states]
        else:
            kept = values
        if (
            prior_current is not None
            and prior_current not in kept
            and self.max_checkpoints >= 2
        ):
            kept = [
                *kept[: max(self.max_checkpoints - 1, 0)],
                prior_current,
            ]
            if newest not in kept:
                kept[0] = newest
        kept_states = {item.state_sha256 for item in kept}
        removed = [
            item
            for item in [*self.records, newest]
            if item.state_sha256 not in kept_states
        ]
        return kept, removed

    def _write_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": FRONTIER_SCHEMA,
            "max_checkpoints": self.max_checkpoints,
            "records": [item.as_dict() for item in self.records],
            "retained_count": len(self.records),
            "current_checkpoint": self.current_checkpoint,
            "evicted_total": self.evicted_total,
            "objectives": {
                "deployable_bytes": "minimize",
                "behavior_score": "maximize",
            },
            "bounded_retained_frontier": True,
            "full_historical_frontier_retained": self.evicted_total == 0,
        }
        payload["manifest_sha256"] = _record_sha256(payload, "manifest_sha256")
        production_runner._write_json(self.manifest_path, payload)
        return payload

    def contains_state(self, state_sha256: str) -> bool:
        return any(item.state_sha256 == state_sha256 for item in self.records)

    def contains_content(self, model_content_sha256: str) -> bool:
        return any(
            item.model_content_sha256 == model_content_sha256
            for item in self.records
        )

    @property
    def current_record(self) -> FrontierRecord | None:
        if self.current_checkpoint is None:
            return None
        return next(
            (
                item
                for item in self.records
                if item.checkpoint == self.current_checkpoint
            ),
            None,
        )

    def reconcile_from_journal(self, replay: JournalReplay) -> dict[str, Any] | None:
        """Treat verified journal checkpoints as authoritative frontier inputs."""

        if not replay.artifact_verified_records:
            if self.manifest_validation_error is not None:
                raise QwenUnifiedSurgeryRunnerError(
                    "Frontier is corrupt and no accepted journal checkpoint can rebuild it."
                )
            return None
        recovered: list[FrontierRecord] = []
        for commit in replay.artifact_verified_records:
            checkpoint = str(Path(commit["checkpoint"]).resolve())
            try:
                Path(checkpoint).relative_to(self.root)
            except ValueError:
                continue
            content = str(commit["model_content_sha256"])
            recovered.append(
                FrontierRecord(
                    accepted_transaction_total=int(
                        commit["accepted_transaction_total"]
                    ),
                    state_sha256=content,
                    model_content_sha256=content,
                    deployable_bytes=int(commit["accounting"]["deployable_bytes"]),
                    behavior_score=float(commit["behavior_score"]),
                    checkpoint=checkpoint,
                    artifact_tree_sha256=str(commit["artifact_tree_sha256"]),
                    artifact_total_bytes=int(commit["artifact_total_bytes"]),
                    transaction_sha256=str(commit["transaction_sha256"]),
                    reload_verified=True,
                )
            )
        if not recovered:
            raise QwenUnifiedSurgeryRunnerError(
                "Accepted journal checkpoints cannot rebuild this frontier root."
            )
        selected_checkpoint = (
            None
            if replay.selected is None
            else str(Path(replay.selected["checkpoint"]).resolve())
        )
        newest = next(
            (
                item
                for item in recovered
                if item.checkpoint == selected_checkpoint
            ),
            max(recovered, key=lambda item: item.accepted_transaction_total),
        )
        retained: list[FrontierRecord] = []
        for record in sorted(recovered, key=lambda item: item.accepted_transaction_total):
            values = [
                item
                for item in [*retained, record]
                if not any(
                    other != item and self._dominates(other, item)
                    for other in [*retained, record]
                )
            ]
            retained = values[-self.max_checkpoints :]
        if newest not in retained:
            prefix = (
                []
                if self.max_checkpoints == 1
                else retained[-(self.max_checkpoints - 1) :]
            )
            retained = [*prefix, newest]
        ordered_verified = sorted(
            replay.artifact_verified_records,
            key=lambda item: int(item["seq"]),
        )
        if self.max_checkpoints >= 2 and len(ordered_verified) >= 2:
            fallback_checkpoint = str(
                Path(ordered_verified[-2]["checkpoint"]).resolve()
            )
            fallback = next(
                (
                    item
                    for item in recovered
                    if item.checkpoint == fallback_checkpoint
                ),
                None,
            )
            if fallback is not None and fallback not in retained:
                retained = [
                    newest,
                    fallback,
                    *[
                        item
                        for item in retained
                        if item not in {newest, fallback}
                    ],
                ][: self.max_checkpoints]
        self.records = retained
        self.current_checkpoint = newest.checkpoint
        self.manifest_validation_error = None
        return self._write_manifest()

    def record(
        self,
        runtime: nn.Module,
        *,
        accepted_transaction_total: int,
        state_sha256: str,
        accounting: Qwen3UnifiedSurgeryAccounting,
        behavior_score: float,
        transaction_sha256: str | None,
    ) -> dict[str, Any]:
        del state_sha256  # Allocator counters/history are not checkpoint identity.
        memory_content = _model_content_identity(runtime)
        content_sha256 = str(memory_content["model_content_sha256"])
        if self.contains_content(content_sha256):
            existing = next(
                item
                for item in self.records
                if item.model_content_sha256 == content_sha256
            )
            self.current_checkpoint = existing.checkpoint
            return self._write_manifest()
        destination = self.root / (
            f"state-{accepted_transaction_total:06d}-{content_sha256[:16]}"
        )
        if destination.exists():
            raise QwenUnifiedSurgeryRunnerError(
                f"Checkpoint destination already exists: {destination}."
            )
        parent_rows = getattr(runtime, "parent_token_rows", None)
        if not isinstance(parent_rows, Tensor):
            raise QwenUnifiedSurgeryRunnerError(
                "Prepared runtime lacks parent_token_rows checkpoint provenance."
            )
        contract = codebook_runner._tokenizer_checkpoint_contract(
            self.tokenizer,
            tokenizer_name=self.tokenizer_name,
            parent_rows=tuple(int(item) for item in parent_rows.tolist()),
        )
        artifact_plan = plan_qwen3_codebook_artifact(
            runtime,
            tokenizer_contract=contract,
        )
        if (
            accounting.deployable_bytes != artifact_plan.artifact_total_bytes
            or (
                accounting.artifact_plan_sha256
                and accounting.artifact_plan_sha256
                != artifact_plan.artifact_plan_sha256
            )
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Allocator accounting differs from the canonical checkpoint plan."
            )
        with tempfile.TemporaryDirectory(
            dir=self.output_directory, prefix=".unified-checkpoint."
        ) as temporary:
            staged = Path(temporary) / "checkpoint"
            save_qwen3_codebook_checkpoint(
                runtime,
                staged,
                tokenizer_contract=contract,
            )
            codebook_runner._publish_checkpoint_directory(staged, destination)
        reload_verified = False
        if self.verify_reload:
            reloaded = load_qwen3_codebook_checkpoint(destination, device="cpu")
            try:
                expected_report = qwen3_codebook_parameter_report(runtime)
                actual_report = qwen3_codebook_parameter_report(reloaded)
                reloaded_content = _model_content_identity(reloaded)
                if actual_report != expected_report:
                    raise QwenUnifiedSurgeryRunnerError(
                        "Reloaded frontier checkpoint accounting differs from memory."
                    )
                if reloaded_content != memory_content:
                    raise QwenUnifiedSurgeryRunnerError(
                        "Reloaded checkpoint model content differs from memory."
                    )
            finally:
                del reloaded
                gc.collect()
            reload_verified = True
        tree = production_runner.artifact_tree_report(destination)
        if int(tree["total_bytes"]) != artifact_plan.artifact_total_bytes:
            raise QwenUnifiedSurgeryRunnerError(
                "Published frontier bytes differ from the canonical artifact plan."
            )
        newest = FrontierRecord(
            accepted_transaction_total=_nonnegative_int(
                accepted_transaction_total, "accepted_transaction_total"
            ),
            state_sha256=content_sha256,
            model_content_sha256=content_sha256,
            deployable_bytes=accounting.deployable_bytes,
            behavior_score=float(behavior_score),
            checkpoint=str(destination),
            artifact_tree_sha256=str(tree["tree_sha256"]),
            artifact_total_bytes=int(tree["total_bytes"]),
            transaction_sha256=transaction_sha256,
            reload_verified=reload_verified,
        )
        kept, removed = self._retain(newest)
        if newest not in kept:
            raise QwenUnifiedSurgeryRunnerError(
                "Bounded frontier policy attempted to evict the current checkpoint."
            )
        self.records = kept
        self.current_checkpoint = newest.checkpoint
        self.evicted_total += len({item.state_sha256 for item in removed})
        self._pending_removals.extend(
            item
            for item in removed
            if item.checkpoint != self.current_checkpoint
        )
        manifest = self._write_manifest()
        return manifest

    def finalize_pending_removals(self) -> None:
        pending, self._pending_removals = self._pending_removals, []
        for item in pending:
            if item.checkpoint != self.current_checkpoint:
                self._remove_checkpoint(item.checkpoint)


@dataclass(frozen=True, slots=True)
class UnifiedLoopConfig:
    target_deployable_bytes: int
    candidate_validation_mode: CandidateValidationMode = "independent"
    max_attempts: int = 64
    max_history_records: int = 128
    max_history_json_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_deployable_bytes",
            _nonnegative_int(self.target_deployable_bytes, "target_deployable_bytes"),
        )
        if self.candidate_validation_mode not in {"independent", "provisional"}:
            raise QwenUnifiedSurgeryRunnerError(
                "candidate_validation_mode must be independent or provisional."
            )
        for name in (
            "max_attempts",
            "max_history_records",
            "max_history_json_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))


def resolve_deployable_target(
    initial_deployable_bytes: int,
    *,
    target_deployable_bytes: int | None,
    target_savings_bytes: int | None,
) -> int:
    """Resolve a maximum checkpoint-artifact budget from one request."""

    initial = _positive_int(initial_deployable_bytes, "initial_deployable_bytes")
    if (target_deployable_bytes is None) == (target_savings_bytes is None):
        raise QwenUnifiedSurgeryRunnerError(
            "Specify exactly one absolute deployable-byte target or savings target."
        )
    if target_deployable_bytes is not None:
        return _nonnegative_int(
            target_deployable_bytes, "target_deployable_bytes"
        )
    savings = _nonnegative_int(target_savings_bytes, "target_savings_bytes")
    if savings > initial:
        raise QwenUnifiedSurgeryRunnerError(
            "target_savings_bytes exceeds exact incumbent deployable bytes."
        )
    return initial - savings


def execute_unified_surgery_loop(
    *,
    runtime: nn.Module,
    allocator: Qwen3UnifiedSurgeryAllocator,
    config: UnifiedLoopConfig,
    gate: ParentRelativeBehavioralGate,
    refresh_evidence: Callable[
        [Qwen3UnifiedSurgeryRefreshContext], Qwen3SensitivityEvidence
    ],
    current_evidence: Callable[[], Qwen3SensitivityEvidence],
    checkpoint_manager: ParetoCheckpointManager,
    report_path: Path,
    ledger_path: Path,
    candidate_validator: IndependentCandidateValidator | None,
    resume_report: Mapping[str, Any] | None = None,
    commit_journal: AcceptedCommitJournal | None = None,
    run_contract_sha256: str | None = None,
    initial_model_content_sha256: str | None = None,
    resume_commit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Drive discovery, validation, Pareto selection, and atomic transactions."""

    mode = config.candidate_validation_mode
    if mode == "independent" and candidate_validator is None:
        raise QwenUnifiedSurgeryRunnerError(
            "Independent mode requires an IndependentCandidateValidator."
        )
    state = dict(resume_report or {})
    state.setdefault("schema", RUN_SCHEMA)
    state.setdefault("attempts_total", 0)
    state.setdefault("accepted_transactions_total", 0)
    state.setdefault("rejected_transactions_total", 0)
    state.setdefault("transaction_history", [])
    state.setdefault("transaction_history_omitted", 0)
    state.setdefault("excluded_candidate_sha256s", [])
    parent_commit_seq = (
        0 if resume_commit is None else int(resume_commit["seq"])
    )
    parent_model_content_sha256 = (
        initial_model_content_sha256
        if resume_commit is None
        else str(resume_commit["model_content_sha256"])
    )
    if commit_journal is not None and (
        not run_contract_sha256 or not parent_model_content_sha256
    ):
        raise QwenUnifiedSurgeryRunnerError(
            "Durable execution requires a run contract and parent model content."
        )
    state["status"] = "running"
    state["candidate_validation_mode"] = mode
    state["target_deployable_bytes"] = config.target_deployable_bytes
    initial = allocator.accounting()
    state.setdefault("initial_accounting", initial.as_dict())
    excluded = set(str(item) for item in state["excluded_candidate_sha256s"])
    attempts_this_process = 0
    while allocator.accounting().deployable_bytes > config.target_deployable_bytes:
        if attempts_this_process >= config.max_attempts:
            raise QwenUnifiedSurgeryRunnerError(
                f"max_attempts={config.max_attempts} reached before byte target."
            )
        parent_accounting = allocator.accounting()
        parent_state = allocator._state_sha256()
        gate.bind_parent(runtime, parent_state, parent_accounting)
        raw_proposals = allocator.discover_proposals(
            excluded_sha256s=frozenset(excluded)
        )
        validation_summary = None
        if mode == "independent":
            assert candidate_validator is not None
            validation_summary = candidate_validator.validate(raw_proposals)
        requested = parent_accounting.deployable_bytes - config.target_deployable_bytes
        batch = allocator.select_batch(
            requested,
            excluded_sha256s=frozenset(excluded),
            unsafe_allow_unvalidated_candidates=mode == "provisional",
        )
        transaction = allocator.apply_batch(
            batch,
            evaluator=gate,
            refresh_evidence=refresh_evidence,
            unsafe_allow_unvalidated_candidates=mode == "provisional",
        )
        attempts_this_process += 1
        state["attempts_total"] = int(state["attempts_total"]) + 1
        compact = {
            "attempt_total": state["attempts_total"],
            "parent_state_sha256": parent_state,
            "batch": batch.as_dict(),
            "transaction": transaction.as_dict(),
            "candidate_validation": (
                None
                if validation_summary is None
                else {
                    "round_index": validation_summary["round_index"],
                    "validated_candidate_count": validation_summary[
                        "validated_candidate_count"
                    ],
                    "validation_fingerprint": validation_summary[
                        "validation_fingerprint"
                    ],
                    "summary_artifact": validation_summary["summary_artifact"],
                    "summary_artifact_sha256": validation_summary[
                        "summary_artifact_sha256"
                    ],
                }
            ),
            "gate_artifact": None if gate.last_artifact is None else str(gate.last_artifact),
            "zero_optimizer_steps": 0,
            "backward_called": False,
            "recovery_steps": 0,
        }
        production_runner._append_transaction_ledger(ledger_path, compact)
        history = state["transaction_history"]
        if not isinstance(history, list):
            raise QwenUnifiedSurgeryRunnerError("Resume transaction history is invalid.")
        omitted = production_runner._append_bounded_history(
            history,
            compact,
            max_records=config.max_history_records,
            max_json_bytes=config.max_history_json_bytes,
        )
        state["transaction_history_omitted"] = (
            int(state["transaction_history_omitted"]) + omitted
        )
        if transaction.accepted:
            committed = allocator.accounting()
            if committed != transaction.committed_accounting:
                raise QwenUnifiedSurgeryRunnerError(
                    "Allocator accounting changed after an accepted commit."
                )
            if committed.deployable_bytes >= parent_accounting.deployable_bytes:
                raise QwenUnifiedSurgeryRunnerError(
                    "Accepted surgery did not reduce exact deployable bytes."
                )
            next_accepted_total = int(state["accepted_transactions_total"]) + 1
            excluded.clear()
            committed_state = allocator._state_sha256()
            gate.promote_accepted_candidate(committed_state)
            if gate.last_metrics is None:
                raise QwenUnifiedSurgeryRunnerError("Accepted gate metrics are missing.")
            transaction_payload = transaction.as_dict()
            transaction_sha256 = production_runner._canonical_sha256(
                transaction_payload
            )
            refreshed = current_evidence()
            if refreshed.fingerprints.model_manifest.root != refreshed.fingerprints.model:
                raise QwenUnifiedSurgeryRunnerError(
                    "Accepted evidence refresh has inconsistent model provenance."
                )
            checkpoint_manager.record(
                runtime,
                accepted_transaction_total=next_accepted_total,
                state_sha256=committed_state,
                accounting=committed,
                behavior_score=float(gate.last_metrics["candidate_behavior_score"]),
                transaction_sha256=transaction_sha256,
            )
            frontier_record = checkpoint_manager.current_record
            if frontier_record is None:
                raise QwenUnifiedSurgeryRunnerError(
                    "Accepted checkpoint is absent from the verified frontier."
                )
            lineage = _evidence_lineage(refreshed)
            durable_commit = None
            if commit_journal is not None:
                if gate.last_artifact is None:
                    raise QwenUnifiedSurgeryRunnerError(
                        "Accepted commit has no immutable gate artifact."
                    )
                durable_commit = commit_journal.append(
                    parent_commit_seq=parent_commit_seq,
                    parent_model_content_sha256=str(
                        parent_model_content_sha256
                    ),
                    accepted_transaction_total=next_accepted_total,
                    model_content_sha256=frontier_record.model_content_sha256,
                    checkpoint=Path(frontier_record.checkpoint),
                    artifact_tree={
                        "tree_sha256": frontier_record.artifact_tree_sha256,
                        "total_bytes": frontier_record.artifact_total_bytes,
                    },
                    accounting=committed,
                    run_contract_sha256=str(run_contract_sha256),
                    transaction_payload=transaction_payload,
                    gate_artifact=gate.last_artifact,
                    candidate_validation_artifact=(
                        None
                        if validation_summary is None
                        else Path(str(validation_summary["summary_artifact"]))
                    ),
                    candidate_validation_round_id=(
                        None
                        if validation_summary is None
                        else str(validation_summary["round_id"])
                    ),
                    evidence_lineage=lineage,
                    behavior_score=float(
                        gate.last_metrics["candidate_behavior_score"]
                    ),
                )
                parent_commit_seq = int(durable_commit["seq"])
                parent_model_content_sha256 = str(
                    durable_commit["model_content_sha256"]
                )
                checkpoint_manager.finalize_pending_removals()
            state["accepted_transactions_total"] = next_accepted_total
            state["current_commit_seq"] = parent_commit_seq
            state["current_model_content_sha256"] = (
                frontier_record.model_content_sha256
            )
            state["accepted_evidence_lineage"] = lineage
            state["durable_commit_record_sha256"] = (
                None
                if durable_commit is None
                else durable_commit["record_sha256"]
            )
        else:
            state["rejected_transactions_total"] = (
                int(state["rejected_transactions_total"]) + 1
            )
            if allocator.accounting() != parent_accounting or transaction.rollback_exact is not True:
                raise QwenUnifiedSurgeryRunnerError(
                    "Rejected main transaction did not restore exact parent accounting."
                )
            excluded.update(transaction.proposal_sha256s)
            excluded.update(
                item.candidate.candidate_sha256 for item in batch.proposals
            )
        state["excluded_candidate_sha256s"] = sorted(excluded)
        state["current_accounting"] = allocator.accounting().as_dict()
        state["allocator_diagnostics"] = {
            name: getattr(allocator.diagnostics, name)
            for name in allocator.diagnostics.__dataclass_fields__
        }
        state["frontier_manifest"] = str(checkpoint_manager.manifest_path)
        state["current_checkpoint"] = checkpoint_manager.current_checkpoint
        state["zero_optimizer_steps"] = 0
        state["backward_called"] = False
        state["recovery_steps"] = 0
        production_runner._write_json(report_path, state)

    final = allocator.accounting()
    final_state = allocator._state_sha256()
    gate.bind_parent(runtime, final_state, final)
    parent_score, _ = _behavior_score(gate.parent_cache.parent_report or {})
    final_content = _model_content_identity(runtime)
    if not checkpoint_manager.contains_content(
        str(final_content["model_content_sha256"])
    ):
        checkpoint_manager.record(
            runtime,
            accepted_transaction_total=int(state["accepted_transactions_total"]),
            state_sha256=final_state,
            accounting=final,
            behavior_score=parent_score,
            transaction_sha256=None,
        )
    state["current_accounting"] = final.as_dict()
    state["target_reached"] = final.deployable_bytes <= config.target_deployable_bytes
    state["target_overshoot_bytes"] = max(
        0, config.target_deployable_bytes - final.deployable_bytes
    )
    state["target_contract"] = "maximum-canonical-checkpoint-artifact-bytes"
    state["target_is_upper_bound"] = True
    state["target_equality_required"] = False
    state["target_headroom_bytes"] = max(
        0, config.target_deployable_bytes - final.deployable_bytes
    )
    state["exact_checkpoint_artifact_byte_accounting"] = True
    state["exact_deployable_byte_accounting"] = False
    state["standalone_deployable_bytes_accounted"] = False
    state["unaccounted_deployment_dependencies"] = [
        "actual_tokenizer_asset_files",
        "neural_forge_codebook_runtime_and_python_dependencies",
    ]
    state["current_checkpoint"] = checkpoint_manager.current_checkpoint
    state["frontier_manifest"] = str(checkpoint_manager.manifest_path)
    surgery_total = int(state["accepted_transactions_total"])
    if surgery_total == 0:
        state["status"] = "target_reached_without_surgery"
    elif mode == "independent":
        state["status"] = "target_reached_independently_validated"
    else:
        state["status"] = "target_reached_provisional_unvalidated_candidate_risk"
    state["candidate_marginal_claim"] = (
        mode == "independent" and surgery_total > 0
    )
    state["independent_validation_vacuous"] = (
        mode == "independent" and surgery_total == 0
    )
    state["certified"] = False
    state["certification_note"] = (
        "adaptive transaction gate passed; untouched one-shot certification is not "
        "part of this runner; tokenizer assets and the Neural Forge runtime are not "
        "packaged by this checkpoint-artifact budget"
    )
    state["zero_optimizer_steps"] = 0
    state["backward_called"] = False
    state["recovery_steps"] = 0
    production_runner._write_json(report_path, state)
    return state


def _load_prompts(path: Path, count: int) -> list[str]:
    prompts = codebook_runner.load_calibration_prompts(path)
    selected, _report = codebook_runner.select_calibration_prompts(prompts, count)
    return selected


def _checkpoint_manifest_identity(directory: Path) -> dict[str, Any]:
    checkpoint = directory.expanduser().resolve()
    manifest_path = checkpoint / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenUnifiedSurgeryRunnerError(
            "Codebook checkpoint manifest is unreadable."
        ) from exc
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("files"), Mapping):
        raise QwenUnifiedSurgeryRunnerError(
            "Codebook checkpoint manifest lacks its verified file contract."
        )
    tree = production_runner.artifact_tree_report(checkpoint)
    return {
        "checkpoint_path": str(checkpoint),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": production_runner._file_sha256(manifest_path),
        "declared_files_sha256": production_runner._canonical_sha256(
            dict(manifest["files"])
        ),
        "artifact_tree_sha256": tree["tree_sha256"],
        "artifact_total_bytes": tree["total_bytes"],
        "schema": manifest.get("schema"),
    }


def _tokenizer_contract_identity(tokenizer: Any) -> dict[str, Any]:
    try:
        return production_runner._tokenizer_identity(tokenizer)
    except Exception as exc:
        raise QwenUnifiedSurgeryRunnerError(
            "Tokenizer cannot be bound to an immutable vocabulary contract."
        ) from exc


def _source_contract_identity(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_codebook_checkpoint is not None:
        return {
            "kind": "codebook-checkpoint",
            "identity": _checkpoint_manifest_identity(
                args.input_codebook_checkpoint
            ),
        }
    source = str(args.source_model)
    path = Path(source).expanduser()
    if path.is_dir():
        tree = production_runner.artifact_tree_report(path.resolve())
        return {
            "kind": "local-model-directory",
            "path": str(path.resolve()),
            "artifact_tree_sha256": tree["tree_sha256"],
            "artifact_total_bytes": tree["total_bytes"],
        }
    return {"kind": "model-identifier", "identifier": source}


def _contract_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {
            str(key): _contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_contract_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise QwenUnifiedSurgeryRunnerError(
        f"Unsupported immutable run-contract value: {type(value).__name__}."
    )


def _evaluator_contract_identity(args: argparse.Namespace) -> dict[str, Any]:
    source_path = inspect.getsourcefile(evaluate_loaded_text_model)
    if source_path is None:
        raise QwenUnifiedSurgeryRunnerError(
            "Evaluator implementation has no hashable source file."
        )
    return {
        "callable": (
            f"{evaluate_loaded_text_model.__module__}."
            f"{evaluate_loaded_text_model.__qualname__}"
        ),
        "source_file": str(Path(source_path).resolve()),
        "source_file_sha256": production_runner._file_sha256(Path(source_path)),
        "generation_health_contract": dict(
            production_runner.GENERATION_HEALTH_CONTRACT
        ),
        "device": str(args.device),
        "max_new_tokens": int(args.max_new_tokens),
        "max_length": int(args.max_length),
    }


def persist_failure_report(
    report_path: Path,
    fallback: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    """Mark failure without replacing newer resumable transactional state."""

    latest = dict(fallback)
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, Mapping):
            latest = dict(loaded)
            for key, value in fallback.items():
                latest.setdefault(key, value)
    latest["status"] = "failed"
    latest["error"] = f"{type(error).__name__}: {error}"
    latest["zero_optimizer_steps"] = 0
    latest["backward_called"] = False
    latest["recovery_steps"] = 0
    production_runner._write_json(report_path, latest)
    return latest


def _load_or_prepare_runtime(
    args: argparse.Namespace,
    tokenizer: Any,
    resume_report: Mapping[str, Any] | None,
    identity_prompt: str,
    *,
    resume_checkpoint: Path | None = None,
    source_identity: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any], bool]:
    checkpoint = (
        str(resume_checkpoint.resolve())
        if resume_checkpoint is not None
        else None if resume_report is None else resume_report.get("current_checkpoint")
    )
    if isinstance(checkpoint, str) and Path(checkpoint).is_dir():
        runtime = load_qwen3_codebook_checkpoint(checkpoint, device=args.device)
        report = qwen3_codebook_parameter_report(runtime)
        if report.logical_layers != args.expected_layers:
            raise QwenUnifiedSurgeryRunnerError(
                "Resume checkpoint logical depth differs from this run."
            )
        return runtime.eval(), {
            "resumed_checkpoint": checkpoint,
            "parameter_report": report.as_dict(),
            "identity_preparation_repeated": False,
        }, True
    if args.input_codebook_checkpoint is not None:
        checkpoint_path = args.input_codebook_checkpoint.expanduser().resolve()
        runtime = load_qwen3_codebook_checkpoint(checkpoint_path, device=args.device)
        report = qwen3_codebook_parameter_report(runtime)
        if report.logical_layers != args.expected_layers:
            raise QwenUnifiedSurgeryRunnerError(
                "Input codebook checkpoint logical depth differs from this run."
            )
        codebook_runner._assert_zero_gradients(runtime, "loaded codebook incumbent")
        codebook_runner._assert_model_finite(runtime, "loaded codebook incumbent")
        return runtime.eval(), {
            "input_codebook_checkpoint": str(checkpoint_path),
            "input_checkpoint_manifest": dict(
                (source_identity or {}).get("identity")
                or _checkpoint_manifest_identity(checkpoint_path)
            ),
            "parameter_report": report.as_dict(),
            "identity_preparation_repeated": False,
            "incumbent_prototype_depths": {
                "attention": report.attention_prototypes_per_group,
                "neuron": report.neuron_prototypes_per_neuron,
            },
        }, False
    if not args.source_model:
        raise QwenUnifiedSurgeryRunnerError(
            "Dense preparation requires --source-model."
        )
    try:
        import transformers
    except ImportError as exc:
        raise QwenUnifiedSurgeryRunnerError("transformers is required.") from exc
    runtime = transformers.AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=codebook_runner._dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    runtime.to(args.device)
    runtime.eval()
    runtime, identity = prepare_qwen3_identity_codebook(
        runtime,
        tokenizer,
        prompt=identity_prompt,
        device=args.device,
        max_length=args.max_length,
        expected_layers=args.expected_layers,
    )
    return runtime, identity, False


def _run_binding(
    args: argparse.Namespace,
    split_provenance: Mapping[str, Any],
    *,
    resolved_target_deployable_bytes: int,
    initial_deployable_bytes: int,
    tokenizer_identity: Mapping[str, Any] | None = None,
    initial_model_content_sha256: str | None = None,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_source_identity = dict(
        source_identity or _source_contract_identity(args)
    )
    bound_arguments = {
        name: _contract_value(value)
        for name, value in sorted(vars(args).items())
        if name not in {"output", "resume"}
    }
    payload: dict[str, Any] = {
        "schema": RUN_CONTRACT_SCHEMA,
        "source_model": None if args.source_model is None else str(args.source_model),
        "input_codebook_checkpoint": (
            None
            if args.input_codebook_checkpoint is None
            else str(args.input_codebook_checkpoint.expanduser().resolve())
        ),
        "input_codebook_checkpoint_manifest_identity_sha256": (
            None
            if args.input_codebook_checkpoint is None
            else _canonical_sha256(resolved_source_identity.get("identity"))
        ),
        "tokenizer": str(args.tokenizer or args.source_model),
        "requested_target_deployable_bytes": args.target_deployable_bytes,
        "requested_target_savings_bytes": args.target_savings_bytes,
        "resolved_target_deployable_bytes": _nonnegative_int(
            resolved_target_deployable_bytes,
            "resolved_target_deployable_bytes",
        ),
        "initial_deployable_bytes": _positive_int(
            initial_deployable_bytes, "initial_deployable_bytes"
        ),
        "candidate_validation_mode": args.candidate_validation_mode,
        "expected_layers": int(args.expected_layers),
        "calibration_file": str(args.calibration_file.resolve()),
        "calibration_file_sha256": production_runner._file_sha256(
            args.calibration_file
        ),
        "sensitivity_gate_file": str(args.sensitivity_gate_file.resolve()),
        "sensitivity_gate_file_sha256": production_runner._file_sha256(
            args.sensitivity_gate_file
        ),
        "transaction_gate_suite": str(args.transaction_gate_suite.resolve()),
        "transaction_gate_suite_sha256": production_runner._file_sha256(
            args.transaction_gate_suite
        ),
        "candidate_validation_suite": (
            None
            if args.candidate_validation_suite is None
            else str(args.candidate_validation_suite.resolve())
        ),
        "candidate_validation_suite_sha256": (
            None
            if args.candidate_validation_suite is None
            else production_runner._file_sha256(args.candidate_validation_suite)
        ),
        "split_provenance_sha256": split_provenance["provenance_sha256"],
        "initial_model_content_sha256": initial_model_content_sha256,
        "source_identity": resolved_source_identity,
        "tokenizer_identity": dict(tokenizer_identity or {}),
        "evaluator_identity": _evaluator_contract_identity(args),
        "bound_arguments": bound_arguments,
        "runner_source_sha256": production_runner._file_sha256(
            Path(__file__).resolve()
        ),
    }
    payload["binding_sha256"] = _record_sha256(payload, "binding_sha256")
    payload["run_contract_sha256"] = payload["binding_sha256"]
    _assert_run_contract_integrity(payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    _prepare_run_output(output, resume=bool(args.resume))
    report_path = output / "run_report.json"
    resume_report: dict[str, Any] | None = None
    if args.resume and report_path.is_file():
        resume_report = _read_json_mapping(report_path, "Run report")
    journal = AcceptedCommitJournal(output)
    preliminary_replay: JournalReplay | None = None
    if journal.initial_path.is_file():
        preliminary_replay = journal.replay()
    elif resume_report is not None:
        raise QwenUnifiedSurgeryRunnerError(
            "Existing run report predates the durable initial-incumbent contract."
        )
    resume_checkpoint = (
        None
        if preliminary_replay is None or preliminary_replay.selected is None
        else Path(str(preliminary_replay.selected["checkpoint"]))
    )
    try:
        import transformers
    except ImportError as exc:
        raise QwenUnifiedSurgeryRunnerError("transformers is required.") from exc
    tokenizer_name = args.tokenizer or args.source_model
    if not tokenizer_name:
        raise QwenUnifiedSurgeryRunnerError(
            "Checkpoint input requires --tokenizer when --source-model is omitted."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer_identity = _tokenizer_contract_identity(tokenizer)
    source_identity = _source_contract_identity(args)
    calibration_prompts = _load_prompts(args.calibration_file, args.calibration_count)
    sensitivity_gate_prompts = _load_prompts(
        args.sensitivity_gate_file, args.sensitivity_gate_count
    )
    calibration = encode_prompt_rows(
        tokenizer,
        calibration_prompts,
        max_length=args.max_length,
        name="sensitivity-calibration",
    )
    sensitivity_gate = encode_prompt_rows(
        tokenizer,
        sensitivity_gate_prompts,
        max_length=args.max_length,
        name="sensitivity-gate",
    )
    transaction_suite = bind_evaluation_suite(
        tokenizer,
        args.transaction_gate_suite,
        name="transaction-gate",
        max_new_tokens=args.max_new_tokens,
    )
    validation_suite = (
        None
        if args.candidate_validation_suite is None
        else bind_evaluation_suite(
            tokenizer,
            args.candidate_validation_suite,
            name="candidate-validation",
            max_new_tokens=args.max_new_tokens,
        )
    )
    split_provenance = build_split_provenance(
        calibration,
        sensitivity_gate,
        transaction_suite,
        validation_suite,
        mode=args.candidate_validation_mode,
    )
    runtime, identity_report, resumed = _load_or_prepare_runtime(
        args,
        tokenizer,
        None,
        calibration_prompts[0],
        resume_checkpoint=resume_checkpoint,
        source_identity=source_identity,
    )
    parent_token_rows = getattr(runtime, "parent_token_rows", None)
    if not isinstance(parent_token_rows, Tensor):
        raise QwenUnifiedSurgeryRunnerError(
            "Prepared runtime lacks parent_token_rows tokenizer provenance."
        )
    artifact_tokenizer_contract = codebook_runner._tokenizer_checkpoint_contract(
        tokenizer,
        tokenizer_name=str(tokenizer_name),
        parent_rows=tuple(
            int(item)
            for item in parent_token_rows.detach().to(device="cpu").tolist()
        ),
    )
    layers = int(runtime.config.num_hidden_layers)
    intermediate = int(runtime.config.intermediate_size)
    groups = int(runtime.config.num_key_value_heads)
    neuron_candidates = _bounded_refs(
        layers, intermediate, args.neuron_sensitivity_candidates, Qwen3NeuronRef
    )
    attention_candidates = _bounded_refs(
        layers,
        groups,
        args.attention_sensitivity_candidates,
        Qwen3AttentionGroupRef,
    )
    sensitivity_limits = Qwen3SensitivityLimits(
        max_calibration_rows=max(args.calibration_count, 1),
        max_gate_rows=max(args.sensitivity_gate_count, 1),
        max_sequence_length=args.max_length,
        max_score_rows=max(args.calibration_count, 1),
        max_neuron_candidates=max(args.neuron_sensitivity_candidates, 1),
        max_attention_candidates=max(args.attention_sensitivity_candidates, 1),
        candidate_chunk_size=args.sensitivity_candidate_chunk_size,
        max_candidate_forward_rows=args.max_candidate_forward_rows,
        max_collector_bytes=args.max_collector_bytes,
        max_process_rss_increase_bytes=args.max_process_rss_increase_bytes,
        max_cuda_allocated_increase_bytes=args.max_cuda_allocated_increase_bytes,
        max_cuda_reserved_increase_bytes=args.max_cuda_reserved_increase_bytes,
    )

    def collect(
        *,
        parent_manifest: Qwen3ModelFingerprintManifest | None = None,
        changed_prefixes: Sequence[str] = (),
        accepted_batch_sha256: str | None = None,
    ) -> Qwen3SensitivityEvidence:
        return collect_qwen3_sensitivity_evidence(
            runtime,
            calibration.input_ids,
            sensitivity_gate.input_ids,
            calibration_attention_mask=calibration.attention_mask,
            gate_attention_mask=sensitivity_gate.attention_mask,
            score_positions=(-1,),
            neuron_candidates=neuron_candidates,
            attention_group_candidates=attention_candidates,
            neuron_high_damage_anchor_count=args.neuron_high_damage_anchors,
            attention_high_damage_anchor_count=args.attention_high_damage_anchors,
            sketch_dim=args.sensitivity_sketch_dim,
            limits=sensitivity_limits,
            incremental_parent_manifest=parent_manifest,
            changed_state_prefixes=changed_prefixes,
            accepted_transaction_sha256=accepted_batch_sha256,
        )

    evidence_holder = [collect()]
    allocator = Qwen3UnifiedSurgeryAllocator.from_evidence(
        runtime,
        evidence_holder[0],
        artifact_tokenizer_contract=artifact_tokenizer_contract,
        limits=Qwen3SurgeryAllocatorLimits(
            max_proposals_per_modality=args.max_proposals_per_modality,
            max_batch_size=args.max_batch_size,
            max_knapsack_states=args.max_knapsack_states,
            max_expansion_work=args.max_expansion_work,
            max_attempts=args.max_attempts,
            history_limit=args.allocator_history_limit,
            max_overshoot_bytes=args.max_overshoot_bytes,
            max_overshoot_ratio=args.max_overshoot_ratio,
            max_process_rss_increase_bytes=args.max_process_rss_increase_bytes,
            max_cuda_allocated_increase_bytes=args.max_cuda_allocated_increase_bytes,
            max_cuda_reserved_increase_bytes=args.max_cuda_reserved_increase_bytes,
        ),
    )
    accepted_before = (
        0
        if preliminary_replay is None
        else preliminary_replay.accepted_transaction_total
    )
    if accepted_before:
        allocator._accepted_transactions = accepted_before
        allocator._evidence_epoch = accepted_before
        allocator._accounting_cache = None
    current_initial_accounting = allocator.accounting()
    current_model_content = _model_content_identity(runtime)
    if preliminary_replay is not None:
        initial_record = preliminary_replay.initial
        prior_binding = (resume_report or {}).get("run_contract") or (
            resume_report or {}
        ).get("run_binding") or initial_record.get("run_contract")
        if not isinstance(prior_binding, Mapping):
            raise QwenUnifiedSurgeryRunnerError(
                "Resume report lacks its immutable run contract."
            )
        resolved_target = _nonnegative_int(
            prior_binding.get("resolved_target_deployable_bytes"),
            "persisted resolved target",
        )
        bound_initial_deployable = _positive_int(
            initial_record["accounting"].get("deployable_bytes"),
            "persisted initial deployable bytes",
        )
        initial_model_content_sha256 = str(
            initial_record["model_content_sha256"]
        )
        expected_current_content = preliminary_replay.model_content_sha256
        if (
            current_model_content["model_content_sha256"]
            != expected_current_content
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Loaded resume checkpoint model content differs from the journal."
            )
    else:
        bound_initial_deployable = current_initial_accounting.deployable_bytes
        initial_model_content_sha256 = str(
            current_model_content["model_content_sha256"]
        )
        resolved_target = resolve_deployable_target(
            bound_initial_deployable,
            target_deployable_bytes=args.target_deployable_bytes,
            target_savings_bytes=args.target_savings_bytes,
        )
    binding = _run_binding(
        args,
        split_provenance,
        resolved_target_deployable_bytes=resolved_target,
        initial_deployable_bytes=bound_initial_deployable,
        tokenizer_identity=tokenizer_identity,
        initial_model_content_sha256=initial_model_content_sha256,
        source_identity=source_identity,
    )
    if preliminary_replay is None:
        journal.record_or_verify_initial(
            model_content=current_model_content,
            accounting=current_initial_accounting,
            run_contract=binding,
            source_identity=source_identity,
        )
        replay = journal.replay(
            expected_run_contract_sha256=str(binding["run_contract_sha256"])
        )
    else:
        prior_contract = (resume_report or {}).get("run_contract") or (
            resume_report or {}
        ).get("run_binding") or preliminary_replay.initial.get("run_contract")
        if prior_contract != binding:
            raise QwenUnifiedSurgeryRunnerError(
                "Resume evaluator, tokenizer, policy, inputs, or limits drifted."
            )
        replay = journal.replay(
            expected_run_contract_sha256=str(binding["run_contract_sha256"])
        )
        if replay.commit_seq != preliminary_replay.commit_seq:
            raise QwenUnifiedSurgeryRunnerError(
                "Journal replay changed while binding the run contract."
            )
        if replay.selected is not None:
            _verify_evidence_lineage(
                evidence_holder[0], replay.selected["evidence_lineage"]
            )
    executor = SuiteExecutor(
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    transaction_parent_cache = CachedParentSuite(
        suite=transaction_suite,
        executor=executor,
        output_directory=output / "transaction_parent_cache",
    )
    gate = ParentRelativeBehavioralGate(
        parent_cache=transaction_parent_cache,
        retention_floors=RetainedParentFloors(
            behavior=args.min_behavior_retention,
            choice=args.min_choice_retention,
            generation=args.min_generation_retention,
        ),
        hard_loss_limits=(
            args.max_behavior_losses,
            args.max_choice_losses,
            args.max_generation_losses,
        ),
        output_directory=output / "transaction_gates",
    )
    candidate_validator = None
    if args.candidate_validation_mode == "independent":
        assert validation_suite is not None
        validation_cache = CachedParentSuite(
            suite=validation_suite,
            executor=executor,
            output_directory=output / "validation_parent_cache",
        )
        candidate_validator = IndependentCandidateValidator(
            allocator=allocator,
            evidence=lambda: evidence_holder[0],
            parent_cache=validation_cache,
            output_directory=output / "candidate_validation",
            max_candidates=args.max_candidate_validations,
            discovery_row_fingerprints=tuple(
                sorted(
                    {
                        *calibration.path_domain.token_content_sha256s,
                        *sensitivity_gate.path_domain.token_content_sha256s,
                    }
                )
            ),
        )
    pending_refresh: dict[str, Qwen3SensitivityEvidence] = {}

    def refresh(
        context: Qwen3UnifiedSurgeryRefreshContext,
    ) -> Qwen3SensitivityEvidence:
        kinds = {item.kind for item in context.batch.proposals}
        prefixes = tuple(
            value
            for kind, value in (
                ("neuron", "neuron_codebook.registry."),
                ("attention", "attention_codebook.registry."),
            )
            if kind in kinds
        )
        fresh = collect(
            parent_manifest=evidence_holder[0].fingerprints.model_manifest,
            changed_prefixes=prefixes,
            accepted_batch_sha256=context.batch.batch_sha256,
        )
        pending_refresh[context.batch.batch_sha256] = fresh
        return fresh

    class _EvidenceView:
        def __call__(self) -> Qwen3SensitivityEvidence:
            if pending_refresh:
                _batch_sha, fresh = pending_refresh.popitem()
                evidence_holder[0] = fresh
            return evidence_holder[0]

    evidence_view = _EvidenceView()
    checkpoint_manager = ParetoCheckpointManager(
        output_directory=output,
        tokenizer=tokenizer,
        tokenizer_name=str(tokenizer_name),
        max_checkpoints=args.max_frontier_checkpoints,
        verify_reload=args.verify_checkpoint_reload,
    )
    checkpoint_manager.reconcile_from_journal(replay)
    base_report = dict(resume_report or {})
    if int(base_report.get("current_commit_seq", replay.commit_seq)) != replay.commit_seq:
        base_report["excluded_candidate_sha256s"] = []
    base_report.update(
        {
            "schema": RUN_SCHEMA,
            "run_binding": binding,
            "run_contract": binding,
            "run_contract_sha256": binding["run_contract_sha256"],
            "requested_target_deployable_bytes": args.target_deployable_bytes,
            "requested_target_savings_bytes": args.target_savings_bytes,
            "resolved_target_deployable_bytes": resolved_target,
            "identity": identity_report,
            "resumed": resumed,
            "journal_replay": {
                "selected_commit_seq": replay.commit_seq,
                "accepted_transaction_total": replay.accepted_transaction_total,
                "fallback_reason": replay.fallback_reason,
                "record_count": len(replay.records),
                "artifact_verified_record_count": len(
                    replay.artifact_verified_records
                ),
            },
            "accepted_transactions_total": replay.accepted_transaction_total,
            "initial_accounting": dict(replay.initial["accounting"]),
            "current_commit_seq": replay.commit_seq,
            "current_model_content_sha256": replay.model_content_sha256,
            "current_checkpoint": (
                None
                if replay.selected is None
                else replay.selected["checkpoint"]
            ),
            "accepted_evidence_lineage": (
                None
                if replay.selected is None
                else replay.selected["evidence_lineage"]
            ),
            "split_provenance": split_provenance,
            "sensitivity": {
                "fingerprints": {
                    "model": evidence_holder[0].fingerprints.model,
                    "calibration": evidence_holder[0].fingerprints.calibration,
                    "gate": evidence_holder[0].fingerprints.gate,
                    "runtime": evidence_holder[0].fingerprints.runtime,
                    "cache": evidence_holder[0].fingerprints.cache,
                },
                "report": {
                    name: getattr(evidence_holder[0].report, name)
                    for name in evidence_holder[0].report.__dataclass_fields__
                },
            },
        }
    )
    try:
        return execute_unified_surgery_loop(
            runtime=runtime,
            allocator=allocator,
            config=UnifiedLoopConfig(
                target_deployable_bytes=resolved_target,
                candidate_validation_mode=args.candidate_validation_mode,
                max_attempts=args.max_attempts,
                max_history_records=args.max_history_records,
                max_history_json_bytes=args.max_history_json_bytes,
            ),
            gate=gate,
            refresh_evidence=refresh,
            current_evidence=evidence_view,
            checkpoint_manager=checkpoint_manager,
            report_path=report_path,
            ledger_path=output / "transactions.jsonl",
            candidate_validator=candidate_validator,
            resume_report=base_report,
            commit_journal=journal,
            run_contract_sha256=str(binding["run_contract_sha256"]),
            initial_model_content_sha256=initial_model_content_sha256,
            resume_commit=replay.selected,
        )
    except Exception as exc:
        persist_failure_report(report_path, base_report, exc)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model")
    parser.add_argument("--input-codebook-checkpoint", type=Path)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--max-checkpoint-artifact-bytes",
        "--target-deployable-bytes",
        dest="target_deployable_bytes",
        type=int,
        help=(
            "Maximum canonical four-file checkpoint bytes. The legacy option name "
            "is retained as an alias; runtime/tokenizer assets are not included."
        ),
    )
    target.add_argument(
        "--checkpoint-artifact-savings-bytes",
        "--target-savings-bytes",
        dest="target_savings_bytes",
        type=int,
        help="Minimum canonical checkpoint-artifact bytes to remove from the incumbent.",
    )
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--sensitivity-gate-file", type=Path, required=True)
    parser.add_argument("--candidate-validation-suite", type=Path)
    parser.add_argument("--transaction-gate-suite", type=Path, required=True)
    parser.add_argument(
        "--candidate-validation-mode",
        choices=("independent", "provisional"),
        default="independent",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--expected-layers", type=int, default=36)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--calibration-count", type=int, default=8)
    parser.add_argument("--sensitivity-gate-count", type=int, default=8)
    parser.add_argument("--neuron-sensitivity-candidates", type=int, default=256)
    parser.add_argument("--attention-sensitivity-candidates", type=int, default=128)
    parser.add_argument("--sensitivity-candidate-chunk-size", type=int, default=4)
    parser.add_argument("--max-candidate-forward-rows", type=int, default=256)
    parser.add_argument("--sensitivity-sketch-dim", type=int, default=16)
    parser.add_argument("--neuron-high-damage-anchors", type=int, default=4)
    parser.add_argument("--attention-high-damage-anchors", type=int, default=4)
    parser.add_argument("--max-candidate-validations", type=int, default=16)
    parser.add_argument("--max-proposals-per-modality", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-knapsack-states", type=int, default=4096)
    parser.add_argument("--max-expansion-work", type=int, default=262_144)
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--allocator-history-limit", type=int, default=64)
    parser.add_argument("--max-overshoot-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-overshoot-ratio", type=float, default=1.0)
    parser.add_argument("--max-frontier-checkpoints", type=int, default=3)
    parser.add_argument("--max-history-records", type=int, default=128)
    parser.add_argument("--max-history-json-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-behavior-retention", type=float, default=0.90)
    parser.add_argument("--min-choice-retention", type=float, default=0.90)
    parser.add_argument("--min-generation-retention", type=float, default=0.90)
    parser.add_argument("--max-behavior-losses", type=int)
    parser.add_argument("--max-choice-losses", type=int)
    parser.add_argument("--max-generation-losses", type=int)
    parser.add_argument("--max-collector-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--max-process-rss-increase-bytes", type=int, default=24 * 1024**3
    )
    parser.add_argument(
        "--max-cuda-allocated-increase-bytes", type=int, default=24 * 1024**3
    )
    parser.add_argument(
        "--max-cuda-reserved-increase-bytes", type=int, default=32 * 1024**3
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--verify-checkpoint-reload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_deployable_bytes is not None:
        _nonnegative_int(args.target_deployable_bytes, "target_deployable_bytes")
    if args.target_savings_bytes is not None:
        _nonnegative_int(args.target_savings_bytes, "target_savings_bytes")
    for name in ("max_behavior_losses", "max_choice_losses", "max_generation_losses"):
        value = getattr(args, name)
        if value is not None:
            _nonnegative_int(value, name)
    RetainedParentFloors(
        behavior=args.min_behavior_retention,
        choice=args.min_choice_retention,
        generation=args.min_generation_retention,
    )
    for name in (
        "expected_layers",
        "max_length",
        "calibration_count",
        "sensitivity_gate_count",
        "neuron_sensitivity_candidates",
        "attention_sensitivity_candidates",
        "sensitivity_candidate_chunk_size",
        "max_candidate_forward_rows",
        "sensitivity_sketch_dim",
        "max_candidate_validations",
        "max_proposals_per_modality",
        "max_batch_size",
        "max_knapsack_states",
        "max_expansion_work",
        "max_attempts",
        "allocator_history_limit",
        "max_overshoot_bytes",
        "max_frontier_checkpoints",
        "max_history_records",
        "max_history_json_bytes",
        "max_new_tokens",
        "max_collector_bytes",
        "max_process_rss_increase_bytes",
        "max_cuda_allocated_increase_bytes",
        "max_cuda_reserved_increase_bytes",
    ):
        _positive_int(getattr(args, name), name)
    for name in ("neuron_high_damage_anchors", "attention_high_damage_anchors"):
        _nonnegative_int(getattr(args, name), name)
    if not math.isfinite(args.max_overshoot_ratio) or args.max_overshoot_ratio < 0.0:
        raise QwenUnifiedSurgeryRunnerError(
            "max_overshoot_ratio must be finite and non-negative."
        )
    for name in (
        "calibration_file",
        "sensitivity_gate_file",
        "transaction_gate_suite",
    ):
        if not getattr(args, name).is_file():
            raise FileNotFoundError(getattr(args, name))
    if args.candidate_validation_mode == "independent":
        if (
            args.candidate_validation_suite is None
            or not args.candidate_validation_suite.is_file()
        ):
            raise QwenUnifiedSurgeryRunnerError(
                "Independent mode requires --candidate-validation-suite."
            )
    if args.source_model is None and args.input_codebook_checkpoint is None:
        raise QwenUnifiedSurgeryRunnerError(
            "Provide --source-model or --input-codebook-checkpoint."
        )
    if args.input_codebook_checkpoint is not None:
        if not args.input_codebook_checkpoint.is_dir():
            raise FileNotFoundError(args.input_codebook_checkpoint)
        if args.source_model is None and args.tokenizer is None:
            raise QwenUnifiedSurgeryRunnerError(
                "Checkpoint-only input requires --tokenizer."
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
