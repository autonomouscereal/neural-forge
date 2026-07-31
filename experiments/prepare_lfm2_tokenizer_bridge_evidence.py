#!/usr/bin/env python3
"""Prepare immutable parent evidence for an LFM2 tokenizer bridge.

This command performs one direct-local parent execution phase.  It does not
construct a child, fold a model, train, prune model Parameters, quantize, or
access a model hub.  It:

1. target-tokenizes sealed trajectory text,
2. maps every target ID through the sealed one-row donor transport,
3. captures one content-addressed parent trace cache on those mapped IDs,
4. rebinds an already-sealed canonical-parent behavior baseline to that cache,
5. generates the same behavior prompts once through the mapped target IDs,
6. seals the mapped-parent baseline and its normalization control, and
7. emits every path and digest required by the production folding runner.

An exact rerun verifies and reuses the completed manifest without loading or
executing the parent.  A partial or identity-mismatched directory fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from experiments import run_lfm2_dynamic_catalog as catalog
from experiments import run_lfm2_residual_attention_fold as residual
from neural_forge.dynamic_lfm2_behavior import (
    ENVELOPE_SCHEMA,
    load_lfm2_cached_whole_output_baseline,
    seal_lfm2_cached_whole_output_envelope,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MANIFEST_FILENAME = "neural_forge_lfm2_tokenizer_bridge_evidence.json"
MANIFEST_SCHEMA = "neural-forge-lfm2-tokenizer-bridge-evidence-v1"
TRAJECTORY_TEXT_SCHEMA = "neural-forge-lfm2-bridge-trajectory-text-v1"
TARGET_CALIBRATION_FILENAME = "target_tokenized_calibration.json"
DONOR_CALIBRATION_FILENAME = "mapped_donor_calibration.json"
ORIGINAL_ROWS_FILENAME = "canonical_parent_rows.json"
ORIGINAL_ENVELOPE_FILENAME = "canonical_parent_outputs_bridge_bound.json"
MAPPED_ROWS_FILENAME = "mapped_parent_rows.json"
MAPPED_ENVELOPE_FILENAME = "mapped_parent_outputs.json"
PARENT_CACHE_ROOT_NAME = "parent_trace_cache"


class Lfm2TokenizerBridgeEvidenceError(RuntimeError):
    """Bridge evidence inputs or outputs violate the production contract."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Bridge evidence metadata is not canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise Lfm2TokenizerBridgeEvidenceError(
            f"Cannot hash bridge evidence file {path}."
        ) from exc
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} must be a finite nonnegative float."
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} must be a finite nonnegative float."
        ) from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} must be a finite nonnegative float."
        )
    return normalized


def _fraction(value: Any, name: str) -> float:
    normalized = _nonnegative_float(value, name)
    if normalized > 1.0:
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} must be between zero and one."
        )
    return normalized


def _local_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} must be a regular local file."
        )
    return resolved


def _local_directory(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} must be a real local directory."
        )
    return resolved


def _write_atomic_json(path: Path, value: Any) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        raise Lfm2TokenizerBridgeEvidenceError(
            f"Temporary bridge evidence file already exists: {temporary}."
        )
    payload = _canonical_json(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_canonical_object(path: Path, name: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} is not canonical ASCII JSON."
        ) from exc
    if not isinstance(value, Mapping) or _canonical_json(value) + b"\n" != raw:
        raise Lfm2TokenizerBridgeEvidenceError(
            f"{name} is not a canonical JSON object."
        )
    return dict(value)


def _verify_parent_physical_row_capacity(
    config: BridgeEvidencePreparationConfig,
    artifact: Any,
) -> int:
    config_path = config.parent_model / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise Lfm2TokenizerBridgeEvidenceError(
            "Parent model lacks a regular local config.json."
        )
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Parent model config.json is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, Mapping) or value.get("model_type") != "lfm2":
        raise Lfm2TokenizerBridgeEvidenceError(
            "Parent model config.json is not an LFM2 configuration."
        )
    capacity = _positive_int(
        value.get("vocab_size"),
        "parent config.vocab_size",
    )
    bridge_capacity = artifact.transport_profile.donor_row_capacity
    if bridge_capacity != capacity:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Tokenizer bridge donor row capacity differs from the parent "
            f"model's physical vocabulary rows: {bridge_capacity} != "
            f"{capacity}."
        )
    return capacity


@dataclass(frozen=True, slots=True)
class BridgeEvidencePreparationConfig:
    parent_model: Path
    donor_tokenizer_directory: Path
    target_tokenizer_directory: Path
    target_tokenizer_compilation_receipt: Path
    target_tokenizer_compilation_receipt_sha256: str
    bridge_artifact_sha256: str
    bridge_profile_sha256: str
    trajectory_texts: Path
    trajectory_texts_sha256: str
    original_parent_outputs: Path
    original_parent_outputs_sha256: str
    expected_parent_state_fingerprint: str
    output_directory: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    sample_last_tokens: int = 8
    max_decode_steps: int = 32
    max_capture_bytes: int = 512 * 1024 * 1024
    capture_logit_rtol: float = catalog.DEFAULT_CAPTURE_LOGIT_RTOL
    capture_logit_atol: float = catalog.DEFAULT_CAPTURE_LOGIT_ATOL
    require_capture_top1_agreement: bool = True
    allow_capture_top1_disagreement_within_numeric_error_envelope: bool = True
    capture_causal_circuit_evidence: bool = True
    generation_limit: int = 64
    maximum_prompt_tokens: int = 512
    minimum_parent_task_retention: float = 0.9
    fragment_minimum_characters: int = 12
    maximum_new_generation_limit_saturations: int = 0
    maximum_repeated_ngram_share_regressions: int = 0
    maximum_repeated_bigram_share_delta: float = 0.05
    final_performance_metrics_diagnostic_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "parent_model",
            "donor_tokenizer_directory",
            "target_tokenizer_directory",
        ):
            object.__setattr__(
                self,
                name,
                _local_directory(Path(getattr(self, name)), name),
            )
        for name in (
            "target_tokenizer_compilation_receipt",
            "trajectory_texts",
            "original_parent_outputs",
        ):
            object.__setattr__(
                self,
                name,
                _local_file(Path(getattr(self, name)), name),
            )
        for name in (
            "target_tokenizer_compilation_receipt_sha256",
            "bridge_artifact_sha256",
            "bridge_profile_sha256",
            "trajectory_texts_sha256",
            "original_parent_outputs_sha256",
            "expected_parent_state_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        output = Path(self.output_directory).expanduser().absolute()
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise Lfm2TokenizerBridgeEvidenceError(
                "output_directory must be absent or a real directory."
            )
        object.__setattr__(self, "output_directory", output)
        if not isinstance(self.device, str) or not self.device.strip():
            raise Lfm2TokenizerBridgeEvidenceError(
                "device must be a nonempty local torch device."
            )
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise Lfm2TokenizerBridgeEvidenceError(
                "dtype must be float32, float16, or bfloat16."
            )
        for name in (
            "sample_last_tokens",
            "max_decode_steps",
            "max_capture_bytes",
            "generation_limit",
            "maximum_prompt_tokens",
            "fragment_minimum_characters",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        for name in (
            "maximum_new_generation_limit_saturations",
            "maximum_repeated_ngram_share_regressions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Lfm2TokenizerBridgeEvidenceError(
                    f"{name} must be a nonnegative integer."
                )
        for name in (
            "capture_logit_rtol",
            "capture_logit_atol",
            "maximum_repeated_bigram_share_delta",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "minimum_parent_task_retention",
            _fraction(
                self.minimum_parent_task_retention,
                "minimum_parent_task_retention",
            ),
        )
        if not isinstance(self.require_capture_top1_agreement, bool):
            raise Lfm2TokenizerBridgeEvidenceError(
                "require_capture_top1_agreement must be boolean."
            )
        if not isinstance(
            self.allow_capture_top1_disagreement_within_numeric_error_envelope,
            bool,
        ):
            raise Lfm2TokenizerBridgeEvidenceError(
                "allow_capture_top1_disagreement_within_numeric_error_"
                "envelope must be boolean."
            )
        if self.capture_causal_circuit_evidence is not True:
            raise Lfm2TokenizerBridgeEvidenceError(
                "Production bridge preparation requires "
                "capture_causal_circuit_evidence=True."
            )

    def identity(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-bridge-evidence-config-v1",
            "parent_model": str(self.parent_model),
            "donor_tokenizer_directory": str(
                self.donor_tokenizer_directory
            ),
            "target_tokenizer_directory": str(
                self.target_tokenizer_directory
            ),
            "target_tokenizer_compilation_receipt": str(
                self.target_tokenizer_compilation_receipt
            ),
            "target_tokenizer_compilation_receipt_sha256": (
                self.target_tokenizer_compilation_receipt_sha256
            ),
            "bridge_artifact_sha256": self.bridge_artifact_sha256,
            "bridge_profile_sha256": self.bridge_profile_sha256,
            "trajectory_texts": str(self.trajectory_texts),
            "trajectory_texts_sha256": self.trajectory_texts_sha256,
            "original_parent_outputs": str(self.original_parent_outputs),
            "original_parent_outputs_sha256": (
                self.original_parent_outputs_sha256
            ),
            "expected_parent_state_fingerprint": (
                self.expected_parent_state_fingerprint
            ),
            "output_directory": str(self.output_directory),
            "device": self.device,
            "dtype": self.dtype,
            "capture": {
                "sample_last_tokens": self.sample_last_tokens,
                "max_decode_steps": self.max_decode_steps,
                "max_capture_bytes": self.max_capture_bytes,
                "logit_rtol": self.capture_logit_rtol,
                "logit_atol": self.capture_logit_atol,
                "require_top1_agreement": (
                    self.require_capture_top1_agreement
                ),
                (
                    "allow_top1_disagreement_within_numeric_error_"
                    "envelope"
                ): (
                    self.allow_capture_top1_disagreement_within_numeric_error_envelope
                ),
                "capture_causal_circuit_evidence": (
                    self.capture_causal_circuit_evidence
                ),
            },
            "behavior": {
                "generation_limit": self.generation_limit,
                "maximum_prompt_tokens": self.maximum_prompt_tokens,
                "minimum_parent_task_retention": (
                    self.minimum_parent_task_retention
                ),
                "fragment_minimum_characters": (
                    self.fragment_minimum_characters
                ),
                "maximum_new_generation_limit_saturations": (
                    self.maximum_new_generation_limit_saturations
                ),
                "maximum_repeated_ngram_share_regressions": (
                    self.maximum_repeated_ngram_share_regressions
                ),
                "maximum_repeated_bigram_share_delta": (
                    self.maximum_repeated_bigram_share_delta
                ),
            },
            "network_access": False,
            "child_constructed": False,
            "fold_invoked": False,
            "training_steps": 0,
        }


def _verify_compilation_receipt(
    config: BridgeEvidencePreparationConfig,
    *,
    artifact: Any,
) -> dict[str, Any]:
    source = config.target_tokenizer_compilation_receipt
    if _file_sha256(source) != (
        config.target_tokenizer_compilation_receipt_sha256
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Target-tokenizer compilation receipt differs from its explicit "
            "file digest."
        )
    receipt = _read_canonical_object(
        source,
        "target-tokenizer compilation receipt",
    )
    body = dict(receipt)
    seal = body.pop("receipt_sha256", None)
    expected = {
        "status": "compiled",
        "target_tokenizer_directory": str(
            config.target_tokenizer_directory
        ),
        "bridge_artifact_sha256": config.bridge_artifact_sha256,
        "bridge_profile_sha256": config.bridge_profile_sha256,
        "donor_physical_row_capacity": (
            artifact.transport_profile.donor_row_capacity
        ),
    }
    if (
        seal != _canonical_sha256(body)
        or any(receipt.get(name) != value for name, value in expected.items())
        or artifact.artifact_sha256 != config.bridge_artifact_sha256
        or artifact.profile_fingerprint != config.bridge_profile_sha256
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Target-tokenizer compilation receipt is not bound to the live "
            "sealed bridge."
        )
    if (
        receipt.get("parent_forward_calls") != 0
        or receipt.get("training_steps") != 0
        or receipt.get("model_parameters_pruned") != 0
        or receipt.get("model_parameters_quantized") != 0
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Target-tokenizer compilation receipt has invalid model-operation "
            "claims."
        )
    return receipt


def _read_trajectory_texts(
    config: BridgeEvidencePreparationConfig,
    *,
    donor_tokenizer: Any | None = None,
) -> tuple[dict[str, str], ...]:
    source = config.trajectory_texts
    if _file_sha256(source) != config.trajectory_texts_sha256:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Trajectory text source differs from its explicit digest."
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Trajectory text source is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, list) or not value:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Trajectory text source must be a nonempty JSON list."
        )
    rows = []
    seen = set()
    splits = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise Lfm2TokenizerBridgeEvidenceError(
                f"Trajectory text row {index} is not an object."
            )
        schema = row.get("schema")
        record_id = row.get("id")
        split = row.get("split")
        if schema == TRAJECTORY_TEXT_SCHEMA:
            input_text = row.get("input_text")
            forced_text = row.get("forced_decode_text")
            source_mode = "explicit-bridge-trajectory-text"
        elif schema is None and donor_tokenizer is not None:
            input_text = row.get("prompt")
            forced_text = row.get("parent_output_text")
            input_ids = row.get("input_ids")
            forced_ids = row.get("forced_decode_ids")
            parent_output_ids = row.get("parent_output_ids")
            if (
                not isinstance(input_text, str)
                or not input_text
                or not isinstance(forced_text, str)
                or not forced_text
                or not isinstance(input_ids, list)
                or not input_ids
                or not isinstance(forced_ids, list)
                or not forced_ids
                or forced_ids != parent_output_ids
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in (*input_ids, *forced_ids)
                )
            ):
                raise Lfm2TokenizerBridgeEvidenceError(
                    f"Existing parent-greedy trajectory row {index} is "
                    "malformed."
                )
            canonical_donor_input = catalog._tokenizer_input_ids(
                donor_tokenizer,
                input_text,
                add_special_tokens=True,
            )
            decoded_donor_output = donor_tokenizer.decode(
                forced_ids,
                skip_special_tokens=True,
            )
            if (
                tuple(input_ids) != canonical_donor_input
                or decoded_donor_output != forced_text
            ):
                raise Lfm2TokenizerBridgeEvidenceError(
                    f"Existing parent-greedy trajectory row {index} does not "
                    "match canonical donor tokenization and decoding."
                )
            source_mode = "verified-existing-parent-greedy-text"
        else:
            input_text = None
            forced_text = None
            source_mode = None
        if (
            source_mode is None
            or not isinstance(record_id, str)
            or not record_id
            or record_id in seen
            or split not in {"selection", "fit", "validation"}
            or not isinstance(input_text, str)
            or not input_text
            or not isinstance(forced_text, str)
            or not forced_text
        ):
            raise Lfm2TokenizerBridgeEvidenceError(
                f"Trajectory text row {index} is malformed."
            )
        seen.add(record_id)
        splits.add(split)
        rows.append(
            {
                "id": record_id,
                "split": str(split),
                "input_text": input_text,
                "forced_decode_text": forced_text,
                "source_mode": source_mode,
            }
        )
    if not {"selection", "fit"}.issubset(splits):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Trajectory texts require at least one selection and one fit row."
        )
    return tuple(rows)


def _tokenize_trajectory_rows(
    rows: Sequence[Mapping[str, str]],
    target_tokenizer: Any,
    *,
    max_decode_steps: int,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        input_ids = catalog._tokenizer_input_ids(
            target_tokenizer,
            row["input_text"],
            add_special_tokens=True,
        )
        forced_ids = catalog._tokenizer_input_ids(
            target_tokenizer,
            row["forced_decode_text"],
            add_special_tokens=False,
        )
        if len(forced_ids) > max_decode_steps:
            raise Lfm2TokenizerBridgeEvidenceError(
                f"Trajectory {row['id']!r} has {len(forced_ids)} target "
                f"tokens, exceeding max_decode_steps={max_decode_steps}."
            )
        result.append(
            {
                "schema": (
                    "neural-forge-lfm2-target-tokenized-trajectory-v1"
                ),
                "id": row["id"],
                "split": row["split"],
                "target_tokenization_mode": (
                    "tokenizer-call-add-special-tokens"
                ),
                "target_forced_decode_tokenization_mode": (
                    "tokenizer-call-no-special-tokens"
                ),
                "target_input_texts": [row["input_text"]],
                "target_forced_decode_texts": [
                    row["forced_decode_text"]
                ],
                "input_ids": list(input_ids),
                "attention_mask": [1] * len(input_ids),
                "forced_decode_ids": list(forced_ids),
            }
        )
    return result


def _mapped_donor_calibration_rows(
    source_rows: Sequence[Mapping[str, Any]],
    adapter: Any,
) -> list[dict[str, Any]]:
    result = []
    for source, record in zip(source_rows, adapter.records):
        parent = record.as_parent_cache_batch()
        result.append(
            {
                "schema": (
                    "neural-forge-lfm2-mapped-donor-trajectory-v1"
                ),
                "id": source["id"],
                "split": source["split"],
                "input_ids": [
                    int(value) for value in parent["input_ids"][0].tolist()
                ],
                "attention_mask": [
                    int(value)
                    for value in parent["attention_mask"][0].tolist()
                ],
                "forced_decode_ids": [
                    int(value)
                    for value in parent["forced_decode_ids"][0].tolist()
                ],
                "target_cache_input_sha256": (
                    record.target_cache_input_sha256
                ),
                "mapped_donor_cache_input_sha256": (
                    record.donor_cache_input_sha256
                ),
                "donor_canonical_sequence_equivalence_claimed": False,
            }
        )
    return result


def _original_envelope_and_baseline(
    config: BridgeEvidencePreparationConfig,
    *,
    tokenizer_fingerprint: str,
) -> tuple[dict[str, Any], Any]:
    source = config.original_parent_outputs
    if _file_sha256(source) != config.original_parent_outputs_sha256:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Original canonical-parent output envelope differs from its "
            "explicit digest."
        )
    envelope = _read_canonical_object(
        source,
        "original canonical-parent output envelope",
    )
    body = dict(envelope)
    seal = body.pop("envelope_sha256", None)
    generation = body.get("generation")
    if (
        body.get("schema") != ENVELOPE_SCHEMA
        or seal != catalog.canonical_sha256(body)
        or body.get("parent_state_fingerprint")
        != config.expected_parent_state_fingerprint
        or body.get("tokenizer_fingerprint") != tokenizer_fingerprint
        or not isinstance(generation, Mapping)
        or generation.get("max_new_tokens") != config.generation_limit
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Original canonical-parent output envelope is not bound to the "
            "expected parent, tokenizer, and generation protocol."
        )
    parent_cache_key = body.get("parent_cache_key")
    baseline = load_lfm2_cached_whole_output_baseline(
        source,
        expected_sha256=config.original_parent_outputs_sha256,
        expected_parent_cache_key=_digest(
            parent_cache_key,
            "original parent_cache_key",
        ),
        expected_parent_state_fingerprint=(
            config.expected_parent_state_fingerprint
        ),
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        expected_generation_limit=config.generation_limit,
        fragment_minimum_characters=(
            config.fragment_minimum_characters
        ),
    )
    return envelope, baseline


def _model_device(model: Any) -> Any:
    devices = {value.device for value in model.parameters()}
    if len(devices) != 1:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Mapped-parent generation requires one materialized model device."
        )
    return next(iter(devices))


def _generate_mapped_parent_rows(
    parent: Any,
    donor_tokenizer: Any,
    tokenizer_bridge: Any,
    original_baseline: Any,
    *,
    maximum_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = _model_device(parent)
    mapping = (
        tokenizer_bridge.artifact.transport_profile
        .target_to_donor_row_indices
    )
    donor_capacity = (
        tokenizer_bridge.artifact.transport_profile.donor_row_capacity
    )
    rows = []
    input_evidence = []
    was_training = bool(parent.training)
    parent.eval()
    try:
        with torch.inference_mode():
            for case in original_baseline.cases:
                target_ids = catalog._tokenizer_input_ids(
                    tokenizer_bridge.target_tokenizer,
                    case.prompt,
                    add_special_tokens=True,
                )
                if len(target_ids) > maximum_prompt_tokens:
                    raise Lfm2TokenizerBridgeEvidenceError(
                        f"Behavior prompt {case.case_id!r} exceeds "
                        "maximum_prompt_tokens."
                    )
                if any(token_id >= len(mapping) for token_id in target_ids):
                    raise Lfm2TokenizerBridgeEvidenceError(
                        f"Behavior prompt {case.case_id!r} exceeds the sealed "
                        "target transport rows."
                    )
                mapped_ids = tuple(mapping[token_id] for token_id in target_ids)
                input_ids = torch.tensor(
                    [mapped_ids],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.ones_like(input_ids)
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": original_baseline.generation_limit,
                    "do_sample": False,
                    "use_cache": True,
                }
                pad_token_id = getattr(donor_tokenizer, "pad_token_id", None)
                if pad_token_id is not None:
                    kwargs["pad_token_id"] = int(pad_token_id)
                generated = parent.generate(**kwargs)
                if (
                    not torch.is_tensor(generated)
                    or generated.ndim != 2
                    or generated.shape[0] != 1
                    or generated.shape[1] <= input_ids.shape[1]
                    or not torch.equal(
                        generated[:, : int(input_ids.shape[1])],
                        input_ids,
                    )
                ):
                    raise Lfm2TokenizerBridgeEvidenceError(
                        f"Mapped-parent generation is invalid for "
                        f"{case.case_id!r}."
                    )
                continuation = tuple(
                    int(value)
                    for value in generated[
                        0, int(input_ids.shape[1]) :
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
                if (
                    len(continuation) > original_baseline.generation_limit
                    or any(
                        token_id < 0 or token_id >= donor_capacity
                        for token_id in continuation
                    )
                ):
                    raise Lfm2TokenizerBridgeEvidenceError(
                        f"Mapped-parent continuation is invalid for "
                        f"{case.case_id!r}."
                    )
                output = donor_tokenizer.decode(
                    continuation,
                    skip_special_tokens=True,
                )
                rows.append(
                    {
                        "id": case.case_id,
                        "prompt": case.prompt,
                        "mapped_parent_output_text": output,
                        "mapped_parent_output_ids": list(continuation),
                        "target_prompt_input_ids": list(target_ids),
                        "mapped_parent_input_ids": list(mapped_ids),
                    }
                )
                input_evidence.append(
                    {
                        "id": case.case_id,
                        "target_prompt_input_ids": list(target_ids),
                        "mapped_parent_input_ids": list(mapped_ids),
                        "mapped_parent_output_ids": list(continuation),
                    }
                )
    finally:
        parent.train(was_training)
    report_body = {
        "schema": "neural-forge-lfm2-mapped-parent-generation-v1",
        "case_count": len(rows),
        "generation_calls": len(rows),
        "generation_limit": original_baseline.generation_limit,
        "inputs": input_evidence,
        "parent_loaded_once": True,
        "parent_execution_phase_count": 1,
        "training_steps": 0,
        "backward_calls": 0,
        "child_constructed": False,
        "fold_invoked": False,
    }
    return rows, {
        **report_body,
        "generation_evidence_sha256": _canonical_sha256(report_body),
    }


def _normalization_config(
    config: BridgeEvidencePreparationConfig,
    *,
    mapped_outputs: Path,
    mapped_outputs_sha256: str,
) -> Any:
    return SimpleNamespace(
        tokenizer_bridge_mapped_parent_outputs=mapped_outputs,
        tokenizer_bridge_mapped_parent_outputs_sha256=(
            mapped_outputs_sha256
        ),
        tokenizer_bridge_minimum_parent_task_retention=(
            config.minimum_parent_task_retention
        ),
        whole_output_maximum_new_generation_limit_saturations=(
            config.maximum_new_generation_limit_saturations
        ),
        whole_output_maximum_repeated_ngram_share_regressions=(
            config.maximum_repeated_ngram_share_regressions
        ),
        whole_output_maximum_repeated_bigram_share_delta=(
            config.maximum_repeated_bigram_share_delta
        ),
        whole_output_fragment_minimum_characters=(
            config.fragment_minimum_characters
        ),
        final_performance_metrics_diagnostic_only=(
            config.final_performance_metrics_diagnostic_only
        ),
    )


def _file_record(output: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(output.resolve(strict=True))
    except ValueError as exc:
        raise Lfm2TokenizerBridgeEvidenceError(
            f"Prepared artifact escaped output_directory: {resolved}."
        ) from exc
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _verify_reusable_manifest(
    config: BridgeEvidencePreparationConfig,
) -> dict[str, Any]:
    output = config.output_directory
    manifest_path = output / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Lfm2TokenizerBridgeEvidenceError(
            "Existing output_directory is partial or lacks its sealed "
            "preparation manifest."
        )
    manifest = _read_canonical_object(
        manifest_path,
        "bridge evidence preparation manifest",
    )
    body = dict(manifest)
    seal = body.pop("manifest_sha256", None)
    if (
        body.get("schema") != MANIFEST_SCHEMA
        or seal != _canonical_sha256(body)
        or body.get("configuration_sha256")
        != _canonical_sha256(config.identity())
        or body.get("status") != "prepared"
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Existing bridge evidence manifest has a different identity or "
            "an invalid seal."
        )
    for path, expected, name in (
        (
            config.target_tokenizer_compilation_receipt,
            config.target_tokenizer_compilation_receipt_sha256,
            "target-tokenizer compilation receipt",
        ),
        (
            config.trajectory_texts,
            config.trajectory_texts_sha256,
            "trajectory text source",
        ),
        (
            config.original_parent_outputs,
            config.original_parent_outputs_sha256,
            "original canonical-parent output envelope",
        ),
    ):
        if _file_sha256(path) != expected:
            raise Lfm2TokenizerBridgeEvidenceError(
                f"Reusable {name} differs from the explicit identity."
            )
    donor_tokenizer = residual.load_local_tokenizer(
        config.donor_tokenizer_directory
    )
    target_tokenizer = residual.load_local_tokenizer(
        config.target_tokenizer_directory
    )
    artifact = load_strict_tokenizer_bridge_artifact(
        config.target_tokenizer_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=config.bridge_artifact_sha256,
    )
    if artifact.profile_fingerprint != config.bridge_profile_sha256:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Reusable target-tokenizer bridge profile changed."
        )
    _verify_parent_physical_row_capacity(config, artifact)
    _verify_compilation_receipt(config, artifact=artifact)
    files = body.get("files")
    if not isinstance(files, list) or not files:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Existing bridge evidence manifest has no file inventory."
        )
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            raise Lfm2TokenizerBridgeEvidenceError(
                f"Manifest file record {index} is malformed."
            )
        relative = record.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise Lfm2TokenizerBridgeEvidenceError(
                f"Manifest file record {index} has an unsafe path."
            )
        path = (output / relative).resolve(strict=True)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _file_sha256(path) != record.get("sha256")
        ):
            raise Lfm2TokenizerBridgeEvidenceError(
                f"Prepared bridge evidence file {relative!r} failed "
                "verification."
            )
    cache_value = body.get("sealed_parent_cache_directory")
    if not isinstance(cache_value, str):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Manifest lacks the sealed parent-cache directory."
        )
    cache_path = Path(cache_value).expanduser().resolve(strict=True)
    try:
        cache_path.relative_to(output.resolve(strict=True))
    except ValueError as exc:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Manifest parent cache escaped output_directory."
        ) from exc
    cache = open_lfm2_circuit_trace_cache(cache_path, verify=True)
    if (
        cache.cache_key != body.get("parent_cache_key")
        or cache.parent_fingerprint
        != config.expected_parent_state_fingerprint
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Reusable parent cache differs from the sealed manifest."
        )
    return {
        **manifest,
        "reused_existing_exact_identity": True,
        "parent_model_loads_this_invocation": 0,
        "parent_execution_phases_this_invocation": 0,
        "parent_generation_calls_this_invocation": 0,
    }


def preflight_bridge_evidence_inputs(
    config: BridgeEvidencePreparationConfig,
) -> dict[str, Any]:
    """Validate every non-model input without creating output or loading parent."""

    donor_tokenizer = residual.load_local_tokenizer(
        config.donor_tokenizer_directory
    )
    target_tokenizer = residual.load_local_tokenizer(
        config.target_tokenizer_directory
    )
    artifact = load_strict_tokenizer_bridge_artifact(
        config.target_tokenizer_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=config.bridge_artifact_sha256,
    )
    if artifact.profile_fingerprint != config.bridge_profile_sha256:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Live tokenizer bridge differs from the explicit profile digest."
        )
    parent_physical_row_capacity = _verify_parent_physical_row_capacity(
        config,
        artifact,
    )
    compilation_receipt = _verify_compilation_receipt(
        config,
        artifact=artifact,
    )
    tokenizer_fingerprint = catalog.fingerprint_local_tokenizer(
        config.donor_tokenizer_directory
    )
    original_envelope, original_baseline = (
        _original_envelope_and_baseline(
            config,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
    )
    trajectory_rows = _read_trajectory_texts(
        config,
        donor_tokenizer=donor_tokenizer,
    )
    original_prompts = {
        str(row["prompt"])
        for row in original_envelope["rows"]
        if isinstance(row, Mapping)
    }
    overlapping = sorted(
        {
            row["input_text"]
            for row in trajectory_rows
            if row["input_text"] in original_prompts
        }
    )
    if overlapping:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Trajectory calibration text overlaps held-out behavior prompts."
        )
    target_rows = _tokenize_trajectory_rows(
        trajectory_rows,
        target_tokenizer,
        max_decode_steps=config.max_decode_steps,
    )
    mode_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in trajectory_rows:
        mode = row["source_mode"]
        split = row["split"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
    body = {
        "schema": "neural-forge-lfm2-bridge-evidence-preflight-v1",
        "status": "preflight_validated",
        "configuration_sha256": _canonical_sha256(config.identity()),
        "bridge_artifact_sha256": artifact.artifact_sha256,
        "bridge_profile_sha256": artifact.profile_fingerprint,
        "parent_physical_vocabulary_row_capacity": (
            parent_physical_row_capacity
        ),
        "target_tokenizer_compilation_receipt_sha256": (
            config.target_tokenizer_compilation_receipt_sha256
        ),
        "target_tokenizer_compilation_receipt_seal": (
            compilation_receipt["receipt_sha256"]
        ),
        "trajectory_texts_sha256": config.trajectory_texts_sha256,
        "trajectory_row_count": len(trajectory_rows),
        "trajectory_source_mode_counts": mode_counts,
        "trajectory_split_counts": split_counts,
        "maximum_target_prompt_tokens": max(
            len(row["input_ids"]) for row in target_rows
        ),
        "maximum_target_forced_decode_tokens": max(
            len(row["forced_decode_ids"]) for row in target_rows
        ),
        "original_parent_outputs_sha256": (
            config.original_parent_outputs_sha256
        ),
        "original_behavior_case_count": len(original_baseline.cases),
        "trace_behavior_prompt_overlap_count": 0,
        "expected_parent_state_fingerprint": (
            config.expected_parent_state_fingerprint
        ),
        "parent_model_loaded": False,
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "output_directory_created": False,
        "capture_causal_circuit_evidence_required": True,
        "child_constructed": False,
        "fold_invoked": False,
        "training_steps": 0,
        "network_access": False,
    }
    return {
        **body,
        "preflight_sha256": _canonical_sha256(body),
    }


def prepare_bridge_evidence(
    config: BridgeEvidencePreparationConfig,
) -> dict[str, Any]:
    """Create or exactly reuse all parent-only tokenizer-bridge evidence."""

    if config.output_directory.exists():
        return _verify_reusable_manifest(config)

    donor_tokenizer = residual.load_local_tokenizer(
        config.donor_tokenizer_directory
    )
    target_tokenizer = residual.load_local_tokenizer(
        config.target_tokenizer_directory
    )
    artifact = load_strict_tokenizer_bridge_artifact(
        config.target_tokenizer_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=config.bridge_artifact_sha256,
    )
    if artifact.profile_fingerprint != config.bridge_profile_sha256:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Live tokenizer bridge differs from the explicit profile digest."
        )
    parent_physical_row_capacity = _verify_parent_physical_row_capacity(
        config,
        artifact,
    )
    compilation_receipt = _verify_compilation_receipt(
        config,
        artifact=artifact,
    )
    tokenizer_fingerprint = catalog.fingerprint_local_tokenizer(
        config.donor_tokenizer_directory
    )
    original_envelope, _original_source_baseline = (
        _original_envelope_and_baseline(
            config,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
    )
    trajectory_rows = _read_trajectory_texts(
        config,
        donor_tokenizer=donor_tokenizer,
    )
    original_prompts = {
        str(row["prompt"])
        for row in original_envelope["rows"]
        if isinstance(row, Mapping)
    }
    overlapping = sorted(
        {
            row["input_text"]
            for row in trajectory_rows
            if row["input_text"] in original_prompts
        }
    )
    if overlapping:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Trajectory calibration text overlaps held-out behavior prompts."
        )
    target_rows = _tokenize_trajectory_rows(
        trajectory_rows,
        target_tokenizer,
        max_decode_steps=config.max_decode_steps,
    )

    dtype = residual._torch_dtype(config.dtype)
    parent = residual.load_local_lfm2_model(
        config.parent_model,
        device=config.device,
        dtype=dtype,
    )
    parent_config = getattr(parent, "config", None)
    if parent_config is None:
        raise Lfm2TokenizerBridgeEvidenceError(
            "Loaded LFM2 parent does not expose its runtime configuration."
        )
    # The installability cache must observe the explicit Q/K/softmax/V path.
    # Transformers may otherwise auto-select SDPA even for a strictly local
    # load.  This changes only the execution backend, before any forward, and
    # does not alter learned state or the authenticated serialized identity.
    parent_config._attn_implementation = "eager"
    if getattr(parent_config, "_attn_implementation", None) != "eager":
        raise Lfm2TokenizerBridgeEvidenceError(
            "Unable to force the eager attention backend required for "
            "causal-circuit installability evidence."
        )
    parent_identity = fingerprint_lfm2_parent(parent)
    if (
        parent_identity.identity_sha256
        != config.expected_parent_state_fingerprint
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Loaded parent state differs from the explicit expected "
            "fingerprint."
        )

    output = config.output_directory
    output.mkdir(parents=True)
    target_calibration_path = output / TARGET_CALIBRATION_FILENAME
    donor_calibration_path = output / DONOR_CALIBRATION_FILENAME
    _write_atomic_json(target_calibration_path, target_rows)
    target_calibration = residual.load_calibration_batches(
        target_calibration_path
    )
    canonical_tokenization_evidence_sha256 = (
        catalog._verify_target_calibration_canonical_tokenization(
            target_calibration_path,
            target_calibration,
            target_tokenizer,
        )
    )
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        artifact,
        catalog._target_tokenized_records(target_calibration),
        expected_artifact_sha256=config.bridge_artifact_sha256,
    )
    donor_rows = _mapped_donor_calibration_rows(target_rows, adapter)
    _write_atomic_json(donor_calibration_path, donor_rows)
    donor_calibration = residual.load_calibration_batches(
        donor_calibration_path
    )
    capture_config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=config.sample_last_tokens,
        max_decode_steps=config.max_decode_steps,
        max_capture_bytes=config.max_capture_bytes,
        logit_rtol=config.capture_logit_rtol,
        logit_atol=config.capture_logit_atol,
        require_top1_agreement=config.require_capture_top1_agreement,
        allow_top1_disagreement_within_numeric_error_envelope=(
            config.allow_capture_top1_disagreement_within_numeric_error_envelope
        ),
        capture_causal_circuit_evidence=(
            config.capture_causal_circuit_evidence
        ),
    )
    parent_cache_batches = tuple(
        record.as_parent_cache_batch() for record in adapter.records
    )
    expected_trace_forward_calls = catalog._expected_capture_forwards(
        parent_cache_batches
    )
    parent_cache = build_lfm2_circuit_trace_cache(
        output / PARENT_CACHE_ROOT_NAME,
        parent,
        parent_cache_batches,
        config=capture_config,
        input_device=torch.device(config.device),
    )
    if tuple(donor_calibration.splits) != tuple(target_calibration.splits):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Mapped donor and target calibration splits differ."
        )
    bridge_config = SimpleNamespace(
        uses_tokenizer_bridge=True,
        target_tokenizer_directory=config.target_tokenizer_directory,
        target_tokenizer_calibration=target_calibration_path,
        tokenizer_bridge_artifact_sha256=config.bridge_artifact_sha256,
        tokenizer_bridge_profile_sha256=config.bridge_profile_sha256,
    )
    bridge_runtime = catalog.resolve_production_tokenizer_bridge(
        bridge_config,
        donor_tokenizer=donor_tokenizer,
        parent_cache=parent_cache,
        calibration_splits=target_calibration.splits,
    )
    if (
        bridge_runtime is None
        or bridge_runtime.canonical_tokenization_evidence_sha256
        != canonical_tokenization_evidence_sha256
    ):
        raise Lfm2TokenizerBridgeEvidenceError(
            "Prepared bridge runtime did not reproduce canonical trace "
            "evidence."
        )

    original_rows_path = output / ORIGINAL_ROWS_FILENAME
    original_envelope_path = output / ORIGINAL_ENVELOPE_FILENAME
    _write_atomic_json(original_rows_path, original_envelope["rows"])
    original_receipt = seal_lfm2_cached_whole_output_envelope(
        original_rows_path,
        original_envelope_path,
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        generation_limit=config.generation_limit,
    )
    bridge_original_baseline = load_lfm2_cached_whole_output_baseline(
        original_envelope_path,
        expected_sha256=original_receipt["file_sha256"],
        expected_parent_cache_key=parent_cache.cache_key,
        expected_parent_state_fingerprint=parent_cache.parent_fingerprint,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        expected_generation_limit=config.generation_limit,
        fragment_minimum_characters=(
            config.fragment_minimum_characters
        ),
    )
    mapped_rows, generation_report = _generate_mapped_parent_rows(
        parent,
        donor_tokenizer,
        bridge_runtime,
        bridge_original_baseline,
        maximum_prompt_tokens=config.maximum_prompt_tokens,
    )
    mapped_rows_path = output / MAPPED_ROWS_FILENAME
    mapped_envelope_path = output / MAPPED_ENVELOPE_FILENAME
    _write_atomic_json(mapped_rows_path, mapped_rows)
    mapped_receipt = (
        catalog.seal_lfm2_tokenizer_bridge_mapped_parent_envelope(
            mapped_rows_path,
            mapped_envelope_path,
            original_parent_baseline=bridge_original_baseline,
            tokenizer_bridge=bridge_runtime,
            parent_cache_key=parent_cache.cache_key,
            parent_state_fingerprint=parent_cache.parent_fingerprint,
        )
    )
    _mapped_baseline, normalization_control = (
        catalog._load_bridge_mapped_parent_baseline(
            _normalization_config(
                config,
                mapped_outputs=mapped_envelope_path,
                mapped_outputs_sha256=mapped_receipt["file_sha256"],
            ),
            original_baseline=bridge_original_baseline,
            tokenizer_bridge=bridge_runtime,
            parent_cache=parent_cache,
        )
    )
    parent_after = fingerprint_lfm2_parent(parent)
    if parent_after.as_dict() != parent_identity.as_dict():
        raise Lfm2TokenizerBridgeEvidenceError(
            "Parent state changed during bridge evidence preparation."
        )

    prepared_files = [
        target_calibration_path,
        donor_calibration_path,
        original_rows_path,
        original_envelope_path,
        mapped_rows_path,
        mapped_envelope_path,
    ]
    fold_arguments = {
        "--parent-model": str(config.parent_model),
        "--tokenizer-directory": str(config.donor_tokenizer_directory),
        "--target-tokenizer-directory": str(
            config.target_tokenizer_directory
        ),
        "--target-tokenizer-calibration": str(target_calibration_path),
        "--tokenizer-bridge-artifact-sha256": (
            config.bridge_artifact_sha256
        ),
        "--tokenizer-bridge-profile-sha256": (
            config.bridge_profile_sha256
        ),
        "--tokenizer-bridge-mapped-parent-outputs": str(
            mapped_envelope_path
        ),
        "--tokenizer-bridge-mapped-parent-outputs-sha256": (
            mapped_receipt["file_sha256"]
        ),
        "--tokenizer-bridge-minimum-parent-task-retention": (
            config.minimum_parent_task_retention
        ),
        "--calibration": str(donor_calibration_path),
        "--sealed-parent-cache-directory": str(parent_cache.path),
        "--cached-parent-outputs": str(original_envelope_path),
        "--cached-parent-outputs-sha256": original_receipt["file_sha256"],
        "--whole-output-max-new-tokens": config.generation_limit,
        "--whole-output-maximum-prompt-tokens": (
            config.maximum_prompt_tokens
        ),
    }
    body = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "configuration": config.identity(),
        "configuration_sha256": _canonical_sha256(config.identity()),
        "target_tokenizer_compilation_receipt_sha256": (
            config.target_tokenizer_compilation_receipt_sha256
        ),
        "target_tokenizer_compilation_receipt_seal": (
            compilation_receipt["receipt_sha256"]
        ),
        "bridge_artifact_sha256": artifact.artifact_sha256,
        "bridge_profile_sha256": artifact.profile_fingerprint,
        "parent_physical_vocabulary_row_capacity": (
            parent_physical_row_capacity
        ),
        "donor_tokenizer_semantics_sha256": (
            fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        ),
        "target_tokenizer_semantics_sha256": (
            fingerprint_hf_tokenizer_semantics(target_tokenizer)
        ),
        "parent_state_fingerprint": parent_cache.parent_fingerprint,
        "parent_state_unchanged_after_trace_and_generation": True,
        "parent_cache_key": parent_cache.cache_key,
        "parent_cache_corpus_fingerprint": (
            parent_cache.corpus_fingerprint
        ),
        "sealed_parent_cache_directory": str(parent_cache.path),
        "target_calibration_sha256": _file_sha256(
            target_calibration_path
        ),
        "mapped_donor_calibration_sha256": _file_sha256(
            donor_calibration_path
        ),
        "trace_adapter_evidence_sha256": bridge_runtime.evidence_sha256,
        "canonical_target_tokenization_evidence_sha256": (
            canonical_tokenization_evidence_sha256
        ),
        "original_parent_outputs_file_sha256": (
            original_receipt["file_sha256"]
        ),
        "original_parent_outputs_envelope_sha256": (
            original_receipt["envelope_sha256"]
        ),
        "mapped_parent_outputs_file_sha256": (
            mapped_receipt["file_sha256"]
        ),
        "mapped_parent_outputs_envelope_sha256": (
            mapped_receipt["envelope_sha256"]
        ),
        "mapped_parent_generation": generation_report,
        "normalization_control": dict(normalization_control),
        "fold_runner_arguments": fold_arguments,
        "files": [_file_record(output, path) for path in prepared_files],
        "execution_audit": {
            "parent_model_loads": 1,
            "parent_execution_phases": 1,
            "parent_trace_cache_build_calls": 1,
            "parent_trace_capture_forward_calls": (
                expected_trace_forward_calls
            ),
            "causal_circuit_installability_evidence_requested": True,
            "parent_behavior_generation_calls": len(mapped_rows),
            "canonical_parent_behavior_generation_calls": 0,
            "parent_outputs_reused_after_seal": True,
            "child_models_constructed": 0,
            "fold_invocations": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "model_parameters_pruned": 0,
            "model_parameters_quantized": 0,
            "network_access": False,
        },
    }
    manifest = {
        **body,
        "manifest_sha256": _canonical_sha256(body),
    }
    _write_atomic_json(output / MANIFEST_FILENAME, manifest)
    return {
        **manifest,
        "reused_existing_exact_identity": False,
        "parent_model_loads_this_invocation": 1,
        "parent_execution_phases_this_invocation": 1,
        "parent_generation_calls_this_invocation": len(mapped_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one immutable bridge-specific LFM2 parent trace cache and "
            "mapped-parent behavior envelope; no child and no fold."
        )
    )
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument(
        "--donor-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target-tokenizer-compilation-receipt",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target-tokenizer-compilation-receipt-sha256",
        required=True,
    )
    parser.add_argument("--bridge-artifact-sha256", required=True)
    parser.add_argument("--bridge-profile-sha256", required=True)
    parser.add_argument("--trajectory-texts", required=True, type=Path)
    parser.add_argument("--trajectory-texts-sha256", required=True)
    parser.add_argument(
        "--original-parent-outputs",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--original-parent-outputs-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-parent-state-fingerprint",
        required=True,
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--sample-last-tokens", type=int, default=8)
    parser.add_argument("--max-decode-steps", type=int, default=32)
    parser.add_argument(
        "--max-capture-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument(
        "--capture-logit-rtol",
        type=float,
        default=catalog.DEFAULT_CAPTURE_LOGIT_RTOL,
    )
    parser.add_argument(
        "--capture-logit-atol",
        type=float,
        default=catalog.DEFAULT_CAPTURE_LOGIT_ATOL,
    )
    parser.add_argument(
        "--require-capture-top1-agreement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-capture-top1-disagreement-within-numeric-error-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Treat a cached/full-prefix winner flip as verified only when "
            "both candidates are inside their measured numerical error "
            "envelope; exact winner agreement remains recorded separately."
        ),
    )
    parser.add_argument(
        "--capture-causal-circuit-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Required production setting: retain exact causal attention, "
            "short-convolution state, and final-normalization installability "
            "evidence in the immutable parent cache."
        ),
    )
    parser.add_argument("--generation-limit", type=int, default=64)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--minimum-parent-task-retention",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--fragment-minimum-characters",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--maximum-new-generation-limit-saturations",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--maximum-repeated-ngram-share-regressions",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--maximum-repeated-bigram-share-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate all tokenizer, text, receipt, and canonical-parent "
            "inputs without loading or executing the parent or creating "
            "output."
        ),
    )
    return parser


def config_from_args(
    args: argparse.Namespace,
) -> BridgeEvidencePreparationConfig:
    return BridgeEvidencePreparationConfig(
        parent_model=args.parent_model,
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        target_tokenizer_compilation_receipt=(
            args.target_tokenizer_compilation_receipt
        ),
        target_tokenizer_compilation_receipt_sha256=(
            args.target_tokenizer_compilation_receipt_sha256
        ),
        bridge_artifact_sha256=args.bridge_artifact_sha256,
        bridge_profile_sha256=args.bridge_profile_sha256,
        trajectory_texts=args.trajectory_texts,
        trajectory_texts_sha256=args.trajectory_texts_sha256,
        original_parent_outputs=args.original_parent_outputs,
        original_parent_outputs_sha256=(
            args.original_parent_outputs_sha256
        ),
        expected_parent_state_fingerprint=(
            args.expected_parent_state_fingerprint
        ),
        output_directory=args.output_directory,
        device=args.device,
        dtype=args.dtype,
        sample_last_tokens=args.sample_last_tokens,
        max_decode_steps=args.max_decode_steps,
        max_capture_bytes=args.max_capture_bytes,
        capture_logit_rtol=args.capture_logit_rtol,
        capture_logit_atol=args.capture_logit_atol,
        require_capture_top1_agreement=(
            args.require_capture_top1_agreement
        ),
        allow_capture_top1_disagreement_within_numeric_error_envelope=(
            args.allow_capture_top1_disagreement_within_numeric_error_envelope
        ),
        capture_causal_circuit_evidence=(
            args.capture_causal_circuit_evidence
        ),
        generation_limit=args.generation_limit,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
        minimum_parent_task_retention=(
            args.minimum_parent_task_retention
        ),
        fragment_minimum_characters=args.fragment_minimum_characters,
        maximum_new_generation_limit_saturations=(
            args.maximum_new_generation_limit_saturations
        ),
        maximum_repeated_ngram_share_regressions=(
            args.maximum_repeated_ngram_share_regressions
        ),
        maximum_repeated_bigram_share_delta=(
            args.maximum_repeated_bigram_share_delta
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        receipt = (
            preflight_bridge_evidence_inputs(config)
            if args.preflight_only
            else prepare_bridge_evidence(config)
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": MANIFEST_SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "child_constructed": False,
                    "fold_invoked": False,
                    "training_steps": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
