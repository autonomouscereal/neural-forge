#!/usr/bin/env python3
"""Capture one sealed holdout from an already-frozen local CPCF child.

This command is deliberately post-hoc.  It authenticates the external and
checkpoint-local CPCF receipts, the exact saved checkpoint files, the target
tokenizer package and semantics, and the strict tokenizer bridge before it
loads the child.  It then performs exactly one greedy generation per manifest
case, proves that the child state and gradients did not change, and publishes
an existing-format candidate artifact plus a sealed capture receipt.

There is no parent loader, trace-cache builder, optimizer, backward pass,
training loop, pruning, quantization, network path, or hub loader here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


import torch  # noqa: E402

try:
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
    from experiments import run_lfm2_cpcf_blank_fold as cpcf_runner
    from experiments import run_lfm2_residual_attention_fold as fold_runtime
except ImportError:  # Direct execution from the experiments directory.
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_cpcf_blank_fold as cpcf_runner
    import run_lfm2_residual_attention_fold as fold_runtime

from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_lfm2_intelligence_gate import (  # noqa: E402
    IntelligenceHoldoutManifest,
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (  # noqa: E402
    load_lfm2_intelligence_holdout_candidate_outputs,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (  # noqa: E402
    independent_parameter_inventory,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
    strict_tokenizer_bridge_configuration,
)


CAPTURE_SCHEMA = "neural-forge-lfm2-cpcf-holdout-candidate-capture-v1"
INTERNAL_CPCF_RECEIPT = "cpcf_run_receipt.json"


class Lfm2CpcfHoldoutCaptureError(RuntimeError):
    """Frozen-child provenance or one-time candidate capture failed closed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_file_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value)) + b"\n").hexdigest()


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} must be an explicit local directory."
        )
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} does not exist."
        ) from exc
    if not path.is_dir() or path.is_symlink():
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} is missing or unsafe: {path}."
        )
    return path


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} does not exist."
        ) from exc
    if not path.is_file() or path.is_symlink():
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} is missing or unsafe: {path}."
        )
    return path


def _new_output_file(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} already exists; refusing to overwrite it: {path}."
        )
    if not path.name:
        raise Lfm2CpcfHoldoutCaptureError(f"{name} must name a file.")
    return path


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _read_canonical_json(path: Path, name: str) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} is not canonical ASCII JSON."
        ) from exc
    if not isinstance(document, Mapping):
        raise Lfm2CpcfHoldoutCaptureError(f"{name} must be a JSON object.")
    canonical = canonical_json_bytes(document)
    if raw != canonical + b"\n":
        raise Lfm2CpcfHoldoutCaptureError(
            f"{name} is not encoded exactly as the CPCF canonical producer."
        )
    return raw, dict(document)


def _stage_new_json(path: Path, value: Mapping[str, Any]) -> Path:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise Lfm2CpcfHoldoutCaptureError(
            f"Refusing to replace existing output {destination}."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return temporary


def _publish_new_documents(
    documents: Sequence[tuple[Path, Mapping[str, Any]]],
) -> None:
    destinations = tuple(path.expanduser().absolute() for path, _ in documents)
    if len(set(destinations)) != len(destinations):
        raise Lfm2CpcfHoldoutCaptureError(
            "Candidate and capture-receipt destinations must differ."
        )
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise Lfm2CpcfHoldoutCaptureError(
                f"Refusing to replace existing output {destination}."
            )
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, (_, document) in zip(destinations, documents):
            staged.append((destination, _stage_new_json(destination, document)))
        for destination, temporary in staged:
            os.link(temporary, destination)
            published.append(destination)
    except BaseException:
        for destination in published:
            if destination.exists():
                destination.unlink()
        raise
    finally:
        for _destination, temporary in staged:
            if temporary.exists():
                temporary.unlink()


def _checkpoint_inventory(
    directory: Path,
    *,
    ignored_relative_paths: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], ...]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2CpcfHoldoutCaptureError(
                f"Frozen checkpoint contains a symlink: {path}."
            )
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in ignored_relative_paths:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise Lfm2CpcfHoldoutCaptureError(
            "Frozen checkpoint contains no receiptable saved files."
        )
    return tuple(rows)


def _typed_receipt_inventory(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise Lfm2CpcfHoldoutCaptureError(
            "CPCF receipt has no saved checkpoint inventory."
        )
    rows = []
    previous = None
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise Lfm2CpcfHoldoutCaptureError(
                "CPCF saved-file inventory fields are invalid."
            )
        relative = raw["relative_path"]
        size = raw["size_bytes"]
        sha256 = raw["sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
            or relative == INTERNAL_CPCF_RECEIPT
        ):
            raise Lfm2CpcfHoldoutCaptureError(
                "CPCF saved-file inventory contains an unsafe path."
            )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise Lfm2CpcfHoldoutCaptureError(
                "CPCF saved-file inventory contains an invalid size."
            )
        _digest(sha256, "saved checkpoint file sha256")
        if previous is not None and relative <= previous:
            raise Lfm2CpcfHoldoutCaptureError(
                "CPCF saved-file inventory is not uniquely sorted."
            )
        previous = relative
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": sha256,
            }
        )
    return tuple(rows)


def _require_zero_operation_audit(document: Mapping[str, Any]) -> None:
    claims = document.get("claims")
    expected_claims = {
        "folding_only": True,
        "blank_target_filled": True,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "parent_backed_fallback_used": False,
    }
    if not isinstance(claims, Mapping) or any(
        claims.get(name) is not value
        for name, value in expected_claims.items()
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "CPCF receipt does not prove a folding-only blank child."
        )
    audit = document.get("operation_audit")
    expected_audit = {
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "parent_cache_generation_calls": 0,
        "parent_cache_regeneration_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "quantized_parameters": 0,
        "pruned_parameters": 0,
        "fallback_model_calls": 0,
        "network_access": False,
    }
    if not isinstance(audit, Mapping) or any(
        audit.get(name) != value for name, value in expected_audit.items()
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "CPCF receipt operation audit permits a forbidden operation."
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedCpcfReceipt:
    document: Mapping[str, Any]
    receipt_sha256: str
    external_file_sha256: str
    internal_file_sha256: str
    checkpoint_inventory: tuple[Mapping[str, Any], ...]
    checkpoint_inventory_sha256: str
    expected_parameter_inventory: Mapping[str, Any]
    materialized_ffn_width: int


def _authenticate_cpcf_receipt(
    *,
    checkpoint_directory: Path,
    external_receipt: Path,
    expected_external_file_sha256: str,
    expected_receipt_sha256: str,
    expected_bridge_artifact_sha256: str,
    expected_bridge_profile_sha256: str,
) -> AuthenticatedCpcfReceipt:
    internal_receipt = _local_file(
        checkpoint_directory / INTERNAL_CPCF_RECEIPT,
        "internal_cpcf_receipt",
    )
    if external_receipt == internal_receipt:
        raise Lfm2CpcfHoldoutCaptureError(
            "External and internal CPCF receipts must be distinct files."
        )
    external_raw, external = _read_canonical_json(
        external_receipt,
        "external CPCF receipt",
    )
    internal_raw, internal = _read_canonical_json(
        internal_receipt,
        "internal CPCF receipt",
    )
    external_file_sha = _file_sha256(external_receipt)
    expected_file_sha = _digest(
        expected_external_file_sha256,
        "expected_cpcf_receipt_file_sha256",
    )
    if external_file_sha != expected_file_sha:
        raise Lfm2CpcfHoldoutCaptureError(
            "External CPCF receipt file differs from its expected SHA-256."
        )
    internal_file_sha = _file_sha256(internal_receipt)
    if (
        external_raw != internal_raw
        or external_file_sha != internal_file_sha
        or canonical_json_bytes(external) != canonical_json_bytes(internal)
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "External and internal CPCF receipts are not identical."
        )
    core = dict(external)
    seal = core.pop("receipt_sha256", None)
    expected_seal = _digest(
        expected_receipt_sha256,
        "expected_cpcf_receipt_sha256",
    )
    if (
        external.get("schema") != cpcf_runner.RUN_SCHEMA
        or seal != expected_seal
        or seal != canonical_sha256(core)
        or external.get("status")
        not in {
            "folded_child_requires_manual_intelligence_review",
            "diagnostic_control_fold_requires_manual_review",
        }
        or external.get("fresh_reload_verified") is not True
        or external.get("structural_parameter_compaction_proven") is not True
        or external.get("parent_state_preserved") is not True
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "CPCF receipt seal or frozen-child contract is invalid."
        )
    inputs = external.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("bridge_artifact_sha256")
        != expected_bridge_artifact_sha256
        or inputs.get("bridge_profile_sha256")
        != expected_bridge_profile_sha256
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "CPCF receipt differs from the expected strict tokenizer bridge."
        )
    _require_zero_operation_audit(external)
    receipted_inventory = _typed_receipt_inventory(
        external.get("saved_checkpoint_files")
    )
    observed_inventory = _checkpoint_inventory(
        checkpoint_directory,
        ignored_relative_paths=frozenset({INTERNAL_CPCF_RECEIPT}),
    )
    if canonical_json_bytes(receipted_inventory) != canonical_json_bytes(
        observed_inventory
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Frozen checkpoint files differ from the CPCF receipt."
        )
    expected_parameter_inventory = external.get(
        "reloaded_parameter_inventory"
    )
    if not isinstance(expected_parameter_inventory, Mapping):
        raise Lfm2CpcfHoldoutCaptureError(
            "CPCF receipt has no reloaded parameter inventory."
        )
    aliases = external.get("save_reload_config_aliases")
    fresh_aliases = (
        aliases.get("fresh_reload") if isinstance(aliases, Mapping) else None
    )
    materialized_ffn_width = (
        fresh_aliases.get("materialized_ffn_width")
        if isinstance(fresh_aliases, Mapping)
        else None
    )
    if (
        isinstance(materialized_ffn_width, bool)
        or not isinstance(materialized_ffn_width, int)
        or materialized_ffn_width <= 0
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "CPCF receipt has no valid fresh-reload FFN width."
        )
    return AuthenticatedCpcfReceipt(
        document=external,
        receipt_sha256=str(seal),
        external_file_sha256=external_file_sha,
        internal_file_sha256=internal_file_sha,
        checkpoint_inventory=tuple(observed_inventory),
        checkpoint_inventory_sha256=canonical_sha256(observed_inventory),
        expected_parameter_inventory=dict(expected_parameter_inventory),
        materialized_ffn_width=materialized_ffn_width,
    )


def load_capture_manifest(
    source: Path | None,
    *,
    expected_manifest_sha256: str,
) -> IntelligenceHoldoutManifest:
    """Load the strict v1 suite through the sole future-version dispatch seam.

    The current implementation intentionally delegates to the exact v1 loader,
    which still requires byte-for-byte equality with the built-in 64-case
    suite.  A future v2 loader can be dispatched here without weakening v1.
    """

    return load_lfm2_intelligence_holdout_manifest(
        source,
        expected_manifest_sha256=expected_manifest_sha256,
    )


def seal_capture_candidate_outputs(
    manifest: IntelligenceHoldoutManifest,
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_state_fingerprint: str,
    tokenizer_fingerprint: str,
    eos_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Route candidate sealing through one future manifest-version seam."""

    return seal_lfm2_intelligence_holdout_candidate_outputs(
        manifest,
        rows,
        candidate_state_fingerprint=candidate_state_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        eos_token_ids=eos_token_ids,
    )


def load_sealed_capture_candidate_outputs(
    source: Mapping[str, Any],
    *,
    expected_candidate_outputs_sha256: str,
    expected_manifest_sha256: str,
    expected_candidate_state_fingerprint: str,
    expected_tokenizer_fingerprint: str,
    eos_token_ids: Sequence[int],
) -> Any:
    """Round-trip through the sole future-version candidate loader seam."""

    return load_lfm2_intelligence_holdout_candidate_outputs(
        source,
        expected_candidate_outputs_sha256=(
            expected_candidate_outputs_sha256
        ),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_candidate_state_fingerprint=(
            expected_candidate_state_fingerprint
        ),
        expected_tokenizer_fingerprint=expected_tokenizer_fingerprint,
        eos_token_ids=eos_token_ids,
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise Lfm2CpcfHoldoutCaptureError(
            "The frozen child has no Parameters."
        ) from exc


def _normalized_eos_token_id(tokenizer: Any) -> int | list[int] | None:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [int(item) for item in value]
        return values or None
    if value is None:
        return None
    return int(value)


def generate_candidate_rows_once(
    model: torch.nn.Module,
    tokenizer: Any,
    manifest: Any,
) -> tuple[dict[str, Any], ...]:
    """Greedily generate exactly once at each manifest case's own limit."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    cases = tuple(getattr(manifest, "cases", ()))
    if not cases:
        raise Lfm2CpcfHoldoutCaptureError("Manifest contains no cases.")
    device = _model_device(model)
    was_training = bool(model.training)
    model.eval()
    eos_token_id = _normalized_eos_token_id(tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        if isinstance(eos_token_id, list):
            pad_token_id = eos_token_id[0] if eos_token_id else None
        else:
            pad_token_id = eos_token_id
    rows = []
    try:
        with torch.inference_mode():
            for spec in cases:
                case_id = getattr(spec, "case_id", None)
                prompt = getattr(spec, "prompt", None)
                generation_limit = getattr(spec, "generation_limit", None)
                if (
                    not isinstance(case_id, str)
                    or not case_id
                    or not isinstance(prompt, str)
                    or not prompt
                    or isinstance(generation_limit, bool)
                    or not isinstance(generation_limit, int)
                    or generation_limit <= 0
                ):
                    raise Lfm2CpcfHoldoutCaptureError(
                        "Manifest case identity, prompt, or limit is invalid."
                    )
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                if (
                    not isinstance(encoded, Mapping)
                    or not isinstance(encoded.get("input_ids"), torch.Tensor)
                    or encoded["input_ids"].ndim != 2
                    or int(encoded["input_ids"].shape[0]) != 1
                ):
                    raise Lfm2CpcfHoldoutCaptureError(
                        f"Target tokenizer output is invalid for {case_id}."
                    )
                input_ids = encoded["input_ids"].to(
                    device=device,
                    dtype=torch.long,
                )
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                elif not isinstance(attention_mask, torch.Tensor):
                    raise Lfm2CpcfHoldoutCaptureError(
                        f"Target attention mask is invalid for {case_id}."
                    )
                attention_mask = attention_mask.to(
                    device=device,
                    dtype=torch.long,
                )
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": generation_limit,
                    "do_sample": False,
                    "use_cache": True,
                }
                if pad_token_id is not None:
                    kwargs["pad_token_id"] = int(pad_token_id)
                if eos_token_id is not None:
                    kwargs["eos_token_id"] = eos_token_id
                generated = model.generate(**kwargs)
                if (
                    not isinstance(generated, torch.Tensor)
                    or generated.ndim != 2
                    or int(generated.shape[0]) != 1
                    or int(generated.shape[1]) < int(input_ids.shape[1])
                ):
                    raise Lfm2CpcfHoldoutCaptureError(
                        f"Child generation is invalid for {case_id}."
                    )
                if not torch.equal(
                    generated[0, : int(input_ids.shape[1])],
                    input_ids[0],
                ):
                    raise Lfm2CpcfHoldoutCaptureError(
                        f"Child generation changed the prompt prefix for {case_id}."
                    )
                continuation = generated[0, int(input_ids.shape[1]) :]
                output_token_ids = [
                    int(item)
                    for item in continuation.detach().to("cpu").tolist()
                ]
                if len(output_token_ids) > generation_limit:
                    raise Lfm2CpcfHoldoutCaptureError(
                        f"Child exceeded the generation limit for {case_id}."
                    )
                output = tokenizer.decode(
                    output_token_ids,
                    skip_special_tokens=True,
                )
                if not isinstance(output, str):
                    raise Lfm2CpcfHoldoutCaptureError(
                        f"Decoded child output is invalid for {case_id}."
                    )
                rows.append(
                    {
                        "case_id": case_id,
                        "input_token_count": int(input_ids.shape[1]),
                        "output": output,
                        "output_token_ids": output_token_ids,
                        "generation_limit": generation_limit,
                        "topical_relevance": None,
                    }
                )
    finally:
        model.train(was_training)
    if len(rows) != len(cases):
        raise Lfm2CpcfHoldoutCaptureError(
            "Child generation did not cover the exact manifest."
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    checkpoint_directory: Path
    target_tokenizer_directory: Path
    bridge_directory: Path
    cpcf_receipt: Path
    expected_cpcf_receipt_file_sha256: str
    expected_cpcf_receipt_sha256: str
    expected_manifest_sha256: str
    expected_target_tokenizer_fingerprint: str
    expected_target_tokenizer_semantics_sha256: str
    expected_bridge_artifact_sha256: str
    expected_bridge_profile_sha256: str
    candidate_output_path: Path
    capture_receipt_path: Path
    manifest_path: Path | None = None
    expected_checkpoint_tokenizer_fingerprint: str | None = None
    expected_candidate_state_fingerprint: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        checkpoint = _local_directory(
            self.checkpoint_directory,
            "checkpoint_directory",
        )
        tokenizer = _local_directory(
            self.target_tokenizer_directory,
            "target_tokenizer_directory",
        )
        bridge = _local_directory(
            self.bridge_directory,
            "bridge_directory",
        )
        receipt = _local_file(self.cpcf_receipt, "cpcf_receipt")
        internal_receipt = checkpoint / INTERNAL_CPCF_RECEIPT
        if receipt == internal_receipt:
            raise Lfm2CpcfHoldoutCaptureError(
                "cpcf_receipt must be an external copy, not the internal file."
            )
        manifest = (
            None
            if self.manifest_path is None
            else _local_file(self.manifest_path, "manifest_path")
        )
        candidate = _new_output_file(
            self.candidate_output_path,
            "candidate_output_path",
        )
        capture = _new_output_file(
            self.capture_receipt_path,
            "capture_receipt_path",
        )
        if candidate == capture:
            raise Lfm2CpcfHoldoutCaptureError(
                "Candidate and capture receipt destinations must differ."
            )
        for output in (candidate, capture):
            if any(
                _is_within(output, directory)
                for directory in (checkpoint, tokenizer, bridge)
            ):
                raise Lfm2CpcfHoldoutCaptureError(
                    "Capture outputs cannot be written inside authenticated inputs."
                )
        for name in (
            "expected_cpcf_receipt_file_sha256",
            "expected_cpcf_receipt_sha256",
            "expected_manifest_sha256",
            "expected_target_tokenizer_fingerprint",
            "expected_target_tokenizer_semantics_sha256",
            "expected_bridge_artifact_sha256",
            "expected_bridge_profile_sha256",
        ):
            _digest(getattr(self, name), name)
        if self.expected_candidate_state_fingerprint is not None:
            _digest(
                self.expected_candidate_state_fingerprint,
                "expected_candidate_state_fingerprint",
            )
        if self.expected_checkpoint_tokenizer_fingerprint is not None:
            _digest(
                self.expected_checkpoint_tokenizer_fingerprint,
                "expected_checkpoint_tokenizer_fingerprint",
            )
        if not isinstance(self.device, str) or not self.device:
            raise Lfm2CpcfHoldoutCaptureError("device must be nonempty.")
        fold_runtime._torch_dtype(self.dtype)
        object.__setattr__(self, "checkpoint_directory", checkpoint)
        object.__setattr__(self, "target_tokenizer_directory", tokenizer)
        object.__setattr__(self, "bridge_directory", bridge)
        object.__setattr__(self, "cpcf_receipt", receipt)
        object.__setattr__(self, "manifest_path", manifest)
        object.__setattr__(self, "candidate_output_path", candidate)
        object.__setattr__(self, "capture_receipt_path", capture)

    @property
    def checkpoint_tokenizer_fingerprint(self) -> str:
        """Return the independently expected generation-tokenizer package."""

        return (
            self.expected_target_tokenizer_fingerprint
            if self.expected_checkpoint_tokenizer_fingerprint is None
            else self.expected_checkpoint_tokenizer_fingerprint
        )


def _authenticate_tokenizer_package(
    config: CaptureConfig,
    bridge: Any,
) -> tuple[Any, tuple[int, ...], Mapping[str, Any]]:
    source_fingerprint, source_files = (
        posthoc.fingerprint_bound_local_tokenizer(
            config.target_tokenizer_directory
        )
    )
    checkpoint_fingerprint, checkpoint_files = (
        posthoc.fingerprint_bound_local_tokenizer(
            config.checkpoint_directory
        )
    )
    expected_source_package = config.expected_target_tokenizer_fingerprint
    expected_checkpoint_package = (
        config.checkpoint_tokenizer_fingerprint
    )
    if (
        source_fingerprint != expected_source_package
        or checkpoint_fingerprint != expected_checkpoint_package
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Source or checkpoint target tokenizer package differs from its "
            "separately expected identity."
        )
    source_special = posthoc.load_bound_special_token_ids(
        config.target_tokenizer_directory,
        expected_tokenizer_fingerprint=expected_source_package,
    )
    checkpoint_special = posthoc.load_bound_special_token_ids(
        config.checkpoint_directory,
        expected_tokenizer_fingerprint=expected_checkpoint_package,
    )
    source_special_ids = source_special.get("special_token_ids")
    checkpoint_special_ids = checkpoint_special.get("special_token_ids")
    if (
        not isinstance(source_special_ids, Mapping)
        or not isinstance(checkpoint_special_ids, Mapping)
        or canonical_json_bytes(source_special_ids)
        != canonical_json_bytes(checkpoint_special_ids)
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Source and checkpoint target tokenizer special IDs differ."
        )
    eos_values = checkpoint_special_ids.get("eos")
    if (
        not isinstance(eos_values, list)
        or not eos_values
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in eos_values
        )
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Target tokenizer has no authenticated EOS ID."
        )
    source_tokenizer = fold_runtime.load_local_tokenizer(
        config.target_tokenizer_directory
    )
    if config.target_tokenizer_directory == config.checkpoint_directory:
        checkpoint_tokenizer = source_tokenizer
        tokenizer_load_count = 1
    else:
        checkpoint_tokenizer = fold_runtime.load_local_tokenizer(
            config.checkpoint_directory
        )
        tokenizer_load_count = 2
    source_semantics = fingerprint_hf_tokenizer_semantics(source_tokenizer)
    checkpoint_semantics = fingerprint_hf_tokenizer_semantics(
        checkpoint_tokenizer
    )
    live_eos = _normalized_eos_token_id(checkpoint_tokenizer)
    live_eos_values = (
        live_eos if isinstance(live_eos, list) else [live_eos]
    )
    expected_semantics = config.expected_target_tokenizer_semantics_sha256
    if (
        source_semantics != expected_semantics
        or checkpoint_semantics != expected_semantics
        or bridge.target_tokenizer_semantics_sha256 != expected_semantics
        or live_eos_values != eos_values
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Live target tokenizer semantics or EOS identity differ from the "
            "strict bridge and package."
        )
    metadata = {
        "source_package_fingerprint": expected_source_package,
        "checkpoint_package_fingerprint": expected_checkpoint_package,
        "source_package_files": [dict(value) for value in source_files],
        "checkpoint_package_files": [
            dict(value) for value in checkpoint_files
        ],
        "semantics_sha256": expected_semantics,
        "source_special_token_identity_sha256": source_special.get(
            "special_token_identity_sha256"
        ),
        "checkpoint_special_token_identity_sha256": checkpoint_special.get(
            "special_token_identity_sha256"
        ),
        "eos_token_ids": list(eos_values),
        "tokenizer_load_count": tokenizer_load_count,
        "source_and_checkpoint_packages_identical": (
            expected_source_package == expected_checkpoint_package
        ),
        "source_and_checkpoint_semantics_identical": True,
    }
    return checkpoint_tokenizer, tuple(eos_values), metadata


def _verify_model_bridge_configuration(
    model: torch.nn.Module,
    bridge: Any,
) -> Mapping[str, Any]:
    expected = strict_tokenizer_bridge_configuration(bridge)
    model_config = getattr(model, "config", None)
    observed = {
        name: getattr(model_config, name, None) for name in expected
    }
    bridge_required = getattr(
        model_config,
        "_neural_forge_tokenizer_bridge_required",
        None,
    )
    if observed != expected or bridge_required is not True:
        raise Lfm2CpcfHoldoutCaptureError(
            "Frozen child configuration is not bound to the strict bridge."
        )
    return {
        "configuration": observed,
        "configuration_sha256": canonical_sha256(observed),
        "tokenizer_bridge_required": True,
        "matches_strict_bridge": True,
    }


def capture_lfm2_cpcf_holdout_candidate(
    config: CaptureConfig,
) -> dict[str, Any]:
    """Authenticate, generate once from the child, and publish sealed evidence."""

    if not isinstance(config, CaptureConfig):
        raise TypeError("config must be CaptureConfig.")
    if (
        config.candidate_output_path.exists()
        or config.capture_receipt_path.exists()
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "A capture output appeared after configuration; refusing overwrite."
        )
    cpcf = _authenticate_cpcf_receipt(
        checkpoint_directory=config.checkpoint_directory,
        external_receipt=config.cpcf_receipt,
        expected_external_file_sha256=(
            config.expected_cpcf_receipt_file_sha256
        ),
        expected_receipt_sha256=config.expected_cpcf_receipt_sha256,
        expected_bridge_artifact_sha256=(
            config.expected_bridge_artifact_sha256
        ),
        expected_bridge_profile_sha256=(
            config.expected_bridge_profile_sha256
        ),
    )
    manifest = load_capture_manifest(
        config.manifest_path,
        expected_manifest_sha256=config.expected_manifest_sha256,
    )
    bridge = load_strict_tokenizer_bridge_artifact(
        config.bridge_directory,
        expected_artifact_sha256=config.expected_bridge_artifact_sha256,
    )
    bridge.verify_files()
    if (
        bridge.artifact_sha256 != config.expected_bridge_artifact_sha256
        or bridge.profile_fingerprint
        != config.expected_bridge_profile_sha256
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Strict bridge artifact or profile differs from its expectation."
        )
    tokenizer, eos_token_ids, tokenizer_metadata = (
        _authenticate_tokenizer_package(config, bridge)
    )
    saved_config_audit = cpcf_runner._verify_saved_lfm2_ffn_config(
        config.checkpoint_directory,
        expected_width=cpcf.materialized_ffn_width,
    )
    model = fold_runtime.load_local_lfm2_model(
        config.checkpoint_directory,
        device=config.device,
        dtype=fold_runtime._torch_dtype(config.dtype),
    )
    if not isinstance(model, torch.nn.Module):
        raise Lfm2CpcfHoldoutCaptureError(
            "Local CPCF child loader returned no torch module."
        )
    model.eval()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise Lfm2CpcfHoldoutCaptureError(
            "Frozen child contains gradients before capture."
        )
    live_alias_audit = cpcf_runner._restore_lfm2_ffn_config_alias(
        model,
        "holdout_capture_fresh_child",
    )
    if (
        live_alias_audit.get("materialized_ffn_width")
        != cpcf.materialized_ffn_width
        or live_alias_audit.get("shape_stable") is not True
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Fresh child FFN shape differs from the CPCF receipt."
        )
    live_parameter_inventory = dict(independent_parameter_inventory(model))
    if canonical_json_bytes(live_parameter_inventory) != canonical_json_bytes(
        cpcf.expected_parameter_inventory
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Fresh child parameter inventory differs from the CPCF receipt."
        )
    model_bridge = _verify_model_bridge_configuration(model, bridge)
    state_before = fold_runtime.model_state_sha256(model)
    if (
        config.expected_candidate_state_fingerprint is not None
        and state_before != config.expected_candidate_state_fingerprint
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Frozen child state differs from its expected fingerprint."
        )
    rows = generate_candidate_rows_once(model, tokenizer, manifest)
    state_after = fold_runtime.model_state_sha256(model)
    if (
        state_after != state_before
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Child state or gradients changed during holdout capture."
        )
    live_parameter_inventory_after = dict(
        independent_parameter_inventory(model)
    )
    if canonical_json_bytes(
        live_parameter_inventory_after
    ) != canonical_json_bytes(live_parameter_inventory):
        raise Lfm2CpcfHoldoutCaptureError(
            "Child parameter topology changed during holdout capture."
        )
    inventory_after = _checkpoint_inventory(
        config.checkpoint_directory,
        ignored_relative_paths=frozenset({INTERNAL_CPCF_RECEIPT}),
    )
    if (
        canonical_json_bytes(inventory_after)
        != canonical_json_bytes(cpcf.checkpoint_inventory)
        or _file_sha256(
            config.checkpoint_directory / INTERNAL_CPCF_RECEIPT
        )
        != cpcf.internal_file_sha256
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Frozen checkpoint files changed during holdout capture."
        )
    bridge.verify_files()
    source_package_after, _ = posthoc.fingerprint_bound_local_tokenizer(
        config.target_tokenizer_directory
    )
    checkpoint_package_after, _ = (
        posthoc.fingerprint_bound_local_tokenizer(
            config.checkpoint_directory
        )
    )
    if (
        source_package_after
        != config.expected_target_tokenizer_fingerprint
        or checkpoint_package_after
        != config.checkpoint_tokenizer_fingerprint
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Target tokenizer package changed during holdout capture."
        )
    target_row_capacity = getattr(
        getattr(bridge, "transport_profile", None),
        "target_row_capacity",
        None,
    )
    if (
        isinstance(target_row_capacity, bool)
        or not isinstance(target_row_capacity, int)
        or target_row_capacity <= 0
        or any(
            token_id >= target_row_capacity
            for row in rows
            for token_id in row["output_token_ids"]
        )
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Child output IDs exceed the strict target tokenizer row domain."
        )
    candidate = seal_capture_candidate_outputs(
        manifest,
        rows,
        candidate_state_fingerprint=state_before,
        tokenizer_fingerprint=config.checkpoint_tokenizer_fingerprint,
        eos_token_ids=eos_token_ids,
    )
    loaded_candidate = load_sealed_capture_candidate_outputs(
        candidate,
        expected_candidate_outputs_sha256=candidate[
            "candidate_outputs_sha256"
        ],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_candidate_state_fingerprint=state_before,
        expected_tokenizer_fingerprint=(
            config.checkpoint_tokenizer_fingerprint
        ),
        eos_token_ids=eos_token_ids,
    )
    if (
        loaded_candidate.candidate_outputs_sha256
        != candidate["candidate_outputs_sha256"]
    ):
        raise Lfm2CpcfHoldoutCaptureError(
            "Sealed candidate failed its typed round-trip."
        )
    candidate_file_sha256 = _document_file_sha256(candidate)
    limits = [
        {
            "case_id": row["case_id"],
            "generation_limit": row["generation_limit"],
        }
        for row in rows
    ]
    input_counts = [
        {
            "case_id": row["case_id"],
            "input_token_count": row["input_token_count"],
        }
        for row in rows
    ]
    receipt_core = {
        "schema": CAPTURE_SCHEMA,
        "status": "captured_frozen_child_holdout",
        "manifest": {
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_document_sha256": manifest.document_sha256,
            "case_count": len(manifest.cases),
            "loader_mode": "strict_v1_via_version_dispatch_seam",
        },
        "cpcf_provenance": {
            "external_receipt_file_sha256": cpcf.external_file_sha256,
            "internal_receipt_file_sha256": cpcf.internal_file_sha256,
            "receipt_sha256": cpcf.receipt_sha256,
            "external_internal_receipts_identical": True,
            "saved_checkpoint_file_count": len(cpcf.checkpoint_inventory),
            "saved_checkpoint_files_sha256": (
                cpcf.checkpoint_inventory_sha256
            ),
            "saved_checkpoint_files_match_receipt": True,
            "reloaded_parameter_inventory": live_parameter_inventory,
            "reloaded_parameter_inventory_matches_receipt": True,
            "parameter_inventory_unchanged_after_generation": True,
            "saved_config_audit": dict(saved_config_audit),
            "fresh_reload_alias_audit": dict(live_alias_audit),
        },
        "strict_tokenizer_bridge": {
            "artifact_sha256": bridge.artifact_sha256,
            "profile_sha256": bridge.profile_fingerprint,
            "donor_tokenizer_semantics_sha256": (
                bridge.donor_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                bridge.target_tokenizer_semantics_sha256
            ),
            "target_row_capacity": target_row_capacity,
            "files_verified": True,
            "model_configuration": model_bridge,
        },
        "target_tokenizer": dict(tokenizer_metadata),
        "candidate": {
            "output_file_sha256": candidate_file_sha256,
            "candidate_outputs_sha256": candidate[
                "candidate_outputs_sha256"
            ],
            "candidate_state_fingerprint": state_before,
            "state_before_sha256": state_before,
            "state_after_sha256": state_after,
            "state_unchanged": True,
            "case_count": len(rows),
            "generation_limit_rows_sha256": canonical_sha256(limits),
            "input_token_count_rows_sha256": canonical_sha256(input_counts),
            "all_output_token_ids_are_child_domain_only": True,
            "decoded_text_sealed": True,
            "eos_identity_bound_by_target_package": True,
        },
        "generation_protocol": {
            "mode": "greedy_whole_response",
            "do_sample": False,
            "use_cache": True,
            "one_generate_invocation_per_case": True,
            "per_case_manifest_generation_limits_used": True,
            "global_64_token_cap_used": False,
            "eos_token_ids": list(eos_token_ids),
            "exact_parent_wording_used": False,
            "exact_parent_token_matching_used": False,
        },
        "operation_audit": {
            "child_model_loads": 1,
            "child_tokenizer_loads": tokenizer_metadata[
                "tokenizer_load_count"
            ],
            "candidate_generate_invocations": len(rows),
            "parent_model_loaded": False,
            "parent_forward_passes": 0,
            "parent_generate_invocations": 0,
            "trace_cache_opened": False,
            "training_used": False,
            "training_steps": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "quantization_used": False,
            "pruning_used": False,
            "network_access": False,
            "hub_loader_used": False,
        },
        "claims": {
            "fresh_local_frozen_child_loaded": True,
            "candidate_state_preserved": True,
            "folding_receipt_authenticated": True,
            "strict_bridge_authenticated": True,
            "strict_bridge_artifact_and_target_domain_authenticated": True,
            "live_donor_profile_reconstruction_claimed": False,
            "target_tokenizer_package_authenticated": True,
            "target_tokenizer_semantics_authenticated": True,
            "candidate_outputs_are_not_training": True,
            "candidate_outputs_are_not_exact_token_matching": True,
            "intelligence_retention_scored_here": False,
        },
    }
    capture_receipt = {
        **receipt_core,
        "capture_receipt_sha256": canonical_sha256(receipt_core),
    }
    capture_receipt_file_sha256 = _document_file_sha256(capture_receipt)
    _publish_new_documents(
        (
            (config.candidate_output_path, candidate),
            (config.capture_receipt_path, capture_receipt),
        )
    )
    published_hashes_match = (
        _file_sha256(config.candidate_output_path)
        == candidate_file_sha256
        and _file_sha256(config.capture_receipt_path)
        == capture_receipt_file_sha256
    )
    if not published_hashes_match:
        for path in (
            config.candidate_output_path,
            config.capture_receipt_path,
        ):
            if path.exists():
                path.unlink()
        raise Lfm2CpcfHoldoutCaptureError(
            "Published capture files differ from their staged hashes."
        )
    return capture_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-directory",
        dest="checkpoint_directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--cpcf-receipt", required=True, type=Path)
    parser.add_argument(
        "--expected-cpcf-receipt-file-sha256",
        required=True,
    )
    parser.add_argument("--expected-cpcf-receipt-sha256", required=True)
    parser.add_argument(
        "--manifest",
        "--manifest-path",
        dest="manifest_path",
        type=Path,
    )
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-target-tokenizer-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--expected-checkpoint-tokenizer-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--expected-target-tokenizer-semantics-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-bridge-artifact-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-bridge-profile-sha256",
        required=True,
    )
    parser.add_argument("--expected-candidate-state-fingerprint")
    parser.add_argument(
        "--output",
        "--output-path",
        dest="candidate_output_path",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--capture-receipt",
        dest="capture_receipt_path",
        required=True,
        type=Path,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    return parser


def config_from_args(args: argparse.Namespace) -> CaptureConfig:
    return CaptureConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    receipt = capture_lfm2_cpcf_holdout_candidate(config)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "manifest_sha256": receipt["manifest"]["manifest_sha256"],
                "case_count": receipt["manifest"]["case_count"],
                "candidate_outputs_sha256": receipt["candidate"][
                    "candidate_outputs_sha256"
                ],
                "candidate_state_fingerprint": receipt["candidate"][
                    "candidate_state_fingerprint"
                ],
                "capture_receipt_sha256": receipt[
                    "capture_receipt_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
