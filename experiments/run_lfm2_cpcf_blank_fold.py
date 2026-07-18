#!/usr/bin/env python3
"""Compile a sealed LFM2 parent cache into a severe dense blank child.

This is the production CPCF orchestration path.  It accepts only explicit
local model, tokenizer, bridge, corpus, and sealed-cache paths.  The parent is
loaded to expose its immutable parameters, but it is never executed or asked
to generate.  Parent causal evidence is streamed from the already completed
cache exactly once into one resident phase collection, then reused for:

* a target-native tied vocabulary interface;
* independent predictive residual coordinates at every macrocell boundary;
* sequential operator and MLP compilation on actual folded-upstream states;
* a target-coordinate final RMSNorm; and
* standalone whole-response generation from a freshly reloaded child.

The runner has no optimizer, backward pass, training loop, quantization,
pruning, parent-backed fallback, hub loader, or cache-generation path.
Structural dry-plan mode is deliberately non-behavioral: it may prove a fresh
dense blank's exact parameter count, but cannot claim that a child was folded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any


# Direct ``python experiments/<script>.py`` execution puts only the
# experiments directory on sys.path.  Add the canonical repository root so
# sibling experiment helpers remain importable without requiring PYTHONPATH
# or a module-style invocation.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


# Set the offline contract before importing any Transformers-backed module.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


RUN_SCHEMA = "neural-forge-lfm2-cpcf-blank-fold-run-v1"
DRY_PLAN_SCHEMA = "neural-forge-lfm2-cpcf-blank-dry-plan-v1"
TRAJECTORY_SCHEMA = "neural-forge-lfm2-bridge-trajectory-text-v1"
RESUME_INVOCATION_SCHEMA = "neural-forge-lfm2-cpcf-runner-resume-invocation-v1"
RESUME_COMPILER_FINGERPRINT_SCHEMA = (
    "neural-forge-lfm2-cpcf-compiler-implementation-fingerprint-v1"
)
RESUME_BOUNDARY_POLICY_SCHEMA = "neural-forge-lfm2-cpcf-boundary-policy-binding-v1"
RESUME_LATEST_POINTER_SCHEMA = (
    "neural-forge-lfm2-cpcf-latest-resumable-checkpoint-pointer-v1"
)
STANDALONE_EVALUATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-standalone-whole-response-evaluation-v1"
)
MINIMUM_CLAIMABLE_COMPRESSION_RATIO = Decimal("100")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LAYER_PARAMETER_PATH = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_DTYPES = ("bfloat16", "float16", "float32")
_UNSET = object()


class Lfm2CpcfBlankFoldRunnerError(RuntimeError):
    """The strict local CPCF run cannot be completed honestly."""


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
        raise Lfm2CpcfBlankFoldRunnerError(
            "Runner evidence must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_directory(value: Path, name: str) -> Path:
    unresolved = value.expanduser().absolute()
    try:
        unresolved_stat = unresolved.lstat()
    except OSError as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be an explicit nonsymlink local directory: {unresolved}."
        ) from exc
    if _is_link_or_junction(unresolved) or not stat.S_ISDIR(unresolved_stat.st_mode):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be an explicit nonsymlink local directory: {unresolved}."
        )
    path = unresolved.resolve(strict=True)
    if not path.is_dir() or _is_link_or_junction(path):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must resolve to a nonsymlink local directory: {path}."
        )
    return path


def _local_file(value: Path, name: str) -> Path:
    unresolved = value.expanduser().absolute()
    try:
        unresolved_stat = unresolved.lstat()
    except OSError as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be an explicit nonsymlink local file: {unresolved}."
        ) from exc
    if _is_link_or_junction(unresolved) or not stat.S_ISREG(unresolved_stat.st_mode):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be an explicit nonsymlink local file: {unresolved}."
        )
    path = unresolved.resolve(strict=True)
    if not path.is_file() or _is_link_or_junction(path):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must resolve to a nonsymlink local file: {path}."
        )
    return path


def _new_output_directory(value: Path) -> Path:
    path = value.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    result = resolved_parent / path.name
    if result.exists() or result.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(
            f"Output directory already exists; refusing to overwrite it: {result}."
        )
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(
            f"Receipt already exists; refusing to overwrite it: {destination}."
        )
    temporary = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    payload = _canonical_json(dict(value)) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfBlankFoldRunnerError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfBlankFoldRunnerError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfBlankFoldRunnerError(f"{name} must be finite and nonnegative.")
    return result


def _tensor_payload_sha256(value: Any, name: str) -> str:
    """Bind one materialized tensor without changing or retaining it."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency
        raise Lfm2CpcfBlankFoldRunnerError(
            "PyTorch is required to fingerprint CPCF tensor inputs."
        ) from exc
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must be a materialized dense unquantized tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _compiler_implementation_fingerprint(
    modules: Sequence[Any],
) -> str:
    """Seal the exact local Python sources that implement this fold."""

    records = []
    seen_names = set()
    for module in modules:
        name = getattr(module, "__name__", None)
        raw_path = getattr(module, "__file__", None)
        if (
            not isinstance(name, str)
            or not name
            or name in seen_names
            or not isinstance(raw_path, str)
            or not raw_path
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                "Compiler modules must have unique names and local source files."
            )
        path = Path(raw_path).expanduser().resolve(strict=True)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.suffix not in {".py", ".pyw"}
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Compiler module {name!r} is not backed by a local Python source."
            )
        seen_names.add(name)
        records.append(
            {
                "module": name,
                "source_file": path.name,
                "source_sha256": _file_sha256(path),
            }
        )
    if not records:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Compiler implementation fingerprint cannot be empty."
        )
    return _sha(
        {
            "schema": RESUME_COMPILER_FINGERPRINT_SCHEMA,
            "modules": sorted(records, key=lambda item: item["module"]),
        }
    )


def _resume_compiler_fingerprint() -> str:
    """Fingerprint the complete local CPCF numerical implementation."""

    import neural_forge.dynamic_fold_executor as dynamic_fold_executor
    import neural_forge.hf_closed_form_causal_circuit_fold as closed_form
    import neural_forge.hf_lfm2_cpcf_boundary_bundle as boundary_bundle
    import neural_forge.hf_lfm2_cpcf_layer_compiler as layer_compiler
    import neural_forge.hf_lfm2_cpcf_macrocell_replay as macrocell_replay
    import neural_forge.hf_lfm2_cpcf_model_compiler as model_compiler
    import neural_forge.hf_lfm2_cpcf_model_fold as model_fold
    import neural_forge.hf_lfm2_cpcf_resume_artifact as resume_artifact
    import neural_forge.hf_lfm2_cpcf_resume_bridge as resume_bridge
    import neural_forge.hf_lfm2_cpcf_short_conv_fold as short_conv_v3
    import neural_forge.hf_lfm2_cpcf_short_conv_v4 as short_conv_v4
    import neural_forge.hf_lfm2_cpcf_short_conv_v4_fold as short_conv_v4_fold
    import neural_forge.hf_lfm2_cpcf_short_conv_v4_policy as short_conv_v4_policy
    import neural_forge.hf_lfm2_structural_residual_attention_fold as residual_fold
    import neural_forge.hf_tied_causal_interface_fold as tied_interface

    return _compiler_implementation_fingerprint(
        (
            SimpleNamespace(
                __name__="experiments.run_lfm2_cpcf_blank_fold",
                __file__=__file__,
            ),
            dynamic_fold_executor,
            closed_form,
            boundary_bundle,
            layer_compiler,
            macrocell_replay,
            model_compiler,
            model_fold,
            resume_artifact,
            resume_bridge,
            short_conv_v3,
            short_conv_v4,
            short_conv_v4_fold,
            short_conv_v4_policy,
            residual_fold,
            tied_interface,
        )
    )


@dataclass(frozen=True, slots=True)
class _PinnedResumeCheckpointSelection:
    reference: Any
    source_path: Path
    latest_pointer: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, str]:
        return self.reference.as_dict()


def _artifact_reference_from_mapping(
    value: Any,
    *,
    artifact_kind: str,
    name: str,
) -> Any:
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfArtifactReference,
        Lfm2CpcfResumeArtifactError,
    )

    if not isinstance(value, Mapping) or set(value) != {
        "artifact_kind",
        "artifact_key",
        "manifest_sha256",
    }:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must have the exact pinned reference fields."
        )
    try:
        reference = Lfm2CpcfArtifactReference(**dict(value))
    except (
        TypeError,
        ValueError,
        Lfm2CpcfResumeArtifactError,
    ) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(f"{name} is malformed.") from exc
    if reference.artifact_kind != artifact_kind:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} must reference a {artifact_kind} artifact."
        )
    return reference


def _validate_latest_resume_pointer(
    value: Any,
    *,
    expected_invocation_sha256: Any = _UNSET,
    expected_boundary_binding_key: Any = _UNSET,
    expected_boundary_reference: Any = _UNSET,
    expected_full_policy_sha256: Any = _UNSET,
    expected_compiler_fingerprint_sha256: Any = _UNSET,
    expected_base_checkpoint_reference: Any = _UNSET,
    expected_previous_checkpoint_reference: Any = _UNSET,
    expected_checkpoint_reference: Any = _UNSET,
    expected_next_layer_index: Any = _UNSET,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Latest resume pointer must be a JSON object."
        )
    try:
        pointer = json.loads(_canonical_json(dict(value)).decode("ascii"))
    except Lfm2CpcfBlankFoldRunnerError as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Latest resume pointer is not finite canonical JSON."
        ) from exc
    expected_fields = {
        "schema",
        "invocation_sha256",
        "boundary_binding_key",
        "boundary_reference",
        "full_policy_sha256",
        "compiler_fingerprint_sha256",
        "base_checkpoint_reference",
        "previous_checkpoint_reference",
        "checkpoint_reference",
        "next_layer_index",
        "event_report_sha256",
        "resumable_prefix",
        "promotion_authorized",
        "pointer_sha256",
    }
    if set(pointer) != expected_fields:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Latest resume pointer has an unexpected field set."
        )
    body = dict(pointer)
    claimed = body.pop("pointer_sha256", None)
    next_layer_index = pointer.get("next_layer_index")
    if (
        pointer.get("schema") != RESUME_LATEST_POINTER_SCHEMA
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or _sha(body) != claimed
        or isinstance(next_layer_index, bool)
        or not isinstance(next_layer_index, int)
        or next_layer_index < 0
        or pointer.get("resumable_prefix") is not True
        or pointer.get("promotion_authorized") is not False
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Latest resume pointer self-seal or fixed safety flags are invalid."
        )
    for name in (
        "invocation_sha256",
        "boundary_binding_key",
        "full_policy_sha256",
        "compiler_fingerprint_sha256",
        "event_report_sha256",
    ):
        if (
            not isinstance(pointer.get(name), str)
            or _SHA256.fullmatch(pointer[name]) is None
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Latest resume pointer {name} is invalid."
            )
    boundary_reference = _artifact_reference_from_mapping(
        pointer["boundary_reference"],
        artifact_kind="boundary",
        name="latest pointer boundary_reference",
    )
    checkpoint_reference = _artifact_reference_from_mapping(
        pointer["checkpoint_reference"],
        artifact_kind="checkpoint",
        name="latest pointer checkpoint_reference",
    )
    optional_references = {}
    for name in (
        "base_checkpoint_reference",
        "previous_checkpoint_reference",
    ):
        raw = pointer[name]
        optional_references[name] = (
            None
            if raw is None
            else _artifact_reference_from_mapping(
                raw,
                artifact_kind="checkpoint",
                name=f"latest pointer {name}",
            )
        )
    expectations = (
        (
            "invocation_sha256",
            expected_invocation_sha256,
            pointer["invocation_sha256"],
        ),
        (
            "boundary_binding_key",
            expected_boundary_binding_key,
            pointer["boundary_binding_key"],
        ),
        (
            "boundary_reference",
            expected_boundary_reference,
            boundary_reference,
        ),
        (
            "full_policy_sha256",
            expected_full_policy_sha256,
            pointer["full_policy_sha256"],
        ),
        (
            "compiler_fingerprint_sha256",
            expected_compiler_fingerprint_sha256,
            pointer["compiler_fingerprint_sha256"],
        ),
        (
            "base_checkpoint_reference",
            expected_base_checkpoint_reference,
            optional_references["base_checkpoint_reference"],
        ),
        (
            "previous_checkpoint_reference",
            expected_previous_checkpoint_reference,
            optional_references["previous_checkpoint_reference"],
        ),
        (
            "checkpoint_reference",
            expected_checkpoint_reference,
            checkpoint_reference,
        ),
        (
            "next_layer_index",
            expected_next_layer_index,
            next_layer_index,
        ),
    )
    for name, expected, observed in expectations:
        if expected is not _UNSET and observed != expected:
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Latest resume pointer {name} differs from this invocation."
            )
    return pointer


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(callable(is_junction) and is_junction(path))


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _latest_pointer_directory(store: Any) -> Path:
    root = Path(store.root).resolve(strict=True)
    if _is_link_or_junction(root) or not root.is_dir():
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume store root changed or became unsafe."
        )
    candidate = root / "latest"
    candidate.mkdir(exist_ok=True)
    if (
        _is_link_or_junction(candidate)
        or not candidate.is_dir()
        or candidate.resolve(strict=True).parent != root
    ):
        raise Lfm2CpcfBlankFoldRunnerError("Resume latest-pointer directory is unsafe.")
    result = candidate.resolve(strict=True)
    _fsync_directory(root)
    return result


@contextmanager
def _latest_pointer_lock(directory: Path, invocation_sha256: str):
    lock_path = directory / f".{invocation_sha256}.lock"
    if lock_path.exists() and _is_link_or_junction(lock_path):
        raise Lfm2CpcfBlankFoldRunnerError("Resume latest-pointer lock path is unsafe.")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Could not open the resume latest-pointer lock."
        ) from exc
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    locked = False
    try:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            raise Lfm2CpcfBlankFoldRunnerError(
                "Another process owns this invocation latest-pointer lock."
            ) from exc
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _read_pointer_file(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or _is_link_or_junction(path)
        or path.resolve(strict=True).parent != path.parent.resolve(strict=True)
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume latest pointer path is not a safe regular file."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume latest pointer is unreadable or invalid JSON."
        ) from exc
    return _validate_latest_resume_pointer(value)


def _publish_latest_resume_pointer(
    *,
    store: Any,
    boundary_bindings: Any,
    boundary: Any,
    checkpoint: Any,
    full_policy_sha256: str,
    compiler_fingerprint_sha256: str,
    bridge_artifact_sha256: str,
    base_checkpoint_reference: Any | None,
    previous_checkpoint: Any | None,
    event_report_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    verified_boundary = store.final_reauthenticate_boundary(boundary)
    verified_checkpoint = store.final_reauthenticate_checkpoint(checkpoint)
    boundary_reference = verified_boundary.reference
    checkpoint_reference = verified_checkpoint.reference
    previous_reference = (
        None if previous_checkpoint is None else previous_checkpoint.reference
    )
    next_layer_index = verified_checkpoint.completed_layers["next_layer_index"]
    if (
        isinstance(next_layer_index, bool)
        or not isinstance(next_layer_index, int)
        or next_layer_index < 0
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Verified checkpoint has an invalid next layer."
        )
    invocation_sha256 = _resume_invocation_sha256(
        boundary_bindings=boundary_bindings,
        bridge_artifact_sha256=bridge_artifact_sha256,
        full_policy_sha256=full_policy_sha256,
        compiler_fingerprint_sha256=compiler_fingerprint_sha256,
        checkpoint_reference=base_checkpoint_reference,
    )
    directory = _latest_pointer_directory(store)
    pointer_path = directory / f"{invocation_sha256}.json"
    with _latest_pointer_lock(directory, invocation_sha256):
        existing = (
            _read_pointer_file(pointer_path) if os.path.lexists(pointer_path) else None
        )
        if existing is None:
            previous_next_layer = (
                None
                if previous_checkpoint is None
                else previous_checkpoint.completed_layers["next_layer_index"]
            )
            expected_next = (
                0 if previous_next_layer is None else int(previous_next_layer) + 1
            )
            if next_layer_index != expected_next:
                raise Lfm2CpcfBlankFoldRunnerError(
                    "First latest-pointer publication is not the seed or "
                    "the immediate successor of its base checkpoint."
                )
        else:
            existing = _validate_latest_resume_pointer(
                existing,
                expected_invocation_sha256=invocation_sha256,
                expected_boundary_binding_key=(boundary_bindings.binding_key),
                expected_boundary_reference=boundary_reference,
                expected_full_policy_sha256=full_policy_sha256,
                expected_compiler_fingerprint_sha256=(compiler_fingerprint_sha256),
                expected_base_checkpoint_reference=(base_checkpoint_reference),
            )
            existing_reference = _artifact_reference_from_mapping(
                existing["checkpoint_reference"],
                artifact_kind="checkpoint",
                name="existing latest checkpoint",
            )
            existing_next = existing["next_layer_index"]
            if next_layer_index == existing_next:
                if checkpoint_reference != existing_reference:
                    raise Lfm2CpcfBlankFoldRunnerError(
                        "A different checkpoint occupies the same latest pointer layer."
                    )
                # Progress-report hashes may include process-local object
                # inventory.  The immutable, final-reauthenticated checkpoint
                # reference is the idempotence identity; retain the first
                # self-sealed pointer and its original event evidence.
                return pointer_path, existing
            if (
                next_layer_index != existing_next + 1
                or previous_reference != existing_reference
            ):
                raise Lfm2CpcfBlankFoldRunnerError(
                    "Latest resume pointer may advance only one authenticated "
                    "layer along its checkpoint lineage."
                )
        body = {
            "schema": RESUME_LATEST_POINTER_SCHEMA,
            "invocation_sha256": invocation_sha256,
            "boundary_binding_key": boundary_bindings.binding_key,
            "boundary_reference": boundary_reference.as_dict(),
            "full_policy_sha256": full_policy_sha256,
            "compiler_fingerprint_sha256": (compiler_fingerprint_sha256),
            "base_checkpoint_reference": (
                None
                if base_checkpoint_reference is None
                else base_checkpoint_reference.as_dict()
            ),
            "previous_checkpoint_reference": (
                None if previous_reference is None else previous_reference.as_dict()
            ),
            "checkpoint_reference": checkpoint_reference.as_dict(),
            "next_layer_index": next_layer_index,
            "event_report_sha256": event_report_sha256,
            "resumable_prefix": True,
            "promotion_authorized": False,
        }
        document = {**body, "pointer_sha256": _sha(body)}
        _validate_latest_resume_pointer(
            document,
            expected_invocation_sha256=invocation_sha256,
            expected_boundary_binding_key=boundary_bindings.binding_key,
            expected_boundary_reference=boundary_reference,
            expected_full_policy_sha256=full_policy_sha256,
            expected_compiler_fingerprint_sha256=(compiler_fingerprint_sha256),
            expected_base_checkpoint_reference=(base_checkpoint_reference),
            expected_previous_checkpoint_reference=previous_reference,
            expected_checkpoint_reference=checkpoint_reference,
            expected_next_layer_index=next_layer_index,
        )
        temporary = directory / (f".{invocation_sha256}.stage-{uuid.uuid4().hex}")
        payload = _canonical_json(document) + b"\n"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if (
                store.final_reauthenticate_boundary(verified_boundary).reference
                != boundary_reference
                or store.final_reauthenticate_checkpoint(verified_checkpoint).reference
                != checkpoint_reference
            ):
                raise Lfm2CpcfBlankFoldRunnerError(
                    "Checkpoint changed before latest-pointer replacement."
                )
            os.replace(temporary, pointer_path)
            _fsync_directory(directory)
        finally:
            if os.path.lexists(temporary):
                temporary.unlink()
        observed = _read_pointer_file(pointer_path)
        if observed != document:
            raise Lfm2CpcfBlankFoldRunnerError(
                "Latest resume pointer differs after atomic publication."
            )
        return pointer_path, observed


def _read_resume_checkpoint_reference(
    path: Path,
) -> _PinnedResumeCheckpointSelection:
    """Read a pinned checkpoint reference or its authenticated manifest."""

    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        RESUME_ARTIFACT_SCHEMA,
    )

    source = _local_file(path, "resume_checkpoint_reference")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume checkpoint reference must be valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume checkpoint reference must be a JSON object."
        )
    reference_fields = {
        "artifact_kind",
        "artifact_key",
        "manifest_sha256",
    }
    latest_pointer = None
    if value.get("schema") == RESUME_LATEST_POINTER_SCHEMA:
        latest_pointer = _validate_latest_resume_pointer(value)
        reference = latest_pointer["checkpoint_reference"]
    elif set(value) == reference_fields:
        reference = dict(value)
    elif (
        value.get("schema") == RESUME_ARTIFACT_SCHEMA
        and value.get("artifact_kind") == "checkpoint"
        and reference_fields <= set(value)
    ):
        reference = {field: value[field] for field in sorted(reference_fields)}
    else:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume checkpoint reference must be either the exact pinned "
            "three-field reference, an authenticated checkpoint manifest, "
            "or a self-sealed latest pointer."
        )
    reference = _artifact_reference_from_mapping(
        reference,
        artifact_kind="checkpoint",
        name="resume checkpoint reference",
    )
    return _PinnedResumeCheckpointSelection(
        reference=reference,
        source_path=source,
        latest_pointer=latest_pointer,
    )


@dataclass(frozen=True, slots=True)
class CompressionRequest:
    """One unambiguous parameter-count target.

    Generic diagnostic controls may request any real compaction.  The runner's
    separate promotion gate decides whether a result may represent the current
    100x project objective.
    """

    compression_ratio: str | None
    target_unique_parameters: int | None

    def __post_init__(self) -> None:
        supplied = (
            self.compression_ratio is not None,
            self.target_unique_parameters is not None,
        )
        if sum(supplied) != 1:
            raise Lfm2CpcfBlankFoldRunnerError(
                "Supply exactly one compression ratio or target parameter count."
            )
        if self.compression_ratio is not None:
            try:
                ratio = Decimal(self.compression_ratio)
            except (InvalidOperation, ValueError) as exc:
                raise Lfm2CpcfBlankFoldRunnerError(
                    "compression_ratio must be a finite decimal."
                ) from exc
            if not ratio.is_finite() or ratio <= Decimal("1"):
                raise Lfm2CpcfBlankFoldRunnerError(
                    "compression_ratio must be finite and greater than one."
                )
            object.__setattr__(self, "compression_ratio", str(ratio))
        if self.target_unique_parameters is not None:
            _positive_int(
                self.target_unique_parameters,
                "target_unique_parameters",
            )

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "compression_ratio": self.compression_ratio,
            "target_unique_parameters": self.target_unique_parameters,
        }


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    parent_model_directory: Path
    donor_tokenizer_directory: Path
    target_tokenizer_directory: Path
    bridge_directory: Path
    trace_cache_directory: Path
    trajectory_corpus: Path
    output_directory: Path
    receipt: Path | None
    compression: CompressionRequest
    device: str
    dtype: str
    parent_name: str
    child_name: str
    blank_seed: int
    minimum_hidden_size: int
    maximum_candidates: int
    selected_candidate_rank: int | None
    ridge: float
    boundary_state_reconstruction_weight: float
    boundary_predictive_energy_floor: float
    boundary_validation_retention_floor: float
    evaluation_limit: int
    evaluation_max_new_tokens: int
    structural_dry_plan: bool
    diagnostic_control: bool
    resume_store_directory: Path | None = None
    resume_checkpoint_reference: Path | None = None
    short_conv_fold_version: str = "v3"
    short_conv_v4_ridge_alphas: tuple[float, ...] = (1e-8, 1e-4)
    short_conv_v4_variable_projection_passes: int = 2
    short_conv_v4_writer_trust_radius: float = 4.0
    short_conv_v4_fir_trust_radius: float = 1.0
    short_conv_v4_bcx_trust_radius: float = 0.25
    short_conv_v4_correction_rms_floor: float = 1e-6
    short_conv_v4_context_objective_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise Lfm2CpcfBlankFoldRunnerError(f"dtype must be one of {_DTYPES!r}.")
        if not self.device.strip():
            raise Lfm2CpcfBlankFoldRunnerError("device must be nonempty.")
        if not self.parent_name.strip() or not self.child_name.strip():
            raise Lfm2CpcfBlankFoldRunnerError(
                "parent_name and child_name must be nonempty."
            )
        if isinstance(self.blank_seed, bool) or not isinstance(self.blank_seed, int):
            raise Lfm2CpcfBlankFoldRunnerError("blank_seed must be an integer.")
        for value, name in (
            (self.minimum_hidden_size, "minimum_hidden_size"),
            (self.maximum_candidates, "maximum_candidates"),
            (self.evaluation_limit, "evaluation_limit"),
            (
                self.evaluation_max_new_tokens,
                "evaluation_max_new_tokens",
            ),
        ):
            _positive_int(value, name)
        if self.selected_candidate_rank is not None and (
            isinstance(self.selected_candidate_rank, bool)
            or not isinstance(self.selected_candidate_rank, int)
            or self.selected_candidate_rank < 0
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                "selected_candidate_rank must be nonnegative."
            )
        _nonnegative_float(self.ridge, "ridge")
        _nonnegative_float(
            self.boundary_state_reconstruction_weight,
            "boundary_state_reconstruction_weight",
        )
        for value, name in (
            (
                self.boundary_predictive_energy_floor,
                "boundary_predictive_energy_floor",
            ),
            (
                self.boundary_validation_retention_floor,
                "boundary_validation_retention_floor",
            ),
        ):
            normalized = _nonnegative_float(value, name)
            if normalized > 1.0:
                raise Lfm2CpcfBlankFoldRunnerError(f"{name} must be within [0, 1].")
        if not isinstance(self.structural_dry_plan, bool):
            raise Lfm2CpcfBlankFoldRunnerError("structural_dry_plan must be boolean.")
        if not isinstance(self.diagnostic_control, bool):
            raise Lfm2CpcfBlankFoldRunnerError("diagnostic_control must be boolean.")
        if self.resume_checkpoint_reference is not None and (
            self.resume_store_directory is None
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                "resume_checkpoint_reference requires resume_store_directory."
            )
        if self.structural_dry_plan and (
            self.resume_store_directory is not None
            or self.resume_checkpoint_reference is not None
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                "Resume artifacts are only valid for a real fold, not a "
                "structural dry plan."
            )
        if self.short_conv_fold_version not in {"v3", "v4"}:
            raise Lfm2CpcfBlankFoldRunnerError(
                "short_conv_fold_version must be v3 or v4."
            )
        alphas = tuple(
            _nonnegative_float(value, "short_conv_v4_ridge_alphas")
            for value in self.short_conv_v4_ridge_alphas
        )
        if not alphas or tuple(sorted(set(alphas))) != alphas:
            raise Lfm2CpcfBlankFoldRunnerError(
                "short_conv_v4_ridge_alphas must be unique and increasing."
            )
        object.__setattr__(self, "short_conv_v4_ridge_alphas", alphas)
        if self.short_conv_v4_variable_projection_passes not in {2, 3}:
            raise Lfm2CpcfBlankFoldRunnerError(
                "short_conv_v4_variable_projection_passes must be 2 or 3."
            )
        for value, name in (
            (
                self.short_conv_v4_writer_trust_radius,
                "short_conv_v4_writer_trust_radius",
            ),
            (
                self.short_conv_v4_fir_trust_radius,
                "short_conv_v4_fir_trust_radius",
            ),
            (
                self.short_conv_v4_bcx_trust_radius,
                "short_conv_v4_bcx_trust_radius",
            ),
            (
                self.short_conv_v4_correction_rms_floor,
                "short_conv_v4_correction_rms_floor",
            ),
        ):
            if _nonnegative_float(value, name) <= 0.0:
                raise Lfm2CpcfBlankFoldRunnerError(f"{name} must be positive.")
        context_weight = _nonnegative_float(
            self.short_conv_v4_context_objective_weight,
            "short_conv_v4_context_objective_weight",
        )
        if context_weight > 0.1:
            raise Lfm2CpcfBlankFoldRunnerError(
                "short_conv_v4_context_objective_weight must be within [0, 0.1]."
            )


def _torch_dtype(torch: Any, name: str) -> Any:
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise Lfm2CpcfBlankFoldRunnerError(f"Unsupported dtype {name!r}.") from exc


def _read_trajectory_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Trajectory corpus must be valid UTF-8 JSON."
        ) from exc
    if not isinstance(value, list) or not value:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Trajectory corpus must be a nonempty JSON list."
        )
    result = []
    identifiers = set()
    splits = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Trajectory row {index} is not an object."
            )
        record_id = row.get("id")
        split = row.get("split")
        input_text = row.get("input_text")
        forced_text = row.get("forced_decode_text")
        if (
            row.get("schema") != TRAJECTORY_SCHEMA
            or not isinstance(record_id, str)
            or not record_id
            or record_id in identifiers
            or split not in {"selection", "fit", "validation"}
            or not isinstance(input_text, str)
            or not input_text
            or not isinstance(forced_text, str)
            or not forced_text
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Trajectory row {index} is malformed or duplicated."
            )
        identifiers.add(record_id)
        splits.add(str(split))
        result.append(
            {
                "id": record_id,
                "split": str(split),
                "input_text": input_text,
                "forced_decode_text": forced_text,
            }
        )
    if splits != {"selection", "fit", "validation"}:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Trajectory corpus must cover selection, fit, and validation."
        )
    return tuple(result)


def _token_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
    )
    if not isinstance(encoded, Mapping):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Local tokenizer returned an invalid encoding."
        )
    values = encoded.get("input_ids")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        )
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Local tokenizer returned invalid token IDs."
        )
    return tuple(values)


def _trace_adapter(
    *,
    rows: Sequence[Mapping[str, str]],
    target_tokenizer: Any,
    bridge: Any,
    cache_manifest: Mapping[str, Any],
) -> Any:
    import torch

    from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
        Lfm2TargetTokenizedRecord,
        build_lfm2_strict_tokenizer_bridge_trace_adapter,
    )

    capture = cache_manifest.get("capture")
    corpus = cache_manifest.get("corpus")
    if not isinstance(capture, Mapping) or not isinstance(corpus, Mapping):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Sealed cache lacks capture or corpus metadata."
        )
    max_decode_steps = _positive_int(
        capture.get("max_decode_steps"),
        "cache max_decode_steps",
    )
    records = []
    for row in rows:
        input_ids = _token_ids(
            target_tokenizer,
            row["input_text"],
            add_special_tokens=True,
        )
        forced_ids = _token_ids(
            target_tokenizer,
            row["forced_decode_text"],
            add_special_tokens=False,
        )
        if len(forced_ids) > max_decode_steps:
            raise Lfm2CpcfBlankFoldRunnerError(
                f"Trajectory {row['id']!r} exceeds the sealed decode count."
            )
        records.append(
            Lfm2TargetTokenizedRecord(
                record_id=row["id"],
                split=row["split"],
                input_ids=torch.tensor([input_ids], dtype=torch.long),
                attention_mask=torch.ones(
                    (1, len(input_ids)),
                    dtype=torch.long,
                ),
                forced_decode_ids=torch.tensor(
                    [forced_ids],
                    dtype=torch.long,
                ),
            )
        )
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=bridge.artifact_sha256,
    )
    ordered = corpus.get("ordered_batch_input_sha256")
    observed = tuple(record.donor_cache_input_sha256 for record in adapter.records)
    if (
        not isinstance(ordered, Sequence)
        or isinstance(ordered, (str, bytes))
        or tuple(ordered) != observed
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Canonical target tokenization does not reproduce the sealed "
            "mapped-donor cache corpus."
        )
    return adapter


def _mapped_source_tied_table(
    parent_model: Any,
    bridge: Any,
) -> Any:
    import torch

    input_weight = getattr(parent_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(parent_model.get_output_embeddings(), "weight", None)
    if (
        input_weight is None
        or output_weight is not input_weight
        or input_weight.ndim != 2
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "The current CPCF interface requires one native tied parent table."
        )
    mapping = bridge.transport_profile.target_to_donor_row_indices
    index = torch.tensor(
        mapping,
        device=input_weight.device,
        dtype=torch.long,
    )
    if int(index.max().item()) >= int(input_weight.shape[0]):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Strict bridge addresses a row outside the physical parent table."
        )
    return input_weight.detach().index_select(0, index).to(device="cpu")


def _checkpoint_inventory(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _materialized_lfm2_ffn_width(model: Any, name: str) -> int:
    """Return the exact FFN width jointly proven by config and live weights."""

    config = getattr(model, "config", None)
    core = getattr(model, "model", None)
    layers = tuple(getattr(core, "layers", ()))
    if config is None or getattr(config, "model_type", None) != "lfm2" or not layers:
        raise Lfm2CpcfBlankFoldRunnerError(f"{name} is not a materialized LFM2 model.")
    widths = set()
    for index, layer in enumerate(layers):
        feed_forward = getattr(layer, "feed_forward", None)
        w1 = getattr(getattr(feed_forward, "w1", None), "weight", None)
        w2 = getattr(getattr(feed_forward, "w2", None), "weight", None)
        w3 = getattr(getattr(feed_forward, "w3", None), "weight", None)
        if (
            w1 is None
            or w2 is None
            or w3 is None
            or w1.ndim != 2
            or w2.ndim != 2
            or w3.ndim != 2
            or int(w1.shape[0]) != int(w3.shape[0])
            or int(w1.shape[0]) != int(w2.shape[1])
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                f"{name} layer {index} has inconsistent materialized FFN weights."
            )
        widths.add(int(w1.shape[0]))
    if len(widths) != 1:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} layers do not share one materialized FFN width."
        )
    width = next(iter(widths))
    configured = getattr(config, "intermediate_size", None)
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured != width
        or getattr(config, "block_auto_adjust_ff_dim", None) is not False
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} runtime FFN width is not sealed by intermediate_size with "
            "block_auto_adjust_ff_dim=False."
        )
    return width


def _restore_lfm2_ffn_config_alias(model: Any, name: str) -> dict[str, Any]:
    """Restore the legacy alias consumed by Lfm2Config during local reload."""

    width = _materialized_lfm2_ffn_width(model, name)
    config = model.config
    previous = getattr(config, "block_ff_dim", None)
    if previous is not None and previous != width:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} block_ff_dim conflicts with its materialized FFN width."
        )
    config.block_ff_dim = width
    payload = config.to_dict()
    expected = {
        "intermediate_size": width,
        "block_ff_dim": width,
        "block_auto_adjust_ff_dim": False,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"{name} cannot serialize shape-stable LFM2 FFN aliases: "
            f"{observed} != {expected}."
        )
    body = {
        "scope": name,
        "materialized_ffn_width": width,
        "previous_block_ff_dim": previous,
        "serialized_aliases": observed,
        "shape_stable": True,
    }
    return {**body, "report_sha256": _sha(body)}


def _verify_saved_lfm2_ffn_config(
    directory: Path,
    *,
    expected_width: int,
) -> dict[str, Any]:
    """Fail closed if checkpoint JSON cannot rebuild the exact FFN shape."""

    config_path = directory / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            f"Could not authenticate saved LFM2 config: {config_path}."
        ) from exc
    if not isinstance(payload, Mapping):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Saved LFM2 config must contain a JSON object."
        )
    expected = {
        "intermediate_size": expected_width,
        "block_ff_dim": expected_width,
        "block_auto_adjust_ff_dim": False,
    }
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Saved config.json cannot reconstruct the materialized LFM2 FFN "
            f"shape: {observed} != {expected}."
        )
    body = {
        "scope": "saved_config_json",
        "config_json_sha256": _file_sha256(config_path),
        "serialized_aliases": observed,
        "shape_stable": True,
    }
    return {**body, "report_sha256": _sha(body)}


def _repetition_proxy(token_ids: Sequence[int]) -> dict[str, Any]:
    if not token_ids:
        return {
            "token_count": 0,
            "unique_token_fraction": 0.0,
            "maximum_identical_run": 0,
            "passed": False,
        }
    maximum_run = 1
    run = 1
    for previous, current in zip(token_ids, token_ids[1:]):
        if current == previous:
            run += 1
            maximum_run = max(maximum_run, run)
        else:
            run = 1
    unique_fraction = len(set(token_ids)) / len(token_ids)
    passed = (
        maximum_run <= max(4, int(math.ceil(len(token_ids) * 0.5)))
        and unique_fraction >= 0.10
    )
    return {
        "token_count": len(token_ids),
        "unique_token_fraction": unique_fraction,
        "maximum_identical_run": maximum_run,
        "passed": passed,
    }


def _standalone_whole_response_evaluation(
    model: Any,
    tokenizer: Any,
    *,
    rows: Sequence[Mapping[str, str]],
    device: Any,
    limit: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    validation = [row for row in rows if row["split"] == "validation"][:limit]
    if not validation:
        raise Lfm2CpcfBlankFoldRunnerError(
            "No validation prompts are available for standalone evaluation."
        )
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, Sequence) and not isinstance(eos, (str, bytes)):
        eos = eos[0] if eos else None
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = eos
    results = []
    model.eval()
    with torch.inference_mode():
        for row in validation:
            encoded = tokenizer(
                row["input_text"],
                add_special_tokens=True,
                return_tensors="pt",
            )
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise Lfm2CpcfBlankFoldRunnerError(
                    "Reloaded tokenizer returned no standalone input IDs."
                )
            input_ids = encoded["input_ids"].to(device=device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(device=device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad,
                eos_token_id=eos,
            )
            continuation = generated[0, int(input_ids.shape[1]) :].tolist()
            text = tokenizer.decode(
                continuation,
                skip_special_tokens=True,
            )
            repetition = _repetition_proxy(continuation)
            observed_token_count = len(continuation)
            token_count_matches_ids = repetition["token_count"] == observed_token_count
            within_generation_limit = observed_token_count <= max_new_tokens
            nonempty = bool(text.strip())
            control_count = sum(
                ord(character) < 32 and character not in "\n\r\t" for character in text
            )
            proxy_pass = nonempty and control_count == 0 and repetition["passed"]
            results.append(
                {
                    "record_id": row["id"],
                    "prompt": row["input_text"],
                    "cached_trajectory_reference": row["forced_decode_text"],
                    "child_output": text,
                    "child_output_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "continuation_token_ids": continuation,
                    "observed_continuation_token_count": observed_token_count,
                    "observed_token_count_matches_continuation_token_ids": (
                        token_count_matches_ids
                    ),
                    "observed_token_count_within_generation_limit": (
                        within_generation_limit
                    ),
                    "nonempty": nonempty,
                    "control_character_count": control_count,
                    "repetition_proxy": repetition,
                    "automatic_coherence_proxy_pass": proxy_pass,
                    "manual_intelligence_score": None,
                }
            )
    passed = sum(bool(result["automatic_coherence_proxy_pass"]) for result in results)
    body = {
        "schema": STANDALONE_EVALUATION_SCHEMA,
        "evaluation_mode": "fresh_reload_standalone_whole_response_generation",
        "requested_case_limit": limit,
        "requested_max_new_tokens": max_new_tokens,
        "generation_limit": max_new_tokens,
        "generation_limit_semantics": "maximum_new_tokens_per_case",
        "requested_max_new_tokens_matches_generation_limit": True,
        "case_count": len(results),
        "all_observed_token_counts_match_continuation_token_ids": all(
            bool(result["observed_token_count_matches_continuation_token_ids"])
            for result in results
        ),
        "all_observed_token_counts_within_generation_limit": all(
            bool(result["observed_token_count_within_generation_limit"])
            for result in results
        ),
        "automatic_coherence_proxy_pass_count": passed,
        "automatic_coherence_proxy_pass_fraction": passed / len(results),
        "cases": results,
        "exact_token_match_principal_objective": False,
        "cached_reference_is_not_refitted_or_reexecuted": True,
        "manual_semantic_intelligence_review_required": True,
        "manual_intelligence_retention_proven": False,
        "parent_model_calls": 0,
        "parent_generation_calls": 0,
        "target_generation_calls": len(results),
    }
    return {**body, "report_sha256": _sha(body)}


def _save_reload_evaluate(
    target_model: Any,
    target_tokenizer: Any,
    *,
    output: Path,
    parent_model: Any,
    planned_parameters: int,
    rows: Sequence[Mapping[str, str]],
    device: Any,
    dtype: Any,
    evaluation_limit: int,
    evaluation_max_new_tokens: int,
) -> tuple[
    tuple[
        Any,
        Mapping[str, Any],
        Mapping[str, Any],
        list[dict[str, Any]],
        Path,
        Mapping[str, Any],
    ],
    dict[str, Any],
]:
    from experiments.run_lfm2_residual_attention_fold import (
        load_local_lfm2_model,
        load_local_tokenizer,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        audit_lfm2_cpcf_parameter_compaction,
        independent_parameter_inventory,
    )

    stage = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    if stage.exists() or stage.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(f"Unexpected staging collision: {stage}.")
    stage.mkdir()
    try:
        target_aliases = _restore_lfm2_ffn_config_alias(
            target_model,
            "compiled_target_before_save",
        )
        target_model.save_pretrained(
            stage,
            safe_serialization=True,
        )
        target_tokenizer.save_pretrained(stage)
        saved_aliases = _verify_saved_lfm2_ffn_config(
            stage,
            expected_width=int(target_aliases["materialized_ffn_width"]),
        )
        saved_files = _checkpoint_inventory(stage)
        reloaded = load_local_lfm2_model(
            stage,
            device=device,
            dtype=dtype,
        )
        reloaded_aliases = _restore_lfm2_ffn_config_alias(
            reloaded,
            "freshly_reloaded_target",
        )
        alias_body = {
            "compiled_target": target_aliases,
            "saved_config": saved_aliases,
            "fresh_reload": reloaded_aliases,
            "materialized_width_preserved": (
                target_aliases["materialized_ffn_width"]
                == reloaded_aliases["materialized_ffn_width"]
            ),
            "block_ff_dim_restored_after_loader_consumption": True,
            "shape_stable": True,
        }
        alias_audit = {**alias_body, "report_sha256": _sha(alias_body)}
        reloaded_inventory = independent_parameter_inventory(reloaded)
        compression = audit_lfm2_cpcf_parameter_compaction(
            parent_model,
            reloaded,
            planned_target_unique_parameters=planned_parameters,
            promotion_compression_ratio=float(MINIMUM_CLAIMABLE_COMPRESSION_RATIO),
        )
        reloaded_tokenizer = load_local_tokenizer(stage)
        behavior = _standalone_whole_response_evaluation(
            reloaded,
            reloaded_tokenizer,
            rows=rows,
            device=device,
            limit=evaluation_limit,
            max_new_tokens=evaluation_max_new_tokens,
        )
        return (
            reloaded,
            reloaded_inventory,
            compression,
            saved_files,
            stage,
            alias_audit,
        ), behavior
    except BaseException as error:
        resolved = stage.resolve()
        try:
            if resolved.parent == output.parent.resolve() and resolved.name.startswith(
                f".{output.name}.stage-"
            ):
                shutil.rmtree(resolved)
        except BaseException as cleanup_error:
            error.add_note(
                "Standalone-save staging cleanup also failed: "
                f"{type(cleanup_error).__module__}."
                f"{type(cleanup_error).__qualname__}: {cleanup_error}"
            )
        raise


def _write_dry_plan_receipt(
    *,
    path: Path,
    parent_inventory: Mapping[str, Any],
    child_inventory: Mapping[str, Any],
    compression: Mapping[str, Any],
    blank: Any,
    cache: Any,
    bridge: Any,
) -> dict[str, Any]:
    body = {
        "schema": DRY_PLAN_SCHEMA,
        "status": "structural_dry_plan_only",
        "cache_key": cache.cache_key,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "blank_target": blank.as_dict(),
        "parent_parameter_inventory": dict(parent_inventory),
        "child_parameter_inventory": dict(child_inventory),
        "compression": dict(compression),
        "fresh_dense_blank_synthesized": True,
        "layer_compilation_performed": False,
        "folded_child_created": False,
        "standalone_behavior_evaluated": False,
        "intelligence_retention_claimed": False,
        "parameter_compaction_structurally_proven": True,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "fallback_used": False,
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    _atomic_json(path, receipt)
    return receipt


def _phase_evidence_sha256(phases: Sequence[Any]) -> str:
    rows = []
    for ordinal, phase in enumerate(phases):
        parent_evidence = getattr(phase, "parent_evidence", None)
        rows.append(
            {
                "ordinal": ordinal,
                "batch_index": getattr(phase, "batch_index", None),
                "record_id": getattr(phase, "record_id", None),
                "split": getattr(phase, "split", None),
                "phase": getattr(phase, "phase", None),
                "phase_sha256": getattr(phase, "phase_sha256", None),
                "parent_evidence_sha256": getattr(
                    parent_evidence,
                    "evidence_sha256",
                    None,
                ),
            }
        )
    if not rows:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume evidence cannot bind an empty phase collection."
        )
    try:
        return _sha(rows)
    except Lfm2CpcfBlankFoldRunnerError as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume phase evidence lacks canonical authenticated identities."
        ) from exc


def _build_resume_boundary_bindings(
    *,
    parent_fingerprint: str,
    cache_key: str,
    phases: Sequence[Any],
    trace_sha256: str,
    boundary_bridge_sha256: str,
    tokenizer_sha256: str,
    source_tied_table: Any,
    blank_report: Mapping[str, Any],
    policy: Any,
    compiler_fingerprint_sha256: str,
) -> Any:
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfBoundaryBindings,
    )

    boundary_policy = {
        "schema": RESUME_BOUNDARY_POLICY_SCHEMA,
        "policy": policy.as_dict(),
    }
    try:
        blank_sha = str(blank_report["blank_state_fingerprint"])
        target_sha = str(blank_report["target_topology_fingerprint"])
    except KeyError as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Blank target report lacks its state or topology fingerprint."
        ) from exc
    try:
        return Lfm2CpcfBoundaryBindings(
            parent_sha256=str(parent_fingerprint),
            cache_sha256=str(cache_key),
            evidence_sha256=_phase_evidence_sha256(phases),
            phase_id="all-target-phases",
            trace_sha256=str(trace_sha256),
            bridge_sha256=str(boundary_bridge_sha256),
            tokenizer_sha256=str(tokenizer_sha256),
            source_table_sha256=_tensor_payload_sha256(
                source_tied_table,
                "source_tied_table",
            ),
            blank_sha256=blank_sha,
            target_sha256=target_sha,
            boundary_policy_sha256=_sha(boundary_policy),
            solver_schema_sha256=str(compiler_fingerprint_sha256),
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Live resume inputs cannot form exact boundary bindings."
        ) from exc


def _global_installed_parameter_paths(target_model: Any) -> tuple[str, ...]:
    try:
        parameters = tuple(target_model.named_parameters(remove_duplicate=False))
    except (AttributeError, TypeError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Target model must expose aliased named parameters."
        ) from exc
    names = [str(name) for name, _parameter in parameters]
    if not names or len(names) != len(set(names)):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Target parameter inventory is empty or has duplicate paths."
        )
    return tuple(
        sorted(name for name in names if _LAYER_PARAMETER_PATH.search(name) is None)
    )


def _resume_invocation_sha256(
    *,
    boundary_bindings: Any,
    bridge_artifact_sha256: str,
    full_policy_sha256: str,
    compiler_fingerprint_sha256: str,
    checkpoint_reference: Any | None,
) -> str:
    bindings = dict(boundary_bindings.as_dict())
    # The boundary bridge is a compiler result. Invocation identity binds the
    # inputs and algorithms, while the boundary artifact binds that result.
    bindings.pop("bridge_sha256")
    body = {
        "schema": RESUME_INVOCATION_SCHEMA,
        "boundary_input_bindings": bindings,
        "tokenizer_bridge_artifact_sha256": bridge_artifact_sha256,
        "full_policy_sha256": full_policy_sha256,
        "compiler_fingerprint_sha256": compiler_fingerprint_sha256,
        "resume_checkpoint_reference": (
            None if checkpoint_reference is None else checkpoint_reference.as_dict()
        ),
    }
    return _sha(body)


def _prepare_resume_state(
    *,
    store: Any,
    checkpoint_reference: Any,
    checkpoint_pointer: Mapping[str, Any] | None,
    target_model: Any,
    phases: Sequence[Any],
    parent_fingerprint: str,
    cache_key: str,
    trace_sha256: str,
    tokenizer_sha256: str,
    source_tied_table: Any,
    blank_report: Mapping[str, Any],
    policy: Any,
    bridge_artifact_sha256: str,
    full_policy_sha256: str,
    compiler_fingerprint_sha256: str,
) -> tuple[Any, Mapping[str, Any], Any, Any]:
    from neural_forge.dynamic_fold_executor import fingerprint_model_state
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        Lfm2CpcfModelFoldResumeState,
    )
    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfArtifactReference,
    )
    from neural_forge.hf_lfm2_cpcf_resume_bridge import (
        install_lfm2_cpcf_prefix_checkpoint,
        open_lfm2_cpcf_boundary_bundle,
        reconstruct_lfm2_cpcf_phase_state_bank,
    )

    verified = store.open_checkpoint(checkpoint_reference)
    checkpoint_bindings = verified.manifest["bindings"]
    if (
        checkpoint_bindings["full_policy_sha256"] != full_policy_sha256
        or checkpoint_bindings["compiler_fingerprint_sha256"]
        != compiler_fingerprint_sha256
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume checkpoint policy or compiler implementation differs "
            "from this exact fold invocation."
        )
    raw_boundary_reference = checkpoint_bindings["boundary"]
    try:
        boundary_reference = Lfm2CpcfArtifactReference(
            artifact_kind=str(raw_boundary_reference["artifact_kind"]),
            artifact_key=str(raw_boundary_reference["artifact_key"]),
            manifest_sha256=str(raw_boundary_reference["manifest_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Resume checkpoint has no valid pinned boundary reference."
        ) from exc
    verified_boundary = store.open_boundary(boundary_reference)
    boundary_bridge_sha256 = str(
        verified_boundary.manifest["bindings"]["bridge_sha256"]
    )
    expected_boundary_bindings = _build_resume_boundary_bindings(
        parent_fingerprint=parent_fingerprint,
        cache_key=cache_key,
        phases=phases,
        trace_sha256=trace_sha256,
        boundary_bridge_sha256=boundary_bridge_sha256,
        tokenizer_sha256=tokenizer_sha256,
        source_tied_table=source_tied_table,
        blank_report=blank_report,
        policy=policy,
        compiler_fingerprint_sha256=compiler_fingerprint_sha256,
    )
    if checkpoint_pointer is not None:
        raw_base = checkpoint_pointer["base_checkpoint_reference"]
        base_reference = (
            None
            if raw_base is None
            else _artifact_reference_from_mapping(
                raw_base,
                artifact_kind="checkpoint",
                name="latest pointer base checkpoint",
            )
        )
        expected_invocation = _resume_invocation_sha256(
            boundary_bindings=expected_boundary_bindings,
            bridge_artifact_sha256=bridge_artifact_sha256,
            full_policy_sha256=full_policy_sha256,
            compiler_fingerprint_sha256=(compiler_fingerprint_sha256),
            checkpoint_reference=base_reference,
        )
        _validate_latest_resume_pointer(
            checkpoint_pointer,
            expected_invocation_sha256=expected_invocation,
            expected_boundary_binding_key=(expected_boundary_bindings.binding_key),
            expected_boundary_reference=boundary_reference,
            expected_full_policy_sha256=full_policy_sha256,
            expected_compiler_fingerprint_sha256=(compiler_fingerprint_sha256),
            expected_base_checkpoint_reference=base_reference,
            expected_checkpoint_reference=checkpoint_reference,
            expected_next_layer_index=verified.completed_layers["next_layer_index"],
        )
    verified_boundary = store.open_boundary(
        boundary_reference,
        expected_bindings=expected_boundary_bindings,
    )
    boundary_bundle = open_lfm2_cpcf_boundary_bundle(
        store,
        boundary_reference,
        expected_bindings=expected_boundary_bindings,
    )
    incoming_state_bank = reconstruct_lfm2_cpcf_phase_state_bank(
        store,
        verified,
        phases=phases,
    )
    blank_state_sha256 = fingerprint_model_state(target_model)
    prefix_install_report = install_lfm2_cpcf_prefix_checkpoint(
        store,
        verified,
        target_model,
        expected_blank_state_sha256=blank_state_sha256,
    )
    completed_layer_receipts = tuple(
        verified.completed_layers["completed_layer_receipts"]
    )
    state = Lfm2CpcfModelFoldResumeState(
        prefix_install_report=prefix_install_report,
        full_policy_sha256=full_policy_sha256,
        boundary_bundle=boundary_bundle,
        incoming_state_bank=incoming_state_bank,
        completed_layer_receipts=completed_layer_receipts,
    )
    audit = {
        "checkpoint_reference": checkpoint_reference.as_dict(),
        "boundary_reference": boundary_reference.as_dict(),
        "next_layer_index": state.next_layer_index,
        "prefix_install_report_sha256": prefix_install_report["report_sha256"],
        "full_policy_sha256": full_policy_sha256,
        "compiler_fingerprint_sha256": compiler_fingerprint_sha256,
        "boundary_binding_key": (expected_boundary_bindings.binding_key),
        "parent_evidence_tensors_copied": 0,
        "model_checkpoint_loaders_used": 0,
        "authenticated_safetensor_archive_reads_used": True,
        "self_sealed_latest_pointer_verified": (checkpoint_pointer is not None),
    }
    return state, audit, verified_boundary, verified


def _publish_failure_resume_context(
    *,
    store: Any,
    failure: Any,
    target_model: Any,
    phases: Sequence[Any],
    parent_fingerprint: str,
    cache_key: str,
    trace_sha256: str,
    tokenizer_sha256: str,
    source_tied_table: Any,
    blank_report: Mapping[str, Any],
    policy: Any,
    bridge_artifact_sha256: str,
    full_policy_sha256: str,
    compiler_fingerprint_sha256: str,
    checkpoint_reference: Any | None,
) -> Any:
    from neural_forge.hf_lfm2_cpcf_resume_bridge import (
        flatten_lfm2_cpcf_boundary_bundle,
        publish_lfm2_cpcf_model_fold_failure_context,
    )

    if failure.boundary_bundle is None:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Compiler failed before a resumable boundary bundle existed."
        )
    if len(failure.completed_layer_receipts) >= len(failure.boundary_bundle.layers):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Compiler failure has no unfinished layer; the current model "
            "continuation contract cannot resume a final-norm-only failure."
        )
    flattened = flatten_lfm2_cpcf_boundary_bundle(failure.boundary_bundle)
    bindings = _build_resume_boundary_bindings(
        parent_fingerprint=parent_fingerprint,
        cache_key=cache_key,
        phases=phases,
        trace_sha256=trace_sha256,
        boundary_bridge_sha256=flattened.bridge_sha256,
        tokenizer_sha256=tokenizer_sha256,
        source_tied_table=source_tied_table,
        blank_report=blank_report,
        policy=policy,
        compiler_fingerprint_sha256=compiler_fingerprint_sha256,
    )
    invocation_sha256 = _resume_invocation_sha256(
        boundary_bindings=bindings,
        bridge_artifact_sha256=bridge_artifact_sha256,
        full_policy_sha256=full_policy_sha256,
        compiler_fingerprint_sha256=compiler_fingerprint_sha256,
        checkpoint_reference=checkpoint_reference,
    )
    report = failure.report
    rollback = failure.rollback_proof
    return publish_lfm2_cpcf_model_fold_failure_context(
        store,
        failure,
        boundary_bindings=bindings,
        full_policy_sha256=full_policy_sha256,
        compiler_fingerprint_sha256=compiler_fingerprint_sha256,
        invocation_sha256=invocation_sha256,
        guard_proof_sha256=report["failed_prefix_target_cpu_tensor_snapshot"][
            "snapshot_sha256"
        ],
        parent_proof_sha256=rollback["source_state_fingerprint_before"],
        rollback_proof_sha256=rollback["proof_sha256"],
        global_installed_parameter_paths=(
            _global_installed_parameter_paths(target_model)
        ),
    )


@dataclass(slots=True)
class _Lfm2CpcfResumeProgressPublisher:
    """Publish ordered resumable prefix events without retaining event data."""

    store: Any
    phases: Sequence[Any]
    parent_fingerprint: str
    cache_key: str
    trace_sha256: str
    tokenizer_sha256: str
    source_tied_table: Any
    blank_report: Mapping[str, Any]
    policy: Any
    full_policy_sha256: str
    compiler_fingerprint_sha256: str
    bridge_artifact_sha256: str
    global_installed_parameter_paths: tuple[str, ...]
    base_checkpoint_reference: Any | None = None
    pinned_boundary: Any | None = None
    latest_checkpoint: Any | None = None
    boundary_bindings: Any | None = None
    latest_pointer_path: Path | None = None
    observed_event_count: int = 0
    published_checkpoint_count: int = 0
    skipped_nonresumable_event_count: int = 0

    def __call__(self, event: Any) -> None:
        from neural_forge.hf_lfm2_cpcf_resume_bridge import (
            flatten_lfm2_cpcf_boundary_bundle,
            publish_lfm2_cpcf_model_fold_progress_checkpoint,
        )

        event.verify()
        if event.full_policy_sha256 != self.full_policy_sha256:
            raise Lfm2CpcfBlankFoldRunnerError(
                "Progress event policy differs from the runner-bound policy."
            )
        bundle = event.boundary_bundle
        layer_count = len(bundle.layers)
        next_layer_index = event.next_layer_index
        self.observed_event_count += 1
        if next_layer_index >= layer_count:
            if next_layer_index != layer_count:
                raise Lfm2CpcfBlankFoldRunnerError(
                    "Progress event exceeds the target layer count."
                )
            self.skipped_nonresumable_event_count += 1
            return
        flattened = flatten_lfm2_cpcf_boundary_bundle(bundle)
        if self.boundary_bindings is None:
            bindings = _build_resume_boundary_bindings(
                parent_fingerprint=self.parent_fingerprint,
                cache_key=self.cache_key,
                phases=self.phases,
                trace_sha256=self.trace_sha256,
                boundary_bridge_sha256=flattened.bridge_sha256,
                tokenizer_sha256=self.tokenizer_sha256,
                source_tied_table=self.source_tied_table,
                blank_report=self.blank_report,
                policy=self.policy,
                compiler_fingerprint_sha256=(self.compiler_fingerprint_sha256),
            )
        else:
            bindings = self.boundary_bindings
            if bindings.bridge_sha256 != flattened.bridge_sha256:
                raise Lfm2CpcfBlankFoldRunnerError(
                    "Progress boundary changed after its first publication."
                )
        previous_checkpoint = self.latest_checkpoint
        publications = publish_lfm2_cpcf_model_fold_progress_checkpoint(
            self.store,
            event,
            boundary_bindings=bindings,
            full_policy_sha256=self.full_policy_sha256,
            compiler_fingerprint_sha256=(self.compiler_fingerprint_sha256),
            global_installed_parameter_paths=(self.global_installed_parameter_paths),
            pinned_boundary=self.pinned_boundary,
            previous_checkpoint=previous_checkpoint,
        )
        verified_boundary = self.store.final_reauthenticate_boundary(
            publications.boundary
        )
        verified_checkpoint = self.store.final_reauthenticate_checkpoint(
            publications.checkpoint
        )
        if verified_checkpoint.completed_layers[
            "next_layer_index"
        ] != next_layer_index or (
            self.pinned_boundary is not None
            and verified_boundary.reference != self.pinned_boundary.reference
        ):
            raise Lfm2CpcfBlankFoldRunnerError(
                "Progress publication returned the wrong prefix or boundary."
            )
        pointer_path, _pointer = _publish_latest_resume_pointer(
            store=self.store,
            boundary_bindings=bindings,
            boundary=verified_boundary,
            checkpoint=verified_checkpoint,
            full_policy_sha256=self.full_policy_sha256,
            compiler_fingerprint_sha256=(self.compiler_fingerprint_sha256),
            bridge_artifact_sha256=self.bridge_artifact_sha256,
            base_checkpoint_reference=self.base_checkpoint_reference,
            previous_checkpoint=previous_checkpoint,
            event_report_sha256=event.report_sha256,
        )
        # Update only after every store and lineage reauthentication succeeds.
        # No event, target snapshot, or state-bank tensor is retained.
        self.boundary_bindings = bindings
        self.pinned_boundary = verified_boundary
        self.latest_checkpoint = verified_checkpoint
        self.latest_pointer_path = pointer_path
        self.published_checkpoint_count += 1

    @property
    def latest_checkpoint_manifest(self) -> Path | None:
        if self.latest_checkpoint is None:
            return None
        return self.latest_checkpoint.path / "manifest.json"

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_event_count": self.observed_event_count,
            "published_checkpoint_count": self.published_checkpoint_count,
            "skipped_nonresumable_event_count": (self.skipped_nonresumable_event_count),
            "latest_resumable_checkpoint_reference": (
                None
                if self.latest_checkpoint is None
                else self.latest_checkpoint.reference.as_dict()
            ),
            "latest_resumable_next_layer_index": (
                None
                if self.latest_checkpoint is None
                else self.latest_checkpoint.completed_layers["next_layer_index"]
            ),
            "boundary_reference": (
                None
                if self.pinned_boundary is None
                else self.pinned_boundary.reference.as_dict()
            ),
            "latest_pointer_path": (
                None
                if self.latest_pointer_path is None
                else str(self.latest_pointer_path)
            ),
            "event_objects_retained": 0,
            "parent_evidence_tensors_copied": 0,
        }


def run(config: RunnerConfig) -> dict[str, Any]:
    """Execute one strict local CPCF fold or an explicit structural dry plan."""

    import torch

    from neural_forge.hf_lfm2_circuit_trace_cache import (
        fingerprint_lfm2_parent,
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_blank_target import (
        plan_and_synthesize_lfm2_cpcf_blank,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        Lfm2CpcfModelFoldFailure,
        Lfm2CpcfModelFoldPolicy,
        audit_lfm2_cpcf_parameter_compaction,
        compile_lfm2_cpcf_model_fold,
        independent_parameter_inventory,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
    )
    from neural_forge.hf_tokenizer_bridge import (
        fingerprint_hf_tokenizer_semantics,
        load_strict_tokenizer_bridge_artifact,
    )
    from experiments.run_lfm2_residual_attention_fold import (
        direct_local_io_audit,
        load_local_lfm2_model,
        load_local_tokenizer,
    )

    parent_directory = _local_directory(
        config.parent_model_directory,
        "parent_model_directory",
    )
    donor_tokenizer_directory = _local_directory(
        config.donor_tokenizer_directory,
        "donor_tokenizer_directory",
    )
    target_tokenizer_directory = _local_directory(
        config.target_tokenizer_directory,
        "target_tokenizer_directory",
    )
    bridge_directory = _local_directory(
        config.bridge_directory,
        "bridge_directory",
    )
    cache_directory = _local_directory(
        config.trace_cache_directory,
        "trace_cache_directory",
    )
    trajectory_path = _local_file(
        config.trajectory_corpus,
        "trajectory_corpus",
    )
    output = _new_output_directory(config.output_directory)
    transactional_receipt_path = output / "cpcf_run_receipt.json"
    if config.structural_dry_plan:
        receipt_path = (
            output.with_suffix(".receipt.json")
            if config.receipt is None
            else config.receipt.expanduser().absolute()
        )
    else:
        receipt_path = (
            transactional_receipt_path
            if config.receipt is None
            else config.receipt.expanduser().absolute()
        )
        if receipt_path != transactional_receipt_path:
            raise Lfm2CpcfBlankFoldRunnerError(
                "A full fold receipt must be the atomic internal path "
                f"{transactional_receipt_path}; omit --receipt or pass that "
                "exact path."
            )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise Lfm2CpcfBlankFoldRunnerError(f"Receipt already exists: {receipt_path}.")
    dtype = _torch_dtype(torch, config.dtype)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Lfm2CpcfBlankFoldRunnerError("CUDA was requested but is unavailable.")

    donor_tokenizer = load_local_tokenizer(donor_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    bridge.verify_files()
    donor_fingerprint = fingerprint_hf_tokenizer_semantics(donor_tokenizer)
    target_fingerprint = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    if (
        donor_fingerprint != bridge.donor_tokenizer_semantics_sha256
        or target_fingerprint != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Live local tokenizers differ from the sealed bridge."
        )

    # Metadata is opened first; the installability adapter performs the full
    # cache verification.  We intentionally do not run its separate audit()
    # pass, which would rematerialize every broad-cache tensor a second time.
    cache = open_lfm2_circuit_trace_cache(cache_directory, verify=False)
    parent = load_local_lfm2_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    parent.requires_grad_(False)
    parent.eval()
    parent_fingerprint_before = fingerprint_lfm2_parent(parent)
    if parent_fingerprint_before.identity_sha256 != cache.parent_fingerprint:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Explicit local parent differs from the sealed trace cache."
        )
    parent_inventory = independent_parameter_inventory(parent)
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name=config.parent_name,
        parent_tokenizer_fingerprint=donor_fingerprint,
        evidence_fingerprint=cache.cache_key,
        child_name=config.child_name,
        target_vocabulary_size=(bridge.transport_profile.target_row_capacity),
        child_tokenizer_fingerprint=target_fingerprint,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=config.minimum_hidden_size,
        maximum_candidates=config.maximum_candidates,
        selected_candidate_rank=config.selected_candidate_rank,
        blank_seed=config.blank_seed,
        device=device,
        dtype=dtype,
        **config.compression.as_kwargs(),
    )
    target = blank.model
    target.requires_grad_(False)
    target.eval()
    blank_inventory = independent_parameter_inventory(target)
    if config.compression.target_unique_parameters is not None and (
        blank.unique_parameters != config.compression.target_unique_parameters
        or int(blank_inventory["unique_parameters"])
        != config.compression.target_unique_parameters
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "The synthesized blank does not exactly match the requested "
            "parameter count."
        )
    structural_compression = audit_lfm2_cpcf_parameter_compaction(
        parent,
        target,
        planned_target_unique_parameters=blank.unique_parameters,
        promotion_compression_ratio=float(MINIMUM_CLAIMABLE_COMPRESSION_RATIO),
    )
    if (
        not structural_compression["promotion_ratio_met"]
        and not config.diagnostic_control
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "The independently measured target is a valid diagnostic fold "
            f"but is below the {MINIMUM_CLAIMABLE_COMPRESSION_RATIO}x "
            "project-promotion floor. Pass --diagnostic-control to run it "
            "without claiming project-goal success."
        )
    if config.structural_dry_plan:
        return _write_dry_plan_receipt(
            path=receipt_path,
            parent_inventory=parent_inventory,
            child_inventory=blank_inventory,
            compression=structural_compression,
            blank=blank,
            cache=cache,
            bridge=bridge,
        )

    from neural_forge.hf_lfm2_cpcf_installability import (
        Lfm2CpcfInstallabilityAdapter,
    )
    from neural_forge.hf_lfm2_cpcf_model_compiler import (
        build_lfm2_cpcf_target_phases,
    )

    rows = _read_trajectory_rows(trajectory_path)
    trace_adapter = _trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache_manifest=cache.manifest,
    )
    installability = Lfm2CpcfInstallabilityAdapter(cache)
    # Passing the generator directly avoids a second resident parent-evidence
    # tuple.  Target phases are the sole long-lived owner of the authenticated
    # broad-cache tensors used by all subsequent components.
    phases = build_lfm2_cpcf_target_phases(
        trace_adapter,
        installability.iter_compiler_inputs(),
    )
    if len({phase.batch_index for phase in phases}) != cache.batch_count:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Materialized compiler phases do not cover every cache batch."
        )
    source_tied_table = _mapped_source_tied_table(parent, bridge)
    short_conv_v4_solver_policy = (
        None
        if config.short_conv_fold_version == "v3"
        else Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=config.short_conv_v4_ridge_alphas,
            variable_projection_passes=(
                config.short_conv_v4_variable_projection_passes
            ),
            writer_trust_radius=(config.short_conv_v4_writer_trust_radius),
            fir_trust_radius=config.short_conv_v4_fir_trust_radius,
            bcx_trust_radius=config.short_conv_v4_bcx_trust_radius,
            correction_rms_floor=(config.short_conv_v4_correction_rms_floor),
            context_objective_weight=(config.short_conv_v4_context_objective_weight),
            target_dtype=config.dtype,
        )
    )
    fold_policy = Lfm2CpcfModelFoldPolicy(
        ridge=config.ridge,
        state_reconstruction_weight=(config.boundary_state_reconstruction_weight),
        minimum_predictive_energy_fraction=(config.boundary_predictive_energy_floor),
        minimum_validation_block_retention=(config.boundary_validation_retention_floor),
        enforce_validation_block_floor=True,
        short_conv_v4_solver_policy=short_conv_v4_solver_policy,
        promotion_compression_ratio=float(MINIMUM_CLAIMABLE_COMPRESSION_RATIO),
        require_promotion_ratio=not config.diagnostic_control,
    )
    resume_store = None
    resume_checkpoint_selection = None
    resume_checkpoint_reference = None
    resume_state = None
    resume_audit = None
    resume_verified_boundary = None
    resume_verified_checkpoint = None
    progress_publisher = None
    compiler_fingerprint_sha256 = None
    full_policy_sha256 = None
    if config.resume_store_directory is not None:
        from neural_forge.hf_lfm2_cpcf_resume_artifact import (
            Lfm2CpcfResumeArtifactStore,
        )

        resume_store = Lfm2CpcfResumeArtifactStore(config.resume_store_directory)
        compiler_fingerprint_sha256 = _resume_compiler_fingerprint()
        full_policy_sha256 = _sha(fold_policy.as_dict())
        if config.resume_checkpoint_reference is not None:
            resume_checkpoint_selection = _read_resume_checkpoint_reference(
                config.resume_checkpoint_reference
            )
            resume_checkpoint_reference = resume_checkpoint_selection.reference
            (
                resume_state,
                resume_audit,
                resume_verified_boundary,
                resume_verified_checkpoint,
            ) = _prepare_resume_state(
                store=resume_store,
                checkpoint_reference=resume_checkpoint_reference,
                checkpoint_pointer=(resume_checkpoint_selection.latest_pointer),
                target_model=target,
                phases=phases,
                parent_fingerprint=cache.parent_fingerprint,
                cache_key=cache.cache_key,
                trace_sha256=trace_adapter.evidence_sha256,
                tokenizer_sha256=target_fingerprint,
                source_tied_table=source_tied_table,
                blank_report=blank.as_dict(),
                policy=fold_policy,
                bridge_artifact_sha256=bridge.artifact_sha256,
                full_policy_sha256=full_policy_sha256,
                compiler_fingerprint_sha256=(compiler_fingerprint_sha256),
            )
        progress_publisher = _Lfm2CpcfResumeProgressPublisher(
            store=resume_store,
            phases=phases,
            parent_fingerprint=cache.parent_fingerprint,
            cache_key=cache.cache_key,
            trace_sha256=trace_adapter.evidence_sha256,
            tokenizer_sha256=target_fingerprint,
            source_tied_table=source_tied_table,
            blank_report=blank.as_dict(),
            policy=fold_policy,
            full_policy_sha256=full_policy_sha256,
            compiler_fingerprint_sha256=compiler_fingerprint_sha256,
            bridge_artifact_sha256=bridge.artifact_sha256,
            global_installed_parameter_paths=(
                _global_installed_parameter_paths(target)
            ),
            base_checkpoint_reference=resume_checkpoint_reference,
            pinned_boundary=resume_verified_boundary,
            latest_checkpoint=resume_verified_checkpoint,
        )
    compile_kwargs: dict[str, Any] = {
        "phases": phases,
        "source_tied_table": source_tied_table,
        "blank_target_report": blank.as_dict(),
        "planned_target_unique_parameters": blank.unique_parameters,
        "policy": fold_policy,
    }
    if resume_state is not None:
        compile_kwargs["resume_state"] = resume_state
    if progress_publisher is not None:
        compile_kwargs["progress_observer"] = progress_publisher
    try:
        compiled_fold = compile_lfm2_cpcf_model_fold(
            parent,
            target,
            **compile_kwargs,
        )
    except Lfm2CpcfModelFoldFailure as failure:
        if (
            resume_store is not None
            and compiler_fingerprint_sha256 is not None
            and full_policy_sha256 is not None
        ):
            try:
                publications = _publish_failure_resume_context(
                    store=resume_store,
                    failure=failure,
                    target_model=target,
                    phases=phases,
                    parent_fingerprint=cache.parent_fingerprint,
                    cache_key=cache.cache_key,
                    trace_sha256=trace_adapter.evidence_sha256,
                    tokenizer_sha256=target_fingerprint,
                    source_tied_table=source_tied_table,
                    blank_report=blank.as_dict(),
                    policy=fold_policy,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                    full_policy_sha256=full_policy_sha256,
                    compiler_fingerprint_sha256=(compiler_fingerprint_sha256),
                    checkpoint_reference=(resume_checkpoint_reference),
                )
            except Exception as publication_error:
                failure.add_note(
                    "Resume publication failed without replacing the original "
                    "authenticated model-fold failure: "
                    f"{type(publication_error).__qualname__}: "
                    f"{publication_error}"
                )
            else:
                failure.add_note(
                    "Authenticated failed-prefix diagnostic checkpoint "
                    "published: "
                    f"{publications.checkpoint.path / 'manifest.json'}"
                )
                failure.add_note(
                    "Authenticated failure diagnostic published: "
                    f"{publications.failure.path / 'diagnostic.json'}"
                )
        if (
            progress_publisher is not None
            and progress_publisher.latest_checkpoint_manifest is not None
        ):
            failure.add_note(
                "Latest accepted resumable progress checkpoint remains: "
                f"{progress_publisher.latest_checkpoint_manifest}"
            )
        raise
    except BaseException as error:
        if (
            progress_publisher is not None
            and progress_publisher.latest_checkpoint_manifest is not None
        ):
            error.add_note(
                "Latest accepted resumable progress checkpoint remains "
                "after exact model rollback: "
                f"{progress_publisher.latest_checkpoint_manifest}"
            )
        raise
    compiled_inventory = independent_parameter_inventory(target)
    if int(compiled_inventory["unique_parameters"]) != int(
        blank_inventory["unique_parameters"]
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Folding changed the child topology or parameter count."
        )
    parent_fingerprint_after = fingerprint_lfm2_parent(parent)
    if parent_fingerprint_after.identity_sha256 != (
        parent_fingerprint_before.identity_sha256
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Parent parameters or runtime buffers changed during folding."
        )

    saved, behavior = _save_reload_evaluate(
        target,
        target_tokenizer,
        output=output,
        parent_model=parent,
        planned_parameters=blank.unique_parameters,
        rows=rows,
        device=device,
        dtype=dtype,
        evaluation_limit=config.evaluation_limit,
        evaluation_max_new_tokens=config.evaluation_max_new_tokens,
    )
    (
        reloaded,
        reloaded_inventory,
        reload_compression,
        saved_files,
        stage,
        save_reload_config_aliases,
    ) = saved
    del reloaded
    if (
        reload_compression["target_unique_parameters"]
        != structural_compression["target_unique_parameters"]
    ):
        raise Lfm2CpcfBlankFoldRunnerError(
            "Fresh reload changed the independently measured parameter count."
        )
    operation_audit = {
        "local_parent_model_loads": 1,
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "parent_cache_generation_calls": 0,
        "parent_cache_regeneration_calls": 0,
        "sealed_cache_open_count": 1,
        "resident_parent_evidence_collections": 1,
        "separate_installability_audit_rematerialization": False,
        "target_generation_calls": behavior["target_generation_calls"],
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "quantized_parameters": 0,
        "pruned_parameters": 0,
        "fallback_model_calls": 0,
        "network_access": False,
    }
    body = {
        "schema": RUN_SCHEMA,
        "status": (
            "folded_child_requires_manual_intelligence_review"
            if reload_compression["promotion_ratio_met"]
            else "diagnostic_control_fold_requires_manual_review"
        ),
        "inputs": {
            "parent_model_directory": str(parent_directory),
            "donor_tokenizer_directory": str(donor_tokenizer_directory),
            "target_tokenizer_directory": str(target_tokenizer_directory),
            "bridge_directory": str(bridge_directory),
            "trace_cache_directory": str(cache_directory),
            "trajectory_corpus": str(trajectory_path),
            "trajectory_corpus_sha256": _file_sha256(trajectory_path),
            "cache_key": cache.cache_key,
            "parent_fingerprint": cache.parent_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "bridge_profile_sha256": bridge.profile_fingerprint,
            "trace_adapter_sha256": trace_adapter.evidence_sha256,
        },
        "local_io_audit": direct_local_io_audit(parent_directory),
        "blank_target": blank.as_dict(),
        "parent_parameter_inventory": dict(parent_inventory),
        "blank_parameter_inventory": dict(blank_inventory),
        "compiled_parameter_inventory": dict(compiled_inventory),
        "reloaded_parameter_inventory": dict(reloaded_inventory),
        "compression": dict(reload_compression),
        "compiled_model_fold": compiled_fold.as_dict(),
        "boundary_bundle": compiled_fold.boundary_bundle.as_dict(),
        "interface_install": dict(compiled_fold.interface_install_report),
        "neutral_norm_install": dict(compiled_fold.neutral_norm_install_report),
        "layer_results": [result.as_dict() for result in compiled_fold.layer_results],
        "final_norm": dict(compiled_fold.final_norm_report),
        "standalone_whole_response_evaluation": behavior,
        "saved_checkpoint_files": saved_files,
        "save_reload_config_aliases": dict(save_reload_config_aliases),
        "output_directory": str(output),
        "parent_state_preserved": True,
        "all_target_layers_compiled_sequentially": True,
        "fresh_reload_verified": True,
        "structural_parameter_compaction_proven": True,
        "project_promotion_ratio_met": reload_compression["promotion_ratio_met"],
        "diagnostic_control": config.diagnostic_control,
        "behavioral_intelligence_retention_proven": False,
        "manual_intelligence_review_required": True,
        "operation_audit": operation_audit,
        "claims": {
            "folding_only": True,
            "blank_target_filled": True,
            "training_used": False,
            "quantization_used": False,
            "pruning_used": False,
            "parent_backed_fallback_used": False,
            "exact_token_match_used_as_principal_objective": False,
            "marginal_compaction_reported_as_success": False,
        },
    }
    if resume_audit is not None:
        body["resume_continuation"] = {
            **dict(resume_audit),
            "authenticated_prefix_continuation_used": True,
            "fresh_blank_created_before_prefix_install": True,
            "parent_evidence_tensors_copied": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "promotion_claimed_by_checkpoint": False,
        }
    if progress_publisher is not None:
        body["resume_progress_checkpoints"] = {
            **progress_publisher.as_dict(),
            "synchronous_publication": True,
            "latest_pointer_updated_only_after_final_reauthentication": True,
            "nonresumable_final_layer_event_skipped": True,
            "checkpoint_artifacts_non_promotable": True,
        }
    receipt = {**body, "receipt_sha256": _sha(body)}
    internal_receipt = stage / "cpcf_run_receipt.json"
    _atomic_json(internal_receipt, receipt)
    if output.exists():
        raise Lfm2CpcfBlankFoldRunnerError(
            "Output appeared while the child was being compiled."
        )
    os.replace(stage, output)
    if receipt_path != output / internal_receipt.name:
        raise Lfm2CpcfBlankFoldRunnerError(
            "Canonical receipt did not publish atomically with the child."
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
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
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument(
        "--trace-cache-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--trajectory-corpus", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        help=(
            "Dry plans may use an external receipt. Full folds publish "
            "cpcf_run_receipt.json atomically inside the output directory; "
            "omit this flag or pass that exact future path."
        ),
    )
    parser.add_argument(
        "--resume-store-directory",
        type=Path,
        help=(
            "Content-addressed non-promotable boundary/checkpoint/failure "
            "store. On an authenticated model-fold failure, the completed "
            "prefix and child state bank are published here."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint-reference",
        type=Path,
        help=(
            "Pinned three-field checkpoint reference JSON or the exact "
            "manifest.json published under the resume store, or a self-sealed "
            "invocation latest-pointer JSON. Requires --resume-store-directory."
        ),
    )
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--compression-ratio")
    budget.add_argument("--target-unique-parameters", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=_DTYPES, default="bfloat16")
    parser.add_argument("--parent-name", default="lfm2-parent")
    parser.add_argument("--child-name", default="lfm2-cpcf-folded-child")
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--minimum-hidden-size", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=8)
    parser.add_argument("--selected-candidate-rank", type=int)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument(
        "--short-conv-fold-version",
        choices=("v3", "v4"),
        default="v3",
    )
    parser.add_argument(
        "--short-conv-v4-ridge-alpha",
        action="append",
        type=float,
        dest="short_conv_v4_ridge_alphas",
        help=(
            "Repeat for each strictly increasing v4 ridge candidate. "
            "Defaults to 1e-8 and 1e-4."
        ),
    )
    parser.add_argument(
        "--short-conv-v4-variable-projection-passes",
        choices=(2, 3),
        type=int,
        default=2,
    )
    parser.add_argument(
        "--short-conv-v4-writer-trust-radius",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--short-conv-v4-fir-trust-radius",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--short-conv-v4-bcx-trust-radius",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--short-conv-v4-correction-rms-floor",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--short-conv-v4-context-objective-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--boundary-state-reconstruction-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--boundary-predictive-energy-floor",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--boundary-validation-retention-floor",
        type=float,
        default=0.0,
    )
    parser.add_argument("--evaluation-limit", type=int, default=12)
    parser.add_argument(
        "--evaluation-max-new-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--structural-dry-plan",
        action="store_true",
        help=(
            "Synthesize and count the dense blank, then stop. This mode "
            "cannot claim a folded or behaviorally valid child."
        ),
    )
    parser.add_argument(
        "--diagnostic-control",
        action="store_true",
        help=(
            "Allow a real fold below the current 100x promotion floor while "
            "labeling it explicitly as a non-goal diagnostic control."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        parent_model_directory=args.parent_model_directory,
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        bridge_directory=args.bridge_directory,
        trace_cache_directory=args.trace_cache_directory,
        trajectory_corpus=args.trajectory_corpus,
        output_directory=args.output_directory,
        receipt=args.receipt,
        compression=CompressionRequest(
            compression_ratio=args.compression_ratio,
            target_unique_parameters=args.target_unique_parameters,
        ),
        device=args.device,
        dtype=args.dtype,
        parent_name=args.parent_name,
        child_name=args.child_name,
        blank_seed=args.blank_seed,
        minimum_hidden_size=args.minimum_hidden_size,
        maximum_candidates=args.maximum_candidates,
        selected_candidate_rank=args.selected_candidate_rank,
        ridge=args.ridge,
        boundary_state_reconstruction_weight=(
            args.boundary_state_reconstruction_weight
        ),
        boundary_predictive_energy_floor=(args.boundary_predictive_energy_floor),
        boundary_validation_retention_floor=(args.boundary_validation_retention_floor),
        evaluation_limit=args.evaluation_limit,
        evaluation_max_new_tokens=args.evaluation_max_new_tokens,
        structural_dry_plan=args.structural_dry_plan,
        diagnostic_control=args.diagnostic_control,
        resume_store_directory=args.resume_store_directory,
        resume_checkpoint_reference=args.resume_checkpoint_reference,
        short_conv_fold_version=args.short_conv_fold_version,
        short_conv_v4_ridge_alphas=(
            (1e-8, 1e-4)
            if args.short_conv_v4_ridge_alphas is None
            else tuple(args.short_conv_v4_ridge_alphas)
        ),
        short_conv_v4_variable_projection_passes=(
            args.short_conv_v4_variable_projection_passes
        ),
        short_conv_v4_writer_trust_radius=(args.short_conv_v4_writer_trust_radius),
        short_conv_v4_fir_trust_radius=(args.short_conv_v4_fir_trust_radius),
        short_conv_v4_bcx_trust_radius=args.short_conv_v4_bcx_trust_radius,
        short_conv_v4_correction_rms_floor=(args.short_conv_v4_correction_rms_floor),
        short_conv_v4_context_objective_weight=(
            args.short_conv_v4_context_objective_weight
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(config_from_args(args))
    except (
        Lfm2CpcfBlankFoldRunnerError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"note: {note}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
