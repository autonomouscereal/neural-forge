"""Immutable, content-addressed cache for accepted structural child compiles.

The cache deliberately separates structural compiler identity from downstream
behavior training.  Trainer implementations, optimizer settings, selection
thresholds, and training schedules are rejected from the identity payload so a
trainer-only repair does not invalidate an expensive deterministic compile.

Publication is fail-closed and directory-atomic: a model is written to a
sibling staging directory, freshly reloaded, byte/state/topology verified, and
only then renamed into its final never-overwritten cache path.  The receipt is
self-hashed; every artifact other than the receipt is represented in one
canonical tree hash.  Loading verifies all metadata and files before invoking
the caller-supplied model loader, then verifies the reloaded model state again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COMPILER_IDENTITY_SCHEMA = "neural-forge-compiler-identity-v1"
COMPILED_CHILD_CACHE_SCHEMA = "neural-forge-compiled-child-cache-v1"
MODEL_STATE_FINGERPRINT_SCHEMA = "neural-forge-model-state-fingerprint-v1"
COMPILER_REPORT_FILENAME = "compiler_report.json"
CACHE_RECEIPT_FILENAME = "compiled_child_cache_receipt.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFETENSORS_SHARD = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_IDENTITY_BODY_KEYS = frozenset(
    {
        "schema",
        "parent_artifact_tree_sha256",
        "blank_target_artifact_tree_sha256",
        "tokenizer_sha256",
        "compiler_input_sha256s",
        "compiler_implementation_sha256s",
        "compiler_options",
        "runtime",
        "seed",
        "trainer_identity_included",
    }
)
_IDENTITY_KEYS = _IDENTITY_BODY_KEYS | {"compiler_identity_sha256"}
_RECEIPT_BODY_KEYS = frozenset(
    {
        "schema",
        "compiler_identity",
        "compiler_identity_sha256",
        "compiler_report",
        "model_state",
        "artifact_tree",
        "integrity_verified_immutable_by_contract",
        "trainer_identity_included",
        "fresh_reload_verified",
    }
)
_RECEIPT_KEYS = _RECEIPT_BODY_KEYS | {"receipt_sha256"}
_FORBIDDEN_ARTIFACT_NAME_PARTS = (
    "adapter",
    "factor",
    "optimizer",
    "parent",
    "scheduler",
    "sidecar",
    "trainer",
    "training_args",
)


class CompiledChildCacheError(RuntimeError):
    """Base failure for structural compiled-child caching."""


class CompilerIdentityError(CompiledChildCacheError):
    """A compiler identity is incomplete, malformed, or trainer-dependent."""


class CompiledChildCachePublicationError(CompiledChildCacheError):
    """An immutable cache could not be published atomically."""


class CompiledChildCacheIntegrityError(CompiledChildCacheError):
    """A cache receipt, artifact, report, or model state failed verification."""


@dataclass(frozen=True, slots=True, init=False)
class CompilerIdentity:
    """Canonical identity for structural compilation inputs and implementation.

    The stored identity has no field for a trainer hash, training schedule,
    optimizer, selection policy, or behavior threshold.  Suspicious nested
    keys are rejected as an additional guard against accidentally smuggling
    trainer identity into ``compiler_options`` or ``runtime``.
    """

    _body_json: bytes
    compiler_identity_sha256: str

    def __init__(
        self,
        *,
        parent_artifact_tree_sha256: str,
        blank_target_artifact_tree_sha256: str,
        tokenizer_sha256: str,
        compiler_input_sha256s: Mapping[str, str],
        compiler_implementation_sha256s: Mapping[str, str],
        compiler_options: Mapping[str, Any],
        runtime: Mapping[str, Any],
        seed: int,
    ) -> None:
        inputs = _named_sha256s(
            compiler_input_sha256s,
            "compiler_input_sha256s",
        )
        implementation = _named_sha256s(
            compiler_implementation_sha256s,
            "compiler_implementation_sha256s",
        )
        if not inputs:
            raise CompilerIdentityError(
                "compiler_input_sha256s must bind at least one compiler input."
            )
        if not implementation:
            raise CompilerIdentityError(
                "compiler_implementation_sha256s must bind at least one source file."
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise CompilerIdentityError("seed must be a nonnegative integer.")
        options = _json_value(compiler_options, "compiler_options")
        runtime_value = _json_value(runtime, "runtime")
        if not isinstance(options, dict) or not isinstance(runtime_value, dict):
            raise CompilerIdentityError(
                "compiler_options and runtime must be JSON objects."
            )
        if not runtime_value:
            raise CompilerIdentityError(
                "runtime must bind the compiler's relevant dtype/library settings."
            )
        _reject_trainer_identity(options, "compiler_options")
        _reject_trainer_identity(runtime_value, "runtime")
        body = {
            "schema": COMPILER_IDENTITY_SCHEMA,
            "parent_artifact_tree_sha256": _sha256(
                parent_artifact_tree_sha256,
                "parent_artifact_tree_sha256",
            ),
            "blank_target_artifact_tree_sha256": _sha256(
                blank_target_artifact_tree_sha256,
                "blank_target_artifact_tree_sha256",
            ),
            "tokenizer_sha256": _sha256(tokenizer_sha256, "tokenizer_sha256"),
            "compiler_input_sha256s": inputs,
            "compiler_implementation_sha256s": implementation,
            "compiler_options": options,
            "runtime": runtime_value,
            "seed": seed,
            "trainer_identity_included": False,
        }
        body_json = _canonical_json(body)
        object.__setattr__(self, "_body_json", body_json)
        object.__setattr__(
            self,
            "compiler_identity_sha256",
            hashlib.sha256(body_json).hexdigest(),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompilerIdentity:
        value = _json_value(payload, "compiler identity")
        if not isinstance(value, dict) or set(value) != _IDENTITY_KEYS:
            raise CompilerIdentityError(
                "Compiler identity does not contain exactly the v1 fields."
            )
        if value.get("schema") != COMPILER_IDENTITY_SCHEMA:
            raise CompilerIdentityError("Compiler identity schema is unsupported.")
        if value.get("trainer_identity_included") is not False:
            raise CompilerIdentityError(
                "Compiler identity must exclude all trainer identity."
            )
        result = cls(
            parent_artifact_tree_sha256=value["parent_artifact_tree_sha256"],
            blank_target_artifact_tree_sha256=value[
                "blank_target_artifact_tree_sha256"
            ],
            tokenizer_sha256=value["tokenizer_sha256"],
            compiler_input_sha256s=value["compiler_input_sha256s"],
            compiler_implementation_sha256s=value["compiler_implementation_sha256s"],
            compiler_options=value["compiler_options"],
            runtime=value["runtime"],
            seed=value["seed"],
        )
        claimed = _sha256(
            value["compiler_identity_sha256"],
            "compiler_identity_sha256",
        )
        if claimed != result.compiler_identity_sha256 or value != result.as_dict():
            raise CompilerIdentityError("Compiler identity hash does not verify.")
        return result

    def as_dict(self) -> dict[str, Any]:
        result = json.loads(self._body_json.decode("utf-8"))
        result["compiler_identity_sha256"] = self.compiler_identity_sha256
        return result


@dataclass(frozen=True, slots=True, init=False)
class CompiledChildCacheReceipt:
    """Verified immutable cache receipt."""

    _body_json: bytes
    receipt_sha256: str

    def __init__(
        self, body: Mapping[str, Any], receipt_sha256: str | None = None
    ) -> None:
        normalized = _json_value(body, "cache receipt")
        if not isinstance(normalized, dict) or set(normalized) != _RECEIPT_BODY_KEYS:
            raise CompiledChildCacheIntegrityError(
                "Cache receipt does not contain exactly the v1 body fields."
            )
        body_json = _canonical_json(normalized)
        actual = hashlib.sha256(body_json).hexdigest()
        if (
            receipt_sha256 is not None
            and _sha256(receipt_sha256, "receipt_sha256") != actual
        ):
            raise CompiledChildCacheIntegrityError(
                "Cache receipt self-hash does not verify."
            )
        object.__setattr__(self, "_body_json", body_json)
        object.__setattr__(self, "receipt_sha256", actual)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompiledChildCacheReceipt:
        value = _json_value(payload, "cache receipt")
        if not isinstance(value, dict) or set(value) != _RECEIPT_KEYS:
            raise CompiledChildCacheIntegrityError(
                "Cache receipt does not contain exactly the v1 fields."
            )
        claimed = value.pop("receipt_sha256")
        return cls(value, receipt_sha256=claimed)

    def as_dict(self) -> dict[str, Any]:
        result = json.loads(self._body_json.decode("utf-8"))
        result["receipt_sha256"] = self.receipt_sha256
        return result


@dataclass(frozen=True, slots=True)
class VerifiedCompiledChildCache:
    """Metadata verified before any model loader is invoked."""

    path: Path
    compiler_identity: CompilerIdentity
    compiler_report: Mapping[str, Any]
    receipt: CompiledChildCacheReceipt


@dataclass(frozen=True, slots=True)
class LoadedCompiledChild:
    """Freshly loaded model plus its verified compiler evidence."""

    model: Any
    path: Path
    compiler_identity: CompilerIdentity
    compiler_report: Mapping[str, Any]
    receipt: CompiledChildCacheReceipt


def model_state_fingerprint(model: Any) -> dict[str, Any]:
    """Hash a model's complete state bytes and parameter/buffer topology."""

    torch = _torch()
    state_method = getattr(model, "state_dict", None)
    if not callable(state_method):
        raise CompiledChildCacheIntegrityError("model must expose state_dict().")
    state = state_method()
    if not isinstance(state, Mapping) or not state:
        raise CompiledChildCacheIntegrityError(
            "model.state_dict() must be a nonempty mapping."
        )

    digest = hashlib.sha256()
    key_records: list[dict[str, Any]] = []
    logical_elements = 0
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not name:
            raise CompiledChildCacheIntegrityError(
                "Model state keys must be nonempty strings."
            )
        if not isinstance(tensor, torch.Tensor):
            raise CompiledChildCacheIntegrityError(
                f"Model state {name!r} is not a tensor."
            )
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise CompiledChildCacheIntegrityError(
                f"Model state {name!r} must be materialized and strided."
            )
        value = tensor.detach().to(device="cpu").contiguous()
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise CompiledChildCacheIntegrityError(
                f"Model state {name!r} contains NaN or infinity."
            )
        raw = value.view(torch.uint8).numpy()
        tensor_digest = hashlib.sha256()
        tensor_digest.update(memoryview(raw))
        header = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "tensor_sha256": tensor_digest.hexdigest(),
        }
        encoded_header = _canonical_json(header)
        digest.update(len(encoded_header).to_bytes(8, "big"))
        digest.update(encoded_header)
        key_records.append(header)
        logical_elements += int(value.numel())

    parameter_topology, unique_parameters, logical_parameters = _named_topology(
        model,
        "named_parameters",
        include_requires_grad=True,
    )
    buffer_topology, _unique_buffers, _logical_buffers = _named_topology(
        model,
        "named_buffers",
        include_requires_grad=False,
    )
    if unique_parameters <= 0:
        raise CompiledChildCacheIntegrityError(
            "A compiled child must contain at least one learned parameter."
        )
    keys = [record["name"] for record in key_records]
    from neural_forge.hf_parent_compiler import (
        fingerprint_hf_compiler_model_state,
    )

    return {
        "schema": MODEL_STATE_FINGERPRINT_SCHEMA,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "state_dict_sha256": digest.hexdigest(),
        "compiler_state_sha256": fingerprint_hf_compiler_model_state(model),
        "state_key_sha256": _canonical_sha256(keys),
        "state_manifest_sha256": _canonical_sha256(key_records),
        "state_key_count": len(keys),
        "logical_state_elements": logical_elements,
        "unique_parameter_count": unique_parameters,
        "logical_parameter_count": logical_parameters,
        "parameter_topology_sha256": _canonical_sha256(parameter_topology),
        "buffer_topology_sha256": _canonical_sha256(buffer_topology),
        "all_state_tensors_finite": True,
    }


def publish_compiled_child_cache(
    cache_dir: str | os.PathLike[str],
    *,
    model: Any,
    compiler_report: Mapping[str, Any] | Any,
    compiler_identity: CompilerIdentity,
    model_saver: Callable[[Any, Path], None] | None = None,
    model_loader: Callable[[Path], Any] | None = None,
) -> CompiledChildCacheReceipt:
    """Atomically publish one accepted compiled child into a new cache path.

    ``model_saver`` receives ``(model, staging_directory)``.  When omitted, the
    model's ``save_pretrained(..., safe_serialization=True)`` method is used.
    ``model_loader`` receives the staging directory and must freshly reconstruct
    the model.  When omitted, ``type(model).from_pretrained`` is used locally.
    """

    if not isinstance(compiler_identity, CompilerIdentity):
        raise CompilerIdentityError(
            "compiler_identity must be a validated CompilerIdentity."
        )
    report = _compiler_report(compiler_report)
    live_state = model_state_fingerprint(model)
    _bind_report_to_state(report, live_state)
    destination = _publication_destination(cache_dir, compiler_identity)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CompiledChildCachePublicationError(
            f"Immutable cache path already exists: {destination}"
        )
    saver = _default_model_saver if model_saver is None else model_saver
    loader = _default_model_loader(model) if model_loader is None else model_loader
    if not callable(saver) or not callable(loader):
        raise CompiledChildCachePublicationError(
            "model_saver and model_loader must be callable."
        )

    lock_path = destination.parent / f".{destination.name}.compile-cache.lock"
    lock_fd: int | None = None
    lock_identity: tuple[int, int] | None = None
    staging: Path | None = None
    try:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise CompiledChildCachePublicationError(
                f"Compiled-child cache publication lock already exists: {lock_path}"
            ) from error
        lock_stat = os.fstat(lock_fd)
        lock_identity = (int(lock_stat.st_dev), int(lock_stat.st_ino))
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        if destination.exists():
            raise CompiledChildCachePublicationError(
                f"Immutable cache path already exists: {destination}"
            )
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".staging",
                dir=destination.parent,
            )
        )
        saver(model, staging)
        _assert_reserved_paths_absent(staging)
        _validate_hf_model_artifacts(
            staging,
            expected_state_keys=tuple(sorted(model.state_dict())),
        )
        _write_json_exclusive(staging / COMPILER_REPORT_FILENAME, report)
        _fsync_tree(staging)
        tree_before_reload = _artifact_tree(staging)
        reloaded = loader(staging)
        reloaded_state = model_state_fingerprint(reloaded)
        if reloaded_state != live_state:
            raise CompiledChildCachePublicationError(
                "Freshly reloaded compiled child does not exactly match live state."
            )
        del reloaded
        tree_after_reload = _artifact_tree(staging)
        if tree_after_reload != tree_before_reload:
            raise CompiledChildCachePublicationError(
                "Fresh model reload mutated the staged cache artifacts."
            )
        report_bytes = (staging / COMPILER_REPORT_FILENAME).read_bytes()
        report_record = {
            "path": COMPILER_REPORT_FILENAME,
            "canonical_sha256": _canonical_sha256(report),
            "file_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "size_bytes": len(report_bytes),
        }
        receipt_body = {
            "schema": COMPILED_CHILD_CACHE_SCHEMA,
            "compiler_identity": compiler_identity.as_dict(),
            "compiler_identity_sha256": (compiler_identity.compiler_identity_sha256),
            "compiler_report": report_record,
            "model_state": live_state,
            "artifact_tree": tree_after_reload,
            "integrity_verified_immutable_by_contract": True,
            "trainer_identity_included": False,
            "fresh_reload_verified": True,
        }
        receipt = CompiledChildCacheReceipt(receipt_body)
        _write_json_exclusive(staging / CACHE_RECEIPT_FILENAME, receipt.as_dict())
        _fsync_tree(staging)
        verified = _verify_cache_metadata(
            staging,
            expected_identity=compiler_identity,
            expected_receipt_sha256=receipt.receipt_sha256,
            expected_compiler_report_sha256=report_record["canonical_sha256"],
            expected_state_dict_sha256=live_state["state_dict_sha256"],
        )
        if verified.receipt.as_dict() != receipt.as_dict():
            raise CompiledChildCachePublicationError(
                "Staged cache receipt changed during verification."
            )
        if destination.exists():
            raise CompiledChildCachePublicationError(
                f"Immutable cache path already exists: {destination}"
            )
        os.rename(staging, destination)
        staging = None
        _fsync_directory(destination.parent)
        return receipt
    except CompiledChildCacheError:
        raise
    except BaseException as error:
        raise CompiledChildCachePublicationError(
            "Compiled-child cache publication failed before atomic promotion."
        ) from error
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_identity is not None:
            try:
                current = lock_path.stat()
            except FileNotFoundError:
                current = None
            if (
                current is not None
                and (
                    int(current.st_dev),
                    int(current.st_ino),
                )
                == lock_identity
            ):
                lock_path.unlink()


def verify_compiled_child_cache(
    cache_dir: str | os.PathLike[str],
    *,
    expected_identity: CompilerIdentity,
    expected_receipt_sha256: str | None = None,
    expected_compiler_report_sha256: str | None = None,
    expected_state_dict_sha256: str | None = None,
) -> VerifiedCompiledChildCache:
    """Verify receipt, identity, report, and every artifact without loading a model."""

    return _verify_cache_metadata(
        cache_dir,
        expected_identity=expected_identity,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_compiler_report_sha256=expected_compiler_report_sha256,
        expected_state_dict_sha256=expected_state_dict_sha256,
    )


def load_compiled_child_cache(
    cache_dir: str | os.PathLike[str],
    *,
    expected_identity: CompilerIdentity,
    model_loader: Callable[[Path], Any],
    expected_receipt_sha256: str | None = None,
    expected_compiler_report_sha256: str | None = None,
    expected_state_dict_sha256: str | None = None,
) -> LoadedCompiledChild:
    """Verify metadata first, freshly load one model, then verify its exact state."""

    if not callable(model_loader):
        raise CompiledChildCacheIntegrityError("model_loader must be callable.")
    verified = _verify_cache_metadata(
        cache_dir,
        expected_identity=expected_identity,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_compiler_report_sha256=expected_compiler_report_sha256,
        expected_state_dict_sha256=expected_state_dict_sha256,
    )
    model = model_loader(verified.path)
    actual_state = model_state_fingerprint(model)
    expected_state = verified.receipt.as_dict()["model_state"]
    if actual_state != expected_state:
        raise CompiledChildCacheIntegrityError(
            "Freshly loaded compiled child does not match the cached state receipt."
        )
    # Detect loaders that unexpectedly write into their source directory.
    _verify_cache_metadata(
        verified.path,
        expected_identity=expected_identity,
        expected_receipt_sha256=verified.receipt.receipt_sha256,
        expected_compiler_report_sha256=verified.receipt.as_dict()["compiler_report"][
            "canonical_sha256"
        ],
        expected_state_dict_sha256=expected_state["state_dict_sha256"],
    )
    return LoadedCompiledChild(
        model=model,
        path=verified.path,
        compiler_identity=verified.compiler_identity,
        compiler_report=verified.compiler_report,
        receipt=verified.receipt,
    )


def _verify_cache_metadata(
    cache_dir: str | os.PathLike[str],
    *,
    expected_identity: CompilerIdentity,
    expected_receipt_sha256: str | None,
    expected_compiler_report_sha256: str | None,
    expected_state_dict_sha256: str | None,
) -> VerifiedCompiledChildCache:
    if not isinstance(expected_identity, CompilerIdentity):
        raise CompilerIdentityError(
            "expected_identity must be a validated CompilerIdentity."
        )
    source = Path(cache_dir).expanduser()
    if source.is_symlink():
        raise CompiledChildCacheIntegrityError("Cache root cannot be a symlink.")
    try:
        root = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise CompiledChildCacheIntegrityError(
            "Compiled-child cache is absent."
        ) from error
    if not root.is_dir():
        raise CompiledChildCacheIntegrityError(
            "Compiled-child cache path is not a directory."
        )
    receipt_path = root / CACHE_RECEIPT_FILENAME
    report_path = root / COMPILER_REPORT_FILENAME
    receipt_value = _read_json_object(receipt_path, "cache receipt")
    receipt = CompiledChildCacheReceipt.from_dict(receipt_value)
    receipt_payload = receipt.as_dict()
    if receipt_path.read_bytes() != _canonical_json(receipt_payload) + b"\n":
        raise CompiledChildCacheIntegrityError(
            "Cache receipt file is not in its unique canonical byte encoding."
        )
    if receipt_payload.get("schema") != COMPILED_CHILD_CACHE_SCHEMA:
        raise CompiledChildCacheIntegrityError("Cache receipt schema is unsupported.")
    for name in ("integrity_verified_immutable_by_contract", "fresh_reload_verified"):
        if receipt_payload.get(name) is not True:
            raise CompiledChildCacheIntegrityError(
                f"Cache receipt must prove {name}=true."
            )
    if receipt_payload.get("trainer_identity_included") is not False:
        raise CompiledChildCacheIntegrityError(
            "Cache receipt illegally includes trainer identity."
        )
    if expected_receipt_sha256 is not None and receipt.receipt_sha256 != _sha256(
        expected_receipt_sha256,
        "expected_receipt_sha256",
    ):
        raise CompiledChildCacheIntegrityError("Cache receipt hash is not expected.")
    embedded_identity = CompilerIdentity.from_dict(receipt_payload["compiler_identity"])
    if (
        receipt_payload.get("compiler_identity_sha256")
        != embedded_identity.compiler_identity_sha256
        or embedded_identity.as_dict() != expected_identity.as_dict()
    ):
        raise CompiledChildCacheIntegrityError(
            "Cache compiler identity does not match the requested compile."
        )
    report = _compiler_report(_read_json_object(report_path, "compiler report"))
    report_record = receipt_payload.get("compiler_report")
    if not isinstance(report_record, dict) or set(report_record) != {
        "path",
        "canonical_sha256",
        "file_sha256",
        "size_bytes",
    }:
        raise CompiledChildCacheIntegrityError("Compiler report receipt is malformed.")
    if report_record.get("path") != COMPILER_REPORT_FILENAME:
        raise CompiledChildCacheIntegrityError(
            "Compiler report receipt path is invalid."
        )
    report_bytes = report_path.read_bytes()
    actual_report_sha = _canonical_sha256(report)
    if (
        report_record.get("canonical_sha256") != actual_report_sha
        or report_record.get("file_sha256") != hashlib.sha256(report_bytes).hexdigest()
        or report_record.get("size_bytes") != len(report_bytes)
    ):
        raise CompiledChildCacheIntegrityError(
            "Compiler report content or file hash does not verify."
        )
    if expected_compiler_report_sha256 is not None and actual_report_sha != _sha256(
        expected_compiler_report_sha256,
        "expected_compiler_report_sha256",
    ):
        raise CompiledChildCacheIntegrityError("Compiler report hash is not expected.")
    model_state = receipt_payload.get("model_state")
    _validate_model_state_record(model_state)
    _bind_report_to_state(report, model_state)
    _validate_hf_model_artifacts(
        root,
        expected_state_key_sha256=model_state["state_key_sha256"],
    )
    if expected_state_dict_sha256 is not None and model_state[
        "state_dict_sha256"
    ] != _sha256(expected_state_dict_sha256, "expected_state_dict_sha256"):
        raise CompiledChildCacheIntegrityError(
            "Compiled-child state hash is not expected."
        )
    actual_tree = _artifact_tree(root)
    if receipt_payload.get("artifact_tree") != actual_tree:
        raise CompiledChildCacheIntegrityError(
            "Compiled-child artifact tree contains missing, changed, or extra files."
        )
    return VerifiedCompiledChildCache(
        path=root,
        compiler_identity=embedded_identity,
        compiler_report=report,
        receipt=receipt,
    )


def _compiler_report(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        method = getattr(value, "as_dict", None)
        if not callable(method):
            raise CompiledChildCacheIntegrityError(
                "compiler_report must be a mapping or expose as_dict()."
            )
        value = method()
    report = _json_value(value, "compiler_report")
    if not isinstance(report, dict):
        raise CompiledChildCacheIntegrityError("compiler_report must be a JSON object.")
    if report.get("bootstrap_accepted") is not True:
        raise CompiledChildCacheIntegrityError(
            "Only an accepted structural bootstrap can be cached."
        )
    if report.get("target_budget_exact") is not True:
        raise CompiledChildCacheIntegrityError(
            "Only an exact-budget structural child can be cached."
        )
    unique = report.get("target_unique_parameter_count")
    budget = report.get("target_parameter_budget")
    if (
        isinstance(unique, bool)
        or not isinstance(unique, int)
        or unique <= 0
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget != unique
    ):
        raise CompiledChildCacheIntegrityError(
            "Compiler report target parameter count/budget is invalid."
        )
    _sha256(report.get("target_state_sha256_after"), "target_state_sha256_after")
    return report


def _bind_report_to_state(report: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    if report.get("target_state_sha256_after") != state.get("compiler_state_sha256"):
        raise CompiledChildCacheIntegrityError(
            "Compiler report state hash does not match the serialized child."
        )
    if report.get("target_unique_parameter_count") != state.get(
        "unique_parameter_count"
    ):
        raise CompiledChildCacheIntegrityError(
            "Compiler report parameter count does not match the serialized child."
        )
    logical = report.get("target_logical_parameter_count")
    if logical is not None and logical != state.get("logical_parameter_count"):
        raise CompiledChildCacheIntegrityError(
            "Compiler report logical parameter count does not match the child."
        )


def _validate_model_state_record(value: Any) -> None:
    expected = {
        "schema",
        "model_class",
        "state_dict_sha256",
        "compiler_state_sha256",
        "state_key_sha256",
        "state_manifest_sha256",
        "state_key_count",
        "logical_state_elements",
        "unique_parameter_count",
        "logical_parameter_count",
        "parameter_topology_sha256",
        "buffer_topology_sha256",
        "all_state_tensors_finite",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise CompiledChildCacheIntegrityError("Model-state receipt is malformed.")
    if value.get("schema") != MODEL_STATE_FINGERPRINT_SCHEMA:
        raise CompiledChildCacheIntegrityError(
            "Model-state fingerprint schema is unsupported."
        )
    for name in (
        "state_dict_sha256",
        "compiler_state_sha256",
        "state_key_sha256",
        "state_manifest_sha256",
        "parameter_topology_sha256",
        "buffer_topology_sha256",
    ):
        _sha256(value.get(name), name)
    for name in (
        "state_key_count",
        "logical_state_elements",
        "unique_parameter_count",
        "logical_parameter_count",
    ):
        number = value.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise CompiledChildCacheIntegrityError(
                f"Model-state receipt {name} is invalid."
            )
    if value.get("all_state_tensors_finite") is not True:
        raise CompiledChildCacheIntegrityError(
            "Model-state receipt does not prove finite tensors."
        )
    if not isinstance(value.get("model_class"), str) or not value["model_class"]:
        raise CompiledChildCacheIntegrityError(
            "Model-state receipt model_class is invalid."
        )


def _named_topology(
    model: Any,
    method_name: str,
    *,
    include_requires_grad: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    torch = _torch()
    method = getattr(model, method_name, None)
    if not callable(method):
        raise CompiledChildCacheIntegrityError(f"model must expose {method_name}().")
    try:
        entries = tuple(method(recurse=True, remove_duplicate=False))
    except TypeError:
        entries = tuple(method(recurse=True))
    aliases: dict[int, int] = {}
    records = []
    logical = 0
    for name, tensor in entries:
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise CompiledChildCacheIntegrityError(
                f"{method_name} returned an invalid entry."
            )
        alias = aliases.setdefault(id(tensor), len(aliases))
        record = {
            "name": name,
            "alias": alias,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        if include_requires_grad:
            record["requires_grad"] = bool(tensor.requires_grad)
        records.append(record)
        logical += int(tensor.numel())
    unique = {id(tensor): tensor for _name, tensor in entries}
    return records, sum(int(tensor.numel()) for tensor in unique.values()), logical


def _artifact_tree(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    directories: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise CompiledChildCacheIntegrityError(
                "Compiled-child cache cannot contain symbolic links."
            )
        if path.is_dir():
            directories.append(path.relative_to(root).as_posix())
            continue
        if not path.is_file() or path == root / CACHE_RECEIPT_FILENAME:
            continue
        relative = path.relative_to(root).as_posix()
        metadata = path.stat()
        if metadata.st_nlink != 1:
            raise CompiledChildCacheIntegrityError(
                f"Compiled-child cache file {relative!r} is hard-linked and mutable "
                "outside the immutable tree."
            )
        if relative != COMPILER_REPORT_FILENAME and any(
            fragment in path.name.lower() for fragment in _FORBIDDEN_ARTIFACT_NAME_PARTS
        ):
            raise CompiledChildCacheIntegrityError(
                "Compiled-child cache contains forbidden sidecar-like file "
                f"{relative!r}."
            )
        records.append(
            {
                "path": relative,
                "size_bytes": metadata.st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not records or not any(
        value["path"] == COMPILER_REPORT_FILENAME for value in records
    ):
        raise CompiledChildCacheIntegrityError(
            "Compiled-child cache has no model artifacts or compiler report."
        )
    if len(records) == 1:
        raise CompiledChildCacheIntegrityError(
            "Compiled-child cache contains a compiler report but no model files."
        )
    tree_payload = {"directories": directories, "files": records}
    return {
        "directories": directories,
        "files": records,
        "file_count": len(records),
        "directory_count": len(directories),
        "total_bytes": sum(value["size_bytes"] for value in records),
        "tree_sha256": _canonical_sha256(tree_payload),
        "receipt_excluded_from_tree": True,
    }


def _validate_hf_model_artifacts(
    root: Path,
    *,
    expected_state_keys: Sequence[str] | None = None,
    expected_state_key_sha256: str | None = None,
) -> None:
    """Reject every byte not required by a standard safe HF model checkpoint."""

    if (expected_state_keys is None) == (expected_state_key_sha256 is None):
        raise CompiledChildCacheIntegrityError(
            "Exactly one expected model-state key binding is required."
        )
    allowed_json = {"config.json", "generation_config.json"}
    reserved = {COMPILER_REPORT_FILENAME, CACHE_RECEIPT_FILENAME}
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CompiledChildCacheIntegrityError(
                "Compiled-child model artifacts cannot contain symbolic links."
            )
        if path.is_dir():
            if path != root:
                raise CompiledChildCacheIntegrityError(
                    "Compiled-child model artifacts must be flat."
                )
            continue
        if not path.is_file() or path.parent != root:
            raise CompiledChildCacheIntegrityError(
                "Compiled-child model artifacts contain an unsafe entry."
            )
        if path.stat().st_nlink != 1:
            raise CompiledChildCacheIntegrityError(
                f"Compiled-child model artifact {path.name!r} is hard-linked."
            )
        name = path.name
        if name in reserved:
            continue
        if (
            name not in allowed_json
            and name not in {"model.safetensors", "model.safetensors.index.json"}
            and _SAFETENSORS_SHARD.fullmatch(name) is None
        ):
            raise CompiledChildCacheIntegrityError(
                f"Compiled-child cache contains unapproved model artifact {name!r}."
            )
        files[name] = path
    if "config.json" not in files:
        raise CompiledChildCacheIntegrityError(
            "Compiled-child cache omits config.json."
        )
    for name in allowed_json & set(files):
        _read_json_object(files[name], name)
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - production dependency
        raise CompiledChildCacheIntegrityError(
            "safetensors is required to authenticate compiled-child shards."
        ) from error

    index_path = files.get("model.safetensors.index.json")
    unsharded = files.get("model.safetensors")
    shard_names = {
        name for name in files if _SAFETENSORS_SHARD.fullmatch(name) is not None
    }
    actual_keys: list[str] = []
    try:
        if index_path is None:
            if unsharded is None or shard_names:
                raise CompiledChildCacheIntegrityError(
                    "Unsharded cache must contain exactly model.safetensors."
                )
            with safe_open(unsharded, framework="pt", device="cpu") as handle:
                actual_keys.extend(str(value) for value in handle.keys())
        else:
            if unsharded is not None:
                raise CompiledChildCacheIntegrityError(
                    "Sharded cache cannot also contain model.safetensors."
                )
            index = _read_json_object(index_path, "safetensors index")
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise CompiledChildCacheIntegrityError(
                    "Safetensors index has no weight_map."
                )
            if any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or _SAFETENSORS_SHARD.fullmatch(value) is None
                for key, value in weight_map.items()
            ):
                raise CompiledChildCacheIntegrityError(
                    "Safetensors index contains an invalid key or shard name."
                )
            referenced = set(weight_map.values())
            if referenced != shard_names:
                raise CompiledChildCacheIntegrityError(
                    "Safetensors shards are missing or unreferenced by the index."
                )
            for shard_name in sorted(referenced):
                with safe_open(
                    files[shard_name], framework="pt", device="cpu"
                ) as handle:
                    shard_keys = {str(value) for value in handle.keys()}
                mapped = {
                    str(key) for key, value in weight_map.items() if value == shard_name
                }
                if shard_keys != mapped:
                    raise CompiledChildCacheIntegrityError(
                        f"Safetensors shard {shard_name!r} disagrees with its index."
                    )
                actual_keys.extend(sorted(shard_keys))
    except CompiledChildCacheError:
        raise
    except Exception as error:
        raise CompiledChildCacheIntegrityError(
            "Cannot authenticate compiled-child safetensors content."
        ) from error
    if len(set(actual_keys)) != len(actual_keys):
        raise CompiledChildCacheIntegrityError(
            "Compiled-child safetensors contain duplicate state keys."
        )
    ordered_keys = tuple(sorted(actual_keys))
    if expected_state_keys is not None and ordered_keys != tuple(
        sorted(str(value) for value in expected_state_keys)
    ):
        raise CompiledChildCacheIntegrityError(
            "Compiled-child safetensors do not exactly cover model state keys."
        )
    if expected_state_key_sha256 is not None and _canonical_sha256(
        list(ordered_keys)
    ) != _sha256(expected_state_key_sha256, "expected_state_key_sha256"):
        raise CompiledChildCacheIntegrityError(
            "Compiled-child safetensors key coverage differs from the receipt."
        )


def _assert_reserved_paths_absent(root: Path) -> None:
    for name in (COMPILER_REPORT_FILENAME, CACHE_RECEIPT_FILENAME):
        if (root / name).exists():
            raise CompiledChildCachePublicationError(
                f"Model saver wrote reserved cache path {name!r}."
            )


def _default_model_saver(model: Any, destination: Path) -> None:
    method = getattr(model, "save_pretrained", None)
    if not callable(method):
        raise CompiledChildCachePublicationError(
            "model has no save_pretrained(); supply model_saver."
        )
    method(destination, safe_serialization=True)


def _default_model_loader(model: Any) -> Callable[[Path], Any]:
    method = getattr(type(model), "from_pretrained", None)
    if not callable(method):
        raise CompiledChildCachePublicationError(
            "model type has no from_pretrained(); supply model_loader."
        )

    def load(path: Path) -> Any:
        return method(path, local_files_only=True)

    return load


def _publication_destination(
    value: str | os.PathLike[str],
    compiler_identity: CompilerIdentity,
) -> Path:
    raw = Path(value).expanduser()
    if raw.exists() and raw.is_symlink():
        raise CompiledChildCachePublicationError(
            "Cache publication path cannot be a symlink."
        )
    resolved = raw.resolve(strict=False)
    if resolved.name != compiler_identity.compiler_identity_sha256:
        raise CompiledChildCachePublicationError(
            "Compiled-child cache destination must be named by its compiler "
            "identity SHA-256."
        )
    return resolved


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise CompiledChildCachePublicationError(
            f"Refusing to overwrite cache metadata {path.name!r}."
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompiledChildCacheIntegrityError(f"{label} file is absent or unsafe.")
    if path.stat().st_nlink != 1:
        raise CompiledChildCacheIntegrityError(f"{label} file is hard-linked.")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise CompiledChildCacheIntegrityError(f"{label} file size is invalid.")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CompiledChildCacheIntegrityError(f"{label} is invalid JSON.") from error
    if not isinstance(value, dict):
        raise CompiledChildCacheIntegrityError(f"{label} must be a JSON object.")
    return value


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")


def _json_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CompiledChildCacheIntegrityError(
                f"{name} contains a non-finite float."
            )
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CompiledChildCacheIntegrityError(
                    f"{name} contains an invalid JSON object key."
                )
            result[key] = _json_value(item, f"{name}.{key}")
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{name}[]") for item in value]
    raise CompiledChildCacheIntegrityError(
        f"{name} contains a non-JSON value of type {type(value).__name__}."
    )


def _reject_trainer_identity(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
            forbidden = (
                "trainer" in normalized
                or "optimizer" in normalized
                or normalized == "training"
                or normalized.startswith("training_")
                or normalized.endswith("_training")
                or "training_schedule" in normalized
                or "behavior_threshold" in normalized
                or "candidate_threshold" in normalized
                or "sealed_threshold" in normalized
                or normalized in {"selection_policy", "evaluation_contract_sha256"}
            )
            if forbidden:
                raise CompilerIdentityError(
                    f"{path}.{key} is trainer-dependent and cannot enter compiler identity."
                )
            _reject_trainer_identity(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_trainer_identity(item, f"{path}[{index}]")
    elif isinstance(value, str):
        normalized_value = value.strip().lower().replace("-", "_")
        if any(
            fragment in normalized_value
            for fragment in (
                "behavior_distillation",
                "optimizer",
                "target_native_behavior",
                "trainer",
            )
        ):
            raise CompilerIdentityError(
                f"{path} contains trainer-dependent identity text."
            )


def _named_sha256s(value: Mapping[str, str], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CompilerIdentityError(f"{name} must be a mapping.")
    result: dict[str, str] = {}
    for raw_name, digest in value.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise CompilerIdentityError(f"{name} contains an invalid name.")
        entry_name = raw_name.strip()
        _reject_trainer_identity({entry_name: None}, name)
        result[entry_name] = _sha256(digest, f"{name}.{entry_name}")
    return {key: result[key] for key in sorted(result)}


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise CompiledChildCacheIntegrityError(f"{name} must be a SHA-256 string.")
    normalized = value.strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise CompiledChildCacheIntegrityError(
            f"{name} must be 64 lowercase hex digits."
        )
    return normalized


def _canonical_json(value: Any) -> bytes:
    normalized = _json_value(value, "canonical JSON")
    try:
        return json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:  # pragma: no cover - normalized above
        raise CompiledChildCacheIntegrityError(
            "Value cannot be encoded as canonical JSON."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise CompiledChildCachePublicationError(
                "Staged cache cannot contain symbolic links."
            )
        # Windows requires a writable descriptor for ``os.fsync``.  These are
        # still-private staging files, so opening read/write is safe and does
        # not alter their bytes.
        with path.open("rb+") as handle:
            os.fsync(handle.fileno())
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Windows and some network filesystems do not support directory fsync.
            pass
    finally:
        os.close(descriptor)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise CompiledChildCacheError(
            "PyTorch is required for compiled-child state verification."
        ) from error
    return torch


__all__ = [
    "CACHE_RECEIPT_FILENAME",
    "COMPILED_CHILD_CACHE_SCHEMA",
    "COMPILER_IDENTITY_SCHEMA",
    "COMPILER_REPORT_FILENAME",
    "MODEL_STATE_FINGERPRINT_SCHEMA",
    "CompiledChildCacheError",
    "CompiledChildCacheIntegrityError",
    "CompiledChildCachePublicationError",
    "CompiledChildCacheReceipt",
    "CompilerIdentity",
    "CompilerIdentityError",
    "LoadedCompiledChild",
    "VerifiedCompiledChildCache",
    "load_compiled_child_cache",
    "model_state_fingerprint",
    "publish_compiled_child_cache",
    "verify_compiled_child_cache",
]
