"""Immutable, content-addressed catalogs for dynamic fold evidence.

The store persists only canonical JSON metadata.  It never stores tensors,
weights, optimizer state, training state, or a partially compiled child.
Callers bind one catalog to an authenticated source/session/provider/codec
identity, then atomically publish the complete tensor-free catalog under that
binding hash.  Existing paths are never repaired or overwritten: corruption
and identity disagreement fail closed.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256


DYNAMIC_EVIDENCE_CATALOG_BINDING_SCHEMA = (
    "neural-forge-dynamic-evidence-catalog-binding-v1"
)
DYNAMIC_EVIDENCE_CATALOG_DOCUMENT_SCHEMA = (
    "neural-forge-dynamic-evidence-catalog-document-v1"
)
DYNAMIC_EVIDENCE_CATALOG_ARTIFACT_SCHEMA = (
    "neural-forge-dynamic-evidence-catalog-artifact-v1"
)

CATALOG_FILENAME = "catalog.json"
MANIFEST_FILENAME = "manifest.json"

_ARTIFACT_FILENAMES = frozenset({CATALOG_FILENAME, MANIFEST_FILENAME})
_MAX_JSON_BYTES = 64 * 1024 * 1024
_STAGE_PREFIX = ".dynamic-evidence-catalog-stage-"


class DynamicEvidenceCatalogArtifactError(RuntimeError):
    """Base failure for a dynamic-evidence catalog artifact."""


class DynamicEvidenceCatalogArtifactIntegrityError(
    DynamicEvidenceCatalogArtifactError
):
    """A binding, catalog, path, or manifest failed verification."""


class DynamicEvidenceCatalogArtifactPublicationError(
    DynamicEvidenceCatalogArtifactError
):
    """A catalog could not be published atomically and immutably."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:/-"
            for character in value
        )
    ):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} must be a nonempty stable identifier."
        )
    return value


def _normalized_json(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} must be canonical JSON."
        ) from exc


def _canonical_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction(path)) if callable(is_junction) else False


def _regular_file(path: Path, *, name: str) -> os.stat_result:
    try:
        result = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"Cannot stat {name}."
        ) from exc
    if not stat.S_ISREG(result.st_mode) or _is_link_or_junction(path):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} must be a real regular file."
        )
    return result


def _real_directory(path: Path, *, name: str) -> os.stat_result:
    try:
        result = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"Cannot stat {name}."
        ) from exc
    if not stat.S_ISDIR(result.st_mode) or _is_link_or_junction(path):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} must be a real directory."
        )
    return result


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return int(value.st_dev), int(value.st_ino)


def _read_canonical_json(path: Path, *, name: str) -> dict[str, Any]:
    before = _regular_file(path, name=name)
    size = int(before.st_size)
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} has an invalid byte size."
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"Cannot read {name}."
        ) from exc
    after = _regular_file(path, name=name)
    if _stat_identity(after) != _stat_identity(before) or after.st_size != size:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} changed while it was read."
        )
    try:
        value = json.loads(raw.decode("ascii"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} is not valid canonical JSON."
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_file_bytes(value):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            f"{name} is not a canonical JSON object."
        )
    return value


def _write_new_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical_file_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DynamicEvidenceCatalogArtifactPublicationError(
            f"Could not write immutable catalog file {path.name}."
        ) from exc


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""

    if os.name == "posix":
        try:
            library = ctypes.CDLL(None, use_errno=True)
            renameat2 = library.renameat2
        except (AttributeError, OSError):
            renameat2 = None
        if renameat2 is None:
            raise DynamicEvidenceCatalogArtifactPublicationError(
                "This POSIX runtime lacks atomic no-replace directory rename."
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number))
        raise OSError(error_number, os.strerror(error_number))
    # Windows directory rename is atomic and refuses an existing destination.
    os.rename(source, destination)


@dataclass(frozen=True, slots=True, init=False)
class DynamicEvidenceCatalogBinding:
    """Exact authenticated identity under which one catalog is reusable."""

    catalog_kind: str
    source_sha256: str
    provider_sha256: str
    authenticated_input_sha256: str
    codec_sha256: str
    binding_sha256: str
    _body_json: bytes

    def __init__(
        self,
        *,
        catalog_kind: str,
        source_sha256: str,
        provider_sha256: str,
        authenticated_input_sha256: str,
        codec_sha256: str,
        bindings: Mapping[str, Any],
    ) -> None:
        normalized_bindings = _normalized_json(bindings, "bindings")
        if not isinstance(normalized_bindings, dict):
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "bindings must be a JSON object."
            )
        body = {
            "schema": DYNAMIC_EVIDENCE_CATALOG_BINDING_SCHEMA,
            "catalog_kind": _identifier(catalog_kind, "catalog_kind"),
            "source_sha256": _digest(source_sha256, "source_sha256"),
            "provider_sha256": _digest(provider_sha256, "provider_sha256"),
            "authenticated_input_sha256": _digest(
                authenticated_input_sha256,
                "authenticated_input_sha256",
            ),
            "codec_sha256": _digest(codec_sha256, "codec_sha256"),
            "bindings": normalized_bindings,
            "tensor_payloads_resident": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
        }
        body_json = canonical_json_bytes(body)
        object.__setattr__(self, "catalog_kind", body["catalog_kind"])
        object.__setattr__(self, "source_sha256", body["source_sha256"])
        object.__setattr__(self, "provider_sha256", body["provider_sha256"])
        object.__setattr__(
            self,
            "authenticated_input_sha256",
            body["authenticated_input_sha256"],
        )
        object.__setattr__(self, "codec_sha256", body["codec_sha256"])
        object.__setattr__(self, "_body_json", body_json)
        object.__setattr__(
            self,
            "binding_sha256",
            hashlib.sha256(body_json).hexdigest(),
        )

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> DynamicEvidenceCatalogBinding:
        normalized = _normalized_json(value, "binding")
        expected = {
            "schema",
            "catalog_kind",
            "source_sha256",
            "provider_sha256",
            "authenticated_input_sha256",
            "codec_sha256",
            "bindings",
            "tensor_payloads_resident",
            "parent_forward_calls",
            "training_steps",
            "binding_sha256",
        }
        if not isinstance(normalized, dict) or set(normalized) != expected:
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "Catalog binding does not contain exactly the v1 fields."
            )
        if (
            normalized["schema"] != DYNAMIC_EVIDENCE_CATALOG_BINDING_SCHEMA
            or normalized["tensor_payloads_resident"] is not False
            or normalized["parent_forward_calls"] != 0
            or normalized["training_steps"] != 0
        ):
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "Catalog binding invariants are invalid."
            )
        result = cls(
            catalog_kind=normalized["catalog_kind"],
            source_sha256=normalized["source_sha256"],
            provider_sha256=normalized["provider_sha256"],
            authenticated_input_sha256=normalized[
                "authenticated_input_sha256"
            ],
            codec_sha256=normalized["codec_sha256"],
            bindings=normalized["bindings"],
        )
        if (
            _digest(normalized["binding_sha256"], "binding_sha256")
            != result.binding_sha256
            or normalized != result.as_dict()
        ):
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "Catalog binding seal is invalid."
            )
        return result

    @property
    def bindings(self) -> Mapping[str, Any]:
        return MappingProxyType(self.as_dict()["bindings"])

    def as_dict(self) -> dict[str, Any]:
        result = json.loads(self._body_json.decode("ascii"))
        result["binding_sha256"] = self.binding_sha256
        return result


@dataclass(frozen=True, slots=True, init=False)
class DynamicEvidenceCatalogArtifact:
    """One fully verified catalog artifact and its load disposition."""

    path: Path
    binding: DynamicEvidenceCatalogBinding
    disposition: str
    manifest_sha256: str
    catalog_sha256: str
    _catalog_json: bytes
    _manifest_json: bytes

    def __init__(
        self,
        *,
        path: Path,
        binding: DynamicEvidenceCatalogBinding,
        disposition: str,
        catalog: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        if disposition not in {"loaded", "published"}:
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "Artifact disposition must be loaded or published."
            )
        catalog_value = _normalized_json(catalog, "catalog payload")
        manifest_value = _normalized_json(manifest, "catalog manifest")
        if not isinstance(catalog_value, dict) or not isinstance(
            manifest_value, dict
        ):
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "Artifact catalog and manifest must be JSON objects."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self,
            "manifest_sha256",
            _digest(manifest_value["manifest_sha256"], "manifest_sha256"),
        )
        object.__setattr__(
            self,
            "catalog_sha256",
            _digest(catalog_value["catalog_sha256"], "catalog_sha256"),
        )
        object.__setattr__(
            self,
            "_catalog_json",
            canonical_json_bytes(catalog_value),
        )
        object.__setattr__(
            self,
            "_manifest_json",
            canonical_json_bytes(manifest_value),
        )

    @property
    def payload(self) -> Mapping[str, Any]:
        document = json.loads(self._catalog_json.decode("ascii"))
        return MappingProxyType(document["payload"])

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            json.loads(self._manifest_json.decode("ascii"))
        )

    def as_dict(self) -> Mapping[str, Any]:
        body = {
            "schema": DYNAMIC_EVIDENCE_CATALOG_ARTIFACT_SCHEMA,
            "path": str(self.path),
            "binding_sha256": self.binding.binding_sha256,
            "manifest_sha256": self.manifest_sha256,
            "catalog_sha256": self.catalog_sha256,
            "disposition": self.disposition,
            "immutable": True,
            "tensor_payloads_resident": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
        }
        return MappingProxyType(
            {
                **body,
                "receipt_sha256": canonical_sha256(body),
            }
        )

    def verify(self) -> DynamicEvidenceCatalogArtifact:
        """Freshly verify both files and the original authenticated binding."""

        return _load_artifact(
            self.path,
            binding=self.binding,
            disposition="loaded",
            expected_catalog_sha256=self.catalog_sha256,
            expected_manifest_sha256=self.manifest_sha256,
        )


def _catalog_document(
    binding: DynamicEvidenceCatalogBinding,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _normalized_json(payload, "catalog payload")
    if not isinstance(normalized, dict):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "catalog payload must be a JSON object."
        )
    body = {
        "schema": DYNAMIC_EVIDENCE_CATALOG_DOCUMENT_SCHEMA,
        "binding_sha256": binding.binding_sha256,
        "catalog_kind": binding.catalog_kind,
        "payload": normalized,
        "payload_sha256": canonical_sha256(normalized),
        "tensor_payloads_resident": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
    }
    return {
        **body,
        "catalog_sha256": canonical_sha256(body),
    }


def _manifest_document(
    binding: DynamicEvidenceCatalogBinding,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    catalog_bytes = _canonical_file_bytes(catalog)
    body = {
        "schema": DYNAMIC_EVIDENCE_CATALOG_ARTIFACT_SCHEMA,
        "binding": binding.as_dict(),
        "binding_sha256": binding.binding_sha256,
        "catalog": {
            "path": CATALOG_FILENAME,
            "size_bytes": len(catalog_bytes),
            "file_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "catalog_sha256": catalog["catalog_sha256"],
            "payload_sha256": catalog["payload_sha256"],
        },
        "inventory": [CATALOG_FILENAME, MANIFEST_FILENAME],
        "immutable": True,
        "tensor_payloads_resident": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
    }
    return {
        **body,
        "manifest_sha256": canonical_sha256(body),
    }


def _validate_catalog_document(
    value: Mapping[str, Any],
    *,
    binding: DynamicEvidenceCatalogBinding,
) -> dict[str, Any]:
    normalized = _normalized_json(value, "catalog document")
    expected = {
        "schema",
        "binding_sha256",
        "catalog_kind",
        "payload",
        "payload_sha256",
        "tensor_payloads_resident",
        "parent_forward_calls",
        "training_steps",
        "catalog_sha256",
    }
    if not isinstance(normalized, dict) or set(normalized) != expected:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog document does not contain exactly the v1 fields."
        )
    if (
        normalized["schema"] != DYNAMIC_EVIDENCE_CATALOG_DOCUMENT_SCHEMA
        or normalized["binding_sha256"] != binding.binding_sha256
        or normalized["catalog_kind"] != binding.catalog_kind
        or normalized["tensor_payloads_resident"] is not False
        or normalized["parent_forward_calls"] != 0
        or normalized["training_steps"] != 0
    ):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog document binding or zero-operation invariants changed."
        )
    payload = normalized["payload"]
    if not isinstance(payload, dict):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog document payload must be a JSON object."
        )
    claimed_payload = _digest(
        normalized["payload_sha256"],
        "payload_sha256",
    )
    if claimed_payload != canonical_sha256(payload):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog payload seal is invalid."
        )
    body = dict(normalized)
    claimed_catalog = _digest(body.pop("catalog_sha256"), "catalog_sha256")
    if claimed_catalog != canonical_sha256(body):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog document seal is invalid."
        )
    return normalized


def _validate_manifest(
    value: Mapping[str, Any],
    *,
    binding: DynamicEvidenceCatalogBinding,
    catalog: Mapping[str, Any],
    catalog_bytes: bytes,
) -> dict[str, Any]:
    normalized = _normalized_json(value, "catalog manifest")
    expected = {
        "schema",
        "binding",
        "binding_sha256",
        "catalog",
        "inventory",
        "immutable",
        "tensor_payloads_resident",
        "parent_forward_calls",
        "training_steps",
        "manifest_sha256",
    }
    if not isinstance(normalized, dict) or set(normalized) != expected:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog manifest does not contain exactly the v1 fields."
        )
    if (
        normalized["schema"] != DYNAMIC_EVIDENCE_CATALOG_ARTIFACT_SCHEMA
        or normalized["binding_sha256"] != binding.binding_sha256
        or normalized["binding"] != binding.as_dict()
        or normalized["inventory"] != [CATALOG_FILENAME, MANIFEST_FILENAME]
        or normalized["immutable"] is not True
        or normalized["tensor_payloads_resident"] is not False
        or normalized["parent_forward_calls"] != 0
        or normalized["training_steps"] != 0
    ):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog manifest binding or immutable invariants changed."
        )
    DynamicEvidenceCatalogBinding.from_dict(normalized["binding"])
    catalog_record = normalized["catalog"]
    if not isinstance(catalog_record, dict) or set(catalog_record) != {
        "path",
        "size_bytes",
        "file_sha256",
        "catalog_sha256",
        "payload_sha256",
    }:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog manifest file record is invalid."
        )
    if (
        catalog_record["path"] != CATALOG_FILENAME
        or catalog_record["size_bytes"] != len(catalog_bytes)
        or catalog_record["file_sha256"]
        != hashlib.sha256(catalog_bytes).hexdigest()
        or catalog_record["catalog_sha256"] != catalog["catalog_sha256"]
        or catalog_record["payload_sha256"] != catalog["payload_sha256"]
    ):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog manifest file binding is invalid."
        )
    body = dict(normalized)
    claimed = _digest(body.pop("manifest_sha256"), "manifest_sha256")
    if claimed != canonical_sha256(body):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog manifest seal is invalid."
        )
    return normalized


def _load_artifact(
    path: Path,
    *,
    binding: DynamicEvidenceCatalogBinding,
    disposition: str,
    expected_catalog_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> DynamicEvidenceCatalogArtifact:
    _real_directory(path, name="dynamic evidence catalog artifact")
    try:
        names = {entry.name for entry in path.iterdir()}
    except OSError as exc:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Cannot inspect dynamic evidence catalog artifact."
        ) from exc
    if names != _ARTIFACT_FILENAMES:
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Dynamic evidence catalog artifact inventory is not exact."
        )
    catalog_path = path / CATALOG_FILENAME
    manifest_path = path / MANIFEST_FILENAME
    catalog = _validate_catalog_document(
        _read_canonical_json(catalog_path, name="catalog document"),
        binding=binding,
    )
    catalog_bytes = _canonical_file_bytes(catalog)
    manifest = _validate_manifest(
        _read_canonical_json(manifest_path, name="catalog manifest"),
        binding=binding,
        catalog=catalog,
        catalog_bytes=catalog_bytes,
    )
    if (
        expected_catalog_sha256 is not None
        and _digest(expected_catalog_sha256, "expected_catalog_sha256")
        != catalog["catalog_sha256"]
    ):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog document changed after authenticated load."
        )
    if (
        expected_manifest_sha256 is not None
        and _digest(expected_manifest_sha256, "expected_manifest_sha256")
        != manifest["manifest_sha256"]
    ):
        raise DynamicEvidenceCatalogArtifactIntegrityError(
            "Catalog manifest changed after authenticated load."
        )
    return DynamicEvidenceCatalogArtifact(
        path=path,
        binding=binding,
        disposition=disposition,
        catalog=catalog,
        manifest=manifest,
    )


class DynamicEvidenceCatalogArtifactStore:
    """Content-addressed immutable store for tensor-free evidence catalogs."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        candidate = Path(root).expanduser().absolute()
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DynamicEvidenceCatalogArtifactPublicationError(
                f"Could not create catalog store root {candidate}."
            ) from exc
        root_stat = _real_directory(candidate, name="catalog store root")
        try:
            self.root = candidate.resolve(strict=True)
        except OSError as exc:
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "Could not resolve catalog store root."
            ) from exc
        if self.root != candidate or _is_link_or_junction(candidate):
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "Catalog store root must not traverse a link or junction."
            )
        self._root_identity = _stat_identity(root_stat)

    def _authenticate_root(self) -> None:
        current = _real_directory(self.root, name="catalog store root")
        if _stat_identity(current) != self._root_identity:
            raise DynamicEvidenceCatalogArtifactIntegrityError(
                "Catalog store root changed identity."
            )

    def path_for(self, binding: DynamicEvidenceCatalogBinding) -> Path:
        if not isinstance(binding, DynamicEvidenceCatalogBinding):
            raise TypeError("binding must be DynamicEvidenceCatalogBinding.")
        return self.root / binding.binding_sha256

    def load(
        self,
        binding: DynamicEvidenceCatalogBinding,
    ) -> DynamicEvidenceCatalogArtifact | None:
        """Load an exact hit; an existing invalid path is a hard failure."""

        self._authenticate_root()
        destination = self.path_for(binding)
        if not os.path.lexists(destination):
            return None
        return _load_artifact(
            destination,
            binding=binding,
            disposition="loaded",
        )

    def publish(
        self,
        binding: DynamicEvidenceCatalogBinding,
        payload: Mapping[str, Any],
    ) -> DynamicEvidenceCatalogArtifact:
        """Publish once, or authenticate an identical concurrent winner."""

        self._authenticate_root()
        document = _catalog_document(binding, payload)
        expected_payload = document["payload"]
        existing = self.load(binding)
        if existing is not None:
            if dict(existing.payload) != expected_payload:
                raise DynamicEvidenceCatalogArtifactPublicationError(
                    "Immutable catalog binding already contains another payload."
                )
            return existing

        stage = Path(
            tempfile.mkdtemp(prefix=_STAGE_PREFIX, dir=self.root)
        ).absolute()
        stage_identity = _stat_identity(
            _real_directory(stage, name="owned catalog staging directory")
        )
        promoted = False
        destination = self.path_for(binding)
        try:
            _write_new_canonical_json(stage / CATALOG_FILENAME, document)
            manifest = _manifest_document(binding, document)
            _write_new_canonical_json(stage / MANIFEST_FILENAME, manifest)
            _fsync_directory(stage)
            staged = _load_artifact(
                stage,
                binding=binding,
                disposition="published",
            )
            if dict(staged.payload) != expected_payload:
                raise DynamicEvidenceCatalogArtifactPublicationError(
                    "Staged catalog changed during fresh verification."
                )
            try:
                _atomic_rename_noreplace(stage, destination)
            except FileExistsError:
                winner = _load_artifact(
                    destination,
                    binding=binding,
                    disposition="loaded",
                )
                if dict(winner.payload) != expected_payload:
                    raise DynamicEvidenceCatalogArtifactPublicationError(
                        "Concurrent catalog winner has another payload."
                    )
                return winner
            promoted = True
            _fsync_directory(self.root)
            published = _load_artifact(
                destination,
                binding=binding,
                disposition="published",
                expected_catalog_sha256=staged.catalog_sha256,
                expected_manifest_sha256=staged.manifest_sha256,
            )
            if dict(published.payload) != expected_payload:
                raise DynamicEvidenceCatalogArtifactPublicationError(
                    "Published catalog changed after atomic promotion."
                )
            return published
        except DynamicEvidenceCatalogArtifactError:
            raise
        except BaseException as exc:
            raise DynamicEvidenceCatalogArtifactPublicationError(
                "Catalog publication failed before atomic completion."
            ) from exc
        finally:
            if not promoted and os.path.lexists(stage):
                try:
                    current = _real_directory(
                        stage,
                        name="owned catalog staging directory",
                    )
                except DynamicEvidenceCatalogArtifactIntegrityError:
                    pass
                else:
                    if (
                        _stat_identity(current) == stage_identity
                        and stage.parent == self.root
                        and stage.name.startswith(_STAGE_PREFIX)
                    ):
                        shutil.rmtree(stage)

    def load_or_build(
        self,
        binding: DynamicEvidenceCatalogBinding,
        builder: Callable[[], Mapping[str, Any]],
    ) -> DynamicEvidenceCatalogArtifact:
        """Load a hit, otherwise build and atomically publish one catalog."""

        if not callable(builder):
            raise TypeError("builder must be callable.")
        existing = self.load(binding)
        if existing is not None:
            return existing
        return self.publish(binding, builder())


__all__ = [
    "CATALOG_FILENAME",
    "DYNAMIC_EVIDENCE_CATALOG_ARTIFACT_SCHEMA",
    "DYNAMIC_EVIDENCE_CATALOG_BINDING_SCHEMA",
    "DYNAMIC_EVIDENCE_CATALOG_DOCUMENT_SCHEMA",
    "DynamicEvidenceCatalogArtifact",
    "DynamicEvidenceCatalogArtifactError",
    "DynamicEvidenceCatalogArtifactIntegrityError",
    "DynamicEvidenceCatalogArtifactPublicationError",
    "DynamicEvidenceCatalogArtifactStore",
    "DynamicEvidenceCatalogBinding",
    "MANIFEST_FILENAME",
]
