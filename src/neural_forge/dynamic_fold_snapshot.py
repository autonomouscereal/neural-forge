"""Whole-model tensor snapshot storage for dynamic fold rollback.

The executor retains parameter/buffer object bindings, module modes, and hook
registries in memory.  This module stores only tensor values.  The default
backend preserves the original all-in-memory behavior; the disk backend writes
bounded CPU chunks to fsynced raw files and never retains a whole-model CPU
clone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol


SNAPSHOT_SCHEMA = "neural-forge-dynamic-fold-state-snapshot-v1"
_SNAPSHOT_PREFIX = ".neural-forge-fold-snapshot-"


class DynamicFoldSnapshotError(RuntimeError):
    """Snapshot configuration, capture, restoration, or cleanup failed."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicFoldSnapshotError(f"{name} must be a positive integer.")
    return value


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(callable(is_junction) and is_junction(path))


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DynamicFoldSnapshotError(
            "Snapshot metadata is not canonical JSON."
        ) from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Windows does not permit opening directories this way.  Every data
        # file and manifest is still flushed and fsynced individually.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise DynamicFoldSnapshotError(
            f"Could not write snapshot file {path}."
        ) from exc


def _resolve_disk_root(root: Path) -> Path:
    candidate = root.expanduser().absolute()
    if not candidate.exists():
        raise DynamicFoldSnapshotError(f"Snapshot root does not exist: {candidate}.")
    if not candidate.is_dir() or _is_link_or_junction(candidate):
        raise DynamicFoldSnapshotError(
            "Snapshot root must be a real directory, not a file, symlink, "
            f"or junction: {candidate}."
        )
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise DynamicFoldSnapshotError(
            f"Could not resolve snapshot root {candidate}."
        ) from exc


def _safe_remove_snapshot(path: Path, root: Path) -> None:
    if not path.exists():
        return
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise DynamicFoldSnapshotError(
            "Could not resolve snapshot cleanup paths."
        ) from exc
    if (
        resolved_path.parent != resolved_root
        or not resolved_path.name.startswith(_SNAPSHOT_PREFIX)
        or not resolved_path.is_dir()
        or _is_link_or_junction(path)
    ):
        raise DynamicFoldSnapshotError(
            "Refusing to recursively remove an unsafe snapshot path."
        )
    shutil.rmtree(resolved_path)
    _fsync_directory(resolved_root)


@dataclass(frozen=True, slots=True)
class DynamicFoldSnapshotConfig:
    """Select in-memory or bounded disk-backed rollback storage."""

    backend: str = "memory"
    root_directory: Path | str | None = None
    chunk_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in {"memory", "disk"}:
            raise DynamicFoldSnapshotError(
                "snapshot backend must be 'memory' or 'disk'."
            )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(
            self,
            "chunk_bytes",
            _positive_int(self.chunk_bytes, "chunk_bytes"),
        )
        root = self.root_directory
        if backend == "memory":
            if root is not None:
                raise DynamicFoldSnapshotError(
                    "Memory snapshots do not accept root_directory."
                )
            return
        normalized = Path(tempfile.gettempdir()) if root is None else Path(root)
        object.__setattr__(self, "root_directory", normalized)

    @classmethod
    def memory(cls) -> "DynamicFoldSnapshotConfig":
        return cls()

    @classmethod
    def disk(
        cls,
        root_directory: Path | str | None = None,
        *,
        chunk_bytes: int = 64 * 1024 * 1024,
    ) -> "DynamicFoldSnapshotConfig":
        return cls(
            backend="disk",
            root_directory=root_directory,
            chunk_bytes=chunk_bytes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "root_directory": (
                None
                if self.root_directory is None
                else str(Path(self.root_directory).expanduser().absolute())
            ),
            "chunk_bytes": self.chunk_bytes,
        }


class CapturedModelState(Protocol):
    backend: str

    @property
    def names(self) -> tuple[str, ...]: ...

    @property
    def artifact_path(self) -> Path | None: ...

    def restore(self, current_state: Mapping[str, Any]) -> None: ...

    def cleanup(self) -> None: ...


@dataclass(slots=True)
class InMemoryCapturedModelState:
    """Original behavior: one detached CPU clone per named state tensor."""

    state_values: tuple[tuple[str, Any], ...]
    backend: str = "memory"

    @classmethod
    def capture(
        cls,
        values: Sequence[tuple[str, Any]],
    ) -> "InMemoryCapturedModelState":
        return cls(
            tuple(
                (
                    name,
                    value.detach().to(device="cpu").contiguous().clone(),
                )
                for name, value in values
            )
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _value in self.state_values)

    @property
    def artifact_path(self) -> Path | None:
        return None

    def restore(self, current_state: Mapping[str, Any]) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional boundary
            raise DynamicFoldSnapshotError(
                "PyTorch is required for snapshot restoration."
            ) from exc
        if set(current_state) != set(self.names):
            raise DynamicFoldSnapshotError(
                "The model state topology changed during folding."
            )
        with torch.no_grad():
            for name, value in self.state_values:
                current = current_state[name]
                current.copy_(value.to(device=current.device, dtype=current.dtype))

    def cleanup(self) -> None:
        self.state_values = ()


@dataclass(frozen=True, slots=True)
class _DiskTensorEntry:
    name: str
    filename: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "numel": self.numel,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(slots=True)
class DiskCapturedModelState:
    """Raw, fsynced, chunked tensor files under one safely-owned directory."""

    root: Path
    path: Path
    entries: tuple[_DiskTensorEntry, ...]
    chunk_bytes: int
    manifest_sha256: str
    cleaned: bool = False
    backend: str = "disk"

    @classmethod
    def capture(
        cls,
        values: Sequence[tuple[str, Any]],
        config: DynamicFoldSnapshotConfig,
    ) -> "DiskCapturedModelState":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional boundary
            raise DynamicFoldSnapshotError(
                "PyTorch is required for disk snapshots."
            ) from exc
        if config.backend != "disk" or config.root_directory is None:
            raise DynamicFoldSnapshotError(
                "Disk capture requires a disk snapshot configuration."
            )
        root = _resolve_disk_root(Path(config.root_directory))
        try:
            directory = Path(
                tempfile.mkdtemp(prefix=_SNAPSHOT_PREFIX, dir=root)
            ).resolve(strict=True)
        except OSError as exc:
            raise DynamicFoldSnapshotError(
                f"Could not create a snapshot under {root}."
            ) from exc
        entries: list[_DiskTensorEntry] = []
        try:
            for index, (name, value) in enumerate(values):
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(value, torch.Tensor)
                    or value.device.type == "meta"
                    or value.layout != torch.strided
                ):
                    raise DynamicFoldSnapshotError(
                        "Disk snapshots require named materialized dense tensors."
                    )
                filename = f"tensor-{index:08d}.bin"
                destination = directory / filename
                digest = hashlib.sha256()
                size_bytes = 0
                source = value.detach()
                if not source.is_contiguous():
                    source = source.contiguous()
                flat = source.view(-1)
                element_size = int(source.element_size())
                elements_per_chunk = max(
                    1,
                    config.chunk_bytes // element_size,
                )
                try:
                    with destination.open("xb") as handle:
                        for start in range(
                            0,
                            int(flat.numel()),
                            elements_per_chunk,
                        ):
                            count = min(
                                elements_per_chunk,
                                int(flat.numel()) - start,
                            )
                            cpu_chunk = (
                                flat.narrow(0, start, count)
                                .to(device="cpu")
                                .contiguous()
                            )
                            payload = cpu_chunk.view(torch.uint8).numpy().tobytes()
                            handle.write(payload)
                            digest.update(payload)
                            size_bytes += len(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                except OSError as exc:
                    raise DynamicFoldSnapshotError(
                        f"Could not write tensor snapshot {destination}."
                    ) from exc
                expected_size = int(value.numel() * value.element_size())
                if size_bytes != expected_size:
                    raise DynamicFoldSnapshotError(
                        f"Snapshot byte count differs for {name!r}."
                    )
                entries.append(
                    _DiskTensorEntry(
                        name=name,
                        filename=filename,
                        dtype=str(value.dtype),
                        shape=tuple(int(size) for size in value.shape),
                        numel=int(value.numel()),
                        size_bytes=size_bytes,
                        sha256=digest.hexdigest(),
                    )
                )
            body = {
                "schema": SNAPSHOT_SCHEMA,
                "backend": "disk",
                "chunk_bytes": config.chunk_bytes,
                "entries": [entry.as_dict() for entry in entries],
            }
            manifest_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
            manifest = {
                **body,
                "manifest_sha256": manifest_sha256,
            }
            _write_fsynced(
                directory / "manifest.json",
                _canonical_json(manifest) + b"\n",
            )
            _fsync_directory(directory)
            _fsync_directory(root)
            return cls(
                root=root,
                path=directory,
                entries=tuple(entries),
                chunk_bytes=config.chunk_bytes,
                manifest_sha256=manifest_sha256,
            )
        except BaseException:
            _safe_remove_snapshot(directory, root)
            raise

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def artifact_path(self) -> Path | None:
        return None if self.cleaned else self.path

    @staticmethod
    def _torch_dtype(torch: Any, value: str) -> Any:
        prefix = "torch."
        if not value.startswith(prefix):
            raise DynamicFoldSnapshotError(f"Unsupported snapshot dtype {value!r}.")
        result = getattr(torch, value[len(prefix) :], None)
        if result is None:
            raise DynamicFoldSnapshotError(
                f"Installed PyTorch lacks snapshot dtype {value!r}."
            )
        return result

    def _verify_manifest(self) -> None:
        manifest_path = self.path / "manifest.json"
        try:
            raw = manifest_path.read_bytes()
            manifest = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DynamicFoldSnapshotError(
                "Could not read the disk snapshot manifest."
            ) from exc
        if raw != _canonical_json(manifest) + b"\n":
            raise DynamicFoldSnapshotError("Disk snapshot manifest is not canonical.")
        body = dict(manifest)
        observed = body.pop("manifest_sha256", None)
        expected = hashlib.sha256(_canonical_json(body)).hexdigest()
        if (
            observed != expected
            or observed != self.manifest_sha256
            or body.get("schema") != SNAPSHOT_SCHEMA
        ):
            raise DynamicFoldSnapshotError(
                "Disk snapshot manifest integrity check failed."
            )

    def restore(self, current_state: Mapping[str, Any]) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional boundary
            raise DynamicFoldSnapshotError(
                "PyTorch is required for disk snapshot restoration."
            ) from exc
        if self.cleaned or not self.path.exists():
            raise DynamicFoldSnapshotError("Cannot restore a cleaned disk snapshot.")
        self._verify_manifest()
        if set(current_state) != set(self.names):
            raise DynamicFoldSnapshotError(
                "The model state topology changed during folding."
            )
        with torch.no_grad():
            for entry in self.entries:
                current = current_state[entry.name]
                if (
                    not isinstance(current, torch.Tensor)
                    or current.device.type == "meta"
                    or current.layout != torch.strided
                    or tuple(int(size) for size in current.shape) != entry.shape
                    or str(current.dtype) != entry.dtype
                    or int(current.numel()) != entry.numel
                ):
                    raise DynamicFoldSnapshotError(
                        f"Current tensor contract differs for {entry.name!r}."
                    )
                source_path = self.path / entry.filename
                if (
                    source_path.parent != self.path
                    or source_path.is_symlink()
                    or not source_path.is_file()
                    or source_path.stat().st_size != entry.size_bytes
                ):
                    raise DynamicFoldSnapshotError(
                        f"Snapshot tensor file is unsafe for {entry.name!r}."
                    )
                dtype = self._torch_dtype(torch, entry.dtype)
                element_size = int(current.element_size())
                elements_per_chunk = max(
                    1,
                    self.chunk_bytes // element_size,
                )
                temporary = None
                if current.is_contiguous():
                    target_flat = current.view(-1)
                else:
                    temporary = torch.empty(
                        entry.shape,
                        dtype=dtype,
                        device="cpu",
                    )
                    target_flat = temporary.view(-1)
                digest = hashlib.sha256()
                offset = 0
                try:
                    with source_path.open("rb") as handle:
                        while offset < entry.numel:
                            count = min(
                                elements_per_chunk,
                                entry.numel - offset,
                            )
                            expected_bytes = count * element_size
                            payload = handle.read(expected_bytes)
                            if len(payload) != expected_bytes:
                                raise DynamicFoldSnapshotError(
                                    f"Snapshot tensor {entry.name!r} is truncated."
                                )
                            digest.update(payload)
                            chunk = torch.frombuffer(
                                bytearray(payload),
                                dtype=dtype,
                            )
                            target_flat.narrow(0, offset, count).copy_(
                                chunk.to(device=target_flat.device)
                            )
                            offset += count
                        if handle.read(1):
                            raise DynamicFoldSnapshotError(
                                f"Snapshot tensor {entry.name!r} has trailing bytes."
                            )
                except OSError as exc:
                    raise DynamicFoldSnapshotError(
                        f"Could not restore tensor {entry.name!r}."
                    ) from exc
                if digest.hexdigest() != entry.sha256:
                    raise DynamicFoldSnapshotError(
                        f"Snapshot tensor hash differs for {entry.name!r}."
                    )
                if temporary is not None:
                    current.copy_(temporary.to(device=current.device))

    def cleanup(self) -> None:
        if self.cleaned:
            return
        _safe_remove_snapshot(self.path, self.root)
        self.cleaned = True


def capture_model_state(
    values: Sequence[tuple[str, Any]],
    config: DynamicFoldSnapshotConfig | None = None,
) -> CapturedModelState:
    """Capture named tensor values using the explicitly selected backend."""

    selected = config or DynamicFoldSnapshotConfig.memory()
    if not isinstance(selected, DynamicFoldSnapshotConfig):
        raise DynamicFoldSnapshotError(
            "snapshot config must be DynamicFoldSnapshotConfig."
        )
    normalized = tuple(values)
    names = tuple(name for name, _value in normalized)
    if not normalized or len(set(names)) != len(names):
        raise DynamicFoldSnapshotError(
            "Snapshot state names must be nonempty and unique."
        )
    if selected.backend == "memory":
        return InMemoryCapturedModelState.capture(normalized)
    return DiskCapturedModelState.capture(normalized, selected)


def snapshot_directory_inventory(
    root: Path | str,
) -> Mapping[str, tuple[str, ...]]:
    """Read-only test/operations helper for active snapshot directories."""

    resolved = _resolve_disk_root(Path(root))
    result = {
        path.name: tuple(sorted(child.name for child in path.iterdir()))
        for path in resolved.iterdir()
        if path.is_dir() and path.name.startswith(_SNAPSHOT_PREFIX)
    }
    return MappingProxyType(result)


__all__ = [
    "CapturedModelState",
    "DiskCapturedModelState",
    "DynamicFoldSnapshotConfig",
    "DynamicFoldSnapshotError",
    "InMemoryCapturedModelState",
    "SNAPSHOT_SCHEMA",
    "capture_model_state",
    "snapshot_directory_inventory",
]
