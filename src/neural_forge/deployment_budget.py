"""Fail-closed accounting for serialized deployment artifacts.

Callers must classify every file required by the deployed runtime, including
tokenizers, indices, codebooks, and runtime metadata.  This module returns an
in-memory report and never writes one into the artifact tree.  A serialized
copy of the report may be written outside the artifact set so that the report
does not create circular accounting or invalidate its own file inventory.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TypeAlias


class ArtifactAccountingError(ValueError):
    """Raised when an artifact tree cannot be accounted for exactly."""


class ArtifactCategory(str, Enum):
    """Required categories for every serialized deployment file."""

    LEARNED_WEIGHTS = "learned_weights"
    INDICES = "indices"
    TOKENIZER = "tokenizer"
    METADATA = "metadata"
    AUXILIARY = "auxiliary"


_CATEGORY_NAMES = tuple(category.value for category in ArtifactCategory)
_LOWER_HEX = frozenset("0123456789abcdef")
_HASH_CHUNK_BYTES = 1024 * 1024

PathInput: TypeAlias = str | os.PathLike[str]
CategoryInput: TypeAlias = str | ArtifactCategory
CategoryMapping: TypeAlias = (
    Mapping[PathInput, CategoryInput] | Iterable[tuple[PathInput, CategoryInput]]
)


def _validate_nonnegative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactAccountingError(f"{name} must be a non-negative integer.")


def _validate_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactAccountingError(f"{name} must be a positive integer.")


def _validate_optional_count(value: object, name: str) -> None:
    if value is not None:
        _validate_nonnegative_integer(value, name)


def _validate_optional_flops(value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactAccountingError(
            "flops_per_token must be a finite non-negative number."
        )
    if not math.isfinite(value) or value < 0:
        raise ArtifactAccountingError(
            "flops_per_token must be a finite non-negative number."
        )


@dataclass(frozen=True, slots=True)
class CategoryByteTotals(Mapping[str, int]):
    """Immutable serialized-byte totals for all required categories."""

    learned_weights: int
    indices: int
    tokenizer: int
    metadata: int
    auxiliary: int

    def __post_init__(self) -> None:
        for category in _CATEGORY_NAMES:
            _validate_nonnegative_integer(getattr(self, category), category)

    def __getitem__(self, category: str) -> int:
        key = category.value if isinstance(category, ArtifactCategory) else category
        if key not in _CATEGORY_NAMES:
            raise KeyError(category)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_CATEGORY_NAMES)

    def __len__(self) -> int:
        return len(_CATEGORY_NAMES)

    @property
    def total_serialized_bytes(self) -> int:
        return sum(self[category] for category in _CATEGORY_NAMES)

    def as_dict(self) -> dict[str, int]:
        """Return a detached plain-dict representation."""

        return {category: self[category] for category in _CATEGORY_NAMES}


@dataclass(frozen=True, slots=True)
class ArtifactFileReport:
    """Immutable accounting evidence for one root-relative regular file."""

    relative_path: str
    category: ArtifactCategory
    serialized_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ArtifactAccountingError(
                "Artifact relative_path must be a non-empty string."
            )
        normalized = PurePosixPath(self.relative_path)
        if (
            normalized.is_absolute()
            or not normalized.parts
            or ".." in normalized.parts
            or "\\" in self.relative_path
            or normalized.as_posix() != self.relative_path
        ):
            raise ArtifactAccountingError(
                f"Artifact path is not canonical and root-relative: "
                f"{self.relative_path!r}."
            )
        if not isinstance(self.category, ArtifactCategory):
            raise ArtifactAccountingError(
                "Artifact category must be an ArtifactCategory value."
            )
        _validate_nonnegative_integer(
            self.serialized_bytes,
            f"serialized_bytes for {self.relative_path}",
        )
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in _LOWER_HEX for character in self.sha256)
        ):
            raise ArtifactAccountingError(
                f"Invalid SHA-256 digest for {self.relative_path!r}."
            )


@dataclass(frozen=True, slots=True)
class DeploymentBudgetReport:
    """Immutable storage, parameter, compute, and memory accounting report."""

    artifact_root: str
    files: tuple[ArtifactFileReport, ...]
    category_totals: CategoryByteTotals
    actual_total_serialized_bytes: int
    source_serialized_bytes: int
    source_to_deployable_byte_ratio: float
    unique_parameters: int | None = None
    logical_parameters: int | None = None
    flops_per_token: int | float | None = None
    kv_cache_bytes_per_token: int | None = None
    peak_load_bytes: int | None = None
    peak_runtime_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_root, str) or not os.path.isabs(
            self.artifact_root
        ):
            raise ArtifactAccountingError("artifact_root must be an absolute path.")
        if not isinstance(self.files, tuple) or not all(
            isinstance(file_report, ArtifactFileReport) for file_report in self.files
        ):
            raise ArtifactAccountingError(
                "files must be a tuple of ArtifactFileReport values."
            )
        paths = tuple(file_report.relative_path for file_report in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ArtifactAccountingError(
                "Artifact file reports must have unique, sorted relative paths."
            )
        if not isinstance(self.category_totals, CategoryByteTotals):
            raise ArtifactAccountingError(
                "category_totals must be a CategoryByteTotals value."
            )

        computed_by_category = {category: 0 for category in _CATEGORY_NAMES}
        for file_report in self.files:
            computed_by_category[file_report.category.value] += (
                file_report.serialized_bytes
            )
        computed_totals = CategoryByteTotals(**computed_by_category)
        if computed_totals != self.category_totals:
            raise ArtifactAccountingError(
                "Category byte totals do not match the per-file inventory."
            )

        _validate_positive_integer(
            self.actual_total_serialized_bytes,
            "actual_total_serialized_bytes",
        )
        file_total = sum(file_report.serialized_bytes for file_report in self.files)
        if self.actual_total_serialized_bytes != file_total:
            raise ArtifactAccountingError(
                "Actual total serialized bytes do not match the per-file inventory."
            )
        if (
            self.category_totals.total_serialized_bytes
            != self.actual_total_serialized_bytes
        ):
            raise ArtifactAccountingError(
                "Category byte totals do not match actual total serialized bytes."
            )

        _validate_positive_integer(
            self.source_serialized_bytes,
            "source_serialized_bytes",
        )
        expected_ratio = (
            self.source_serialized_bytes / self.actual_total_serialized_bytes
        )
        if (
            isinstance(self.source_to_deployable_byte_ratio, bool)
            or not isinstance(self.source_to_deployable_byte_ratio, (int, float))
            or not math.isfinite(self.source_to_deployable_byte_ratio)
            or not math.isclose(
                self.source_to_deployable_byte_ratio,
                expected_ratio,
                rel_tol=1e-15,
                abs_tol=0.0,
            )
        ):
            raise ArtifactAccountingError(
                "source_to_deployable_byte_ratio does not match serialized bytes."
            )

        _validate_optional_count(self.unique_parameters, "unique_parameters")
        _validate_optional_count(self.logical_parameters, "logical_parameters")
        _validate_optional_flops(self.flops_per_token)
        _validate_optional_count(
            self.kv_cache_bytes_per_token,
            "kv_cache_bytes_per_token",
        )
        _validate_optional_count(self.peak_load_bytes, "peak_load_bytes")
        _validate_optional_count(self.peak_runtime_bytes, "peak_runtime_bytes")

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_serialized_bytes(self) -> int:
        """Alias emphasizing that this is the complete deployable byte total."""

        return self.actual_total_serialized_bytes


def build_deployment_budget_report(
    artifact_root: PathInput,
    file_categories: CategoryMapping,
    *,
    source_serialized_bytes: int,
    unique_parameters: int | None = None,
    logical_parameters: int | None = None,
    flops_per_token: int | float | None = None,
    kv_cache_bytes_per_token: int | None = None,
    peak_load_bytes: int | None = None,
    peak_runtime_bytes: int | None = None,
) -> DeploymentBudgetReport:
    """Account for every regular file in a deployment artifact tree.

    ``file_categories`` keys may be root-relative or absolute paths.  A mapping
    or an iterable of ``(path, category)`` pairs is accepted; the iterable form
    allows duplicate declarations to be detected explicitly.  Every enumerated
    regular file must appear exactly once.  Missing paths, aliases, paths that
    escape the root, symlinks, special files, and unclassified files fail the
    operation before a report is returned.

    ``source_serialized_bytes`` is the complete serialized size of the source
    artifact, not only its learned tensors.  Callers are likewise responsible
    for classifying every runtime-required deployable file.  Persist this report
    outside ``artifact_root`` when it should not become part of that artifact's
    runtime accounting.
    """

    _validate_positive_integer(
        source_serialized_bytes,
        "source_serialized_bytes",
    )
    _validate_optional_count(unique_parameters, "unique_parameters")
    _validate_optional_count(logical_parameters, "logical_parameters")
    _validate_optional_flops(flops_per_token)
    _validate_optional_count(
        kv_cache_bytes_per_token,
        "kv_cache_bytes_per_token",
    )
    _validate_optional_count(peak_load_bytes, "peak_load_bytes")
    _validate_optional_count(peak_runtime_bytes, "peak_runtime_bytes")

    root = _resolve_artifact_root(artifact_root)
    inventory = _enumerate_regular_files(root)
    inventory_set = set(inventory)
    classifications: dict[Path, ArtifactCategory] = {}
    declared_as: dict[Path, str] = {}

    for raw_path, raw_category in _iter_category_entries(file_categories):
        category = _parse_category(raw_category)
        normalized_path = _normalize_classified_path(root, raw_path)
        if normalized_path in classifications:
            previous = declared_as[normalized_path]
            raise ArtifactAccountingError(
                "Duplicate category mapping for artifact path "
                f"{os.fspath(raw_path)!r}; it was already declared as "
                f"{previous!r}."
            )
        if normalized_path not in inventory_set:
            raise ArtifactAccountingError(
                "Classified path was not enumerated as a regular artifact file: "
                f"{os.fspath(raw_path)!r}."
            )
        classifications[normalized_path] = category
        declared_as[normalized_path] = os.fspath(raw_path)

    unclassified = [
        _relative_name(root, path) for path in inventory if path not in classifications
    ]
    if unclassified:
        raise ArtifactAccountingError(
            "Unclassified artifact files: " + ", ".join(unclassified)
        )

    file_reports: list[ArtifactFileReport] = []
    byte_totals = {category: 0 for category in _CATEGORY_NAMES}
    for path in inventory:
        digest, serialized_bytes = _hash_regular_file(path)
        category = classifications[path]
        file_reports.append(
            ArtifactFileReport(
                relative_path=_relative_name(root, path),
                category=category,
                serialized_bytes=serialized_bytes,
                sha256=digest,
            )
        )
        byte_totals[category.value] += serialized_bytes

    category_totals = CategoryByteTotals(**byte_totals)
    actual_total = category_totals.total_serialized_bytes
    if actual_total <= 0:
        raise ArtifactAccountingError(
            "Deployment artifacts must contain at least one serialized byte."
        )

    return DeploymentBudgetReport(
        artifact_root=str(root),
        files=tuple(file_reports),
        category_totals=category_totals,
        actual_total_serialized_bytes=actual_total,
        source_serialized_bytes=source_serialized_bytes,
        source_to_deployable_byte_ratio=(source_serialized_bytes / actual_total),
        unique_parameters=unique_parameters,
        logical_parameters=logical_parameters,
        flops_per_token=flops_per_token,
        kv_cache_bytes_per_token=kv_cache_bytes_per_token,
        peak_load_bytes=peak_load_bytes,
        peak_runtime_bytes=peak_runtime_bytes,
    )


def _resolve_artifact_root(artifact_root: PathInput) -> Path:
    try:
        raw_root = Path(os.fspath(artifact_root))
    except TypeError as error:
        raise ArtifactAccountingError(
            "artifact_root must be a filesystem path."
        ) from error
    if not raw_root.is_absolute():
        raw_root = Path.cwd() / raw_root
    try:
        root_info = os.lstat(raw_root)
    except FileNotFoundError as error:
        raise ArtifactAccountingError(
            f"Artifact root does not exist: {raw_root}."
        ) from error
    except OSError as error:
        raise ArtifactAccountingError(
            f"Unable to inspect artifact root {raw_root}: {error}."
        ) from error
    if _is_link_like(raw_root, root_info):
        raise ArtifactAccountingError(
            f"Artifact root may not be a symlink or junction: {raw_root}."
        )
    if not stat.S_ISDIR(root_info.st_mode):
        raise ArtifactAccountingError(f"Artifact root is not a directory: {raw_root}.")
    try:
        return raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ArtifactAccountingError(
            f"Unable to resolve artifact root {raw_root}: {error}."
        ) from error


def _enumerate_regular_files(root: Path) -> tuple[Path, ...]:
    pending = [root]
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as error:
            raise ArtifactAccountingError(
                f"Unable to enumerate artifact directory {directory}: {error}."
            ) from error

        for entry in children:
            path = Path(entry.path)
            try:
                path_info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ArtifactAccountingError(
                    f"Unable to inspect artifact path {path}: {error}."
                ) from error
            if _is_link_like(path, path_info):
                _reject_link(root, path)
            if stat.S_ISDIR(path_info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(path_info.st_mode):
                files.append(path)
            else:
                raise ArtifactAccountingError(
                    "Deployment artifact contains a non-regular filesystem "
                    f"entry: {_relative_name(root, path)}."
                )

    return tuple(sorted(files, key=lambda path: _relative_name(root, path)))


def _reject_link(root: Path, path: Path) -> None:
    relative_path = _relative_name(root, path)
    try:
        target = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ArtifactAccountingError(
            f"Unsafe or broken symlink in deployment artifact: {relative_path}."
        ) from error
    if not _is_within(target, root):
        raise ArtifactAccountingError(
            f"Unsafe symlink escapes artifact root: {relative_path} -> {target}."
        )
    raise ArtifactAccountingError(
        "Symlinks and junctions are not permitted in deployment artifacts: "
        f"{relative_path}."
    )


def _is_link_like(path: Path, path_info: os.stat_result) -> bool:
    if stat.S_ISLNK(path_info.st_mode):
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _iter_category_entries(
    file_categories: CategoryMapping,
) -> Iterator[tuple[PathInput, CategoryInput]]:
    if isinstance(file_categories, Mapping):
        entries: Iterable[object] = file_categories.items()
    else:
        if isinstance(file_categories, (str, bytes, os.PathLike)):
            raise ArtifactAccountingError(
                "file_categories must be a mapping or iterable of path/category pairs."
            )
        try:
            entries = iter(file_categories)
        except TypeError as error:
            raise ArtifactAccountingError(
                "file_categories must be a mapping or iterable of path/category pairs."
            ) from error

    for index, entry in enumerate(entries):
        try:
            raw_path, raw_category = entry  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise ArtifactAccountingError(
                f"Category declaration {index} is not a path/category pair."
            ) from error
        yield raw_path, raw_category


def _parse_category(raw_category: CategoryInput) -> ArtifactCategory:
    if isinstance(raw_category, ArtifactCategory):
        return raw_category
    try:
        return ArtifactCategory(raw_category)
    except (TypeError, ValueError) as error:
        allowed = ", ".join(_CATEGORY_NAMES)
        raise ArtifactAccountingError(
            f"Unknown artifact category {raw_category!r}; expected one of: {allowed}."
        ) from error


def _normalize_classified_path(root: Path, raw_path: PathInput) -> Path:
    try:
        path = Path(os.fspath(raw_path))
    except TypeError as error:
        raise ArtifactAccountingError(
            f"Artifact category key is not a filesystem path: {raw_path!r}."
        ) from error
    candidate = path if path.is_absolute() else root / path
    lexical_path = Path(os.path.abspath(candidate))
    if not _is_within(lexical_path, root):
        raise ArtifactAccountingError(
            f"Classified artifact path is outside artifact root: {raw_path!r}."
        )
    try:
        resolved_path = lexical_path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ArtifactAccountingError(
            f"Unable to resolve classified artifact path {raw_path!r}: {error}."
        ) from error
    if not _is_within(resolved_path, root):
        raise ArtifactAccountingError(
            f"Classified artifact path is outside artifact root: {raw_path!r}."
        )
    try:
        path_info = os.lstat(lexical_path)
    except FileNotFoundError as error:
        raise ArtifactAccountingError(
            f"Classified artifact path does not exist: {raw_path!r}."
        ) from error
    except OSError as error:
        raise ArtifactAccountingError(
            f"Unable to inspect classified path {raw_path!r}: {error}."
        ) from error
    if _is_link_like(lexical_path, path_info):
        _reject_link(root, lexical_path)
    if not stat.S_ISREG(path_info.st_mode):
        raise ArtifactAccountingError(
            f"Classified artifact path is not a regular file: {raw_path!r}."
        )
    try:
        return lexical_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ArtifactAccountingError(
            f"Unable to resolve classified artifact path {raw_path!r}: {error}."
        ) from error


def _hash_regular_file(path: Path) -> tuple[str, int]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ArtifactAccountingError(
            f"Unable to inspect artifact file before hashing {path}: {error}."
        ) from error
    if _is_link_like(path, before) or not stat.S_ISREG(before.st_mode):
        raise ArtifactAccountingError(
            f"Artifact changed to a non-regular file before hashing: {path}."
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactAccountingError(
            f"Unable to open artifact file for hashing {path}: {error}."
        ) from error

    digest = hashlib.sha256()
    bytes_read = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            raise ArtifactAccountingError(
                f"Artifact changed while it was opened for hashing: {path}."
            )
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        opened_after = os.fstat(descriptor)
    except OSError as error:
        raise ArtifactAccountingError(
            f"Unable to hash artifact file {path}: {error}."
        ) from error
    finally:
        os.close(descriptor)

    try:
        after = os.lstat(path)
    except OSError as error:
        raise ArtifactAccountingError(
            f"Unable to inspect artifact file after hashing {path}: {error}."
        ) from error
    if (
        _is_link_like(path, after)
        or not stat.S_ISREG(after.st_mode)
        or not _same_file(before, opened_after)
        or not _same_file(before, after)
        or before.st_size != opened_after.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != opened_after.st_mtime_ns
        or before.st_mtime_ns != after.st_mtime_ns
        or bytes_read != before.st_size
    ):
        raise ArtifactAccountingError(
            f"Artifact changed while it was being hashed: {path}."
        )
    return digest.hexdigest(), bytes_read


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _relative_name(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ArtifactAccountingError(
            f"Artifact path is outside artifact root: {path}."
        ) from error


__all__ = [
    "ArtifactAccountingError",
    "ArtifactCategory",
    "ArtifactFileReport",
    "CategoryByteTotals",
    "DeploymentBudgetReport",
    "build_deployment_budget_report",
]
