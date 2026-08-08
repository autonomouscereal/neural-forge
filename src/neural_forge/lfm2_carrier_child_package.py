"""Reusable counted child-package wiring for LFM2 semantic carrier routes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
    CountedChildStatePlan,
    build_counted_child_state_plan,
)
from .hf_lfm2_direct_local import load_lfm2_direct_local_tokenizer
from .lfm2_carrier_bridge_publication import (
    AuthenticatedPackagedLfm2CarrierRoutes,
    Lfm2PortableCarrierRuntimeReceipt,
    Lfm2CarrierBridgePublicationError,
    bind_lfm2_carrier_bridge_publication_to_child,
    bind_packaged_lfm2_carrier_routes,
    fingerprint_lfm2_carrier_runtime_owned_tensor_state,
    load_authenticated_lfm2_span_evidence,
    load_lfm2_carrier_bridge_publication,
    load_packaged_lfm2_carrier_routes,
    package_lfm2_carrier_routes_with_child,
    relocate_and_bind_packaged_lfm2_carrier_routes,
)
from .lfm2_carrier_route_runtime import (
    install_lfm2_semantic_step_generate,
    run_lfm2_resumable_adaptive_carrier_manual_review,
)
from .lfm2_semantic_source_lexicon import (
    AuthenticatedLfm2SemanticSourceLexicon,
    load_packaged_lfm2_semantic_source_lexicon,
    package_lfm2_semantic_source_lexicon,
)


CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS = 229_693_184
CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS = 57_420_144
CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS = 53_552_368
CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS = 1_933_888
CURRENT_LFM2_CARRIER_TARGET_ROWS = 4_096
CURRENT_LFM2_CARRIER_HIDDEN_SIZE = 816
CURRENT_LFM2_CARRIER_COMPACTION_RATIO = (
    CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS
    / CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
)
CARRIER_CHILD_PACKAGE_SCHEMA = "neural-forge-lfm2-carrier-child-package-v1"
CARRIER_CHILD_PACKAGE_RECEIPT = "carrier-child-package-receipt.json"


class Lfm2CarrierChildPackageError(RuntimeError):
    """A carrier registration, package, loader, or count is invalid."""


@dataclass(frozen=True)
class Lfm2CarrierCountedAbiProfile:
    """One sealed carrier counted-ABI tuple, admitted or refused as a whole.

    The declared compaction ratio, the counted target total, the dense body
    remainder, and the residual width are *one* fact about a child, not four
    independent knobs.  A request that pairs one profile's ratio with another
    profile's budget describes no child this lane can build, so the tuple is
    matched coherently and a partial match is a refusal.
    """

    profile_id: str
    compression_ratio: float
    counted_target_total_parameter_elements: int
    counted_dense_body_parameter_elements: int
    hidden_size: int

    def matches(
        self,
        *,
        compression_ratio: float | None,
        counted_target_total_parameter_elements: int,
        counted_dense_body_parameter_elements: int,
        hidden_size: int,
    ) -> bool:
        return bool(
            compression_ratio is not None
            and float(compression_ratio) == self.compression_ratio
            and counted_target_total_parameter_elements
            == self.counted_target_total_parameter_elements
            and counted_dense_body_parameter_elements
            == self.counted_dense_body_parameter_elements
            and hidden_size == self.hidden_size
        )

    def describe(self) -> str:
        return (
            f"ratio {self.compression_ratio:g} / counted total "
            f"{self.counted_target_total_parameter_elements:,} / dense body "
            f"{self.counted_dense_body_parameter_elements:,} / hidden "
            f"{self.hidden_size}"
        )


# The proven 4x lane.  Every published carrier child to date is this tuple.
LFM2_CARRIER_EXACT_4X_COUNTED_ABI = Lfm2CarrierCountedAbiProfile(
    profile_id="exact-4x-h816",
    compression_ratio=4.0,
    counted_target_total_parameter_elements=(
        CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
    ),
    counted_dense_body_parameter_elements=(
        CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS
    ),
    hidden_size=CURRENT_LFM2_CARRIER_HIDDEN_SIZE,
)
# The exact-2x ceiling rung derived in NOTES-W13T §2.  229,693,184 is even, so
# the half is exact with no walk-down; the two route channels are a fixed
# subtraction rather than a fraction of the budget, so the body is the same
# 2 x 1,933,888 short; and H816 carries an exact operator/MLP lattice at this
# budget (45 exact widths, 672..1024), so the rung moves one variable -- the
# budget -- against the 4x arm.
LFM2_CARRIER_EXACT_2X_COUNTED_ABI = Lfm2CarrierCountedAbiProfile(
    profile_id="exact-2x-h816",
    compression_ratio=2.0,
    counted_target_total_parameter_elements=114_846_592,
    counted_dense_body_parameter_elements=110_978_816,
    hidden_size=CURRENT_LFM2_CARRIER_HIDDEN_SIZE,
)
# The complete admitted set.  This is an allowlist, not a relaxation: anything
# that is not one of these tuples is refused exactly as before.
ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES: tuple[
    Lfm2CarrierCountedAbiProfile, ...
] = (
    LFM2_CARRIER_EXACT_4X_COUNTED_ABI,
    LFM2_CARRIER_EXACT_2X_COUNTED_ABI,
)


# How far a profile's *declared* ratio may sit from the counted ratio it
# labels.  The declared ratio is a rung name, not a derived quantity: the 4x
# lane declares 4.0 while 229,693,184 / 57,420,144 is 4.0002195745, because
# 57,420,144 is canonical for reasons upstream of this arithmetic
# (docs/CURRENT_FOLDING_GOAL.md; NOTES-W13T §2.2).  The exact-2x rung is
# genuinely exact.  A tenth of a percent admits the former and still refuses a
# profile whose label misdescribes its own budget.
_ADMITTED_CARRIER_RATIO_LABEL_TOLERANCE = 1e-3


def _seal_admitted_carrier_counted_abi() -> None:
    """Refuse to import an allowlist whose own arithmetic does not close."""

    identifiers = set()
    for profile in ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES:
        if profile.profile_id in identifiers:
            raise Lfm2CarrierChildPackageError(
                "Admitted carrier ABI profile identifiers must be unique."
            )
        identifiers.add(profile.profile_id)
        routes = (
            profile.counted_target_total_parameter_elements
            - profile.counted_dense_body_parameter_elements
        )
        counted_ratio = (
            CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS
            / profile.counted_target_total_parameter_elements
        )
        if (
            profile.compression_ratio <= 0.0
            or profile.counted_dense_body_parameter_elements <= 0
            or profile.hidden_size != CURRENT_LFM2_CARRIER_HIDDEN_SIZE
            or routes != 2 * CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS
            or abs(counted_ratio - profile.compression_ratio)
            > _ADMITTED_CARRIER_RATIO_LABEL_TOLERANCE
            * profile.compression_ratio
        ):
            raise Lfm2CarrierChildPackageError(
                "Admitted carrier ABI profile does not close on the sealed "
                f"229,693,184-element source: {profile.describe()}."
            )


_seal_admitted_carrier_counted_abi()


def describe_admitted_lfm2_carrier_counted_abi() -> str:
    """Render every admitted tuple for a fail-closed refusal message."""

    return "; ".join(
        profile.describe()
        for profile in ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES
    )


def admitted_lfm2_carrier_counted_abi_profile(
    *,
    compression_ratio: float | None,
    counted_target_total_parameter_elements: int,
    counted_dense_body_parameter_elements: int,
    hidden_size: int,
) -> Lfm2CarrierCountedAbiProfile | None:
    """Return the one admitted profile this request *entirely* matches."""

    for profile in ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES:
        if profile.matches(
            compression_ratio=compression_ratio,
            counted_target_total_parameter_elements=(
                counted_target_total_parameter_elements
            ),
            counted_dense_body_parameter_elements=(
                counted_dense_body_parameter_elements
            ),
            hidden_size=hidden_size,
        ):
            return profile
    return None


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CarrierChildPackageError(f"{name} must be a lowercase SHA-256.")
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or "://" in str(value):
        raise Lfm2CarrierChildPackageError(f"{name} must be an ordinary local path.")
    return Path(value).expanduser().absolute()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CarrierChildPackageError(f"{name} must be a positive integer.")
    return value


def _ordinary_child_directory(value: Any) -> Path:
    path = _path(value, "carrier child directory")
    try:
        root = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Lfm2CarrierChildPackageError(
            "Carrier child directory is missing."
        ) from exc
    if path.is_symlink() or not root.is_dir():
        raise Lfm2CarrierChildPackageError(
            "Carrier child directory must be a real directory."
        )
    return root


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_receipt(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Lfm2CarrierChildPackageError(
                    f"Carrier child receipt repeats key {key!r}."
                )
            result[key] = value
        return result

    if (
        path.is_symlink()
        or not path.is_file()
        or getattr(path.stat(), "st_nlink", 1) != 1
    ):
        raise Lfm2CarrierChildPackageError(
            "Carrier child package receipt is missing or unsafe."
        )
    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except Lfm2CarrierChildPackageError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CarrierChildPackageError(
            "Carrier child package receipt is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(result, dict):
        raise Lfm2CarrierChildPackageError(
            "Carrier child package receipt must be an object."
        )
    return result


def _child_inventory(
    root: Path,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    records: list[dict[str, Any]] = []
    directories: list[str] = []
    try:
        entries = tuple(sorted(root.rglob("*"), key=lambda path: path.as_posix()))
    except OSError as exc:
        raise Lfm2CarrierChildPackageError(
            "Carrier child package tree could not be enumerated."
        ) from exc
    for path in entries:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise Lfm2CarrierChildPackageError(
                f"Carrier child package contains link-like path {relative!r}."
            )
        if path.is_dir():
            directories.append(relative)
            continue
        if not path.is_file() or getattr(path.stat(), "st_nlink", 1) != 1:
            raise Lfm2CarrierChildPackageError(
                f"Carrier child package file {relative!r} is unsafe."
            )
        if relative == CARRIER_CHILD_PACKAGE_RECEIPT:
            continue
        records.append(
            {
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    if not records:
        raise Lfm2CarrierChildPackageError("Carrier child package is empty.")
    return tuple(records), tuple(directories)


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2CarrierChildPackage:
    """One complete child-local inventory plus its route/lexicon identities."""

    root: Path
    receipt: Mapping[str, Any]
    routes: AuthenticatedPackagedLfm2CarrierRoutes
    semantic_source_lexicon: AuthenticatedLfm2SemanticSourceLexicon

    @property
    def package_sha256(self) -> str:
        return str(self.receipt["package_sha256"])


def load_authenticated_lfm2_carrier_child_package(
    child_directory: Path,
    *,
    child_model: Any | None = None,
    expected_package_sha256: str | None = None,
) -> AuthenticatedLfm2CarrierChildPackage:
    """Authenticate the exact child tree using only its sealed local receipt."""

    root = _ordinary_child_directory(child_directory)
    raw = _strict_receipt(root / CARRIER_CHILD_PACKAGE_RECEIPT)
    seal = raw.pop("package_sha256", None)
    if (
        raw.get("schema") != CARRIER_CHILD_PACKAGE_SCHEMA
        or raw.get("status") != "complete"
        or raw.get("standalone_child_local_runtime") is not True
        or raw.get("external_source_publication_required_at_load") is not False
        or raw.get("external_source_tokenizer_required_at_load") is not False
        or raw.get("continuous_parameter_count_credit_from_lexicon") is not False
        or not isinstance(seal, str)
        or seal != canonical_sha256(raw)
    ):
        raise Lfm2CarrierChildPackageError(
            "Carrier child package receipt or self-seal is invalid."
        )
    if expected_package_sha256 is not None and seal != _digest(
        expected_package_sha256,
        "expected carrier child package SHA-256",
    ):
        raise Lfm2CarrierChildPackageError(
            "Carrier child package belongs to a different sealed child."
        )
    records, directories = _child_inventory(root)
    if raw.get("child_file_records") != list(records) or raw.get(
        "child_directories"
    ) != list(directories):
        raise Lfm2CarrierChildPackageError(
            "Carrier child package tree differs from its sealed inventory."
        )
    base_records = [
        value for value in records if "/" not in str(value["relative_path"])
    ]
    if raw.get("base_root_file_records") != base_records:
        raise Lfm2CarrierChildPackageError(
            "Carrier child base checkpoint inventory is not authentic."
        )
    source_artifact_sha256 = _digest(
        raw.get("source_bridge_artifact_sha256"),
        "sealed source bridge artifact SHA-256",
    )
    semantic_sha256 = _digest(
        raw.get("semantic_source_tokenizer_sha256"),
        "sealed semantic-source tokenizer SHA-256",
    )
    target_rows = _positive_int(raw.get("target_rows"), "sealed target_rows")
    hidden_size = _positive_int(raw.get("hidden_size"), "sealed hidden_size")
    packaged = load_packaged_lfm2_carrier_routes(
        root,
        expected_source_artifact_sha256=source_artifact_sha256,
        child_model=child_model,
        expected_target_rows=target_rows,
        expected_hidden_size=hidden_size,
    )
    lexicon = load_packaged_lfm2_semantic_source_lexicon(
        root,
        expected_tokenizer_semantics_sha256=semantic_sha256,
    )
    if (
        raw.get("carrier_route_package_sha256") != packaged.package_sha256
        or raw.get("semantic_source_lexicon_package_sha256")
        != lexicon.package_sha256
        or raw.get("route_allocations")
        != [value.as_dict() for value in packaged.allocations]
        or raw.get("continuous_carrier_parameter_elements")
        != sum(value.parameter_elements for value in packaged.allocations)
    ):
        raise Lfm2CarrierChildPackageError(
            "Carrier child nested package identities or counts changed."
        )
    receipt = {**raw, "package_sha256": seal}
    return AuthenticatedLfm2CarrierChildPackage(
        root=root,
        receipt=MappingProxyType(receipt),
        routes=packaged,
        semantic_source_lexicon=lexicon,
    )


def seal_lfm2_carrier_child_package(
    child_directory: Path,
    *,
    child_model: Any,
    expected_source_artifact_sha256: str,
    expected_tokenizer_semantics_sha256: str,
) -> AuthenticatedLfm2CarrierChildPackage:
    """Seal base checkpoint files, routes, and lexicon into one child inventory."""

    root = _ordinary_child_directory(child_directory)
    destination = root / CARRIER_CHILD_PACKAGE_RECEIPT
    if destination.exists() or destination.is_symlink():
        raise Lfm2CarrierChildPackageError(
            "Carrier child package already has an outer receipt."
        )
    source_sha256 = _digest(
        expected_source_artifact_sha256,
        "source bridge artifact SHA-256",
    )
    semantic_sha256 = _digest(
        expected_tokenizer_semantics_sha256,
        "semantic-source tokenizer SHA-256",
    )
    packaged = load_packaged_lfm2_carrier_routes(
        root,
        expected_source_artifact_sha256=source_sha256,
        child_model=child_model,
    )
    lexicon = load_packaged_lfm2_semantic_source_lexicon(
        root,
        expected_tokenizer_semantics_sha256=semantic_sha256,
    )
    input_binding = packaged.receipt["input_table_binding"]
    shape = input_binding.get("shape") if isinstance(input_binding, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
    ):
        raise Lfm2CarrierChildPackageError(
            "Carrier route package has no concrete child table shape."
        )
    records, directories = _child_inventory(root)
    base_records = [
        value for value in records if "/" not in str(value["relative_path"])
    ]
    body = {
        "schema": CARRIER_CHILD_PACKAGE_SCHEMA,
        "status": "complete",
        "standalone_child_local_runtime": True,
        "external_source_publication_required_at_load": False,
        "external_source_tokenizer_required_at_load": False,
        "source_bridge_artifact_sha256": source_sha256,
        "semantic_source_tokenizer_sha256": semantic_sha256,
        "carrier_route_package_sha256": packaged.package_sha256,
        "semantic_source_lexicon_package_sha256": lexicon.package_sha256,
        "target_rows": int(shape[0]),
        "hidden_size": int(shape[1]),
        "route_allocations": [value.as_dict() for value in packaged.allocations],
        "continuous_carrier_parameter_elements": sum(
            value.parameter_elements for value in packaged.allocations
        ),
        "continuous_parameter_count_credit_from_lexicon": False,
        "base_root_file_records": base_records,
        "child_file_records": list(records),
        "child_directories": list(directories),
    }
    receipt = {**body, "package_sha256": canonical_sha256(body)}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".carrier-child-receipt.",
        dir=root,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return load_authenticated_lfm2_carrier_child_package(
        root,
        child_model=child_model,
        expected_package_sha256=str(receipt["package_sha256"]),
    )


class Lfm2CarrierChildPackageRegistration:
    """Authenticate one source publication and derive its exact counted plan."""

    __slots__ = (
        "publication_directory",
        "expected_artifact_sha256",
        "span_evidence_json",
        "expected_span_evidence_sha256",
        "expected_span_evidence_file_sha256",
        "expected_target_rows",
        "expected_hidden_size",
        "expected_input_route_elements",
        "expected_output_route_elements",
        "fingerprint",
    )

    def __init__(
        self,
        *,
        publication_directory: Path,
        expected_artifact_sha256: str,
        span_evidence_json: Path,
        expected_span_evidence_sha256: str,
        expected_span_evidence_file_sha256: str,
        expected_target_rows: int,
        expected_hidden_size: int,
        expected_input_route_elements: int,
        expected_output_route_elements: int,
    ) -> None:
        self.publication_directory = _path(
            publication_directory,
            "carrier publication directory",
        )
        self.expected_artifact_sha256 = _digest(
            expected_artifact_sha256,
            "carrier artifact SHA-256",
        )
        self.span_evidence_json = _path(
            span_evidence_json,
            "semantic span evidence",
        )
        self.expected_span_evidence_sha256 = _digest(
            expected_span_evidence_sha256,
            "semantic span evidence SHA-256",
        )
        self.expected_span_evidence_file_sha256 = _digest(
            expected_span_evidence_file_sha256,
            "semantic span evidence file SHA-256",
        )
        self.expected_target_rows = _positive_int(
            expected_target_rows,
            "expected_target_rows",
        )
        self.expected_hidden_size = _positive_int(
            expected_hidden_size,
            "expected_hidden_size",
        )
        self.expected_input_route_elements = _positive_int(
            expected_input_route_elements,
            "expected_input_route_elements",
        )
        self.expected_output_route_elements = _positive_int(
            expected_output_route_elements,
            "expected_output_route_elements",
        )
        self.fingerprint = canonical_sha256(self.as_dict())

    @classmethod
    def current_exact4x(
        cls,
        *,
        publication_directory: Path,
        expected_artifact_sha256: str,
        span_evidence_json: Path,
        expected_span_evidence_sha256: str,
        expected_span_evidence_file_sha256: str,
    ) -> "Lfm2CarrierChildPackageRegistration":
        return cls(
            publication_directory=publication_directory,
            expected_artifact_sha256=expected_artifact_sha256,
            span_evidence_json=span_evidence_json,
            expected_span_evidence_sha256=expected_span_evidence_sha256,
            expected_span_evidence_file_sha256=(
                expected_span_evidence_file_sha256
            ),
            expected_target_rows=CURRENT_LFM2_CARRIER_TARGET_ROWS,
            expected_hidden_size=CURRENT_LFM2_CARRIER_HIDDEN_SIZE,
            expected_input_route_elements=(
                CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS
            ),
            expected_output_route_elements=(
                CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "publication_directory": str(self.publication_directory),
            "expected_artifact_sha256": self.expected_artifact_sha256,
            "span_evidence_json": str(self.span_evidence_json),
            "expected_span_evidence_sha256": (
                self.expected_span_evidence_sha256
            ),
            "expected_span_evidence_file_sha256": (
                self.expected_span_evidence_file_sha256
            ),
            "expected_target_rows": self.expected_target_rows,
            "expected_hidden_size": self.expected_hidden_size,
            "expected_input_route_elements": self.expected_input_route_elements,
            "expected_output_route_elements": self.expected_output_route_elements,
        }

    def _authenticated(self) -> tuple[Any, dict[str, Any]]:
        try:
            publication = load_lfm2_carrier_bridge_publication(
                self.publication_directory,
                expected_artifact_sha256=self.expected_artifact_sha256,
            )
            spans = load_authenticated_lfm2_span_evidence(
                self.span_evidence_json,
                expected_file_sha256=self.expected_span_evidence_file_sha256,
                expected_evidence_sha256=self.expected_span_evidence_sha256,
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Lfm2CarrierChildPackageError(
                "Carrier source publication or span evidence is not authentic."
            ) from exc
        allocations = publication.allocations
        tensors = publication.tensors
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise Lfm2CarrierChildPackageError(
                "PyTorch is required for carrier child packaging."
            ) from exc
        semantic_rows = int(spans["donor_row_count"])
        direct = tensors["target_to_donor"]
        if (
            tuple(value.state_id for value in allocations)
            != (
                "semantic-carrier-route.c_in",
                "semantic-carrier-route.c_out",
            )
            or allocations[0].parameter_elements
            != self.expected_input_route_elements
            or allocations[1].parameter_elements
            != self.expected_output_route_elements
            or any(value.dtype != "float32" for value in allocations)
            or spans.get("target_row_count") != self.expected_target_rows
            or not isinstance(direct, torch.Tensor)
            or direct.dtype is not torch.int64
            or direct.ndim != 1
            or int(direct.numel()) != self.expected_target_rows
            or bool((direct < 0).any())
            or bool((direct >= semantic_rows).any())
            or int(tensors["input_offsets"].numel()) != semantic_rows + 1
            or int(tensors["output_offsets"].numel()) != semantic_rows + 1
            or int(tensors["target_table"].shape[0]) != self.expected_target_rows
        ):
            raise Lfm2CarrierChildPackageError(
                "Carrier source routes differ from their registered child ABI."
            )
        return publication, spans

    def verify(self) -> "Lfm2CarrierChildPackageRegistration":
        self._authenticated()
        if canonical_sha256(self.as_dict()) != self.fingerprint:
            raise Lfm2CarrierChildPackageError("Carrier registration changed.")
        return self

    def allocations(
        self,
    ) -> tuple[ContinuousChildStateAllocation, ContinuousChildStateAllocation]:
        publication, _spans = self._authenticated()
        return publication.allocations

    def counted_plan(
        self,
        *,
        parent_parameter_elements: int,
        target_parameter_elements: int,
    ) -> CountedChildStatePlan:
        return build_counted_child_state_plan(
            parent_unique_parameters=parent_parameter_elements,
            target_total_parameter_elements=target_parameter_elements,
            continuous_allocations=self.allocations(),
        )

    def admitted_counted_plan(
        self,
        profile: Lfm2CarrierCountedAbiProfile,
    ) -> CountedChildStatePlan:
        """Build the counted plan for one admitted carrier ABI tuple.

        The profile object itself must be a member of the sealed allowlist --
        an equal-looking tuple assembled by a caller is not admitted -- and the
        plan the live routes produce must still reserve that profile's exact
        dense body.  Both checks are the fail-closed re-pin the 4x lane always
        had; only the number of admitted answers changed.
        """

        if not any(
            profile is admitted
            for admitted in ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES
        ):
            raise Lfm2CarrierChildPackageError(
                "Carrier counted plans are sealed to the admitted carrier ABI "
                f"({describe_admitted_lfm2_carrier_counted_abi()})."
            )
        plan = self.counted_plan(
            parent_parameter_elements=CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
            target_parameter_elements=(
                profile.counted_target_total_parameter_elements
            ),
        )
        if plan.dense_model_parameter_elements != (
            profile.counted_dense_body_parameter_elements
        ):
            raise Lfm2CarrierChildPackageError(
                "Admitted carrier plan did not reserve its exact H816 dense "
                f"body ({profile.describe()})."
            )
        return plan

    def current_exact4x_counted_plan(self) -> CountedChildStatePlan:
        return self.admitted_counted_plan(LFM2_CARRIER_EXACT_4X_COUNTED_ABI)

    def bind_source(self, child_model: Any) -> Any:
        _publication, spans = self._authenticated()
        try:
            return bind_lfm2_carrier_bridge_publication_to_child(
                child_model,
                self.publication_directory,
                expected_artifact_sha256=self.expected_artifact_sha256,
                span_evidence=spans,
                expected_target_rows=self.expected_target_rows,
                expected_hidden_size=self.expected_hidden_size,
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Lfm2CarrierChildPackageError(
                "Source carrier routes could not bind to the live child."
            ) from exc

    def package(self, child_directory: Path, child_model: Any) -> Any:
        _publication, spans = self._authenticated()
        try:
            return package_lfm2_carrier_routes_with_child(
                self.publication_directory,
                child_directory,
                child_model,
                expected_artifact_sha256=self.expected_artifact_sha256,
                span_evidence=spans,
                expected_target_rows=self.expected_target_rows,
                expected_hidden_size=self.expected_hidden_size,
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Lfm2CarrierChildPackageError(
                "Carrier routes could not be packaged with the child."
            ) from exc

    def load_packaged(
        self,
        child_directory: Path,
        child_model: Any,
    ) -> AuthenticatedPackagedLfm2CarrierRoutes:
        try:
            packaged = load_packaged_lfm2_carrier_routes(
                child_directory,
                expected_source_artifact_sha256=self.expected_artifact_sha256,
                child_model=child_model,
                expected_target_rows=self.expected_target_rows,
                expected_hidden_size=self.expected_hidden_size,
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Lfm2CarrierChildPackageError(
                "Packaged carrier routes failed fresh-child authentication."
            ) from exc
        # A deployed child must be loadable from its own sealed package.  Do
        # not reopen the build-time carrier publication here: the package
        # loader already recomputes both coefficient hashes from the sealed
        # tensors and binds their provenance to ``expected_artifact_sha256``.
        expected = (
            (
                "semantic-carrier-route.c_in",
                self.expected_input_route_elements,
            ),
            (
                "semantic-carrier-route.c_out",
                self.expected_output_route_elements,
            ),
        )
        if any(
            allocation.state_id != state_id
            or allocation.parameter_elements != parameter_elements
            or allocation.artifact_sha256 != self.expected_artifact_sha256
            or allocation.dtype != "float32"
            for allocation, (state_id, parameter_elements) in zip(
                packaged.allocations,
                expected,
                strict=True,
            )
        ):
            raise Lfm2CarrierChildPackageError(
                "Packaged carrier allocations differ from the registered ABI."
            )
        return packaged

    def bind_packaged(
        self,
        child_directory: Path,
        child_model: Any,
    ) -> tuple[AuthenticatedPackagedLfm2CarrierRoutes, Any]:
        packaged = self.load_packaged(child_directory, child_model)
        try:
            runtime = bind_packaged_lfm2_carrier_routes(
                child_model,
                packaged,
                expected_target_rows=self.expected_target_rows,
                expected_hidden_size=self.expected_hidden_size,
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Lfm2CarrierChildPackageError(
                "Packaged carrier routes failed live binding."
            ) from exc
        return packaged, runtime


class Lfm2CarrierChildCheckpointIO:
    """Wrap one local model IO with route+lexicon packaging and public runtime."""

    def __init__(
        self,
        *,
        base: Any,
        registration: Lfm2CarrierChildPackageRegistration,
        semantic_source_tokenizer_directory: Path,
        semantic_source_tokenizer_sha256: str,
        in_memory_semantic_source_tokenizer: Any,
        in_memory_target_tokenizer: Any,
        target_tokenizer_loader: Callable[[Path], Any] = (
            load_lfm2_direct_local_tokenizer
        ),
    ) -> None:
        if not callable(getattr(base, "save", None)) or not callable(
            getattr(base, "load", None)
        ):
            raise Lfm2CarrierChildPackageError(
                "base must expose local save() and load()."
            )
        if not isinstance(registration, Lfm2CarrierChildPackageRegistration):
            raise Lfm2CarrierChildPackageError(
                "registration has the wrong type."
            )
        if not callable(target_tokenizer_loader):
            raise Lfm2CarrierChildPackageError(
                "target_tokenizer_loader must be callable."
            )
        self.base = base
        self.registration = registration.verify()
        self.semantic_source_tokenizer_directory = _path(
            semantic_source_tokenizer_directory,
            "semantic-source tokenizer directory",
        )
        self.semantic_source_tokenizer_sha256 = _digest(
            semantic_source_tokenizer_sha256,
            "semantic-source tokenizer SHA-256",
        )
        self.in_memory_semantic_source_tokenizer = (
            in_memory_semantic_source_tokenizer
        )
        self.in_memory_target_tokenizer = in_memory_target_tokenizer
        self.target_tokenizer_loader = target_tokenizer_loader
        self.expected_child_package_sha256: str | None = None
        self._portable_runtime_model: Any | None = None
        self._portable_runtime: Any | None = None
        base_saver_id = str(getattr(base, "saver_id", "local-saver"))
        base_loader_id = str(getattr(base, "loader_id", "local-loader"))
        suffix = registration.fingerprint[:16]
        self.saver_id = f"{base_saver_id}.carrier-child-v1.{suffix}"
        self.loader_id = f"{base_loader_id}.carrier-child-v1.{suffix}"
        self.resource_fingerprint = canonical_sha256(
            {
                "base_resource_fingerprint": str(
                    getattr(base, "resource_fingerprint", "unsealed-base")
                ),
                "carrier_registration_fingerprint": registration.fingerprint,
                "semantic_source_tokenizer_sha256": (
                    self.semantic_source_tokenizer_sha256
                ),
                "saver_id": self.saver_id,
                "loader_id": self.loader_id,
            }
        )

    @classmethod
    def from_child_directory(
        cls,
        *,
        base: Any,
        child_directory: Path,
        target_tokenizer_loader: Callable[[Path], Any] = (
            load_lfm2_direct_local_tokenizer
        ),
    ) -> "Lfm2CarrierChildCheckpointIO":
        """Construct a fresh-process loader from child-local sealed state only."""

        if not callable(getattr(base, "load", None)):
            raise Lfm2CarrierChildPackageError("base must expose local load().")
        if not callable(target_tokenizer_loader):
            raise Lfm2CarrierChildPackageError(
                "target_tokenizer_loader must be callable."
            )
        package = load_authenticated_lfm2_carrier_child_package(child_directory)
        value = object.__new__(cls)
        value.base = base
        value.registration = None
        value.semantic_source_tokenizer_directory = None
        value.semantic_source_tokenizer_sha256 = str(
            package.receipt["semantic_source_tokenizer_sha256"]
        )
        value.in_memory_semantic_source_tokenizer = None
        value.in_memory_target_tokenizer = None
        value.target_tokenizer_loader = target_tokenizer_loader
        value.expected_child_package_sha256 = package.package_sha256
        value._portable_runtime_model = None
        value._portable_runtime = None
        base_loader_id = str(getattr(base, "loader_id", "local-loader"))
        suffix = package.package_sha256[:16]
        value.saver_id = f"packaged-child-read-only.{suffix}"
        value.loader_id = f"{base_loader_id}.carrier-child-v1.{suffix}"
        value.resource_fingerprint = canonical_sha256(
            {
                "base_resource_fingerprint": str(
                    getattr(base, "resource_fingerprint", "unsealed-base")
                ),
                "carrier_child_package_sha256": package.package_sha256,
                "semantic_source_tokenizer_sha256": (
                    value.semantic_source_tokenizer_sha256
                ),
                "loader_id": value.loader_id,
            }
        )
        return value

    def register_in_memory(self, model: Any) -> Any:
        if self.registration is None:
            raise Lfm2CarrierChildPackageError(
                "A child-local read-only loader cannot bind build-time source state."
            )
        return install_lfm2_semantic_step_generate(
            model,
            semantic_source_tokenizer=self.in_memory_semantic_source_tokenizer,
            target_tokenizer=self.in_memory_target_tokenizer,
            runtime_factory=self.registration.bind_source,
        )

    def save(self, model: Any, destination: Path) -> None:
        if self.registration is None:
            raise Lfm2CarrierChildPackageError(
                "A child-local read-only loader cannot save a new fold."
            )
        self.base.save(model, destination)
        package_lfm2_semantic_source_lexicon(
            self.semantic_source_tokenizer_directory,
            destination,
            expected_tokenizer_semantics_sha256=(
                self.semantic_source_tokenizer_sha256
            ),
        )
        self.registration.package(destination, model)
        seal_lfm2_carrier_child_package(
            destination,
            child_model=model,
            expected_source_artifact_sha256=(
                self.registration.expected_artifact_sha256
            ),
            expected_tokenizer_semantics_sha256=(
                self.semantic_source_tokenizer_sha256
            ),
        )

    def load(self, source: Path) -> Any:
        package_before_load = load_authenticated_lfm2_carrier_child_package(
            source,
            expected_package_sha256=self.expected_child_package_sha256,
        )
        model = self.base.load(source)
        package = load_authenticated_lfm2_carrier_child_package(
            source,
            child_model=model,
            expected_package_sha256=package_before_load.package_sha256,
        )
        target_tokenizer = self.target_tokenizer_loader(Path(source))
        try:
            runtime = bind_packaged_lfm2_carrier_routes(
                model,
                package.routes,
                expected_target_rows=int(package.receipt["target_rows"]),
                expected_hidden_size=int(package.receipt["hidden_size"]),
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Lfm2CarrierChildPackageError(
                "Child-local carrier routes failed live binding."
            ) from exc
        return install_lfm2_semantic_step_generate(
            model,
            semantic_source_tokenizer=package.semantic_source_lexicon.tokenizer,
            target_tokenizer=target_tokenizer,
            runtime=runtime,
        )

    def load_authenticated_portable(
        self,
        source: Path,
        *,
        runtime_device: Any,
        runtime_dtype: Any,
    ) -> tuple[Any, Lfm2PortableCarrierRuntimeReceipt]:
        """Load/authenticate canonical FP32 state before runtime relocation."""

        self._portable_runtime_model = None
        self._portable_runtime = None
        package_before_load = load_authenticated_lfm2_carrier_child_package(
            source,
            expected_package_sha256=self.expected_child_package_sha256,
        )
        canonical_model = self.base.load(source)
        canonical_package = load_authenticated_lfm2_carrier_child_package(
            source,
            child_model=canonical_model,
            expected_package_sha256=package_before_load.package_sha256,
        )
        try:
            relocated_model, runtime, relocation = (
                relocate_and_bind_packaged_lfm2_carrier_routes(
                    canonical_model,
                    canonical_package.routes,
                    provenance_resource_fingerprint=(
                        self.resource_fingerprint
                    ),
                    runtime_device=runtime_device,
                    runtime_dtype=runtime_dtype,
                    expected_target_rows=int(
                        canonical_package.receipt["target_rows"]
                    ),
                    expected_hidden_size=int(
                        canonical_package.receipt["hidden_size"]
                    ),
                )
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Lfm2CarrierChildPackageError(
                "Canonical carrier child failed authenticated portable "
                "runtime relocation."
            ) from exc
        package_after_load = load_authenticated_lfm2_carrier_child_package(
            source,
            child_model=canonical_model,
            expected_package_sha256=canonical_package.package_sha256,
        )
        relocation = replace(
            relocation,
            carrier_child_package_sha256=package_after_load.package_sha256,
        )
        relocation_body = relocation.as_dict()
        if (
            package_after_load.routes.package_sha256
            != relocation_body["carrier_route_package_sha256"]
            or package_after_load.package_sha256
            != relocation_body["carrier_child_package_sha256"]
            or package_after_load.routes.allocations
            != relocation.route_allocations
            or relocation_body["canonical_artifact_rewritten"] is not False
            or relocation_body["canonical_disk_artifact_mutated"] is not False
            or relocation_body["canonical_route_package_rewritten"] is not False
            or relocation_body["executable_forward_smoke_passed"] is not True
            or relocation_body["dtype_or_device_parameter_compaction_credit"]
            != 0
        ):
            raise Lfm2CarrierChildPackageError(
                "Portable carrier relocation changed its sealed child package "
                "or counted-state contract."
            )
        target_tokenizer = self.target_tokenizer_loader(Path(source))
        model = install_lfm2_semantic_step_generate(
            relocated_model,
            semantic_source_tokenizer=(
                package_after_load.semantic_source_lexicon.tokenizer
            ),
            target_tokenizer=target_tokenizer,
            runtime=runtime,
        )
        self._portable_runtime_model = model
        self._portable_runtime = runtime
        return model, relocation

    def observe_authenticated_portable_runtime_state(
        self,
        model: Any,
    ) -> Mapping[str, Any]:
        """Fingerprint fixed route tensors for this exact returned model."""

        if (
            model is not self._portable_runtime_model
            or self._portable_runtime is None
        ):
            raise Lfm2CarrierChildPackageError(
                "Portable route-state observation requires this loader's "
                "latest authenticated runtime model."
            )
        try:
            return MappingProxyType(
                fingerprint_lfm2_carrier_runtime_owned_tensor_state(
                    self._portable_runtime
                )
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Lfm2CarrierChildPackageError(
                "Portable runtime owned route state could not be sealed."
            ) from exc

    def observe_child_directory(
        self,
        child_directory: Path,
    ) -> Sequence[ContinuousChildStateAllocation]:
        model = self.load(child_directory)
        package = load_authenticated_lfm2_carrier_child_package(
            child_directory,
            child_model=model,
            expected_package_sha256=self.expected_child_package_sha256,
        )
        return package.routes.allocations

    def observe_published(
        self,
        result: Any,
        _plan: CountedChildStatePlan,
    ) -> Sequence[ContinuousChildStateAllocation]:
        artifact = getattr(getattr(result, "published", None), "artifact", None)
        root = _path(getattr(artifact, "path", None), "published artifact path")
        child = root / "child"
        if not child.is_dir() or child.is_symlink():
            raise Lfm2CarrierChildPackageError(
                "Published artifact does not contain one real child directory."
            )
        return self.observe_child_directory(child)


def run_lfm2_resumable_review_from_packaged_child(
    *,
    base: Any,
    child_directory: Path,
    prompts: Sequence[Mapping[str, Any]],
    checkpoint_directory: Path,
    published_child_artifact_sha256: str,
    target_tokenizer_loader: Callable[[Path], Any] = (
        load_lfm2_direct_local_tokenizer
    ),
    maximum_prompt_tokens: int = 8_192,
    initial_maximum_new_tokens: int = 1_024,
    retry_maximum_new_tokens: int = 2_048,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fresh-load a child-local package and resume its external review rows."""

    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=base,
        child_directory=child_directory,
        target_tokenizer_loader=target_tokenizer_loader,
    )
    model = loader.load(child_directory)
    return run_lfm2_resumable_adaptive_carrier_manual_review(
        model,
        prompts,
        checkpoint_directory=checkpoint_directory,
        published_child_artifact_sha256=published_child_artifact_sha256,
        maximum_prompt_tokens=maximum_prompt_tokens,
        initial_maximum_new_tokens=initial_maximum_new_tokens,
        retry_maximum_new_tokens=retry_maximum_new_tokens,
        progress_callback=progress_callback,
    )


__all__ = [
    "ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES",
    "AuthenticatedLfm2CarrierChildPackage",
    "CARRIER_CHILD_PACKAGE_RECEIPT",
    "CARRIER_CHILD_PACKAGE_SCHEMA",
    "CURRENT_LFM2_CARRIER_COMPACTION_RATIO",
    "CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS",
    "CURRENT_LFM2_CARRIER_HIDDEN_SIZE",
    "CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS",
    "CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS",
    "CURRENT_LFM2_CARRIER_TARGET_ROWS",
    "CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS",
    "LFM2_CARRIER_EXACT_2X_COUNTED_ABI",
    "LFM2_CARRIER_EXACT_4X_COUNTED_ABI",
    "Lfm2CarrierChildCheckpointIO",
    "Lfm2CarrierChildPackageError",
    "Lfm2CarrierChildPackageRegistration",
    "Lfm2CarrierCountedAbiProfile",
    "admitted_lfm2_carrier_counted_abi_profile",
    "describe_admitted_lfm2_carrier_counted_abi",
    "load_authenticated_lfm2_carrier_child_package",
    "run_lfm2_resumable_review_from_packaged_child",
    "seal_lfm2_carrier_child_package",
]
