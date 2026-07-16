#!/usr/bin/env python3
"""Authenticate the one historical v4 preregistration after source evolution.

The live v4 authenticator remains the authority.  This module first invokes it
unchanged.  Its historical fallback exists only for the exact preregistration
that launched the July 15 v4 tournament and only when the sole live source
change is the behavior trainer.  The fallback verifies a repository-owned copy
of that trainer's exact historical bytes before temporarily presenting the
preregistered implementation map to the full transitive authenticator.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gzip
import hashlib
import re
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import run_qwen_exact_target_native_guided_gud_candidate_tournament as tournament


RECEIPT_SCHEMA = "neural-forge-historical-v4-preregistration-auth-receipt-v1"
V4_PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-gud-candidate-tournament-"
    "preregistration-v1"
)
HISTORICAL_PREREGISTRATION_SHA256 = (
    "8a05c9fbb8407475ccad47d4081a0f6e5feeadd883a778e69011216027d97f09"
)
HISTORICAL_SOURCE_COMMIT = "3181839f64b8421dcb448b3cec2859c8fb5ff905"
HISTORICAL_SOURCE_REPOSITORY_PATH = (
    "src/neural_forge/hf_target_native_behavior_distillation.py"
)
HISTORICAL_SOURCE_GIT_BLOB_OID = "d9be731b8cac0100937b70fb7c7957a0230bd4b8"
HISTORICAL_SOURCE_SHA256 = (
    "88e22e62c500afff70602fa1c4d2f11de40759265737e569115d1f0ac66e4fa5"
)
HISTORICAL_SOURCE_SIZE = 234_133
HISTORICAL_SOURCE_ASSET_RELATIVE_PATH = (
    "historical_assets/hf_target_native_behavior_distillation_3181839.py.gz"
)
HISTORICAL_SOURCE_ASSET_SHA256 = (
    "75252c3f07408035bd552f99582fc8564eb0015bfeb9528e3224cdc1bdb488f4"
)
HISTORICAL_SOURCE_ASSET_SIZE = 38_888
_ALLOWED_HISTORICAL_DELTA = ("behavior_trainer_sha256",)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMPLEMENTATION_OVERRIDE_LOCK = threading.RLock()


class HistoricalV4PreregistrationAuthenticationError(RuntimeError):
    """The bounded historical v4 authentication gate failed closed."""


@dataclasses.dataclass(frozen=True, slots=True)
class HistoricalSourceProvenance:
    """Immutable provenance for the exact source bytes used by the fallback."""

    commit: str
    repository_path: str
    git_blob_oid: str
    source_sha256: str
    source_size: int
    asset_relative_path: str
    asset_sha256: str
    asset_size: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class V4PreregistrationAuthenticationReceipt:
    """Immutable receipt describing which implementation boundary authenticated."""

    schema: str
    mode: Literal["current-live", "exact-historical-source"]
    preregistration: str
    preregistration_sha256: str
    preregistration_schema: str
    evaluation_contract_sha256: str
    implementation_contract_sha256: str
    live_implementation_contract_sha256: str
    implementation_delta_keys: tuple[str, ...]
    full_transitive_authenticator: str
    implementation_override_used: bool
    implementation_override_restored: bool
    historical_source: HistoricalSourceProvenance | None

    def as_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["implementation_delta_keys"] = list(self.implementation_delta_keys)
        return value


@dataclasses.dataclass(frozen=True, slots=True)
class AuthenticatedV4Preregistration:
    """The existing full contract paired with immutable authentication provenance."""

    contract: tournament.AuthenticatedGuidedGUDCandidateTournament
    receipt: V4PreregistrationAuthenticationReceipt


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Mapping[str, str]) -> str:
    return tournament.canonical_sha256(dict(value))


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HistoricalV4PreregistrationAuthenticationError(
            f"{name} must be a lowercase SHA-256."
        )
    return value


def _implementation_map(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise HistoricalV4PreregistrationAuthenticationError(
            f"{name} must be a non-empty implementation mapping."
        )
    implementation: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str):
            raise HistoricalV4PreregistrationAuthenticationError(
                f"{name} contains a non-string key."
            )
        implementation[key] = _validate_sha256(digest, f"{name}.{key}")
    return implementation


def _load_preregistration(
    path: Path | str,
    expected_sha256: str,
) -> tuple[Path, Mapping[str, Any], dict[str, str]]:
    supplied_path = Path(path).expanduser()
    expected = _validate_sha256(expected_sha256, "preregistration_sha256")
    if supplied_path.is_symlink() or not supplied_path.is_file():
        raise HistoricalV4PreregistrationAuthenticationError(
            "V4 preregistration is absent or is a symbolic link."
        )
    preregistration = supplied_path.resolve()
    if _sha256_bytes(preregistration.read_bytes()) != expected:
        raise HistoricalV4PreregistrationAuthenticationError(
            "V4 preregistration changed from its supplied SHA-256."
        )
    try:
        raw = tournament.stage._strict_json(preregistration)
    except Exception as error:
        raise HistoricalV4PreregistrationAuthenticationError(
            "V4 preregistration is not strict JSON."
        ) from error
    if tournament.PREREGISTRATION_SCHEMA != V4_PREREGISTRATION_SCHEMA:
        raise HistoricalV4PreregistrationAuthenticationError(
            "Live v4 preregistration schema constant changed."
        )
    if not isinstance(raw, Mapping) or raw.get("schema") != V4_PREREGISTRATION_SCHEMA:
        raise HistoricalV4PreregistrationAuthenticationError(
            "V4 preregistration schema changed."
        )
    implementation = _implementation_map(
        raw.get("implementation"),
        "preregistration.implementation",
    )
    return preregistration, raw, implementation


def _verify_historical_source_asset() -> HistoricalSourceProvenance:
    supplied_asset = (
        Path(__file__).resolve().parent / HISTORICAL_SOURCE_ASSET_RELATIVE_PATH
    )
    if supplied_asset.is_symlink() or not supplied_asset.is_file():
        raise HistoricalV4PreregistrationAuthenticationError(
            "Historical behavior-trainer source asset is absent or unsafe."
        )
    asset = supplied_asset.resolve()
    packed = asset.read_bytes()
    if (
        len(packed) != HISTORICAL_SOURCE_ASSET_SIZE
        or _sha256_bytes(packed) != HISTORICAL_SOURCE_ASSET_SHA256
    ):
        raise HistoricalV4PreregistrationAuthenticationError(
            "Historical behavior-trainer source asset failed its immutable pin."
        )
    try:
        source = gzip.decompress(packed)
    except (EOFError, OSError) as error:
        raise HistoricalV4PreregistrationAuthenticationError(
            "Historical behavior-trainer source asset cannot be decompressed."
        ) from error
    if (
        len(source) != HISTORICAL_SOURCE_SIZE
        or _sha256_bytes(source) != HISTORICAL_SOURCE_SHA256
    ):
        raise HistoricalV4PreregistrationAuthenticationError(
            "Historical behavior-trainer source bytes failed their exact pin."
        )
    return HistoricalSourceProvenance(
        commit=HISTORICAL_SOURCE_COMMIT,
        repository_path=HISTORICAL_SOURCE_REPOSITORY_PATH,
        git_blob_oid=HISTORICAL_SOURCE_GIT_BLOB_OID,
        source_sha256=HISTORICAL_SOURCE_SHA256,
        source_size=HISTORICAL_SOURCE_SIZE,
        asset_relative_path=HISTORICAL_SOURCE_ASSET_RELATIVE_PATH,
        asset_sha256=HISTORICAL_SOURCE_ASSET_SHA256,
        asset_size=HISTORICAL_SOURCE_ASSET_SIZE,
    )


@contextlib.contextmanager
def _temporary_implementation_contract(
    implementation: Mapping[str, str],
) -> Iterator[None]:
    frozen_items = tuple(sorted(implementation.items()))

    def historical_implementation_contract() -> dict[str, str]:
        return dict(frozen_items)

    with _IMPLEMENTATION_OVERRIDE_LOCK:
        original = tournament._implementation_contract
        tournament._implementation_contract = historical_implementation_contract
        try:
            yield
        finally:
            tournament._implementation_contract = original


def _verify_returned_contract(
    contract: Any,
    *,
    preregistration: Path,
    preregistration_sha256: str,
    implementation: Mapping[str, str],
) -> tournament.AuthenticatedGuidedGUDCandidateTournament:
    if not isinstance(
        contract,
        tournament.AuthenticatedGuidedGUDCandidateTournament,
    ):
        raise HistoricalV4PreregistrationAuthenticationError(
            "Full v4 authenticator returned the wrong contract type."
        )
    if (
        contract.path != preregistration
        or contract.file_sha256 != preregistration_sha256
        or dict(contract.implementation) != dict(implementation)
        or _SHA256.fullmatch(contract.evaluation_contract_sha256) is None
    ):
        raise HistoricalV4PreregistrationAuthenticationError(
            "Full v4 authenticator returned a contract outside the requested boundary."
        )
    return contract


def _receipt(
    *,
    mode: Literal["current-live", "exact-historical-source"],
    contract: tournament.AuthenticatedGuidedGUDCandidateTournament,
    preregistration_schema: str,
    preregistration_implementation: Mapping[str, str],
    live_implementation: Mapping[str, str],
    delta_keys: tuple[str, ...],
    override_used: bool,
    provenance: HistoricalSourceProvenance | None,
) -> V4PreregistrationAuthenticationReceipt:
    return V4PreregistrationAuthenticationReceipt(
        schema=RECEIPT_SCHEMA,
        mode=mode,
        preregistration=str(contract.path),
        preregistration_sha256=contract.file_sha256,
        preregistration_schema=preregistration_schema,
        evaluation_contract_sha256=contract.evaluation_contract_sha256,
        implementation_contract_sha256=_canonical_sha256(
            preregistration_implementation
        ),
        live_implementation_contract_sha256=_canonical_sha256(live_implementation),
        implementation_delta_keys=delta_keys,
        full_transitive_authenticator=(
            "run_qwen_exact_target_native_guided_gud_candidate_tournament."
            "authenticate_guided_gud_candidate_preregistration"
        ),
        implementation_override_used=override_used,
        implementation_override_restored=True,
        historical_source=provenance,
    )


def authenticate_guided_gud_candidate_preregistration_with_history(
    path: Path | str,
    expected_sha256: str,
) -> AuthenticatedV4Preregistration:
    """Run live authentication, then the one exact historical fallback if needed."""

    preregistration, raw, preregistered = _load_preregistration(
        path,
        expected_sha256,
    )
    expected = _validate_sha256(expected_sha256, "preregistration_sha256")
    live = _implementation_map(
        tournament._implementation_contract(),
        "live implementation",
    )
    delta_keys = tuple(
        sorted(
            key
            for key in set(preregistered) | set(live)
            if preregistered.get(key) != live.get(key)
        )
    )
    authenticator = tournament.authenticate_guided_gud_candidate_preregistration
    try:
        contract = authenticator(preregistration, expected)
    except tournament.GuidedGUDCandidateTournamentError as live_error:
        if expected != HISTORICAL_PREREGISTRATION_SHA256:
            raise HistoricalV4PreregistrationAuthenticationError(
                "Historical fallback is not authorized for this preregistration."
            ) from live_error
        if delta_keys != _ALLOWED_HISTORICAL_DELTA:
            raise HistoricalV4PreregistrationAuthenticationError(
                "Historical v4 implementation delta is not solely behavior_trainer_sha256."
            ) from live_error
        if preregistered.get("behavior_trainer_sha256") != HISTORICAL_SOURCE_SHA256:
            raise HistoricalV4PreregistrationAuthenticationError(
                "Historical preregistration is not bound to the preserved trainer source."
            ) from live_error
        provenance = _verify_historical_source_asset()
        with _temporary_implementation_contract(preregistered):
            contract = authenticator(preregistration, expected)
        contract = _verify_returned_contract(
            contract,
            preregistration=preregistration,
            preregistration_sha256=expected,
            implementation=preregistered,
        )
        return AuthenticatedV4Preregistration(
            contract=contract,
            receipt=_receipt(
                mode="exact-historical-source",
                contract=contract,
                preregistration_schema=str(raw["schema"]),
                preregistration_implementation=preregistered,
                live_implementation=live,
                delta_keys=delta_keys,
                override_used=True,
                provenance=provenance,
            ),
        )
    if preregistered != live:
        raise HistoricalV4PreregistrationAuthenticationError(
            "Live v4 authenticator accepted a non-live implementation mapping."
        )
    contract = _verify_returned_contract(
        contract,
        preregistration=preregistration,
        preregistration_sha256=expected,
        implementation=live,
    )
    return AuthenticatedV4Preregistration(
        contract=contract,
        receipt=_receipt(
            mode="current-live",
            contract=contract,
            preregistration_schema=str(raw["schema"]),
            preregistration_implementation=preregistered,
            live_implementation=live,
            delta_keys=delta_keys,
            override_used=False,
            provenance=None,
        ),
    )
