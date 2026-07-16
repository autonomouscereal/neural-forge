from __future__ import annotations

# ruff: noqa: E402

import dataclasses
import gzip
import hashlib
import json
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import authenticate_qwen_exact_target_native_guided_gud_candidate_tournament_history as history
import run_qwen_exact_target_native_guided_gud_candidate_tournament as tournament


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_preregistration(
    path: Path,
    implementation: dict[str, str],
    *,
    schema: str = tournament.PREREGISTRATION_SCHEMA,
) -> str:
    path.write_text(
        json.dumps(
            {"schema": schema, "implementation": implementation},
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return _digest(path)


def _contract(
    path: Path,
    file_sha256: str,
    implementation: dict[str, str],
) -> tournament.AuthenticatedGuidedGUDCandidateTournament:
    return tournament.AuthenticatedGuidedGUDCandidateTournament(
        path=path.resolve(),
        file_sha256=file_sha256,
        source=None,  # type: ignore[arg-type]
        splits={},
        basis=None,  # type: ignore[arg-type]
        site_mask=None,  # type: ignore[arg-type]
        site_mask_path=path.resolve(),
        site_mask_file_sha256="a" * 64,
        arms=(),
        schedule=None,  # type: ignore[arg-type]
        runtime={},
        implementation=dict(implementation),
        evaluation_contract_sha256="e" * 64,
    )


def _install_authenticator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    preregistered: dict[str, str],
    calls: list[dict[str, str]],
    fail_historical: bool = False,
) -> None:
    def authenticate(path: Path | str, expected_sha256: str) -> Any:
        observed = tournament._implementation_contract()
        calls.append(dict(observed))
        if observed != preregistered or fail_historical:
            raise tournament.GuidedGUDCandidateTournamentError(
                "Guided-GUD implementation hashes changed."
            )
        return _contract(Path(path), expected_sha256, observed)

    monkeypatch.setattr(
        tournament,
        "authenticate_guided_gud_candidate_preregistration",
        authenticate,
    )


def _implementation_provider(
    implementation: dict[str, str],
) -> Callable[[], dict[str, str]]:
    def provide() -> dict[str, str]:
        return dict(implementation)

    return provide


def test_historical_asset_is_deterministic_and_exact() -> None:
    asset = (
        Path(history.__file__).resolve().parent
        / history.HISTORICAL_SOURCE_ASSET_RELATIVE_PATH
    )
    packed = asset.read_bytes()
    source = gzip.decompress(packed)

    assert len(packed) == history.HISTORICAL_SOURCE_ASSET_SIZE == 38_888
    assert hashlib.sha256(packed).hexdigest() == (
        history.HISTORICAL_SOURCE_ASSET_SHA256
    )
    assert packed[4:8] == b"\x00\x00\x00\x00"
    assert len(source) == history.HISTORICAL_SOURCE_SIZE == 234_133
    assert hashlib.sha256(source).hexdigest() == history.HISTORICAL_SOURCE_SHA256
    assert history.HISTORICAL_SOURCE_SHA256 == (
        "88e22e62c500afff70602fa1c4d2f11de40759265737e569115d1f0ac66e4fa5"
    )
    assert history.V4_PREREGISTRATION_SCHEMA == tournament.PREREGISTRATION_SCHEMA
    assert history.HISTORICAL_PREREGISTRATION_SHA256 == (
        "8a05c9fbb8407475ccad47d4081a0f6e5feeadd883a778e69011216027d97f09"
    )


def test_current_authentication_runs_unchanged_and_returns_frozen_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = {
        "runner_sha256": "1" * 64,
        "behavior_trainer_sha256": "2" * 64,
    }
    preregistration = tmp_path / "preregistration.json"
    preregistration_sha256 = _write_preregistration(
        preregistration,
        implementation,
    )
    live_contract = _implementation_provider(implementation)
    monkeypatch.setattr(tournament, "_implementation_contract", live_contract)
    calls: list[dict[str, str]] = []
    _install_authenticator(
        monkeypatch,
        preregistered=implementation,
        calls=calls,
    )

    authenticated = (
        history.authenticate_guided_gud_candidate_preregistration_with_history(
            preregistration,
            preregistration_sha256,
        )
    )

    assert calls == [implementation]
    assert authenticated.contract.implementation == implementation
    assert authenticated.receipt.mode == "current-live"
    assert authenticated.receipt.implementation_override_used is False
    assert authenticated.receipt.implementation_override_restored is True
    assert authenticated.receipt.historical_source is None
    assert tournament._implementation_contract is live_contract
    with pytest.raises(dataclasses.FrozenInstanceError):
        authenticated.receipt.mode = "exact-historical-source"  # type: ignore[misc]


def test_exact_historical_fallback_verifies_source_and_reuses_full_authenticator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistered = {
        "runner_sha256": "1" * 64,
        "behavior_trainer_sha256": history.HISTORICAL_SOURCE_SHA256,
        "builder_sha256": "3" * 64,
    }
    live = dict(preregistered)
    live["behavior_trainer_sha256"] = "9" * 64
    preregistration = tmp_path / "preregistration.json"
    preregistration_sha256 = _write_preregistration(
        preregistration,
        preregistered,
    )
    monkeypatch.setattr(
        history,
        "HISTORICAL_PREREGISTRATION_SHA256",
        preregistration_sha256,
    )
    live_contract = _implementation_provider(live)
    monkeypatch.setattr(tournament, "_implementation_contract", live_contract)
    calls: list[dict[str, str]] = []
    _install_authenticator(
        monkeypatch,
        preregistered=preregistered,
        calls=calls,
    )

    authenticated = (
        history.authenticate_guided_gud_candidate_preregistration_with_history(
            preregistration,
            preregistration_sha256,
        )
    )

    assert calls == [live, preregistered]
    assert tournament._implementation_contract is live_contract
    assert authenticated.contract.implementation == preregistered
    receipt = authenticated.receipt
    assert receipt.mode == "exact-historical-source"
    assert receipt.implementation_delta_keys == ("behavior_trainer_sha256",)
    assert receipt.implementation_override_used is True
    assert receipt.implementation_override_restored is True
    assert receipt.historical_source is not None
    assert receipt.historical_source.commit == history.HISTORICAL_SOURCE_COMMIT
    assert (
        receipt.historical_source.source_sha256
        == (preregistered["behavior_trainer_sha256"])
    )
    assert receipt.as_dict()["implementation_delta_keys"] == ["behavior_trainer_sha256"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.historical_source.source_size = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "extra_change",
    [False, True],
)
def test_historical_fallback_rejects_every_nonexact_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_change: bool,
) -> None:
    preregistered = {
        "runner_sha256": "1" * 64,
        "behavior_trainer_sha256": history.HISTORICAL_SOURCE_SHA256,
    }
    live = dict(preregistered)
    live["behavior_trainer_sha256"] = "9" * 64
    if extra_change:
        live["runner_sha256"] = "8" * 64
    preregistration = tmp_path / "preregistration.json"
    preregistration_sha256 = _write_preregistration(
        preregistration,
        preregistered,
    )
    live_contract = _implementation_provider(live)
    monkeypatch.setattr(tournament, "_implementation_contract", live_contract)
    calls: list[dict[str, str]] = []
    _install_authenticator(
        monkeypatch,
        preregistered=preregistered,
        calls=calls,
    )
    if extra_change:
        monkeypatch.setattr(
            history,
            "HISTORICAL_PREREGISTRATION_SHA256",
            preregistration_sha256,
        )
        match = "not solely behavior_trainer_sha256"
    else:
        match = "not authorized for this preregistration"

    with pytest.raises(
        history.HistoricalV4PreregistrationAuthenticationError,
        match=match,
    ):
        history.authenticate_guided_gud_candidate_preregistration_with_history(
            preregistration,
            preregistration_sha256,
        )

    assert calls == [live]
    assert tournament._implementation_contract is live_contract


def test_historical_fallback_rejects_unpreserved_trainer_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistered = {
        "runner_sha256": "1" * 64,
        "behavior_trainer_sha256": "7" * 64,
    }
    live = dict(preregistered)
    live["behavior_trainer_sha256"] = "9" * 64
    preregistration = tmp_path / "preregistration.json"
    preregistration_sha256 = _write_preregistration(
        preregistration,
        preregistered,
    )
    monkeypatch.setattr(
        history,
        "HISTORICAL_PREREGISTRATION_SHA256",
        preregistration_sha256,
    )
    live_contract = _implementation_provider(live)
    monkeypatch.setattr(tournament, "_implementation_contract", live_contract)
    calls: list[dict[str, str]] = []
    _install_authenticator(
        monkeypatch,
        preregistered=preregistered,
        calls=calls,
    )

    with pytest.raises(
        history.HistoricalV4PreregistrationAuthenticationError,
        match="not bound to the preserved trainer source",
    ):
        history.authenticate_guided_gud_candidate_preregistration_with_history(
            preregistration,
            preregistration_sha256,
        )

    assert calls == [live]
    assert tournament._implementation_contract is live_contract


def test_tampered_source_asset_fails_before_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistered = {
        "runner_sha256": "1" * 64,
        "behavior_trainer_sha256": history.HISTORICAL_SOURCE_SHA256,
    }
    live = dict(preregistered)
    live["behavior_trainer_sha256"] = "9" * 64
    preregistration = tmp_path / "preregistration.json"
    preregistration_sha256 = _write_preregistration(
        preregistration,
        preregistered,
    )
    monkeypatch.setattr(
        history,
        "HISTORICAL_PREREGISTRATION_SHA256",
        preregistration_sha256,
    )
    original_asset = (
        Path(history.__file__).resolve().parent
        / history.HISTORICAL_SOURCE_ASSET_RELATIVE_PATH
    ).read_bytes()
    tampered_asset = tmp_path / "tampered.py.gz"
    tampered_asset.write_bytes(original_asset[:-1] + bytes([original_asset[-1] ^ 1]))
    monkeypatch.setattr(
        history,
        "HISTORICAL_SOURCE_ASSET_RELATIVE_PATH",
        str(tampered_asset),
    )
    live_contract = _implementation_provider(live)
    monkeypatch.setattr(tournament, "_implementation_contract", live_contract)
    calls: list[dict[str, str]] = []
    _install_authenticator(
        monkeypatch,
        preregistered=preregistered,
        calls=calls,
    )

    with pytest.raises(
        history.HistoricalV4PreregistrationAuthenticationError,
        match="failed its immutable pin",
    ):
        history.authenticate_guided_gud_candidate_preregistration_with_history(
            preregistration,
            preregistration_sha256,
        )

    assert calls == [live]
    assert tournament._implementation_contract is live_contract


def test_override_restores_live_global_when_full_historical_authentication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistered = {
        "runner_sha256": "1" * 64,
        "behavior_trainer_sha256": history.HISTORICAL_SOURCE_SHA256,
    }
    live = dict(preregistered)
    live["behavior_trainer_sha256"] = "9" * 64
    preregistration = tmp_path / "preregistration.json"
    preregistration_sha256 = _write_preregistration(
        preregistration,
        preregistered,
    )
    monkeypatch.setattr(
        history,
        "HISTORICAL_PREREGISTRATION_SHA256",
        preregistration_sha256,
    )
    live_contract = _implementation_provider(live)
    monkeypatch.setattr(tournament, "_implementation_contract", live_contract)
    calls: list[dict[str, str]] = []
    _install_authenticator(
        monkeypatch,
        preregistered=preregistered,
        calls=calls,
        fail_historical=True,
    )

    with pytest.raises(tournament.GuidedGUDCandidateTournamentError):
        history.authenticate_guided_gud_candidate_preregistration_with_history(
            preregistration,
            preregistration_sha256,
        )

    assert calls == [live, preregistered]
    assert tournament._implementation_contract is live_contract


def test_hash_and_schema_preflight_fail_before_authenticator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = {
        "runner_sha256": "1" * 64,
        "behavior_trainer_sha256": "2" * 64,
    }
    preregistration = tmp_path / "preregistration.json"
    actual_sha256 = _write_preregistration(
        preregistration,
        implementation,
        schema="wrong-schema",
    )
    calls: list[dict[str, str]] = []
    _install_authenticator(
        monkeypatch,
        preregistered=implementation,
        calls=calls,
    )

    with pytest.raises(
        history.HistoricalV4PreregistrationAuthenticationError,
        match="changed from its supplied SHA-256",
    ):
        history.authenticate_guided_gud_candidate_preregistration_with_history(
            preregistration,
            "0" * 64,
        )
    with pytest.raises(
        history.HistoricalV4PreregistrationAuthenticationError,
        match="schema changed",
    ):
        history.authenticate_guided_gud_candidate_preregistration_with_history(
            preregistration,
            actual_sha256,
        )

    assert calls == []
