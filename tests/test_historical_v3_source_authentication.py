from __future__ import annotations

# ruff: noqa: E402

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_exact_target_native_guided_candidate_tournament as v3
import run_qwen_exact_target_native_guided_continuation as guided
import run_qwen_exact_target_native_guided_gud_candidate_tournament as v4
import run_qwen_exact_target_native_stage as stage


HISTORICAL_GUIDED_IMPLEMENTATION = {
    "runner_sha256": (
        "47867a4ec5f4d5f61a9e8903699137887da169c9501ef958b5dde82216db361b"
    ),
    "builder_sha256": (
        "5d8781989e040517435af17448339057f0653e828f44715b618d961a97eacf6a"
    ),
    "base_runner_sha256": (
        "f1fb34691c1e1770644cbe1e6bb8d95b972e93f274ee8f50c875ff047b9195a0"
    ),
    "trainer_module": guided.GUIDED_TRAINER_MODULE,
    "trainer_sha256": (
        "ad0cb5c33acd495d3e20548298a95a4af26761c6494ab7e8645c81fef45411c1"
    ),
    "guided_adapter_sha256": (
        "7b01e317a02d5dd670525836a193631a1062e6a1c4240490306725cc1aaf20fb"
    ),
    "progressive_bridge_sha256": (
        "4b754d7c7bc17071cbacc14aa3e36d202f5b2368ff476978d125b348c7ceef55"
    ),
    "compiled_child_cache_sha256": (
        "74c7e0ead2b43b82c1c2089750f7583a66461f6f9a56dc3b51c5e7af097835df"
    ),
}
HISTORICAL_TOURNAMENT_IMPLEMENTATION = {
    "runner_sha256": (
        "31707ed24ce6f916e9e8f64b3a46eca6a0ad8f7eb96debc46c7108d41978b890"
    ),
    "builder_sha256": (
        "479c70d98bd5f3707a9f75a5a13a22edbeb1bbf66d0e324e15005adb126d702e"
    ),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_guided_and_tournament_pins_authenticate_from_exact_commit() -> None:
    assert guided._authenticate_implementation(
        HISTORICAL_GUIDED_IMPLEMENTATION,
        historical_git_revision=v3.HISTORICAL_V3_SOURCE_COMMIT,
    ) == HISTORICAL_GUIDED_IMPLEMENTATION
    assert v3._authenticate_tournament_implementation(
        HISTORICAL_TOURNAMENT_IMPLEMENTATION,
        historical_git_revision=v3.HISTORICAL_V3_SOURCE_COMMIT,
    ) == HISTORICAL_TOURNAMENT_IMPLEMENTATION


def test_default_authentication_still_requires_live_implementation() -> None:
    with pytest.raises(
        guided.GuidedContinuationError,
        match="implementation source pins changed",
    ):
        guided._authenticate_implementation(HISTORICAL_GUIDED_IMPLEMENTATION)
    with pytest.raises(
        v3.CandidateTournamentError,
        match="implementation hashes changed",
    ):
        v3._authenticate_tournament_implementation(
            HISTORICAL_TOURNAMENT_IMPLEMENTATION
        )


def test_historical_authentication_rejects_tamper_and_abbreviated_revision() -> None:
    tampered = dict(HISTORICAL_GUIDED_IMPLEMENTATION)
    tampered["trainer_sha256"] = "0" * 64
    with pytest.raises(
        guided.GuidedContinuationError,
        match="failed its SHA-256 pin",
    ):
        guided._authenticate_implementation(
            tampered,
            historical_git_revision=v3.HISTORICAL_V3_SOURCE_COMMIT,
        )
    with pytest.raises(
        guided.GuidedContinuationError,
        match="full lowercase commit SHA",
    ):
        guided._authenticate_implementation(
            HISTORICAL_GUIDED_IMPLEMENTATION,
            historical_git_revision="565aad1",
        )


def test_v4_v3_source_authentication_is_the_only_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings: dict[str, dict[str, str]] = {}
    for name in ("preregistration", "run-report", "winner"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        bindings[name] = {"path": str(path), "sha256": _digest(path)}
    observed: dict[str, Any] = {}

    def stop_after_call(path: Path, digest: str, **kwargs: Any) -> Any:
        observed.update(path=path, digest=digest, kwargs=kwargs)
        raise RuntimeError("stop after historical source call")

    monkeypatch.setattr(
        v3,
        "authenticate_candidate_tournament_preregistration",
        stop_after_call,
    )
    source = {
        "tournament_preregistration": bindings["preregistration"],
        "tournament_run_report": bindings["run-report"],
        "winner_selection": bindings["winner"],
        "expected_winner": {
            "arm_id": v4.EXPECTED_SOURCE_ARM_ID,
            "step": v4.EXPECTED_SOURCE_STEP,
            "runtime_factor_state_sha256": "1" * 64,
            "portable_factor_state_sha256": (
                v4.EXPECTED_SOURCE_PORTABLE_V3_SHA256
            ),
            "factor_pack_file_sha256": v4.EXPECTED_SOURCE_FACTOR_PACK_FILE_SHA256,
        },
    }
    with pytest.raises(
        v4.GuidedGUDCandidateTournamentError,
        match="Completed v3 winner/frontier authentication failed",
    ):
        v4.authenticate_v3_winner_source(source, base=tmp_path)
    assert observed["kwargs"] == {
        "historical_git_revision": v3.HISTORICAL_V3_SOURCE_COMMIT
    }
    assert observed["path"] == Path(bindings["preregistration"]["path"])
    assert observed["digest"] == bindings["preregistration"]["sha256"]


def test_live_guided_callers_do_not_pass_historical_opt_in() -> None:
    # The public default is part of the fail-closed contract, not just behavior
    # inferred from the current callers.
    assert (
        guided.authenticate_guided_preregistration.__kwdefaults__
        == {"historical_git_revision": None}
    )
    assert (
        v3.authenticate_candidate_tournament_preregistration.__kwdefaults__
        == {"historical_git_revision": None}
    )
    assert stage._file_sha256(Path(guided.__file__).resolve()) != (
        HISTORICAL_GUIDED_IMPLEMENTATION["runner_sha256"]
    )
