from __future__ import annotations

# ruff: noqa: E402

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_exact_target_native_guided_candidate_tournament as v3
import run_qwen_exact_target_native_guided_continuation as guided
import run_qwen_exact_target_native_guided_gud_candidate_tournament as v4
import run_qwen_exact_target_native_guided_winner_publication as publication
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_metrics(index: int) -> dict[str, Any]:
    action_matches = 49 - index
    token_matches = 48 - index
    lcp_tokens = 27 - index
    return {
        "split": "calibration-selection",
        "action_count": 192,
        "action_agreement_count": action_matches,
        "action_accuracy": action_matches / 192,
        "parent_action_mean_logprob": -4.0 - index,
        "parent_action_topk_fraction": 0.2,
        "parent_action_mean_rank": 100.0 + index,
        "parent_action_worst_rank": 1000 + index,
        "rollout_count": 24,
        "exact_rollout_count": 0,
        "exact_rollout_fraction": 0.0,
        "rollout_token_count": 192,
        "token_match_count": token_matches,
        "token_match_fraction": token_matches / 192,
        "lcp_token_count": lcp_tokens,
        "lcp_fraction": lcp_tokens / 192,
        "lcp_at_least_4_count": 0,
        "repeat_fraction_excess": 0.4 + index / 100,
        "adjacent_repeat_fraction_excess": 0.55 + index / 100,
        "longest_run_excess": 5 + index,
        "report_sha256": f"{index + 1:064x}",
    }


def _historical_v1_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    evaluation_sha = "e" * 64
    preregistration = tmp_path / "preregistration.json"
    _write_json(preregistration, {})
    contract = SimpleNamespace(
        path=preregistration,
        evaluation_contract_sha256=evaluation_sha,
    )
    runtime_ids = ["1" * 64, "2" * 64, "3" * 64]
    portable_ids = ["4" * 64, "5" * 64, "6" * 64]
    retained: list[dict[str, Any]] = []
    for index in range(3):
        pack_dir = tmp_path / "retained" / portable_ids[index]
        pack_dir.mkdir(parents=True)
        pack_path = pack_dir / "factor_pack.pt"
        torch.save({"fixture": torch.tensor([index], dtype=torch.float32)}, pack_path)
        metrics = _candidate_metrics(index)
        row = {
            "arm_id": v3.ARM_IDS[index],
            "arm_index": index,
            "step": 12 if index == 0 else 0,
            "rank": list(v3.tournament_behavior_rank(metrics)),
            "runtime_factor_state_sha256": runtime_ids[index],
            "portable_factor_state_sha256": portable_ids[index],
            "checkpoint_sha256": f"{index + 7:064x}",
            "metrics": metrics,
            "factor_pack": str(pack_path),
            "factor_pack_sha256": _digest(pack_path),
        }
        _write_json(pack_dir / "candidate_record.json", row)
        retained.append(row)
    frontier = {
        "schema": v4.HISTORICAL_V3_FRONTIER_SCHEMA,
        "evaluation_contract_sha256": evaluation_sha,
        "completed_arms": list(v3.ARM_IDS),
        "retained_count": 3,
        "retained": retained,
        "distinct_full_v3_states_only": True,
        "portable_content_identity_used": True,
        "step_zero_ranked": True,
        "later_states_require_pareto_safety_vs_step_zero": True,
        "sealed_metrics_used": False,
        "internal_numeric_matching_used": False,
    }
    frontier_path = tmp_path / "candidate_frontier.json"
    _write_json(frontier_path, frontier)
    winner_selection = {
        "schema": v4.HISTORICAL_V3_WINNER_SCHEMA,
        "evaluation_contract_sha256": evaluation_sha,
        "winner": retained[0],
        "top_three_factor_state_sha256s": runtime_ids,
        "top_three_portable_factor_state_sha256s": portable_ids,
        "candidate_only": True,
        "sealed_evaluation_accessed": False,
        "model_checkpoint_written": False,
        "publication_attempted": False,
        "stop_reason": "candidate-winner-selected-before-sealed",
    }
    winner_path = tmp_path / "winner_selection.json"
    _write_json(winner_path, winner_selection)
    run_report = {
        "schema": v4.HISTORICAL_V3_RUN_SCHEMA,
        "status": "candidate-winner-selected",
        "evaluation_contract_sha256": evaluation_sha,
        "winner_selection": winner_selection,
        "top_three_factor_state_sha256s": runtime_ids,
        "top_three_portable_factor_state_sha256s": portable_ids,
        "sealed_forward_count": 0,
        "sealed_evaluation_accessed": False,
        "sealed_evaluation_locked": True,
        "sealed_suite_supplied_to_trainer": False,
        "selection_uses_internal_numeric_targets": False,
        "model_checkpoint_written": False,
        "publication_attempted": False,
        "publication_authorized": False,
        "stop_after_candidate_winner": True,
    }
    run_path = tmp_path / "run_report.json"
    _write_json(run_path, run_report)
    prereg_sha = _digest(preregistration)
    run_sha = _digest(run_path)
    winner_sha = _digest(winner_path)
    frontier_sha = _digest(frontier_path)
    monkeypatch.setattr(
        v3,
        "authenticate_candidate_tournament_preregistration",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        v4.GuidedCheckpointRecoverySink,
        "_validate_v3_factor_payload",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        v4,
        "portable_v3_factor_state_sha256",
        lambda _payload: portable_ids[0],
    )
    monkeypatch.setattr(
        v4, "EXPECTED_SOURCE_TOURNAMENT_PREREGISTRATION_SHA256", prereg_sha
    )
    monkeypatch.setattr(v4, "EXPECTED_SOURCE_RUN_REPORT_SHA256", run_sha)
    monkeypatch.setattr(v4, "EXPECTED_SOURCE_WINNER_SELECTION_SHA256", winner_sha)
    monkeypatch.setattr(
        v4, "EXPECTED_SOURCE_CANDIDATE_FRONTIER_SHA256", frontier_sha
    )
    monkeypatch.setattr(
        v4, "EXPECTED_SOURCE_EVALUATION_CONTRACT_SHA256", evaluation_sha
    )
    monkeypatch.setattr(v4, "EXPECTED_SOURCE_RUNTIME_V3_SHA256", runtime_ids[0])
    monkeypatch.setattr(v4, "EXPECTED_SOURCE_PORTABLE_V3_SHA256", portable_ids[0])
    monkeypatch.setattr(
        v4, "EXPECTED_SOURCE_FACTOR_PACK_FILE_SHA256", retained[0]["factor_pack_sha256"]
    )
    source = {
        "tournament_preregistration": {
            "path": str(preregistration),
            "sha256": prereg_sha,
        },
        "tournament_run_report": {"path": str(run_path), "sha256": run_sha},
        "winner_selection": {"path": str(winner_path), "sha256": winner_sha},
        "expected_winner": {
            "arm_id": v4.EXPECTED_SOURCE_ARM_ID,
            "step": v4.EXPECTED_SOURCE_STEP,
            "runtime_factor_state_sha256": runtime_ids[0],
            "portable_factor_state_sha256": portable_ids[0],
            "factor_pack_file_sha256": retained[0]["factor_pack_sha256"],
        },
    }
    return source, contract, run_report, winner_selection


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


def test_v4_authenticates_only_the_exact_historical_v1_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, _, _ = _historical_v1_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        publication,
        "_winner_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live v2 publication parser must not handle historical v1")
        ),
    )

    authenticated = v4.authenticate_v3_winner_source(source, base=tmp_path)

    assert authenticated.arm_id == v4.EXPECTED_SOURCE_ARM_ID
    assert authenticated.step == v4.EXPECTED_SOURCE_STEP
    assert authenticated.candidate_frontier["schema"] == (
        v4.HISTORICAL_V3_FRONTIER_SCHEMA
    )
    assert authenticated.candidate_frontier["callback_preimages_persisted"] is False
    assert authenticated.candidate_frontier["publication_authorized"] is False
    assert "observed_callbacks" not in authenticated.candidate_frontier


def test_live_v2_publication_parser_remains_strict_for_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, contract, run_report, winner_selection = _historical_v1_fixture(
        tmp_path,
        monkeypatch,
    )

    with pytest.raises(
        publication.GuidedWinnerPublicationError,
        match="sealed-free winner boundary",
    ):
        publication._winner_payload(contract, run_report, winner_selection)


def test_historical_v1_hard_pin_rejects_coordinated_run_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, run_report, _ = _historical_v1_fixture(tmp_path, monkeypatch)
    run_report["sealed_forward_count"] = 1
    run_path = Path(source["tournament_run_report"]["path"])
    _write_json(run_path, run_report)
    source["tournament_run_report"]["sha256"] = _digest(run_path)

    with pytest.raises(
        v4.GuidedGUDCandidateTournamentError,
        match="run, winner, or evaluation identity changed",
    ):
        v4.authenticate_v3_winner_source(source, base=tmp_path)


def test_historical_v1_structure_rejects_frontier_relabeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _, _, _ = _historical_v1_fixture(tmp_path, monkeypatch)
    frontier_path = tmp_path / "candidate_frontier.json"
    frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
    frontier["retained"][0]["step"] = 0
    _write_json(frontier_path, frontier)
    monkeypatch.setattr(
        v4,
        "EXPECTED_SOURCE_CANDIDATE_FRONTIER_SHA256",
        _digest(frontier_path),
    )

    with pytest.raises(
        v4.GuidedGUDCandidateTournamentError,
        match="candidate record is absent or changed|winner, frontier, and terminal",
    ):
        v4.authenticate_v3_winner_source(source, base=tmp_path)
