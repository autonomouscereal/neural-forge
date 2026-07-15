from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_exact_target_native_guided_gud_winner_publication as publication  # noqa: E402
import run_qwen_exact_target_native_stage as stage  # noqa: E402
import prepare_qwen_exact_target_native_guided_gud_winner_publication as winner_prepare  # noqa: E402
import retain_qwen_exact_target_candidates as retention  # noqa: E402
from neural_forge.hf_exact_candidate_frontier import (  # noqa: E402
    ExactCandidateFrontier,
    ExactCandidateFrontierError,
    artifact_tree,
    authenticate_terminal_candidate,
    file_sha256,
)
from neural_forge.hf_target_native_behavior_distillation import (  # noqa: E402
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _with_self_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = publication.canonical_sha256(result)
    return result


def _compiler_report() -> dict[str, Any]:
    return {
        "bootstrap_accepted": True,
        "target_budget_exact": True,
        "target_unique_parameter_count": stage.QWEN3_1P4B_EXACT_PARAMETERS,
    }


def _metrics(
    *,
    split: str = "calibration-selection",
    action_accuracy: float = 1.0,
) -> dict[str, Any]:
    return {
        "split": split,
        "action_count": 192,
        "action_agreement_count": round(192 * action_accuracy),
        "action_accuracy": action_accuracy,
        "parent_action_mean_logprob": -1.0,
        "parent_action_topk_fraction": 1.0,
        "parent_action_mean_rank": 1.0,
        "parent_action_worst_rank": 1,
        "rollout_count": 24,
        "exact_rollout_count": 6,
        "exact_rollout_fraction": 0.25,
        "rollout_token_count": 192,
        "token_match_count": 120,
        "token_match_fraction": 0.625,
        "lcp_token_count": 80,
        "lcp_fraction": 80 / 192,
        "lcp_at_least_4_count": 6,
        "repeat_fraction_excess": 0.1,
        "adjacent_repeat_fraction_excess": 0.1,
        "longest_run_excess": 0,
        "report_sha256": _digest(f"metrics:{split}:{action_accuracy}"),
    }


def _thresholds() -> dict[str, Any]:
    return stage.BehaviorThresholds(
        minimum_action_accuracy=0.5,
        minimum_exact_rollout_fraction=0.25,
        minimum_token_match_fraction=0.5,
        minimum_lcp_fraction=0.35,
        minimum_action_count=192,
        minimum_rollout_count=24,
    ).as_dict()


def _split(tmp_path: Path, name: str, count: int) -> Any:
    path = tmp_path / f"{name}.json"
    path.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        name=name,
        path=path,
        sha256=file_sha256(path),
        expected_count=count,
        row_ids=tuple(f"{name}-{index}" for index in range(count)),
        prompt_sha256s=tuple(_digest(f"{name}-{index}") for index in range(count)),
        payload={"prompts": []},
    )


def _arm(index: int) -> Any:
    arm_id = publication.tournament.ARM_IDS[index]
    method = publication.tournament.BASIS_METHODS[
        index // len(publication.tournament.MASK_MODES)
    ]
    mode = publication.tournament.MASK_MODES[
        index % len(publication.tournament.MASK_MODES)
    ]
    payload = {
        "arm_id": arm_id,
        "arm_index": index,
        "basis_method": method,
        "mask_mode": mode,
    }
    return SimpleNamespace(
        arm_id=arm_id,
        arm_index=index,
        basis_method=method,
        mask_mode=mode,
        site_mask_enabled=mode == "masked",
        as_dict=lambda: dict(payload),
    )


def _source_contract(tmp_path: Path) -> Any:
    source_prereg = tmp_path / "source-v3-preregistration.json"
    source_run = tmp_path / "source-v3-run-report.json"
    source_winner = tmp_path / "source-v3-winner-selection.json"
    source_frontier = tmp_path / "source-v3-candidate-frontier.json"
    source_pack = tmp_path / "source-v3-factor-pack.pt"
    for path in (source_prereg, source_run, source_winner, source_frontier):
        path.write_text("{}\n", encoding="utf-8")
    source_pack.write_bytes(b"source-v3")
    source_guided = SimpleNamespace(execution_contract=SimpleNamespace())
    v3_tournament = SimpleNamespace(
        evaluation_contract_sha256=_digest("v3-evaluation"),
        source=source_guided,
    )
    v3_source = SimpleNamespace(
        tournament_contract=v3_tournament,
        tournament_preregistration_path=source_prereg,
        tournament_preregistration_sha256=file_sha256(source_prereg),
        run_report_path=source_run,
        run_report_sha256=file_sha256(source_run),
        winner_selection_path=source_winner,
        winner_selection_sha256=file_sha256(source_winner),
        candidate_frontier={
            "path": str(source_frontier),
            "sha256": file_sha256(source_frontier),
        },
        runtime_factor_state_sha256=_digest("source-runtime-v3"),
        portable_factor_state_sha256=_digest("source-portable-v3"),
        factor_pack_path=source_pack,
        factor_pack_file_sha256=file_sha256(source_pack),
        factor_payload={"schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA},
    )
    source_basis_hashes = {
        method: {role: _digest(f"{method}:{role}") for role in ("gate", "up", "down")}
        for method in publication.tournament.BASIS_METHODS
    }
    basis_path = tmp_path / "guided-gud-basis-bundle.pt"
    basis_path.write_bytes(b"guided-gud-basis")
    return SimpleNamespace(
        path=tmp_path / "tournament-preregistration.json",
        file_sha256=_digest("tournament-preregistration"),
        evaluation_contract_sha256=_digest("v4-evaluation"),
        source=v3_source,
        splits={
            "fit": _split(tmp_path, "fit", 192),
            "candidate": _split(tmp_path, "candidate", 24),
            "sealed": _split(tmp_path, "sealed", 32),
        },
        arms=tuple(_arm(index) for index in range(8)),
        basis=SimpleNamespace(
            path=basis_path,
            file_sha256=file_sha256(basis_path),
            payload_sha256=_digest("basis-payload"),
            audit_report_sha256=_digest("basis-audit"),
            basis_manifest_sha256=_digest("basis-manifest"),
            source_basis_tensor_sha256s=source_basis_hashes,
        ),
        site_mask=SimpleNamespace(mask_contract_sha256=_digest("site-mask")),
        schedule=SimpleNamespace(steps=64, as_dict=lambda: {"steps": 64}),
        runtime={
            "dtype": "float16",
            "parent_device_map": "cuda:0",
            "target_device_map": "cuda:1",
        },
    )


def _winner() -> dict[str, Any]:
    metrics = _metrics()
    admission = {
        "passed": True,
        "winner_arm_id": publication.tournament.ARM_IDS[0],
        "winner_step": 16,
        "winner_callback_ordinal": 1,
        "step_zero_callback_ordinal": 0,
        "declared_pareto_safe_vs_step0": True,
        "objectives_equal_to_step_zero": False,
        "dominates_step_zero": True,
        "winner_metrics_sha256": publication.canonical_sha256(metrics),
        "step_zero_metrics_sha256": _digest("step-zero-metrics"),
        "recomputed_from_authenticated_output_metrics": True,
        "internal_numeric_matching_consulted": False,
    }
    admission["decision_sha256"] = publication.canonical_sha256(admission)
    return {
        "arm_id": publication.tournament.ARM_IDS[0],
        "arm_index": 0,
        "basis_method": publication.tournament.BASIS_METHODS[0],
        "mask_mode": publication.tournament.MASK_MODES[0],
        "site_mask_enabled": False,
        "step": 16,
        "rank": [0],
        "runtime_factor_state_sha256": _digest("runtime-v4"),
        "portable_factor_state_sha256": _digest("portable-v4"),
        "checkpoint_sha256": _digest("checkpoint-v4"),
        "metrics": metrics,
        "metrics_sha256": publication.canonical_sha256(metrics),
        "factor_pack": "factor_pack.pt",
        "factor_pack_sha256": _digest("factor-pack-v4"),
        "candidate_frontier": {
            "schema": publication.tournament.FRONTIER_SCHEMA,
            "path": "candidate_frontier.json",
            "sha256": _digest("frontier-v4"),
            "observed_callback_count": 96,
            "pending_callback_count": 0,
            "completed_arms": list(publication.tournament.ARM_IDS),
        },
        "tournament_pareto_admission": admission,
    }


def test_model_free_contract_binds_eight_arms_96_callbacks_and_dense_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_contract(tmp_path)
    tournament_prereg = source.path
    tournament_report = tmp_path / "run_report.json"
    winner_selection = tmp_path / "winner_selection.json"
    for path in (tournament_prereg, tournament_report, winner_selection):
        path.write_text("{}\n", encoding="utf-8")
    source.file_sha256 = file_sha256(tournament_prereg)
    winner = _winner()
    winner["factor_pack"] = str(tmp_path / "factor_pack.pt")
    monkeypatch.setattr(
        publication.tournament,
        "authenticate_guided_gud_candidate_preregistration",
        lambda *_a: source,
    )
    monkeypatch.setattr(
        publication,
        "_winner_payload",
        lambda *_a: (winner, {"schema": BEHAVIOR_FACTOR_PACK_V4_SCHEMA}),
    )
    implementation = {
        name: _digest(name)
        for name in (
            "runner_sha256",
            "builder_sha256",
            "guided_gud_tournament_runner_sha256",
            "v3_publication_runner_sha256",
            "guided_runner_sha256",
            "exact_stage_runner_sha256",
            "trainer_sha256",
            "checkpoint_recovery_sha256",
            "progressive_bridge_sha256",
            "guided_qkvo_adapter_sha256",
            "guided_gud_adapter_sha256",
            "independent_authenticator_sha256",
        )
    }
    monkeypatch.setattr(
        publication,
        "_implementation_hashes",
        lambda **_kw: implementation,
    )
    payload, authenticated, observed_winner, _ = (
        publication.build_winner_publication_preregistration_payload(
            tournament_preregistration=tournament_prereg,
            tournament_preregistration_sha256=file_sha256(tournament_prereg),
            tournament_run_report=tournament_report,
            tournament_run_report_sha256=file_sha256(tournament_report),
            winner_selection=winner_selection,
            winner_selection_sha256=file_sha256(winner_selection),
            sealed_consumption_marker=tmp_path / "sealed_consumption.json",
            dtype="float16",
            parent_device_map="cuda:0",
            target_device_map="cuda:1",
            builder_path=Path(publication.__file__),
        )
    )

    assert authenticated is source
    assert observed_winner is winner
    assert payload["tournament"]["expected_arm_count"] == 8
    assert payload["tournament"]["expected_callback_count"] == 96
    assert payload["splits"]["sealed"]["expected_count"] == 32
    semantic = payload["semantic_contract"]
    assert semantic["winner_arm"] == source.arms[0].as_dict()
    assert semantic["published_exact_unique_parameters"] == 1_410_567_168
    assert semantic["published_adapter_parameter_count"] == 0
    assert semantic["published_parent_runtime_reference_count"] == 0
    assert semantic["published_source_sized_sidecar_parameter_count"] == 0
    assert semantic["sealed_forward_limit"] == 1
    assert semantic["post_bake_sealed_forbidden"] is True
    assert payload["publication_authorized_before_sealed_pass"] is False
    assert not (tmp_path / "sealed_consumption.json").exists()

    prereg = tmp_path / "publication-preregistration.json"
    prereg.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        publication,
        "build_winner_publication_preregistration_payload",
        lambda **_kw: (
            payload,
            source,
            winner,
            {"schema": BEHAVIOR_FACTOR_PACK_V4_SCHEMA},
        ),
    )
    contract = publication.authenticate_winner_publication_preregistration(
        prereg,
        file_sha256(prereg),
    )
    assert contract.preflight_report()["tournament_callback_count"] == 96
    contract.sealed_marker.write_text("already consumed\n", encoding="utf-8")
    with pytest.raises(
        publication.GuidedGUDWinnerPublicationError,
        match="preflight cannot pass",
    ):
        contract.preflight_report()
    contract.sealed_marker.unlink()

    tampered = dict(payload)
    tampered["tournament"] = dict(payload["tournament"])
    tampered["tournament"]["expected_callback_count"] = 95
    prereg.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(
        publication.GuidedGUDWinnerPublicationError,
        match="eight arms and 96 callbacks",
    ):
        publication.authenticate_winner_publication_preregistration(
            prereg,
            file_sha256(prereg),
        )


def test_run_receipts_require_all_eight_exact_rollbacks_and_no_sealed(
    tmp_path: Path,
) -> None:
    source = _source_contract(tmp_path)
    arms = []
    reset = _digest("one-compiled-child")
    for index, arm in enumerate(source.arms):
        runtime_v3 = _digest(f"runtime-v3:{index}")
        arms.append(
            {
                "arm": arm.as_dict(),
                "reset_state_sha256": reset,
                "rollback_state_sha256": reset,
                "runtime_rebound_source_v3_factor_state_sha256": runtime_v3,
                "trainer_initial_factor_state_sha256": runtime_v3,
                "post_optimizer_mask_verification_count": (
                    64 if arm.site_mask_enabled else 0
                ),
                "guided_gud_parameter_count": 4_792_320,
                "sealed_forward_count": 0,
            }
        )
    publication._authenticate_run_arms(source, {"arms": arms})

    arms[7]["sealed_forward_count"] = 1
    with pytest.raises(
        publication.GuidedGUDWinnerPublicationError,
        match="arm receipt 7",
    ):
        publication._authenticate_run_arms(source, {"arms": arms})


def test_frontier_reconstructs_all_96_callbacks_and_locks_post_step_winner(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "tournament-preregistration.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    evaluation_sha = _digest("v4-evaluation")
    contract = SimpleNamespace(path=contract_path, evaluation_contract_sha256=evaluation_sha)
    portable = (_digest("weak"), _digest("strong"), _digest("medium"))
    metric_rows = (_metrics(action_accuracy=0.5), _metrics(), _metrics(action_accuracy=0.75))
    observations = []
    for ordinal in range(publication.tournament.EXPECTED_CALLBACK_COUNT):
        arm_index = ordinal // len(publication.tournament.CHECKPOINT_STEPS)
        step = publication.tournament.CHECKPOINT_STEPS[
            ordinal % len(publication.tournament.CHECKPOINT_STEPS)
        ]
        identity_index = ordinal if ordinal < 3 else 0
        metrics = metric_rows[identity_index]
        observations.append(
            {
                "ordinal": ordinal,
                "arm_id": publication.tournament.ARM_IDS[arm_index],
                "arm_index": arm_index,
                "step": step,
                "checkpoint_sha256": _digest(f"checkpoint:{ordinal}"),
                "runtime_factor_state_sha256": _digest(
                    f"runtime:{identity_index}"
                ),
                "portable_factor_state_sha256": portable[identity_index],
                "pareto_safe_vs_step0": True,
                "metrics": metrics,
                "metrics_sha256": publication.canonical_sha256(metrics),
                "rank": list(
                    publication.tournament.v3_tournament.tournament_behavior_rank(
                        metrics
                    )
                ),
                "eligible_for_retention": True,
                "disposition": (
                    "retained-top-three" if ordinal < 3 else "duplicate-retained-state"
                ),
                "retained_after_callback": True,
                "factor_pack_created": ordinal < 3,
                "evicted_portable_factor_state_sha256s": [],
                "terminal_verified": True,
                "terminal_pareto_safe_vs_step0": True,
            }
        )
    retained = []
    # Rollout-first rank orders strong, medium, then weak.
    for identity_index, ordinal in ((1, 1), (2, 2), (0, 0)):
        pack = tmp_path / f"factor-{identity_index}.pt"
        pack.write_bytes(f"factor-{identity_index}".encode())
        observation = observations[ordinal]
        retained.append(
            {
                "arm_id": observation["arm_id"],
                "arm_index": observation["arm_index"],
                "step": observation["step"],
                "rank": observation["rank"],
                "runtime_factor_state_sha256": observation[
                    "runtime_factor_state_sha256"
                ],
                "portable_factor_state_sha256": observation[
                    "portable_factor_state_sha256"
                ],
                "checkpoint_sha256": observation["checkpoint_sha256"],
                "metrics": observation["metrics"],
                "factor_pack": str(pack),
                "factor_pack_sha256": file_sha256(pack),
            }
        )
    frontier = {
        "schema": publication.tournament.FRONTIER_SCHEMA,
        "evaluation_contract_sha256": evaluation_sha,
        "completed_arms": list(publication.tournament.ARM_IDS),
        "retained_count": 3,
        "retained": retained,
        "observed_callback_count": 96,
        "pending_callback_count": 0,
        "observed_callbacks": observations,
        "distinct_full_v4_states_only": True,
        "portable_content_identity_used": True,
        "step_zero_ranked": True,
        "later_states_require_pareto_safety_vs_step_zero": True,
        "sealed_metrics_used": False,
        "internal_numeric_matching_used": False,
        "callback_metrics_are_durable_without_factor_pack_retention": True,
        "maximum_retained_factor_packs": 3,
    }
    frontier_path = tmp_path / "candidate_frontier.json"
    frontier_path.write_text(
        json.dumps(frontier, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    binding = {
        "schema": publication.tournament.FRONTIER_SCHEMA,
        "path": str(frontier_path),
        "sha256": file_sha256(frontier_path),
        "observed_callback_count": 96,
        "pending_callback_count": 0,
        "completed_arms": list(publication.tournament.ARM_IDS),
    }
    run_report = {"candidate_frontier": binding}
    winner_selection = {"candidate_frontier": binding}

    authenticated, admission = publication._authenticated_frontier_binding(
        contract,
        run_report,
        winner_selection,
    )

    assert authenticated["observed_callback_count"] == 96
    assert admission["passed"] is True
    assert admission["winner_arm_id"] == publication.tournament.ARM_IDS[0]
    assert admission["winner_step"] == 2
    assert admission["dominates_step_zero"] is True

    frontier["observed_callbacks"][95]["terminal_verified"] = False
    frontier_path.write_text(
        json.dumps(frontier, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered_binding = dict(binding)
    tampered_binding["sha256"] = file_sha256(frontier_path)
    with pytest.raises(
        publication.GuidedGUDWinnerPublicationError,
        match="unauthenticated callback evidence",
    ):
        publication._authenticated_frontier_binding(
            contract,
            {"candidate_frontier": tampered_binding},
            {"candidate_frontier": tampered_binding},
        )


def test_preparer_atomically_writes_only_model_free_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tournament_prereg = tmp_path / "tournament.json"
    run_report = tmp_path / "run_report.json"
    winner_selection = tmp_path / "winner_selection.json"
    for path in (tournament_prereg, run_report, winner_selection):
        path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "publication"
    winner = _winner()
    payload = {
        "schema": publication.PREREGISTRATION_SCHEMA,
        "semantic_contract_sha256": _digest("semantic"),
    }
    monkeypatch.setattr(
        winner_prepare.publication,
        "build_winner_publication_preregistration_payload",
        lambda **_kw: (
            payload,
            SimpleNamespace(),
            winner,
            {"schema": BEHAVIOR_FACTOR_PACK_V4_SCHEMA},
        ),
    )
    monkeypatch.setattr(
        winner_prepare.publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: SimpleNamespace(
            sealed_marker=output / "sealed_consumption.json"
        ),
    )

    report = winner_prepare.prepare_winner_publication(
        output_dir=output,
        tournament_preregistration=tournament_prereg,
        tournament_run_report=run_report,
        winner_selection=winner_selection,
        dtype="float16",
        parent_device_map="cuda:0",
        target_device_map="cuda:1",
    )

    assert report["status"] == "ready"
    assert report["authenticated_arm_count"] == 8
    assert report["authenticated_callback_count"] == 96
    assert report["winner_locked_before_sealed"] is True
    assert report["sealed_reranking_forbidden"] is True
    assert sorted(path.name for path in output.iterdir()) == [
        "build_report.json",
        "preregistration.json",
    ]
    assert not list(tmp_path.glob(f".{output.name}.*.staging"))


class _FakeTransaction:
    def __init__(self, events: list[str], runtime: "_FakeRuntime") -> None:
        self.events = events
        self.runtime = runtime
        self.is_live = True
        self.qkvo_handle: Any = object()
        self.swiglu_handle: Any = object()

    def installation_report(self) -> dict[str, Any]:
        qkvo = _with_self_hash(
            {
                "schema": "neural-forge-dense-gqa-qkvo-residual-installation-v2",
                "adapter_parameter_count": 101,
            },
            "installation_sha256",
        )
        gud = _with_self_hash(
            {
                "schema": (
                    "neural-forge-dense-swiglu-projection-residual-"
                    "installation-v2"
                ),
                "adapter_parameter_count": 202,
            },
            "installation_sha256",
        )
        return _with_self_hash(
            {
                "schema": (
                    "neural-forge-exact-target-compiler-transaction-"
                    "installation-v1"
                ),
                "exact_native_parameters": stage.QWEN3_1P4B_EXACT_PARAMETERS,
                "temporary_qkvo_factor_parameters": 101,
                "temporary_gate_up_down_factor_parameters": 202,
                "temporary_lexical_factor_parameters": 303,
                "temporary_factor_parameters": 606,
                "published_parameters_if_promoted": (
                    stage.QWEN3_1P4B_EXACT_PARAMETERS
                ),
                "zero_step_identity": True,
                "parent_compiler_report_sha256": publication.canonical_sha256(
                    _compiler_report()
                ),
                "dense_qkvo_installation": qkvo,
                "dense_gate_up_down_installation": gud,
            },
            "installation_transaction_sha256",
        )

    def rollback(self) -> None:
        self.events.append("rollback-base")
        self.is_live = False

    def bake_for_publication(self, receipt: Any) -> dict[str, Any]:
        assert receipt.passed is True
        self.events.append("bake-qkvo-and-gud")
        qkvo = _with_self_hash(
            {
                "schema": "neural-forge-guided-dense-gqa-qkvo-residual-bake-v1",
                "inherited_adapter_parameters_removed": 101,
                "guided_adapter_parameters_removed": 7_127_040,
                "adapter_parameters_removed": 7_127_141,
                "factor_concatenation_used": False,
                "factor_regrouping_used": False,
                "inherited_registry_removed": True,
                "guided_registry_removed": True,
                "projection_wrappers_removed": True,
            },
            "bake_sha256",
        )
        gud = _with_self_hash(
            {
                "schema": (
                    "neural-forge-guided-dense-swiglu-projection-residual-"
                    "bake-v1"
                ),
                "inherited_adapter_parameters_removed": 202,
                "guided_adapter_parameters_removed": 4_792_320,
                "adapter_parameters_removed": 4_792_522,
                "model_parameter_count_after_bake": (
                    stage.QWEN3_1P4B_EXACT_PARAMETERS
                ),
                "factor_concatenation_used": False,
                "factor_regrouping_used": False,
                "inherited_registry_removed": True,
                "guided_registry_removed": True,
                "projection_wrappers_removed": True,
            },
            "bake_sha256",
        )
        lexical = _with_self_hash(
            {
                "schema": "neural-forge-dense-lexical-residual-bake-v1",
                "adapter_parameters_removed": 303,
            },
            "bake_sha256",
        )
        return _with_self_hash(
            {
                "schema": (
                    "neural-forge-exact-target-compiler-transaction-"
                    "publication-v1"
                ),
                "exact_unique_parameters": stage.QWEN3_1P4B_EXACT_PARAMETERS,
                "exact_budget_match": True,
                "dense_qkvo_bake": qkvo,
                "dense_gate_up_down_bake": gud,
                "dense_lexical_bake": lexical,
                "external_behavior_gate": receipt.as_dict(),
                "parent_compiler_report_sha256": publication.canonical_sha256(
                    _compiler_report()
                ),
                "published_adapter_parameter_count": 0,
                "published_parent_runtime_reference_count": 0,
                "published_source_sized_sidecar_parameter_count": 0,
                "guided_registry_suspended_for_lexical_bake": True,
                "guided_gate_up_down_registry_suspended_for_lexical_bake": True,
                "guided_gate_up_down_registry_suspended_for_qkvo_bake": True,
                "nested_publication_bake_order": [
                    "lexical",
                    "qkvo-inherited-then-guided",
                    "gate-up-down-inherited-then-guided",
                ],
            },
            "publication_transaction_sha256",
        )

    def restore_prepublication_native(self) -> None:
        self.events.append("restore-prepublication-native")
        self.is_live = False


class _FakeRuntime:
    def __init__(
        self,
        events: list[str],
        *,
        candidate_matches: bool = True,
    ) -> None:
        self.events = events
        self.candidate_matches = candidate_matches
        self.sealed_forward_count = 0
        self.model = SimpleNamespace()
        self.transaction = _FakeTransaction(events, self)
        self._guided_tranche = SimpleNamespace(
            set_training_stage=lambda value: events.append(f"qkvo-stage-{value}")
        )
        self._guided_gud_tranche = SimpleNamespace()
        self.load_count = 0

    def prepare_compiler_cache(self, _contract: Any) -> dict[str, Any]:
        return {"cache_authenticated": True}

    def load_models(self, _contract: Any) -> tuple[Any, Any, Any]:
        self.load_count += 1
        self.events.append("load-models")
        return object(), object(), object()

    def compile(self, *_args: Any) -> Any:
        self.events.append("compile")
        return SimpleNamespace(child_model=self.model, report=_compiler_report())

    def state_sha256(self, _model: Any) -> str:
        return _digest("compiled")

    def install_transaction(self, _compiled: Any) -> _FakeTransaction:
        self.events.append("install-base")
        return self.transaction

    def restore_source_v2(self, *_args: Any) -> dict[str, Any]:
        self.events.append("restore-v2")
        return {
            "schema": "neural-forge-target-native-factor-pack-v2",
            "factor_state_sha256": _digest("source-v2"),
            "factor_count": 18,
            "temporary_restore_clone_released_before_guided_install": True,
            "full_tensor_or_model_clone_retained": False,
        }

    def install_guided_transaction(self, *_args: Any) -> dict[str, Any]:
        self.events.append("install-guided-qkvo")
        inherited = self.transaction.installation_report()[
            "dense_qkvo_installation"
        ]
        return _with_self_hash(
            {
                "schema": (
                    "neural-forge-guided-dense-gqa-qkvo-residual-"
                    "installation-v1"
                ),
                "inherited_adapter_parameter_count": 101,
                "guided_adapter_parameter_count": 7_127_040,
                "adapter_parameter_count": 7_127_141,
                "audit_report_sha256": _digest("qkvo-audit-report"),
                "basis_envelope_file_sha256": _bytes_digest(
                    b"guided-qkvo-basis"
                ),
                "basis_payload_sha256": _digest("qkvo-basis-payload"),
                "basis_manifest_sha256": _digest("qkvo-basis-manifest"),
                "source_basis_tensor_sha256s": {
                    role: _digest(f"qkvo-basis:{role}")
                    for role in ("q", "k", "v", "o")
                },
                "inherited_installation_sha256": inherited[
                    "installation_sha256"
                ],
                "factor_concatenation_used": False,
                "factor_regrouping_used": False,
                "zero_guided_left_preserves_inherited_function_exactly": True,
            },
            "installation_sha256",
        )

    def install_guided_gud_transaction(self, *_args: Any) -> dict[str, Any]:
        self.events.append("install-guided-gud")
        inherited = self.transaction.installation_report()[
            "dense_gate_up_down_installation"
        ]
        return _with_self_hash(
            {
                "schema": (
                    "neural-forge-guided-dense-swiglu-projection-residual-"
                    "installation-v1"
                ),
                "inherited_adapter_parameter_count": 202,
                "guided_adapter_parameter_count": 4_792_320,
                "adapter_parameter_count": 4_792_522,
                "audit_report_sha256": _digest("gud-audit-report"),
                "basis_envelope_file_sha256": _bytes_digest(
                    b"v4-guided-gud-basis"
                ),
                "basis_payload_sha256": _digest("gud-basis-payload"),
                "basis_manifest_sha256": _digest("gud-basis-manifest"),
                "source_basis_tensor_sha256s": {
                    role: _digest(f"gud-basis:{role}")
                    for role in ("gate", "up", "down")
                },
                "inherited_installation_sha256": inherited[
                    "installation_sha256"
                ],
                "factor_concatenation_used": False,
                "factor_regrouping_used": False,
                "zero_guided_left_preserves_inherited_function_exactly": True,
            },
            "installation_sha256",
        )

    def rollback_v4_installation(self, _transaction: Any) -> None:
        self.events.append("rollback-guided-gud-and-qkvo")

    def handoff_v4_for_publication(
        self,
        _transaction: Any,
        _contract: Any,
    ) -> dict[str, Any]:
        self.events.append("handoff-qkvo-and-gud")
        return {
            "handoff_complete": True,
            "transaction_qkvo_handle_is_guided_composite": True,
            "transaction_gud_handle_is_guided_composite": True,
            "old_only_publication_paths_disabled": True,
            "sequential_dense_bake": [
                "lexical",
                "inherited-and-guided-qkvo",
                "inherited-and-guided-gud",
            ],
        }

    def evaluate_publication_split(
        self,
        _model: Any,
        _parent: Any,
        contract: Any,
        split_name: str,
    ) -> dict[str, Any]:
        self.events.append(f"evaluate-{split_name}")
        if split_name == "sealed":
            assert contract.sealed_marker.exists()
            self.sealed_forward_count += 1
            return _metrics(split="heldout-evaluation")
        if split_name == "fit":
            return _metrics(split="calibration-exploratory")
        result = dict(contract.winner["metrics"])
        if not self.candidate_matches and self.events.count("evaluate-candidate") == 1:
            result["action_accuracy"] = 0.0
        return result

    def save_reload_certify(
        self,
        _model: Any,
        _tokenizer: Any,
        output: Path,
    ) -> dict[str, Any]:
        self.events.append("save-reload")
        child = output / "accepted_child"
        child.mkdir()
        (child / "model.bin").write_bytes(b"standalone-child")
        tree = artifact_tree(child)
        return {
            "path": str(child),
            "fresh_reload_performed": True,
            "exact_budget_match": True,
            "exact_unique_parameters": stage.QWEN3_1P4B_EXACT_PARAMETERS,
            "state_keys_match_pre_save": True,
            "state_bytes_match_pre_save": True,
            "state_dict_sha256": _digest("published-state"),
            "published_adapter_parameter_count": 0,
            "published_parent_runtime_reference_count": 0,
            "published_source_sized_sidecar_parameter_count": 0,
            "artifact_tree": tree,
        }

    def independent_reload_certify(
        self,
        child: Path,
        expected_parameters: int,
    ) -> dict[str, Any]:
        self.events.append("independent-reload")
        return {
            "fresh_reload_performed": True,
            "loaded_from_candidate_path": True,
            "exact_unique_parameters": expected_parameters,
            "state_dict_sha256": _digest("published-state"),
            "published_adapter_parameter_count": 0,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_parameter_count": 0,
            "artifact_tree": artifact_tree(child),
        }


def _execution_contract(
    tmp_path: Path,
) -> publication.AuthenticatedGuidedGUDWinnerPublication:
    winner = _winner()

    def split_bindings(prefix: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name in ("fit", "candidate", "sealed"):
            path = tmp_path / f"{prefix}-{name}.json"
            path.write_text("{}\n", encoding="utf-8")
            result[name] = {"path": str(path), "sha256": file_sha256(path)}
        return result

    guided_source_run = tmp_path / "guided-source-run-report.json"
    guided_source_factor = tmp_path / "guided-source-factor-pack.pt"
    guided_audit_prereg = tmp_path / "guided-audit-preregistration.json"
    guided_audit_run = tmp_path / "guided-audit-run-report.json"
    guided_basis = tmp_path / "guided-qkvo-basis-pack.pt"
    guided_source_run.write_text("{}\n", encoding="utf-8")
    guided_source_factor.write_bytes(b"guided-source-factor-pack")
    guided_audit_prereg.write_text("{}\n", encoding="utf-8")
    guided_audit_run.write_text("{}\n", encoding="utf-8")
    guided_basis.write_bytes(b"guided-qkvo-basis")
    guided_splits = split_bindings("guided")
    compiler_report = _compiler_report()
    source_v2_factor_sha = _digest("source-v2")
    qkvo_basis_tensor_shas = {
        role: _digest(f"qkvo-basis:{role}") for role in ("q", "k", "v", "o")
    }
    guided_prereg = tmp_path / "source-guided-preregistration.json"
    _write_json(
        guided_prereg,
        {
            "schema": (
                "neural-forge-qwen-exact-target-native-guided-"
                "continuation-preregistration-v1"
            ),
            "evaluation_contract_sha256": _digest("guided-evaluation"),
            "thresholds": {
                name: _thresholds() for name in ("fit", "candidate", "sealed")
            },
            "source": {
                "run_report": {
                    "path": str(guided_source_run),
                    "sha256": file_sha256(guided_source_run),
                },
                "factor_pack": {
                    "path": str(guided_source_factor),
                    "sha256": file_sha256(guided_source_factor),
                },
                "factor_state_sha256": source_v2_factor_sha,
                "compiler_report_sha256": publication.canonical_sha256(
                    compiler_report
                ),
            },
            "audit": {
                "preregistration": {
                    "path": str(guided_audit_prereg),
                    "sha256": file_sha256(guided_audit_prereg),
                },
                "run_report": {
                    "path": str(guided_audit_run),
                    "sha256": file_sha256(guided_audit_run),
                },
                "guided_basis_pack": {
                    "path": str(guided_basis),
                    "sha256": file_sha256(guided_basis),
                },
                "basis_envelope_file_sha256": file_sha256(guided_basis),
                "audit_report_self_sha256": _digest("qkvo-audit-report"),
                "basis_payload_sha256": _digest("qkvo-basis-payload"),
                "basis_manifest_sha256": _digest("qkvo-basis-manifest"),
                "basis_tensor_sha256s": qkvo_basis_tensor_shas,
            },
            "splits": guided_splits,
        },
    )
    source_v3_evaluation = _digest("source-v3-evaluation")
    source_v3_splits = split_bindings("source-v3")
    source_v3_site_mask = tmp_path / "source-v3-site-mask.json"
    source_v3_site_mask.write_text("{}\n", encoding="utf-8")
    source_v3_prereg = tmp_path / "source-v3-tournament-preregistration.json"
    _write_json(
        source_v3_prereg,
        {
            "schema": (
                "neural-forge-qwen-exact-target-native-guided-candidate-"
                "tournament-preregistration-v1"
            ),
            "sealed_before_model_access": True,
            "source_guided": {
                "path": str(guided_prereg),
                "sha256": file_sha256(guided_prereg),
            },
            "splits": source_v3_splits,
            "site_mask": {
                "path": str(source_v3_site_mask),
                "sha256": file_sha256(source_v3_site_mask),
            },
            "evaluation_contract_sha256": source_v3_evaluation,
        },
    )
    source_v3_run = tmp_path / "source-v3-tournament-run-report.json"
    source_v3_winner = tmp_path / "source-v3-winner-selection.json"
    source_v3_frontier = tmp_path / "source-v3-candidate-frontier.json"
    source_v3_run.write_text("{}\n", encoding="utf-8")
    source_v3_winner.write_text("{}\n", encoding="utf-8")
    source_v3_frontier.write_text("{}\n", encoding="utf-8")
    source_v3_factor = tmp_path / "source-v3-factor-pack.pt"
    source_v3_factor.write_bytes(b"source-v3-factor-pack")
    source_v3_runtime = _digest("source-runtime-v3")
    source_v3_portable = _digest("source-portable-v3")
    target_evaluation = _digest("guided-gud-evaluation")
    v4_splits = split_bindings("v4")
    v4_basis = tmp_path / "v4-guided-gud-basis-bundle.pt"
    v4_site_mask = tmp_path / "v4-guided-gud-site-mask.json"
    v4_basis.write_bytes(b"v4-guided-gud-basis")
    v4_site_mask.write_text("{}\n", encoding="utf-8")
    v4_tournament_prereg = tmp_path / "v4-tournament-preregistration.json"
    _write_json(
        v4_tournament_prereg,
        {
            "schema": (
                "neural-forge-qwen-exact-target-native-guided-gud-"
                "candidate-tournament-preregistration-v1"
            ),
            "sealed_before_model_access": True,
            "source_v3_winner": {
                "tournament_preregistration": {
                    "path": str(source_v3_prereg),
                    "sha256": file_sha256(source_v3_prereg),
                },
                "tournament_run_report": {
                    "path": str(source_v3_run),
                    "sha256": file_sha256(source_v3_run),
                },
                "winner_selection": {
                    "path": str(source_v3_winner),
                    "sha256": file_sha256(source_v3_winner),
                },
                "expected_winner": {
                    "runtime_factor_state_sha256": source_v3_runtime,
                    "portable_factor_state_sha256": source_v3_portable,
                    "factor_pack_file_sha256": file_sha256(source_v3_factor),
                },
            },
            "splits": v4_splits,
            "basis_bundle": {
                "path": str(v4_basis),
                "sha256": file_sha256(v4_basis),
            },
            "site_mask": {
                "path": str(v4_site_mask),
                "sha256": file_sha256(v4_site_mask),
            },
            "evaluation_contract_sha256": target_evaluation,
        },
    )
    v4_run = tmp_path / "v4-tournament-run-report.json"
    v4_selection = tmp_path / "v4-winner-selection.json"
    v4_frontier = tmp_path / "candidate_frontier.json"
    v4_winner_factor = tmp_path / "winner-factor-pack.pt"
    for path, payload in (
        (v4_run, b"{}\n"),
        (v4_selection, b"{}\n"),
        (v4_frontier, b"{}\n"),
        (v4_winner_factor, b"winner-v4-factor-pack"),
    ):
        path.write_bytes(payload)
    winner["factor_pack"] = str(v4_winner_factor)
    winner["factor_pack_sha256"] = file_sha256(v4_winner_factor)
    winner["candidate_frontier"] = {
        "schema": publication.tournament.FRONTIER_SCHEMA,
        "path": str(v4_frontier),
        "sha256": file_sha256(v4_frontier),
        "observed_callback_count": 96,
        "pending_callback_count": 0,
        "completed_arms": list(publication.tournament.ARM_IDS),
    }
    splits = v4_splits
    gud_basis_tensor_shas = {
        role: _digest(f"gud-basis:{role}") for role in ("gate", "up", "down")
    }
    semantic_contract = {
        "guided_gud_tournament_evaluation_contract_sha256": target_evaluation,
        "source_v3_tournament_evaluation_contract_sha256": source_v3_evaluation,
        "source_v3_run_report_sha256": file_sha256(source_v3_run),
        "source_v3_winner_selection_sha256": file_sha256(source_v3_winner),
        "source_v3_candidate_frontier_sha256": file_sha256(source_v3_frontier),
        "source_v3_factor_pack_file_sha256": file_sha256(source_v3_factor),
        "winner_portable_v4_factor_state_sha256": winner[
            "portable_factor_state_sha256"
        ],
        "winner_candidate_metrics_sha256": winner["metrics_sha256"],
        "candidate_frontier_sha256": winner["candidate_frontier"]["sha256"],
        "winner_basis_payload_sha256": _digest("gud-basis-payload"),
        "winner_basis_bundle_file_sha256": file_sha256(v4_basis),
        "winner_basis_audit_report_sha256": _digest("gud-audit-report"),
        "winner_basis_manifest_sha256": _digest("gud-basis-manifest"),
        "winner_source_basis_tensor_sha256s": gud_basis_tensor_shas,
        "tournament_pareto_admission_decision_sha256": winner[
            "tournament_pareto_admission"
        ]["decision_sha256"],
        "splits": splits,
    }
    semantic = publication.canonical_sha256(semantic_contract)
    prereg_payload = {
        "schema": publication.PREREGISTRATION_SCHEMA,
        "tournament": {
            "preregistration": str(v4_tournament_prereg),
            "preregistration_sha256": file_sha256(v4_tournament_prereg),
            "run_report": str(v4_run),
            "run_report_sha256": file_sha256(v4_run),
            "winner_selection": str(v4_selection),
            "winner_selection_sha256": file_sha256(v4_selection),
            "candidate_frontier": dict(winner["candidate_frontier"]),
            "expected_arm_count": 8,
            "expected_callback_count": 96,
        },
        "source_v3": {
            "tournament_preregistration": str(source_v3_prereg),
            "tournament_preregistration_sha256": file_sha256(source_v3_prereg),
            "tournament_run_report": str(source_v3_run),
            "tournament_run_report_sha256": file_sha256(source_v3_run),
            "winner_selection": str(source_v3_winner),
            "winner_selection_sha256": file_sha256(source_v3_winner),
            "candidate_frontier": {
                "path": str(source_v3_frontier),
                "sha256": file_sha256(source_v3_frontier),
            },
            "runtime_factor_state_sha256": source_v3_runtime,
            "portable_factor_state_sha256": source_v3_portable,
            "factor_pack": str(source_v3_factor),
            "factor_pack_sha256": file_sha256(source_v3_factor),
        },
        "winner": winner,
        "splits": splits,
        "sealed_consumption": {
            "marker": str(tmp_path / "sealed_consumption.json"),
            "schema": publication.SEALED_CONSUMPTION_SCHEMA,
            "must_be_absent_at_preflight": True,
            "exclusive_create_immediately_before_forward": True,
            "maximum_forward_count": 1,
        },
        "runtime": {
            "dtype": "float16",
            "parent_device_map": "cuda:0",
            "target_device_map": "cuda:1",
        },
        "implementation": {"runner_sha256": _digest("runner")},
        "semantic_contract": semantic_contract,
        "semantic_contract_sha256": semantic,
        "model_free_preregistration": True,
        "model_weights_loaded": False,
        "gpu_work_launched": False,
        "publication_authorized_before_sealed_pass": False,
    }
    prereg = tmp_path / "preregistration.json"
    prereg.write_text(
        json.dumps(prereg_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_guided = SimpleNamespace(execution_contract=SimpleNamespace())
    source = SimpleNamespace(
        tournament_contract=SimpleNamespace(source=source_guided),
        factor_payload={"schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA},
    )
    tournament_contract = SimpleNamespace(
        source=source,
        splits={
            "candidate": SimpleNamespace(sha256=splits["candidate"]["sha256"]),
            "sealed": SimpleNamespace(sha256=splits["sealed"]["sha256"]),
        },
        schedule=SimpleNamespace(as_dict=lambda: {"steps": 64}),
    )
    return publication.AuthenticatedGuidedGUDWinnerPublication(
        path=prereg,
        file_sha256=file_sha256(prereg),
        tournament=tournament_contract,
        winner=winner,
        factor_payload={"schema": BEHAVIOR_FACTOR_PACK_V4_SCHEMA},
        sealed_marker=tmp_path / "sealed_consumption.json",
        runtime={
            "dtype": "float16",
            "parent_device_map": "cuda:0",
            "target_device_map": "cuda:1",
        },
        raw=prereg_payload,
    )


@pytest.fixture
def execution_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        publication.tournament.v3_tournament,
        "validate_candidate_tournament_cuda_preflight",
        lambda *_a, **_kw: {
            "status": "passed",
            "models_loaded": False,
            "model_forward_calls": 0,
            "gpu_kernel_work_launched": False,
        },
    )
    monkeypatch.setattr(
        publication.dataclasses,
        "replace",
        lambda _source, **_kw: SimpleNamespace(
            thresholds={
                name: stage.BehaviorThresholds.from_mapping(
                    _thresholds(), f"thresholds.{name}"
                )
                for name in ("fit", "candidate", "sealed")
            }
        ),
    )
    monkeypatch.setattr(stage, "_validate_continuation_compilation", lambda *_a: None)
    states = iter(
        (
            SimpleNamespace(schema="v2", factor_state_sha256=_digest("source-v2")),
            SimpleNamespace(
                schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
                factor_state_sha256=_digest("runtime-v3"),
            ),
        )
    )
    monkeypatch.setattr(
        publication,
        "capture_target_native_factor_state",
        lambda *_a, **_kw: next(states),
    )
    monkeypatch.setattr(
        publication,
        "authenticate_guided_factor_sources",
        lambda *_a: SimpleNamespace(name="qkvo-binding"),
    )
    monkeypatch.setattr(
        publication,
        "rebind_recovered_v3_factor_payload",
        lambda *_a: {
            "schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            "factor_state_sha256": _digest("runtime-v3"),
        },
    )
    monkeypatch.setattr(
        publication,
        "authenticate_guided_gud_factor_sources",
        lambda *_a: SimpleNamespace(name="gud-binding"),
    )
    monkeypatch.setattr(
        publication,
        "rebind_recovered_v4_factor_payload",
        lambda *_a: {
            "schema": BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
            "factor_state_sha256": _digest("runtime-v4"),
        },
    )

    def restore(_transaction: Any, payload: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            schema=payload["schema"],
            factor_state_sha256=payload["factor_state_sha256"],
        )

    monkeypatch.setattr(publication, "restore_target_native_factor_payload", restore)
    monkeypatch.setattr(
        publication,
        "portable_v4_factor_state_sha256",
        lambda *_a: _digest("portable-v4"),
    )


def _execute(
    contract: publication.AuthenticatedGuidedGUDWinnerPublication,
    output: Path,
    runtime: _FakeRuntime,
) -> dict[str, Any]:
    return publication.execute_winner_publication(
        contract,
        output,
        dtype="float16",
        parent_device_map="cuda:0",
        target_device_map="cuda:1",
        runtime=runtime,
    )


def test_publication_restores_both_factor_families_and_seals_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    events: list[str] = []
    runtime = _FakeRuntime(events)
    report = _execute(contract, tmp_path / "run", runtime)

    assert report["status"] == "accepted"
    assert report["exact_unique_parameters"] == 1_410_567_168
    assert report["published_adapter_parameter_count"] == 0
    assert report["published_parent_runtime_reference_count"] == 0
    assert report["published_source_sized_sidecar_parameter_count"] == 0
    assert report["sealed_forward_count"] == 1
    assert report["post_bake_sealed_forward_count"] == 0
    assert report["sealed_used_for_reranking"] is False
    assert events.index("install-guided-qkvo") < events.index("install-guided-gud")
    assert events.index("evaluate-candidate") < events.index("evaluate-sealed")
    assert events.index("handoff-qkvo-and-gud") < events.index("bake-qkvo-and-gud")
    assert events.count("evaluate-sealed") == 1
    assert events[-4:] == [
        "evaluate-fit",
        "evaluate-candidate",
        "save-reload",
        "independent-reload",
    ]
    assert report["post_bake_behavior_decisions"]["sealed"] is None
    assert (tmp_path / "sealed_consumption.json").is_file()
    assert (tmp_path / "run" / "accepted_child").is_dir()
    with pytest.raises(
        publication.GuidedGUDWinnerPublicationError,
        match="already consumed",
    ):
        _execute(contract, tmp_path / "rerun", runtime)
    assert runtime.load_count == 1
    assert not (tmp_path / "rerun").exists()


def _authenticate_v4_terminal_candidate(
    contract: publication.AuthenticatedGuidedGUDWinnerPublication,
    output: Path,
    report: dict[str, Any],
) -> Any:
    return authenticate_terminal_candidate(
        output,
        expected_run_report_sha256=file_sha256(output / "run_report.json"),
        expected_preregistration_sha256=contract.file_sha256,
        preregistration_authenticator=lambda *_a: dict(contract.raw),
        accepted_child_authenticator=lambda *_a: report[
            "independent_reload_certificate"
        ],
    )


def test_frontier_authenticates_registers_and_inspects_v4_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))

    candidate = _authenticate_v4_terminal_candidate(contract, output, report)
    assert candidate.terminal_run_schema == publication.RUN_SCHEMA
    assert candidate.selected_factor_state_sha256 == _digest("portable-v4")
    assert candidate.target_contract_sha256 == contract.raw["semantic_contract"][
        "guided_gud_tournament_evaluation_contract_sha256"
    ]
    assert candidate.exact_unique_parameters == 1_410_567_168
    assert any(
        name.startswith("v4_candidate_frontier_")
        for name in candidate.protected_roots
    )
    assert any(
        name.startswith("source_guided_preregistration_")
        for name in candidate.protected_roots
    )

    frontier = ExactCandidateFrontier(tmp_path / "frontier")
    retained = frontier.register(candidate)
    assert retained["candidate_retained"] is True
    inspected = ExactCandidateFrontier(tmp_path / "frontier").inspect()
    assert inspected["retained_record_count"] == 1


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("marker", "marker bytes"),
        ("portable-factor", "factor provenance"),
        ("sealed-rerank", "one-shot exact-child evidence"),
        ("dual-handoff", "dual-composite"),
        ("sidecar", "not exact and parent-free"),
    ),
)
def test_frontier_rejects_v4_terminal_evidence_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
    tamper: str,
    message: str,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    report_path = output / "run_report.json"
    tampered = json.loads(json.dumps(report))
    if tamper == "marker":
        contract.sealed_marker.write_text("{}\n", encoding="utf-8")
    elif tamper == "portable-factor":
        tampered["winner_portable_v4_factor_state_sha256"] = _digest("forged")
    elif tamper == "sealed-rerank":
        tampered["sealed_used_for_reranking"] = True
    elif tamper == "dual-handoff":
        tampered["guided_v4_publication_handoff"][
            "transaction_gud_handle_is_guided_composite"
        ] = False
    else:
        tampered["fresh_reload_certificate"][
            "published_source_sized_sidecar_parameter_count"
        ] = 1
    report_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")

    with pytest.raises(ExactCandidateFrontierError, match=message):
        _authenticate_v4_terminal_candidate(contract, output, report)


@pytest.mark.parametrize(
    "receipt_field",
    (
        "compiler_report",
        "inherited_installation",
        "source_v2_restore",
        "guided_qkvo_installation",
        "guided_gud_installation",
        "publication",
    ),
)
def test_frontier_rejects_missing_v4_compiler_install_and_bake_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
    receipt_field: str,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    tampered = json.loads(json.dumps(report))
    tampered.pop(receipt_field)
    _write_json(output / "run_report.json", tampered)

    with pytest.raises(ExactCandidateFrontierError, match="provenance|schema"):
        _authenticate_v4_terminal_candidate(contract, output, report)


def test_frontier_runtime_v4_sha_cannot_relabel_portable_winner_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    substituted = _digest("attacker-controlled-runtime-v4")
    tampered = json.loads(json.dumps(report))
    tampered["runtime_rebound_v4_factor_state_sha256"] = substituted
    _write_json(output / "run_report.json", tampered)

    candidate = _authenticate_v4_terminal_candidate(contract, output, report)
    assert candidate.selected_factor_state_sha256 == _digest("portable-v4")
    assert candidate.selected_factor_state_sha256 != substituted


def test_frontier_inspect_rejects_v4_to_v3_to_guided_threshold_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    candidate = _authenticate_v4_terminal_candidate(contract, output, report)
    frontier_root = tmp_path / "frontier"
    ExactCandidateFrontier(frontier_root).register(candidate)

    guided_prereg = tmp_path / "source-guided-preregistration.json"
    payload = json.loads(guided_prereg.read_text(encoding="utf-8"))
    payload["thresholds"]["candidate"]["minimum_action_accuracy"] = 0.0
    guided_prereg.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(
        ExactCandidateFrontierError,
        match="schema/dynamic method decision",
    ):
        ExactCandidateFrontier(frontier_root).inspect()


@pytest.mark.parametrize(
    ("root_prefix", "mutation"),
    (
        ("v4_basis_bundle_", "tamper"),
        ("v4_site_mask_", "missing"),
        ("v4_split_candidate_", "tamper"),
        ("source_v3_tournament_run_report_", "missing"),
        ("source_v3_winner_selection_", "tamper"),
        ("source_v3_candidate_frontier_", "missing"),
        ("source_v3_factor_pack_", "tamper"),
        ("source_v3_site_mask_", "missing"),
        ("source_v3_split_sealed_", "tamper"),
        ("source_guided_input_run_report_", "missing"),
        ("source_guided_input_factor_pack_", "tamper"),
        ("source_guided_audit_preregistration_", "missing"),
        ("source_guided_audit_run_report_", "tamper"),
        ("source_guided_basis_pack_", "missing"),
        ("source_guided_split_fit_", "tamper"),
    ),
)
def test_frontier_inspect_rejects_transitive_v4_source_root_tamper_or_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
    root_prefix: str,
    mutation: str,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    candidate = _authenticate_v4_terminal_candidate(contract, output, report)
    frontier_root = tmp_path / "frontier"
    ExactCandidateFrontier(frontier_root).register(candidate)
    protected = next(
        Path(path)
        for name, path in candidate.protected_roots.items()
        if name.startswith(root_prefix)
    )
    if mutation == "missing":
        protected.unlink()
    else:
        protected.write_bytes(b"tampered-transitive-evidence")

    with pytest.raises(
        ExactCandidateFrontierError,
        match="protected path|schema/dynamic method decision",
    ):
        ExactCandidateFrontier(frontier_root).inspect()


def test_frontier_manifest_protects_all_transitive_v4_roots_from_prune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    candidate = _authenticate_v4_terminal_candidate(contract, output, report)
    frontier_root = tmp_path / "frontier"
    frontier = ExactCandidateFrontier(frontier_root)
    frontier.register(candidate)
    manifest = json.loads(
        (frontier_root / "frontier_manifest.json").read_text(encoding="utf-8")
    )
    protected_names = set(manifest["protected_roots"])
    required_prefixes = {
        "v4_basis_bundle_",
        "v4_site_mask_",
        "v4_split_fit_",
        "v4_split_candidate_",
        "v4_split_sealed_",
        "source_v3_tournament_run_report_",
        "source_v3_winner_selection_",
        "source_v3_candidate_frontier_",
        "source_v3_factor_pack_",
        "source_v3_site_mask_",
        "source_v3_split_fit_",
        "source_v3_split_candidate_",
        "source_v3_split_sealed_",
        "source_guided_input_run_report_",
        "source_guided_input_factor_pack_",
        "source_guided_audit_preregistration_",
        "source_guided_audit_run_report_",
        "source_guided_basis_pack_",
        "source_guided_split_fit_",
        "source_guided_split_candidate_",
        "source_guided_split_sealed_",
    }
    assert all(
        any(name.startswith(prefix) for name in protected_names)
        for prefix in required_prefixes
    )
    with pytest.raises(ExactCandidateFrontierError, match="Actual deletion is unsupported"):
        frontier.apply_pending_prune()
    assert all(Path(path).exists() for path in manifest["protected_roots"].values())


def test_frontier_inspect_rejects_retained_v4_marker_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    output = tmp_path / "run"
    report = _execute(contract, output, _FakeRuntime([]))
    candidate = _authenticate_v4_terminal_candidate(contract, output, report)
    frontier_root = tmp_path / "frontier"
    ExactCandidateFrontier(frontier_root).register(candidate)

    contract.sealed_marker.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        ExactCandidateFrontierError,
        match="schema/dynamic method decision",
    ):
        ExactCandidateFrontier(frontier_root).inspect()


def test_retention_authenticates_consumed_v4_prereg_without_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _execution_contract(tmp_path)

    def forbidden_preflight() -> dict[str, Any]:
        raise AssertionError("retention must not rerun consumed-marker preflight")

    monkeypatch.setattr(
        retention,
        "_load_guided_gud_winner_publication_runner",
        lambda: SimpleNamespace(
            authenticate_winner_publication_preregistration=lambda *_a: (
                SimpleNamespace(raw=contract.raw, preflight_report=forbidden_preflight)
            )
        ),
    )
    authenticated = retention._canonical_preregistration_authenticator(
        contract.path,
        contract.file_sha256,
    )
    assert authenticated == contract.raw


def test_candidate_recheck_blocks_sealed_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_patches: None,
) -> None:
    del execution_patches
    contract = _execution_contract(tmp_path)
    contract.winner["tournament_pareto_admission"]["passed"] = False
    monkeypatch.setattr(
        publication,
        "authenticate_winner_publication_preregistration",
        lambda *_a: contract,
    )
    events: list[str] = []
    runtime = _FakeRuntime(events)
    report = _execute(contract, tmp_path / "rejected", runtime)

    assert report["status"] == "rejected"
    assert report["reason"] == "v4-winner-candidate-recheck-rejected"
    assert report["sealed_forward_count"] == 0
    assert "evaluate-sealed" not in events
    assert "handoff-qkvo-and-gud" not in events
    assert "bake-qkvo-and-gud" not in events
    assert not (tmp_path / "sealed_consumption.json").exists()


def test_dual_composite_handoff_rebinds_both_transaction_handles() -> None:
    events: list[str] = []

    class Composite:
        def __init__(self, inherited: Any, name: str) -> None:
            self.inherited_handle = inherited
            self.name = name

        def assert_live(self) -> None:
            events.append(f"assert-{self.name}")

    transaction = SimpleNamespace(qkvo_handle=object(), swiglu_handle=object())
    runtime = object.__new__(
        publication.ProductionGuidedGUDWinnerPublicationRuntime
    )
    runtime._guided_tranche = Composite(transaction.qkvo_handle, "qkvo")
    runtime._guided_gud_tranche = Composite(transaction.swiglu_handle, "gud")

    receipt = runtime.handoff_v4_for_publication(transaction, SimpleNamespace())

    assert transaction.qkvo_handle is runtime._guided_tranche
    assert transaction.swiglu_handle is runtime._guided_gud_tranche
    assert events == ["assert-qkvo", "assert-gud"]
    assert receipt["transaction_qkvo_handle_is_guided_composite"] is True
    assert receipt["transaction_gud_handle_is_guided_composite"] is True
