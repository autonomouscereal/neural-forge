from __future__ import annotations

# ruff: noqa: E402

import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_exact_target_native_guided_gud_basis_audit as producer
import run_qwen_exact_target_native_guided_gud_candidate_tournament as tournament
import neural_forge.hf_gud_output_jacobian_replay as replay
import neural_forge.hf_guided_checkpoint_recovery as recovery
import neural_forge.hf_target_native_behavior_distillation as behavior
from neural_forge.hf_gud_output_jacobian_audit import GUDStageGeometry
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
    BehaviorDomainSuite,
    BehaviorPrompt,
    CausalBehaviorMetrics,
    DurableBehaviorCheckpoint,
    DurableFactorStateSnapshot,
    DurableFactorTensorSnapshot,
    GuidedFactorSourceBinding,
    GuidedGUDFactorSourceBinding,
)


BASE_NAMES = {
    *(
        f"qkvo.{role}_{side}"
        for role in ("q", "k", "v", "o")
        for side in ("left", "right")
    ),
    *(
        f"gud.{role}_{side}"
        for role in ("gate", "up", "down")
        for side in ("left", "right")
    ),
    *(
        f"lexical.{role}_{side}"
        for role in ("embedding", "lm_head")
        for side in ("left", "right")
    ),
}
GUIDED_NAMES = {
    f"guided_qkvo.{role}_{side}"
    for role in ("q", "k", "v", "o")
    for side in ("left", "right")
}
GUIDED_GUD_NAMES = {
    f"guided_gud.{role}_{side}"
    for role in ("gate", "up", "down")
    for side in ("left", "right")
}
V3_NAMES = tuple(sorted(BASE_NAMES | GUIDED_NAMES))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source_binding() -> GuidedFactorSourceBinding:
    return GuidedFactorSourceBinding(
        source_v2_factor_state_sha256=_digest("source-v2"),
        source_v2_tensor_sha256s=tuple(
            (name, _digest(f"source:{name}")) for name in sorted(BASE_NAMES)
        ),
        audit_report_sha256=_digest("audit"),
        basis_envelope_file_sha256=_digest("envelope"),
        basis_payload_sha256=_digest("payload"),
        basis_manifest_sha256=_digest("manifest"),
        source_basis_tensor_sha256s=tuple(
            (role, _digest(f"basis:{role}")) for role in ("q", "k", "v", "o")
        ),
        inherited_installation_sha256=_digest("installation"),
        inherited_factor_state_sha256_at_install=_digest("inherited"),
        adapter_site_indices_sha256=_digest("adapter-sites"),
        audit_site_indices_sha256=_digest("audit-sites"),
    )


def _v3_state(seed: int = 3) -> DurableFactorStateSnapshot:
    binding = _source_binding()
    tensors = []
    rows = []
    for index, name in enumerate(V3_NAMES):
        value = torch.tensor([seed, index + 1], dtype=torch.float32)
        digest = behavior._tensor_sha256(value)
        tensors.append(
            DurableFactorTensorSnapshot(
                name=name,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                tensor_sha256=digest,
                raw_bytes=value.view(torch.uint8).numpy().tobytes(),
            )
        )
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": digest,
            }
        )
    factor_sha = behavior._factor_state_sha256(
        rows,
        schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=4,
        guided_source_binding=binding,
    )
    return DurableFactorStateSnapshot(
        schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        tensors=tuple(tensors),
        factor_state_sha256=factor_sha,
        lexical_rank=4,
        guided_source_binding=binding,
    )


def _v4_state(seed: int = 7) -> DurableFactorStateSnapshot:
    source = _v3_state(seed)
    assert isinstance(source.guided_source_binding, GuidedFactorSourceBinding)
    source_rows = [
        {
            "name": value.name,
            "shape": list(value.shape),
            "dtype": value.dtype,
            "tensor_sha256": value.tensor_sha256,
        }
        for value in source.tensors
    ]
    gud_binding = GuidedGUDFactorSourceBinding(
        source_v3_factor_state_sha256=source.factor_state_sha256,
        source_v3_factor_manifest_sha256=behavior._canonical_sha256(source_rows),
        source_v3_tensor_sha256s=tuple(
            (value.name, value.tensor_sha256) for value in source.tensors
        ),
        source_v3_lexical_rank=4,
        source_guided_qkvo_binding=source.guided_source_binding,
        audit_report_sha256=_digest("gud-audit"),
        basis_envelope_file_sha256=_digest("gud-envelope"),
        basis_payload_sha256=_digest("gud-payload"),
        basis_manifest_sha256=_digest("gud-manifest"),
        source_basis_tensor_sha256s=tuple(
            (role, _digest(f"gud-basis:{role}"))
            for role in ("gate", "up", "down")
        ),
        inherited_installation_sha256=_digest("gud-installation"),
        inherited_factor_state_sha256_at_install=_digest("gud-inherited"),
        adapter_site_indices_sha256=_digest("gud-adapter-sites"),
        audit_site_indices_sha256=_digest("gud-audit-sites"),
    )
    snapshots = list(source.tensors)
    for index, name in enumerate(sorted(GUIDED_GUD_NAMES)):
        value = torch.tensor([seed, 100 + index], dtype=torch.float32)
        digest = behavior._tensor_sha256(value)
        snapshots.append(
            DurableFactorTensorSnapshot(
                name=name,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                tensor_sha256=digest,
                raw_bytes=value.view(torch.uint8).numpy().tobytes(),
            )
        )
    snapshots.sort(key=lambda value: value.name)
    rows = [
        {
            "name": value.name,
            "shape": list(value.shape),
            "dtype": value.dtype,
            "tensor_sha256": value.tensor_sha256,
        }
        for value in snapshots
    ]
    factor_sha = behavior._factor_state_sha256(
        rows,
        schema=BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        lexical_rank=4,
        guided_source_binding=source.guided_source_binding,
        guided_gud_source_binding=gud_binding,
    )
    return DurableFactorStateSnapshot(
        schema=BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
        tensors=tuple(snapshots),
        factor_state_sha256=factor_sha,
        lexical_rank=4,
        guided_source_binding=source.guided_source_binding,
        guided_gud_source_binding=gud_binding,
    )


def _metrics() -> CausalBehaviorMetrics:
    return CausalBehaviorMetrics(
        split="calibration-selection",
        action_count=24,
        action_agreement_count=3,
        action_accuracy=0.125,
        parent_action_mean_logprob=-4.0,
        parent_action_topk_fraction=0.1,
        parent_action_mean_rank=100.0,
        parent_action_worst_rank=500,
        rollout_count=24,
        exact_rollout_count=0,
        exact_rollout_fraction=0.0,
        rollout_token_count=192,
        token_match_count=5,
        token_match_fraction=5 / 192,
        lcp_token_count=2,
        lcp_fraction=2 / 192,
        lcp_at_least_4_count=0,
        repeat_fraction_excess=0.1,
        adjacent_repeat_fraction_excess=0.2,
        longest_run_excess=2,
        report_sha256=_digest("metrics"),
    )


def _patch_small_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tournament,
        "EXPECTED_GEOMETRY",
        GUDStageGeometry(4, 3, 5).to_dict(),
    )
    monkeypatch.setattr(tournament, "EXPECTED_RANK", 1)


def _replay_result(
    *,
    source_state: DurableFactorStateSnapshot | None = None,
    basis_domain: BehaviorDomainSuite | None = None,
    seed: int = 17,
) -> replay.GUDOutputJacobianReplayResult:
    source = _v3_state() if source_state is None else source_state
    payload = source.as_factor_payload()
    rows = payload["factors"]
    config = replay.GUDOutputJacobianReplayConfig(
        prompt_count=24,
        horizon=8,
        rank=1,
        sketch_width=2,
        balanced_envelope_width=2,
        balanced_stream_rows=1,
        fingerprint_lanes=2,
        fingerprint_chunk_elements=64,
        max_callback_rows=64,
        seed=seed,
        max_bytes=64 * 1024 * 1024,
    )
    packs = {
        method: {
            "gate": torch.full((4, 1, 3), float(index + 1)),
            "up": torch.full((4, 1, 3), float(index + 2)),
            "down": torch.full((4, 1, 5), float(index + 3)),
        }
        for index, method in enumerate(tournament.BASIS_METHODS)
    }
    manifest = [
        {
            "candidate": method,
            "role": role,
            "shape": list(packs[method][role].shape),
            "dtype": str(packs[method][role].dtype),
            "sha256": replay._tensor_sha256(packs[method][role]),
        }
        for method in tournament.BASIS_METHODS
        for role in tournament.GUD_ROLES
    ]
    diagnostics = [
        {
            "layer": layer,
            "role": role,
            "guided_capture_fraction": 0.2 + layer / 10,
            "guided_over_random_ratio": 1.5 + layer,
            "normal_fraction": 0.3 + layer / 20,
        }
        for layer in range(4)
        for role in tournament.GUD_ROLES
    ]
    domain = basis_domain or BehaviorDomainSuite(
        split_name="fit",
        prompts=tuple(BehaviorPrompt((index + 1,)) for index in range(24)),
        artifact_sha256=_digest("candidate24"),
    )
    schedule = tournament.guided_gud_basis_audit_schedule()
    report: dict[str, Any] = {
        "schema": replay.GUD_OUTPUT_JACOBIAN_REPLAY_SCHEMA,
        "basis_pack_schema": replay.GUD_COMPLETE_BASIS_PACK_SCHEMA,
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "geometry": GUDStageGeometry(4, 3, 5).to_dict(),
        "basis_domain": domain.as_dict(),
        "basis_domain_sha256": domain.domain_sha256,
        "basis_artifact_sha256": domain.artifact_sha256,
        "behavior_schedule": schedule.as_dict(),
        "behavior_schedule_sha256": tournament.canonical_sha256(schedule.as_dict()),
        "two_pass_replay_identical": True,
        "site_count": 12,
        "separate_projection_hook_count": 12,
        "gate_up_separate_hooks": True,
        "full_downstream_loss_output_delta_captured": True,
        "role_local_projection_input_captured": True,
        "internal_weight_matching_used": False,
        "source_v3_runtime_factor_state_sha256_before": source.factor_state_sha256,
        "source_v3_runtime_factor_state_sha256_after": source.factor_state_sha256,
        "source_v3_portable_factor_state_sha256": (
            recovery.portable_v3_factor_state_sha256(payload)
        ),
        "source_v3_factor_manifest_sha256": tournament.canonical_sha256(rows),
        "source_v3_tensor_sha256s": {
            row["name"]: row["tensor_sha256"] for row in rows
        },
        "source_v3_factor_count": 26,
        "source_v3_lexical_rank": 4,
        "source_guided_qkvo_binding": source.guided_source_binding.as_dict(),
        "source_guided_qkvo_binding_sha256": (
            source.guided_source_binding.binding_sha256
        ),
        "source_v3_state_unchanged": True,
        "site_diagnostics": diagnostics,
        "site_diagnostics_sha256": tournament.canonical_sha256(diagnostics),
        "basis_pack_manifest": manifest,
        "basis_pack_manifest_sha256": tournament.canonical_sha256(manifest),
        "candidate_order": list(tournament.BASIS_METHODS),
    }
    report["report_sha256"] = tournament.canonical_sha256(report)
    return replay.GUDOutputJacobianReplayResult(
        basis_packs=packs,
        site_reports=(),
        report=replay.GUDOutputJacobianReplayReport(report),
    )


def _save_bundle(
    tmp_path: Path,
    result: replay.GUDOutputJacobianReplayResult,
) -> tournament.AuthenticatedGuidedGUDBasisBundle:
    path = tmp_path / "basis.pt"
    torch.save(tournament.guided_gud_basis_bundle_from_audit(result), path)
    return tournament.authenticate_guided_gud_basis_bundle(
        path,
        tournament.stage._file_sha256(path),
    )


def test_exact_v4_arm_stage_and_retention_contract() -> None:
    assert tournament.ARM_IDS == tuple(
        f"{method}-{mode}"
        for method in tournament.BASIS_METHODS
        for mode in tournament.MASK_MODES
    )
    assert len(tournament.ARM_IDS) == 8
    assert tournament.EXPECTED_CALLBACK_COUNT == 96
    assert tournament.MAX_RETAINED_STATES == 3
    assert tournament.EXPECTED_GUIDED_GUD_PARAMETER_COUNT == 4_792_320
    stages = tournament.guided_gud_stage_contract()
    assert [(row["start_step"], row["end_step"]) for row in stages] == [
        (1, 8),
        (9, 24),
        (25, 64),
    ]
    assert [row["guided_gud_stage"] for row in stages] == [
        "guided-left",
        "guided",
        "joint",
    ]
    assert tournament.guided_gud_candidate_schedule().sealed_evaluation_policy == (
        "disabled"
    )


def test_replay_basis_roundtrip_and_tensor_tamper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_geometry(monkeypatch)
    result = _replay_result()
    bundle = _save_bundle(tmp_path, result)

    assert bundle.rank == 1
    assert set(bundle.source_v3_tensor_sha256s) == recovery._V3_FACTOR_NAMES
    assert bundle.source_v3_state_unchanged is True

    raw = torch.load(bundle.path, map_location="cpu", weights_only=True)
    replay_manifest = result.report.to_dict()["basis_pack_manifest"]
    installer_manifest = []
    for method in tournament.BASIS_METHODS:
        for role in tournament.GUD_ROLES:
            value = raw["guided_right_rows"][method][role]
            replay_digest = behavior._output_jacobian_tensor_sha256(value)
            installer_digest = tournament.gud_adapter._tensor_sha256(value)
            matching = next(
                row
                for row in replay_manifest
                if row["candidate"] == method and row["role"] == role
            )
            assert matching["sha256"] == replay_digest
            assert raw["source_basis_tensor_sha256s"][method][role] == (
                installer_digest
            )
            assert replay_digest != installer_digest
            installer_manifest.append(
                {
                    "candidate": method,
                    "role": role,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": installer_digest,
                }
            )
    assert bundle.replay_basis_manifest_sha256 == tournament.canonical_sha256(
        replay_manifest
    )
    assert bundle.basis_manifest_sha256 == tournament.canonical_sha256(
        installer_manifest
    )

    installation_probe = (
        tournament.gud_adapter.GuidedSparseDenseSwiGLUProjectionResidualRegistry(
            range(4),
            hidden_size=3,
            intermediate_size=5,
            rank=1,
            guided_right_rows=bundle.rows_for("balanced"),
            source_basis_tensor_sha256s=(
                bundle.source_basis_tensor_sha256s["balanced"]
            ),
            audit_report_sha256=bundle.audit_report_sha256,
            basis_envelope_file_sha256=bundle.file_sha256,
            basis_payload_sha256=bundle.payload_sha256,
            basis_manifest_sha256=bundle.basis_manifest_sha256,
            dtype=torch.float32,
            device="cpu",
        )
    )
    assert installation_probe.source_basis_tensor_sha256s == (
        bundle.source_basis_tensor_sha256s["balanced"]
    )

    tampered = copy.deepcopy(raw)
    tampered["guided_right_rows"]["balanced"]["gate"][0, 0, 0] += 1.0
    path = tmp_path / "tampered.pt"
    torch.save(tampered, path)
    with pytest.raises(tournament.GuidedGUDCandidateTournamentError):
        tournament.authenticate_guided_gud_basis_bundle(
            path,
            tournament.stage._file_sha256(path),
        )


def test_replay_bundle_normalizes_numpy_scalars_for_weights_only_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_geometry(monkeypatch)
    result = _replay_result()
    report = result.report.to_dict()
    report["ordered_site_reports"] = [
        {
            "balanced_dependency_report": {
                "rank_tolerances": {
                    "controllability": np.float64(1.25e-12),
                    "observability": np.float64(2.5e-12),
                    "hankel": np.float64(5.0e-12),
                }
            }
        }
    ]
    report.pop("report_sha256")
    report["report_sha256"] = tournament.canonical_sha256(report)
    canonical_before = tournament.canonical_json_bytes(report)
    production_shaped = replay.GUDOutputJacobianReplayResult(
        basis_packs=result.basis_packs,
        site_reports=result.site_reports,
        report=replay.GUDOutputJacobianReplayReport(report),
    )

    payload = tournament.guided_gud_basis_bundle_from_audit(production_shaped)
    normalized_report = payload["replay_audit_report"]
    tolerances = normalized_report["ordered_site_reports"][0][
        "balanced_dependency_report"
    ]["rank_tolerances"]
    assert all(type(value) is float for value in tolerances.values())
    assert tournament.canonical_json_bytes(normalized_report) == canonical_before
    assert normalized_report["report_sha256"] == report["report_sha256"]
    safe_metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"basis_payload_sha256", "guided_right_rows"}
    }
    pre_normalized_metadata = {
        **safe_metadata,
        "replay_audit_report": report,
    }
    assert tournament.canonical_json_bytes(pre_normalized_metadata) == (
        tournament.canonical_json_bytes(safe_metadata)
    )
    assert (
        tournament.canonical_sha256(pre_normalized_metadata)
        == payload["basis_payload_sha256"]
    )

    path = tmp_path / "numpy-scalar-normalized.pt"
    torch.save(payload, path)
    restricted = torch.load(path, map_location="cpu", weights_only=True)
    assert restricted["replay_audit_report"] == normalized_report
    authenticated = tournament.authenticate_guided_gud_basis_bundle(
        path,
        tournament.stage._file_sha256(path),
    )
    assert authenticated.audit_report_sha256 == report["report_sha256"]


def test_replay_bundle_rejects_full_config_seed_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_geometry(monkeypatch)
    result = _replay_result()
    report = result.report.to_dict()
    report["config"] = {**report["config"], "seed": report["config"]["seed"] + 1}
    report.pop("report_sha256")
    report["report_sha256"] = tournament.canonical_sha256(report)
    tampered = replay.GUDOutputJacobianReplayResult(
        basis_packs=result.basis_packs,
        site_reports=result.site_reports,
        report=replay.GUDOutputJacobianReplayReport(report),
    )

    with pytest.raises(
        tournament.GuidedGUDCandidateTournamentError,
        match="configuration|dimensions",
    ):
        tournament.guided_gud_basis_bundle_from_audit(tampered)


def test_mask_is_rederived_and_rejects_self_consistent_cherry_pick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_geometry(monkeypatch)
    bundle = _save_bundle(tmp_path, _replay_result())
    expected, _ = tournament.guided_gud_layer_mask_from_replay_report(
        bundle.replay_audit_report
    )
    crafted = expected.as_dict()
    crafted["gate_up_layer_mask"] = [True, False, True, False]
    crafted["derivation_report"]["gate_up_selected_layers"] = [0, 2]
    derivation = dict(crafted["derivation_report"])
    derivation.pop("derivation_sha256")
    crafted["derivation_report"]["derivation_sha256"] = (
        tournament.canonical_sha256(derivation)
    )
    crafted["derivation_sha256"] = crafted["derivation_report"][
        "derivation_sha256"
    ]
    crafted.pop("mask_contract_sha256")
    crafted["mask_contract_sha256"] = tournament.canonical_sha256(crafted)
    self_consistent = tournament.GuidedGUDLayerMask.from_dict(crafted)

    with pytest.raises(
        tournament.GuidedGUDCandidateTournamentError,
        match="deterministic replay derivation",
    ):
        tournament.authenticate_guided_gud_mask_derivation(
            self_consistent,
            bundle,
        )


def test_basis_source_rejects_an_arbitrary_fit_labeled_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_geometry(monkeypatch)
    state = _v3_state()
    expected_domain = BehaviorDomainSuite(
        split_name="fit",
        prompts=tuple(BehaviorPrompt((index + 1,)) for index in range(24)),
        artifact_sha256=_digest("candidate24"),
    )
    wrong_domain = BehaviorDomainSuite(
        split_name="fit",
        prompts=tuple(reversed(expected_domain.prompts)),
        artifact_sha256=expected_domain.artifact_sha256,
    )
    bundle = _save_bundle(
        tmp_path,
        _replay_result(source_state=state, basis_domain=wrong_domain),
    )
    candidate = SimpleNamespace(
        name="candidate",
        payload={"unused": True},
        sha256=expected_domain.artifact_sha256,
    )
    source = tournament.AuthenticatedV3WinnerSource(
        tournament_contract=SimpleNamespace(
            splits={"candidate": candidate, "sealed": SimpleNamespace(name="sealed")},
            source=SimpleNamespace(execution_contract=SimpleNamespace(seed=17)),
        ),
        tournament_preregistration_path=tmp_path / "source.json",
        tournament_preregistration_sha256=_digest("prereg"),
        run_report_path=tmp_path / "run.json",
        run_report_sha256=_digest("run"),
        winner_selection_path=tmp_path / "winner.json",
        winner_selection_sha256=_digest("winner"),
        arm_id=tournament.EXPECTED_SOURCE_ARM_ID,
        step=tournament.EXPECTED_SOURCE_STEP,
        runtime_factor_state_sha256=state.factor_state_sha256,
        portable_factor_state_sha256=recovery.portable_v3_factor_state_sha256(
            state.as_factor_payload()
        ),
        factor_pack_path=tmp_path / "factor.pt",
        factor_pack_file_sha256=_digest("factor"),
        factor_payload=state.as_factor_payload(),
        candidate_frontier={},
    )
    monkeypatch.setattr(tournament, "_prompt_records", lambda _binding: ())
    monkeypatch.setattr(
        tournament.split_tools,
        "_assert_globally_disjoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tournament.behavior,
        "load_behavior_domain_suite",
        lambda *_args, **_kwargs: expected_domain,
    )
    monkeypatch.setattr(
        tournament,
        "guided_gud_basis_audit_config",
        lambda _seed: replay.GUDOutputJacobianReplayConfig(
            prompt_count=24,
            horizon=8,
            rank=1,
            sketch_width=2,
            balanced_envelope_width=2,
            balanced_stream_rows=1,
            fingerprint_lanes=2,
            fingerprint_chunk_elements=64,
            max_callback_rows=64,
            seed=17,
            max_bytes=64 * 1024 * 1024,
        ),
    )
    with pytest.raises(
        tournament.GuidedGUDCandidateTournamentError,
        match="exact old candidate24",
    ):
        tournament.authenticate_guided_gud_basis_source(bundle, source)


def test_sink_keeps_only_lightweight_receipts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sink"
    output.mkdir()
    sink = tournament.GuidedGUDTopThreeSink(
        output,
        evaluation_contract_sha256=_digest("evaluation"),
    )
    callback = sink.callback_for_arm(tournament.ARM_IDS[0], 0)
    event = DurableBehaviorCheckpoint(
        step=0,
        metrics=_metrics(),
        factor_state=_v4_state(),
        pareto_safe_vs_step0=True,
    )
    callback(event)

    stored = sink._events[tournament.ARM_IDS[0]][0]
    assert type(stored) is tournament._DurableV4CheckpointReceipt
    assert not isinstance(stored, DurableBehaviorCheckpoint)
    assert not hasattr(stored, "factor_state")
    assert not hasattr(stored, "raw_bytes")
    retained = next(iter(sink._retained.values()))
    retained.pack_path.write_bytes(retained.pack_path.read_bytes() + b"tamper")
    with pytest.raises(
        tournament.GuidedGUDCandidateTournamentError,
        match="changed before selection",
    ):
        sink._reauthenticate_retained_state(retained)


def test_failure_cleanup_attempts_every_rollback_layer() -> None:
    calls: list[str] = []

    class GUD:
        is_applied = True

        def rollback_guided_only(self) -> None:
            calls.append("gud-guided")
            raise RuntimeError("guided failed")

        def rollback(self) -> None:
            calls.append("gud-fallback")
            raise RuntimeError("fallback failed")

    class Transaction:
        def rollback(self) -> None:
            calls.append("transaction")

    fake = SimpleNamespace(
        compiled=SimpleNamespace(child_model=object()),
        compiled_state_sha256="exact",
        rollback_guided_installation=lambda _transaction: (
            calls.append("qkvo"),
            (_ for _ in ()).throw(RuntimeError("qkvo failed")),
        ),
        state_sha256=lambda _model: "exact",
    )
    with pytest.raises(
        tournament.GuidedGUDCandidateTournamentError,
        match="rollback layers failed",
    ):
        tournament.ProductionGuidedGUDCandidateTournamentRuntime._rollback_guided_gud_arm(
            fake,
            Transaction(),
            GUD(),
        )
    assert calls == ["gud-guided", "gud-fallback", "qkvo", "transaction"]


def test_production_basis_launcher_runs_restore_replay_publish_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_geometry(monkeypatch)
    result = _replay_result()
    state = _v3_state()
    binding = state.guided_source_binding
    assert isinstance(binding, GuidedFactorSourceBinding)
    candidate = SimpleNamespace(
        name="candidate",
        payload={"prompts": []},
        sha256=_digest("candidate24"),
    )
    contract = SimpleNamespace(
        runtime={
            "dtype": "float16",
            "parent_device_map": "cuda:0",
            "target_device_map": "cuda:1",
        },
        source=SimpleNamespace(execution_contract=SimpleNamespace(seed=17)),
        splits={"candidate": candidate},
    )
    source = tournament.AuthenticatedV3WinnerSource(
        tournament_contract=contract,
        tournament_preregistration_path=tmp_path / "prereg.json",
        tournament_preregistration_sha256=_digest("prereg"),
        run_report_path=tmp_path / "source-run.json",
        run_report_sha256=_digest("source-run"),
        winner_selection_path=tmp_path / "winner.json",
        winner_selection_sha256=_digest("winner"),
        arm_id=tournament.EXPECTED_SOURCE_ARM_ID,
        step=tournament.EXPECTED_SOURCE_STEP,
        runtime_factor_state_sha256=state.factor_state_sha256,
        portable_factor_state_sha256=result.report.payload[
            "source_v3_portable_factor_state_sha256"
        ],
        factor_pack_path=tmp_path / "factor.pt",
        factor_pack_file_sha256=_digest("factor"),
        factor_payload=state.as_factor_payload(),
        candidate_frontier={},
    )
    calls: list[str] = []

    class Tranche:
        training_stage = "guided"

        def set_training_stage(self, stage_name: str) -> None:
            calls.append(f"stage:{stage_name}")
            self.training_stage = stage_name

    class Transaction:
        def rollback(self) -> None:
            calls.append("transaction-rollback")

    class Runtime:
        parent = object()
        compiled = SimpleNamespace(child_model=object())
        compiled_state_sha256 = "compiled"
        _guided_tranche = Tranche()

        def prepare(self, received: Any) -> dict[str, Any]:
            assert received is contract
            calls.append("prepare")
            return {"compiled_state_sha256": "compiled"}

        def install_transaction(self, _compiled: Any) -> Transaction:
            calls.append("install-base")
            return Transaction()

        def restore_source_v2(self, _transaction: Any, _source: Any) -> None:
            calls.append("restore-v2")

        def install_guided_transaction(self, _transaction: Any, _source: Any) -> None:
            calls.append("install-guided-qkvo")

        def rollback_guided_installation(self, _transaction: Any) -> None:
            calls.append("guided-qkvo-rollback")

        def state_sha256(self, _model: Any) -> str:
            return "compiled"

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(producer, "_reauthenticate_source", lambda value: value)
    monkeypatch.setattr(
        producer.behavior,
        "capture_target_native_factor_state",
        lambda _transaction: object(),
    )
    monkeypatch.setattr(
        producer.behavior,
        "authenticate_guided_factor_sources",
        lambda *_args: binding,
    )
    monkeypatch.setattr(
        producer,
        "rebind_recovered_v3_factor_payload",
        lambda *_args: {"factor_state_sha256": "runtime-v3"},
    )
    monkeypatch.setattr(
        producer,
        "portable_v3_factor_state_sha256",
        lambda *_args: source.portable_factor_state_sha256,
    )
    monkeypatch.setattr(
        producer.behavior,
        "restore_target_native_factor_payload",
        lambda *_args, **_kwargs: SimpleNamespace(
            factor_state_sha256="runtime-v3",
            schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        ),
    )
    monkeypatch.setattr(
        producer.behavior,
        "load_behavior_domain_suite",
        lambda *_args, **_kwargs: BehaviorDomainSuite(
            split_name="fit",
            prompts=tuple(BehaviorPrompt((index + 1,)) for index in range(24)),
            artifact_sha256=candidate.sha256,
        ),
    )
    monkeypatch.setattr(
        producer.tournament,
        "authenticate_guided_gud_basis_source",
        lambda *_args: {"sealed_prompt_count_replayed": 0},
    )

    def audit(**kwargs: Any) -> replay.GUDOutputJacobianReplayResult:
        calls.append("replay")
        assert kwargs["guided_tranche"].training_stage == "joint"
        assert kwargs["basis_domain"].split_name == "fit"
        return result

    output = tmp_path / "audit-output"
    report = producer.execute_guided_gud_basis_audit(
        source,
        output,
        runtime=Runtime(),
        audit_runner=audit,
    )

    assert report["status"] == "basis-audit-complete-before-training"
    assert report["compiled_state_sha256_after_rollback"] == "compiled"
    assert report["sealed_prompt_count_replayed"] == 0
    assert (output / "guided_gud_basis_bundle.pt").is_file()
    assert (output / "gud_output_jacobian_replay_report.json").is_file()
    assert (output / "guided_gud_layer_mask.json").is_file()
    assert (output / "guided_gud_layer_mask_derivation.json").is_file()
    assert calls == [
        "prepare",
        "install-base",
        "restore-v2",
        "install-guided-qkvo",
        "stage:joint",
        "replay",
        "guided-qkvo-rollback",
        "transaction-rollback",
        "close",
    ]

    original_rollback = producer._rollback_exact
    rollback_attempts = 0

    def cleanup_then_fail(runner: Any, transaction: Any) -> str:
        nonlocal rollback_attempts
        rollback_attempts += 1
        original_rollback(runner, transaction)
        raise producer.GuidedGUDBasisAuditError("post-cleanup receipt failure")

    monkeypatch.setattr(producer, "_rollback_exact", cleanup_then_fail)
    with pytest.raises(
        producer.GuidedGUDBasisAuditError,
        match="post-cleanup receipt failure",
    ):
        producer.execute_guided_gud_basis_audit(
            source,
            tmp_path / "failed-audit-output",
            runtime=Runtime(),
            audit_runner=audit,
        )
    assert rollback_attempts == 1
