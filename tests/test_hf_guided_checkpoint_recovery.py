from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

import neural_forge.hf_guided_checkpoint_recovery as recovery
import neural_forge.hf_target_native_behavior_distillation as behavior
from neural_forge.hf_guided_checkpoint_recovery import (
    GuidedCheckpointRecoveryAuthenticationError,
    GuidedCheckpointRecoveryOrderError,
    GuidedCheckpointRecoveryPublicationError,
    GuidedCheckpointRecoverySink,
    HistoricalCheckpointExpectation,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BehaviorCheckpoint,
    CausalBehaviorMetrics,
    DurableBehaviorCheckpoint,
    DurableFactorStateSnapshot,
    DurableFactorTensorSnapshot,
    GuidedFactorSourceBinding,
)


SOURCE_RUN_SHA = "1" * 64
RECOVERY_PREREG_SHA = "2" * 64
RECOVERY_REPORT_SHA = "3" * 64

QKVO_GUD_NAMES = {
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
}
LEXICAL_NAMES = {
    f"lexical.{role}_{side}"
    for role in ("embedding", "lm_head")
    for side in ("left", "right")
}
GUIDED_NAMES = {
    f"guided_qkvo.{role}_{side}"
    for role in ("q", "k", "v", "o")
    for side in ("left", "right")
}
V3_NAMES = tuple(sorted(QKVO_GUD_NAMES | LEXICAL_NAMES | GUIDED_NAMES))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_binding() -> GuidedFactorSourceBinding:
    return GuidedFactorSourceBinding(
        source_v2_factor_state_sha256=_digest("source-v2-state"),
        source_v2_tensor_sha256s=tuple(
            (name, _digest(f"source:{name}"))
            for name in sorted(QKVO_GUD_NAMES | LEXICAL_NAMES)
        ),
        audit_report_sha256=_digest("audit-report"),
        basis_envelope_file_sha256=_digest("basis-envelope"),
        basis_payload_sha256=_digest("basis-payload"),
        basis_manifest_sha256=_digest("basis-manifest"),
        source_basis_tensor_sha256s=tuple(
            (role, _digest(f"basis:{role}")) for role in ("q", "k", "v", "o")
        ),
        inherited_installation_sha256=_digest("inherited-installation"),
        inherited_factor_state_sha256_at_install=_digest("inherited-state"),
        adapter_site_indices_sha256=_digest("adapter-sites"),
        audit_site_indices_sha256=_digest("audit-sites"),
    )


def _runtime_binding(
    source: GuidedFactorSourceBinding,
    *,
    installation_label: str = "runtime-installation",
    audit_label: str | None = None,
) -> GuidedFactorSourceBinding:
    payload = source.as_dict()
    payload["inherited_installation_sha256"] = _digest(installation_label)
    if audit_label is not None:
        payload["audit_report_sha256"] = _digest(audit_label)
    return GuidedFactorSourceBinding(**payload)


def _factor_state(seed: int) -> DurableFactorStateSnapshot:
    binding = _source_binding()
    snapshots: list[DurableFactorTensorSnapshot] = []
    rows: list[dict[str, object]] = []
    for index, name in enumerate(V3_NAMES):
        tensor = torch.tensor(
            [float(seed), float(index), float(seed + index + 1)],
            dtype=torch.float32,
        )
        # Build the fixture in the trainer's real legacy factor-pack hash
        # domain.  The recovery sink must interoperate with this digest rather
        # than validating tensors against a sink-local convention.
        digest = behavior._tensor_sha256(tensor)
        snapshots.append(
            DurableFactorTensorSnapshot(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                tensor_sha256=digest,
                raw_bytes=tensor.view(torch.uint8).numpy().tobytes(),
            )
        )
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
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
        tensors=tuple(snapshots),
        factor_state_sha256=factor_sha,
        lexical_rank=4,
        guided_source_binding=binding,
    )


def _state_with_binding(
    source: DurableFactorStateSnapshot,
    binding: GuidedFactorSourceBinding,
) -> DurableFactorStateSnapshot:
    rows = [
        {
            "name": value.name,
            "shape": list(value.shape),
            "dtype": value.dtype,
            "tensor_sha256": value.tensor_sha256,
        }
        for value in source.tensors
    ]
    factor_sha = behavior._factor_state_sha256(
        rows,
        schema=BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
        lexical_rank=source.lexical_rank,
        guided_source_binding=binding,
    )
    return DurableFactorStateSnapshot(
        schema=source.schema,
        tensors=source.tensors,
        factor_state_sha256=factor_sha,
        lexical_rank=source.lexical_rank,
        guided_source_binding=binding,
    )


def _metrics(step: int, *, action_offset: int = 0) -> CausalBehaviorMetrics:
    action_agreements = 10 + step // 8 + action_offset
    token_matches = 12 + step // 8 + action_offset
    lcp_tokens = 4 + step // 16 + action_offset
    exact_rollouts = min(2, step // 40)
    return CausalBehaviorMetrics(
        split="calibration-selection",
        action_count=192,
        action_agreement_count=action_agreements,
        action_accuracy=action_agreements / 192,
        parent_action_mean_logprob=-4.0 + step / 100,
        parent_action_topk_fraction=0.2 + step / 1000,
        parent_action_mean_rank=4000.0 - step,
        parent_action_worst_rank=12000 - step,
        rollout_count=24,
        exact_rollout_count=exact_rollouts,
        exact_rollout_fraction=exact_rollouts / 24,
        rollout_token_count=192,
        token_match_count=token_matches,
        token_match_fraction=token_matches / 192,
        lcp_token_count=lcp_tokens,
        lcp_fraction=lcp_tokens / 192,
        lcp_at_least_4_count=min(24, lcp_tokens // 4),
        repeat_fraction_excess=max(0.0, 0.4 - step / 400),
        adjacent_repeat_fraction_excess=max(0.0, 0.55 - step / 400),
        longest_run_excess=max(0, 4 - step // 20),
        report_sha256=_digest(f"metrics-report:{step}:{action_offset}"),
    )


def _event(
    step: int,
    factor_state: DurableFactorStateSnapshot,
    *,
    metrics: CausalBehaviorMetrics | None = None,
) -> DurableBehaviorCheckpoint:
    return DurableBehaviorCheckpoint(
        step=step,
        metrics=_metrics(step) if metrics is None else metrics,
        factor_state=factor_state,
        pareto_safe_vs_step0=step in {0, 40, 64},
    )


def _expectation(
    event: DurableBehaviorCheckpoint,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> HistoricalCheckpointExpectation:
    metrics = event.metrics.as_dict()
    return HistoricalCheckpointExpectation(
        step=event.step,
        metrics=metrics,
        metrics_sha256=recovery.canonical_sha256(metrics),
        factor_state_sha256=event.factor_state.factor_state_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )


def _terminal_row(
    event: DurableBehaviorCheckpoint,
    *,
    pareto_frontier: bool = False,
) -> dict[str, object]:
    return BehaviorCheckpoint(
        step=event.step,
        metrics=event.metrics,
        factor_state_sha256=event.factor_state.factor_state_sha256,
        pareto_safe_vs_step0=event.pareto_safe_vs_step0,
        pareto_frontier=pareto_frontier,
    ).as_dict()


def _fixture_events(
    *,
    deduplicate_40_56: bool = False,
) -> tuple[DurableBehaviorCheckpoint, ...]:
    step0 = _factor_state(0)
    step40 = _factor_state(40)
    step56 = step40 if deduplicate_40_56 else _factor_state(56)
    step64 = _factor_state(64)
    return (
        _event(0, step0),
        _event(40, step40),
        _event(56, step56),
        _event(64, step64),
    )


def _sink(
    output: Path,
    events: tuple[DurableBehaviorCheckpoint, ...],
    *,
    retained_steps: tuple[int, ...] = (40, 56, 64),
    prebind_checkpoint_sha: bool = False,
) -> GuidedCheckpointRecoverySink:
    return GuidedCheckpointRecoverySink(
        output,
        expectations=tuple(
            _expectation(
                event,
                expected_checkpoint_sha256=(
                    event.checkpoint_sha256 if prebind_checkpoint_sha else None
                ),
            )
            for event in events
        ),
        retained_steps=retained_steps,
        recovery_preregistration_sha256=RECOVERY_PREREG_SHA,
        recovery_report_sha256=RECOVERY_REPORT_SHA,
        source_run_report_sha256=SOURCE_RUN_SHA,
        historical_guided_source_binding=_source_binding(),
    )


def test_recovery_publishes_three_full_v3_states_atomically(tmp_path: Path) -> None:
    events = _fixture_events()
    output = tmp_path / "recovered"
    sink = _sink(output, events)
    staging = sink.staging_root
    assert staging is not None and staging.is_dir()

    for event in events:
        sink(event)
        assert not output.exists()

    publication = sink.finalize(
        trainer_evidence_sha256=_digest("trainer-evidence"),
        terminal_checkpoint_rows=tuple(
            _terminal_row(event, pareto_frontier=event.step == 40) for event in events
        ),
    )

    assert sink.state == "finalized"
    assert sink.staging_root is None
    assert not staging.exists()
    assert output.is_dir()
    assert publication.manifest_path == output / "research_manifest.json"
    assert publication.retained_steps == (40, 56, 64)
    manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    declared = manifest.pop("manifest_sha256")
    assert recovery.canonical_sha256(manifest) == declared
    manifest["manifest_sha256"] = declared
    assert manifest["historical_durable_checkpoint_sha_required"] is False
    assert manifest["replay_durable_checkpoint_sha_self_authenticated"] is True
    assert manifest["contains_parent_state"] is False
    assert manifest["contains_native_child_state"] is False
    assert manifest["contains_optimizer_state"] is False
    assert [row["step"] for row in manifest["observed_events"]] == [0, 40, 56, 64]
    assert all(
        row["historical_checkpoint_sha256"] is None
        and row["durable_checkpoint_sha256_historically_prebound"] is False
        for row in manifest["observed_events"]
    )

    for event in events[1:]:
        candidate = output / "retained" / event.factor_state.factor_state_sha256
        assert set(path.name for path in candidate.iterdir()) == {
            "factor_pack.pt",
            "candidate_record.json",
        }
        factor_pack = torch.load(
            candidate / "factor_pack.pt", map_location="cpu", weights_only=True
        )
        assert factor_pack["schema"] == BEHAVIOR_FACTOR_PACK_V3_SCHEMA
        assert factor_pack["factor_count"] == 26
        assert set(factor_pack["factor_tensors"]) == set(V3_NAMES)
        assert factor_pack["guided_source_binding_sha256"] == (
            event.factor_state.guided_source_binding.binding_sha256
        )
        assert factor_pack["parent_state_included"] is False
        assert factor_pack["native_child_state_included"] is False
        assert factor_pack["optimizer_state_included"] is False
        record = json.loads(
            (candidate / "candidate_record.json").read_text(encoding="utf-8")
        )
        assert record["recovery_preregistration_sha256"] == RECOVERY_PREREG_SHA
        assert record["recovery_report_sha256"] == RECOVERY_REPORT_SHA
        assert record["source_run_report_sha256"] == SOURCE_RUN_SHA
        assert record["retained_steps"] == [event.step]
        assert record["events"][0]["metrics"] == event.metrics.as_dict()

    sink.cleanup()
    assert output.is_dir()


def test_recovery_deduplicates_equal_factor_states_by_sha(tmp_path: Path) -> None:
    events = _fixture_events(deduplicate_40_56=True)
    output = tmp_path / "deduplicated"
    sink = _sink(output, events)
    for event in events:
        sink(event)
    publication = sink.finalize(
        trainer_evidence_sha256=_digest("trainer-deduplicated"),
        terminal_checkpoint_rows=tuple(_terminal_row(event) for event in events),
    )

    retained = list((output / "retained").iterdir())
    assert len(retained) == 2
    shared = output / "retained" / events[1].factor_state.factor_state_sha256
    record = json.loads((shared / "candidate_record.json").read_text(encoding="utf-8"))
    assert record["retained_steps"] == [40, 56]
    assert [row["step"] for row in record["events"]] == [40, 56]
    assert publication.retained_factor_state_sha256s == tuple(
        sorted({event.factor_state.factor_state_sha256 for event in events[1:]})
    )


@pytest.mark.parametrize("mode", ("duplicate", "reordered"))
def test_recovery_rejects_duplicate_or_reordered_callbacks_and_cleans_owned_root(
    tmp_path: Path,
    mode: str,
) -> None:
    events = _fixture_events()
    output = tmp_path / mode
    sentinel = tmp_path / f"{mode}.sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    sink = _sink(output, events)
    staging = sink.staging_root
    assert staging is not None

    if mode == "duplicate":
        sink(events[0])
        bad_event = events[0]
        match = "duplicated"
    else:
        bad_event = events[1]
        match = "expected step 0"
    with pytest.raises(GuidedCheckpointRecoveryOrderError, match=match):
        sink(bad_event)

    assert sink.state == "failed"
    assert sink.staging_root is None
    assert not staging.exists()
    assert not output.exists()
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_recovery_rejects_missing_callbacks_at_finalize(tmp_path: Path) -> None:
    events = _fixture_events()
    output = tmp_path / "missing"
    sink = _sink(output, events)
    staging = sink.staging_root
    sink(events[0])
    with pytest.raises(GuidedCheckpointRecoveryOrderError, match="missing callbacks"):
        sink.finalize(
            trainer_evidence_sha256=_digest("trainer-missing"),
            terminal_checkpoint_rows=(_terminal_row(events[0]),),
        )
    assert staging is not None and not staging.exists()
    assert not output.exists()


def test_recovery_base_exception_cleans_only_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _fixture_events()
    output = tmp_path / "interrupted"
    sentinel = tmp_path / "interrupted.sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    sink = _sink(output, events)
    staging = sink.staging_root

    def interrupted(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt("sink interrupted")

    monkeypatch.setattr(sink, "_authenticate_event", interrupted)
    with pytest.raises(KeyboardInterrupt, match="sink interrupted"):
        sink(events[0])
    assert sink.state == "failed"
    assert sink.staging_root is None
    assert staging is not None and not staging.exists()
    assert not output.exists()
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_recovery_rejects_tampered_checkpoint_metrics_and_factor_bytes(
    tmp_path: Path,
) -> None:
    events = _fixture_events()

    checkpoint_sink = _sink(tmp_path / "checkpoint", events)
    bad_checkpoint = copy.deepcopy(events[0])
    object.__setattr__(bad_checkpoint, "checkpoint_sha256", "f" * 64)
    with pytest.raises(
        GuidedCheckpointRecoveryAuthenticationError,
        match="checkpoint SHA failed",
    ):
        checkpoint_sink(bad_checkpoint)

    metrics_sink = _sink(tmp_path / "metrics", events)
    bad_metrics = _event(
        0,
        events[0].factor_state,
        metrics=_metrics(0, action_offset=1),
    )
    with pytest.raises(
        GuidedCheckpointRecoveryAuthenticationError,
        match="metrics differ",
    ):
        metrics_sink(bad_metrics)

    factor_sink = _sink(tmp_path / "factor", events)
    bad_factor = copy.deepcopy(events[0])
    tensor = bad_factor.factor_state.tensors[0]
    raw = bytearray(tensor.raw_bytes)
    raw[0] ^= 1
    object.__setattr__(tensor, "raw_bytes", bytes(raw))
    with pytest.raises(
        GuidedCheckpointRecoveryAuthenticationError,
        match="factor bytes failed",
    ):
        factor_sink(bad_factor)


def test_recovery_rejects_terminal_rows_that_do_not_match_callbacks(
    tmp_path: Path,
) -> None:
    events = _fixture_events()
    output = tmp_path / "terminal-tamper"
    sink = _sink(output, events)
    for event in events:
        sink(event)
    rows = [_terminal_row(event) for event in events]
    rows[2]["factor_state_sha256"] = "f" * 64
    with pytest.raises(
        GuidedCheckpointRecoveryAuthenticationError,
        match="differs from its callback",
    ):
        sink.finalize(
            trainer_evidence_sha256=_digest("trainer-terminal"),
            terminal_checkpoint_rows=rows,
        )
    assert not output.exists()


def test_optional_historical_durable_checkpoint_sha_is_verified(
    tmp_path: Path,
) -> None:
    events = _fixture_events()
    sink = _sink(
        tmp_path / "prebound",
        events,
        retained_steps=(40,),
        prebind_checkpoint_sha=True,
    )
    for event in events:
        sink(event)
    publication = sink.finalize(
        trainer_evidence_sha256=_digest("trainer-prebound"),
        terminal_checkpoint_rows=tuple(_terminal_row(event) for event in events),
    )
    manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    assert all(
        row["durable_checkpoint_sha256_historically_prebound"] is True
        for row in manifest["observed_events"]
    )


def test_constructor_rejects_more_than_three_or_noncanonical_retained_steps(
    tmp_path: Path,
) -> None:
    events = _fixture_events()
    expectations = tuple(_expectation(event) for event in events)
    kwargs = {
        "expectations": expectations,
        "recovery_preregistration_sha256": RECOVERY_PREREG_SHA,
        "recovery_report_sha256": RECOVERY_REPORT_SHA,
        "source_run_report_sha256": SOURCE_RUN_SHA,
        "historical_guided_source_binding": _source_binding(),
    }
    with pytest.raises(GuidedCheckpointRecoveryOrderError, match="one to three"):
        GuidedCheckpointRecoverySink(
            tmp_path / "four",
            retained_steps=(0, 40, 56, 64),
            **kwargs,
        )
    with pytest.raises(GuidedCheckpointRecoveryOrderError, match="one to three"):
        GuidedCheckpointRecoverySink(
            tmp_path / "reordered-retained",
            retained_steps=(64, 40),
            **kwargs,
        )
    with pytest.raises(GuidedCheckpointRecoveryOrderError, match="one to three"):
        GuidedCheckpointRecoverySink(
            tmp_path / "unknown-retained",
            retained_steps=(48,),
            **kwargs,
        )


def test_expectation_rejects_incomplete_or_false_metrics_hash() -> None:
    event = _fixture_events()[0]
    metrics = event.metrics.as_dict()
    incomplete = dict(metrics)
    incomplete.pop("lcp_fraction")
    with pytest.raises(
        GuidedCheckpointRecoveryAuthenticationError,
        match="complete canonical",
    ):
        HistoricalCheckpointExpectation(
            step=0,
            metrics=incomplete,
            metrics_sha256=recovery.canonical_sha256(incomplete),
            factor_state_sha256=event.factor_state.factor_state_sha256,
        )
    with pytest.raises(
        GuidedCheckpointRecoveryAuthenticationError,
        match="metrics SHA",
    ):
        HistoricalCheckpointExpectation(
            step=0,
            metrics=metrics,
            metrics_sha256="f" * 64,
            factor_state_sha256=event.factor_state.factor_state_sha256,
        )


def test_atomic_publication_failure_cleans_only_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _fixture_events()
    output = tmp_path / "rename-failure"
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    sentinel = sibling / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    sink = _sink(output, events, retained_steps=(40,))
    for event in events:
        sink(event)
    staging = sink.staging_root
    original_rename = os.rename

    def fail_final_rename(source: object, destination: object) -> None:
        if Path(destination) == output:
            raise OSError("simulated final rename failure")
        original_rename(source, destination)

    monkeypatch.setattr(recovery.os, "rename", fail_final_rename)
    with pytest.raises(OSError, match="simulated final rename failure"):
        sink.finalize(
            trainer_evidence_sha256=_digest("trainer-rename-failure"),
            terminal_checkpoint_rows=tuple(_terminal_row(event) for event in events),
        )
    assert staging is not None and not staging.exists()
    assert not output.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_cleanup_is_idempotent_and_never_removes_published_output(
    tmp_path: Path,
) -> None:
    events = _fixture_events()
    sink = _sink(tmp_path / "cleaned", events, retained_steps=(40,))
    staging = sink.staging_root
    sink.cleanup()
    sink.cleanup()
    assert sink.state == "cleaned"
    assert staging is not None and not staging.exists()

    existing = tmp_path / "already-exists"
    existing.mkdir()
    with pytest.raises(GuidedCheckpointRecoveryPublicationError, match="absent"):
        _sink(existing, events, retained_steps=(40,))


def test_cross_process_recovery_reconstructs_legacy_identity_and_rebinds_pack(
    tmp_path: Path,
) -> None:
    historical_events = _fixture_events()
    historical_binding = _source_binding()
    live_binding = _runtime_binding(historical_binding)
    runtime_events = tuple(
        _event(
            event.step,
            _state_with_binding(event.factor_state, live_binding),
            metrics=event.metrics,
        )
        for event in historical_events
    )
    output = tmp_path / "cross-process"
    sink = GuidedCheckpointRecoverySink(
        output,
        expectations=tuple(_expectation(event) for event in historical_events),
        retained_steps=(40, 56, 64),
        recovery_preregistration_sha256=RECOVERY_PREREG_SHA,
        recovery_report_sha256=RECOVERY_REPORT_SHA,
        source_run_report_sha256=SOURCE_RUN_SHA,
        historical_guided_source_binding=historical_binding,
    )
    for event in runtime_events:
        sink(event)
    publication = sink.finalize(
        trainer_evidence_sha256=_digest("historical-evidence"),
        runtime_trainer_evidence_sha256=_digest("runtime-evidence"),
        terminal_checkpoint_rows=tuple(
            _terminal_row(event) for event in runtime_events
        ),
    )

    manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    assert manifest["trainer_evidence_runtime_identity_normalized"] is True
    assert all(
        row["runtime_factor_state_sha256"]
        != row["legacy_rebound_factor_state_sha256"]
        == row["factor_state_sha256"]
        for row in manifest["observed_events"]
    )
    historical = historical_events[1]
    retained = (
        output
        / "retained"
        / historical.factor_state.factor_state_sha256
        / "factor_pack.pt"
    )
    pack = torch.load(retained, map_location="cpu", weights_only=True)
    assert pack["guided_source_binding_sha256"] == historical_binding.binding_sha256
    rebound = recovery.rebind_recovered_v3_factor_payload(pack, live_binding)
    assert rebound["factor_state_sha256"] == (
        runtime_events[1].factor_state.factor_state_sha256
    )
    assert recovery.portable_v3_factor_state_sha256(pack) == (
        recovery.portable_v3_factor_state_sha256(rebound)
    )


def test_cross_process_recovery_rejects_source_drift_and_retains_diagnostic(
    tmp_path: Path,
) -> None:
    historical = _fixture_events()
    drifted_binding = _runtime_binding(
        _source_binding(),
        audit_label="different-audit",
    )
    runtime = _event(
        historical[0].step,
        _state_with_binding(historical[0].factor_state, drifted_binding),
        metrics=historical[0].metrics,
    )
    sink = GuidedCheckpointRecoverySink(
        tmp_path / "source-drift",
        expectations=tuple(_expectation(event) for event in historical),
        retained_steps=(40,),
        recovery_preregistration_sha256=RECOVERY_PREREG_SHA,
        recovery_report_sha256=RECOVERY_REPORT_SHA,
        source_run_report_sha256=SOURCE_RUN_SHA,
        historical_guided_source_binding=_source_binding(),
    )
    with pytest.raises(
        GuidedCheckpointRecoveryAuthenticationError,
        match="historical legacy",
    ):
        sink(runtime)
    diagnostic = sink.failure_diagnostic
    assert diagnostic is not None
    assert diagnostic["portable_source_binding_matches"] is False
    assert diagnostic["observed_runtime_factor_state_sha256"] == (
        runtime.factor_state.factor_state_sha256
    )
    assert diagnostic["observed_legacy_rebound_factor_state_sha256"] == (
        historical[0].factor_state.factor_state_sha256
    )
    declared = diagnostic["diagnostic_sha256"]
    assert (
        recovery.canonical_sha256(
            {
                name: value
                for name, value in diagnostic.items()
                if name != "diagnostic_sha256"
            }
        )
        == declared
    )
    assert sink.state == "failed"
    assert sink.staging_root is None


def test_portable_rebind_rejects_live_source_content_drift() -> None:
    state = _factor_state(40)
    pack = state.as_factor_payload()
    drifted = _runtime_binding(
        _source_binding(),
        installation_label="another-process",
        audit_label="tampered-audit",
    )
    with pytest.raises(
        GuidedCheckpointRecoveryAuthenticationError,
        match="source content differs",
    ):
        recovery.rebind_recovered_v3_factor_payload(pack, drifted)
