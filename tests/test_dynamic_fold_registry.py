from __future__ import annotations

from dataclasses import replace

import pytest

import neural_forge.dynamic_fold_registry as registry_module
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_registry import (
    ClosureLevel,
    DurableExperimentLedger,
    DynamicFoldRegistryError,
    ExperimentLedger,
    ExperimentSignature,
    LedgerRecord,
    RecordStatus,
    default_method_registry,
)


def _signature(**changes) -> ExperimentSignature:
    values = {
        "method_id": "coupled_gqa_medoid_endpoint",
        "method_version": 1,
        "family_id": "coupled_attention",
        "parent_graph_fingerprints": (canonical_sha256({"parent": 1}),),
        "parent_state_fingerprints": (
            ("parent", canonical_sha256({"parent_state": 1})),
        ),
        "child_graph_fingerprint": canonical_sha256({"child": "topology"}),
        "child_state_fingerprint": canonical_sha256({"child": "state-1"}),
        "evidence_fingerprint": canonical_sha256({"evidence": 1}),
        "source_unit_ids": ("parent:layer.000.attention",),
        "target_unit_ids": ("layer.000.attention",),
        "topology_transition": "attention->attention",
        "selection_policy": "automatic_complete_circuit_alignment",
        "fit_objective": "closed_loop_parent_relative_residual_endpoint",
        "promotion_gate": "automatic_behavior_then_final_manual_review",
        "zero_gradient": True,
        "configuration": (("target_heads", 7),),
    }
    values.update(changes)
    return ExperimentSignature(**values)


def test_registry_refuses_settled_closed_lanes() -> None:
    registry = default_method_registry()

    with pytest.raises(DynamicFoldRegistryError, match="closed"):
        registry.require_selectable("one_shot_pca_axis_fold")
    with pytest.raises(DynamicFoldRegistryError, match="training_only"):
        registry.require_selectable("knowledge_distillation")
    assert registry.require_selectable("lfm2_closed_loop_layer").selectable


@pytest.mark.parametrize(
    "field",
    ("source_kinds", "target_kinds", "evidence_channels", "coupled_roles"),
)
def test_method_specs_require_nonempty_executable_contracts(field: str) -> None:
    method = default_method_registry().get("copy_exact")

    with pytest.raises(DynamicFoldRegistryError, match=field):
        replace(method, **{field: ()})


def test_novelty_levels_ignore_cosmetic_state_but_bind_material_changes() -> None:
    base = _signature()
    new_state = replace(
        base,
        child_state_fingerprint=canonical_sha256({"child": "state-2"}),
    )
    new_config = replace(base, configuration=(("target_heads", 6),))

    assert base.family_sha256 == new_state.family_sha256
    assert base.lane_sha256 == new_state.lane_sha256
    assert base.configuration_sha256 == new_state.configuration_sha256
    assert base.candidate_sha256 != new_state.candidate_sha256
    assert base.lane_sha256 == new_config.lane_sha256
    assert base.configuration_sha256 != new_config.configuration_sha256


def test_exact_rejection_blocks_only_same_state_and_evidence() -> None:
    signature = _signature()
    record = LedgerRecord(
        signature=signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.REJECTED,
        reason_class="behavior_regression",
        reason="Candidate failed the free-running behavior gate.",
    )
    ledger = ExperimentLedger((record,))

    assert ledger.decision(signature).allowed is False
    changed = replace(
        signature,
        child_state_fingerprint=canonical_sha256({"child": "accepted-edit"}),
    )
    assert ledger.decision(changed).allowed is True


def test_append_only_hash_chain_is_enforced(tmp_path) -> None:
    first = LedgerRecord(
        signature=_signature(),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.REJECTED,
        reason_class="gate",
        reason="rejected",
    )
    ledger = ExperimentLedger((first,))
    second = LedgerRecord(
        signature=replace(
            _signature(),
            child_state_fingerprint=canonical_sha256({"child": 2}),
        ),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
        previous_record_sha256=first.record_sha256,
    )
    ledger.append(second)
    path = tmp_path / "ledger.jsonl"
    ledger.write_jsonl(path)

    assert len(path.read_text(encoding="ascii").splitlines()) == 2
    with pytest.raises(DynamicFoldRegistryError, match="hash chain"):
        ledger.append(replace(second, previous_record_sha256=None))


def test_durable_ledger_round_trips_and_appends_without_rewrite(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = ExperimentLedger()
    first = LedgerRecord(
        signature=_signature(),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.REJECTED,
        reason_class="gate",
        reason="rejected",
    )
    ledger.append_jsonl(path, first)
    prefix = path.read_bytes()
    second = LedgerRecord(
        signature=replace(
            _signature(),
            child_state_fingerprint=canonical_sha256({"child": 2}),
        ),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
        previous_record_sha256=first.record_sha256,
    )
    ledger.append_jsonl(path, second)

    assert path.read_bytes().startswith(prefix)
    loaded = ExperimentLedger.read_jsonl(path)
    assert loaded.fingerprint == ledger.fingerprint
    assert loaded.records == (first, second)


def test_durable_ledger_rejects_tampering_and_stale_writer(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = LedgerRecord(
        signature=_signature(),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.REJECTED,
        reason_class="gate",
        reason="rejected",
    )
    ledger = ExperimentLedger()
    ledger.append_jsonl(path, first)
    stale = ExperimentLedger()
    second = LedgerRecord(
        signature=replace(
            _signature(),
            child_state_fingerprint=canonical_sha256({"child": 3}),
        ),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
        previous_record_sha256=first.record_sha256,
    )
    with pytest.raises(DynamicFoldRegistryError, match="stale"):
        stale.append_jsonl(path, second)

    tampered = path.read_text(encoding="ascii").replace(
        '"reason":"rejected"',
        '"reason":"changed"',
    )
    path.write_text(tampered, encoding="ascii", newline="\n")
    with pytest.raises(DynamicFoldRegistryError, match="fingerprint"):
        ExperimentLedger.read_jsonl(path)


def test_newer_supersession_reopens_an_older_exact_acceptance() -> None:
    signature = _signature()
    accepted = LedgerRecord(
        signature=signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="transaction_gate",
        reason="provisionally accepted",
    )
    superseded = LedgerRecord(
        signature=signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.SUPERSEDED,
        reason_class="global_rollback",
        reason="A later final gate rolled the run back.",
        previous_record_sha256=accepted.record_sha256,
    )
    ledger = ExperimentLedger((accepted, superseded))

    assert ledger.decision(signature).allowed is True


def test_compensating_supersession_reopens_pruned_publication() -> None:
    signature = _signature()
    accepted = LedgerRecord(
        signature=signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="transaction_gate",
        reason="provisionally accepted",
    )
    ledger = ExperimentLedger((accepted,))

    closures = ledger.supersede_records(
        (accepted.record_sha256,),
        reason_class="artifact_rollback",
        reason="The provisional artifact was pruned.",
    )

    assert len(closures) == 1
    assert closures[0].status == RecordStatus.SUPERSEDED.value
    assert closures[0].previous_record_sha256 == accepted.record_sha256
    assert ledger.decision(signature).allowed is True


def test_durable_ledger_persists_every_append_and_reopens(tmp_path) -> None:
    path = tmp_path / "production-ledger.jsonl"
    ledger = DurableExperimentLedger(path)
    accepted = LedgerRecord(
        signature=_signature(),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
    )

    ledger.append(accepted)
    reopened = DurableExperimentLedger(path)

    assert reopened.records == (accepted,)
    assert reopened.fingerprint == ledger.fingerprint
    assert path.read_bytes().endswith(b"\n")


def test_durable_ledger_can_export_and_verify_a_canonical_snapshot(tmp_path) -> None:
    path = tmp_path / "production-ledger.jsonl"
    export = tmp_path / "exported-ledger.jsonl"
    ledger = DurableExperimentLedger(path)
    accepted = LedgerRecord(
        signature=_signature(),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
    )
    ledger.append(accepted)

    ledger.write_jsonl(export)

    loaded = ExperimentLedger.read_jsonl(export)
    assert loaded.records == ledger.records
    assert loaded.fingerprint == ledger.fingerprint


def test_append_batch_rejects_an_invalid_extension_without_partial_mutation() -> None:
    ledger = ExperimentLedger()
    first = LedgerRecord(
        signature=_signature(),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
    )
    invalid_second = LedgerRecord(
        signature=replace(
            _signature(),
            child_state_fingerprint=canonical_sha256({"child": "second"}),
        ),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
        previous_record_sha256=None,
    )

    with pytest.raises(DynamicFoldRegistryError, match="hash chain"):
        ledger.append_batch((first, invalid_second))

    assert ledger.records == ()


def test_durable_batch_failure_preserves_memory_and_disk_atomically(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "atomic-ledger.jsonl"
    ledger = DurableExperimentLedger(path)
    before = path.read_bytes()
    first = LedgerRecord(
        signature=_signature(),
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
    )

    def fail_replace(_path, _records):
        raise OSError("simulated durable commit failure")

    monkeypatch.setattr(registry_module, "_write_records_atomic", fail_replace)
    with pytest.raises(OSError, match="simulated durable commit failure"):
        ledger.append_batch((first,))

    assert ledger.records == ()
    assert path.read_bytes() == before


def test_two_phase_supersession_blocks_before_artifact_removal() -> None:
    signature = _signature()
    accepted = LedgerRecord(
        signature=signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.ACCEPTED,
        reason_class="gate",
        reason="accepted",
    )
    ledger = ExperimentLedger((accepted,))

    pending = ledger.prepare_supersession(
        (accepted.record_sha256,),
        reason_class="artifact_rollback_pending",
        reason="Artifact removal is pending.",
    )

    assert pending[0].status == RecordStatus.SUPERSESSION_PENDING.value
    assert ledger.decision(signature).allowed is False
    assert "pending" in ledger.decision(signature).reason

    ledger.supersede_records(
        (pending[0].record_sha256,),
        reason_class="artifact_rollback",
        reason="Artifact removal completed.",
    )
    assert ledger.decision(signature).allowed is True
