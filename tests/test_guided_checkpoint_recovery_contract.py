from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import neural_forge.hf_guided_checkpoint_recovery as sink_module
import neural_forge.hf_target_native_behavior_distillation as behavior

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import prepare_qwen_exact_target_native_guided_checkpoint_recovery as prepare  # noqa: E402
import run_qwen_exact_target_native_guided_checkpoint_recovery as recovery  # noqa: E402
import run_qwen_exact_target_native_stage as stage  # noqa: E402
from neural_forge.hf_target_native_behavior_distillation import (  # noqa: E402
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    BehaviorCheckpoint,
    CausalBehaviorMetrics,
    DurableBehaviorCheckpoint,
    DurableFactorStateSnapshot,
    DurableFactorTensorSnapshot,
    GuidedFactorSourceBinding,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


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


def _binding() -> GuidedFactorSourceBinding:
    return GuidedFactorSourceBinding(
        source_v2_factor_state_sha256=_digest("source-v2"),
        source_v2_tensor_sha256s=tuple(
            (name, _digest(f"source:{name}"))
            for name in sorted(QKVO_GUD_NAMES | LEXICAL_NAMES)
        ),
        audit_report_sha256=_digest("audit"),
        basis_envelope_file_sha256=_digest("envelope"),
        basis_payload_sha256=_digest("basis-payload"),
        basis_manifest_sha256=_digest("basis-manifest"),
        source_basis_tensor_sha256s=tuple(
            (role, _digest(f"basis:{role}")) for role in ("q", "k", "v", "o")
        ),
        inherited_installation_sha256=_digest("installation"),
        inherited_factor_state_sha256_at_install=_digest("inherited-state"),
        adapter_site_indices_sha256=_digest("adapter-sites"),
        audit_site_indices_sha256=_digest("audit-sites"),
    )


def _state(seed: int) -> DurableFactorStateSnapshot:
    binding = _binding()
    snapshots = []
    rows = []
    for index, name in enumerate(V3_NAMES):
        tensor = torch.tensor([seed, index, seed + index + 1], dtype=torch.float32)
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


def _metrics(step: int) -> CausalBehaviorMetrics:
    action = 20 + step // 8
    token = 22 + step // 8
    lcp = 10 + step // 16
    return CausalBehaviorMetrics(
        split="calibration-selection",
        action_count=192,
        action_agreement_count=action,
        action_accuracy=action / 192,
        parent_action_mean_logprob=-6.0,
        parent_action_topk_fraction=0.25,
        parent_action_mean_rank=4000.0 - step,
        parent_action_worst_rank=30000,
        rollout_count=24,
        exact_rollout_count=0,
        exact_rollout_fraction=0.0,
        rollout_token_count=192,
        token_match_count=token,
        token_match_fraction=token / 192,
        lcp_token_count=lcp,
        lcp_fraction=lcp / 192,
        lcp_at_least_4_count=0,
        repeat_fraction_excess=0.4,
        adjacent_repeat_fraction_excess=0.5,
        longest_run_excess=4,
        report_sha256=_digest(f"metrics:{step}"),
    )


def _events() -> tuple[DurableBehaviorCheckpoint, ...]:
    return tuple(
        DurableBehaviorCheckpoint(
            step=step,
            metrics=_metrics(step),
            factor_state=_state(step),
            pareto_safe_vs_step0=step == 0,
        )
        for step in recovery.EXPECTED_CHECKPOINT_STEPS
    )


def _terminal_rows(events) -> list[dict[str, object]]:
    return [
        BehaviorCheckpoint(
            step=event.step,
            metrics=event.metrics,
            factor_state_sha256=event.factor_state.factor_state_sha256,
            pareto_safe_vs_step0=event.pareto_safe_vs_step0,
            pareto_frontier=event.step == 0,
        ).as_dict()
        for event in events
    ]


def _training() -> dict[str, object]:
    return behavior.BehaviorDistillationSchedule(
        steps=64,
        learning_rate=1e-4,
        checkpoint_steps=recovery.EXPECTED_CHECKPOINT_STEPS,
        fit_horizon=8,
        evaluation_horizon=8,
        fit_prompts_per_step=12,
        parent_only_steps=8,
        warmup_fit_horizon=4,
        guided_continuation_enabled=True,
        guided_lexical_head_unlock_step=41,
        learning_rate_decay_step_1=24,
        learning_rate_decay_scale_1=2 / 3,
        learning_rate_decay_step_2=40,
        learning_rate_decay_scale_2=0.5,
        learning_rate_decay_step_3=56,
        learning_rate_decay_scale_3=0.5,
        max_train_prefixes=12_288,
        max_evaluation_prefixes=20_000,
        selection_policy="rollout-first-acquisition",
    ).as_dict()


def _split(name: str, marker: int) -> stage.SplitBinding:
    count = {"fit": 144, "candidate": 24, "sealed": 32}[name]
    rows = tuple(f"{name}-{index}" for index in range(count))
    prompts = tuple(_digest(f"{marker}:{value}") for value in rows)
    payload = {"schema": "fixture", "name": name, "marker": marker}
    return stage.SplitBinding(
        name=name,
        path=Path(f"/{name}.json"),
        sha256=_digest(f"split:{name}:{marker}"),
        expected_count=count,
        payload=payload,
        row_ids=rows,
        prompt_sha256s=prompts,
    )


def _historical_evidence(events) -> dict[str, object]:
    training = _training()
    return {
        "schedule_sha256": sink_module.canonical_sha256(training),
        "domain_binding_sha256": _digest("domain"),
        "selected_step": 0,
        "checkpoints": _terminal_rows(events),
        "training_audit": {"optimizer_steps": 64, "optimizer_class": "AdamW"},
        "sealed_metrics": None,
        "sealed_evaluation_unlocked": False,
        "candidate_gate_passed": False,
        "internal_numeric_matching_used": False,
    }


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return sink_module.file_sha256(path)


def _contract_fixture(tmp_path: Path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    events = _events()
    splits = {name: _split(name, 1) for name in ("fit", "candidate", "sealed")}
    training = _training()
    source = {"factor_state_sha256": _digest("source-factor")}
    audit = {"audit_report_sha256": _digest("audit-report")}
    runtime = {
        "dtype": "float16",
        "attention_implementation": "eager",
        "explicit_execute_required": True,
    }
    split_rows = {
        name: {
            "path": f"{name}.json",
            "sha256": split.sha256,
            "expected_count": split.expected_count,
        }
        for name, split in splits.items()
    }
    common = {
        "source": source,
        "audit": audit,
        "training": training,
        "seed": 0,
        "runtime": runtime,
        "splits": split_rows,
    }
    current_path = tmp_path / "current.json"
    historical_path = tmp_path / "historical.json"
    current_sha = _write_json(current_path, common)
    _write_json(
        tmp_path / "build_report.json",
        {
            "preregistration_sha256": current_sha,
            "ready_for_explicit_preflight": True,
            "training_launched": False,
            "model_weights_loaded": False,
            "model_checkpoint_created": False,
        },
    )
    historical_sha = _write_json(historical_path, common)
    evidence = _historical_evidence(events)
    evidence_sha = sink_module.canonical_sha256(evidence)
    historical_report = {
        "preregistration_sha256": historical_sha,
        "status": "rejected",
        "reason": "fit-gate-rejected",
        "rollback_exact": True,
        "rollback_state_sha256": recovery.EXPECTED_ROLLBACK_STATE_SHA256,
        "model_checkpoint_written": False,
        "trainer_evidence_sha256": evidence_sha,
        "trainer_evidence": evidence,
        "compiled_child_cache": {
            "compiler_identity_sha256": _digest("compiler-identity"),
            "receipt_sha256": _digest("cache-receipt"),
        },
    }
    historical_report_path = tmp_path / "historical-run.json"
    historical_report_sha = _write_json(historical_report_path, historical_report)
    cache = SimpleNamespace(
        expected_compiler_identity_sha256=_digest("compiler-identity"),
        expected_receipt_sha256=_digest("cache-receipt"),
    )
    threshold = SimpleNamespace(as_dict=lambda: {})
    current_contract = SimpleNamespace(
        execution_contract=SimpleNamespace(
            splits=splits,
            compiler_cache=cache,
            training=training,
            thresholds={"candidate": threshold},
            seed=0,
            continuation=SimpleNamespace(factor_payload={}),
        ),
        evaluation_contract_sha256=_digest("current-evaluation"),
        artifacts=SimpleNamespace(source_run_report_sha256=_digest("source-run")),
    )
    monkeypatch.setattr(
        recovery.guided,
        "authenticate_guided_preregistration",
        lambda *_args, **_kwargs: current_contract,
    )
    monkeypatch.setattr(
        recovery,
        "_resolve_split",
        lambda _row, *, base, name: splits[name],
    )
    return SimpleNamespace(
        events=events,
        splits=splits,
        current_path=current_path,
        current_sha=current_sha,
        historical_path=historical_path,
        historical_sha=historical_sha,
        historical_report_path=historical_report_path,
        historical_report_sha=historical_report_sha,
        historical_report=historical_report,
        current_contract=current_contract,
    )


def _build_payload(fixture):
    return recovery.build_recovery_preregistration_payload(
        current_guided_preregistration=fixture.current_path,
        current_guided_preregistration_sha256=fixture.current_sha,
        historical_guided_preregistration=fixture.historical_path,
        historical_guided_preregistration_sha256=fixture.historical_sha,
        historical_guided_run_report=fixture.historical_report_path,
        historical_guided_run_report_sha256=fixture.historical_report_sha,
        builder_path=Path(prepare.__file__).resolve(),
    )[0]


def test_model_free_preparer_binds_complete_historical_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _contract_fixture(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    report = prepare.prepare_guided_checkpoint_recovery(
        output_dir=output,
        current_guided_preregistration=fixture.current_path,
        historical_guided_preregistration=fixture.historical_path,
        historical_guided_run_report=fixture.historical_report_path,
    )
    assert report["model_weights_loaded"] is False
    assert report["training_launched"] is False
    assert report["retained_steps"] == [40, 56, 64]
    prereg = json.loads((output / "preregistration.json").read_text())
    assert len(prereg["checkpoint_expectations"]) == 12
    assert prereg["sealed_policy"] == "historical-default-candidate-gated"


def test_recovery_rejects_tampered_historical_or_current_prereg(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _contract_fixture(tmp_path, monkeypatch)
    fixture.historical_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(recovery.GuidedCheckpointRecoveryRunError, match="binding"):
        _build_payload(fixture)

    fixture = _contract_fixture(tmp_path / "second", monkeypatch)
    current = json.loads(fixture.current_path.read_text())
    current["source"] = {"factor_state_sha256": _digest("drift")}
    fixture.current_sha = _write_json(fixture.current_path, current)
    _write_json(
        fixture.current_path.with_name("build_report.json"),
        {
            "preregistration_sha256": fixture.current_sha,
            "ready_for_explicit_preflight": True,
            "training_launched": False,
            "model_weights_loaded": False,
            "model_checkpoint_created": False,
        },
    )
    with pytest.raises(recovery.GuidedCheckpointRecoveryRunError, match="source drift"):
        _build_payload(fixture)


def test_recovery_rejects_split_semantic_drift(tmp_path: Path, monkeypatch) -> None:
    fixture = _contract_fixture(tmp_path, monkeypatch)
    drift = _split("candidate", 9)
    monkeypatch.setattr(
        recovery,
        "_resolve_split",
        lambda _row, *, base, name: (
            drift if name == "candidate" else fixture.splits[name]
        ),
    )
    with pytest.raises(recovery.GuidedCheckpointRecoveryRunError, match="candidate"):
        _build_payload(fixture)


def _authenticated_contract(tmp_path: Path, events):
    evidence = _historical_evidence(events)
    evidence_sha = sink_module.canonical_sha256(evidence)
    expectations = tuple(
        sink_module.HistoricalCheckpointExpectation(
            step=event.step,
            metrics=event.metrics.as_dict(),
            metrics_sha256=sink_module.canonical_sha256(event.metrics.as_dict()),
            factor_state_sha256=event.factor_state.factor_state_sha256,
        )
        for event in events
    )
    raw = {
        "historical_guided": {
            "run_report_sha256": _digest("historical-run"),
            "trainer_evidence_sha256": evidence_sha,
            "schedule_sha256": evidence["schedule_sha256"],
            "domain_binding_sha256": evidence["domain_binding_sha256"],
            "training_audit": evidence["training_audit"],
            "checkpoint_rows": evidence["checkpoints"],
        },
        "current_guided": {},
        "execution_runtime": recovery.EXPECTED_RUNTIME,
        "source_implementation": {},
        "semantic_contract_sha256": _digest("semantic"),
        "checkpoint_expectations_sha256": _digest("expectations"),
    }
    return recovery.AuthenticatedGuidedCheckpointRecovery(
        path=tmp_path / "recovery.json",
        file_sha256=_digest("recovery-prereg"),
        current_contract=SimpleNamespace(
            artifacts=SimpleNamespace(source_run_report_sha256=_digest("source-run"))
        ),
        expectations=expectations,
        retained_steps=(40, 56, 64),
        raw=raw,
    )


def _guided_report(events, contract, *, reason="fit-gate-rejected", sealed=None):
    evidence = _historical_evidence(events)
    evidence_sha = sink_module.canonical_sha256(evidence)
    return {
        "status": "rejected",
        "reason": reason,
        "rollback_exact": True,
        "rollback_state_sha256": recovery.EXPECTED_ROLLBACK_STATE_SHA256,
        "model_checkpoint_written": False,
        "trainer_evidence": evidence,
        "trainer_evidence_sha256": evidence_sha,
        "pre_bake_behavior_decisions": {
            "fit": {"passed": False},
            "candidate": {"passed": False},
            "sealed": sealed,
        },
        "preregistration_sha256": getattr(contract, "file_sha256", _digest("x")),
    }


def _fake_executor(events, *, missing_step=None, reason="fit-gate-rejected", sealed=None):
    def execute(contract, output, *, runtime):
        output.mkdir(parents=True)
        for event in events:
            if event.step != missing_step:
                runtime.recovery_sink(event)
        report = _guided_report(events, contract, reason=reason, sealed=sealed)
        stage._atomic_json(output / "run_report.json", report)
        return report

    return execute


def _runtime_factory(sink, **_kwargs):
    return SimpleNamespace(recovery_sink=sink)


def test_successful_wrapper_publishes_exactly_three_after_terminal_match(
    tmp_path: Path, monkeypatch
) -> None:
    events = _events()
    contract = _authenticated_contract(tmp_path, events)
    monkeypatch.setattr(
        recovery, "authenticate_recovery_preregistration", lambda *_args: contract
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    output = tmp_path / "run"
    report = recovery.execute_authenticated_guided_checkpoint_recovery(
        contract,
        output,
        runtime_factory=_runtime_factory,
        guided_executor=_fake_executor(events),
    )
    assert report["status"] == "complete"
    research = output / "recovered_research"
    manifest = json.loads((research / "research_manifest.json").read_text())
    assert manifest["retained_steps"] == [40, 56, 64]
    assert manifest["retained_unique_factor_state_count"] == 3
    assert len(list((research / "retained").iterdir())) == 3
    assert report["sealed_evaluation_accessed"] is False


@pytest.mark.parametrize(
    ("missing_step", "reason", "sealed"),
    [
        (48, "fit-gate-rejected", None),
        (None, "candidate-gate-rejected", None),
        (None, "fit-gate-rejected", {"passed": True}),
    ],
)
def test_missing_checkpoint_or_terminal_mismatch_prevents_publication(
    tmp_path: Path,
    monkeypatch,
    missing_step,
    reason,
    sealed,
) -> None:
    events = _events()
    contract = _authenticated_contract(tmp_path, events)
    monkeypatch.setattr(
        recovery, "authenticate_recovery_preregistration", lambda *_args: contract
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    output = tmp_path / f"run-{missing_step}-{reason}-{bool(sealed)}"
    with pytest.raises(Exception):
        recovery.execute_authenticated_guided_checkpoint_recovery(
            contract,
            output,
            runtime_factory=_runtime_factory,
            guided_executor=_fake_executor(
                events,
                missing_step=missing_step,
                reason=reason,
                sealed=sealed,
            ),
        )
    assert not (output / "recovered_research").exists()
    assert json.loads((output / "run_report.json").read_text())["status"] == "failed"


def test_runtime_copies_exact_schedule_and_wires_callback(monkeypatch) -> None:
    observed = {}
    result = object()

    class FakeSchedule:
        sealed_evaluation_policy = "candidate-gated"

        def __init__(self, **values):
            self.values = values

        def as_dict(self):
            return dict(self.values)

    class FakeResult:
        pass

    result = FakeResult()

    def trainer(**kwargs):
        observed.update(kwargs)
        return result

    fake_module = SimpleNamespace(
        load_behavior_domain_suite=lambda name, payload, artifact_sha256: (
            name,
            payload,
            artifact_sha256,
        ),
        BehaviorDistillationSchedule=FakeSchedule,
        CandidateBehaviorThresholds=lambda **values: values,
        train_target_native_behavior_distillation=trainer,
        BehaviorDistillationResult=FakeResult,
    )
    monkeypatch.setattr(recovery.importlib, "import_module", lambda _name: fake_module)
    sink = object()
    runtime = recovery.RecoveryProductionGuidedRuntime(sink)
    runtime._guided_tranche = object()
    training = {"steps": 64}
    split = SimpleNamespace(payload={}, sha256=_digest("split"))
    base = SimpleNamespace(
        splits={name: split for name in ("fit", "candidate", "sealed")},
        training=training,
        thresholds={"candidate": SimpleNamespace(as_dict=lambda: {})},
        seed=0,
        continuation=SimpleNamespace(factor_payload={"schema": "v2"}),
    )
    returned = runtime.train_guided(
        object(), object(), SimpleNamespace(execution_contract=base)
    )
    assert returned is result
    assert set(observed["domains"]) == {"fit", "candidate", "sealed"}
    assert observed["durable_checkpoint_callback"] is sink
    assert observed["schedule"].sealed_evaluation_policy == "candidate-gated"
