from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import prepare_qwen_exact_target_native_guided_continuation as prepare  # noqa: E402
import run_qwen_exact_target_native_guided_continuation as guided  # noqa: E402
import run_qwen_exact_target_native_stage as stage  # noqa: E402
import replay_qwen_unified_v8_commit as replay  # noqa: E402


def _metrics(
    *,
    split: str = "calibration-selection",
    lcp4: int = 1,
    repeat: float = 0.1,
    adjacent: float = 0.1,
    action_accuracy: float = 1.0,
) -> dict[str, object]:
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
        "lcp_at_least_4_count": lcp4,
        "repeat_fraction_excess": repeat,
        "adjacent_repeat_fraction_excess": adjacent,
        "longest_run_excess": 0,
        "report_sha256": "a" * 64,
    }


def _method_evidence(
    *,
    selected_step: int = 24,
    lcp4: int = 1,
    repeat: float = 0.1,
    adjacent: float = 0.1,
) -> dict[str, object]:
    step0 = _metrics(lcp4=0, repeat=0.2, adjacent=0.2)
    candidate = _metrics(lcp4=lcp4, repeat=repeat, adjacent=adjacent)
    promoted = selected_step > 0
    gate = {
        "selected_step": selected_step,
        "selected_step_is_promoted": promoted,
        "minimum_lcp_at_least_4_count": 1,
        "observed_lcp_at_least_4_count": lcp4,
        "baseline_repeat_fraction_excess": 0.2,
        "candidate_repeat_fraction_excess": repeat,
        "repeat_fraction_nonregression": repeat <= 0.2,
        "baseline_adjacent_repeat_fraction_excess": 0.2,
        "candidate_adjacent_repeat_fraction_excess": adjacent,
        "adjacent_repeat_fraction_nonregression": adjacent <= 0.2,
        "passed": promoted and lcp4 >= 1 and repeat <= 0.2 and adjacent <= 0.2,
    }
    return {
        "selected_step": selected_step,
        "checkpoints": [
            {"step": 0, "metrics": step0},
            {"step": selected_step, "metrics": candidate},
        ],
        "candidate_metrics": candidate,
        "guided_candidate_gate": gate,
    }


def test_guided_method_gate_requires_one_contiguous_lcp_of_four() -> None:
    # Four rollouts that each match only one token still have lcp4=0.  Aggregate
    # LCP tokens cannot substitute for one context-preserving four-token path.
    passed, decision = guided._guided_method_gate(_method_evidence(lcp4=0))
    assert passed is False
    assert decision["checks"]["candidate_has_one_lcp_at_least_4_rollout"] is False
    assert decision["four_isolated_one_token_lcps_cannot_substitute"] is True


@pytest.mark.parametrize(
    ("selected_step", "lcp4", "repeat", "adjacent", "expected"),
    [
        (24, 1, 0.2, 0.2, True),
        (0, 1, 0.2, 0.2, False),
        (24, 1, 0.21, 0.2, False),
        (24, 1, 0.2, 0.21, False),
    ],
)
def test_guided_method_gate_binds_promotion_and_repetition_nonregression(
    selected_step: int,
    lcp4: int,
    repeat: float,
    adjacent: float,
    expected: bool,
) -> None:
    passed, _decision = guided._guided_method_gate(
        _method_evidence(
            selected_step=selected_step,
            lcp4=lcp4,
            repeat=repeat,
            adjacent=adjacent,
        )
    )
    assert passed is expected


def _split_binding(
    tmp_path: Path,
    name: str,
    rows: list[tuple[int, ...]],
    *,
    path: Path | None = None,
) -> stage.SplitBinding:
    payload = {
        "schema": stage.SPLIT_SCHEMA,
        "split": name,
        "sealed_before_model_access": name == "sealed",
        "prompts": [
            {"prompt_id": f"{name}-{index}", "token_ids": list(tokens)}
            for index, tokens in enumerate(rows)
        ],
    }
    target = tmp_path / f"{name}.json" if path is None else path
    return stage.SplitBinding(
        name=name,
        path=target,
        sha256=hashlib.sha256(repr(payload).encode()).hexdigest(),
        expected_count=len(rows),
        payload=payload,
        row_ids=tuple(f"{name}-{index}" for index in range(len(rows))),
        prompt_sha256s=tuple(stage._canonical_sha256(list(tokens)) for tokens in rows),
    )


def test_split_evolution_is_exact_144_fresh24_reused_locked32(tmp_path: Path) -> None:
    fit_rows = [(1, index + 10, 2) for index in range(120)]
    old_candidate = [(3, index + 200, 4) for index in range(24)]
    sealed_rows = [(5, index + 300, 6) for index in range(32)]
    source_sealed = _split_binding(tmp_path, "sealed", sealed_rows)
    source = SimpleNamespace(
        splits={
            "fit": _split_binding(tmp_path, "fit", fit_rows),
            "candidate": _split_binding(tmp_path, "candidate", old_candidate),
            "sealed": source_sealed,
        }
    )
    active = {
        "fit": _split_binding(tmp_path, "fit", fit_rows + old_candidate),
        "candidate": _split_binding(
            tmp_path, "candidate", [(7, index + 400, 8) for index in range(24)]
        ),
        "sealed": source_sealed,
    }
    guided._authenticate_split_evolution(active, source)
    assert active["sealed"] is source_sealed


def _historical_evidence() -> tuple[dict[str, object], dict[str, object]]:
    implementation: dict[str, object] = {
        "trainer_module": stage.CANONICAL_TRAINER_MODULE
    }
    evidence: dict[str, object] = {"authenticated_before_model_access": True}
    for field, (key, relative) in stage._HISTORICAL_IMPLEMENTATION_BLOBS.items():
        blob = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{prepare.HISTORICAL_AUDIT_COMMIT}:{relative}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(blob).hexdigest()
        implementation[field] = digest
        row: dict[str, object] = {
            "path": f"C:/historical/neural-forge/{relative}",
            "sha256": digest,
        }
        if key in stage._HISTORICAL_IMPLEMENTATION_MODULES:
            row["module"] = stage._HISTORICAL_IMPLEMENTATION_MODULES[key]
        evidence[key] = row
    return implementation, evidence


def test_historical_sources_authenticate_from_commit_not_live_files() -> None:
    implementation, evidence = _historical_evidence()
    required = set(implementation)
    result = stage._authenticate_historical_implementation_blobs(
        implementation,
        evidence,
        prepare.HISTORICAL_AUDIT_COMMIT,
        required_fields=required,
    )
    assert result == evidence
    assert implementation["qkvo_adapter_sha256"] != stage._file_sha256(
        ROOT / "src/neural_forge/hf_dense_gqa_qkvo_residual_adapter.py"
    )


def test_historical_source_tamper_and_nonfull_revision_fail() -> None:
    implementation, evidence = _historical_evidence()
    bad = dict(evidence)
    bad["trainer"] = {**bad["trainer"], "sha256": "0" * 64}
    with pytest.raises(stage.ExactTargetNativeStageError):
        stage._authenticate_historical_implementation_blobs(
            implementation,
            bad,
            prepare.HISTORICAL_AUDIT_COMMIT,
            required_fields=set(implementation),
        )
    with pytest.raises(stage.ExactTargetNativeStageError):
        stage._authenticate_historical_implementation_blobs(
            implementation,
            evidence,
            "e57d0b2",
            required_fields=set(implementation),
        )


def test_authenticate_guided_preregistration_constructs_complete_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_threshold = stage.BehaviorThresholds(
        minimum_action_accuracy=0.1,
        minimum_exact_rollout_fraction=0.0,
        minimum_token_match_fraction=0.0,
        minimum_lcp_fraction=0.0,
        minimum_action_count=192,
        minimum_rollout_count=24,
    )
    source_splits = {
        name: _split_binding(tmp_path, name, [(index + 1,) for index in range(count)])
        for name, count in {"fit": 120, "candidate": 24, "sealed": 32}.items()
    }
    source_contract = stage.AuthenticatedPreRegistration(
        path=tmp_path / "source.json",
        file_sha256="1" * 64,
        parent=SimpleNamespace(),
        target=SimpleNamespace(),
        splits=source_splits,
        compiler_split=source_splits["fit"],
        thresholds={name: source_threshold for name in source_splits},
        compiler={},
        training={},
        seed=0,
        evaluation_contract_sha256="2" * 64,
        artifact_authentication={},
        implementation_authentication={},
        continuation=None,
        compiler_cache=None,
        lexical_residual=None,
        output_jacobian_audit=None,
        historical_implementation_revision=prepare.HISTORICAL_AUDIT_COMMIT,
    )
    active_splits = {
        name: _split_binding(tmp_path, name, [(index + 500,) for index in range(count)])
        for name, count in guided.GUIDED_SPLIT_COUNTS.items()
    }
    artifacts = SimpleNamespace(
        source_run_report_sha256="3" * 64,
        source_factor_pack_sha256="4" * 64,
        source_factor_state_sha256="5" * 64,
        audit_preregistration_sha256="6" * 64,
        audit_run_report_sha256="7" * 64,
        audit_report_self_sha256="8" * 64,
        audit_report_canonical_sha256="9" * 64,
        basis_envelope_file_sha256="a" * 64,
        basis_domain_artifact_sha256="b" * 64,
        basis_payload_sha256="c" * 64,
        basis_manifest_sha256="d" * 64,
        basis_tensor_sha256s={role: role * 64 for role in guided.GUIDED_ROLES},
    )
    implementation = {"authenticated": True}
    runtime = {
        "dtype": "float16",
        "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
        "explicit_execute_required": True,
    }
    thresholds = {
        name: source_threshold.as_dict() for name in ("fit", "candidate", "sealed")
    }
    thresholds["fit"].update(
        {"minimum_action_count": 1_152, "minimum_rollout_count": 144}
    )
    threshold_objects = {
        name: stage.BehaviorThresholds.from_mapping(
            thresholds[name], f"thresholds.{name}"
        )
        for name in thresholds
    }
    evaluation_sha = stage._canonical_sha256(
        guided._guided_evaluation_preimage(
            artifacts=artifacts,
            splits=active_splits,
            thresholds=threshold_objects,
            training={},
            implementation=implementation,
            runtime=runtime,
            method_gate=guided.GUIDED_METHOD_GATE,
        )
    )
    preregistration = {
        "schema": guided.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "seed": 0,
        "implementation": {},
        "source": {},
        "audit": {},
        "splits": {name: {} for name in active_splits},
        "thresholds": thresholds,
        "training": {},
        "method_gate": dict(guided.GUIDED_METHOD_GATE),
        "runtime": runtime,
        "evaluation_contract_sha256": evaluation_sha,
    }
    prereg_path = tmp_path / "guided-preregistration.json"
    prereg_path.write_text(
        json.dumps(preregistration, sort_keys=True) + "\n", encoding="utf-8"
    )
    prereg_sha = stage._file_sha256(prereg_path)
    monkeypatch.setattr(
        guided, "_authenticate_implementation", lambda _value: implementation
    )
    monkeypatch.setattr(
        guided,
        "_authenticate_persisted_audit",
        lambda *_args, **_kwargs: (source_contract, artifacts),
    )
    monkeypatch.setattr(
        stage,
        "_load_split",
        lambda _value, *, name, base: active_splits[name],
    )
    monkeypatch.setattr(guided, "_authenticate_split_evolution", lambda *_args: None)
    monkeypatch.setattr(guided, "_authenticate_training", lambda _value, **_kwargs: {})

    contract = guided.authenticate_guided_preregistration(prereg_path, prereg_sha)

    assert dataclasses.is_dataclass(contract)
    assert contract.method_gate == guided.GUIDED_METHOD_GATE
    assert (
        contract.source_contract.historical_implementation_revision
        == prepare.HISTORICAL_AUDIT_COMMIT
    )
    assert contract.execution_contract.historical_implementation_revision is None


def test_guided_builder_publication_uses_marked_verified_copy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "guided"
    staging = tmp_path / ".guided.abc.staging"
    staging.mkdir()
    (staging / "preregistration.json").write_text("{}\n", encoding="utf-8")
    tree = replay._artifact_tree(staging)

    def denied(source, destination):
        del source, destination
        raise PermissionError(errno.EACCES, "simulated CIFS rename denial")

    monkeypatch.setattr(replay.os, "replace", denied)
    publication = replay._publish_staged_checkpoint(staging, output, expected_tree=tree)
    assert publication["mode"] == "verified-copy-fallback"
    assert replay._artifact_tree(output) == tree
    assert not (output / replay._INCOMPLETE_PROMOTION_MARKER).exists()
    assert prepare._remove_owned_staging(staging, output=output, expected_tree=tree)


def test_guided_builder_corrupt_copy_fails_closed_and_removes_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "guided"
    staging = tmp_path / ".guided.abc.staging"
    staging.mkdir()
    source_file = staging / "preregistration.json"
    source_file.write_text("{}\n", encoding="utf-8")
    tree = replay._artifact_tree(staging)
    real_copytree = replay.shutil.copytree

    def denied(source, destination):
        del source, destination
        raise PermissionError(errno.EACCES, "simulated CIFS rename denial")

    def corrupt_copy(source, destination, **kwargs):
        result = real_copytree(source, destination, **kwargs)
        (Path(destination) / "preregistration.json").write_text(
            "corrupt\n", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(replay.os, "replace", denied)
    monkeypatch.setattr(replay.shutil, "copytree", corrupt_copy)
    with pytest.raises(Exception, match="copy differs"):
        replay._publish_staged_checkpoint(staging, output, expected_tree=tree)
    assert not output.exists()
    assert staging.exists()
    prepare._remove_owned_staging(staging, output=output)


class _FakeTransaction:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.is_live = True
        self.qkvo_handle = object()

    def installation_report(self) -> dict[str, object]:
        return {
            "zero_step_identity": True,
            "temporary_lexical_factor_parameters": stage.LEXICAL_FACTOR_PARAMETERS,
        }

    def rollback(self) -> None:
        self.events.append("rollback-base")
        self.is_live = False


class _FakeGuidedRuntime:
    def __init__(self, events: list[str], *, accept: bool) -> None:
        self.events = events
        self.accept = accept
        self.model = SimpleNamespace()
        self.transaction = _FakeTransaction(events)

    def load_models(self, contract):
        del contract
        return object(), object(), object()

    def compile(self, parent, target, tokenizer, contract):
        del parent, target, tokenizer, contract
        return SimpleNamespace(child_model=self.model, report={"compiled": True})

    def state_sha256(self, model):
        del model
        return "b" * 64

    def install_transaction(self, compiled):
        del compiled
        self.events.append("install-base")
        return self.transaction

    def restore_source_v2(self, transaction, contract):
        del transaction, contract
        self.events.append("restore-v2")
        return {"factor_state_sha256": "c" * 64}

    def install_guided_transaction(self, transaction, contract):
        del transaction, contract
        self.events.append("install-guided")
        return {"inherited_installation_sha256": "d" * 64}

    def train_guided(self, transaction, parent, contract):
        del transaction, parent, contract
        self.events.append("train")
        return SimpleNamespace(
            fit_metrics=_metrics(action_accuracy=1.0 if self.accept else 0.0),
            candidate_metrics=_metrics(),
            sealed_metrics=_metrics(split="heldout-evaluation"),
            sealed_evaluation_unlocked=True,
            candidate_gate_passed=True,
        )

    def rollback_guided_installation(self, transaction):
        del transaction
        self.events.append("rollback-guided")

    def handoff_guided_for_publication(self, transaction, contract):
        del contract
        self.events.append("handoff-guided")
        transaction.qkvo_handle = "guided-composite"
        return {"handoff_complete": True}

    def evaluate_baked_behavior(self, model, parent, contract, split_name):
        del model, parent, contract
        self.events.append(f"post-bake-{split_name}")
        split = (
            "heldout-evaluation"
            if split_name == "sealed"
            else (
                "calibration-exploratory"
                if split_name == "fit"
                else "calibration-selection"
            )
        )
        return _metrics(split=split)

    def save_reload_certify(self, model, tokenizer, output):
        del model, tokenizer, output
        self.events.append("fresh-reload")
        return {
            "fresh_reload_performed": True,
            "state_bytes_match_pre_save": True,
            "published_adapter_parameter_count": 0,
            "published_parent_runtime_reference_count": 0,
            "published_source_sized_sidecar_parameter_count": 0,
        }


class _PublishingTransaction(_FakeTransaction):
    def bake_for_publication(self, receipt):
        del receipt
        assert self.qkvo_handle == "guided-composite"
        self.events.append("publish-guided-composite")
        return {"guided_sequential_bake": True}

    def restore_prepublication_native(self):
        self.events.append("restore-publication")


def _fake_contract() -> SimpleNamespace:
    threshold = stage.BehaviorThresholds(
        minimum_action_accuracy=0.5,
        minimum_exact_rollout_fraction=0.25,
        minimum_token_match_fraction=0.5,
        minimum_lcp_fraction=0.35,
        minimum_action_count=192,
        minimum_rollout_count=24,
    )
    base = SimpleNamespace(
        continuation=None,
        thresholds={"fit": threshold, "candidate": threshold, "sealed": threshold},
    )
    artifacts = SimpleNamespace(
        source_factor_state_sha256="1" * 64,
        audit_report_self_sha256="2" * 64,
        audit_report_canonical_sha256="3" * 64,
        basis_envelope_file_sha256="4" * 64,
        basis_domain_artifact_sha256="5" * 64,
        basis_payload_sha256="6" * 64,
    )
    return SimpleNamespace(
        execution_contract=base,
        artifacts=artifacts,
        file_sha256="7" * 64,
        evaluation_contract_sha256="8" * 64,
        preflight_report=lambda: {"schema": guided.RUN_SCHEMA},
    )


def _minimal_authenticated_guided_contract(tmp_path: Path):
    return guided.AuthenticatedGuidedContinuation(
        path=tmp_path / "preregistration.json",
        file_sha256="a" * 64,
        source_contract=SimpleNamespace(
            historical_implementation_revision=prepare.HISTORICAL_AUDIT_COMMIT
        ),
        execution_contract=SimpleNamespace(historical_implementation_revision=None),
        artifacts=SimpleNamespace(),
        evaluation_contract_sha256="b" * 64,
        implementation_authentication={"runner_sha256": "c" * 64},
        source_authentication={},
        training={},
        method_gate=dict(guided.GUIDED_METHOD_GATE),
    )


def test_production_execution_reauthenticates_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _minimal_authenticated_guided_contract(tmp_path)
    output = tmp_path / "run"

    def stale(*_args, **_kwargs):
        raise guided.GuidedContinuationError("stale guided source")

    monkeypatch.setattr(guided, "authenticate_guided_preregistration", stale)
    with pytest.raises(guided.GuidedContinuationError, match="stale guided source"):
        guided.execute_authenticated_guided_continuation(
            contract,
            output,
            runtime=guided.ProductionGuidedRuntime(),
        )
    assert not output.exists()


def test_execution_refresh_requires_cleared_live_historical_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _minimal_authenticated_guided_contract(tmp_path)
    stale_execution = dataclasses.replace(
        contract,
        execution_contract=SimpleNamespace(
            historical_implementation_revision=prepare.HISTORICAL_AUDIT_COMMIT
        ),
    )
    monkeypatch.setattr(
        guided,
        "authenticate_guided_preregistration",
        lambda *_args, **_kwargs: stale_execution,
    )
    with pytest.raises(guided.GuidedContinuationError, match="clearing the historical"):
        guided._refresh_guided_execution_contract(contract)


def _fake_v3_evidence(accept: bool) -> tuple[str, dict[str, object], dict[str, object]]:
    evidence = _method_evidence()
    evidence.update(
        {
            "fit_metrics": _metrics(action_accuracy=1.0 if accept else 0.0),
            "sealed_metrics": _metrics(split="heldout-evaluation") if accept else None,
        }
    )
    payload = {
        "schema": guided.TRAINER_FACTOR_PACK_V3_SCHEMA,
        "factor_tensors": {"factor": torch.zeros(1)},
    }
    return "9" * 64, evidence, payload


def test_execute_restores_v2_before_guided_install_and_rolls_back_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    runtime = _FakeGuidedRuntime(events, accept=False)
    monkeypatch.setattr(
        guided,
        "_validate_v3_evidence",
        lambda result, contract, installation: _fake_v3_evidence(False),
    )
    monkeypatch.setattr(stage, "_factor_acquisition_claim", lambda evidence: {})
    report = guided.execute_authenticated_guided_continuation(
        _fake_contract(), tmp_path / "run", runtime=runtime
    )
    assert report["status"] == "rejected"
    assert events[:4] == ["install-base", "restore-v2", "install-guided", "train"]
    assert events[-2:] == ["rollback-guided", "rollback-base"]


def test_acceptance_hands_composite_to_atomic_publication_before_bake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    runtime = _FakeGuidedRuntime(events, accept=True)
    runtime.transaction = _PublishingTransaction(events)
    monkeypatch.setattr(
        guided,
        "_validate_v3_evidence",
        lambda result, contract, installation: _fake_v3_evidence(True),
    )
    monkeypatch.setattr(stage, "_factor_acquisition_claim", lambda evidence: {})
    report = guided.execute_authenticated_guided_continuation(
        _fake_contract(), tmp_path / "run", runtime=runtime
    )
    assert report["status"] == "accepted"
    assert events.index("handoff-guided") < events.index("publish-guided-composite")
    assert report["fresh_reload_certificate"]["published_adapter_parameter_count"] == 0


def test_cli_requires_literal_execute_and_rejects_implicit_output_root() -> None:
    with pytest.raises(SystemExit):
        guided.build_parser().parse_args(
            [
                "--preregistration",
                "preregistration.json",
                "--preregistration-sha256",
                "a" * 64,
                "--output-root",
                "run",
            ]
        )


def test_cli_execute_and_preflight_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        guided.build_parser().parse_args(
            [
                "--preregistration",
                "preregistration.json",
                "--preregistration-sha256",
                "a" * 64,
                "--preflight",
                "--execute",
            ]
        )


def test_cli_execute_requires_output_root_before_runtime_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        guided,
        "authenticate_guided_preregistration",
        lambda *_args, **_kwargs: object(),
    )

    def forbidden_runtime(**_kwargs):
        raise AssertionError("runtime must not be constructed without --output-root")

    monkeypatch.setattr(guided, "ProductionGuidedRuntime", forbidden_runtime)
    with pytest.raises(SystemExit, match="--output-root is required with --execute"):
        guided.main(
            [
                "--preregistration",
                "preregistration.json",
                "--preregistration-sha256",
                "a" * 64,
                "--execute",
            ]
        )
