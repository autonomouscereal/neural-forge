from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_exact_target_native_stage as runner  # noqa: E402


def _write_json(path: Path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, label: str) -> tuple[Path, str]:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "label": label}), encoding="utf-8"
    )
    weight = root / "model.safetensors"
    weight.write_bytes(f"{label}-weights".encode())
    return root, hashlib.sha256(weight.read_bytes()).hexdigest()


def _split(path: Path, name: str, token: int) -> tuple[Path, str]:
    payload = {
        "schema": runner.SPLIT_SCHEMA,
        "split": name,
        "sealed_before_model_access": name == "sealed",
        "prompts": [
            {"token_ids": [1, token, token + 1]},
            {"token_ids": [1, token + 2, token + 3]},
        ],
    }
    return path, _write_json(path, payload)


def _threshold(*, accuracy: float = 0.75) -> dict[str, object]:
    return {
        "minimum_action_accuracy": accuracy,
        "minimum_exact_rollout_fraction": 0.5,
        "minimum_token_match_fraction": 0.5,
        "minimum_lcp_fraction": 0.5,
        "minimum_action_count": 2,
        "minimum_rollout_count": 2,
    }


def _contract_files(tmp_path: Path) -> tuple[Path, str]:
    parent, parent_weight = _artifact(tmp_path / "parent", "parent")
    target, target_weight = _artifact(tmp_path / "target", "target")
    fit, fit_sha = _split(tmp_path / "fit.json", "fit", 10)
    candidate, candidate_sha = _split(tmp_path / "candidate.json", "candidate", 20)
    sealed, sealed_sha = _split(tmp_path / "sealed.json", "sealed", 30)
    payload = {
        "schema": runner.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "implementation": {
            "runner_sha256": hashlib.sha256(
                Path(runner.__file__).read_bytes()
            ).hexdigest(),
            "trainer_module": runner.CANONICAL_TRAINER_MODULE,
            "trainer_sha256": hashlib.sha256(
                (
                    Path(__file__).resolve().parents[1]
                    / "src"
                    / "neural_forge"
                    / "hf_target_native_behavior_distillation.py"
                ).read_bytes()
            ).hexdigest(),
        },
        "seed": 17,
        "parent": {
            "path": str(parent),
            "exact_unique_parameters": runner.QWEN3_4B_EXACT_PARAMETERS,
            "required_files": {"model.safetensors": parent_weight},
        },
        "blank_target": {
            "path": str(target),
            "exact_unique_parameters": runner.QWEN3_1P4B_EXACT_PARAMETERS,
            "required_files": {"model.safetensors": target_weight},
        },
        "splits": {
            "fit": {"path": str(fit), "sha256": fit_sha, "expected_count": 2},
            "candidate": {
                "path": str(candidate),
                "sha256": candidate_sha,
                "expected_count": 2,
            },
            "sealed": {
                "path": str(sealed),
                "sha256": sealed_sha,
                "expected_count": 2,
            },
        },
        "thresholds": {
            "fit": _threshold(),
            "candidate": _threshold(),
            "sealed": _threshold(),
        },
        "compiler": {
            "capture_batch_size": 1,
            "max_prompt_tokens": 32,
            "bootstrap_kwargs": {},
        },
        "training": {
            "steps": 2,
            "learning_rate": 0.001,
            "selection_policy": "rollout-first-acquisition",
        },
    }
    preregistration = tmp_path / "preregistration.json"
    return preregistration, _write_json(preregistration, payload)


def _metrics(*, accuracy: float = 1.0) -> dict[str, object]:
    return {
        "action_count": 4,
        "action_agreement_count": round(4 * accuracy),
        "action_accuracy": accuracy,
        "rollout_count": 4,
        "exact_rollout_count": round(4 * accuracy),
        "exact_rollout_fraction": accuracy,
        "rollout_token_count": 8,
        "token_match_count": round(8 * accuracy),
        "token_match_fraction": accuracy,
        "lcp_token_count": round(8 * accuracy),
        "lcp_fraction": accuracy,
    }


class _Report:
    def as_dict(self):
        return {
            "bootstrap_accepted": True,
            "target_budget_exact": True,
            "target_unique_parameter_count": runner.QWEN3_1P4B_EXACT_PARAMETERS,
            "capability_proven": False,
        }


class _Transaction:
    def __init__(self, model) -> None:
        self.model = model
        self.is_live = True
        self.rolled_back = False
        self.baked = False

    def installation_report(self):
        return {
            "zero_step_identity": True,
            "temporary_qkvo_factor_parameters": 120,
            "temporary_gate_up_down_factor_parameters": 90,
            "role_complete_attention_qkvo": True,
            "role_complete_mlp_gate_up_down": True,
        }

    def rollback(self):
        self.is_live = False
        self.rolled_back = True
        return {"restored": True}

    def bake_for_publication(self, receipt):
        assert receipt.internal_similarity_only is False
        self.is_live = False
        self.baked = True
        return {
            "exact_budget_match": True,
            "published_adapter_parameter_count": 0,
            "weight_or_activation_similarity_can_promote": False,
        }


@dataclass
class _TrainingResult:
    fit_metrics: object
    candidate_metrics: object
    sealed_metrics: object | None
    candidate_gate_passed: bool
    sealed_evaluation_unlocked: bool
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        self.evidence_sha256 = runner._canonical_sha256(self.evidence_payload())

    def evidence_payload(self):
        return {
            "checkpoints": [{"step": 0, "factor_state_sha256": "f" * 64}],
            "selected_step": 0,
            "selection_policy": "rollout-first-acquisition",
            "fit_metrics": self.fit_metrics,
            "candidate_metrics": self.candidate_metrics,
            "sealed_metrics": self.sealed_metrics,
            "training_audit": {
                "gradient_activation": {
                    "every_qkvo_gud_factor_gradient_activated": True
                }
            },
            "factor_pack_manifest": {
                "factor_state_sha256": "f" * 64,
                "factors": ["q.left", "gud.left"],
            },
            "domain_binding_sha256": "d" * 64,
            "schedule_sha256": "c" * 64,
            "candidate_thresholds": _threshold(),
            "candidate_gate_passed": self.candidate_gate_passed,
            "sealed_evaluation_unlocked": self.sealed_evaluation_unlocked,
            "internal_numeric_matching_used": False,
        }

    def factor_pack_payload(self):
        return {
            "schema": "test-factor-pack",
            "factors": {"q.left": torch.ones(2), "gud.left": torch.zeros(3)},
        }


class _Runtime:
    def __init__(self, result: _TrainingResult) -> None:
        self.result = result
        self.model = SimpleNamespace(state="compiled")
        self.transaction = _Transaction(self.model)
        self.save_calls = 0

    def load_models(self, contract):
        del contract
        return object(), self.model, object()

    def compile(self, parent, target, tokenizer, contract):
        del parent, tokenizer, contract
        assert target is self.model
        return SimpleNamespace(child_model=self.model, report=_Report())

    def state_sha256(self, model):
        assert model is self.model
        return hashlib.sha256(model.state.encode()).hexdigest()

    def install_transaction(self, compiled):
        assert compiled.child_model is self.model
        return self.transaction

    def train(self, transaction, parent, contract):
        del parent, contract
        assert transaction is self.transaction
        return self.result

    def factor_pack_payload(self, result):
        return result.factor_pack_payload()

    def save_reload_certify(self, model, tokenizer, output_root):
        del tokenizer
        assert model is self.model
        self.save_calls += 1
        child = output_root / "accepted_child"
        child.mkdir()
        (child / "model.safetensors").write_bytes(b"baked")
        return {
            "fresh_reload_performed": True,
            "exact_unique_parameters": runner.QWEN3_1P4B_EXACT_PARAMETERS,
            "state_keys_match_pre_save": True,
            "published_adapter_parameter_count": 0,
            "model_save_pretrained_calls": 1,
        }


def test_preflight_authenticates_everything_without_loading_models(tmp_path) -> None:
    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    report = contract.preflight_report()
    assert report["status"] == "preflight-passed"
    assert report["qkvo"] == {
        "site_count": 120,
        "rank": 8,
        "all_target_layers": True,
        "all_physical_groups": True,
    }
    assert report["gate_up_down"]["layer_count"] == 30
    assert report["models_loaded"] is False
    assert report["gpu_work_launched"] is False
    assert report["internal_numeric_matching_can_promote"] is False


def test_hash_mismatch_fails_before_any_model_access(tmp_path) -> None:
    path, _digest = _contract_files(tmp_path)
    with pytest.raises(runner.ExactTargetNativeStageError, match="hash mismatch"):
        runner.authenticate_preregistration(path, "0" * 64)


def test_overlapping_token_prompts_are_rejected(tmp_path) -> None:
    path, _digest = _contract_files(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_path = Path(payload["splits"]["candidate"]["path"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    fit_path = Path(payload["splits"]["fit"]["path"])
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    candidate["prompts"][0] = fit["prompts"][0]
    payload["splits"]["candidate"]["sha256"] = _write_json(candidate_path, candidate)
    digest = _write_json(path, payload)
    with pytest.raises(runner.ExactTargetNativeStageError, match="overlap"):
        runner.authenticate_preregistration(path, digest)


def test_candidate_output_failure_retains_only_factor_pack_and_report(tmp_path) -> None:
    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    # An irrelevant perfect numeric similarity must not rescue failed actions.
    candidate = {**_metrics(accuracy=0.25), "weight_similarity": 1.0}
    runtime = _Runtime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=candidate,
            sealed_metrics=None,
            candidate_gate_passed=False,
            sealed_evaluation_unlocked=False,
        )
    )
    output = tmp_path / "rejected"
    report = runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert report["status"] == "rejected"
    assert report["reason"] == "candidate-gate-rejected"
    assert report["model_checkpoint_written"] is False
    assert report["rollback_exact"] is True
    assert runtime.transaction.rolled_back is True
    assert runtime.save_calls == 0
    assert {path.name for path in output.iterdir()} == {
        "rejected_factor_pack.pt",
        "run_report.json",
    }
    assert (
        report["behavior_decisions"]["candidate"]["internal_numeric_matching_consulted"]
        is False
    )


def test_sealed_failure_rolls_back_without_checkpoint(tmp_path) -> None:
    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _Runtime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=_metrics(),
            sealed_metrics=_metrics(accuracy=0.25),
            candidate_gate_passed=True,
            sealed_evaluation_unlocked=True,
        )
    )
    report = runner.execute_authenticated_stage(
        contract, tmp_path / "sealed-reject", runtime=runtime
    )
    assert report["reason"] == "sealed-gate-rejected"
    assert runtime.transaction.rolled_back is True
    assert runtime.save_calls == 0


def test_accept_bakes_once_and_records_fresh_reload_certificate(tmp_path) -> None:
    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _Runtime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=_metrics(),
            sealed_metrics=_metrics(),
            candidate_gate_passed=True,
            sealed_evaluation_unlocked=True,
        )
    )
    output = tmp_path / "accepted"
    report = runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert report["status"] == "accepted"
    assert runtime.transaction.baked is True
    assert runtime.transaction.rolled_back is False
    assert runtime.save_calls == 1
    assert report["model_save_count"] == 1
    assert report["selection_used_internal_numeric_matching"] is False
    assert report["fresh_reload_certificate"]["fresh_reload_performed"] is True
    assert (output / "accepted_child" / "model.safetensors").is_file()
    assert not (output / "rejected_factor_pack.pt").exists()


def test_sealed_access_before_candidate_gate_fails_closed(tmp_path) -> None:
    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _Runtime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=_metrics(accuracy=0.25),
            sealed_metrics=_metrics(),
            candidate_gate_passed=False,
            sealed_evaluation_unlocked=True,
        )
    )
    with pytest.raises(
        runner.ExactTargetNativeStageError,
        match="Sealed evaluation was accessed",
    ):
        runner.execute_authenticated_stage(
            contract, tmp_path / "bad-unlock", runtime=runtime
        )
    assert runtime.transaction.rolled_back is True


def test_post_publish_report_failure_removes_exact_accepted_child(
    tmp_path, monkeypatch
) -> None:
    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _Runtime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=_metrics(),
            sealed_metrics=_metrics(),
            candidate_gate_passed=True,
            sealed_evaluation_unlocked=True,
        )
    )

    def fail_report(*_args, **_kwargs):
        raise OSError("injected report fsync failure")

    monkeypatch.setattr(runner, "_atomic_json", fail_report)
    output = tmp_path / "report-failure"
    with pytest.raises(OSError, match="injected report"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "accepted_child").exists()


def test_gate_rejects_out_of_domain_fraction_and_runtime_is_source_pinned() -> None:
    assert runner.CANONICAL_ATTENTION_IMPLEMENTATION == "eager"
    with pytest.raises(runner.ExactTargetNativeStageError, match=r"\[0, 1\]"):
        runner._behavior_gate(
            {**_metrics(), "action_accuracy": 2.0},
            runner.BehaviorThresholds.from_mapping(_threshold(), "test"),
        )
    with pytest.raises(runner.ExactTargetNativeStageError, match="pinned"):
        runner.ProductionStageRuntime(trainer_module="malicious.trainer")
    with pytest.raises(runner.ExactTargetNativeStageError, match="pinned"):
        runner.ProductionStageRuntime(dtype="bfloat16")
