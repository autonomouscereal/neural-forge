from __future__ import annotations

import hashlib
import json
import math
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


def _continuation_contract_files(
    tmp_path: Path,
    monkeypatch,
    *,
    compiler_report: dict[str, object] | None = None,
    envelope_reason: str = "fit-gate-rejected",
) -> tuple[Path, str, str]:
    monkeypatch.setattr(runner, "TARGET_NATIVE_FACTOR_PARAMETERS", 14)
    path, _digest = _contract_files(tmp_path)
    names = tuple(
        sorted(
            {
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
        )
    )
    monkeypatch.setattr(
        runner,
        "_expected_continuation_factor_shapes",
        lambda *, lexical_rank: {
            name: (1,)
            for name in names
            if lexical_rank is None or not name.startswith("lexical.")
        },
    )
    tensors = {
        name: torch.tensor([float(index + 1)], dtype=torch.float32)
        for index, name in enumerate(names)
    }
    rows = [
        {
            "name": name,
            "shape": [1],
            "dtype": "torch.float32",
            "tensor_sha256": runner._factor_tensor_sha256(tensors[name]),
        }
        for name in names
    ]
    factor_state_sha = runner._canonical_sha256(rows)
    source_prereg_sha = "1" * 64
    source_evaluation_sha = "2" * 64
    factor_payload = {
        "schema": runner.TRAINER_FACTOR_PACK_SCHEMA,
        "factor_count": 14,
        "factors": rows,
        "factor_state_sha256": factor_state_sha,
        "parent_state_included": False,
        "native_child_state_included": False,
        "optimizer_state_included": False,
        "factor_tensors": tensors,
    }
    envelope = {
        "schema": runner.FACTOR_PACK_SCHEMA,
        "reason": envelope_reason,
        "preregistration_sha256": source_prereg_sha,
        "evaluation_contract_sha256": source_evaluation_sha,
        "qkvo_site_count": runner.QKVO_SITE_COUNT,
        "qkvo_rank": runner.QKVO_RANK,
        "swiglu_layer_count": runner.SWIGLU_LAYER_COUNT,
        "swiglu_rank": runner.SWIGLU_RANK,
        "factor_pack": factor_payload,
        "contains_model_checkpoint": False,
        "contains_parent_parameters": False,
        "contains_activations_or_hidden_states": False,
    }
    factor_path = tmp_path / "source_factor_pack.pt"
    torch.save(envelope, factor_path)
    factor_file_sha = hashlib.sha256(factor_path.read_bytes()).hexdigest()
    source_compiler_report = compiler_report or _Report().as_dict()
    compiled_state_sha = hashlib.sha256(b"compiled").hexdigest()
    trainer_evidence = {
        "factor_pack_manifest": {"factor_state_sha256": factor_state_sha}
    }
    source_report = {
        "status": "rejected",
        "reason": "fit-gate-rejected",
        "rollback_exact": True,
        "model_checkpoint_written": False,
        "standalone_compaction_passed": False,
        "preregistration_sha256": source_prereg_sha,
        "evaluation_contract_sha256": source_evaluation_sha,
        "factor_pack": {"path": str(factor_path), "sha256": factor_file_sha},
        "trainer_evidence": trainer_evidence,
        "trainer_evidence_sha256": runner._canonical_sha256(trainer_evidence),
        "factor_acquisition": {
            "selected_factor_state_sha256": factor_state_sha,
            "target_native_factors_acquired_behavior": True,
        },
        "compiler_report": source_compiler_report,
        "rollback_state_sha256": compiled_state_sha,
    }
    source_report_path = tmp_path / "source_run_report.json"
    source_report_sha = _write_json(source_report_path, source_report)
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    cache_root = tmp_path / "compiler-cache"
    cache_root.mkdir()
    preregistration["compiler_cache"] = {"root": str(cache_root)}
    preregistration["implementation"]["compiler_cache_sha256"] = runner._file_sha256(
        Path(runner.__file__).resolve().parents[1]
        / "src"
        / "neural_forge"
        / "hf_compiled_child_cache.py"
    )
    preregistration["continuation"] = {
        "factor_pack": {"path": str(factor_path), "sha256": factor_file_sha},
        "source_run_report": {
            "path": str(source_report_path),
            "sha256": source_report_sha,
        },
        "expected_factor_state_sha256": factor_state_sha,
        "expected_compiler_report_sha256": runner._canonical_sha256(
            source_compiler_report
        ),
        "expected_compiled_state_sha256": compiled_state_sha,
    }
    return path, _write_json(path, preregistration), factor_state_sha


class _Transaction:
    def __init__(self, model) -> None:
        self.model = model
        self.is_live = True
        self.rolled_back = False
        self.baked = False
        self.restored_prepublication = False

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

    def restore_prepublication_native(self):
        was_live = self.is_live
        self.is_live = False
        self.baked = False
        self.rolled_back = self.rolled_back or was_live
        self.restored_prepublication = True
        return {"restored": True}


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
    def __init__(
        self,
        result: _TrainingResult,
        *,
        post_bake_metrics: dict[str, object] | None = None,
    ) -> None:
        self.result = result
        self.parent = object()
        self.model = SimpleNamespace(state="compiled")
        self.transaction = _Transaction(self.model)
        self.save_calls = 0
        self.evaluate_calls: list[str] = []
        self.post_bake_metrics = post_bake_metrics or {
            name: _metrics() for name in ("fit", "candidate", "sealed")
        }

    def load_models(self, contract):
        del contract
        return self.parent, self.model, object()

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

    def evaluate_baked_behavior(
        self, model, parent, contract, split_name
    ):
        del contract
        assert model is self.model
        assert parent is self.parent
        assert self.transaction.baked is True
        self.evaluate_calls.append(split_name)
        return self.post_bake_metrics[split_name]

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


def test_continuation_factor_and_source_report_authenticate_before_models(
    tmp_path, monkeypatch
) -> None:
    path, digest, factor_state_sha = _continuation_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    assert contract.continuation is not None
    assert contract.continuation.factor_state_sha256 == factor_state_sha
    assert contract.continuation.factor_payload["factor_count"] == 14
    assert (
        contract.preflight_report()["continuation"]["authenticated_before_model_access"]
        is True
    )


def test_continuation_rejects_factor_report_provenance_disagreement(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _continuation_contract_files(
        tmp_path,
        monkeypatch,
        envelope_reason="candidate-gate-rejected",
    )
    with pytest.raises(
        runner.ExactTargetNativeStageError, match="provenance disagrees"
    ):
        runner.authenticate_preregistration(path, digest)


def test_continuation_compiler_mismatch_aborts_before_install(
    tmp_path, monkeypatch
) -> None:
    source_compiler = {**_Report().as_dict(), "source_marker": "different"}
    path, digest, _state = _continuation_contract_files(
        tmp_path,
        monkeypatch,
        compiler_report=source_compiler,
    )
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
    runtime.install_calls = 0
    runtime.finalize_calls = 0

    def forbidden_install(_compiled):
        runtime.install_calls += 1
        raise AssertionError("installation must remain unreachable")

    runtime.install_transaction = forbidden_install

    def forbidden_finalize(_compiled, _contract):
        runtime.finalize_calls += 1
        raise AssertionError("cache publication must remain unreachable")

    runtime.finalize_compiled_base = forbidden_finalize
    with pytest.raises(runner.ExactTargetNativeStageError, match="compiler report"):
        runner.execute_authenticated_stage(
            contract, tmp_path / "compiler-mismatch", runtime=runtime
    )
    assert runtime.finalize_calls == 0
    assert runtime.install_calls == 0


def test_production_runtime_passes_authenticated_initial_factor_payload(
    tmp_path, monkeypatch
) -> None:
    path, digest, factor_state_sha = _continuation_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    captured: dict[str, object] = {}

    class _Schedule:
        def __init__(self, **values) -> None:
            self.values = values

    class _Threshold:
        def __init__(self, **values) -> None:
            self.values = values

    class _Result:
        pass

    result = _Result()

    def trainer(**kwargs):
        captured.update(kwargs)
        return result

    module = SimpleNamespace(
        load_behavior_domain_suite=lambda *args, **kwargs: (args, kwargs),
        BehaviorDistillationSchedule=_Schedule,
        CandidateBehaviorThresholds=_Threshold,
        train_target_native_behavior_distillation=trainer,
        BehaviorDistillationResult=_Result,
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: module)
    runtime = runner.ProductionStageRuntime()
    assert runtime.train(object(), object(), contract) is result
    initial = captured["initial_factor_payload"]
    assert isinstance(initial, dict)
    assert initial["factor_state_sha256"] == factor_state_sha


def test_production_post_bake_evaluator_uses_authenticated_suite_and_schedule(
    tmp_path, monkeypatch
) -> None:
    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    parent = object()
    child = object()
    metrics = _metrics()
    captured: dict[str, object] = {}

    class _Schedule:
        def __init__(self, **values) -> None:
            captured["schedule"] = values
            self.evaluation_horizon = 13
            self.top_k = 7

    def loader(name, payload, *, artifact_sha256):
        captured["loader"] = (name, payload, artifact_sha256)
        return "authenticated-candidate-suite"

    def evaluator(parent_model, child_model, suite, **kwargs):
        captured["evaluator"] = (parent_model, child_model, suite, kwargs)
        return metrics

    module = SimpleNamespace(
        BehaviorDistillationSchedule=_Schedule,
        load_behavior_domain_suite=loader,
        evaluate_behavior_domain=evaluator,
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: module)

    runtime = runner.ProductionStageRuntime()
    result = runtime.evaluate_baked_behavior(
        child, parent, contract, "candidate"
    )

    assert result is metrics
    binding = contract.splits["candidate"]
    assert captured["loader"] == (
        "candidate",
        binding.payload,
        binding.sha256,
    )
    assert captured["evaluator"] == (
        parent,
        child,
        "authenticated-candidate-suite",
        {
            "horizon": 13,
            "top_k": 7,
            "split": "calibration-selection",
        },
    )


def test_lexical_continuation_accepts_authenticated_v1_base_factor_pack(
    tmp_path, monkeypatch
) -> None:
    path, _digest, factor_state_sha = _continuation_contract_files(
        tmp_path, monkeypatch
    )
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    target_config = Path(preregistration["blank_target"]["path"]) / "config.json"
    target_config.write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "vocab_size": runner.LEXICAL_VOCAB_SIZE,
                "hidden_size": runner.LEXICAL_HIDDEN_SIZE,
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )
    source_root = Path(runner.__file__).resolve().parents[1] / "src" / "neural_forge"
    preregistration["implementation"].update(
        {
            "lexical_adapter_sha256": runner._file_sha256(
                source_root / "hf_dense_lexical_residual_adapter.py"
            ),
            "progressive_bridge_sha256": runner._file_sha256(
                source_root / "hf_progressive_target_bridge.py"
            ),
            "qkvo_adapter_sha256": runner._file_sha256(
                source_root / "hf_dense_gqa_qkvo_residual_adapter.py"
            ),
            "swiglu_adapter_sha256": runner._file_sha256(
                source_root / "hf_dense_swiglu_projection_residual_adapter.py"
            ),
        }
    )
    preregistration["lexical_residual"] = {
        "rank": runner.LEXICAL_RANK,
        "vocab_size": runner.LEXICAL_VOCAB_SIZE,
        "hidden_size": runner.LEXICAL_HIDDEN_SIZE,
        "untied_required": True,
    }
    digest = _write_json(path, preregistration)

    contract = runner.authenticate_preregistration(path, digest)

    assert contract.lexical_residual is not None
    assert contract.lexical_residual.rank == 32
    assert contract.continuation is not None
    assert contract.continuation.factor_state_sha256 == factor_state_sha
    assert contract.continuation.trainer_factor_pack_schema == (
        runner.TRAINER_FACTOR_PACK_SCHEMA
    )
    assert contract.continuation.lexical_rank is None


def test_v2_factor_pack_authenticates_all_18_factors_and_lexical_rank(
    monkeypatch,
) -> None:
    names = tuple(
        sorted(
            {
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
        )
    )
    tensors = {
        name: torch.tensor([float(index + 1)], dtype=torch.float32)
        for index, name in enumerate(names)
    }
    rows = [
        {
            "name": name,
            "shape": [1],
            "dtype": "torch.float32",
            "tensor_sha256": runner._factor_tensor_sha256(tensors[name]),
        }
        for name in names
    ]
    state_sha = runner._canonical_sha256(
        {
            "schema": runner.TRAINER_FACTOR_PACK_V2_SCHEMA,
            "lexical_rank": runner.LEXICAL_RANK,
            "factors": rows,
        }
    )
    monkeypatch.setattr(
        runner,
        "_expected_continuation_factor_shapes",
        lambda *, lexical_rank: {
            name: (1,)
            for name in names
            if lexical_rank is not None or not name.startswith("lexical.")
        },
    )
    monkeypatch.setattr(runner, "LEXICAL_TARGET_NATIVE_FACTOR_PARAMETERS", len(names))
    payload = {
        "schema": runner.TRAINER_FACTOR_PACK_V2_SCHEMA,
        "factor_count": len(names),
        "factors": rows,
        "factor_state_sha256": state_sha,
        "parent_state_included": False,
        "native_child_state_included": False,
        "optimizer_state_included": False,
        "lexical_rank": runner.LEXICAL_RANK,
        "factor_tensors": tensors,
    }
    envelope = {
        "schema": runner.FACTOR_PACK_V2_SCHEMA,
        "reason": "fit-gate-rejected",
        "preregistration_sha256": "1" * 64,
        "evaluation_contract_sha256": "2" * 64,
        "qkvo_site_count": runner.QKVO_SITE_COUNT,
        "qkvo_rank": runner.QKVO_RANK,
        "swiglu_layer_count": runner.SWIGLU_LAYER_COUNT,
        "swiglu_rank": runner.SWIGLU_RANK,
        "lexical_rank": runner.LEXICAL_RANK,
        "factor_pack": payload,
        "contains_model_checkpoint": False,
        "contains_parent_parameters": False,
        "contains_activations_or_hidden_states": False,
    }

    authenticated = runner._authenticate_continuation_factor_payload(
        envelope, lexical_rank=runner.LEXICAL_RANK
    )

    assert authenticated[1] == state_sha
    assert authenticated[3] == runner.FACTOR_PACK_V2_SCHEMA
    assert authenticated[4] == runner.TRAINER_FACTOR_PACK_V2_SCHEMA
    assert authenticated[5] == runner.LEXICAL_RANK
    with pytest.raises(runner.ExactTargetNativeStageError, match="lexical rank"):
        runner._authenticate_continuation_factor_payload(
            envelope, lexical_rank=16
        )


def test_production_continuation_factor_shapes_match_pinned_parameter_totals() -> None:
    base_shapes = runner._expected_continuation_factor_shapes(lexical_rank=None)
    lexical_shapes = runner._expected_continuation_factor_shapes(
        lexical_rank=runner.LEXICAL_RANK
    )

    assert len(base_shapes) == runner.TARGET_NATIVE_FACTOR_COUNT
    assert sum(math.prod(shape) for shape in base_shapes.values()) == (
        runner.TARGET_NATIVE_FACTOR_PARAMETERS
    )
    assert len(lexical_shapes) == runner.LEXICAL_TARGET_NATIVE_FACTOR_COUNT
    assert sum(math.prod(shape) for shape in lexical_shapes.values()) == (
        runner.LEXICAL_TARGET_NATIVE_FACTOR_PARAMETERS
    )
    assert sum(
        math.prod(shape)
        for name, shape in lexical_shapes.items()
        if name.startswith("lexical.")
    ) == runner.LEXICAL_FACTOR_PARAMETERS


def test_continuation_rejects_same_numel_wrong_factor_reshape(
    tmp_path, monkeypatch
) -> None:
    path, _digest, _state = _continuation_contract_files(tmp_path, monkeypatch)
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    factor_path = Path(preregistration["continuation"]["factor_pack"]["path"])
    envelope = torch.load(factor_path, map_location="cpu", weights_only=True)
    payload = envelope["factor_pack"]
    name = "qkvo.q_left"
    reshaped = payload["factor_tensors"][name].reshape(())
    payload["factor_tensors"][name] = reshaped
    for row in payload["factors"]:
        if row["name"] == name:
            row["shape"] = []
            row["tensor_sha256"] = runner._factor_tensor_sha256(reshaped)
            break
    payload["factor_state_sha256"] = runner._canonical_sha256(payload["factors"])

    with pytest.raises(runner.ExactTargetNativeStageError, match="exact stage-1 shape"):
        runner._authenticate_continuation_factor_payload(
            envelope, lexical_rank=None
        )


def test_required_compiler_cache_miss_aborts_before_model_access(
    tmp_path, monkeypatch
) -> None:
    path, _digest = _contract_files(tmp_path)
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    cache_root = tmp_path / "required-cache"
    cache_root.mkdir()
    preregistration["implementation"]["compiler_cache_sha256"] = runner._file_sha256(
        Path(runner.__file__).resolve().parents[1]
        / "src"
        / "neural_forge"
        / "hf_compiled_child_cache.py"
    )
    preregistration["compiler_cache"] = {
        "root": str(cache_root),
        "require_hit": True,
        "expected_compiler_identity_sha256": "a" * 64,
        "expected_receipt_sha256": "b" * 64,
        "expected_compiler_report_sha256": "c" * 64,
        "expected_state_dict_sha256": "d" * 64,
        "expected_artifact_tree_sha256": "e" * 64,
    }
    contract = runner.authenticate_preregistration(
        path, _write_json(path, preregistration)
    )
    runtime = runner.ProductionStageRuntime()
    monkeypatch.setattr(
        runtime,
        "_compiler_identity_for_contract",
        lambda _contract: SimpleNamespace(compiler_identity_sha256="a" * 64),
    )

    with pytest.raises(runner.ExactTargetNativeStageError, match="requires an exact hit"):
        runtime.prepare_compiler_cache(contract)


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
    assert runtime.evaluate_calls == []
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
    assert runtime.evaluate_calls == ["fit", "candidate", "sealed"]
    assert report["model_save_count"] == 1
    assert report["selection_used_internal_numeric_matching"] is False
    assert report["fresh_reload_certificate"]["fresh_reload_performed"] is True
    assert all(
        decision["passed"]
        for decision in report["pre_bake_behavior_decisions"].values()
    )
    assert all(
        decision["passed"]
        for decision in report["post_bake_behavior_decisions"].values()
    )
    assert (output / "accepted_child" / "model.safetensors").is_file()
    assert not (output / "rejected_factor_pack.pt").exists()


def test_post_bake_candidate_failure_retains_pack_and_restores_exact_base(
    tmp_path,
) -> None:
    class _MutatingTransaction(_Transaction):
        def bake_for_publication(self, receipt):
            payload = super().bake_for_publication(receipt)
            self.model.state = "baked"
            return payload

        def restore_prepublication_native(self):
            self.model.state = "compiled"
            return super().restore_prepublication_native()

    class _MutatingRuntime(_Runtime):
        def __init__(self, result, **kwargs):
            super().__init__(result, **kwargs)
            self.transaction = _MutatingTransaction(self.model)

    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _MutatingRuntime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=_metrics(),
            sealed_metrics=_metrics(),
            candidate_gate_passed=True,
            sealed_evaluation_unlocked=True,
        ),
        post_bake_metrics={
            "fit": _metrics(),
            "candidate": _metrics(accuracy=0.25),
            "sealed": _metrics(),
        },
    )
    output = tmp_path / "post-bake-candidate-reject"

    report = runner.execute_authenticated_stage(contract, output, runtime=runtime)

    assert report["status"] == "rejected"
    assert report["reason"] == "post-bake-candidate-gate-rejected"
    assert runtime.evaluate_calls == ["fit", "candidate"]
    assert runtime.transaction.restored_prepublication is True
    assert runtime.model.state == "compiled"
    assert report["rollback_exact"] is True
    assert report["rollback_state_sha256"] == hashlib.sha256(b"compiled").hexdigest()
    assert report["prepublication_native_restored"] is True
    assert report["model_checkpoint_written"] is False
    assert runtime.save_calls == 0
    assert report["pre_bake_behavior_decisions"]["candidate"]["passed"] is True
    assert report["post_bake_behavior_decisions"]["candidate"]["passed"] is False
    assert report["post_bake_behavior_decisions"]["sealed"] is None
    assert {artifact.name for artifact in output.iterdir()} == {
        "rejected_factor_pack.pt",
        "run_report.json",
    }
    envelope = torch.load(output / "rejected_factor_pack.pt", weights_only=True)
    assert envelope["reason"] == "post-bake-candidate-gate-rejected"


def test_post_bake_evaluator_error_restores_exact_base_and_fails(tmp_path) -> None:
    class _MutatingTransaction(_Transaction):
        def bake_for_publication(self, receipt):
            payload = super().bake_for_publication(receipt)
            self.model.state = "baked"
            return payload

        def restore_prepublication_native(self):
            self.model.state = "compiled"
            return super().restore_prepublication_native()

    class _FailingEvaluatorRuntime(_Runtime):
        def __init__(self, result):
            super().__init__(result)
            self.transaction = _MutatingTransaction(self.model)

        def evaluate_baked_behavior(
            self, model, parent, contract, split_name
        ):
            del model, parent, contract, split_name
            raise RuntimeError("injected post-bake evaluator failure")

    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _FailingEvaluatorRuntime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=_metrics(),
            sealed_metrics=_metrics(),
            candidate_gate_passed=True,
            sealed_evaluation_unlocked=True,
        )
    )
    output = tmp_path / "post-bake-evaluator-error"

    with pytest.raises(RuntimeError, match="post-bake evaluator"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)

    assert runtime.model.state == "compiled"
    assert runtime.transaction.restored_prepublication is True
    assert runtime.save_calls == 0
    assert not (output / "accepted_child").exists()
    assert not (output / "rejected_factor_pack.pt").exists()
    failure = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["last_completed_phase"] == "native-factor-bake"
    assert failure["prepublication_native_restored"] is True
    assert failure["rollback_error"] is None
    assert failure["model_checkpoint_promoted"] is False


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


def test_training_failure_report_retains_completed_gpu_and_compiler_phases(
    tmp_path,
) -> None:
    class _FailingTrainingRuntime(_Runtime):
        def train(self, transaction, parent, contract):
            del transaction, parent, contract
            raise RuntimeError("injected training failure")

    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _FailingTrainingRuntime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=_metrics(),
            sealed_metrics=_metrics(),
            candidate_gate_passed=True,
            sealed_evaluation_unlocked=True,
        )
    )
    output = tmp_path / "training-failure"
    with pytest.raises(RuntimeError, match="injected training failure"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    report = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["models_loaded"] is True
    assert report["gpu_work_launched"] is True
    assert report["last_completed_phase"] == "behavior-training-started"
    assert report["compiler_report"]["bootstrap_accepted"] is True
    assert report["installation"]["zero_step_identity"] is True
    assert report["rollback_error"] is None
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
    assert runtime.transaction.restored_prepublication is True


def test_post_bake_save_failure_restores_exact_compiled_child_bytes(tmp_path) -> None:
    class _MutatingTransaction(_Transaction):
        def bake_for_publication(self, receipt):
            payload = super().bake_for_publication(receipt)
            self.model.state = "baked"
            return payload

        def restore_prepublication_native(self):
            self.model.state = "compiled"
            return super().restore_prepublication_native()

    class _FailingSaveRuntime(_Runtime):
        def __init__(self, result):
            super().__init__(result)
            self.transaction = _MutatingTransaction(self.model)

        def save_reload_certify(self, model, tokenizer, output_root):
            del model, tokenizer, output_root
            raise OSError("injected save failure after bake")

    path, digest = _contract_files(tmp_path)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _FailingSaveRuntime(
        _TrainingResult(
            fit_metrics=_metrics(),
            candidate_metrics=_metrics(),
            sealed_metrics=_metrics(),
            candidate_gate_passed=True,
            sealed_evaluation_unlocked=True,
        )
    )
    output = tmp_path / "post-bake-save-failure"

    with pytest.raises(OSError, match="after bake"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)

    assert runtime.model.state == "compiled"
    assert runtime.transaction.restored_prepublication is True
    failure = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    assert failure["prepublication_native_restored"] is True
    assert failure["rollback_error"] is None


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
