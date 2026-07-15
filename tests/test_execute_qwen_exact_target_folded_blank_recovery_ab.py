from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import execute_qwen_exact_target_folded_blank_recovery_ab as executor  # noqa: E402
from neural_forge import hf_recovery  # noqa: E402


def _digest(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return _file_sha(path)


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _behavior(step: int, *, correct: int, loss: float | None) -> dict[str, Any]:
    total = 2
    generation_correct = 1
    generation_total = 2
    behavior_breakdown = {
        "reasoning": {
            "correct": correct,
            "total": total,
            "pass_rate": correct / total,
            "failed_ids": [f"b-fail-{index}" for index in range(total - correct)],
            "passed_ids": [f"b-pass-{index}" for index in range(correct)],
        }
    }
    generation_breakdown = {
        "generation": {
            "correct": generation_correct,
            "total": generation_total,
            "pass_rate": generation_correct / generation_total,
            "failed_ids": ["g-fail"],
            "passed_ids": ["g-pass"],
        }
    }
    return {
        "step": step,
        "loss": loss,
        "correct": correct,
        "total": total,
        "pass_rate": correct / total,
        "generation_health_correct": generation_correct,
        "generation_health_total": generation_total,
        "generation_health_rate": generation_correct / generation_total,
        "behavior_category_breakdown": behavior_breakdown,
        "generation_category_breakdown": generation_breakdown,
        "behavior_category_summary": hf_recovery._category_breakdown_summary(
            behavior_breakdown
        ),
        "generation_category_summary": hf_recovery._category_breakdown_summary(
            generation_breakdown
        ),
        "passed_ids": [f"b-pass-{index}" for index in range(correct)],
        "generation_passed_prompts": ["generation prompt"],
    }


def _event(index: int, behavior: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "neural-forge.recovery-behavior-checkpoint-observer.v1",
        "operation": "recover_cross_tokenizer_causal_lm",
        "checkpoint_index": index,
        "step": behavior["step"],
        "behavior": behavior,
    }


def _stepwise_event(step: int) -> dict[str, Any]:
    return {
        "step": step,
        "rows": [
            {
                "prompt_input_ids": [10 + step],
                "rollout_input_ids": [10 + step, 20 + step],
            }
        ],
    }


def _fixture_contract(tmp_path: Path) -> tuple[Any, dict[str, Any], Path]:
    source_state_body = {"state_dict_sha256": _digest("source-state")}
    source_state = {
        **source_state_body,
        "sha256": executor._canonical_sha256(source_state_body),
    }
    source_fingerprint_body = {
        "class": "fake.SourceModel",
        "model_state_fingerprint": source_state,
        "config_sha256": _digest("source-config"),
        "generation_config_sha256": _digest("source-generation-config"),
    }
    source_fingerprint = {
        **source_fingerprint_body,
        "sha256": executor._canonical_sha256(source_fingerprint_body),
    }
    source_tokenizer_body = {"class": "fake.SourceTokenizer"}
    source_tokenizer_fingerprint = {
        **source_tokenizer_body,
        "sha256": executor._canonical_sha256(source_tokenizer_body),
    }
    candidate_tokenizer_body = {"class": "fake.CandidateTokenizer"}
    candidate_tokenizer_fingerprint = {
        **candidate_tokenizer_body,
        "sha256": executor._canonical_sha256(candidate_tokenizer_body),
    }
    teacher_parameters = {"max_length": 8}
    teacher_replay_body = {
        "schema": "fake-teacher-token-replay",
        "parameters": teacher_parameters,
        "tokenizer_fingerprints": {
            "source": source_tokenizer_fingerprint,
            "candidate": candidate_tokenizer_fingerprint,
        },
        "source_model_fingerprint": source_fingerprint,
        "calibration_texts": ["calibration"],
        "precomputed_teacher_generated_texts": ["teacher continuation"],
        "on_policy_replay_bank": [["rollout one"], ["rollout two"]],
        "stepwise_replay_bank": [_stepwise_event(1), _stepwise_event(2)],
    }
    teacher_replay = {
        **teacher_replay_body,
        "payload_sha256": executor._canonical_sha256(teacher_replay_body),
    }
    replay_body = {
        "schema": "fake-replay",
        "status": "frozen",
        "teacher_token_replay": teacher_replay,
        "behavior_eval_suite": {"split": "candidate", "probes": ["p"]},
        "frozen_before_preregistration": True,
        "contains_sealed_data": False,
        "publication_authorized": False,
    }
    replay_payload = {
        **replay_body,
        "replay_payload_sha256": executor._canonical_sha256(replay_body),
    }
    replay_path = tmp_path / "replay.json"
    replay_sha = _write_json(replay_path, replay_payload)
    frozen_on_policy = hf_recovery._freeze_on_policy_replay_bank(
        teacher_replay["on_policy_replay_bank"]
    )
    frozen_stepwise = hf_recovery._freeze_stepwise_replay_bank(
        teacher_replay["stepwise_replay_bank"]
    )
    stepwise_rows = [
        {
            "event_index": index,
            "optimizer_step": event.step,
            "row_count": len(event.rows),
            "prompt_token_count": sum(len(row.prompt_input_ids) for row in event.rows),
            "rollout_token_count": sum(
                len(row.rollout_input_ids) for row in event.rows
            ),
            "continuation_token_count": sum(
                len(row.rollout_input_ids) - len(row.prompt_input_ids)
                for row in event.rows
            ),
            "event_sha256": hf_recovery._stepwise_replay_event_sha256(event),
        }
        for index, event in enumerate(frozen_stepwise)
    ]
    on_policy_rows = [
        {
            "event_index": index,
            "text_count": len(batch),
            "ordered_text_sha256": hf_recovery._ordered_recovery_texts_sha256(batch),
        }
        for index, batch in enumerate(frozen_on_policy)
    ]
    recovery_kwargs = {
        "steps": 2,
        "batch_size": 1,
        "max_length": 8,
        "teacher_generate_lm_weight": 0.1,
        "stepwise_donor_kd_weight": 0.2,
        "stepwise_donor_max_new_tokens": 1,
        "on_policy_anchor_kd_weight": 0.2,
        "on_policy_interval": 2,
        "on_policy_max_new_tokens": 1,
        "behavior_eval_checkpoint_steps": [0, 1, 2],
        "local_files_only": True,
        "trust_remote_code": False,
        "adamw_foreach": False,
        "save_best_loss": False,
        "save_best_behavior": False,
    }
    blank_cache = tmp_path / "blank-cache"
    blank_cache.mkdir()
    prereg = tmp_path / "preregistration.json"
    prereg.write_text("{}", encoding="utf-8")
    raw = {
        "semantic_contract_sha256": _digest("semantic"),
        "shared_execution_contract_sha256": _digest("shared"),
        "compiled_blank": {
            "cache_path": str(blank_cache),
            "compiler_identity_sha256": _digest("compiler-identity"),
            "state_dict_sha256": _digest("blank-state"),
            "receipt_sha256": _digest("blank-receipt"),
            "config_payload_sha256": _digest("config"),
            "state_key_sha256": _digest("keys"),
            "parameter_topology_sha256": _digest("parameters"),
            "buffer_topology_sha256": _digest("buffers"),
        },
        "v4_candidate": {
            "factor_pack_schema": "fake-v4-pack",
            "factor_source_binding_sha256": _digest("source-binding"),
            "winner": {
                "portable_factor_state_sha256": _digest("portable-v4"),
                "runtime_factor_state_sha256": _digest("runtime-v4"),
                "factor_pack_sha256": _digest("factor-pack"),
            },
        },
        "replay_bank": {
            "path": str(replay_path),
            "file_sha256": replay_sha,
            "payload_sha256": replay_payload["replay_payload_sha256"],
            "teacher_token_replay_schema": teacher_replay["schema"],
            "teacher_token_replay_payload_sha256": teacher_replay["payload_sha256"],
            "teacher_token_replay_parameters_sha256": (
                executor._canonical_sha256(teacher_parameters)
            ),
            "source_tokenizer_fingerprint_sha256": (
                source_tokenizer_fingerprint["sha256"]
            ),
            "candidate_tokenizer_fingerprint_sha256": (
                candidate_tokenizer_fingerprint["sha256"]
            ),
            "source_model_fingerprint_sha256": source_fingerprint["sha256"],
            "source_model_state_dict_sha256": source_state["state_dict_sha256"],
            "calibration_text_count": 1,
            "calibration_texts_sha256": (
                hf_recovery._ordered_recovery_texts_sha256(["calibration"])
            ),
            "teacher_generated_text_count": 1,
            "teacher_generated_texts_sha256": (
                hf_recovery._ordered_recovery_texts_sha256(["teacher continuation"])
            ),
            "on_policy_event_count": 2,
            "on_policy_events": on_policy_rows,
            "on_policy_replay_bank_sha256": (
                hf_recovery._on_policy_replay_bank_sha256(frozen_on_policy)
            ),
            "stepwise_event_count": 2,
            "stepwise_event_steps": [1, 2],
            "stepwise_events": stepwise_rows,
            "stepwise_replay_bank_schema": hf_recovery._STEPWISE_REPLAY_SCHEMA,
            "stepwise_replay_bank_sha256": (
                hf_recovery._stepwise_replay_bank_sha256(frozen_stepwise)
            ),
            "behavior_eval_suite_sha256": executor._canonical_sha256(
                replay_payload["behavior_eval_suite"]
            ),
        },
        "recovery_plan": {
            "recovery_kwargs": recovery_kwargs,
            "seeds": {"python": 1, "torch": 2, "cuda": 3, "data_order": 4},
            "checkpoint_steps": [0, 1, 2],
            "on_policy_event_steps": [1, 2],
            "stepwise_event_steps": [1, 2],
            "stepwise_replay_bank_required": True,
        },
        "implementation": {
            "recovery_implementation_sha256": _digest("recovery-code"),
            "behavior_evaluator_implementation_sha256": _digest(
                "behavior-evaluator-code"
            ),
        },
    }
    contract = SimpleNamespace(
        path=prereg,
        file_sha256=_file_sha(prereg),
        raw=raw,
    )
    return contract, replay_payload, blank_cache


def _replay_report(
    contract: Any,
    invocation: dict[str, Any],
    behaviors: list[dict[str, Any]],
    *,
    tamper_consumption: bool = False,
) -> dict[str, Any]:
    replay = contract.raw["replay_bank"]
    stepwise_preflight_events = [
        {
            "event_index": row["event_index"],
            "optimizer_step": row["optimizer_step"],
            "scheduled_batch_index": 0,
            "row_count": row["row_count"],
            "raw_text_count": 1,
            "raw_texts_sha256": _digest(f"raw-{row['event_index']}"),
            "prompt_texts_sha256": _digest(f"prompt-{row['event_index']}"),
            "prompt_token_counts": [1],
            "rollout_token_counts": [2],
            "continuation_token_counts": [1],
            "kd_position_count": row["continuation_token_count"],
            "event_sha256": row["event_sha256"],
        }
        for row in replay["stepwise_events"]
    ]
    preflight_body = {
        "schema": hf_recovery._STEPWISE_REPLAY_PREFLIGHT_SCHEMA,
        "status": "validated",
        "event_count": 2,
        "replay_bank_sha256": replay["stepwise_replay_bank_sha256"],
        "candidate_vocab_limit": 100,
        "prompt_max_length": 7,
        "stepwise_donor_max_new_tokens": 1,
        "max_length": 8,
        "events": stepwise_preflight_events,
    }
    preflight = {
        **preflight_body,
        "sha256": hf_recovery._stepwise_replay_preflight_sha256(preflight_body),
    }
    consumption = [
        {
            "event_index": row["event_index"],
            "step": row["optimizer_step"],
            "row_count": row["row_count"],
            "prompt_token_counts": [1],
            "continuation_token_counts": [1],
            "event_sha256": row["event_sha256"],
            "kd_position_count": row["continuation_token_count"],
        }
        for row in replay["stepwise_events"]
    ]
    if tamper_consumption:
        consumption[0]["event_sha256"] = _digest("tampered")
    on_policy_consumption = [
        {
            "event_index": row["event_index"],
            "step": contract.raw["recovery_plan"]["on_policy_event_steps"][index],
            "text_count": row["text_count"],
            "ordered_text_sha256": row["ordered_text_sha256"],
        }
        for index, row in enumerate(replay["on_policy_events"])
    ]
    report = {
        **{
            name: value
            for name, value in contract.raw["recovery_plan"]["recovery_kwargs"].items()
            if name != "behavior_eval_checkpoint_steps"
        },
        "operation": "recover_cross_tokenizer_causal_lm",
        "source_model": invocation["source_model"],
        "candidate_model": invocation["candidate_model"],
        "output_dir": invocation["output_dir"],
        "steps": 2,
        "local_files_only": True,
        "adamw_foreach": False,
        "adamw_foreach_explicit": True,
        "save_best_loss": False,
        "save_best_behavior": False,
        "param_count": executor.recovery_ab.EXACT_UNIQUE_PARAMETERS,
        "calibration_text_count": 1,
        "teacher_generated_text_source": "precomputed",
        "teacher_generated_text_count": 1,
        "teacher_generated_ordered_text_sha256": replay[
            "teacher_generated_texts_sha256"
        ],
        "behavior_eval_checkpoint_steps": [0, 1, 2],
        "behavior_checkpoint_observer_calls": 3,
        "behavior_checkpoint_observer_read_only_payload": True,
        "behavior_history": behaviors,
        "stepwise_dynamic_candidate_generation": False,
        "stepwise_replay_bank_supplied": True,
        "stepwise_replay_bank_schema": replay["stepwise_replay_bank_schema"],
        "stepwise_replay_bank_event_count": 2,
        "stepwise_replay_bank_sha256": replay["stepwise_replay_bank_sha256"],
        "stepwise_scheduled_steps": [1, 2],
        "stepwise_replay_preflight": preflight,
        "stepwise_replay_consumption": consumption,
        "on_policy_dynamic_candidate_generation": False,
        "on_policy_replay_bank_supplied": True,
        "on_policy_replay_bank_event_count": 2,
        "on_policy_replay_bank_sha256": replay["on_policy_replay_bank_sha256"],
        "on_policy_scheduled_steps": [1, 2],
        "on_policy_replay_consumption": on_policy_consumption,
        "losses": [1.0, 0.5],
    }
    return report


def _fake_dependencies(
    tmp_path: Path,
    contract: Any,
    *,
    mismatch_training_zero: bool = False,
    tamper_consumption: bool = False,
    tamper_source_fingerprint: bool = False,
    tamper_candidate_tokenizer: bool = False,
) -> tuple[executor.ExecutorDependencies, list[str]]:
    calls: list[str] = []
    folded_initial = _digest("folded-initial")

    def authenticate(path, digest):
        calls.append("authenticate")
        assert Path(path).resolve() == contract.path.resolve()
        assert digest == contract.file_sha256
        return contract

    def authenticate_teacher_replay(payload):
        calls.append("authenticate-teacher-replay")
        return payload

    def materialize(_contract, output, *, folded_artifact_hooks, runtime):
        del folded_artifact_hooks, runtime
        calls.append("materialize")
        root = output / executor.FOLDED_INIT_DIRECTORY_NAME
        child = root / executor.child_artifact.ARTIFACT_DIRECTORY_NAME
        child.mkdir(parents=True)
        (child / "config.json").write_text("{}", encoding="utf-8")
        receipts = {
            "folded": {"initial_state_sha256": folded_initial},
            "blank": {
                "initial_state_sha256": _contract.raw["compiled_blank"][
                    "state_dict_sha256"
                ]
            },
        }
        return executor.MaterializationProduct(
            receipts=receipts,
            folded_publication_root=root,
            folded_candidate_path=child,
            blank_candidate_path=Path(_contract.raw["compiled_blank"]["cache_path"]),
            source_model_path=tmp_path / "source-model",
            trace={"compiled_blank_bytes_restored_exactly": True},
        )

    def authenticate_materializations(_contract, receipts, *, folded_artifact_hooks):
        del folded_artifact_hooks
        calls.append("authenticate-materializations")
        return {
            "status": "materialization-authenticated",
            "receipts": receipts,
            "authenticated_materialization_reports": {
                "folded": {
                    "publication_root": str(
                        tmp_path / "output" / executor.FOLDED_INIT_DIRECTORY_NAME
                    )
                },
                "blank": None,
            },
            "optimizer_steps": 0,
            "folded_artifact_independent_fresh_reload_verified": True,
            "sealed_data_accessed": False,
            "publication_authorized": False,
            "accepted_artifact": False,
        }

    def authenticate_source_model(
        _contract,
        *,
        source_model_path,
        expected_fingerprint,
        expected_source_tokenizer_fingerprint,
        expected_candidate_tokenizer_fingerprint,
        candidate_model_paths,
    ):
        calls.append("authenticate-source-model")
        assert _contract is contract
        assert set(candidate_model_paths) == {"folded", "blank"}
        parent_tree = _digest("parent-tree")
        actual_fingerprint = dict(expected_fingerprint)
        if tamper_source_fingerprint:
            actual_fingerprint["sha256"] = _digest("wrong-source-model")
        candidate_fingerprints = {
            arm_id: expected_candidate_tokenizer_fingerprint
            for arm_id in ("folded", "blank")
        }
        if tamper_candidate_tokenizer:
            candidate_fingerprints["blank"] = {
                **expected_candidate_tokenizer_fingerprint,
                "sha256": _digest("wrong-candidate-tokenizer"),
            }
        return {
            "status": "source-model-authenticated",
            "source_model_path": str(source_model_path),
            "source_model_fingerprint": actual_fingerprint,
            "source_model_fingerprint_sha256": actual_fingerprint["sha256"],
            "source_tokenizer_fingerprint": (expected_source_tokenizer_fingerprint),
            "source_tokenizer_fingerprint_sha256": (
                expected_source_tokenizer_fingerprint["sha256"]
            ),
            "candidate_tokenizer_fingerprints": candidate_fingerprints,
            "candidate_tokenizer_fingerprint_sha256": (
                expected_candidate_tokenizer_fingerprint["sha256"]
            ),
            "source_artifact_tree_sha256": parent_tree,
            "compiler_parent_artifact_tree_sha256": parent_tree,
            "compiled_blank_compiler_identity_sha256": _contract.raw["compiled_blank"][
                "compiler_identity_sha256"
            ],
            "fresh_model_load_verified": True,
            "source_fingerprint_exactly_matches_teacher_token_replay": True,
            "source_tokenizer_exactly_matches_teacher_token_replay": True,
            "both_candidate_tokenizers_exactly_match_teacher_token_replay": True,
            "source_tree_exactly_matches_compiler_parent_lineage": True,
            "artifact_tree_unchanged_by_authentication": True,
            "candidate_artifact_trees_unchanged_by_authentication": True,
            "local_files_only": True,
            "trust_remote_code": False,
        }

    zero_by_arm = {
        "folded": _event(0, _behavior(0, correct=1, loss=None)),
        "blank": _event(0, _behavior(0, correct=0, loss=None)),
    }

    def zero_evaluator(
        _contract,
        *,
        arm_id,
        candidate_path,
        source_model_path,
        replay_payload,
        work_root,
        folded_artifact_hooks,
        dependencies,
    ):
        zero_child = work_root / f"{arm_id}_zero_child"
        zero_child.mkdir()
        (zero_child / "model.safetensors").write_text(
            arm_id,
            encoding="utf-8",
        )
        (work_root / f"{arm_id}_zero_report.json").write_text(
            "{}",
            encoding="utf-8",
        )
        del (
            candidate_path,
            source_model_path,
            replay_payload,
            folded_artifact_hooks,
            dependencies,
        )
        calls.append(f"zero:{arm_id}")
        event = zero_by_arm[arm_id]
        initial = (
            folded_initial
            if arm_id == "folded"
            else _contract.raw["compiled_blank"]["state_dict_sha256"]
        )
        return executor.ZeroStepEvaluation(
            event=MappingProxyType(event),
            event_sha256=executor._canonical_sha256(event),
            behavior=MappingProxyType(event["behavior"]),
            report={},
            report_file_sha256=_digest(f"zero-report-{arm_id}"),
            clean_artifact={"state_fingerprint": {"state_dict_sha256": initial}},
        )

    zero_authenticated = {"value": False}

    def authenticate_zero(
        _contract, materialization, receipts, *, folded_artifact_hooks
    ):
        del materialization, folded_artifact_hooks
        calls.append("authenticate-zero")
        assert set(receipts) == {"folded", "blank"}
        zero_authenticated["value"] = True
        return {
            "status": "zero-step-authenticated",
            "training_launched": False,
            "optimizer_steps": 0,
            "sealed_data_accessed": False,
            "publication_authorized": False,
            "accepted_artifact": False,
        }

    def recover(**invocation):
        arm_id = (
            "folded"
            if executor.FOLDED_INIT_DIRECTORY_NAME in invocation["candidate_model"]
            else "blank"
        )
        calls.append(f"train-start:{arm_id}")
        assert zero_authenticated["value"] is True
        zero_roots = list((tmp_path / "output").glob(".zero-step-*"))
        assert len(zero_roots) == 1
        assert not any(zero_roots[0].iterdir())
        correct0 = 1 if arm_id == "folded" else 0
        if mismatch_training_zero and arm_id == "folded":
            correct0 = 2
        behaviors = [
            _behavior(0, correct=correct0, loss=None),
            _behavior(1, correct=1, loss=1.0),
            _behavior(2, correct=2 if arm_id == "folded" else 1, loss=0.5),
        ]
        observer = invocation["behavior_checkpoint_observer"]
        for index, behavior in enumerate(behaviors):
            observer(_event(index, behavior))
            if behavior["step"] > 0:
                calls.append(f"optimizer:{arm_id}:{behavior['step']}")
        output = Path(invocation["output_dir"])
        output.mkdir()
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (output / name).write_text(name, encoding="utf-8")
        report = _replay_report(
            contract,
            invocation,
            behaviors,
            tamper_consumption=tamper_consumption,
        )
        _write_json(output / executor.RECOVERY_REPORT_FILENAME, report)
        return report

    def clean_artifact(path, *, folded_artifact_hooks, expected_state_sha256):
        del folded_artifact_hooks, expected_state_sha256
        calls.append(f"clean:{path.name}")
        assert not (path / executor.RECOVERY_REPORT_FILENAME).exists()
        assert {item.name for item in path.iterdir()} == {
            "config.json",
            "tokenizer.json",
            "model.safetensors",
        }
        return {
            "path": str(path),
            "state_fingerprint": {"state_dict_sha256": _digest(f"final-{path.name}")},
            "artifact_tree": {"tree_sha256": _digest(f"tree-{path.name}")},
            "report_detached_before_reload": True,
        }

    def atomic_json(path, payload):
        _write_json(path, payload)

    deps = executor.ExecutorDependencies(
        authenticate_contract=authenticate,
        authenticate_teacher_replay=authenticate_teacher_replay,
        materialize_initializations=materialize,
        authenticate_materializations=authenticate_materializations,
        authenticate_source_model=authenticate_source_model,
        authenticate_zero_steps=authenticate_zero,
        zero_step_evaluator=zero_evaluator,
        recover=recover,
        reset_rng=lambda seeds: calls.append(
            "reset:" + executor._canonical_sha256(seeds)
        ),
        authenticate_clean_artifact=clean_artifact,
        load_json=lambda path: json.loads(path.read_text(encoding="utf-8")),
        file_sha256=_file_sha,
        atomic_json=atomic_json,
        release_memory=lambda: calls.append("release"),
    )
    return deps, calls


def test_experimental_contract_has_exact_runner_fields(tmp_path: Path) -> None:
    contract, _replay, _blank = _fixture_contract(tmp_path)
    payload = executor.build_experimental_recovery_contract(
        contract,
        native_child_state_sha256=_digest("native"),
        live_factor_state_sha256=_digest("live"),
        restored_runtime_factor_state_sha256=_digest("runtime-v4"),
    )
    assert set(payload) == {
        "schema",
        "recovery_ab_preregistration_sha256",
        "recovery_ab_semantic_contract_sha256",
        "shared_execution_contract_sha256",
        "factor_state_sha256",
        "factor_payload_schema",
        "native_child_state_sha256",
        "blank_control_state_sha256",
        "compiled_blank_cache_state_dict_sha256",
        "restored_runtime_factor_state_sha256",
        "restored_portable_factor_state_sha256",
        "bridge_native_child_state_sha256",
        "bridge_live_factor_state_sha256",
        "exact_unique_parameters",
        "publication_authorized",
        "sealed_data_accessed",
    }
    assert payload["blank_control_state_sha256"] == _digest("native")
    assert payload["publication_authorized"] is False


def test_frozen_observer_validates_exact_behavior_and_step_zero() -> None:
    zero = _event(0, _behavior(0, correct=1, loss=None))
    observer = executor.FrozenBehaviorObserver(
        expected_steps=[0, 2],
        authenticated_zero_event=zero,
    )
    observer(zero)
    observer(_event(1, _behavior(2, correct=2, loss=0.5)))
    records = observer.event_records()
    assert [row["event"]["step"] for row in records] == [0, 2]
    assert all(len(row["event_sha256"]) == 64 for row in records)

    malformed = _event(0, _behavior(0, correct=1, loss=None))
    malformed["behavior"]["pass_rate"] = 0.9
    with pytest.raises(
        executor.FoldedBlankRecoveryExecutionError, match="inconsistent"
    ):
        executor.FrozenBehaviorObserver(expected_steps=[0])(malformed)

    no_generation = _behavior(0, correct=1, loss=None)
    no_generation.update(
        {
            "generation_health_correct": 0,
            "generation_health_total": 0,
            "generation_health_rate": 0.0,
            "generation_category_breakdown": {},
            "generation_category_summary": hf_recovery._category_breakdown_summary({}),
            "generation_passed_prompts": [],
        }
    )
    executor.FrozenBehaviorObserver(expected_steps=[0])(_event(0, no_generation))


def test_full_injected_execution_authenticates_zero_before_training_and_cleans_reports(
    tmp_path: Path,
) -> None:
    contract, _replay, _blank = _fixture_contract(tmp_path)
    deps, calls = _fake_dependencies(tmp_path, contract)
    report = executor.execute_recovery_ab(
        preregistration=contract.path,
        preregistration_sha256=contract.file_sha256,
        output_root=tmp_path / "output",
        folded_artifact_hooks=object(),
        dependencies=deps,
    )
    assert calls.count("authenticate") == 2
    assert calls.index("authenticate-teacher-replay") < calls.index("materialize")
    assert calls.index("authenticate-materializations") < calls.index(
        "authenticate-source-model"
    )
    assert calls.index("authenticate-source-model") < calls.index("zero:folded")
    assert calls.index("zero:folded") < calls.index("authenticate-zero")
    assert calls.index("zero:blank") < calls.index("authenticate-zero")
    assert calls.index("authenticate-zero") < calls.index("train-start:folded")
    assert calls.index("train-start:folded") < calls.index("train-start:blank")
    assert report["status"] == "experimental-unaccepted"
    assert report["publication_attempted"] is False
    assert report["sealed_evaluation_accessed"] is False
    assert report["trajectory_summary"]["checkpoint_count"] == 3
    assert report["trajectory_summary"]["optimizer_step_horizon"] == 2
    assert report["trajectory_summary"]["sealed_capability_claim"] is False
    assert (
        report["execution"]["both_zero_steps_authenticated_before_any_optimizer_step"]
        is True
    )
    assert (
        report["execution"]["source_model_authenticated_before_zero_step_and_training"]
        is True
    )
    assert (
        report["arms"]["folded"]["recovery_static_inputs_sha256"]
        == report["arms"]["blank"]["recovery_static_inputs_sha256"]
    )
    for arm_id in ("folded", "blank"):
        final = tmp_path / "output" / f"{arm_id}_final"
        assert not (final / executor.RECOVERY_REPORT_FILENAME).exists()
        assert (
            tmp_path
            / "output"
            / executor.EVIDENCE_DIRECTORY_NAME
            / f"{arm_id}_recovery_report.json"
        ).is_file()
    persisted = json.loads(
        (tmp_path / "output" / executor.COMPARISON_REPORT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    body = dict(persisted)
    claimed = body.pop("report_payload_sha256")
    assert executor._canonical_sha256(body) == claimed
    assert not list((tmp_path / "output").glob(".zero-step-*"))


def test_training_step_zero_mismatch_raises_before_optimizer(tmp_path: Path) -> None:
    contract, _replay, _blank = _fixture_contract(tmp_path)
    deps, calls = _fake_dependencies(
        tmp_path,
        contract,
        mismatch_training_zero=True,
    )
    with pytest.raises(
        executor.FoldedBlankRecoveryExecutionError,
        match="differs from the authenticated zero-step",
    ):
        executor.execute_recovery_ab(
            preregistration=contract.path,
            preregistration_sha256=contract.file_sha256,
            output_root=tmp_path / "output",
            folded_artifact_hooks=object(),
            dependencies=deps,
        )
    assert "train-start:folded" in calls
    assert not any(value.startswith("optimizer:") for value in calls)


def test_source_fingerprint_mismatch_fails_before_zero_or_training(
    tmp_path: Path,
) -> None:
    contract, _replay, _blank = _fixture_contract(tmp_path)
    deps, calls = _fake_dependencies(
        tmp_path,
        contract,
        tamper_source_fingerprint=True,
    )
    with pytest.raises(
        executor.FoldedBlankRecoveryExecutionError,
        match="did not authenticate against replay and compiler lineage",
    ):
        executor.execute_recovery_ab(
            preregistration=contract.path,
            preregistration_sha256=contract.file_sha256,
            output_root=tmp_path / "output",
            folded_artifact_hooks=object(),
            dependencies=deps,
        )
    assert "authenticate-source-model" in calls
    assert not any(
        value.startswith(("zero:", "train-start:", "optimizer:")) for value in calls
    )


def test_candidate_tokenizer_mismatch_fails_before_zero_or_training(
    tmp_path: Path,
) -> None:
    contract, _replay, _blank = _fixture_contract(tmp_path)
    deps, calls = _fake_dependencies(
        tmp_path,
        contract,
        tamper_candidate_tokenizer=True,
    )
    with pytest.raises(
        executor.FoldedBlankRecoveryExecutionError,
        match="did not authenticate against replay and compiler lineage",
    ):
        executor.execute_recovery_ab(
            preregistration=contract.path,
            preregistration_sha256=contract.file_sha256,
            output_root=tmp_path / "output",
            folded_artifact_hooks=object(),
            dependencies=deps,
        )
    assert "authenticate-source-model" in calls
    assert not any(
        value.startswith(("zero:", "train-start:", "optimizer:")) for value in calls
    )


def test_replay_consumption_tampering_fails_closed(tmp_path: Path) -> None:
    contract, _replay, _blank = _fixture_contract(tmp_path)
    deps, _calls = _fake_dependencies(
        tmp_path,
        contract,
        tamper_consumption=True,
    )
    with pytest.raises(
        executor.FoldedBlankRecoveryExecutionError,
        match="Stepwise replay event order",
    ):
        executor.execute_recovery_ab(
            preregistration=contract.path,
            preregistration_sha256=contract.file_sha256,
            output_root=tmp_path / "output",
            folded_artifact_hooks=object(),
            dependencies=deps,
        )


def test_clean_authenticator_rejects_embedded_recovery_report(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / executor.RECOVERY_REPORT_FILENAME).write_text("{}", encoding="utf-8")
    hooks = SimpleNamespace(
        model_loader=lambda _path: pytest.fail("loader must not run"),
        native_certifier=lambda _model: {},
        runtime_auditor=lambda _model, _path: {},
    )
    with pytest.raises(
        executor.FoldedBlankRecoveryExecutionError,
        match="remains inside",
    ):
        executor._authenticate_clean_artifact(
            child,
            folded_artifact_hooks=hooks,
            expected_state_sha256=None,
        )


def test_restore_v4_winner_uses_live_rebinding_and_composite_handoff(
    monkeypatch,
) -> None:
    calls: list[str] = []
    qkvo = SimpleNamespace(
        set_training_stage=lambda stage_name: calls.append(f"qkvo-stage:{stage_name}")
    )
    gud = object()

    class Runtime:
        _guided_tranche = qkvo
        _guided_gud_tranche = gud

        def restore_source_v2(self, _transaction, _source):
            calls.append("restore-v2")
            return {"restored": True}

        def install_guided_transaction(self, _transaction, _source):
            calls.append("install-qkvo")
            return {"installed": "qkvo"}

        def install_guided_gud_transaction(self, _transaction, _binding):
            calls.append("install-gud")
            return {"installed": "gud"}

        def handoff_v4_for_publication(self, _transaction, _binding):
            calls.append("handoff")
            return {
                "handoff_complete": True,
                "transaction_qkvo_handle_is_guided_composite": True,
                "transaction_gud_handle_is_guided_composite": True,
            }

    transaction = SimpleNamespace(
        installation_report=lambda: {"installed": "inherited"}
    )
    source_v2 = SimpleNamespace(factor_state_sha256=_digest("v2"))
    source_v3 = SimpleNamespace(factor_state_sha256=_digest("runtime-v3"))
    capture_values = iter((source_v2, source_v3))

    monkeypatch.setattr(
        executor.publication,
        "capture_target_native_factor_state",
        lambda *_args, **_kwargs: next(capture_values),
    )
    monkeypatch.setattr(
        executor.publication,
        "authenticate_guided_factor_sources",
        lambda *_args: calls.append("authenticate-qkvo") or {"qkvo": "live"},
    )
    monkeypatch.setattr(
        executor.publication,
        "rebind_recovered_v3_factor_payload",
        lambda _payload, _binding: {"factor_state_sha256": _digest("runtime-v3")},
    )
    monkeypatch.setattr(
        executor.publication,
        "authenticate_guided_gud_factor_sources",
        lambda *_args: calls.append("authenticate-gud") or {"gud": "live"},
    )
    monkeypatch.setattr(
        executor.publication,
        "rebind_recovered_v4_factor_payload",
        lambda _payload, _qkvo, _gud: {"factor_state_sha256": _digest("runtime-v4")},
    )
    restore_values = iter(
        (
            SimpleNamespace(
                schema=executor.publication.BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
                factor_state_sha256=_digest("runtime-v3"),
            ),
            SimpleNamespace(
                schema=executor.publication.BEHAVIOR_FACTOR_PACK_V4_SCHEMA,
                factor_state_sha256=_digest("runtime-v4"),
            ),
        )
    )
    monkeypatch.setattr(
        executor.publication,
        "restore_target_native_factor_payload",
        lambda *_args, **_kwargs: next(restore_values),
    )
    monkeypatch.setattr(
        executor.publication,
        "portable_v4_factor_state_sha256",
        lambda *_args: _digest("portable-v4"),
    )
    v4_contract = SimpleNamespace(
        source=SimpleNamespace(factor_payload={}),
    )
    contract = SimpleNamespace(
        v4=SimpleNamespace(
            contract=v4_contract,
            factor_payload={},
            winner={},
        ),
        raw={
            "v4_candidate": {
                "winner": {
                    "runtime_factor_state_sha256": _digest("runtime-v4"),
                    "portable_factor_state_sha256": _digest("portable-v4"),
                }
            }
        },
    )
    trace = executor._restore_v4_winner(
        contract=contract,
        runtime=Runtime(),
        transaction=transaction,
        source_guided=object(),
    )
    assert calls == [
        "restore-v2",
        "install-qkvo",
        "authenticate-qkvo",
        "qkvo-stage:joint",
        "install-gud",
        "authenticate-gud",
        "handoff",
    ]
    assert trace["runtime_rebound_v4_factor_state_sha256"] == _digest("runtime-v4")
    assert trace["portable_v4_factor_state_sha256"] == _digest("portable-v4")
