from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    plan_qwen3_codebook_artifact,
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)
from neural_forge.hf_qwen_surgery_allocator import (  # noqa: E402
    Qwen3UnifiedSurgeryAccounting,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_qwen_unified_surgery.py"
)
SPEC = importlib.util.spec_from_file_location("run_qwen_unified_surgery", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runtime_identity(value: str) -> dict:
    payload = {
        "schema": "neural-forge-unified-runtime-environment-v1",
        "label": value,
    }
    payload["identity_sha256"] = MODULE._canonical_sha256(payload)
    return payload


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    name_or_path = "tiny-test-tokenizer"
    padding_side = "right"

    def __init__(self, size: int = 32) -> None:
        self.tokens = tuple(f"token-{index}" for index in range(size))

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]

    def __call__(self, text, **kwargs):
        values = [text] if isinstance(text, str) else list(text)
        rows = [
            [1, *[3 + (ord(character) % 25) for character in value], 2]
            for value in values
        ]
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            rows = [row[: int(kwargs["max_length"])] for row in rows]
        width = max(len(row) for row in rows)
        if len(rows) > 1 and not kwargs.get("padding"):
            raise AssertionError("Batched tokenizer calls must request padding.")
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            start = width - len(row) if self.padding_side == "left" else 0
            ids[index, start : start + len(row)] = torch.tensor(row)
            mask[index, start : start + len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _tiny_runtime(seed: int = 771, layers: int = 2):
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=64,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()
    tokenizer = _Tokenizer()
    return (
        prepare_qwen3_codebook_model(
            model,
            tokenizer,
            tokenizer,
            attention_prototypes_per_group=layers,
            neuron_prototypes_per_neuron=layers,
        ),
        tokenizer,
    )


def _accounting(value: int = 10_000) -> Qwen3UnifiedSurgeryAccounting:
    return Qwen3UnifiedSurgeryAccounting(
        source_unique_parameters=20_000,
        child_unique_parameters=value // 2,
        parameter_tensor_bytes=value - 100,
        buffer_tensor_bytes=0,
        tensor_bytes=value - 100,
        neuron_registry_bytes=2_000,
        attention_registry_bytes=1_000,
        index_bytes=100,
        scale_bytes=0,
        metadata_bytes=100,
        deployable_bytes=value,
        parameter_report_sha256=_digest(f"parameter:{value}"),
        neuron_registry_sha256=_digest(f"neuron:{value}"),
        attention_registry_sha256=_digest(f"attention:{value}"),
    )


def _artifact_accounting(
    runtime,
    tokenizer: _Tokenizer | None = None,
) -> Qwen3UnifiedSurgeryAccounting:
    resolved_tokenizer = tokenizer or _Tokenizer()
    parent_rows = runtime.parent_token_rows.detach().to(device="cpu").tolist()
    contract = MODULE.codebook_runner._tokenizer_checkpoint_contract(
        resolved_tokenizer,
        tokenizer_name="tiny",
        parent_rows=tuple(int(item) for item in parent_rows),
    )
    plan = plan_qwen3_codebook_artifact(
        runtime,
        tokenizer_contract=contract,
    )
    report = qwen3_codebook_parameter_report(runtime)
    return replace(
        _accounting(plan.artifact_total_bytes),
        source_unique_parameters=report.source_unique_parameter_count,
        child_unique_parameters=report.unique_parameter_count,
        parameter_tensor_bytes=plan.runtime_parameter_storage_bytes,
        buffer_tensor_bytes=plan.persistent_buffer_storage_bytes,
        tensor_bytes=plan.safetensors_payload_bytes,
        index_bytes=report.index_buffer_bytes,
        metadata_bytes=(
            plan.artifact_total_bytes - plan.safetensors_payload_bytes
        ),
        logical_state_tensor_bytes=plan.logical_state_tensor_bytes,
        runtime_registered_storage_bytes=plan.runtime_registered_storage_bytes,
        artifact_payload_bytes=plan.safetensors_payload_bytes,
        artifact_header_bytes=plan.safetensors_header_bytes,
        artifact_config_bytes=plan.config_file_bytes,
        artifact_tokenizer_contract_bytes=plan.tokenizer_contract_file_bytes,
        artifact_manifest_bytes=plan.manifest_file_bytes,
        artifact_plan_sha256=plan.artifact_plan_sha256,
        artifact_tokenizer_contract_resolved=True,
    )


def _report(
    *,
    behavior_correct: int = 10,
    choice_correct: int = 10,
    generation_correct: int = 10,
    total: int = 10,
    prompt_prefix: str = "gate",
) -> dict:
    behavior_items = [
        {
            "id": f"b{index}",
            "prompt": f"{prompt_prefix} behavior {index}",
            "passed": index < behavior_correct,
            "output": "ok",
        }
        for index in range(total)
    ]
    choice_items = [
        {
            "id": f"c{index}",
            "prediction": 0 if index < choice_correct else 1,
            "answer": 0,
            "correct": index < choice_correct,
            "scores": [1.0, 0.0] if index < choice_correct else [0.0, 1.0],
            "choices": [" A", " B"],
        }
        for index in range(total)
    ]
    generation_items = [
        {
            "id": f"g{index}",
            "prompt": f"{prompt_prefix} generation {index}",
            "passed": index < generation_correct,
            "output": "ok",
        }
        for index in range(total)
    ]
    return {
        "model": "test",
        "param_count": 1,
        "eval_suite": {
            "source": "test",
            "text_eval_count": 1,
            "choice_probe_count": total,
            "generation_prompt_count": total,
            "behavior_probe_count": total,
        },
        "perplexity": {"mean": 1000.0, "items": [1000.0]},
        "choice_accuracy": {
            "correct": choice_correct,
            "total": total,
            "accuracy": choice_correct / total,
            "items": choice_items,
        },
        "generations": generation_items,
        "generation_diagnostics": generation_items,
        "generation_health": {
            "correct": generation_correct,
            "total": total,
            "pass_rate": generation_correct / total,
            "contract": dict(MODULE.production_runner.GENERATION_HEALTH_CONTRACT),
        },
        "behavior_suite": {
            "correct": behavior_correct,
            "total": total,
            "pass_rate": behavior_correct / total,
            "items": behavior_items,
        },
    }


def _bound_suite(
    total: int = 10,
    *,
    prompt_prefix: str = "gate",
) -> MODULE.BoundEvaluationSuite:
    return MODULE.bind_evaluation_suite(
        _Tokenizer(),
        {
            "text_evals": ["text"],
            "choice_probes": [
                {
                    "id": f"c{index}",
                    "prompt": f"{prompt_prefix} choice {index}",
                    "choices": [" A", " B"],
                    "answer": 0,
                }
                for index in range(total)
            ],
            "generation_prompts": [
                {
                    "id": f"g{index}",
                    "prompt": f"{prompt_prefix} generation {index}",
                    "relevance_rules": [],
                }
                for index in range(total)
            ],
            "behavior_probes": [
                {
                    "id": f"b{index}",
                    "prompt": f"{prompt_prefix} behavior {index}",
                    "kind": "contains",
                    "expected": "ok",
                }
                for index in range(total)
            ],
        },
        name="gate",
        max_new_tokens=1,
    )


def _fake_evaluator(report: dict):
    def evaluate(_model, _tokenizer, output_file, **kwargs):
        result = copy.deepcopy(report)
        identity = kwargs.get("identity_binding")
        if identity is not None:
            result["identity_binding"] = dict(identity)
        suite = kwargs.get("eval_suite")
        if isinstance(suite, dict) and isinstance(result.get("eval_suite"), dict):
            result["eval_suite"]["source"] = suite.get("source", "<dict>")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(json.dumps(result), encoding="utf-8")
        return result

    return evaluate


def _labeled_evaluator(report: dict):
    evaluate = _fake_evaluator(report)

    def labeled(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        result["model"] = str(kwargs["model_label"])
        Path(args[2]).write_text(json.dumps(result), encoding="utf-8")
        return result

    return labeled


def _lineage(value: str = "lineage") -> dict:
    payload = {
        "model_root": _digest(f"{value}:model"),
        "mode": "incremental",
        "parent_root": _digest(f"{value}:parent"),
        "accepted_transaction_sha256": _digest(f"{value}:transaction"),
        "changed_state_prefixes": ["neuron_codebook.registry."],
        "tensors_hashed": 1,
        "tensors_reused": 2,
        "calibration_sha256": _digest(f"{value}:calibration"),
        "gate_sha256": _digest(f"{value}:gate"),
        "runtime_sha256": _digest(f"{value}:runtime"),
        "measurement_sha256": _digest(f"{value}:measurement"),
        "cache_sha256": _digest(f"{value}:cache"),
    }
    payload["lineage_sha256"] = MODULE._canonical_sha256(payload)
    return payload


def _accepted_transaction_proof(
    root: Path,
    name: str,
    *,
    parent_accounting: dict,
    committed_accounting: dict,
    transaction_index: int = 1,
) -> tuple[dict, Path, float, str, str]:
    parent_state = _digest(f"{name}:parent-state")
    candidate_state = _digest(f"{name}:candidate-state")
    batch_sha256 = _digest(f"{name}:batch")
    proposal_sha256 = _digest(f"{name}:validated-proposal")
    candidate_report = _report()
    candidate_path = root / f"{name}.candidate.json"
    candidate_path.write_text(json.dumps(candidate_report), encoding="utf-8")
    score, count = MODULE._behavior_score(candidate_report)
    metrics = {
        "parent_relative": True,
        "evaluation_kind": MODULE.surgery_allocator._BEHAVIORAL_GATE_KIND,
        "parent_state_sha256": parent_state,
        "candidate_state_sha256": candidate_state,
        "evaluation_suite_sha256": _digest(f"{name}:suite"),
        "parent_outputs_sha256": _digest(f"{name}:parent-outputs"),
        "candidate_outputs_sha256": MODULE._canonical_sha256(
            MODULE._behavior_outputs(candidate_report)
        ),
        "behavioral_sample_count": count,
        "parent_behavior_score": score,
        "candidate_behavior_score": score,
    }
    metrics["evaluation_fingerprint"] = (
        MODULE.surgery_allocator._behavioral_evaluation_fingerprint(metrics)
    )
    transaction = {
        "transaction_index": transaction_index,
        "evidence_epoch_before": transaction_index - 1,
        "evidence_epoch_after": transaction_index,
        "batch_sha256": batch_sha256,
        "proposal_sha256s": [proposal_sha256],
        "accepted": True,
        "decision": {
            "accepted": True,
            "reason": "test accepted gate",
            "metrics": metrics,
        },
        "parent_accounting": parent_accounting,
        "proposed_accounting": committed_accounting,
        "committed_accounting": committed_accounting,
        "rollback_exact": None,
        "measured_behavioral_improvement_claim": False,
        "zero_recovery": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "name": name,
    }
    gate = root / f"{name}.gate.json"
    gate.write_text(
        json.dumps(
            {
                "context": {
                    "transaction_index": transaction_index,
                    "parent_state_sha256": parent_state,
                    "candidate_state_sha256": candidate_state,
                    "batch_sha256": batch_sha256,
                },
                "metrics": metrics,
                "verdict": {"passed": True},
                "candidate_report": str(candidate_path.resolve()),
                "candidate_report_sha256": MODULE.production_runner._file_sha256(
                    candidate_path
                ),
            }
        ),
        encoding="utf-8",
    )
    return transaction, gate, score, parent_state, proposal_sha256


def _write_candidate_validation_proof(
    root: Path,
    *,
    parent_state: str,
    proposal_sha256: str,
) -> tuple[Path, str]:
    round_dir = root / "candidate_validation" / "round-000001"
    round_dir.mkdir(parents=True)
    claim = {
        "schema": "neural-forge-independent-candidate-round-v1",
        "round_index": 1,
        "directory": str(round_dir.resolve()),
    }
    claim["round_id"] = MODULE._canonical_sha256(claim)
    (round_dir / "round_claim.json").write_text(
        json.dumps(claim), encoding="utf-8"
    )
    suite_sha256 = _digest("candidate-suite")
    discovery = _digest("candidate-discovery")
    validation = _digest("candidate-validation")
    candidate_sha256 = _digest("candidate")
    parent_report = _report(prompt_prefix="validation")
    parent_outputs = MODULE._canonical_sha256(
        MODULE._behavior_outputs(parent_report)
    )
    parent_cache = round_dir / "parent-cache.json"
    parent_cache.write_text(
        json.dumps(
            {
                "schema": "neural-forge-cached-parent-suite-v1",
                "suite_sha256": suite_sha256,
                "parent_state_sha256": parent_state,
                "parent_outputs_sha256": parent_outputs,
                "report": parent_report,
            }
        ),
        encoding="utf-8",
    )
    candidate_report = _report(prompt_prefix="validation")
    candidate_path = round_dir / "candidate-0001-neuron.json"
    candidate_path.write_text(json.dumps(candidate_report), encoding="utf-8")
    candidate_outputs = MODULE._canonical_sha256(
        MODULE._behavior_outputs(candidate_report)
    )
    parent_fp = MODULE._canonical_sha256(
        {
            "role": "independent-candidate-marginal-parent",
            "candidate_sha256": candidate_sha256,
            "parent_state_sha256": parent_state,
            "suite_sha256": suite_sha256,
            "outputs_sha256": parent_outputs,
        }
    )
    candidate_fp = MODULE._canonical_sha256(
        {
            "role": "independent-candidate-marginal",
            "candidate_sha256": candidate_sha256,
            "parent_state_sha256": parent_state,
            "suite_sha256": suite_sha256,
            "outputs_sha256": candidate_outputs,
        }
    )
    rows = tuple(_digest(f"validation-row-{index}") for index in range(30))
    record = MODULE.Qwen3CandidateValidationDamage(
        kind="neuron",
        candidate_sha256=candidate_sha256,
        parent_state_sha256=parent_state,
        discovery_fingerprint=discovery,
        validation_fingerprint=validation,
        discovery_row_fingerprints=(_digest("discovery-row"),),
        validation_row_fingerprints=rows,
        marginal_damage=0.0,
        metric_name="independent_behavioral_failure_rate",
        parent_metric_value=0.0,
        candidate_metric_value=0.0,
        metric_direction="lower_is_better",
        parent_evaluation_fingerprint=parent_fp,
        candidate_evaluation_fingerprint=candidate_fp,
        validation_sample_count=30,
    )
    audit = {
        "proposal_sha256": _digest("unvalidated-proposal"),
        "validated_proposal_sha256": proposal_sha256,
        "candidate_sha256": candidate_sha256,
        "kind": "neuron",
        "marginal_damage": 0.0,
        "rollback_exact": True,
        "candidate_error": 0.0,
        "candidate_behavioral_sample_count": 30,
        "candidate_outputs_sha256": candidate_outputs,
        "candidate_evaluation_fingerprint": candidate_fp,
        "artifact": str(candidate_path.resolve()),
        "artifact_sha256": MODULE.production_runner._file_sha256(candidate_path),
    }
    summary = {
        "schema": MODULE.CANDIDATE_VALIDATION_SUMMARY_SCHEMA,
        "validation_class": "completed-independent-candidate-validation",
        "round_index": 1,
        "round_id": claim["round_id"],
        "parent_state_sha256": parent_state,
        "discovery_fingerprint": discovery,
        "validation_fingerprint": validation,
        "suite_sha256": suite_sha256,
        "proposal_sequence_sha256": MODULE._canonical_sha256(
            [audit["proposal_sha256"]]
        ),
        "validated_proposal_sequence_sha256": MODULE._canonical_sha256(
            [proposal_sha256]
        ),
        "validated_candidate_count": 1,
        "records": [record.as_dict()],
        "audits": [audit],
        "every_probe_rolled_back_exactly": True,
        "parent_forward_evaluations_for_round": 1,
        "parent_cache_artifact": str(parent_cache.resolve()),
        "parent_cache_artifact_sha256": MODULE.production_runner._file_sha256(
            parent_cache
        ),
    }
    summary["summary_sha256"] = MODULE._record_sha256(summary, "summary_sha256")
    path = round_dir / "validation_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path, str(claim["round_id"])


def _journal_with_commit(
    tmp_path: Path,
    *,
    candidate_validation_mode: str = "provisional",
):
    runtime, tokenizer = _tiny_runtime(seed=911)
    manager = MODULE.ParetoCheckpointManager(
        output_directory=tmp_path,
        tokenizer=tokenizer,
        tokenizer_name="tiny",
        max_checkpoints=3,
        verify_reload=True,
    )
    journal = MODULE.AcceptedCommitJournal(tmp_path)
    content = MODULE._model_content_identity(runtime)
    artifact_accounting = _artifact_accounting(runtime)
    contract = {
        "schema": MODULE.RUN_CONTRACT_SCHEMA,
        "name": "test-contract",
        "candidate_validation_mode": candidate_validation_mode,
    }
    contract_sha256 = MODULE._canonical_sha256(contract)
    contract["binding_sha256"] = contract_sha256
    contract["run_contract_sha256"] = contract_sha256
    journal.record_or_verify_initial(
        model_content=content,
        accounting=artifact_accounting,
        run_contract=contract,
        source_identity={"kind": "test"},
    )
    transaction_payload, gate, behavior_score, parent_state, proposal_sha256 = (
        _accepted_transaction_proof(
            tmp_path,
            "transaction-one",
            parent_accounting=artifact_accounting.as_dict(),
            committed_accounting=artifact_accounting.as_dict(),
        )
    )
    validation_artifact = None
    validation_round_id = None
    if candidate_validation_mode == "independent":
        validation_artifact, validation_round_id = (
            _write_candidate_validation_proof(
                tmp_path,
                parent_state=parent_state,
                proposal_sha256=proposal_sha256,
            )
        )
    manager.record(
        runtime,
        accepted_transaction_total=1,
        state_sha256=_digest("ephemeral-one"),
        accounting=artifact_accounting,
        behavior_score=behavior_score,
        transaction_sha256=_digest("transaction-one"),
    )
    frontier = manager.current_record
    assert frontier is not None
    commit = journal.append(
        parent_commit_seq=0,
        parent_model_content_sha256=content["model_content_sha256"],
        accepted_transaction_total=1,
        model_content_sha256=frontier.model_content_sha256,
        checkpoint=Path(frontier.checkpoint),
        artifact_tree={
            "tree_sha256": frontier.artifact_tree_sha256,
            "total_bytes": frontier.artifact_total_bytes,
        },
        accounting=artifact_accounting,
        run_contract_sha256=contract["run_contract_sha256"],
        transaction_payload=transaction_payload,
        gate_artifact=gate,
        candidate_validation_artifact=validation_artifact,
        candidate_validation_round_id=validation_round_id,
        evidence_lineage=_lineage("one"),
        behavior_score=behavior_score,
    )
    return runtime, tokenizer, manager, journal, commit, contract, gate


def test_prompt_rows_are_left_padded_and_split_overlap_fails() -> None:
    tokenizer = _Tokenizer()
    calibration = MODULE.encode_prompt_rows(
        tokenizer, ["a", "long"], max_length=32
    )
    gate_rows = MODULE.encode_prompt_rows(tokenizer, ["gate"], max_length=32)

    assert calibration.attention_mask[:, -1].tolist() == [True, True]
    assert calibration.attention_mask[0, 0].item() is False
    suite = _bound_suite()
    report = MODULE.build_split_provenance(
        calibration,
        gate_rows,
        suite,
        None,
        mode="provisional",
    )
    assert report["all_token_paths_disjoint"] is True

    overlapping = MODULE.bind_evaluation_suite(
        tokenizer,
        {
            **dict(suite.suite),
            "generation_prompts": [
                {"id": "overlap-generation", "prompt": "a", "relevance_rules": []}
            ],
        },
        name="overlapping",
        max_new_tokens=1,
    )
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="overlap"):
        MODULE.build_split_provenance(
            calibration,
            gate_rows,
            overlapping,
            None,
            mode="provisional",
        )


def test_split_provenance_allows_boilerplate_but_rejects_truncated_prefix() -> None:
    tokenizer = _Tokenizer()
    calibration = MODULE.encode_prompt_rows(
        tokenizer,
        ["shared boilerplate calibration unique ending"],
        max_length=128,
    )
    gate_rows = MODULE.encode_prompt_rows(
        tokenizer,
        ["shared boilerplate sensitivity unique ending"],
        max_length=128,
    )
    independent = _bound_suite(
        total=1,
        prompt_prefix="shared boilerplate transaction unique ending",
    )
    report = MODULE.build_split_provenance(
        calibration,
        gate_rows,
        independent,
        None,
        mode="provisional",
    )
    assert report["all_token_paths_disjoint"] is True
    assert all(
        item["token_prefix"] == 0
        for item in report["pairwise_overlaps"].values()
    )

    truncated = MODULE.encode_prompt_rows(
        tokenizer,
        ["abcdefghijklmno"],
        max_length=10,
    )
    leaked = _bound_suite(
        total=1,
        prompt_prefix="abcdefghijklmno with a held-out suffix",
    )
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="token_prefix"):
        MODULE.build_split_provenance(
            truncated,
            gate_rows,
            leaked,
            None,
            mode="provisional",
        )


def test_absolute_ninety_percent_floors_replace_zero_loss_gate(tmp_path: Path) -> None:
    suite = _bound_suite()
    candidate = _report(behavior_correct=9, choice_correct=9, generation_correct=9)
    executor = MODULE.SuiteExecutor(
        tokenizer=object(),
        device="cpu",
        max_new_tokens=1,
        evaluator=_fake_evaluator(candidate),
    )
    cache = MODULE.CachedParentSuite(
        suite=suite,
        executor=executor,
        output_directory=tmp_path / "cache",
    )
    parent_state = "a" * 64
    cache.store_parent(parent_state, _report(), source="test")
    gate = MODULE.ParentRelativeBehavioralGate(
        parent_cache=cache,
        retention_floors=MODULE.RetainedParentFloors(),
        hard_loss_limits=(None, None, None),
        output_directory=tmp_path / "gates",
    )
    gate.bind_parent(torch.nn.Identity(), parent_state, _accounting())
    context = SimpleNamespace(
        runtime=object(),
        transaction_index=1,
        parent_state_sha256=parent_state,
        proposed_state_sha256="b" * 64,
        proposed_accounting=_accounting(9_000),
        batch=SimpleNamespace(batch_sha256="c" * 64),
    )

    decision = gate(context)

    assert decision.accepted is True
    audit = json.loads(gate.last_artifact.read_text(encoding="utf-8"))
    assert audit["verdict"]["retained_parent_ratios"] == {
        "behavior": 0.9,
        "choice": 0.9,
        "generation": 0.9,
    }
    assert audit["verdict"]["perplexity_used_for_acceptance"] is False


def test_absolute_floor_rejects_below_ninety_percent(tmp_path: Path) -> None:
    suite = _bound_suite()
    candidate = _report(behavior_correct=8, choice_correct=9, generation_correct=9)
    executor = MODULE.SuiteExecutor(
        tokenizer=object(),
        device="cpu",
        max_new_tokens=1,
        evaluator=_fake_evaluator(candidate),
    )
    cache = MODULE.CachedParentSuite(
        suite=suite,
        executor=executor,
        output_directory=tmp_path / "cache",
    )
    parent_state = "a" * 64
    cache.store_parent(parent_state, _report(), source="test")
    gate = MODULE.ParentRelativeBehavioralGate(
        parent_cache=cache,
        retention_floors=MODULE.RetainedParentFloors(),
        hard_loss_limits=(None, None, None),
        output_directory=tmp_path / "gates",
    )
    gate.bind_parent(torch.nn.Identity(), parent_state, _accounting())
    context = SimpleNamespace(
        runtime=object(),
        transaction_index=1,
        parent_state_sha256=parent_state,
        proposed_state_sha256="b" * 64,
        proposed_accounting=_accounting(9_000),
        batch=SimpleNamespace(batch_sha256="c" * 64),
    )

    assert gate(context).accepted is False


def test_failure_report_preserves_latest_resume_state(tmp_path: Path) -> None:
    path = tmp_path / "run_report.json"
    latest = {
        "schema": MODULE.RUN_SCHEMA,
        "status": "running",
        "attempts_total": 7,
        "accepted_transactions_total": 3,
        "current_checkpoint": "checkpoint-3",
        "current_accounting": {"deployable_bytes": 1234},
        "transaction_history": [{"attempt": 7}],
    }
    MODULE.production_runner._write_json(path, latest)
    fallback = {"schema": MODULE.RUN_SCHEMA, "attempts_total": 0}

    persisted = MODULE.persist_failure_report(path, fallback, RuntimeError("exhausted"))

    assert persisted["status"] == "failed"
    assert persisted["attempts_total"] == 7
    assert persisted["accepted_transactions_total"] == 3
    assert persisted["current_checkpoint"] == "checkpoint-3"
    assert persisted["current_accounting"] == {"deployable_bytes": 1234}
    assert persisted["transaction_history"] == [{"attempt": 7}]


def test_parser_targets_are_mutually_exclusive_and_floors_default_to_point_nine() -> None:
    common = [
        "--source-model",
        "qwen",
        "--output",
        "out",
        "--calibration-file",
        "cal.jsonl",
        "--sensitivity-gate-file",
        "gate.jsonl",
        "--candidate-validation-suite",
        "validation.json",
        "--transaction-gate-suite",
        "selection.json",
    ]
    args = MODULE.build_parser().parse_args([*common, "--target-savings-bytes", "42"])
    assert args.target_deployable_bytes is None
    assert args.target_savings_bytes == 42
    assert (
        args.min_behavior_retention,
        args.min_choice_retention,
        args.min_generation_retention,
    ) == (0.9, 0.9, 0.9)
    with pytest.raises(SystemExit):
        MODULE.build_parser().parse_args(
            [
                *common,
                "--target-savings-bytes",
                "42",
                "--target-deployable-bytes",
                "100",
            ]
        )


def test_target_savings_resolves_from_verified_incumbent_accounting() -> None:
    assert MODULE.resolve_deployable_target(
        12_345,
        target_deployable_bytes=None,
        target_savings_bytes=2_345,
    ) == 10_000
    assert MODULE.resolve_deployable_target(
        12_345,
        target_deployable_bytes=9_876,
        target_savings_bytes=None,
    ) == 9_876
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="exceeds"):
        MODULE.resolve_deployable_target(
            100,
            target_deployable_bytes=None,
            target_savings_bytes=101,
        )


def test_existing_codebook_checkpoint_loads_without_dense_identity_rebuild(
    tmp_path: Path,
) -> None:
    runtime, tokenizer = _tiny_runtime()
    checkpoint = tmp_path / "incumbent-p1"
    parent_rows = runtime.parent_token_rows.detach().to(device="cpu").tolist()
    save_qwen3_codebook_checkpoint(
        runtime,
        checkpoint,
        tokenizer_contract=MODULE.codebook_runner._tokenizer_checkpoint_contract(
            tokenizer,
            tokenizer_name="tiny",
            parent_rows=tuple(int(value) for value in parent_rows),
        ),
    )
    args = argparse.Namespace(
        input_codebook_checkpoint=checkpoint,
        source_model=None,
        device="cpu",
        expected_layers=2,
        dtype="float32",
        trust_remote_code=False,
        local_files_only=True,
        max_length=16,
    )

    loaded, report, resumed = MODULE._load_or_prepare_runtime(
        args, tokenizer, None, "unused"
    )

    assert resumed is False
    assert report["identity_preparation_repeated"] is False
    assert report["input_codebook_checkpoint"] == str(checkpoint.resolve())
    assert qwen3_codebook_parameter_report(loaded) == qwen3_codebook_parameter_report(
        runtime
    )


def test_frontier_checkpoint_is_standalone_and_reload_verified(tmp_path: Path) -> None:
    runtime, tokenizer = _tiny_runtime(seed=772)
    manager = MODULE.ParetoCheckpointManager(
        output_directory=tmp_path,
        tokenizer=tokenizer,
        tokenizer_name="tiny",
        max_checkpoints=1,
        verify_reload=True,
    )
    accounting = _artifact_accounting(runtime)

    manifest = manager.record(
        runtime,
        accepted_transaction_total=0,
        state_sha256="d" * 64,
        accounting=accounting,
        behavior_score=1.0,
        transaction_sha256=None,
    )

    assert manifest["retained_count"] == 1
    assert manager.records[0].reload_verified is True
    assert manager.records[0].state_sha256 == manager.records[0].model_content_sha256
    assert manager.records[0].state_sha256 != "d" * 64
    checkpoint = Path(manager.records[0].checkpoint)
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "manifest.json").is_file()
    loaded = MODULE.load_qwen3_codebook_checkpoint(checkpoint, device="cpu")
    assert MODULE._model_content_identity(loaded) == MODULE._model_content_identity(
        runtime
    )

    manager.record(
        runtime,
        accepted_transaction_total=99,
        state_sha256="e" * 64,
        accounting=accounting,
        behavior_score=0.5,
        transaction_sha256=_digest("different-ephemeral-transaction"),
    )
    assert len(manager.records) == 1
    assert Path(manager.current_checkpoint) == checkpoint


def test_independent_candidate_validation_forces_measurement_rollback(
    tmp_path: Path,
) -> None:
    parent_state = "1" * 64
    proposal = SimpleNamespace(
        kind="neuron",
        candidate=SimpleNamespace(candidate_sha256="2" * 64),
        individual_physical_bytes_saved=100,
        prototype_bytes_saved=100,
        scale_setup_bytes=0,
        risk=0.5,
        proposal_sha256="3" * 64,
    )
    evidence = SimpleNamespace(
        fingerprints=SimpleNamespace(
            model="0" * 64,
            runtime="1" * 64,
            measurement="2" * 64,
            sketch_basis="3" * 64,
            cache="4" * 64,
            calibration_row_fingerprints=("5" * 64,),
            gate_row_fingerprints=("6" * 64,),
        )
    )

    class FakeAllocator:
        def __init__(self) -> None:
            self.runtime = torch.nn.Identity()
            self.installed = ()
            self.apply_calls = 0

        def _state_sha256(self):
            return parent_state

        def accounting(self):
            return _accounting()

        def apply_batch(self, batch, *, evaluator, **kwargs):
            self.apply_calls += 1
            assert kwargs["unsafe_allow_unvalidated_candidates"] is True
            decision = evaluator(
                SimpleNamespace(
                    runtime=object(),
                    proposed_accounting=_accounting(9_900),
                )
            )
            assert decision.accepted is False
            return SimpleNamespace(accepted=False, rollback_exact=True)

        def install_candidate_validation(self, records):
            self.installed = tuple(records)

    suite = _bound_suite(prompt_prefix="validation")
    executor = MODULE.SuiteExecutor(
        tokenizer=object(),
        device="cpu",
        max_new_tokens=1,
        evaluator=_fake_evaluator(
            _report(behavior_correct=9, prompt_prefix="validation")
        ),
    )
    cache = MODULE.CachedParentSuite(
        suite=suite,
        executor=executor,
        output_directory=tmp_path / "cache",
    )
    cache.store_parent(
        parent_state,
        _report(prompt_prefix="validation"),
        source="test",
    )
    allocator = FakeAllocator()
    validator = MODULE.IndependentCandidateValidator(
        allocator=allocator,
        evidence=lambda: evidence,
        parent_cache=cache,
        output_directory=tmp_path / "validation",
        max_candidates=1,
        discovery_row_fingerprints=("5" * 64, "6" * 64),
    )

    summary = validator.validate((proposal,))

    assert allocator.apply_calls == 1
    assert len(allocator.installed) == 1
    assert allocator.installed[0].has_complete_candidate_marginal_measurement
    assert summary["every_probe_rolled_back_exactly"] is True


def test_candidate_discovery_manifest_binds_exact_and_semantic_order() -> None:
    operations = ("8" * 64, "9" * 64)
    proposals = tuple(
        SimpleNamespace(
            kind=kind,
            candidate=SimpleNamespace(
                candidate_sha256=str(index + 2) * 64,
                operation_sha256=operation,
            ),
            proposal_sha256=str(index + 4) * 64,
        )
        for index, (kind, operation) in enumerate(
            zip(("neuron", "attention"), operations, strict=True)
        )
    )
    evidence = SimpleNamespace(
        fingerprints=SimpleNamespace(
            model="0" * 64,
            runtime="1" * 64,
            measurement="2" * 64,
            sketch_basis="3" * 64,
            cache="4" * 64,
        )
    )
    manifest = MODULE._candidate_discovery_manifest(
        proposals,
        attempt_total=1,
        parent_state_sha256="5" * 64,
        evidence=evidence,
    )
    MODULE._assert_candidate_discovery_manifest(manifest)
    assert [
        item["candidate_operation_sha256"] for item in manifest["records"]
    ] == list(operations)
    reversed_manifest = MODULE._candidate_discovery_manifest(
        tuple(reversed(proposals)),
        attempt_total=2,
        parent_state_sha256="5" * 64,
        evidence=evidence,
    )
    assert reversed_manifest["semantic_operation_set_sha256"] == manifest[
        "semantic_operation_set_sha256"
    ]
    assert reversed_manifest["ordered_semantic_operation_sequence_sha256"] != (
        manifest["ordered_semantic_operation_sequence_sha256"]
    )
    tampered = copy.deepcopy(manifest)
    tampered["records"][0]["candidate_operation_sha256"] = "a" * 64
    with pytest.raises(
        MODULE.QwenUnifiedSurgeryRunnerError,
        match="exact integrity",
    ):
        MODULE._assert_candidate_discovery_manifest(tampered)


def test_candidate_import_preflight_rejects_source_compiler_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_run = tmp_path / "source"
    source_run.mkdir()
    current_manifest = MODULE._implementation_source_manifest()
    source_manifest = copy.deepcopy(current_manifest)
    source_manifest["files"][0]["sha256"] = _digest("changed-source")
    source_manifest["manifest_sha256"] = MODULE._canonical_sha256(
        source_manifest["files"]
    )
    policy = {"schema": "deterministic-test", "seed": 0}
    source_contract = {
        "schema": MODULE.RUN_CONTRACT_SCHEMA,
        "implementation_source_manifest": source_manifest,
        "deterministic_execution": policy,
    }
    digest = MODULE._canonical_sha256(source_contract)
    source_contract["binding_sha256"] = digest
    source_contract["run_contract_sha256"] = digest
    MODULE.production_runner._write_json(
        source_run / "initial_incumbent.json",
        {"run_contract": source_contract},
    )
    monkeypatch.setattr(
        MODULE,
        "_candidate_validation_import_contract",
        lambda _path: {"source_run": str(source_run)},
    )
    with pytest.raises(
        MODULE.QwenUnifiedSurgeryRunnerError,
        match="implementation differs.*fresh independent validation",
    ):
        MODULE._preflight_candidate_validation_source_implementation(
            tmp_path / "raw-round",
            implementation_source_manifest=current_manifest,
            deterministic_execution=policy,
        )


def test_candidate_validation_rejects_wrong_discovery_rows_before_forward(
    tmp_path: Path,
) -> None:
    evidence = SimpleNamespace(
        fingerprints=SimpleNamespace(
            model="0" * 64,
            runtime="1" * 64,
            measurement="2" * 64,
            sketch_basis="3" * 64,
            cache="4" * 64,
            calibration_row_fingerprints=("5" * 64,),
            gate_row_fingerprints=("6" * 64,),
        )
    )

    class NoForwardAllocator:
        runtime = torch.nn.Identity()
        apply_calls = 0

        def _state_sha256(self):
            return "1" * 64

        def accounting(self):
            return _accounting()

        def apply_batch(self, *_args, **_kwargs):
            self.apply_calls += 1
            raise AssertionError("wrong provenance must fail before candidate forward")

    allocator = NoForwardAllocator()
    validator = MODULE.IndependentCandidateValidator(
        allocator=allocator,
        evidence=lambda: evidence,
        parent_cache=SimpleNamespace(),
        output_directory=tmp_path / "validation",
        max_candidates=1,
        discovery_row_fingerprints=("7" * 64,),
    )
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="exact sensitivity"):
        validator.validate((SimpleNamespace(),))
    assert allocator.apply_calls == 0


def test_candidate_validation_round_ids_are_persistent_and_exclusive(
    tmp_path: Path,
) -> None:
    parent_state = "1" * 64
    proposal = SimpleNamespace(
        kind="neuron",
        candidate=SimpleNamespace(candidate_sha256="2" * 64),
        individual_physical_bytes_saved=100,
        prototype_bytes_saved=100,
        scale_setup_bytes=0,
        risk=0.5,
        proposal_sha256="3" * 64,
    )
    evidence = SimpleNamespace(
        fingerprints=SimpleNamespace(
            cache="4" * 64,
            calibration_row_fingerprints=("5" * 64,),
            gate_row_fingerprints=("6" * 64,),
        )
    )

    class FakeAllocator:
        runtime = torch.nn.Identity()

        def _state_sha256(self):
            return parent_state

        def accounting(self):
            return _accounting()

        def apply_batch(self, _batch, *, evaluator, **_kwargs):
            decision = evaluator(
                SimpleNamespace(runtime=object(), proposed_accounting=_accounting())
            )
            assert decision.accepted is False
            return SimpleNamespace(accepted=False, rollback_exact=True)

        def install_candidate_validation(self, _records):
            return None

    suite = _bound_suite(prompt_prefix="validation")
    cache = MODULE.CachedParentSuite(
        suite=suite,
        executor=MODULE.SuiteExecutor(
            tokenizer=object(),
            device="cpu",
            max_new_tokens=1,
                evaluator=_fake_evaluator(_report(prompt_prefix="validation")),
        ),
        output_directory=tmp_path / "cache",
    )
    cache.store_parent(
        parent_state,
        _report(prompt_prefix="validation"),
        source="test",
    )
    output = tmp_path / "validation"
    first = MODULE.IndependentCandidateValidator(
        allocator=FakeAllocator(),
        evidence=lambda: evidence,
        parent_cache=cache,
        output_directory=output,
        max_candidates=1,
        discovery_row_fingerprints=("5" * 64, "6" * 64),
    ).validate((proposal,))
    second = MODULE.IndependentCandidateValidator(
        allocator=FakeAllocator(),
        evidence=lambda: evidence,
        parent_cache=cache,
        output_directory=output,
        max_candidates=1,
        discovery_row_fingerprints=("5" * 64, "6" * 64),
    ).validate((proposal,))

    assert first["round_index"] == 1
    assert second["round_index"] == 2
    assert first["round_id"] != second["round_id"]
    assert (output / "round-000001" / "validation_summary.json").is_file()
    assert (output / "round-000002" / "validation_summary.json").is_file()


def test_candidate_validation_import_verifies_full_source_and_reuses_prefix_once(
    tmp_path: Path,
) -> None:
    parent_state = "1" * 64
    proposals = tuple(
        SimpleNamespace(
            kind="neuron",
            candidate=SimpleNamespace(candidate_sha256=str(index) * 64),
            individual_physical_bytes_saved=100,
            prototype_bytes_saved=100,
            scale_setup_bytes=0,
            risk=0.5,
            proposal_sha256=str(index + 2) * 64,
        )
        for index in (2, 3)
    )
    evidence = SimpleNamespace(
        fingerprints=SimpleNamespace(
            model="0" * 64,
            runtime="1" * 64,
            measurement="2" * 64,
            sketch_basis="3" * 64,
            cache="4" * 64,
            calibration_row_fingerprints=("5" * 64,),
            gate_row_fingerprints=("6" * 64,),
        )
    )

    class FakeAllocator:
        runtime = torch.nn.Identity()

        def __init__(self, state: str = parent_state) -> None:
            self.state = state
            self.installed = ()
            self.apply_calls = 0

        def _state_sha256(self):
            return self.state

        def accounting(self):
            return _accounting()

        def apply_batch(self, _batch, *, evaluator, **_kwargs):
            self.apply_calls += 1
            decision = evaluator(
                SimpleNamespace(runtime=object(), proposed_accounting=_accounting())
            )
            assert decision.accepted is False
            return SimpleNamespace(accepted=False, rollback_exact=True)

        def install_candidate_validation(self, records):
            self.installed = tuple(records)

    suite = _bound_suite(prompt_prefix="validation")
    source_allocator = FakeAllocator()
    source = MODULE.IndependentCandidateValidator(
        allocator=source_allocator,
        evidence=lambda: evidence,
        parent_cache=MODULE.CachedParentSuite(
            suite=suite,
            executor=MODULE.SuiteExecutor(
                tokenizer=object(),
                device="cpu",
                max_new_tokens=1,
                evaluator=_fake_evaluator(
                    _report(behavior_correct=9, prompt_prefix="validation")
                ),
            ),
            output_directory=tmp_path / "source_parent_cache",
        ),
        output_directory=tmp_path / "source_validation",
        max_candidates=2,
        discovery_row_fingerprints=("5" * 64, "6" * 64),
    ).validate(proposals)
    source_path = Path(source["summary_artifact"])
    source_sha256 = MODULE.production_runner._file_sha256(source_path)

    def no_forward(*_args, **_kwargs):
        raise AssertionError("import must not execute parent or candidate forwards")

    imported_allocator = FakeAllocator()
    imported = MODULE.IndependentCandidateValidator(
        allocator=imported_allocator,
        evidence=lambda: evidence,
        parent_cache=MODULE.CachedParentSuite(
            suite=suite,
            executor=MODULE.SuiteExecutor(
                tokenizer=object(),
                device="cpu",
                max_new_tokens=1,
                evaluator=no_forward,
            ),
            output_directory=tmp_path / "import_parent_cache",
        ),
        output_directory=tmp_path / "import_validation",
        max_candidates=1,
        discovery_row_fingerprints=("5" * 64, "6" * 64),
        import_summary=source_path,
        import_summary_sha256=source_sha256,
    ).validate(proposals)

    assert source_allocator.apply_calls == 2
    assert imported_allocator.apply_calls == 0
    assert len(imported_allocator.installed) == 1
    assert imported["imported"] is True
    assert imported["imported_count"] == 1
    assert imported["source_count"] == 2
    assert imported["parent_forward_evaluations_for_round"] == 0

    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    Path(source_payload["audits"][1]["artifact"]).write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="report hash"):
        MODULE.IndependentCandidateValidator(
            allocator=FakeAllocator(),
            evidence=lambda: evidence,
            parent_cache=MODULE.CachedParentSuite(
                suite=suite,
                executor=MODULE.SuiteExecutor(
                    tokenizer=object(),
                    device="cpu",
                    max_new_tokens=1,
                    evaluator=no_forward,
                ),
                output_directory=tmp_path / "tampered_tail_cache",
            ),
            output_directory=tmp_path / "tampered_tail",
            max_candidates=1,
            discovery_row_fingerprints=("5" * 64, "6" * 64),
            import_summary=source_path,
            import_summary_sha256=source_sha256,
        ).validate(proposals)


def test_candidate_validation_import_rejects_tampering_and_parent_mismatch(
    tmp_path: Path,
) -> None:
    parent_state = "1" * 64
    proposal = SimpleNamespace(
        kind="neuron",
        candidate=SimpleNamespace(candidate_sha256="2" * 64),
        individual_physical_bytes_saved=100,
        prototype_bytes_saved=100,
        scale_setup_bytes=0,
        risk=0.5,
        proposal_sha256="3" * 64,
    )
    evidence = SimpleNamespace(
        fingerprints=SimpleNamespace(
            cache="4" * 64,
            calibration_row_fingerprints=("5" * 64,),
            gate_row_fingerprints=("6" * 64,),
        )
    )

    class FakeAllocator:
        runtime = torch.nn.Identity()

        def __init__(self, state: str) -> None:
            self.state = state

        def _state_sha256(self):
            return self.state

        def accounting(self):
            return _accounting()

        def apply_batch(self, _batch, *, evaluator, **_kwargs):
            evaluator(SimpleNamespace(runtime=object(), proposed_accounting=_accounting()))
            return SimpleNamespace(accepted=False, rollback_exact=True)

        def install_candidate_validation(self, _records):
            return None

    suite = _bound_suite(prompt_prefix="validation")
    source = MODULE.IndependentCandidateValidator(
        allocator=FakeAllocator(parent_state),
        evidence=lambda: evidence,
        parent_cache=MODULE.CachedParentSuite(
            suite=suite,
            executor=MODULE.SuiteExecutor(
                tokenizer=object(),
                device="cpu",
                max_new_tokens=1,
                evaluator=_fake_evaluator(_report(prompt_prefix="validation")),
            ),
            output_directory=tmp_path / "source_parent_cache",
        ),
        output_directory=tmp_path / "source_validation",
        max_candidates=1,
        discovery_row_fingerprints=("5" * 64, "6" * 64),
    ).validate((proposal,))
    source_path = Path(source["summary_artifact"])
    source_sha256 = MODULE.production_runner._file_sha256(source_path)

    def importer(state: str, output: str):
        return MODULE.IndependentCandidateValidator(
            allocator=FakeAllocator(state),
            evidence=lambda: evidence,
            parent_cache=MODULE.CachedParentSuite(
                suite=suite,
                executor=MODULE.SuiteExecutor(
                    tokenizer=object(),
                    device="cpu",
                    max_new_tokens=1,
                    evaluator=lambda *_args, **_kwargs: None,
                ),
                output_directory=tmp_path / f"{output}_cache",
            ),
            output_directory=tmp_path / output,
            max_candidates=1,
            discovery_row_fingerprints=("5" * 64, "6" * 64),
            import_summary=source_path,
            import_summary_sha256=source_sha256,
        )

    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="exact parent"):
        importer("9" * 64, "stale").validate((proposal,))

    summary = json.loads(source_path.read_text(encoding="utf-8"))
    artifact = Path(summary["audits"][0]["artifact"])
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="report hash"):
        importer(parent_state, "tampered").validate((proposal,))


def test_failed_raw_round_salvage_is_zero_forward_and_truth_labeled(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "failed-source-run"
    source_round_root = source_run / "candidate_validation"
    parent_state = "1" * 64
    runtime, tokenizer = _tiny_runtime(seed=955)
    proposals = tuple(
        SimpleNamespace(
            kind="neuron",
            candidate=SimpleNamespace(candidate_sha256=str(index) * 64),
            individual_physical_bytes_saved=100,
            prototype_bytes_saved=100,
            scale_setup_bytes=0,
            risk=0.5,
            proposal_sha256=str(index + 4) * 64,
        )
        for index in (2, 3)
    )
    evidence = SimpleNamespace(
        fingerprints=SimpleNamespace(
            model="0" * 64,
            runtime="1" * 64,
            measurement="2" * 64,
            sketch_basis="3" * 64,
            cache="4" * 64,
            calibration_row_fingerprints=("5" * 64,),
            gate_row_fingerprints=("6" * 64,),
        )
    )

    class FakeAllocator:
        def __init__(self, state: str = parent_state) -> None:
            self.runtime = runtime
            self.state = state
            self.installed = ()
            self.apply_calls = 0

        def _state_sha256(self):
            return self.state

        def accounting(self):
            return _artifact_accounting(runtime, tokenizer)

        def apply_batch(self, _batch, *, evaluator, **_kwargs):
            self.apply_calls += 1
            decision = evaluator(
                SimpleNamespace(
                    runtime=object(),
                    proposed_accounting=_artifact_accounting(runtime, tokenizer),
                )
            )
            assert decision.accepted is False
            return SimpleNamespace(accepted=False, rollback_exact=True)

        def install_candidate_validation(self, records):
            self.installed = tuple(records)

    suite = _bound_suite(prompt_prefix="validation")
    source_allocator = FakeAllocator()
    source_summary = MODULE.IndependentCandidateValidator(
        allocator=source_allocator,
        evidence=lambda: evidence,
        parent_cache=MODULE.CachedParentSuite(
            suite=suite,
            executor=MODULE.SuiteExecutor(
                tokenizer=object(),
                device="cpu",
                max_new_tokens=1,
                evaluator=_labeled_evaluator(
                    _report(behavior_correct=9, prompt_prefix="validation")
                ),
            ),
            output_directory=source_run / "validation_parent_cache",
        ),
        output_directory=source_round_root,
        max_candidates=2,
        discovery_row_fingerprints=("5" * 64, "6" * 64),
    ).validate(proposals)
    source_summary_path = Path(source_summary["summary_artifact"])
    source_round = source_summary_path.parent
    source_summary_path.unlink()

    content = MODULE._model_content_identity(runtime)
    source_manifest = MODULE._implementation_source_manifest()
    deterministic_execution = MODULE._configure_deterministic_execution()
    source_contract = {
        "schema": MODULE.RUN_CONTRACT_SCHEMA,
        "candidate_validation_mode": "independent",
        "candidate_validation_suite_sha256": _digest("suite-file"),
        "split_provenance_sha256": _digest("split"),
        "tokenizer_identity": {"tokenizer_sha256": _digest("tokenizer")},
        "evaluator_identity": {"evaluator": "test"},
        "calibration_file_sha256": _digest("calibration"),
        "sensitivity_gate_file_sha256": _digest("sensitivity"),
        "initial_model_content_sha256": content["model_content_sha256"],
        "implementation_source_manifest": source_manifest,
        "deterministic_execution": deterministic_execution,
    }
    contract_sha256 = MODULE._canonical_sha256(source_contract)
    source_contract["binding_sha256"] = contract_sha256
    source_contract["run_contract_sha256"] = contract_sha256
    MODULE.AcceptedCommitJournal(source_run).record_or_verify_initial(
        model_content=content,
        accounting=_artifact_accounting(runtime, tokenizer),
        run_contract=source_contract,
        source_identity={"kind": "test"},
    )
    MODULE.production_runner._write_json(
        source_run / "run_report.json",
        {
            "schema": MODULE.RUN_SCHEMA,
            "status": "failed",
            "accepted_transactions_total": 0,
            "current_commit_seq": 0,
            "current_checkpoint": None,
            "error": (
                "Qwen3SurgeryAllocatorStateError: Candidate validation discovery "
                "provenance does not match evidence."
            ),
        },
    )
    (source_run / "checkpoints").mkdir()
    discovery = MODULE._candidate_discovery_manifest(
        proposals,
        attempt_total=1,
        parent_state_sha256=parent_state,
        evidence=evidence,
    )
    discovery_path = source_run / "candidate_discovery" / "attempt-000001.json"
    MODULE._write_json_exclusive_atomic(discovery_path, discovery)
    import_contract = MODULE._candidate_validation_import_contract(source_round)

    def no_forward(*_args, **_kwargs):
        raise AssertionError("failed-round salvage must not execute forwards")

    rebound_parent_state = "7" * 64
    target_allocator = FakeAllocator(rebound_parent_state)
    salvaged = MODULE.IndependentCandidateValidator(
        allocator=target_allocator,
        evidence=lambda: evidence,
        parent_cache=MODULE.CachedParentSuite(
            suite=suite,
            executor=MODULE.SuiteExecutor(
                tokenizer=object(),
                device="cpu",
                max_new_tokens=1,
                evaluator=no_forward,
            ),
            output_directory=tmp_path / "target_parent_cache",
        ),
        output_directory=tmp_path / "target_validation",
        max_candidates=1,
        discovery_row_fingerprints=("5" * 64, "6" * 64),
        import_summary=source_round,
        import_contract=import_contract,
        current_run_contract=source_contract,
    ).validate(tuple(reversed(proposals)))

    assert source_allocator.apply_calls == 2
    assert target_allocator.apply_calls == 0
    assert len(target_allocator.installed) == 1
    assert target_allocator.installed[0].candidate_sha256 == "3" * 64
    assert salvaged["import_class"] == "failed-round-artifact-salvage"
    assert salvaged["source_count"] == 2
    assert salvaged["source_parent_state_sha256"] == parent_state
    assert salvaged["parent_state_sha256"] == rebound_parent_state
    assert salvaged["parent_state_rebound_across_process"] is True
    assert salvaged["every_probe_rolled_back_exactly"] is None
    assert salvaged["audits"][0]["rollback_exact"] is None
    limitation = salvaged["import_provenance"][
        "parent_link_inference_limitation"
    ]
    assert "source failed before validation_summary.json" in limitation
    assert MODULE._semantic_candidate_validation_commit_proof_is_valid(
        Path(salvaged["summary_artifact"]),
        transaction={
            "proposal_sha256s": [
                salvaged["audits"][0]["validated_proposal_sha256"]
            ]
        },
        gate_parent_state_sha256=rebound_parent_state,
        output_directory=tmp_path,
        run_contract={"candidate_validation_import": import_contract},
    )

    second_source_candidate = sorted(source_round.glob("candidate-*.json"))[1]
    second_source_candidate.write_text("{}", encoding="utf-8")
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="tree changed"):
        MODULE.IndependentCandidateValidator(
            allocator=FakeAllocator(),
            evidence=lambda: evidence,
            parent_cache=MODULE.CachedParentSuite(
                suite=suite,
                executor=MODULE.SuiteExecutor(
                    tokenizer=object(),
                    device="cpu",
                    max_new_tokens=1,
                    evaluator=no_forward,
                ),
                output_directory=tmp_path / "tampered_raw_cache",
            ),
            output_directory=tmp_path / "tampered_raw",
            max_candidates=1,
            discovery_row_fingerprints=("5" * 64, "6" * 64),
            import_summary=source_round,
            import_contract=import_contract,
            current_run_contract=source_contract,
        ).validate(proposals)


def test_failed_round_salvage_never_becomes_completed_candidate_marginal_claim() -> None:
    assert MODULE._candidate_marginal_truth_label(
        "independent",
        1,
        ("failed-round-artifact-salvage",),
    ) == (
        False,
        "failed-round-artifact-salvage-with-independent-transaction-gate",
        True,
    )
    assert MODULE._candidate_marginal_truth_label(
        "independent",
        1,
        ("completed-independent-candidate-validation",),
    ) == (True, "completed-independent-candidate-validation", False)


def test_missing_newest_checkpoint_replays_last_verified_commit(
    tmp_path: Path,
) -> None:
    runtime, _tokenizer, manager, journal, first, contract, gate = (
        _journal_with_commit(tmp_path)
    )
    with torch.no_grad():
        next(runtime.parameters()).add_(0.125)
    second_accounting = _artifact_accounting(runtime)
    manager.record(
        runtime,
        accepted_transaction_total=2,
        state_sha256=_digest("ephemeral-two"),
        accounting=second_accounting,
        behavior_score=1.0,
        transaction_sha256=_digest("transaction-two"),
    )
    frontier = manager.current_record
    assert frontier is not None
    transaction_two, gate_two, score_two, _parent_state, _proposal = (
        _accepted_transaction_proof(
            tmp_path,
            "transaction-two",
            parent_accounting=first["accounting"],
            committed_accounting=second_accounting.as_dict(),
            transaction_index=2,
        )
    )
    second = journal.append(
        parent_commit_seq=first["seq"],
        parent_model_content_sha256=first["model_content_sha256"],
        accepted_transaction_total=2,
        model_content_sha256=frontier.model_content_sha256,
        checkpoint=Path(frontier.checkpoint),
        artifact_tree={
            "tree_sha256": frontier.artifact_tree_sha256,
            "total_bytes": frontier.artifact_total_bytes,
        },
        accounting=second_accounting,
        run_contract_sha256=contract["run_contract_sha256"],
        transaction_payload=transaction_two,
        gate_artifact=gate_two,
        candidate_validation_artifact=None,
        candidate_validation_round_id=None,
        evidence_lineage=_lineage("two"),
        behavior_score=score_two,
    )
    manager.finalize_pending_removals()
    assert Path(first["checkpoint"]).is_dir()
    shutil.rmtree(second["checkpoint"])

    replay = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )

    assert replay.selected is not None
    assert replay.selected["seq"] == first["seq"]
    assert replay.accepted_transaction_total == 1
    assert replay.fallback_reason is not None

    recovered = MODULE.load_qwen3_codebook_checkpoint(
        Path(first["checkpoint"]), device="cpu"
    )
    with torch.no_grad():
        next(recovered.parameters()).sub_(0.25)
    recovered_manager = MODULE.ParetoCheckpointManager(
        output_directory=tmp_path,
        tokenizer=_Tokenizer(),
        tokenizer_name="tiny",
        max_checkpoints=3,
        verify_reload=True,
    )
    recovered_manager.reconcile_from_journal(replay)
    recovered_accounting = _artifact_accounting(recovered)
    recovered_manager.record(
        recovered,
        accepted_transaction_total=2,
        state_sha256=_digest("recovered-ephemeral-two"),
        accounting=recovered_accounting,
        behavior_score=1.0,
        transaction_sha256=_digest("recovered-transaction-two"),
    )
    recovered_frontier = recovered_manager.current_record
    assert recovered_frontier is not None
    recovery_transaction, recovery_gate, recovery_score, _parent_state, _proposal = (
        _accepted_transaction_proof(
            tmp_path,
            "recovered-transaction-two",
            parent_accounting=first["accounting"],
            committed_accounting=recovered_accounting.as_dict(),
            transaction_index=2,
        )
    )
    recovery_commit = journal.append(
        parent_commit_seq=first["seq"],
        parent_model_content_sha256=first["model_content_sha256"],
        accepted_transaction_total=2,
        model_content_sha256=recovered_frontier.model_content_sha256,
        checkpoint=Path(recovered_frontier.checkpoint),
        artifact_tree={
            "tree_sha256": recovered_frontier.artifact_tree_sha256,
            "total_bytes": recovered_frontier.artifact_total_bytes,
        },
        accounting=recovered_accounting,
        run_contract_sha256=contract["run_contract_sha256"],
        transaction_payload=recovery_transaction,
        gate_artifact=recovery_gate,
        candidate_validation_artifact=None,
        candidate_validation_round_id=None,
        evidence_lineage=_lineage("recovered-two"),
        behavior_score=recovery_score,
    )
    replay_after_recovery = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )
    assert recovery_commit["seq"] == 3
    assert replay_after_recovery.selected["seq"] == 3
    assert replay_after_recovery.accepted_transaction_total == 2


def test_corrupt_journal_is_rejected_but_corrupt_frontier_rebuilds(
    tmp_path: Path,
) -> None:
    _runtime, tokenizer, _manager, journal, commit, contract, _gate = (
        _journal_with_commit(tmp_path)
    )
    manifest_path = tmp_path / "frontier_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_checkpoint"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rebuilt = MODULE.ParetoCheckpointManager(
        output_directory=tmp_path,
        tokenizer=tokenizer,
        tokenizer_name="tiny",
        max_checkpoints=3,
        verify_reload=True,
    )
    assert rebuilt.manifest_validation_error is not None
    replay = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )
    rebuilt.reconcile_from_journal(replay)
    assert rebuilt.current_checkpoint == commit["checkpoint"]
    assert rebuilt.manifest_validation_error is None

    journal_path = tmp_path / "accepted_commit_journal" / "commit-000001.json"
    corrupted = json.loads(journal_path.read_text(encoding="utf-8"))
    corrupted["behavior_score"] = 0.0
    journal_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="integrity"):
        journal.replay()


def test_initial_incumbent_and_evidence_lineage_are_immutable(
    tmp_path: Path,
) -> None:
    runtime, _tokenizer, _manager, journal, commit, contract, _gate = (
        _journal_with_commit(tmp_path)
    )
    replay = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )
    assert replay.selected is not None
    assert replay.selected["evidence_lineage"] == _lineage("one")
    assert replay.initial["model_content_sha256"] == MODULE._model_content_identity(
        runtime
    )["model_content_sha256"]
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="policy differs"):
        journal.replay(expected_run_contract_sha256=_digest("drifted-policy"))

    with torch.no_grad():
        next(runtime.parameters()).mul_(0.0)
    with pytest.raises(
        MODULE.QwenUnifiedSurgeryRunnerError, match="Initial incumbent differs"
    ):
        journal.record_or_verify_initial(
            model_content=MODULE._model_content_identity(runtime),
            accounting=_accounting(10_000),
            run_contract=contract,
            source_identity={"kind": "test"},
        )


def test_independent_journal_replay_requires_every_durable_proof(
    tmp_path: Path,
) -> None:
    _runtime, _tokenizer, _manager, journal, commit, contract, gate = (
        _journal_with_commit(
            tmp_path, candidate_validation_mode="independent"
        )
    )
    journal_path = tmp_path / "accepted_commit_journal" / "commit-000001.json"
    original_record = json.loads(journal_path.read_text(encoding="utf-8"))
    validation_path = Path(commit["candidate_validation_artifact"])
    validation_bytes = validation_path.read_bytes()

    tampered = copy.deepcopy(original_record)
    tampered["transaction_payload"]["name"] = "tampered"
    tampered["record_sha256"] = MODULE._record_sha256(
        tampered, "record_sha256"
    )
    journal_path.write_text(json.dumps(tampered), encoding="utf-8")
    replay = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )
    assert replay.selected is None

    journal_path.write_text(json.dumps(original_record), encoding="utf-8")
    validation_path.unlink()
    replay = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )
    assert replay.selected is None

    validation_path.write_bytes(validation_bytes)
    gate.unlink()
    replay = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )
    assert replay.selected is None


def test_run_contract_binds_policy_evaluator_tokenizer_and_limits(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.jsonl"
    sensitivity = tmp_path / "sensitivity.jsonl"
    validation = tmp_path / "validation.json"
    transaction = tmp_path / "transaction.json"
    for path in (calibration, sensitivity, validation, transaction):
        path.write_text("{}\n", encoding="utf-8")
    common = [
        "--source-model",
        "qwen",
        "--output",
        str(tmp_path / "out"),
        "--target-savings-bytes",
        "42",
        "--calibration-file",
        str(calibration),
        "--sensitivity-gate-file",
        str(sensitivity),
        "--candidate-validation-suite",
        str(validation),
        "--transaction-gate-suite",
        str(transaction),
    ]
    args = MODULE.build_parser().parse_args(common)
    provenance = {"provenance_sha256": _digest("splits")}

    def bind(value, tokenizer_value="tokenizer-a"):
        return MODULE._run_binding(
            value,
            provenance,
            resolved_target_deployable_bytes=9_958,
            initial_deployable_bytes=10_000,
            tokenizer_identity={"tokenizer_sha256": _digest(tokenizer_value)},
            initial_model_content_sha256=_digest("initial-model"),
            source_identity={"kind": "test", "sha256": _digest("source")},
        )

    baseline = bind(args)
    for name, value in (
        ("min_behavior_retention", 0.91),
        ("sensitivity_sketch_dim", 17),
        ("max_batch_size", 7),
        ("max_new_tokens", 95),
        ("max_collector_bytes", 123456),
    ):
        changed = argparse.Namespace(**copy.deepcopy(vars(args)))
        setattr(changed, name, value)
        assert bind(changed)["run_contract_sha256"] != baseline[
            "run_contract_sha256"
        ]
    assert bind(args, "tokenizer-b")["run_contract_sha256"] != baseline[
        "run_contract_sha256"
    ]


def test_run_contract_binds_full_source_manifest_and_runtime_identity(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    package = source_root / "src" / "neural_forge"
    experiments = source_root / "experiments"
    package.mkdir(parents=True)
    experiments.mkdir()
    implementation = package / "implementation.py"
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    for name in (
        "run_qwen_unified_surgery.py",
        "run_qwen_codebook_in_memory.py",
        "run_qwen_compensated_neuron_fold.py",
        "prepare_qwen_unified_splits.py",
    ):
        (experiments / name).write_text(f"# {name}\n", encoding="utf-8")
    first_manifest = MODULE._implementation_source_manifest(source_root)
    implementation.write_text("VALUE = 2\n", encoding="utf-8")
    second_manifest = MODULE._implementation_source_manifest(source_root)
    assert first_manifest["manifest_sha256"] != second_manifest["manifest_sha256"]

    calibration = tmp_path / "calibration.jsonl"
    sensitivity = tmp_path / "sensitivity.jsonl"
    validation = tmp_path / "validation.json"
    transaction = tmp_path / "transaction.json"
    for path in (calibration, sensitivity, validation, transaction):
        path.write_text("{}\n", encoding="utf-8")
    args = MODULE.build_parser().parse_args(
        [
            "--source-model",
            "qwen",
            "--output",
            str(tmp_path / "out"),
            "--target-savings-bytes",
            "42",
            "--calibration-file",
            str(calibration),
            "--sensitivity-gate-file",
            str(sensitivity),
            "--candidate-validation-suite",
            str(validation),
            "--transaction-gate-suite",
            str(transaction),
        ]
    )
    kwargs = {
        "resolved_target_deployable_bytes": 9_958,
        "initial_deployable_bytes": 10_000,
        "tokenizer_identity": {"tokenizer_sha256": _digest("tokenizer")},
        "initial_model_content_sha256": _digest("initial-model"),
        "source_identity": {"kind": "test", "sha256": _digest("source")},
    }
    provenance = {"provenance_sha256": _digest("splits")}
    first = MODULE._run_binding(
        args,
        provenance,
        **kwargs,
        implementation_source_manifest=first_manifest,
        runtime_environment_identity=_runtime_identity("runtime-a"),
    )
    source_drift = MODULE._run_binding(
        args,
        provenance,
        **kwargs,
        implementation_source_manifest=second_manifest,
        runtime_environment_identity=_runtime_identity("runtime-a"),
    )
    runtime_drift = MODULE._run_binding(
        args,
        provenance,
        **kwargs,
        implementation_source_manifest=first_manifest,
        runtime_environment_identity=_runtime_identity("runtime-b"),
    )
    assert first["run_contract_sha256"] != source_drift["run_contract_sha256"]
    assert first["run_contract_sha256"] != runtime_drift["run_contract_sha256"]

    import_run = tmp_path / "import-run"
    import_round = import_run / "candidate_validation" / "round-000001"
    import_round.mkdir(parents=True)
    (import_run / "initial_incumbent.json").write_text("{}\n", encoding="utf-8")
    (import_round / "round_claim.json").write_text("{}\n", encoding="utf-8")
    imported_summary = import_round / "validation_summary.json"
    imported_summary.write_text('{"round": 1}\n', encoding="utf-8")
    import_args = argparse.Namespace(**copy.deepcopy(vars(args)))
    import_args.import_candidate_validation_summary = imported_summary
    imported = MODULE._run_binding(
        import_args,
        provenance,
        **kwargs,
        implementation_source_manifest=first_manifest,
        runtime_environment_identity=_runtime_identity("runtime-a"),
    )
    first_import_hash = MODULE.production_runner._file_sha256(imported_summary)
    assert imported["import_candidate_validation_summary"] == str(
        imported_summary.resolve()
    )
    assert imported["import_candidate_validation_summary_sha256"] == first_import_hash
    imported_summary.write_text('{"round": 2}\n', encoding="utf-8")
    import_drift = MODULE._run_binding(
        import_args,
        provenance,
        **kwargs,
        implementation_source_manifest=first_manifest,
        runtime_environment_identity=_runtime_identity("runtime-a"),
    )
    assert imported["run_contract_sha256"] != import_drift["run_contract_sha256"]


def test_runtime_identity_records_cudnn_probe_failure(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(MODULE.torch.backends.cudnn, "is_available", lambda: True)

    def fail_version():
        raise RuntimeError("heterogeneous GPU incompatibility")

    monkeypatch.setattr(MODULE.torch.backends.cudnn, "version", fail_version)

    identity = MODULE._runtime_environment_identity("cpu")

    assert identity["torch"]["cudnn"] is None
    assert identity["torch"]["cudnn_probe_error"] == (
        "RuntimeError: heterogeneous GPU incompatibility"
    )
    MODULE._assert_runtime_environment_identity(identity)


def test_global_attempt_ceiling_survives_resume(tmp_path: Path) -> None:
    class FakeAllocator:
        def __init__(self, *, fail_discovery: bool = False) -> None:
            self.discovery_calls = 0
            self.fail_discovery = fail_discovery

        def accounting(self):
            return _accounting(10_000)

        def _state_sha256(self):
            return _digest("parent")

        def discover_proposals(self, **_kwargs):
            self.discovery_calls += 1
            if self.fail_discovery:
                raise RuntimeError("simulated process failure after durable claim")
            raise AssertionError("global attempt ceiling must stop before discovery")

    report_path = tmp_path / "run_report.json"
    first_allocator = FakeAllocator(fail_discovery=True)
    with pytest.raises(RuntimeError, match="simulated process failure"):
        MODULE.execute_unified_surgery_loop(
            runtime=torch.nn.Identity(),
            allocator=first_allocator,
            config=MODULE.UnifiedLoopConfig(
                target_deployable_bytes=9_000,
                candidate_validation_mode="provisional",
                max_attempts=1,
            ),
            gate=SimpleNamespace(bind_parent=lambda *_args: None),
            refresh_evidence=lambda _context: None,
            current_evidence=lambda: None,
            checkpoint_manager=SimpleNamespace(),
            report_path=report_path,
            ledger_path=tmp_path / "transactions.jsonl",
            candidate_validator=None,
        )
    claimed = json.loads(report_path.read_text(encoding="utf-8"))
    assert claimed["attempts_total"] == 1

    allocator = FakeAllocator()
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="global max_attempts"):
        MODULE.execute_unified_surgery_loop(
            runtime=torch.nn.Identity(),
            allocator=allocator,
            config=MODULE.UnifiedLoopConfig(
                target_deployable_bytes=9_000,
                candidate_validation_mode="provisional",
                max_attempts=1,
            ),
            gate=SimpleNamespace(),
            refresh_evidence=lambda _context: None,
            current_evidence=lambda: None,
            checkpoint_manager=SimpleNamespace(),
            report_path=report_path,
            ledger_path=tmp_path / "transactions.jsonl",
            candidate_validator=None,
            resume_report=claimed,
        )
    assert allocator.discovery_calls == 0


def test_journal_replay_rejects_rehashed_semantically_failed_gate(
    tmp_path: Path,
) -> None:
    _runtime, _tokenizer, _manager, journal, commit, contract, gate = (
        _journal_with_commit(tmp_path)
    )
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    gate_payload["verdict"]["passed"] = False
    gate.write_text(json.dumps(gate_payload), encoding="utf-8")
    journal_path = tmp_path / "accepted_commit_journal" / "commit-000001.json"
    record = json.loads(journal_path.read_text(encoding="utf-8"))
    record["gate_artifact_sha256"] = MODULE.production_runner._file_sha256(gate)
    record["record_sha256"] = MODULE._record_sha256(record, "record_sha256")
    journal_path.write_text(json.dumps(record), encoding="utf-8")

    replay = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )
    assert replay.selected is None
    assert replay.fallback_reason is not None


def test_independent_journal_rejects_rehashed_validation_parent_mismatch(
    tmp_path: Path,
) -> None:
    _runtime, _tokenizer, _manager, journal, commit, contract, _gate = (
        _journal_with_commit(tmp_path, candidate_validation_mode="independent")
    )
    summary_path = Path(commit["candidate_validation_artifact"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["parent_state_sha256"] = _digest("wrong-parent")
    summary["summary_sha256"] = MODULE._record_sha256(summary, "summary_sha256")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    journal_path = tmp_path / "accepted_commit_journal" / "commit-000001.json"
    record = json.loads(journal_path.read_text(encoding="utf-8"))
    record["candidate_validation_artifact_sha256"] = (
        MODULE.production_runner._file_sha256(summary_path)
    )
    record["record_sha256"] = MODULE._record_sha256(record, "record_sha256")
    journal_path.write_text(json.dumps(record), encoding="utf-8")

    replay = journal.replay(
        expected_run_contract_sha256=contract["run_contract_sha256"]
    )
    assert replay.selected is None


def test_stale_output_and_no_resume_reuse_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="--no-resume"):
        MODULE._prepare_run_output(output, resume=False)
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="no durable"):
        MODULE._prepare_run_output(output, resume=True)


def test_vacuous_baseline_is_rejected(tmp_path: Path) -> None:
    suite = _bound_suite()
    cache = MODULE.CachedParentSuite(
        suite=suite,
        executor=MODULE.SuiteExecutor(
            tokenizer=object(),
            device="cpu",
            max_new_tokens=1,
            evaluator=_fake_evaluator(_report()),
        ),
        output_directory=tmp_path / "cache",
    )
    parent_state = "a" * 64
    cache.store_parent(
        parent_state, _report(behavior_correct=0), source="vacuous-test"
    )
    gate = MODULE.ParentRelativeBehavioralGate(
        parent_cache=cache,
        retention_floors=MODULE.RetainedParentFloors(),
        hard_loss_limits=(None, None, None),
        output_directory=tmp_path / "gates",
    )
    with pytest.raises(MODULE.QwenUnifiedSurgeryRunnerError, match="vacuous"):
        gate.bind_parent(torch.nn.Identity(), parent_state, _accounting())


def test_independent_mode_makes_no_claim_when_target_needs_no_surgery(
    tmp_path: Path,
) -> None:
    runtime, tokenizer = _tiny_runtime(seed=944)
    artifact_accounting = _artifact_accounting(runtime)
    suite = _bound_suite()
    executor = MODULE.SuiteExecutor(
        tokenizer=object(),
        device="cpu",
        max_new_tokens=1,
        evaluator=_fake_evaluator(_report()),
    )
    cache = MODULE.CachedParentSuite(
        suite=suite,
        executor=executor,
        output_directory=tmp_path / "cache",
    )
    state_sha = _digest("no-surgery-state")
    cache.store_parent(state_sha, _report(), source="test")
    gate = MODULE.ParentRelativeBehavioralGate(
        parent_cache=cache,
        retention_floors=MODULE.RetainedParentFloors(),
        hard_loss_limits=(None, None, None),
        output_directory=tmp_path / "gates",
    )

    class FakeAllocator:
        def accounting(self):
            return artifact_accounting

        def _state_sha256(self):
            return state_sha

    result = MODULE.execute_unified_surgery_loop(
        runtime=runtime,
        allocator=FakeAllocator(),
        config=MODULE.UnifiedLoopConfig(
            target_deployable_bytes=artifact_accounting.deployable_bytes,
            candidate_validation_mode="independent",
        ),
        gate=gate,
        refresh_evidence=lambda _context: None,
        current_evidence=lambda: None,
        checkpoint_manager=MODULE.ParetoCheckpointManager(
            output_directory=tmp_path,
            tokenizer=tokenizer,
            tokenizer_name="tiny",
            max_checkpoints=1,
        ),
        report_path=tmp_path / "run_report.json",
        ledger_path=tmp_path / "transactions.jsonl",
        candidate_validator=SimpleNamespace(),
    )

    assert result["status"] == "target_reached_without_surgery"
    assert result["candidate_marginal_claim"] is False
    assert result["independent_validation_vacuous"] is True
