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


def _accepted_transaction_payload(name: str) -> dict:
    return {
        "accepted": True,
        "proposal_sha256s": [_digest(f"{name}:proposal")],
        "name": name,
    }


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
    gate = tmp_path / "gate.json"
    gate.write_text('{"passed": true}\n', encoding="utf-8")
    validation_artifact = None
    validation_round_id = None
    if candidate_validation_mode == "independent":
        validation_round_id = _digest("validation-round-one")
        validation_artifact = (
            tmp_path
            / "candidate_validation"
            / "round-000001"
            / "validation_summary.json"
        )
        validation_artifact.parent.mkdir(parents=True)
        validation_artifact.write_text(
            json.dumps(
                {
                    "round_id": validation_round_id,
                    "parent_state_sha256": _digest("parent-one"),
                    "audits": [],
                }
            ),
            encoding="utf-8",
        )
    manager.record(
        runtime,
        accepted_transaction_total=1,
        state_sha256=_digest("ephemeral-one"),
        accounting=artifact_accounting,
        behavior_score=0.99,
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
        transaction_payload=_accepted_transaction_payload("transaction-one"),
        gate_artifact=gate,
        candidate_validation_artifact=validation_artifact,
        candidate_validation_round_id=validation_round_id,
        evidence_lineage=_lineage("one"),
        behavior_score=0.99,
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


def test_missing_newest_checkpoint_replays_last_verified_commit(
    tmp_path: Path,
) -> None:
    runtime, _tokenizer, manager, journal, first, contract, gate = (
        _journal_with_commit(tmp_path)
    )
    with torch.no_grad():
        next(runtime.parameters()).add_(0.125)
    manager.record(
        runtime,
        accepted_transaction_total=2,
        state_sha256=_digest("ephemeral-two"),
        accounting=_artifact_accounting(runtime),
        behavior_score=1.0,
        transaction_sha256=_digest("transaction-two"),
    )
    frontier = manager.current_record
    assert frontier is not None
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
        accounting=_artifact_accounting(runtime),
        run_contract_sha256=contract["run_contract_sha256"],
        transaction_payload=_accepted_transaction_payload("transaction-two"),
        gate_artifact=gate,
        candidate_validation_artifact=None,
        candidate_validation_round_id=None,
        evidence_lineage=_lineage("two"),
        behavior_score=1.0,
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
    recovered_manager.record(
        recovered,
        accepted_transaction_total=2,
        state_sha256=_digest("recovered-ephemeral-two"),
        accounting=_artifact_accounting(recovered),
        behavior_score=1.0,
        transaction_sha256=_digest("recovered-transaction-two"),
    )
    recovered_frontier = recovered_manager.current_record
    assert recovered_frontier is not None
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
        accounting=_artifact_accounting(recovered),
        run_contract_sha256=contract["run_contract_sha256"],
        transaction_payload=_accepted_transaction_payload(
            "recovered-transaction-two"
        ),
        gate_artifact=gate,
        candidate_validation_artifact=None,
        candidate_validation_round_id=None,
        evidence_lineage=_lineage("recovered-two"),
        behavior_score=1.0,
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
