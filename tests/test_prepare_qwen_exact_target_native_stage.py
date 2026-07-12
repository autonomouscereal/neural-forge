from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from neural_forge.hf_compiled_child_cache import (
    CompilerIdentity,
    model_state_fingerprint,
    publish_compiled_child_cache,
)


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import prepare_qwen_exact_target_native_stage as builder  # noqa: E402
import run_qwen_exact_target_native_stage as runner  # noqa: E402


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"

    def __call__(self, text, **_kwargs):
        pieces = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text)
        return {
            "input_ids": [
                3 + int(hashlib.sha256(piece.encode()).hexdigest()[:7], 16) % 100_000
                for piece in pieces
            ]
        }


class _AuditCacheModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        destination.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), destination / "model.safetensors")
        (destination / "config.json").write_text(
            json.dumps({"model_type": "audit-cache"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, source: Path, *, local_files_only: bool = True):
        assert local_files_only is True
        model = cls()
        model.load_state_dict(
            load_file(source / "model.safetensors", device="cpu"), strict=True
        )
        return model


def _artifact(path: Path, files: dict[str, bytes]) -> Path:
    path.mkdir()
    for name, payload in files.items():
        (path / name).write_bytes(payload)
    return path


def _corpus(path: Path, count: int = 100) -> tuple[Path, str, int]:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "prompt": (
                            f"Natural corpus prompt identifier {index:04d} requests "
                            "a concise and context-sensitive answer."
                        )
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), count


def _write_json(path: Path, value) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_split(path: Path, role: str, count: int, offset: int) -> tuple[Path, str]:
    payload = {
        "schema": runner.SPLIT_SCHEMA,
        "split": role,
        "sealed_before_model_access": role == "sealed",
        "prompts": [
            {
                "id": f"{role}-{index}",
                "token_ids": [1, offset + index, offset + count + index],
            }
            for index in range(count)
        ],
    }
    return path, _write_json(path, payload)


def _full_threshold(count: int) -> dict[str, object]:
    return {
        "minimum_action_accuracy": 0.5,
        "minimum_exact_rollout_fraction": 0.25,
        "minimum_token_match_fraction": 0.25,
        "minimum_lcp_fraction": 0.25,
        "minimum_action_count": count * 8,
        "minimum_rollout_count": count,
        "maximum_repeat_fraction_excess": 0.2,
        "maximum_adjacent_repeat_fraction_excess": 0.25,
        "maximum_longest_run_excess": 2,
    }


def _audit_metrics(role: str, count: int, *, passing: bool = False) -> dict[str, object]:
    action_count = count * 8
    rollout_count = count
    rollout_tokens = count * 8
    agreement = action_count if passing else 0
    exact = rollout_count if passing else 0
    token_matches = rollout_tokens if passing else 0
    return {
        "split": role,
        "action_count": action_count,
        "action_agreement_count": agreement,
        "action_accuracy": agreement / action_count,
        "parent_action_mean_logprob": -1.0,
        "parent_action_topk_fraction": 1.0 if passing else 0.0,
        "parent_action_mean_rank": 1.0 if passing else 100.0,
        "parent_action_worst_rank": 1 if passing else 1000,
        "rollout_count": rollout_count,
        "exact_rollout_count": exact,
        "exact_rollout_fraction": exact / rollout_count,
        "rollout_token_count": rollout_tokens,
        "token_match_count": token_matches,
        "token_match_fraction": token_matches / rollout_tokens,
        "lcp_token_count": token_matches,
        "lcp_fraction": token_matches / rollout_tokens,
        "lcp_at_least_4_count": rollout_count if passing else 0,
        "repeat_fraction_excess": 0.0,
        "adjacent_repeat_fraction_excess": 0.0,
        "longest_run_excess": 0,
        "report_sha256": hashlib.sha256(f"metrics:{role}".encode()).hexdigest(),
    }


def _audit_source_bundle(
    tmp_path: Path,
    monkeypatch,
    *,
    candidate_count: int = 24,
    relative_bindings: bool = False,
    sealed_source: bool = True,
) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = _artifact(
        tmp_path / "audit-parent",
        {
            "config.json": b'{"model_type":"qwen3"}',
            "model.safetensors": b"parent-weight-bytes",
        },
    )
    target = _artifact(
        tmp_path / "audit-target",
        {
            "config.json": json.dumps(
                {
                    "model_type": "qwen3",
                    "vocab_size": runner.LEXICAL_VOCAB_SIZE,
                    "hidden_size": runner.LEXICAL_HIDDEN_SIZE,
                    "tie_word_embeddings": False,
                },
                sort_keys=True,
            ).encode(),
            "model.safetensors": b"target-weight-bytes",
        },
    )
    parent_weight_sha = hashlib.sha256(b"parent-weight-bytes").hexdigest()
    target_weight_sha = hashlib.sha256(b"target-weight-bytes").hexdigest()
    fit, fit_sha = _audit_split(tmp_path / "audit-fit.json", "fit", 120, 1000)
    candidate, candidate_sha = _audit_split(
        tmp_path / "audit-candidate.json", "candidate", candidate_count, 3000
    )
    sealed, sealed_sha = _audit_split(
        tmp_path / "audit-sealed.json", "sealed", 32, 5000
    )
    compiler_fit, compiler_fit_sha = _audit_split(
        tmp_path / "audit-compiler-fit.json", "fit", 24, 7000
    )
    thresholds = {
        "fit": _full_threshold(120),
        "candidate": _full_threshold(candidate_count),
        "sealed": _full_threshold(32),
    }
    compiler = {
        "capture_batch_size": 1,
        "max_prompt_tokens": 128,
        "canonical_validation_count": 4,
        "compiler_seed": 0,
        "bootstrap_kwargs": {
            "heldout_gate_policy": "output_only",
            "max_operator_delta_ratio": 0.1,
            "trace_pooling": "last_valid",
        },
        "candidate_split_used_during_bootstrap": False,
        "sealed_split_used_during_bootstrap": False,
    }
    training = {
        "steps": 48,
        "learning_rate": 1e-4,
        "fit_horizon": 8,
        "evaluation_horizon": 8,
        "selection_policy": "rollout-first-acquisition",
    }
    lexical_residual = {
        "rank": runner.LEXICAL_RANK,
        "vocab_size": runner.LEXICAL_VOCAB_SIZE,
        "hidden_size": runner.LEXICAL_HIDDEN_SIZE,
        "untied_required": True,
    }
    bound = lambda path: path.name if relative_bindings else str(path)  # noqa: E731
    cache_root = tmp_path / "audit-cache"
    cache_root.mkdir()
    cache_model = _AuditCacheModel()
    cache_model_state = model_state_fingerprint(cache_model)
    compiler_report = {
        "schema": "neural-forge-hf-parent-compiler-v1",
        "bootstrap_accepted": True,
        "target_budget_exact": True,
        "target_unique_parameter_count": cache_model_state["unique_parameter_count"],
        "target_logical_parameter_count": cache_model_state[
            "logical_parameter_count"
        ],
        "target_parameter_budget": cache_model_state["unique_parameter_count"],
        "target_state_sha256_after": cache_model_state["compiler_state_sha256"],
        "execution_graph_fingerprint": "8" * 64,
        "capability_proven": False,
    }
    parent_tree_sha = runner._artifact_tree(parent)["tree_sha256"]
    target_tree_sha = runner._artifact_tree(target)["tree_sha256"]
    cache_identity = CompilerIdentity(
        parent_artifact_tree_sha256=parent_tree_sha,
        blank_target_artifact_tree_sha256=target_tree_sha,
        tokenizer_sha256=hashlib.sha256(b"synthetic-tokenizer").hexdigest(),
        compiler_input_sha256s={
            "compiler_fit": compiler_fit_sha,
            "behavior_fit": fit_sha,
        },
        compiler_implementation_sha256s={
            "parent_compiler": "8" * 64,
            "blank_bootstrap": "9" * 64,
        },
        compiler_options=compiler,
        runtime={
            "dtype": "float16",
            "attention_implementation": runner.CANONICAL_ATTENTION_IMPLEMENTATION,
            "torch_version": str(torch.__version__),
        },
        seed=0,
    )
    cache_path = cache_root / cache_identity.compiler_identity_sha256
    cache_receipt = publish_compiled_child_cache(
        cache_path,
        model=cache_model,
        compiler_report=compiler_report,
        compiler_identity=cache_identity,
    )
    receipt_payload = cache_receipt.as_dict()
    cache_hashes = {
        "compiler_identity_sha256": cache_identity.compiler_identity_sha256,
        "receipt_sha256": cache_receipt.receipt_sha256,
        "compiler_report_sha256": receipt_payload["compiler_report"][
            "canonical_sha256"
        ],
        "state_dict_sha256": receipt_payload["model_state"]["state_dict_sha256"],
        "artifact_tree_sha256": receipt_payload["artifact_tree"]["tree_sha256"],
    }
    source_cache = {
        "root": bound(cache_root),
        "require_hit": True,
        **{f"expected_{name}": value for name, value in cache_hashes.items()},
    }
    historical_root = tmp_path / "historical-source"
    historical_root.mkdir()
    historical_files = {
        name: historical_root / f"{name}.py"
        for name in (
            "runner",
            "trainer",
            "compiler_cache",
            "lexical_adapter",
            "progressive_bridge",
            "qkvo_adapter",
            "swiglu_adapter",
        )
    }
    historical_hashes: dict[str, str] = {}
    for name, path in historical_files.items():
        path.write_text(f"# historical {name}\n", encoding="utf-8")
        historical_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_implementation = {
        "runner_sha256": historical_hashes["runner"],
        "trainer_module": runner.CANONICAL_TRAINER_MODULE,
        "trainer_sha256": historical_hashes["trainer"],
        "compiler_cache_sha256": historical_hashes["compiler_cache"],
        "lexical_adapter_sha256": historical_hashes["lexical_adapter"],
        "progressive_bridge_sha256": historical_hashes["progressive_bridge"],
        "qkvo_adapter_sha256": historical_hashes["qkvo_adapter"],
        "swiglu_adapter_sha256": historical_hashes["swiglu_adapter"],
    }
    source_preregistration = {
        "schema": runner.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": sealed_source,
        "seed": 0,
        "implementation": source_implementation,
        "parent": {
            "path": bound(parent),
            "exact_unique_parameters": runner.QWEN3_4B_EXACT_PARAMETERS,
            "required_files": {"model.safetensors": parent_weight_sha},
            "tree_sha256": parent_tree_sha,
        },
        "blank_target": {
            "path": bound(target),
            "exact_unique_parameters": runner.QWEN3_1P4B_EXACT_PARAMETERS,
            "required_files": {"model.safetensors": target_weight_sha},
            "tree_sha256": target_tree_sha,
        },
        "compiler_split": {
            "path": bound(compiler_fit),
            "sha256": compiler_fit_sha,
            "expected_count": 24,
        },
        "splits": {
            "fit": {"path": bound(fit), "sha256": fit_sha, "expected_count": 120},
            "candidate": {
                "path": bound(candidate),
                "sha256": candidate_sha,
                "expected_count": candidate_count,
            },
            "sealed": {
                "path": bound(sealed),
                "sha256": sealed_sha,
                "expected_count": 32,
            },
        },
        "thresholds": thresholds,
        "compiler": compiler,
        "training": training,
        "lexical_residual": lexical_residual,
        "compiler_cache": source_cache,
        "runtime": {
            "dtype": "float16",
            "attention_implementation": runner.CANONICAL_ATTENTION_IMPLEMENTATION,
            "explicit_execute_required": True,
        },
    }
    source_prereg_path = tmp_path / "terminal-preregistration.json"
    source_prereg_sha = _write_json(source_prereg_path, source_preregistration)

    parent_binding = runner._artifact_binding(
        source_preregistration["parent"],
        name="parent",
        base=tmp_path,
        expected_parameters=runner.QWEN3_4B_EXACT_PARAMETERS,
    )
    target_binding = runner._artifact_binding(
        source_preregistration["blank_target"],
        name="blank_target",
        base=tmp_path,
        expected_parameters=runner.QWEN3_1P4B_EXACT_PARAMETERS,
    )
    source_artifact_authentication = {
        "parent": runner._authenticate_artifact(parent_binding),
        "blank_target": runner._authenticate_artifact(target_binding),
        "split_sha256s": {
            "fit": fit_sha,
            "candidate": candidate_sha,
            "sealed": sealed_sha,
        },
        "compiler_split_sha256": compiler_fit_sha,
    }
    source_implementation_authentication = {
        "runner": {
            "path": str(historical_files["runner"]),
            "sha256": historical_hashes["runner"],
        },
        "trainer": {
            "module": runner.CANONICAL_TRAINER_MODULE,
            "path": str(historical_files["trainer"]),
            "sha256": historical_hashes["trainer"],
        },
        "compiler_cache": {
            "path": str(historical_files["compiler_cache"]),
            "sha256": historical_hashes["compiler_cache"],
        },
        "lexical_adapter": {
            "path": str(historical_files["lexical_adapter"]),
            "sha256": historical_hashes["lexical_adapter"],
        },
        "progressive_bridge": {
            "path": str(historical_files["progressive_bridge"]),
            "sha256": historical_hashes["progressive_bridge"],
        },
        "qkvo_adapter": {
            "path": str(historical_files["qkvo_adapter"]),
            "sha256": historical_hashes["qkvo_adapter"],
        },
        "swiglu_adapter": {
            "path": str(historical_files["swiglu_adapter"]),
            "sha256": historical_hashes["swiglu_adapter"],
        },
        "authenticated_before_model_access": True,
    }
    normalized_thresholds = {
        role: runner.BehaviorThresholds.from_mapping(value, role).as_dict()
        for role, value in thresholds.items()
    }
    historical_evaluation_parent_evidence = {
        "source_split": "authenticated-compiler-split-deterministic-partition",
        "candidate_split_used": False,
        "sealed_split_used": False,
        "compiler_options": compiler,
    }
    source_evaluation_sha = runner._canonical_sha256(
        {
            "split_sha256s": source_artifact_authentication["split_sha256s"],
            "compiler_split_sha256": compiler_fit_sha,
            "thresholds": normalized_thresholds,
            "selection_domain": "causal-output-actions-and-free-rollouts-only",
            "qkvo_sites": runner.QKVO_SITE_COUNT,
            "qkvo_rank": runner.QKVO_RANK,
            "swiglu_layers": runner.SWIGLU_LAYER_COUNT,
            "swiglu_rank": runner.SWIGLU_RANK,
            "lexical_residual": lexical_residual,
            "implementation": source_implementation_authentication,
            "runtime_dtype": "float16",
            "runtime_attention_implementation": (
                runner.CANONICAL_ATTENTION_IMPLEMENTATION
            ),
            "canonical_parent_evidence": historical_evaluation_parent_evidence,
        }
    )
    compiler_split_binding = runner._load_split(
        source_preregistration["compiler_split"], name="fit", base=tmp_path
    )
    calibration_ids = list(compiler_split_binding.row_ids[:20])
    validation_ids = list(compiler_split_binding.row_ids[20:])
    report_parent_evidence = {
        "source_split": "compiler_split",
        "source_fit_artifact_sha256": compiler_fit_sha,
        "calibration_row_count": 20,
        "calibration_row_ids_sha256": runner._canonical_sha256(calibration_ids),
        "validation_row_count": 4,
        "validation_row_ids_sha256": runner._canonical_sha256(validation_ids),
        "candidate_split_used": False,
        "sealed_split_used": False,
        "partition_fixed_before_model_access": True,
    }

    factor_names = tuple(
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
    monkeypatch.setattr(runner, "LEXICAL_TARGET_NATIVE_FACTOR_PARAMETERS", 18)
    monkeypatch.setattr(
        runner,
        "_expected_continuation_factor_shapes",
        lambda *, lexical_rank: {name: (1,) for name in factor_names},
    )
    factor_tensors = {
        name: torch.tensor([float(index + 1)], dtype=torch.float32)
        for index, name in enumerate(factor_names)
    }
    factor_rows = [
        {
            "name": name,
            "shape": [1],
            "dtype": "torch.float32",
            "tensor_sha256": runner._factor_tensor_sha256(factor_tensors[name]),
        }
        for name in factor_names
    ]
    factor_state_sha = runner._canonical_sha256(
        {
            "schema": runner.TRAINER_FACTOR_PACK_V2_SCHEMA,
            "lexical_rank": runner.LEXICAL_RANK,
            "factors": factor_rows,
        }
    )
    factor_manifest = {
        "schema": runner.TRAINER_FACTOR_PACK_V2_SCHEMA,
        "factor_count": 18,
        "factors": factor_rows,
        "factor_state_sha256": factor_state_sha,
        "parent_state_included": False,
        "native_child_state_included": False,
        "optimizer_state_included": False,
        "lexical_rank": runner.LEXICAL_RANK,
    }
    factor_envelope = {
        "schema": runner.FACTOR_PACK_V2_SCHEMA,
        "reason": "fit-gate-rejected",
        "preregistration_sha256": source_prereg_sha,
        "evaluation_contract_sha256": source_evaluation_sha,
        "qkvo_site_count": runner.QKVO_SITE_COUNT,
        "qkvo_rank": runner.QKVO_RANK,
        "swiglu_layer_count": runner.SWIGLU_LAYER_COUNT,
        "swiglu_rank": runner.SWIGLU_RANK,
        "lexical_rank": runner.LEXICAL_RANK,
        "factor_pack": {**factor_manifest, "factor_tensors": factor_tensors},
        "contains_model_checkpoint": False,
        "contains_parent_parameters": False,
        "contains_activations_or_hidden_states": False,
    }
    factor_path = tmp_path / "terminal-rejected-factor-pack.pt"
    torch.save(factor_envelope, factor_path)
    factor_sha = hashlib.sha256(factor_path.read_bytes()).hexdigest()
    fit_metrics = _audit_metrics("fit", 120)
    candidate_metrics = _audit_metrics("candidate", candidate_count)
    step0_state_sha = "0" * 64
    training_audit = {
        "optimizer_steps": training["steps"],
        "optimizer_class": "AdamW",
        "parent_greedy_prefix_count": 120,
        "child_on_policy_prefix_count": 120,
        "parent_output_forward_count": 960,
        "child_output_forward_count": 960,
        "parent_greedy_cache_prompt_count": 120,
        "parent_greedy_cache_forward_count": 960,
        "invariant_parent_greedy_rows_cached_once": True,
        "gradient_activation": {
            "factor_names": list(factor_names),
            "finite_gradient_factor_names": list(factor_names),
            "nonzero_gradient_factor_names": list(factor_names),
            "backward_observation_count": 48,
            "every_qkvo_gud_factor_gradient_activated": True,
            "every_lexical_factor_gradient_activated": True,
            "every_combined_factor_gradient_activated": True,
            "combined_factor_count": 18,
        },
        "teacher_signal": "parent-causal-output-logits-and-actions",
        "parent_greedy_prefix_training": True,
        "child_greedy_on_policy_prefix_training": True,
        "weight_targets_used": False,
        "activation_targets_used": False,
        "hidden_state_targets_used": False,
        "attention_map_targets_used": False,
        "cache_value_targets_used": False,
    }
    trainer_evidence = {
        "checkpoints": [
            {"step": 0, "factor_state_sha256": step0_state_sha},
            {"step": 24, "factor_state_sha256": factor_state_sha},
        ],
        "selected_step": 24,
        "selection_policy": "rollout-first-acquisition",
        "fit_metrics": fit_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_gate_passed": False,
        "sealed_evaluation_unlocked": False,
        "sealed_metrics": None,
        "training_audit": training_audit,
        "factor_pack_manifest": factor_manifest,
        "domain_binding_sha256": "1" * 64,
        "schedule_sha256": "2" * 64,
        "candidate_thresholds": normalized_thresholds["candidate"],
        "selection_uses_internal_numeric_targets": False,
        "internal_numeric_matching_used": False,
    }
    fit_passed, fit_decision = runner._behavior_gate(
        fit_metrics,
        runner.BehaviorThresholds.from_mapping(thresholds["fit"], "fit"),
    )
    candidate_passed, candidate_decision = runner._behavior_gate(
        candidate_metrics,
        runner.BehaviorThresholds.from_mapping(
            thresholds["candidate"], "candidate"
        ),
    )
    assert fit_passed is False and candidate_passed is False
    pre_bake_decisions = {
        "fit": fit_decision,
        "candidate": candidate_decision,
        "sealed": None,
    }
    factor_acquisition = runner._factor_acquisition_claim(trainer_evidence)
    source_report = {
        "schema": runner.RUN_SCHEMA,
        "status": "rejected",
        "reason": "fit-gate-rejected",
        "preregistration": str(source_prereg_path),
        "preregistration_sha256": source_prereg_sha,
        "evaluation_contract_sha256": source_evaluation_sha,
        "parent": {
            "path": str(parent),
            "exact_unique_parameters": runner.QWEN3_4B_EXACT_PARAMETERS,
        },
        "target": {
            "path": str(target),
            "exact_unique_parameters": runner.QWEN3_1P4B_EXACT_PARAMETERS,
            "reuses_existing_staging_artifact": False,
        },
        "splits": {
            "fit": {"path": str(fit), "sha256": fit_sha, "row_count": 120},
            "candidate": {
                "path": str(candidate),
                "sha256": candidate_sha,
                "row_count": candidate_count,
            },
            "sealed": {"path": str(sealed), "sha256": sealed_sha, "row_count": 32},
        },
        "compiler_split": {
            "path": str(compiler_fit),
            "sha256": compiler_fit_sha,
            "row_count": 24,
            "reuses_behavior_fit": False,
        },
        "thresholds": normalized_thresholds,
        "canonical_parent_evidence": report_parent_evidence,
        "lexical_residual": lexical_residual,
        "selection_domain": "causal-output-actions-and-free-rollouts-only",
        "runtime_dtype": "float16",
        "runtime_attention_implementation": runner.CANONICAL_ATTENTION_IMPLEMENTATION,
        "artifact_authentication": source_artifact_authentication,
        "implementation_authentication": source_implementation_authentication,
        "compiler_cache": {
            **source_cache,
            "root": str(cache_root),
            "content_addressed": True,
            "copied_into_run_output": False,
        },
        "compiled_child_cache": {
            "disposition": "hit",
            "path": str(cache_path),
            **cache_hashes,
            "verified_before_factor_install": True,
        },
        "compiler_report": compiler_report,
        "trainer_evidence": trainer_evidence,
        "trainer_evidence_sha256": runner._canonical_sha256(trainer_evidence),
        "behavior_decisions": pre_bake_decisions,
        "pre_bake_behavior_decisions": pre_bake_decisions,
        "post_bake_behavior_decisions": None,
        "factor_acquisition": factor_acquisition,
        "factor_pack": {"path": str(factor_path), "sha256": factor_sha},
        "rollback_exact": True,
        "rollback_state_sha256": "3" * 64,
        "model_checkpoint_written": False,
        "standalone_compaction_passed": False,
    }
    source_report_path = tmp_path / "terminal-run-report.json"
    _write_json(source_report_path, source_report)
    return SimpleNamespace(
        source_report=source_report,
        source_report_path=source_report_path,
        source_preregistration=source_preregistration,
        source_prereg_path=source_prereg_path,
        factor_path=factor_path,
        candidate_path=candidate,
        cache_path=cache_path,
        cache_root=cache_root,
        historical_files=historical_files,
        factor_state_sha=factor_state_sha,
        source_prereg_sha=source_prereg_sha,
        source_evaluation_sha=source_evaluation_sha,
    )


def _rewrite_audit_source_report(bundle: SimpleNamespace) -> None:
    evidence = bundle.source_report.get("trainer_evidence")
    if isinstance(evidence, dict):
        bundle.source_report["trainer_evidence_sha256"] = runner._canonical_sha256(
            evidence
        )
    _write_json(bundle.source_report_path, bundle.source_report)


def test_builder_seals_real_curated_corpus_without_model_creation(tmp_path) -> None:
    parent = _artifact(
        tmp_path / "parent",
        {"config.json": b"parent-config", "model.safetensors": b"parent"},
    )
    target = _artifact(
        tmp_path / "target",
        {"config.json": b"target-config", "model.safetensors": b"target"},
    )
    target_weight_sha = hashlib.sha256(b"target").hexdigest()
    corpus, corpus_sha, corpus_count = _corpus(tmp_path / "corpus.jsonl")
    output = tmp_path / "sealed"
    report = builder.prepare_stage(
        output_dir=output,
        parent=parent,
        target=target,
        corpus_path=corpus,
        corpus_sha256=corpus_sha,
        corpus_line_count=corpus_count,
        tokenizer_loader=lambda _path: _Tokenizer(),
        target_required_files={"model.safetensors": target_weight_sha},
    )
    assert report["model_weights_loaded"] is False
    assert report["model_checkpoint_created"] is False
    assert report["training_launched"] is False
    assert {path.name for path in output.iterdir()} == {
        "fit.json",
        "candidate.json",
        "sealed.json",
        "preregistration.json",
        "build_report.json",
    }
    counts = {
        name: len(json.loads((output / f"{name}.json").read_text())["prompts"])
        for name in ("fit", "candidate", "sealed")
    }
    assert counts == {"fit": 24, "candidate": 12, "sealed": 16}
    selected = [
        item
        for name in ("fit", "candidate", "sealed")
        for item in json.loads((output / f"{name}.json").read_text())["prompts"]
    ]
    assert len({item["source_line_number"] for item in selected}) == 52
    assert len({item["source_prompt_sha256"] for item in selected}) == 52
    preregistration = json.loads(
        (output / "preregistration.json").read_text(encoding="utf-8")
    )
    assert preregistration["training"] == {
        "steps": 24,
        "learning_rate": 0.001,
        "checkpoint_steps": [2, 4, 8, 12, 16, 20],
        "fit_horizon": 8,
        "evaluation_horizon": 8,
        "fit_prompts_per_step": 3,
        "temperature": 1.0,
        "action_loss_weight": 1.0,
        "distribution_loss_weight": 0.1,
        "top_k": 5,
        "gradient_clip_norm": 1.0,
        "max_prompt_tokens": 128,
        "max_train_prefixes": 1152,
        "max_evaluation_prefixes": 5000,
        "max_checkpoints": 8,
        "selection_policy": "rollout-first-acquisition",
    }
    assert preregistration["compiler"]["bootstrap_kwargs"] == {
        "heldout_gate_policy": "output_only",
        "max_operator_delta_ratio": 0.1,
        "trace_pooling": "last_valid",
    }
    assert preregistration["corpus_provenance"]["synthetic_replicas"] == 0
    assert preregistration["corpus_provenance"]["prompt_text_modified"] is False
    assert preregistration["thresholds"]["candidate"] == {
        "minimum_action_accuracy": 0.5,
        "minimum_exact_rollout_fraction": 1 / 12,
        "minimum_token_match_fraction": 0.2,
        "minimum_lcp_fraction": 0.125,
        "minimum_action_count": 96,
        "minimum_rollout_count": 12,
    }
    authenticated = runner.authenticate_preregistration(
        output / "preregistration.json",
        report["preregistration_sha256"],
    )
    assert authenticated.preflight_report()["models_loaded"] is False
    assert (
        authenticated.preflight_report()["canonical_parent_evidence"][
            "candidate_split_used"
        ]
        is False
    )
    with pytest.raises(builder.StagePreparationError, match="must not already exist"):
        builder.prepare_stage(
            output_dir=output,
            parent=parent,
            target=target,
            corpus_path=corpus,
            corpus_sha256=corpus_sha,
            corpus_line_count=corpus_count,
            tokenizer_loader=lambda _path: _Tokenizer(),
            target_required_files={"model.safetensors": target_weight_sha},
        )


def test_remote_wrapper_defaults_preflight_and_requires_explicit_execute() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "remote_run_qwen_exact_target_native_stage.sh"
    ).read_text(encoding="utf-8")
    assert "MODE=preflight" in script
    assert '[[ ${1:-} == "--execute" ]]' in script
    assert "--preflight" in script
    assert "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,6}" in script
    assert "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" in script
    assert "parent logical cuda:0, child logical cuda:1" in script
    assert 'ROOT=$(realpath -e "$ROOT")' in script
    assert 'test "$ROOT" = "/home/cereal/model-surgery-lab/neural-forge-src"' in script
    assert "nohup" not in script
    assert not any(line.rstrip().endswith("&") for line in script.splitlines())
    assert '[[ ${1:-} == "--lexical-continuation" ]]' in script
    assert "LEXICAL_CONTINUATION_SOURCE_REPORT" in script
    assert "--lexical-continuation-source-report" in script
    assert "--source-split-dir" in script


def test_lexical_builder_seals_nonrestarting_schedule_and_required_cache_hit(
    tmp_path, monkeypatch
) -> None:
    parent = _artifact(
        tmp_path / "parent",
        {"config.json": b"parent-config", "tokenizer.json": b"tokenizer"},
    )
    target_config = json.dumps(
        {
            "model_type": "qwen3",
            "vocab_size": runner.LEXICAL_VOCAB_SIZE,
            "hidden_size": runner.LEXICAL_HIDDEN_SIZE,
            "tie_word_embeddings": False,
        }
    ).encode()
    target = _artifact(
        tmp_path / "target",
        {"config.json": target_config, "model.safetensors": b"target"},
    )
    original_splits = tmp_path / "original"
    original_splits.mkdir()
    compiler_fit = original_splits / "fit.json"
    compiler_fit.write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(
        builder.continuation_splits.PINNED_ORIGINAL_SPLIT_SHA256S,
        "fit",
        hashlib.sha256(compiler_fit.read_bytes()).hexdigest(),
    )
    source_splits = tmp_path / "source-splits"
    source_splits.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    identity_sha = "1" * 64
    cache_path = cache_root / identity_sha
    cache_path.mkdir()
    factor_path = tmp_path / "source_factor_pack.pt"
    factor_path.write_bytes(b"authenticated-factor-pack")
    factor_sha = hashlib.sha256(factor_path.read_bytes()).hexdigest()
    factor_state_sha = "2" * 64
    compiler_report = {"bootstrap_accepted": True, "target_budget_exact": True}
    compiler_report_sha = runner._canonical_sha256(compiler_report)
    source_report = {
        "status": "rejected",
        "rollback_exact": True,
        "rollback_state_sha256": "3" * 64,
        "splits": {
            role: {
                "sha256": builder.continuation_splits.PINNED_RESIDUAL_ATTENTION_SPLIT_SHA256S[
                    role
                ],
                "row_count": builder.continuation_splits.RESIDUAL_ATTENTION_COUNTS[
                    role
                ],
            }
            for role in ("fit", "candidate", "sealed")
        },
        "factor_pack": {"path": str(factor_path), "sha256": factor_sha},
        "trainer_evidence": {
            "sealed_evaluation_unlocked": False,
            "sealed_metrics": None,
            "factor_pack_manifest": {
                "schema": runner.TRAINER_FACTOR_PACK_SCHEMA,
                "factor_state_sha256": factor_state_sha,
            },
        },
        "compiler_report": compiler_report,
        "compiled_child_cache": {
            "disposition": "published",
            "path": str(cache_path),
            "compiler_identity_sha256": identity_sha,
            "receipt_sha256": "4" * 64,
            "compiler_report_sha256": compiler_report_sha,
            "state_dict_sha256": "5" * 64,
            "artifact_tree_sha256": "6" * 64,
            "verified_before_factor_install": True,
        },
    }
    source_report_path = tmp_path / "source_run_report.json"
    source_report_path.write_text(
        json.dumps(source_report, sort_keys=True) + "\n", encoding="utf-8"
    )

    def fake_split_builder(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir()
        for role in ("fit", "candidate"):
            (output / f"{role}.json").write_text(
                json.dumps({"schema": "test", "split": role}) + "\n",
                encoding="utf-8",
            )
        return {"build_content_sha256": "7" * 64}

    monkeypatch.setattr(
        builder.continuation_splits,
        "prepare_lexical_continuation_splits",
        fake_split_builder,
    )
    monkeypatch.setattr(builder, "_default_target_required_files", lambda: {})
    monkeypatch.setattr(
        builder.stage,
        "authenticate_preregistration",
        lambda _path, _sha: SimpleNamespace(evaluation_contract_sha256="8" * 64),
    )
    output = tmp_path / "lexical-prereg"

    report = builder.prepare_lexical_continuation_stage(
        output_dir=output,
        parent=parent,
        target=target,
        source_run_report=source_report_path,
        source_split_dir=source_splits,
        original_split_dir=original_splits,
        compiler_cache_root=cache_root,
        corpus_path=tmp_path / "unused-corpus.jsonl",
    )

    preregistration = json.loads((output / "preregistration.json").read_text())
    assert report["mode"] == "authenticated-lexical-attention-continuation"
    assert preregistration["lexical_residual"] == {
        "rank": 32,
        "vocab_size": 151936,
        "hidden_size": 1536,
        "untied_required": True,
    }
    assert preregistration["splits"]["fit"]["expected_count"] == 120
    assert preregistration["splits"]["candidate"]["expected_count"] == 24
    assert preregistration["splits"]["sealed"] == {
        "path": str(source_splits / "sealed.json"),
        "sha256": builder.continuation_splits.PINNED_RESIDUAL_ATTENTION_SPLIT_SHA256S[
            "sealed"
        ],
        "expected_count": 32,
    }
    assert preregistration["compiler_cache"]["require_hit"] is True
    assert {
        "qkvo_adapter_sha256",
        "swiglu_adapter_sha256",
    }.issubset(preregistration["implementation"])
    assert (
        preregistration["compiler_cache"]["expected_compiler_identity_sha256"]
        == identity_sha
    )
    schedule = preregistration["training"]
    assert schedule["steps"] == 48
    assert schedule["fit_prompts_per_step"] == 12
    assert schedule["qkvo_frozen_steps"] == 8
    assert schedule["gud_frozen_steps"] == 24
    assert schedule["lexical_embedding_frozen_steps"] == 24
    assert schedule["lexical_head_frozen_steps"] == 0
    assert schedule["learning_rate_decay_step_3"] == 40
    assert schedule["learning_rate_decay_scale_3"] == 0.5
    assert schedule["gud_learning_rate_scale"] == 0.25
    assert schedule["lexical_embedding_learning_rate_scale"] == 0.25
    assert schedule["max_checkpoints"] == 10
    assert preregistration["thresholds"]["fit"]["minimum_action_count"] == 960
    assert (
        preregistration["thresholds"]["candidate"]["minimum_action_count"]
        == 192
    )
    assert preregistration["thresholds"]["sealed"]["minimum_action_count"] == 256


def test_output_jacobian_audit_builder_reuses_terminal_v2_artifacts_model_free(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    output = tmp_path / "jacobian-audit-prereg"

    report = builder.prepare_output_jacobian_audit_stage(
        output_dir=output,
        source_run_report=bundle.source_report_path,
    )

    assert {path.name for path in output.iterdir()} == {
        "preregistration.json",
        "build_report.json",
    }
    assert report["mode"] == builder.OUTPUT_JACOBIAN_AUDIT_MODE
    assert report["audit_only"] is True
    assert report["model_weights_loaded"] is False
    assert report["training_launched"] is False
    assert report["source_preregistration_sha256"] == bundle.source_prereg_sha
    assert report["source_evaluation_contract_sha256"] == bundle.source_evaluation_sha
    assert report["source_factor_state_sha256"] == bundle.factor_state_sha
    preregistration = json.loads((output / "preregistration.json").read_text())
    assert preregistration["output_jacobian_audit"] == {
        "schema": runner.OUTPUT_JACOBIAN_AUDIT_SCHEMA,
        "source_exposed_candidate_as_fit": True,
        "basis_split_role": "candidate",
        "prompt_count": 24,
        "horizon": 8,
        "lanes": ["parent-greedy", "child-greedy"],
        "rank": 8,
        "sketch_width": 16,
        "min_normal_fraction": 0.20,
        "min_guided_over_random_ratio": 1.50,
        "max_bytes": 268_435_456,
    }
    assert {
        role: preregistration["splits"][role]["expected_count"]
        for role in ("fit", "candidate", "sealed")
    } == builder.OUTPUT_JACOBIAN_AUDIT_COUNTS
    assert (
        preregistration["splits"]["candidate"]["path"]
        == str(bundle.candidate_path)
    )
    assert preregistration["continuation"]["factor_pack"]["path"] == str(
        bundle.factor_path
    )
    expected_implementation = {
        "runner_sha256",
        "trainer_module",
        "trainer_sha256",
        "compiler_cache_sha256",
        "lexical_adapter_sha256",
        "progressive_bridge_sha256",
        "qkvo_adapter_sha256",
        "swiglu_adapter_sha256",
        "output_jacobian_basis_sha256",
        "output_jacobian_sketch_sha256",
        "qkvo_output_jacobian_audit_sha256",
    }
    assert set(preregistration["implementation"]) == expected_implementation
    authenticated = runner.authenticate_preregistration(
        output / "preregistration.json", report["preregistration_sha256"]
    )
    assert authenticated.output_jacobian_audit is not None
    assert authenticated.preflight_report()["models_loaded"] is False


@pytest.mark.parametrize(
    "case",
    (
        "status",
        "report_schema",
        "reason",
        "rollback",
        "checkpoint",
        "standalone",
        "publication",
        "post_bake",
        "fresh_reload",
        "candidate_missing",
        "candidate_passed",
        "candidate_decision_missing",
        "sealed_unlocked",
        "sealed_metrics",
        "internal_matching",
        "selection_numeric",
        "weight_target",
    ),
)
def test_output_jacobian_audit_builder_rejects_every_terminal_source_gate(
    tmp_path, monkeypatch, case
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    report = bundle.source_report
    evidence = report["trainer_evidence"]
    if case == "status":
        report["status"] = "accepted"
    elif case == "report_schema":
        report["schema"] = "wrong-run-schema"
    elif case == "reason":
        report["reason"] = "candidate-gate-rejected"
    elif case == "rollback":
        report["rollback_exact"] = False
    elif case == "checkpoint":
        report["model_checkpoint_written"] = True
    elif case == "standalone":
        report["standalone_compaction_passed"] = True
    elif case == "publication":
        report["publication"] = {"exact_budget_match": True}
    elif case == "post_bake":
        report["post_bake_behavior_decisions"] = {"fit": {"passed": False}}
    elif case == "fresh_reload":
        report["fresh_reload_certificate"] = {"loaded": True}
    elif case == "candidate_missing":
        evidence["candidate_metrics"] = None
    elif case == "candidate_passed":
        evidence["candidate_gate_passed"] = True
    elif case == "candidate_decision_missing":
        report["pre_bake_behavior_decisions"]["candidate"] = None
    elif case == "sealed_unlocked":
        evidence["sealed_evaluation_unlocked"] = True
    elif case == "sealed_metrics":
        evidence["sealed_metrics"] = {"action_count": 256}
    elif case == "internal_matching":
        evidence["internal_numeric_matching_used"] = True
    elif case == "selection_numeric":
        evidence["selection_uses_internal_numeric_targets"] = True
    elif case == "weight_target":
        evidence["training_audit"]["weight_targets_used"] = True
    _rewrite_audit_source_report(bundle)

    with pytest.raises(builder.StagePreparationError):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "rejected-audit",
            source_run_report=bundle.source_report_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", runner.TRAINER_FACTOR_PACK_SCHEMA),
        ("factor_count", 17),
        ("lexical_rank", 31),
        ("factor_state_sha256", "f" * 64),
    ),
)
def test_output_jacobian_audit_builder_rejects_manifest_schema_rank_and_state(
    tmp_path, monkeypatch, field, value
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    bundle.source_report["trainer_evidence"]["factor_pack_manifest"][field] = value
    _rewrite_audit_source_report(bundle)

    with pytest.raises(builder.StagePreparationError, match="18-factor v2 state"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "bad-manifest-audit",
            source_run_report=bundle.source_report_path,
        )


def test_output_jacobian_audit_builder_rejects_non_v2_factor_envelope(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    envelope = torch.load(bundle.factor_path, map_location="cpu", weights_only=True)
    envelope["schema"] = runner.FACTOR_PACK_SCHEMA
    torch.save(envelope, bundle.factor_path)
    bundle.source_report["factor_pack"]["sha256"] = hashlib.sha256(
        bundle.factor_path.read_bytes()
    ).hexdigest()
    _rewrite_audit_source_report(bundle)

    with pytest.raises(builder.StagePreparationError, match="lexical-v2 payload"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "bad-pack-audit",
            source_run_report=bundle.source_report_path,
        )


def test_output_jacobian_audit_builder_rejects_split_tamper_and_wrong_count(
    tmp_path, monkeypatch
) -> None:
    tampered = _audit_source_bundle(tmp_path / "tampered", monkeypatch)
    tampered.candidate_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(builder.StagePreparationError, match="candidate split"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "tampered-output",
            source_run_report=tampered.source_report_path,
        )

    wrong_count = _audit_source_bundle(
        tmp_path / "wrong-count", monkeypatch, candidate_count=23
    )
    with pytest.raises(builder.StagePreparationError, match="24-row"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "wrong-count-output",
            source_run_report=wrong_count.source_report_path,
        )


@pytest.mark.parametrize(
    "case", ("disposition", "verified", "hash", "cache_path")
)
def test_output_jacobian_audit_builder_rejects_unverified_cache_source(
    tmp_path, monkeypatch, case
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    cache = bundle.source_report["compiled_child_cache"]
    if case == "disposition":
        cache["disposition"] = "published"
    elif case == "verified":
        cache["verified_before_factor_install"] = False
    elif case == "hash":
        cache["receipt_sha256"] = "0" * 64
    elif case == "cache_path":
        cache["path"] = str(bundle.cache_path.parent)
    _rewrite_audit_source_report(bundle)

    with pytest.raises(builder.StagePreparationError, match="cache"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "bad-cache-output",
            source_run_report=bundle.source_report_path,
        )


def test_output_jacobian_audit_builder_authenticates_historical_prereg_hash_and_pins(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(tmp_path / "prereg-tamper", monkeypatch)
    bundle.source_prereg_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(builder.StagePreparationError, match="path/SHA-256"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "prereg-tamper-output",
            source_run_report=bundle.source_report_path,
        )

    pin_bundle = _audit_source_bundle(tmp_path / "pin-tamper", monkeypatch)
    pin_bundle.source_report["implementation_authentication"]["qkvo_adapter"][
        "sha256"
    ] = "0" * 64
    _rewrite_audit_source_report(pin_bundle)
    with pytest.raises(builder.StagePreparationError, match="qkvo_adapter"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "pin-tamper-output",
            source_run_report=pin_bundle.source_report_path,
        )


def test_output_jacobian_audit_cli_modes_are_mutually_exclusive(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="at most one"):
        builder.main(
            [
                "--continuation-source-report",
                str(report),
                "--output-jacobian-audit-source-report",
                str(report),
            ]
        )
    with pytest.raises(SystemExit, match="at most one"):
        builder.main(
            [
                "--lexical-continuation-source-report",
                str(report),
                "--output-jacobian-audit-source-report",
                str(report),
            ]
        )


def test_output_jacobian_audit_builder_refuses_existing_output(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    output = tmp_path / "already-exists"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(builder.StagePreparationError, match="must not already exist"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=output,
            source_run_report=bundle.source_report_path,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_output_jacobian_audit_builder_rejects_unsealed_source_preregistration(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch, sealed_source=False)
    with pytest.raises(builder.StagePreparationError, match="exact lexical stage"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "unsealed-output",
            source_run_report=bundle.source_report_path,
        )


def test_output_jacobian_historical_paths_must_exist_but_may_have_evolved(
    tmp_path, monkeypatch
) -> None:
    missing = _audit_source_bundle(tmp_path / "missing", monkeypatch)
    missing.historical_files["qkvo_adapter"].unlink()
    with pytest.raises(builder.StagePreparationError, match="qkvo_adapter"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "missing-output",
            source_run_report=missing.source_report_path,
        )

    evolved = _audit_source_bundle(tmp_path / "evolved", monkeypatch)
    evolved.historical_files["runner"].write_text(
        "# live checkout evolved after the authenticated historical run\n",
        encoding="utf-8",
    )
    result = builder.prepare_output_jacobian_audit_stage(
        output_dir=tmp_path / "evolved-output",
        source_run_report=evolved.source_report_path,
    )
    assert result["mode"] == builder.OUTPUT_JACOBIAN_AUDIT_MODE


def test_output_jacobian_rejects_historical_preimage_as_run_partition_evidence(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    bundle.source_report["canonical_parent_evidence"] = {
        "source_split": "authenticated-compiler-split-deterministic-partition",
        "candidate_split_used": False,
        "sealed_split_used": False,
        "compiler_options": bundle.source_preregistration["compiler"],
    }
    _rewrite_audit_source_report(bundle)
    with pytest.raises(builder.StagePreparationError, match="partition evidence"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "wrong-partition-output",
            source_run_report=bundle.source_report_path,
        )


def test_output_jacobian_recomputes_fit_gate_and_rejects_forged_fit_reason(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    evidence = bundle.source_report["trainer_evidence"]
    evidence["fit_metrics"] = _audit_metrics("fit", 120, passing=True)
    _passed, fit_decision = runner._behavior_gate(
        evidence["fit_metrics"],
        runner.BehaviorThresholds.from_mapping(
            bundle.source_preregistration["thresholds"]["fit"], "fit"
        ),
    )
    decisions = bundle.source_report["pre_bake_behavior_decisions"]
    decisions["fit"] = fit_decision
    bundle.source_report["behavior_decisions"] = decisions
    _rewrite_audit_source_report(bundle)
    with pytest.raises(builder.StagePreparationError, match="decisions"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / "fit-forgery-output",
            source_run_report=bundle.source_report_path,
        )


@pytest.mark.parametrize("case", ("report-acquisition", "selected-checkpoint"))
def test_output_jacobian_rejects_acquisition_and_checkpoint_forgery(
    tmp_path, monkeypatch, case
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    if case == "report-acquisition":
        bundle.source_report["factor_acquisition"]["selected_step"] = 40
    else:
        checkpoints = bundle.source_report["trainer_evidence"]["checkpoints"]
        checkpoints[1]["factor_state_sha256"] = "f" * 64
    _rewrite_audit_source_report(bundle)
    with pytest.raises(builder.StagePreparationError):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / f"{case}-output",
            source_run_report=bundle.source_report_path,
        )


@pytest.mark.parametrize("case", ("empty", "tampered"))
def test_output_jacobian_public_cache_verification_rejects_artifact_corruption(
    tmp_path, monkeypatch, case
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    if case == "empty":
        for path in bundle.cache_path.iterdir():
            path.unlink()
    else:
        config = bundle.cache_path / "config.json"
        config.write_bytes(config.read_bytes() + b"tamper")
    with pytest.raises(builder.StagePreparationError, match="public verification"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=tmp_path / f"cache-{case}-output",
            source_run_report=bundle.source_report_path,
        )


def test_output_jacobian_normalizes_all_copied_external_paths(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(
        tmp_path, monkeypatch, relative_bindings=True
    )
    output = tmp_path / "absolute-bindings-output"
    builder.prepare_output_jacobian_audit_stage(
        output_dir=output,
        source_run_report=bundle.source_report_path,
    )
    preregistration = json.loads((output / "preregistration.json").read_text())
    paths = [
        preregistration["parent"]["path"],
        preregistration["blank_target"]["path"],
        preregistration["compiler_cache"]["root"],
        preregistration["compiler_split"]["path"],
        preregistration["continuation"]["factor_pack"]["path"],
        preregistration["continuation"]["source_run_report"]["path"],
        *(binding["path"] for binding in preregistration["splits"].values()),
    ]
    assert all(Path(value).is_absolute() for value in paths)
    assert (
        preregistration["continuation"]["expected_compiled_state_sha256"]
        != preregistration["compiler_cache"]["expected_state_dict_sha256"]
    )


def test_output_jacobian_atomic_staging_cleans_up_on_live_preflight_failure(
    tmp_path, monkeypatch
) -> None:
    bundle = _audit_source_bundle(tmp_path, monkeypatch)
    output = tmp_path / "atomic-output"

    def fail_authentication(_path, _sha):
        raise runner.ExactTargetNativeStageError("injected live preflight failure")

    monkeypatch.setattr(builder.stage, "authenticate_preregistration", fail_authentication)
    with pytest.raises(runner.ExactTargetNativeStageError, match="injected"):
        builder.prepare_output_jacobian_audit_stage(
            output_dir=output,
            source_run_report=bundle.source_report_path,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".atomic-output.*.staging"))
