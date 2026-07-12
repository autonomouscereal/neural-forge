from __future__ import annotations

import hashlib
import json
import math
import sys
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
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


def _audit_binding_payload() -> dict[str, object]:
    return {
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


def _audit_contract_files(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, str, str]:
    path, _digest, _v1_state = _continuation_contract_files(tmp_path, monkeypatch)
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
    candidate_path = Path(preregistration["splits"]["candidate"]["path"])
    candidate_payload = {
        "schema": runner.SPLIT_SCHEMA,
        "split": "candidate",
        "sealed_before_model_access": False,
        "prompts": [
            {"id": f"audit-candidate-{index}", "token_ids": [1, 100 + index, 500]}
            for index in range(24)
        ],
    }
    preregistration["splits"]["candidate"] = {
        "path": str(candidate_path),
        "sha256": _write_json(candidate_path, candidate_payload),
        "expected_count": 24,
    }
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
    factor_path = Path(preregistration["continuation"]["factor_pack"]["path"])
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
        "factor_pack": {
            "schema": runner.TRAINER_FACTOR_PACK_V2_SCHEMA,
            "factor_count": len(names),
            "factors": rows,
            "factor_state_sha256": state_sha,
            "parent_state_included": False,
            "native_child_state_included": False,
            "optimizer_state_included": False,
            "lexical_rank": runner.LEXICAL_RANK,
            "factor_tensors": tensors,
        },
        "contains_model_checkpoint": False,
        "contains_parent_parameters": False,
        "contains_activations_or_hidden_states": False,
    }
    torch.save(envelope, factor_path)
    factor_sha = hashlib.sha256(factor_path.read_bytes()).hexdigest()
    source_report_path = Path(
        preregistration["continuation"]["source_run_report"]["path"]
    )
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_report["factor_pack"] = {
        "path": str(factor_path),
        "sha256": factor_sha,
    }
    source_report["trainer_evidence"]["factor_pack_manifest"] = {
        "factor_state_sha256": state_sha
    }
    source_report["trainer_evidence_sha256"] = runner._canonical_sha256(
        source_report["trainer_evidence"]
    )
    source_report["factor_acquisition"]["selected_factor_state_sha256"] = state_sha
    source_report_sha = _write_json(source_report_path, source_report)
    preregistration["continuation"]["factor_pack"] = {
        "path": str(factor_path),
        "sha256": factor_sha,
    }
    preregistration["continuation"]["source_run_report"] = {
        "path": str(source_report_path),
        "sha256": source_report_sha,
    }
    preregistration["continuation"]["expected_factor_state_sha256"] = state_sha
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
            "output_jacobian_basis_sha256": runner._file_sha256(
                source_root / "hf_output_jacobian_orthogonal_basis.py"
            ),
            "output_jacobian_sketch_sha256": runner._file_sha256(
                source_root / "hf_output_jacobian_sketch.py"
            ),
            "qkvo_output_jacobian_audit_sha256": runner._file_sha256(
                source_root / "hf_qkvo_output_jacobian_audit.py"
            ),
        }
    )
    preregistration["lexical_residual"] = {
        "rank": runner.LEXICAL_RANK,
        "vocab_size": runner.LEXICAL_VOCAB_SIZE,
        "hidden_size": runner.LEXICAL_HIDDEN_SIZE,
        "untied_required": True,
    }
    preregistration["compiler_cache"].update(
        {
            "require_hit": True,
            "expected_compiler_identity_sha256": "a" * 64,
            "expected_receipt_sha256": "b" * 64,
            "expected_compiler_report_sha256": "c" * 64,
            "expected_state_dict_sha256": "d" * 64,
            "expected_artifact_tree_sha256": "e" * 64,
        }
    )
    preregistration["output_jacobian_audit"] = _audit_binding_payload()
    return path, _write_json(path, preregistration), state_sha


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

    def evaluate_baked_behavior(self, model, parent, contract, split_name):
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


@dataclass(frozen=True)
class _AuditGateReport:
    passed: bool
    seed: int
    factor_state_sha256: str
    basis_manifest_sha256: str
    diagnostic_roles: tuple[str, ...] | None = None

    def to_dict(self):
        config = {
            "prompt_count": 24,
            "horizon": 8,
            "lanes": ["parent-greedy", "child-greedy"],
            "rank": 8,
            "sketch_width": 16,
            "seed": self.seed,
            "min_normal_fraction": 0.20,
            "min_guided_over_random_ratio": 1.50,
            "max_bytes": 268_435_456,
        }
        native_sha = hashlib.sha256(b"compiled").hexdigest()
        site_indices = [
            [layer, group]
            for layer in range(runner.TARGET_LAYER_COUNT)
            for group in range(runner.TARGET_PHYSICAL_GROUPS)
        ]
        diagnostic_roles = (
            ("q", "k", "v", "o")
            if self.diagnostic_roles is None
            else self.diagnostic_roles
        )
        final_passed = self.passed and bool(diagnostic_roles)
        installable_roles = diagnostic_roles if final_passed else ()
        replay_blocks = 32
        callback_count = 2 * replay_blocks * runner.QKVO_SITE_COUNT
        authenticator_sha = "8" * 64
        coefficient_sha = "9" * 64
        maximum_replay_blocks = 24 * 8 * 2
        fingerprint_upper = (
            maximum_replay_blocks
            * runner.QKVO_SITE_COUNT
            * runner.QKVO_FINGERPRINT_VALUES_PER_WRAPPER
            * 4
        )
        actual_fingerprint = (
            replay_blocks
            * runner.QKVO_SITE_COUNT
            * runner.QKVO_FINGERPRINT_VALUES_PER_WRAPPER
            * 4
        )
        persistent_upper = 128_778_240
        largest_operation = 3_437_664
        wrapper_transient = 4_354_944
        coefficient_bytes = 17_694_720
        hash_staging_limit = runner.QKVO_TENSOR_HASH_STAGING_LIMIT_BYTES
        final_pack_bytes = 19_660_800
        token_path_scalar_bytes = runner.QKVO_TOKEN_PATH_SCALAR_TRANSIENT_BYTES
        phase1_persistent = 120_000_000
        phase1_transient = 3_000_000
        phase1_peak = sum(
            (
                phase1_persistent,
                phase1_transient,
                wrapper_transient,
                fingerprint_upper,
                fingerprint_upper,
                coefficient_bytes,
                hash_staging_limit,
                token_path_scalar_bytes,
            )
        )
        phase2_persistent = 110_000_000
        phase2_transient = 2_500_000
        phase2_peak = sum(
            (
                phase2_persistent,
                phase2_transient,
                wrapper_transient,
                2 * actual_fingerprint,
                coefficient_bytes,
                hash_staging_limit,
                token_path_scalar_bytes,
            )
        )
        accounted_peak = sum(
            (
                persistent_upper,
                largest_operation,
                wrapper_transient,
                fingerprint_upper,
                fingerprint_upper,
                coefficient_bytes,
                hash_staging_limit,
                final_pack_bytes,
                final_pack_bytes,
                token_path_scalar_bytes,
            )
        )
        memory_scope = {
            "bounded": "audit-explicit-auxiliary-tensors-only",
            "includes": [
                "sketch-accumulators-and-bases",
                "deferred-replay-authenticator-buffers",
                "replay-authenticator-coefficients",
                "bounded-tensor-hash-staging",
                "final-guided-cpu-pack",
                "basis-payload-cpu-clone",
                "token-path-planning-and-replay-argmax-scalar",
            ],
            "excludes": [
                "parent-and-child-model-weights",
                "caller-owned-authenticated-continuation-factor-payload",
                "standard-forward-backward-autograd-activations",
                "framework-allocator-reservations-and-overhead",
                "python-object-overhead",
            ],
        }
        factor_shapes = runner._expected_continuation_factor_shapes(
            lexical_rank=runner.LEXICAL_RANK
        )
        factor_numel = sum(math.prod(shape) for shape in factor_shapes.values())
        factor_nbytes = factor_numel * 4
        largest_factor_nbytes = max(
            math.prod(shape) * 4 for shape in factor_shapes.values()
        )
        observed_hash_staging = hash_staging_limit
        memory = {
            "persistent_upper_bound_bytes": persistent_upper,
            "largest_single_operation_bytes": largest_operation,
            "wrapper_hook_transient_bytes": wrapper_transient,
            "maximum_replay_blocks": maximum_replay_blocks,
            "deferred_fingerprint_buffer_upper_bound_bytes": fingerprint_upper,
            "deferred_fingerprint_host_staging_upper_bound_bytes": fingerprint_upper,
            "replay_authenticator_coefficient_bytes": coefficient_bytes,
            "tensor_hash_cpu_staging_limit_bytes": hash_staging_limit,
            "final_guided_cpu_pack_bytes": final_pack_bytes,
            "basis_payload_cpu_clone_bytes": final_pack_bytes,
            "replay_plan_cached_parent_logit_row_count": 0,
            "replay_plan_cached_parent_logit_bytes": 0,
            "replay_plan_cached_child_logit_row_count": 0,
            "replay_plan_cached_child_logit_bytes": 0,
            "replay_plan_cached_tensor_count": 0,
            "replay_plan_cached_tensor_bytes": 0,
            "token_path_scalar_transient_bytes": token_path_scalar_bytes,
            "accounted_auxiliary_peak_bytes": accounted_peak,
            "max_auxiliary_bytes": 268_435_456,
            "memory_scope": memory_scope,
            "phase1_actual_persistent_bytes": phase1_persistent,
            "phase1_actual_max_batch_transient_bytes": phase1_transient,
            "phase1_wrapper_hook_transient_bytes": wrapper_transient,
            "phase1_actual_peak_bytes": phase1_peak,
            "manual_plan_covers_phase1": 1,
            "actual_replay_blocks": replay_blocks,
            "actual_deferred_fingerprint_buffer_bytes": actual_fingerprint,
            "phase2_actual_persistent_bytes": phase2_persistent,
            "phase2_actual_max_batch_transient_bytes": phase2_transient,
            "phase2_wrapper_hook_transient_bytes": wrapper_transient,
            "phase2_actual_peak_bytes": phase2_peak,
            "manual_plan_covers_phase2": 1,
            "actual_replay_authenticator_coefficient_bytes": coefficient_bytes,
            "actual_final_guided_cpu_pack_bytes": final_pack_bytes,
            "finalized_result_tensors_retained_on_accumulator_device": 0,
            "all_sketches_drained_after_finalize": 1,
            "maximum_observed_tensor_hash_cpu_staging_bytes": observed_hash_staging,
            "factor_state_capture_mode": runner.QKVO_FACTOR_CAPTURE_MODE,
            "factor_state_total_nbytes": factor_nbytes,
            "factor_state_retained_cpu_clone_bytes": 0,
            "factor_state_before_after_capture_overlap_bytes": 0,
        }
        callback_sources = {"collector.backward_hook": "a" * 64}

        def gate_metrics(role_passed: bool) -> dict[str, object]:
            return {
                "normal_fraction": 0.25 if role_passed else 0.10,
                "guided_over_random_ratio": 2.0 if role_passed else 1.0,
                "normal_fraction_gate_passed": role_passed,
                "guided_over_random_gate_passed": role_passed,
                "passed": role_passed,
            }

        payload = {
            "schema": runner.OUTPUT_JACOBIAN_AUDIT_SCHEMA,
            "passed": final_passed,
            "config": config,
            "config_sha256": runner._canonical_sha256(config),
            "site_count": 120,
            "role_count": 4,
            "sketch_count": 480,
            "site_indices": site_indices,
            "site_indices_sha256": runner._canonical_sha256(site_indices),
            "basis_prompt_count": 24,
            "basis_source_role": "source-exposed-candidate",
            "source_exposed_candidate_as_fit": True,
            "new_candidate_accessed": False,
            "sealed_accessed": False,
            "optimizer_created": False,
            "optimizer_used": False,
            "optimizer_steps": 0,
            "hooks_registered": 120,
            "hooks_removed": True,
            "active_hook_count_after": 0,
            "transaction_live_after": True,
            "rollback_compatible_after": True,
            "native_state_sha256_before": native_sha,
            "native_state_sha256_after": native_sha,
            "inherited_qkvo_state_sha256_before": "5" * 64,
            "inherited_qkvo_state_sha256_after": "5" * 64,
            "factor_state_sha256_before": self.factor_state_sha256,
            "factor_state_sha256_after": self.factor_state_sha256,
            "factor_state_capture": {
                "mode": runner.QKVO_FACTOR_CAPTURE_MODE,
                "factor_count": len(factor_shapes),
                "total_numel": factor_numel,
                "total_nbytes": factor_nbytes,
                "largest_factor_nbytes": largest_factor_nbytes,
                "retained_cpu_clone_bytes_before_sketch": 0,
                "retained_cpu_clone_bytes_after_audit": 0,
                "maximum_observed_hash_staging_bytes": observed_hash_staging,
                "hash_staging_limit_bytes": hash_staging_limit,
                "before_after_captures_overlap": False,
            },
            "factor_parameter_versions_sha256_before": "7" * 64,
            "factor_parameter_versions_sha256_after": "7" * 64,
            "factor_gradients_cleared": True,
            "pass1_replay_sha256": "6" * 64,
            "pass2_replay_sha256": "6" * 64,
            "pass1_full_element_replay_authenticator_sha256": authenticator_sha,
            "pass2_full_element_replay_authenticator_sha256": authenticator_sha,
            "probabilistic_replay_authenticator_match": True,
            "replay_authentication_kind": runner.QKVO_REPLAY_AUTHENTICATOR_KIND,
            "replay_authentication_binds_every_element": True,
            "replay_authentication_is_collision_free": False,
            "replay_authentication_projection_lanes": (
                runner.QKVO_REPLAY_PROJECTION_LANES
            ),
            "replay_loss_block_count": replay_blocks,
            "replay_plan_evidence": {
                "mode": "token-paths-only-no-cached-logit-tensors",
                "prompt_count": 24,
                "cached_parent_logit_row_count": 0,
                "cached_parent_logit_bytes": 0,
                "cached_child_logit_row_count": 0,
                "cached_child_logit_bytes": 0,
                "cached_tensor_count": 0,
                "cached_tensor_bytes": 0,
                "token_path_scalar_transient_bytes": token_path_scalar_bytes,
                "parent_logits_recomputed_under_no_grad": True,
                "parent_logits_sha256_in_replay_evidence": True,
                "parent_logit_recomputations_per_pass": replay_blocks,
                "parent_logit_recomputations_total": 2 * replay_blocks,
            },
            "accounted_auxiliary_peak_bytes": accounted_peak,
            "max_auxiliary_bytes": 268_435_456,
            "memory_scope": memory_scope,
            "memory_accounting": memory,
            "hotpath_instrumentation": {
                "wrapper_gradient_callback_count": callback_count,
                "callback_sync_evidence": "callback_source_audit",
                "pass_boundary_device_to_host_transfer_call_count": 2,
                "fingerprint_records_hashed": callback_count,
                "fingerprint_values_per_wrapper": (
                    runner.QKVO_FINGERPRINT_VALUES_PER_WRAPPER
                ),
                "pass_full_element_replay_authenticator_sha256": {
                    "pass1": authenticator_sha,
                    "pass2": authenticator_sha,
                },
                "replay_authenticator_kind": runner.QKVO_REPLAY_AUTHENTICATOR_KIND,
                "replay_authenticator_binds_every_element": True,
                "replay_authenticator_is_collision_free": False,
                "replay_authenticator_projection_lanes": (
                    runner.QKVO_REPLAY_PROJECTION_LANES
                ),
                "replay_authenticator_seed": int.from_bytes(
                    hashlib.sha256(
                        (
                            f"{runner.OUTPUT_JACOBIAN_AUDIT_SCHEMA}:{self.seed}:"
                            "full-element-replay-authenticator"
                        ).encode("ascii")
                    ).digest()[:8],
                    "little",
                )
                % (2**63),
                "replay_authenticator_coefficient_sha256": coefficient_sha,
                "replay_authenticator_coefficient_bytes": coefficient_bytes,
                "callback_source_audit": {
                    "kind": runner.QKVO_CALLBACK_SOURCE_AUDIT_KIND,
                    "target_count": len(callback_sources),
                    "target_source_sha256": callback_sources,
                    "call_graph_sha256": runner._canonical_sha256(callback_sources),
                    "banned_host_sync_primitive_count": 0,
                    "runtime_cuda_profiler_used": False,
                    "limitation": "source audit only",
                },
            },
            "aggregate": gate_metrics(self.passed),
            "per_role": {
                role: gate_metrics(role in diagnostic_roles)
                for role in ("q", "k", "v", "o")
            },
            "diagnostic_passed_roles": list(diagnostic_roles),
            "authorized_roles": list(installable_roles),
            "installable_roles": list(installable_roles),
            "role_authorization_policy": runner.QKVO_ROLE_AUTHORIZATION_POLICY,
            "installer_must_ignore_unauthorized_role_tensors": True,
            "all_four_role_tensors_retained_for_diagnostics": True,
            "basis_manifest_sha256": self.basis_manifest_sha256,
        }
        payload["report_sha256"] = runner._canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class _AuditResult:
    report: _AuditGateReport
    basis_payload: Mapping[str, object]

    def basis_pack_payload(self):
        return self.basis_payload


def _fake_basis_pack(
    factor_state_sha256: str,
    *,
    authorized_for_training: bool,
    diagnostic_roles: tuple[str, ...] = ("q", "k", "v", "o"),
    installable_roles: tuple[str, ...] = ("q", "k", "v", "o"),
) -> dict[str, object]:
    specs = (
        ("q_orthogonal_right", (120, 8, 1536)),
        ("k_orthogonal_right", (120, 8, 1536)),
        ("v_orthogonal_right", (120, 8, 1536)),
        ("o_orthogonal_right", (120, 8, 512)),
    )
    tensors = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in specs}
    factors = [
        {
            "name": name,
            "shape": list(shape),
            "dtype": "torch.float32",
            "tensor_sha256": runner._output_jacobian_tensor_sha256(tensors[name]),
            "numel": int(tensors[name].numel()),
            "nbytes": int(tensors[name].numel() * tensors[name].element_size()),
        }
        for name, shape in specs
    ]
    return {
        "schema": runner.GUIDED_QKVO_RIGHT_BASIS_SCHEMA,
        "factor_state_sha256": factor_state_sha256,
        "site_indices": [
            [layer, group]
            for layer in range(runner.TARGET_LAYER_COUNT)
            for group in range(runner.TARGET_PHYSICAL_GROUPS)
        ],
        "site_indices_sha256": runner._canonical_sha256(
            [
                [layer, group]
                for layer in range(runner.TARGET_LAYER_COUNT)
                for group in range(runner.TARGET_PHYSICAL_GROUPS)
            ]
        ),
        "basis_manifest_sha256": runner._canonical_sha256(factors),
        "tensor_count": 4,
        "total_numel": 4_915_200,
        "total_nbytes": 19_660_800,
        "factors": factors,
        "factor_tensors": tensors,
        "contains_model_state": False,
        "contains_parent_state": False,
        "contains_optimizer_state": False,
        "contains_gradients": False,
        "contains_activations": False,
        "contains_hidden_states": False,
        "diagnostic_passed_roles": list(diagnostic_roles),
        "authorized_roles": list(installable_roles),
        "installable_roles": list(installable_roles),
        "authorized_for_training": authorized_for_training,
        "installer_must_ignore_unauthorized_role_tensors": True,
        "all_four_role_tensors_retained_for_diagnostics": True,
    }


class _AuditTransaction(_Transaction):
    def installation_report(self):
        return {
            **super().installation_report(),
            "temporary_lexical_factor_parameters": runner.LEXICAL_FACTOR_PARAMETERS,
            "role_complete_lexical_boundary": True,
            "lexical_embedding_and_lm_head_untied": True,
        }

    def bake_for_publication(self, receipt):
        del receipt
        raise AssertionError("audit-only branch must never bake")


class _AuditRuntime(_Runtime):
    def __init__(
        self,
        *,
        passed: bool = True,
        fail: bool = False,
        diagnostic_roles: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            _TrainingResult(
                fit_metrics=_metrics(),
                candidate_metrics=_metrics(),
                sealed_metrics=_metrics(),
                candidate_gate_passed=True,
                sealed_evaluation_unlocked=True,
            )
        )
        self.transaction = _AuditTransaction(self.model)
        self.audit_calls = 0
        self.train_calls = 0
        self.passed = passed
        self.fail = fail
        self.diagnostic_roles = (
            ("q", "k", "v", "o")
            if diagnostic_roles is None
            else diagnostic_roles
        )

    def run_output_jacobian_audit(self, transaction, parent, contract):
        assert transaction is self.transaction
        assert parent is self.parent
        assert contract.output_jacobian_audit is not None
        self.audit_calls += 1
        if self.fail:
            raise RuntimeError("injected Jacobian audit failure")
        assert contract.continuation is not None
        continuation_tensors = contract.continuation.factor_payload["factor_tensors"]
        restored_clone_bytes = sum(
            int(value.numel() * value.element_size())
            for value in continuation_tensors.values()
        )
        final_passed = self.passed and bool(self.diagnostic_roles)
        installable_roles = self.diagnostic_roles if final_passed else ()
        basis_payload = _fake_basis_pack(
            contract.continuation.factor_state_sha256,
            authorized_for_training=final_passed,
            diagnostic_roles=self.diagnostic_roles,
            installable_roles=installable_roles,
        )
        return runner.OutputJacobianAuditExecution(
            result=_AuditResult(
                _AuditGateReport(
                    self.passed,
                    contract.seed,
                    contract.continuation.factor_state_sha256,
                    str(basis_payload["basis_manifest_sha256"]),
                    self.diagnostic_roles,
                ),
                basis_payload,
            ),
            restore_mode="full",
            restored_factor_state_sha256=(contract.continuation.factor_state_sha256),
            restored_factor_schema=runner.TRAINER_FACTOR_PACK_V2_SCHEMA,
            restored_factor_lexical_rank=runner.LEXICAL_RANK,
            restored_state_clone_tensor_count=len(continuation_tensors),
            restored_state_clone_cpu_bytes=restored_clone_bytes,
            restored_state_released_before_audit=True,
            live_target_native_factor_state_clone_tensor_count_during_audit=0,
            live_target_native_factor_state_clone_cpu_bytes_during_audit=0,
            post_audit_full_factor_capture_helper_called=False,
            post_audit_factor_verification_mode=(
                "audit-v2-streamed-before-after-factor-evidence"
            ),
        )

    def train(self, transaction, parent, contract):
        del transaction, parent, contract
        self.train_calls += 1
        raise AssertionError("audit-only branch must never train")

    def evaluate_baked_behavior(self, model, parent, contract, split_name):
        del model, parent, contract, split_name
        raise AssertionError("audit-only branch must never evaluate promotion gates")

    def save_reload_certify(self, model, tokenizer, output_root):
        del model, tokenizer, output_root
        raise AssertionError("audit-only branch must never publish a checkpoint")


class _MutatedAuditReport:
    def __init__(self, source, mutator) -> None:
        self.source = source
        self.mutator = mutator

    def to_dict(self):
        payload = self.source.to_dict()
        payload.pop("report_sha256")
        self.mutator(payload)
        payload["report_sha256"] = runner._canonical_sha256(payload)
        return payload


class _MutatedAuditRuntime(_AuditRuntime):
    def __init__(
        self,
        *,
        report_mutator=None,
        basis_mutator=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.report_mutator = report_mutator
        self.basis_mutator = basis_mutator

    def run_output_jacobian_audit(self, transaction, parent, contract):
        execution = super().run_output_jacobian_audit(transaction, parent, contract)
        report = execution.result.report
        if self.report_mutator is not None:
            report = _MutatedAuditReport(report, self.report_mutator)
        basis_payload = dict(execution.result.basis_payload)
        if self.basis_mutator is not None:
            self.basis_mutator(basis_payload)
        return runner.OutputJacobianAuditExecution(
            result=_AuditResult(report, basis_payload),
            restore_mode=execution.restore_mode,
            restored_factor_state_sha256=execution.restored_factor_state_sha256,
            restored_factor_schema=execution.restored_factor_schema,
            restored_factor_lexical_rank=execution.restored_factor_lexical_rank,
            restored_state_clone_tensor_count=(
                execution.restored_state_clone_tensor_count
            ),
            restored_state_clone_cpu_bytes=execution.restored_state_clone_cpu_bytes,
            restored_state_released_before_audit=(
                execution.restored_state_released_before_audit
            ),
            live_target_native_factor_state_clone_tensor_count_during_audit=(
                execution.live_target_native_factor_state_clone_tensor_count_during_audit
            ),
            live_target_native_factor_state_clone_cpu_bytes_during_audit=(
                execution.live_target_native_factor_state_clone_cpu_bytes_during_audit
            ),
            post_audit_full_factor_capture_helper_called=(
                execution.post_audit_full_factor_capture_helper_called
            ),
            post_audit_factor_verification_mode=(
                execution.post_audit_factor_verification_mode
            ),
        )


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
    assert "output_jacobian_audit" not in report


def test_output_jacobian_preflight_authenticates_contract_and_sources(
    tmp_path, monkeypatch
) -> None:
    path, digest, factor_state_sha = _audit_contract_files(tmp_path, monkeypatch)

    contract = runner.authenticate_preregistration(path, digest)
    report = contract.preflight_report()

    assert report["output_jacobian_audit"] == _audit_binding_payload()
    assert contract.output_jacobian_audit is not None
    assert contract.output_jacobian_audit.basis_split_role == "candidate"
    assert contract.output_jacobian_audit.prompt_count == 24
    assert contract.continuation is not None
    assert contract.continuation.factor_state_sha256 == factor_state_sha
    assert (
        contract.continuation.trainer_factor_pack_schema
        == runner.TRAINER_FACTOR_PACK_V2_SCHEMA
    )
    assert contract.compiler_cache is not None
    assert contract.compiler_cache.require_hit is True
    for name in (
        "output_jacobian_basis",
        "output_jacobian_sketch",
        "qkvo_output_jacobian_audit",
    ):
        assert report["implementation_authentication"][name]["sha256"]
    assert report["models_loaded"] is False
    assert report["gpu_work_launched"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prompt_count", 23),
        ("basis_split_role", "fit"),
        ("lanes", ["parent-greedy", "sealed"]),
        ("rank", 7),
        ("max_bytes", 268_435_455),
    ),
)
def test_output_jacobian_preflight_rejects_contract_tamper(
    tmp_path, monkeypatch, field, value
) -> None:
    path, _digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    preregistration["output_jacobian_audit"][field] = value

    with pytest.raises(runner.ExactTargetNativeStageError, match="pinned|exactly"):
        runner.authenticate_preregistration(
            path,
            _write_json(path, preregistration),
        )


def test_output_jacobian_source_pin_fails_before_artifact_access(
    tmp_path, monkeypatch
) -> None:
    path, _digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    preregistration["implementation"]["output_jacobian_sketch_sha256"] = "0" * 64
    artifact_accessed = False

    def forbidden_artifact_access(_binding):
        nonlocal artifact_accessed
        artifact_accessed = True
        raise AssertionError("artifact access must remain unreachable")

    monkeypatch.setattr(runner, "_authenticate_artifact", forbidden_artifact_access)
    with pytest.raises(runner.ExactTargetNativeStageError, match="source hash"):
        runner.authenticate_preregistration(
            path,
            _write_json(path, preregistration),
        )
    assert artifact_accessed is False


def test_output_jacobian_requires_authenticated_cache_hit(
    tmp_path, monkeypatch
) -> None:
    path, _digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    preregistration["compiler_cache"]["require_hit"] = False

    with pytest.raises(runner.ExactTargetNativeStageError, match="cache.*hit"):
        runner.authenticate_preregistration(
            path,
            _write_json(path, preregistration),
        )


@pytest.mark.parametrize(
    ("passed", "expected_status"),
    ((True, "diagnostic-passed"), (False, "diagnostic-rejected")),
)
def test_output_jacobian_execute_is_audit_only_and_rolls_back_exactly(
    tmp_path, monkeypatch, passed, expected_status
) -> None:
    path, digest, factor_state_sha = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _AuditRuntime(passed=passed)
    output = tmp_path / f"audit-{passed}"

    report = runner.execute_authenticated_stage(contract, output, runtime=runtime)

    assert report["status"] == expected_status
    assert report["output_jacobian_audit_report"]["passed"] is passed
    assert report["output_jacobian_audit_report_sha256"] == runner._canonical_sha256(
        report["output_jacobian_audit_report"]
    )
    assert contract.continuation is not None
    continuation_tensors = contract.continuation.factor_payload["factor_tensors"]
    expected_restore_clone_bytes = sum(
        int(value.numel() * value.element_size())
        for value in continuation_tensors.values()
    )
    assert report["output_jacobian_restore"] == {
        "mode": "full",
        "factor_pack_schema": runner.TRAINER_FACTOR_PACK_V2_SCHEMA,
        "factor_state_sha256": factor_state_sha,
        "lexical_rank": runner.LEXICAL_RANK,
        "restored_state_clone_tensor_count": len(continuation_tensors),
        "restored_state_clone_cpu_bytes": expected_restore_clone_bytes,
        "restored_state_released_before_audit": True,
        "matches_authenticated_continuation": True,
    }
    assert report["output_jacobian_factor_verification"] == {
        "live_target_native_factor_state_clone_tensor_count_during_audit": 0,
        "live_target_native_factor_state_clone_cpu_bytes_during_audit": 0,
        "post_audit_full_factor_capture_helper_called": False,
        "mode": "audit-v2-streamed-before-after-factor-evidence",
    }
    assert report["sealed_not_accessed"] is True
    assert report["training_started"] is False
    assert report["optimizer_created"] is False
    assert report["optimizer_steps"] == 0
    assert report["training_checkpoints_written"] == 0
    assert report["native_factor_bake_performed"] is False
    assert report["model_checkpoint_written"] is False
    assert report["standalone_compaction_passed"] is False
    assert report["rollback_exact"] is True
    assert runtime.audit_calls == 1
    assert runtime.train_calls == 0
    assert runtime.transaction.rolled_back is True
    assert runtime.transaction.baked is False
    assert runtime.save_calls == 0
    assert runtime.evaluate_calls == []
    assert {artifact.name for artifact in output.iterdir()} == {
        "guided_basis_pack.pt",
        "run_report.json",
    }
    basis_path = output / "guided_basis_pack.pt"
    assert (
        report["guided_basis_pack"]["sha256"]
        == hashlib.sha256(basis_path.read_bytes()).hexdigest()
    )
    envelope = torch.load(basis_path, map_location="cpu", weights_only=True)
    assert envelope["schema"] == runner.GUIDED_BASIS_PACK_SCHEMA
    assert envelope["source_factor_state_sha256"] == factor_state_sha
    assert envelope["contains_model_checkpoint"] is False
    assert envelope["contains_parent_parameters"] is False
    assert envelope["contains_optimizer_state"] is False
    assert envelope["contains_activations_or_hidden_states"] is False
    assert envelope["basis_pack"]["authorized_for_training"] is passed
    assert envelope["basis_pack"]["diagnostic_passed_roles"] == ["q", "k", "v", "o"]
    assert envelope["basis_pack"]["authorized_roles"] == (
        ["q", "k", "v", "o"] if passed else []
    )
    assert envelope["basis_pack"]["installable_roles"] == (
        ["q", "k", "v", "o"] if passed else []
    )
    assert (
        envelope["basis_pack"][
            "installer_must_ignore_unauthorized_role_tensors"
        ]
        is True
    )
    assert (
        envelope["basis_pack"]["all_four_role_tensors_retained_for_diagnostics"]
        is True
    )


def test_output_jacobian_accepts_partial_role_authorization_and_retains_diagnostics(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _AuditRuntime(passed=True, diagnostic_roles=("q", "v"))
    output = tmp_path / "audit-partial-roles"

    report = runner.execute_authenticated_stage(contract, output, runtime=runtime)

    assert report["status"] == "diagnostic-passed"
    assert report["output_jacobian_audit_report"]["diagnostic_passed_roles"] == [
        "q",
        "v",
    ]
    assert report["output_jacobian_audit_report"]["authorized_roles"] == ["q", "v"]
    assert report["output_jacobian_audit_report"]["installable_roles"] == ["q", "v"]
    envelope = torch.load(
        output / "guided_basis_pack.pt", map_location="cpu", weights_only=True
    )
    basis = envelope["basis_pack"]
    assert basis["authorized_for_training"] is True
    assert basis["diagnostic_passed_roles"] == ["q", "v"]
    assert basis["authorized_roles"] == ["q", "v"]
    assert basis["installable_roles"] == ["q", "v"]
    assert set(basis["factor_tensors"]) == {
        "q_orthogonal_right",
        "k_orthogonal_right",
        "v_orthogonal_right",
        "o_orthogonal_right",
    }
    assert basis["installer_must_ignore_unauthorized_role_tensors"] is True
    assert basis["all_four_role_tensors_retained_for_diagnostics"] is True


def test_output_jacobian_aggregate_pass_with_zero_roles_is_diagnostic_rejection(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _AuditRuntime(passed=True, diagnostic_roles=())
    output = tmp_path / "audit-zero-roles"

    report = runner.execute_authenticated_stage(contract, output, runtime=runtime)

    audit = report["output_jacobian_audit_report"]
    assert audit["aggregate"]["passed"] is True
    assert audit["diagnostic_passed_roles"] == []
    assert audit["authorized_roles"] == []
    assert audit["installable_roles"] == []
    assert audit["passed"] is False
    assert report["status"] == "diagnostic-rejected"
    envelope = torch.load(
        output / "guided_basis_pack.pt", map_location="cpu", weights_only=True
    )
    assert envelope["basis_pack"]["authorized_for_training"] is False


def test_output_jacobian_global_rejection_retains_diagnostics_but_installs_nothing(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _AuditRuntime(passed=False, diagnostic_roles=("q", "v"))
    output = tmp_path / "audit-global-reject-diagnostic-roles"

    report = runner.execute_authenticated_stage(contract, output, runtime=runtime)

    audit = report["output_jacobian_audit_report"]
    assert audit["passed"] is False
    assert audit["diagnostic_passed_roles"] == ["q", "v"]
    assert audit["authorized_roles"] == []
    assert audit["installable_roles"] == []
    envelope = torch.load(
        output / "guided_basis_pack.pt", map_location="cpu", weights_only=True
    )
    basis = envelope["basis_pack"]
    assert basis["authorized_for_training"] is False
    assert basis["diagnostic_passed_roles"] == ["q", "v"]
    assert basis["authorized_roles"] == []
    assert basis["installable_roles"] == []


@pytest.mark.parametrize("tamper", ("order", "hash"))
def test_output_jacobian_rejects_noncanonical_site_order_or_hash(
    tmp_path, monkeypatch, tamper
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        if tamper == "order":
            sites = payload["site_indices"]
            sites[0], sites[1] = sites[1], sites[0]
            payload["site_indices_sha256"] = runner._canonical_sha256(sites)
        else:
            payload["site_indices_sha256"] = "0" * 64

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / f"audit-site-{tamper}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="site order/hash"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize("tamper", ("order", "hash"))
def test_output_jacobian_rejects_basis_site_order_or_hash_tamper(
    tmp_path, monkeypatch, tamper
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate_basis(payload):
        if tamper == "order":
            sites = payload["site_indices"]
            sites[0], sites[1] = sites[1], sites[0]
            payload["site_indices_sha256"] = runner._canonical_sha256(sites)
        else:
            payload["site_indices_sha256"] = "0" * 64

    runtime = _MutatedAuditRuntime(basis_mutator=mutate_basis)
    output = tmp_path / f"audit-basis-site-{tamper}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="basis site order"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


def test_output_jacobian_rejects_unauthorized_basis_role_metadata(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate_basis(payload):
        payload["authorized_roles"] = ["q", "k", "v"]

    runtime = _MutatedAuditRuntime(
        passed=True,
        diagnostic_roles=("q", "v"),
        basis_mutator=mutate_basis,
    )
    output = tmp_path / "audit-unauthorized-basis-role"
    with pytest.raises(
        runner.ExactTargetNativeStageError, match="installer authorization metadata"
    ):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("role_authorization_policy", "install-everything"),
        ("installer_must_ignore_unauthorized_role_tensors", False),
        ("all_four_role_tensors_retained_for_diagnostics", False),
    ),
)
def test_output_jacobian_rejects_report_role_policy_tamper(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        payload[field] = bad_value

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / f"audit-role-policy-{field}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="installer"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize("surface", ("report", "basis"))
def test_output_jacobian_rejects_v1_audit_and_basis_schemas(
    tmp_path, monkeypatch, surface
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate_report(payload):
        payload["schema"] = "neural-forge-qkvo-output-jacobian-audit-v1"

    def mutate_basis(payload):
        payload["schema"] = "neural-forge-guided-qkvo-right-basis-v1"

    runtime = _MutatedAuditRuntime(
        report_mutator=mutate_report if surface == "report" else None,
        basis_mutator=mutate_basis if surface == "basis" else None,
    )
    output = tmp_path / f"audit-v1-{surface}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="schema|lifecycle"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("normal_fraction", 0.19),
        ("guided_over_random_ratio", 1.49),
        ("normal_fraction_gate_passed", False),
        ("guided_over_random_gate_passed", False),
        ("passed", False),
    ),
)
def test_output_jacobian_rejects_per_role_gate_tamper(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        payload["per_role"]["q"][field] = bad_value

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / f"audit-role-gate-{field}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="per-role q"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


def test_output_jacobian_rejects_probabilistic_replay_authenticator_disagreement(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        payload["pass2_full_element_replay_authenticator_sha256"] = "7" * 64
        payload["hotpath_instrumentation"][
            "pass_full_element_replay_authenticator_sha256"
        ]["pass2"] = (
            "7" * 64
        )

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / "audit-probabilistic-replay"
    with pytest.raises(runner.ExactTargetNativeStageError, match="probabilistic replay"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("cached_parent_logit_row_count", 1),
        ("cached_parent_logit_bytes", 4),
        ("cached_child_logit_row_count", 1),
        ("cached_child_logit_bytes", 4),
        ("cached_tensor_count", 1),
        ("cached_tensor_bytes", 4),
        ("parent_logits_recomputed_under_no_grad", False),
    ),
)
def test_output_jacobian_rejects_cached_replay_logit_evidence(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        payload["replay_plan_evidence"][field] = bad_value

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / f"audit-cached-replay-logits-{field}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="cached logit"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("callback_sync_evidence", "unverified"),
        ("pass_boundary_device_to_host_transfer_call_count", 3),
        ("wrapper_gradient_callback_count", 7_679),
        ("fingerprint_records_hashed", 7_679),
        ("replay_authenticator_binds_every_element", False),
        ("replay_authenticator_is_collision_free", True),
        ("replay_authenticator_projection_lanes", 1),
    ),
)
def test_output_jacobian_rejects_hotpath_instrumentation_tamper(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        payload["hotpath_instrumentation"][field] = bad_value

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / f"audit-hotpath-{field}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="hot-path"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("finalized_result_tensors_retained_on_accumulator_device", 1),
        ("all_sketches_drained_after_finalize", 0),
        ("actual_final_guided_cpu_pack_bytes", 19_660_796),
        ("basis_payload_cpu_clone_bytes", 19_660_796),
        ("accounted_auxiliary_peak_bytes", 1024),
        ("actual_deferred_fingerprint_buffer_bytes", 8_110_084),
        ("maximum_observed_tensor_hash_cpu_staging_bytes", 4_194_308),
        ("factor_state_retained_cpu_clone_bytes", 4),
        ("replay_plan_cached_parent_logit_bytes", 4),
        ("replay_plan_cached_child_logit_row_count", 1),
        ("replay_plan_cached_tensor_count", 1),
        ("token_path_scalar_transient_bytes", 16),
    ),
)
def test_output_jacobian_rejects_memory_accounting_tamper(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        payload["memory_accounting"][field] = bad_value

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / f"audit-memory-{field}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="auxiliary-memory"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


def test_output_jacobian_rejects_self_consistent_fake_1024_byte_peak(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        payload["accounted_auxiliary_peak_bytes"] = 1024
        payload["memory_accounting"]["accounted_auxiliary_peak_bytes"] = 1024

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / "audit-fake-1024-byte-peak"
    with pytest.raises(runner.ExactTargetNativeStageError, match="auxiliary-memory"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("retained_cpu_clone_bytes_before_sketch", 4),
        ("retained_cpu_clone_bytes_after_audit", 4),
        ("hash_staging_limit_bytes", 4_194_308),
        ("maximum_observed_hash_staging_bytes", 4_194_308),
        ("before_after_captures_overlap", True),
    ),
)
def test_output_jacobian_rejects_factor_capture_bound_tamper(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    def mutate(payload):
        payload["factor_state_capture"][field] = bad_value

    runtime = _MutatedAuditRuntime(report_mutator=mutate)
    output = tmp_path / f"audit-factor-capture-{field}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="factor capture"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert not (output / "guided_basis_pack.pt").exists()


def test_output_jacobian_failure_is_atomic_and_restores_compiled_state(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _AuditRuntime(fail=True)
    output = tmp_path / "audit-failure"

    with pytest.raises(RuntimeError, match="Jacobian audit failure"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)

    assert runtime.audit_calls == 1
    assert runtime.train_calls == 0
    assert runtime.transaction.restored_prepublication is True
    assert runtime.model.state == "compiled"
    assert not (output / "guided_basis_pack.pt").exists()
    failure = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["last_completed_phase"] == "output-jacobian-audit-started"
    assert failure["prepublication_native_restored"] is True
    assert failure["rollback_error"] is None


def test_output_jacobian_pack_save_failure_removes_partial_artifact(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    runtime = _AuditRuntime()
    output = tmp_path / "audit-pack-save-failure"

    def failing_save(path, payload):
        torch.save(dict(payload), path)
        raise OSError("injected basis-pack fsync failure")

    monkeypatch.setattr(runner, "_atomic_torch_save", failing_save)
    with pytest.raises(OSError, match="basis-pack fsync"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)

    assert runtime.transaction.rolled_back is True
    assert runtime.transaction.restored_prepublication is True
    assert not (output / "guided_basis_pack.pt").exists()
    failure = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["rollback_error"] is None


def test_output_jacobian_rejects_unproven_restore_before_pack_write(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    class _BadRestoreRuntime(_AuditRuntime):
        def run_output_jacobian_audit(self, transaction, parent, contract):
            execution = super().run_output_jacobian_audit(transaction, parent, contract)
            return replace(
                execution,
                restored_factor_state_sha256="0" * 64,
            )

    runtime = _BadRestoreRuntime()
    output = tmp_path / "audit-bad-restore"
    with pytest.raises(runner.ExactTargetNativeStageError, match="full authenticated"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert runtime.train_calls == 0
    assert runtime.transaction.restored_prepublication is True
    assert not (output / "guided_basis_pack.pt").exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("restored_state_released_before_audit", False),
        ("live_target_native_factor_state_clone_tensor_count_during_audit", 1),
        ("live_target_native_factor_state_clone_cpu_bytes_during_audit", 4),
        ("post_audit_full_factor_capture_helper_called", True),
        ("post_audit_factor_verification_mode", "full-cpu-recapture"),
    ),
)
def test_output_jacobian_rejects_non_clone_free_execution_attestation(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    class _BadCloneLifecycleRuntime(_AuditRuntime):
        def run_output_jacobian_audit(self, transaction, parent, contract):
            execution = super().run_output_jacobian_audit(
                transaction, parent, contract
            )
            return replace(execution, **{field: bad_value})

    runtime = _BadCloneLifecycleRuntime()
    output = tmp_path / f"audit-bad-clone-lifecycle-{field}"
    with pytest.raises(runner.ExactTargetNativeStageError, match="clone-free"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert runtime.transaction.restored_prepublication is True
    assert not (output / "guided_basis_pack.pt").exists()


def test_output_jacobian_rejects_banned_basis_state_before_pack_write(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    class _BadPackRuntime(_AuditRuntime):
        def run_output_jacobian_audit(self, transaction, parent, contract):
            execution = super().run_output_jacobian_audit(transaction, parent, contract)
            execution.result.basis_payload["contains_optimizer_state"] = True
            return execution

    runtime = _BadPackRuntime()
    output = tmp_path / "audit-bad-pack"
    with pytest.raises(runner.ExactTargetNativeStageError, match="banned"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert runtime.transaction.restored_prepublication is True
    assert not (output / "guided_basis_pack.pt").exists()


def test_output_jacobian_rejects_false_lifecycle_attestation(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    class _TamperedReport:
        def __init__(self, source) -> None:
            self.source = source

        def to_dict(self):
            payload = self.source.to_dict()
            payload["sealed_accessed"] = True
            payload.pop("report_sha256")
            payload["report_sha256"] = runner._canonical_sha256(payload)
            return payload

    class _BadReportRuntime(_AuditRuntime):
        def run_output_jacobian_audit(self, transaction, parent, contract):
            execution = super().run_output_jacobian_audit(transaction, parent, contract)
            return replace(
                execution,
                result=_AuditResult(
                    _TamperedReport(execution.result.report),
                    execution.result.basis_payload,
                ),
            )

    runtime = _BadReportRuntime()
    output = tmp_path / "audit-bad-report"
    with pytest.raises(runner.ExactTargetNativeStageError, match="lifecycle"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert runtime.transaction.restored_prepublication is True
    assert not (output / "guided_basis_pack.pt").exists()


def test_output_jacobian_rejects_decision_that_disagrees_with_metrics(
    tmp_path, monkeypatch
) -> None:
    path, digest, _state = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)

    class _TamperedDecision:
        def __init__(self, source) -> None:
            self.source = source

        def to_dict(self):
            payload = self.source.to_dict()
            payload["passed"] = not payload["passed"]
            payload.pop("report_sha256")
            payload["report_sha256"] = runner._canonical_sha256(payload)
            return payload

    class _BadDecisionRuntime(_AuditRuntime):
        def run_output_jacobian_audit(self, transaction, parent, contract):
            execution = super().run_output_jacobian_audit(
                transaction, parent, contract
            )
            return replace(
                execution,
                result=_AuditResult(
                    _TamperedDecision(execution.result.report),
                    execution.result.basis_payload,
                ),
            )

    runtime = _BadDecisionRuntime()
    output = tmp_path / "audit-bad-decision"
    with pytest.raises(runner.ExactTargetNativeStageError, match="decision"):
        runner.execute_authenticated_stage(contract, output, runtime=runtime)
    assert runtime.transaction.restored_prepublication is True
    assert not (output / "guided_basis_pack.pt").exists()


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


def test_production_runtime_output_jacobian_uses_public_full_restore_and_only_candidate(
    tmp_path, monkeypatch
) -> None:
    path, digest, factor_state_sha = _audit_contract_files(tmp_path, monkeypatch)
    contract = runner.authenticate_preregistration(path, digest)
    transaction = object()
    parent = object()
    captured: dict[str, object] = {"loads": []}

    class _Schedule:
        def __init__(self, **values) -> None:
            captured["schedule"] = values

    class _FactorClone:
        def __init__(self, numel: int, element_size: int) -> None:
            self._numel = numel
            self._element_size = element_size

        def numel(self) -> int:
            return self._numel

        def element_size(self) -> int:
            return self._element_size

    def restore(actual_transaction, payload):
        assert actual_transaction is transaction
        assert payload["factor_state_sha256"] == factor_state_sha
        captured["restore"] = True
        clones = tuple(
            _FactorClone(int(value.numel()), int(value.element_size()))
            for value in payload["factor_tensors"].values()
        )
        captured["restore_clone_refs"] = tuple(weakref.ref(value) for value in clones)
        return SimpleNamespace(
            factor_state_sha256=factor_state_sha,
            schema=runner.TRAINER_FACTOR_PACK_V2_SCHEMA,
            lexical_rank=runner.LEXICAL_RANK,
            tensors=clones,
        )

    def forbidden_capture(_actual_transaction):
        captured["capture_helper_calls"] = (
            int(captured.get("capture_helper_calls", 0)) + 1
        )
        raise AssertionError("audit mode must not recapture the full factor state")

    def loader(name, payload, *, artifact_sha256):
        captured["loads"].append((name, payload, artifact_sha256))
        return SimpleNamespace(split_name=name, prompts=tuple(range(24)))

    trainer_module = SimpleNamespace(
        restore_target_native_factor_payload=restore,
        capture_target_native_factor_state=forbidden_capture,
        load_behavior_domain_suite=loader,
        BehaviorDistillationSchedule=_Schedule,
    )

    class _Config:
        def __init__(self, **values) -> None:
            captured["config"] = values

    class _CanonicalAuditResult:
        pass

    result = _CanonicalAuditResult()

    def audit(*args):
        assert all(
            reference() is None for reference in captured["restore_clone_refs"]
        )
        captured["clone_free_during_audit"] = True
        captured["audit"] = args
        return result

    audit_module = SimpleNamespace(
        QKVOOutputJacobianAuditConfig=_Config,
        QKVOOutputJacobianAuditResult=_CanonicalAuditResult,
        run_qkvo_output_jacobian_audit=audit,
    )

    def module_loader(name):
        if name == runner.CANONICAL_TRAINER_MODULE:
            return trainer_module
        if name == runner.QKVO_OUTPUT_JACOBIAN_AUDIT_MODULE:
            return audit_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(runner.importlib, "import_module", module_loader)
    runtime = runner.ProductionStageRuntime()
    execution = runtime.run_output_jacobian_audit(transaction, parent, contract)

    assert captured["restore"] is True
    assert captured["clone_free_during_audit"] is True
    assert captured.get("capture_helper_calls", 0) == 0
    assert captured["loads"] == [
        (
            "fit",
            contract.splits["candidate"].payload,
            contract.splits["candidate"].sha256,
        )
    ]
    assert captured["config"] == {
        "prompt_count": 24,
        "horizon": 8,
        "lanes": ("parent-greedy", "child-greedy"),
        "rank": 8,
        "sketch_width": 16,
        "seed": contract.seed,
        "min_normal_fraction": 0.20,
        "min_guided_over_random_ratio": 1.50,
        "max_bytes": 268_435_456,
    }
    basis_domain = captured["audit"][2]
    schedule = captured["audit"][3]
    assert captured["audit"] == (
        transaction,
        parent,
        basis_domain,
        schedule,
        captured["audit"][4],
    )
    assert execution.result is result
    assert execution.restore_mode == "full"
    assert execution.restored_factor_state_sha256 == factor_state_sha
    assert execution.restored_factor_lexical_rank == runner.LEXICAL_RANK
    assert execution.restored_state_released_before_audit is True
    assert (
        execution.live_target_native_factor_state_clone_tensor_count_during_audit == 0
    )
    assert execution.live_target_native_factor_state_clone_cpu_bytes_during_audit == 0
    assert execution.post_audit_full_factor_capture_helper_called is False
    assert execution.post_audit_factor_verification_mode == (
        "audit-v2-streamed-before-after-factor-evidence"
    )


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
    result = runtime.evaluate_baked_behavior(child, parent, contract, "candidate")

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
        runner._authenticate_continuation_factor_payload(envelope, lexical_rank=16)


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
    assert (
        sum(
            math.prod(shape)
            for name, shape in lexical_shapes.items()
            if name.startswith("lexical.")
        )
        == runner.LEXICAL_FACTOR_PARAMETERS
    )


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
        runner._authenticate_continuation_factor_payload(envelope, lexical_rank=None)


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

    with pytest.raises(
        runner.ExactTargetNativeStageError, match="requires an exact hit"
    ):
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

        def evaluate_baked_behavior(self, model, parent, contract, split_name):
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
