#!/usr/bin/env python3
"""Reproduce the exact step-16 p35 QKVO winner and pack only its factors.

This runner deliberately stops after update 16, requires the byte-exact adapter
state and exact calibration/heldout behavior from the successful 32-step run,
writes only eight Q/K/V/O factors plus a manifest, reloads those factors into a
fresh adapter, and proves identical calibration/heldout outputs. It never
saves an optimizer, parent tensor, base-model tensor, or full checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import neural_forge.hf_assignment_local_qkvo_residual_adapter as qkvo_adapter_module
import neural_forge.hf_qkvo_factor_pack as factor_pack_module
import neural_forge.hf_qwen_attention_surgery as attention_surgery
import run_qwen_p35_assignment_local_qkvo_residual as qkvo_runner
import run_qwen_p35_functional_residual_adapter as o_runner
import run_qwen_p35_vo_role_scale_search as role_search
import run_qwen_residual_attention_probe as base
from neural_forge.hf_assignment_local_qkvo_residual_adapter import (
    InstalledAssignmentLocalQKVOResidualAdapter,
    install_assignment_local_qkvo_residual_adapter,
)
from neural_forge.hf_functional_decode_trace import FunctionalDecodeLimits
from neural_forge.hf_functional_residual_adapter import (
    evaluate_functional_action_batches,
)
from neural_forge.hf_qkvo_factor_pack import (
    QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
    QKVO_SOURCE_PROVENANCE_SCHEMA,
    adapter_state_sha256,
    file_sha256,
    install_qkvo_factor_pack,
    load_qkvo_factor_pack,
    tensor_sha256,
    write_qkvo_factor_pack,
)
from neural_forge.hf_qwen_attention_surgery import (
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import qwen3_codebook_parameter_report


RUN_SCHEMA = "neural-forge-qwen-p35-qkvo-selected-replay-pack-v1"
EXPECTED_SELECTED_ADAPTER_STATE_SHA256 = (
    "f4cad9b277826a6724d76c9727999dc614ae5c54ef6197b9902d8d4c26c20240"
)
EXPECTED_SOURCE_REPORT_FILE_SHA256 = (
    "a14ccdfc6921d8ba8046d1057b5dad50f3373bba8c5de261b71be6b119d13fb1"
)
EXPECTED_SOURCE_REPORT_SHA256 = (
    "0493bb5114cdfd47ffe36a55a759c266fabd2bbe96198024d11b6c154c88e9bc"
)
EXPECTED_CODEBOOK_MANIFEST_SHA256 = (
    "f62927114d5bc40fa10d239f7eb5d628ba217cd3fe2b7dcac0bdb5fbb68e9b1f"
)
EXPECTED_PYTHON_VERSION = "3.10.12"
EXPECTED_TORCH_VERSION = "2.4.1+cu121"
EXPECTED_GPU_NAME = "Tesla P40"
EXPECTED_CALIBRATION_EVALUATION_SHA256 = (
    "370701ea2b3f521445d8df9adce5bfd403cdeba6ab3af99ddd718d79189ee130"
)
EXPECTED_HELDOUT_EVALUATION_SHA256 = (
    "acdd2449699c25b3ad1eef4a81dcf2f58b774363fb546170113c46c7d4883d49"
)
EXPECTED_HELDOUT_ROLLOUT_COMPARISON_SHA256 = (
    "b8af9fb81de9fd40b88eb33044b74c1f46cec68d1a002079bc36f93acff1ae66"
)
DEFAULT_SOURCE_SUCCESS_REPORT = Path(
    "runs/residual-attention-20260711/"
    "p35_L19G4_to_L21G6_qkvo_residual_rank8_h4_s32_fp16_p40_gpu3.json"
)

EXPECTED_CALIBRATION_EVALUATION = {
    "schema": "neural-forge-functional-residual-adapter-evaluation-v1",
    "split": "calibration-selection",
    "example_count": 48,
    "parent_action_top1_rate": 0.9375,
    "mean_parent_action_rank": 1.0625,
    "mean_parent_action_log_probability": -0.5121914673751841,
    "hard_action_loss": 0.5121914673751841,
    "distribution_loss": 0.6614115284755826,
    "total_loss": 0.5783326243981719,
    "prefix_sources": ["parent-greedy", "student-greedy"],
    "functional_outputs_only": True,
    "attention_residual_cache_weight_similarity_used": False,
    "used_for_selection": True,
    "evaluation_sha256": EXPECTED_CALIBRATION_EVALUATION_SHA256,
}

EXPECTED_HELDOUT_EVALUATION = {
    "schema": "neural-forge-functional-residual-adapter-evaluation-v1",
    "split": "heldout-evaluation",
    "example_count": 64,
    "parent_action_top1_rate": 1.0,
    "mean_parent_action_rank": 1.0,
    "mean_parent_action_log_probability": -0.27040824154391885,
    "hard_action_loss": 0.27040824154391885,
    "distribution_loss": 0.42419432163296733,
    "total_loss": 0.3128276700153947,
    "prefix_sources": ["parent-greedy", "student-greedy"],
    "functional_outputs_only": True,
    "attention_residual_cache_weight_similarity_used": False,
    "used_for_selection": False,
    "evaluation_sha256": EXPECTED_HELDOUT_EVALUATION_SHA256,
}


class QwenP35QKVOSelectedReplayPackError(RuntimeError):
    """The selected-state replay, factor pack, or reload proof failed."""


def _json_value(value: Any) -> Any:
    """Normalize tuple/list representation without changing numeric values."""

    return json.loads(
        json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True)
    )


def _exclusive_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a report without an overwrite race."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise QwenP35QKVOSelectedReplayPackError(
                f"Replay report already exists: {path}."
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _unlink_if_owned(path: Path, expected_sha256: str) -> bool:
    """Delete only the exact bytes this invocation proved it created."""

    resolved = path.resolve()
    if not resolved.exists():
        return True
    try:
        if file_sha256(resolved) != expected_sha256:
            return False
        resolved.unlink()
        return True
    except OSError:
        return False


def load_exact_source_success_report(path: Path) -> dict[str, Any]:
    """Verify the immutable successful-run evidence before replay."""

    resolved = path.resolve()
    if not resolved.is_file() or file_sha256(resolved) != (
        EXPECTED_SOURCE_REPORT_FILE_SHA256
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Source QKVO success report file hash is not exact."
        )
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QwenP35QKVOSelectedReplayPackError(
            "Source QKVO success report is unreadable."
        ) from error
    if (
        not isinstance(report, dict)
        or report.get("report_sha256") != EXPECTED_SOURCE_REPORT_SHA256
        or base._sha256(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
        != EXPECTED_SOURCE_REPORT_SHA256
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Source QKVO success report canonical hash is not exact."
        )
    return report


def assert_exact_execution_environment(
    runtime: nn.Module,
    *,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    """Fail closed unless replay uses the successful run's environment."""

    if (
        platform.python_version() != EXPECTED_PYTHON_VERSION
        or torch.__version__ != EXPECTED_TORCH_VERSION
        or str(device) != "cuda:0"
        or str(dtype) != "float16"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_name(torch.device(device)) != EXPECTED_GPU_NAME
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Exact replay requires Python 3.10.12, torch 2.4.1+cu121, "
            "CUDA_VISIBLE_DEVICES selecting one Tesla P40 as cuda:0, and FP16."
        )
    registry = base._registry(runtime)
    reference = registry.prototype_q_weight
    if reference.device != torch.device(device) or reference.dtype != torch.float16:
        raise QwenP35QKVOSelectedReplayPackError(
            "Loaded p35 registry is not FP16 on the exact replay device."
        )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
    }


def assert_exact_selected_replay(
    report: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> None:
    """Require exact selected state and exact small-gate behavior."""

    training = report.get("training")
    winner = report.get("calibration_winner_lock")
    heldout = report.get("heldout_evaluation")
    if not all(isinstance(value, Mapping) for value in (training, winner, heldout)):
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay report is missing training/winner/heldout contracts."
        )
    assert isinstance(training, Mapping)
    assert isinstance(winner, Mapping)
    assert isinstance(heldout, Mapping)
    history = training.get("calibration_history")
    if not isinstance(history, list) or [value.get("step") for value in history] != [
        0,
        8,
        16,
    ]:
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay did not execute exactly calibration gates 0/8/16."
        )
    selected = history[-1]
    locked_state = artifacts.get("locked_state")
    if (
        training.get("best_step") != 16
        or winner.get("best_step") != 16
        or training.get("adapter_state_sha256")
        != EXPECTED_SELECTED_ADAPTER_STATE_SHA256
        or winner.get("adapter_state_sha256") != EXPECTED_SELECTED_ADAPTER_STATE_SHA256
        or selected.get("adapter_state_sha256")
        != EXPECTED_SELECTED_ADAPTER_STATE_SHA256
        or not isinstance(locked_state, Mapping)
        or adapter_state_sha256(locked_state) != EXPECTED_SELECTED_ADAPTER_STATE_SHA256
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay did not reconstruct exact selected adapter state."
        )
    if _json_value(selected.get("evaluation")) != EXPECTED_CALIBRATION_EVALUATION:
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay calibration metrics are not byte-contract exact."
        )
    post_lock_calibration = _post_lock_calibration_evaluation(report)
    if any(
        post_lock_calibration.get(name) != EXPECTED_CALIBRATION_EVALUATION[name]
        for name in (
            "schema",
            "split",
            "example_count",
            "parent_action_top1_rate",
            "mean_parent_action_rank",
            "prefix_sources",
            "functional_outputs_only",
            "attention_residual_cache_weight_similarity_used",
            "used_for_selection",
        )
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Post-lock calibration changed the selected discrete behavior contract."
        )
    if _json_value(heldout.get("locked_qkvo_action_function")) != (
        EXPECTED_HELDOUT_EVALUATION
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay heldout action evaluation is not exact."
        )
    rollout = heldout.get("locked_qkvo_greedy_rollout")
    if (
        not isinstance(rollout, Mapping)
        or rollout.get("comparison_sha256")
        != EXPECTED_HELDOUT_ROLLOUT_COMPARISON_SHA256
        or rollout.get("exact_rollout_matches") != 8
        or rollout.get("token_matches") != 32
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay heldout greedy outputs are not exact 8/8 and 32/32."
        )
    activation = training.get("parameter_gradient_activation")
    if (
        not isinstance(activation, Mapping)
        or set(activation)
        != {
            "q_left",
            "q_right",
            "k_left",
            "k_right",
            "v_left",
            "v_right",
            "o_left",
            "o_right",
        }
        or not all(value is True for value in activation.values())
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay did not activate all eight Q/K/V/O factor tensors."
        )


def _post_lock_calibration_evaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the evaluation made after the winner state was copied back.

    The step-16 selection-gate evaluation proves which state won.  The trainer
    then copies that locked CPU state back into the live FP16 registry and
    evaluates it once more.  A factor-pack reload must match this post-lock
    evaluation, not continuous diagnostics accumulated at the earlier gate.
    Exact prepack/reload action-logit hashes remain the stronger identity test.
    """

    training = report.get("training")
    if not isinstance(training, Mapping):
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay report has no training contract for post-lock calibration."
        )
    evaluation = _json_value(training.get("final_calibration"))
    if not isinstance(evaluation, dict) or not isinstance(
        evaluation.get("evaluation_sha256"), str
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay report has no valid post-lock calibration evaluation."
        )
    return evaluation


def _functional_discrete_contract(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Extract behavior fields that are invariant to GPU reduction ordering."""

    fields = (
        "schema",
        "split",
        "example_count",
        "parent_action_top1_rate",
        "mean_parent_action_rank",
        "prefix_sources",
        "functional_outputs_only",
        "attention_residual_cache_weight_similarity_used",
        "used_for_selection",
    )
    if not isinstance(evaluation, Mapping) or any(
        name not in evaluation for name in fields
    ):
        raise QwenP35QKVOSelectedReplayPackError(
            "Functional evaluation lacks its discrete behavior contract."
        )
    return _json_value({name: evaluation[name] for name in fields})


def _continuous_evaluation_diagnostics(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Record non-selecting floating aggregates without treating them as identity."""

    fields = (
        "total_loss",
        "hard_action_loss",
        "distribution_loss",
        "mean_parent_action_log_probability",
    )
    return {
        "byte_exact": _json_value(actual) == _json_value(expected),
        "actual_evaluation_sha256": actual.get("evaluation_sha256"),
        "expected_evaluation_sha256": expected.get("evaluation_sha256"),
        "deltas": {
            name: float(actual[name]) - float(expected[name]) for name in fields
        },
        "used_for_identity_or_selection": False,
        "reason": (
            "exact raw action-logit hashes are the identity gate; parallel GPU "
            "loss/log-probability reductions can differ in final floating bits"
        ),
    }


def _registry_geometry(registry: nn.Module) -> dict[str, Any]:
    """Derive adapter geometry from the tentative attention base registry."""

    return {
        "logical_layers": registry.logical_layers,
        "physical_groups": registry.key_value_heads,
        "hidden_size": registry.input_width,
        "query_group_width": registry.query_rows_per_group,
        "head_dim": registry.head_dim,
        "rank": 8,
        "site_count": 1,
    }


def _role_scale_provenance(
    proposed_registry: nn.Module,
    replay_report: Mapping[str, Any],
) -> dict[str, Any]:
    audit = replay_report.get("incumbent_role_scale_audit")
    role_scales = getattr(proposed_registry, "role_scales", None)
    if not isinstance(audit, Mapping) or not isinstance(role_scales, Tensor):
        raise QwenP35QKVOSelectedReplayPackError(
            "Role-scale provenance is unavailable."
        )
    return {
        "schema": QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
        "q": audit["q_scale"],
        "k": audit["k_scale"],
        "v": audit["v_scale"],
        "o": audit["o_scale"],
        "role_scales_trainable": False,
        "table_sha256": tensor_sha256(role_scales),
        "canonical_assignment_axis": audit["canonical_assignment_axis"],
        "source": "tentative-child-base-registry-not-packed",
    }


def _site_manifest(
    site: Any,
    replay_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **site.as_dict(),
        "redirect": dict(replay_report["redirect"]),
        "canonical_assignment_axis": replay_report["incumbent_role_scale_audit"][
            "canonical_assignment_axis"
        ],
    }


def _implementation_provenance() -> dict[str, Any]:
    files = {
        "selected_replay_runner": Path(__file__).resolve(),
        "qkvo_experiment_runner": Path(qkvo_runner.__file__).resolve(),
        "qkvo_adapter_and_trainer": Path(qkvo_adapter_module.__file__).resolve(),
        "factor_pack_module": Path(factor_pack_module.__file__).resolve(),
        "functional_fixture_runner": Path(o_runner.__file__).resolve(),
    }
    rows = [
        {
            "role": role,
            "file": path.name,
            "sha256": file_sha256(path),
        }
        for role, path in sorted(files.items())
    ]
    return {
        "schema": "neural-forge-qkvo-selected-replay-implementation-v1",
        "files": rows,
        "file_count": len(rows),
        "manifest_sha256": base._sha256(rows),
    }


def reproduce_pack_and_verify(
    runtime: nn.Module,
    tokenizer: Any,
    fit_prompts: Sequence[Tensor],
    calibration_prompts: Sequence[Tensor],
    heldout_prompts: Sequence[Tensor],
    *,
    split_provenance: Mapping[str, Any],
    source_success_proof: Mapping[str, Any],
    source_runtime: Mapping[str, Any],
    factor_path: Path,
    manifest_path: Path,
    trace_limits: FunctionalDecodeLimits,
    device: str,
) -> dict[str, Any]:
    """Replay 1..16, write/load factors, verify outputs, and restore p35."""

    del tokenizer  # Broad evaluation is a separate, staged pack consumer.
    artifacts: dict[str, Any] = {}
    replay_report = qkvo_runner.run_qwen_p35_assignment_local_qkvo_residual(
        runtime,
        fit_prompts,
        calibration_prompts,
        heldout_prompts,
        split_provenance=split_provenance,
        rank=qkvo_runner.EXACT_RANK,
        horizon=qkvo_runner.EXACT_HORIZON,
        steps=qkvo_runner.EXACT_SELECTED_REPLAY_STEPS,
        micro_batch_size=qkvo_runner.EXACT_MICRO_BATCH_SIZE,
        q_learning_rate=qkvo_runner.EXACT_Q_LEARNING_RATE,
        k_learning_rate=qkvo_runner.EXACT_K_LEARNING_RATE,
        v_learning_rate=qkvo_runner.EXACT_V_LEARNING_RATE,
        o_learning_rate=qkvo_runner.EXACT_O_LEARNING_RATE,
        calibration_steps=(0, 8, 16),
        initialization_seed=0,
        trace_limits=trace_limits,
        artifact_sink=artifacts,
    )
    assert_exact_selected_replay(replay_report, artifacts)
    post_lock_calibration_evaluation = _post_lock_calibration_evaluation(replay_report)

    parent_registry = base._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    parent_identity = role_search._runtime_state_identity_sha256(runtime)
    parent_installation = attention_surgery._capture_installed(runtime)
    _bridge, _proposal, _candidates, tentative = base._tentative_codebook(
        runtime, (qkvo_runner.EXACT_REDIRECT,)
    )
    codebook, _used_projections = tentative
    proposed_registry = codebook.registry
    site = o_runner._redirected_site(parent_registry, qkvo_runner.EXACT_REDIRECT)
    locked_state = artifacts["locked_state"]
    fixture_sha = split_provenance.get("provenance_sha256")
    if not isinstance(fixture_sha, str):
        raise QwenP35QKVOSelectedReplayPackError(
            "Fixture provenance SHA-256 is missing."
        )
    source_provenance = {
        "schema": QKVO_SOURCE_PROVENANCE_SCHEMA,
        "source_report_file_sha256": EXPECTED_SOURCE_REPORT_FILE_SHA256,
        "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "fixture_provenance_sha256": fixture_sha,
        "calibration_evaluation_sha256": (EXPECTED_CALIBRATION_EVALUATION_SHA256),
        "post_lock_calibration_evaluation_sha256": (
            post_lock_calibration_evaluation["evaluation_sha256"]
        ),
        "heldout_evaluation_sha256": EXPECTED_HELDOUT_EVALUATION_SHA256,
        "heldout_rollout_comparison_sha256": (
            EXPECTED_HELDOUT_ROLLOUT_COMPARISON_SHA256
        ),
        "source_runtime": dict(source_runtime),
        "replay_implementation": _implementation_provenance(),
        "reproduced_environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(torch.device(device)),
            "dtype": "float16",
        },
        "training_schedule": {
            "updates": 16,
            "calibration_steps": [0, 8, 16],
            "horizon": 4,
            "rank": 8,
            "micro_batch_size": 2,
            "initialization_seed": 0,
            "q_learning_rate": 0.001,
            "k_learning_rate": 0.001,
            "v_learning_rate": 0.002,
            "o_learning_rate": 0.002,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "hard_action_weight": 1.0,
            "distribution_weight": 0.1,
            "temperature": 1.0,
            "max_gradient_norm": 1.0,
            "dtype": "float16",
            "update_rule": "manual-adam-per-qkvo-branch",
        },
        "prepack_calibration_action_output_sha256": artifacts[
            "calibration_action_output_manifest"
        ]["manifest_sha256"],
        "prepack_heldout_action_output_sha256": artifacts[
            "heldout_action_output_manifest"
        ]["manifest_sha256"],
    }
    factor_manifest = write_qkvo_factor_pack(
        factor_path,
        manifest_path,
        locked_state,
        selected_adapter_state_sha256=(EXPECTED_SELECTED_ADAPTER_STATE_SHA256),
        site=_site_manifest(site, replay_report),
        rank=8,
        registry_geometry=_registry_geometry(proposed_registry),
        role_scale_provenance=_role_scale_provenance(proposed_registry, replay_report),
        provenance=source_provenance,
    )
    created_output_hashes = {
        factor_path.resolve(): factor_manifest["factor_file_sha256"],
        manifest_path.resolve(): file_sha256(manifest_path.resolve()),
    }
    if (
        factor_manifest.get("factor_tensor_count") != 8
        or factor_manifest.get("factor_parameter_count") != 92_160
        or factor_manifest.get("factor_tensor_bytes") != 184_320
        or factor_manifest.get("optimizer_tensor_count") != 0
        or factor_manifest.get("parent_tensor_count") != 0
        or factor_manifest.get("base_model_tensor_count") != 0
    ):
        for path, digest in created_output_hashes.items():
            _unlink_if_owned(path, digest)
        raise QwenP35QKVOSelectedReplayPackError(
            "Written artifact is not the exact eight-factor 92,160-parameter pack."
        )

    adapter_handle: InstalledAssignmentLocalQKVOResidualAdapter | None = None
    restored = False
    try:
        loaded = load_qkvo_factor_pack(factor_path, manifest_path)
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        reloaded_proposed_snapshot = _json_value(
            qwen3_attention_runtime_snapshot(runtime).as_dict()
        )
        replay_proposed_snapshot = _json_value(
            replay_report["proposed_attention_snapshot"]
        )
        if reloaded_proposed_snapshot != replay_proposed_snapshot:
            raise QwenP35QKVOSelectedReplayPackError(
                "Freshly reconstructed tentative child differs from the child "
                "used to train the selected factors: "
                + json.dumps(
                    {
                        "trained_registry_sha256": replay_proposed_snapshot.get(
                            "registry_sha256"
                        ),
                        "reloaded_registry_sha256": reloaded_proposed_snapshot.get(
                            "registry_sha256"
                        ),
                        "trained_snapshot_sha256": base._sha256(
                            replay_proposed_snapshot
                        ),
                        "reloaded_snapshot_sha256": base._sha256(
                            reloaded_proposed_snapshot
                        ),
                    },
                    sort_keys=True,
                )
            )
        adapter_handle = install_assignment_local_qkvo_residual_adapter(
            runtime,
            proposed_registry,
            sites=(site,),
            rank=8,
            initialization_seed=0,
            require_qwen3_4b_dimensions=True,
            role_scale_table_new_for_redirect=True,
        )
        if any(
            value.dtype != torch.float16
            for value in adapter_handle.adapter_registry.parameters()
        ):
            raise QwenP35QKVOSelectedReplayPackError(
                "Fresh reload registry is not exactly FP16."
            )
        installed_sha = install_qkvo_factor_pack(
            adapter_handle.adapter_registry, loaded
        )
        if installed_sha != EXPECTED_SELECTED_ADAPTER_STATE_SHA256:
            raise QwenP35QKVOSelectedReplayPackError(
                "Reloaded pack did not install the selected state exactly."
            )
        calibration_evaluation = _json_value(
            evaluate_functional_action_batches(
                runtime,
                artifacts["calibration_batches"],
                expected_split="calibration-selection",
                hard_action_weight=1.0,
                distribution_weight=0.1,
                temperature=1.0,
                used_for_selection=True,
            ).as_dict()
        )
        heldout_evaluation = _json_value(
            evaluate_functional_action_batches(
                runtime,
                artifacts["heldout_batches"],
                expected_split="heldout-evaluation",
                hard_action_weight=1.0,
                distribution_weight=0.1,
                temperature=1.0,
                used_for_selection=False,
            ).as_dict()
        )
        calibration_outputs = qkvo_runner._action_output_manifest(
            runtime, artifacts["calibration_batches"]
        )
        heldout_outputs = qkvo_runner._action_output_manifest(
            runtime, artifacts["heldout_batches"]
        )
        reloaded_heldout_greedy, _drivers = o_runner._collect_greedy_split(
            runtime,
            heldout_prompts,
            split="heldout-evaluation",
            horizon=4,
            limits=trace_limits,
        )
        heldout_trace_manifest = o_runner._trace_manifest(reloaded_heldout_greedy)
        reload_checks = {
            "calibration_discrete_evaluation": (
                _functional_discrete_contract(calibration_evaluation)
                == _functional_discrete_contract(post_lock_calibration_evaluation)
            ),
            "heldout_discrete_evaluation": (
                _functional_discrete_contract(heldout_evaluation)
                == _functional_discrete_contract(EXPECTED_HELDOUT_EVALUATION)
            ),
            "calibration_action_outputs": (
                calibration_outputs == artifacts["calibration_action_output_manifest"]
            ),
            "heldout_action_outputs": (
                heldout_outputs == artifacts["heldout_action_output_manifest"]
            ),
            "heldout_greedy_traces": (
                heldout_trace_manifest == artifacts["heldout_greedy_trace_manifest"]
            ),
        }
        if not all(reload_checks.values()):
            diagnostics = {
                "checks": reload_checks,
                "calibration_evaluation_actual_sha256": base._sha256(
                    calibration_evaluation
                ),
                "calibration_evaluation_expected_sha256": base._sha256(
                    post_lock_calibration_evaluation
                ),
                "selection_gate_calibration_evaluation_sha256": base._sha256(
                    EXPECTED_CALIBRATION_EVALUATION
                ),
                "heldout_evaluation_actual_sha256": base._sha256(heldout_evaluation),
                "heldout_evaluation_expected_sha256": base._sha256(
                    EXPECTED_HELDOUT_EVALUATION
                ),
                "calibration_action_outputs_actual_sha256": base._sha256(
                    calibration_outputs
                ),
                "calibration_action_outputs_prepack_sha256": base._sha256(
                    artifacts["calibration_action_output_manifest"]
                ),
                "heldout_action_outputs_actual_sha256": base._sha256(heldout_outputs),
                "heldout_action_outputs_prepack_sha256": base._sha256(
                    artifacts["heldout_action_output_manifest"]
                ),
                "heldout_greedy_traces_actual_sha256": base._sha256(
                    heldout_trace_manifest
                ),
                "heldout_greedy_traces_prepack_sha256": base._sha256(
                    artifacts["heldout_greedy_trace_manifest"]
                ),
            }
            raise QwenP35QKVOSelectedReplayPackError(
                "Reloaded factor pack changed calibration or heldout outputs: "
                + json.dumps(diagnostics, sort_keys=True)
            )
        adapter_handle.assert_live()
        calibration_continuous_diagnostics = _continuous_evaluation_diagnostics(
            calibration_evaluation, post_lock_calibration_evaluation
        )
        heldout_continuous_diagnostics = _continuous_evaluation_diagnostics(
            heldout_evaluation, EXPECTED_HELDOUT_EVALUATION
        )
        reload_verification = {
            "installed_adapter_state_sha256": installed_sha,
            "selection_gate_calibration_evaluation": (EXPECTED_CALIBRATION_EVALUATION),
            "post_lock_calibration_action_evaluation": calibration_evaluation,
            "heldout_action_evaluation": heldout_evaluation,
            "calibration_action_output_manifest": calibration_outputs,
            "heldout_action_output_manifest": heldout_outputs,
            "heldout_greedy_trace_manifest_sha256": base._sha256(
                heldout_trace_manifest
            ),
            "identical_to_prepack_calibration_action_logits": True,
            "identical_to_post_lock_calibration_discrete_contract": True,
            "identical_to_prepack_heldout_action_logits": True,
            "identical_to_source_heldout_discrete_contract": True,
            "identical_to_prepack_heldout_greedy_traces": True,
            "calibration_continuous_diagnostics": (calibration_continuous_diagnostics),
            "heldout_continuous_diagnostics": heldout_continuous_diagnostics,
            "continuous_aggregate_byte_identity_required": False,
            "factor_pack_reconstructed_full_registry_state": True,
        }
    except Exception:
        for path, digest in created_output_hashes.items():
            _unlink_if_owned(path, digest)
        raise
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_error: Exception | None = None
        if adapter_handle is not None and adapter_handle.is_applied:
            try:
                adapter_handle.rollback()
            except Exception as error:  # pragma: no cover
                cleanup_error = error
        try:
            restored = o_runner._restore_parent_exact(
                runtime,
                parent_installation,
                parent_snapshot,
                parent_identity,
            )
            if not restored and cleanup_error is None:
                cleanup_error = QwenP35QKVOSelectedReplayPackError(
                    "Reload verification did not restore p35 exactly."
                )
        except Exception as error:  # pragma: no cover
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            for path, digest in created_output_hashes.items():
                _unlink_if_owned(path, digest)
            if primary_error is not None:
                raise QwenP35QKVOSelectedReplayPackError(
                    "Reload verification failed and p35 cleanup also failed; "
                    f"primary={type(primary_error).__name__}: {primary_error}; "
                    f"cleanup={type(cleanup_error).__name__}: {cleanup_error}"
                ) from cleanup_error
            raise QwenP35QKVOSelectedReplayPackError(
                f"Reload verification cleanup failed: {cleanup_error}"
            ) from cleanup_error

    if not restored:
        for path, digest in created_output_hashes.items():
            _unlink_if_owned(path, digest)
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay completed without exact p35 restoration."
        )
    report = {
        "schema": RUN_SCHEMA,
        "source_success_proof": dict(source_success_proof),
        "source_runtime": dict(source_runtime),
        "selected_replay": replay_report,
        "factor_pack": factor_manifest,
        "reload_verification": reload_verification,
        "broad_evaluation": {
            "status": "separate-staged-pack-consumer-required",
            "reason": (
                "must live-evaluate current-contract p35, then run a freshly "
                "provenance-bound decontaminated zero-loss gate before full suite"
            ),
            "used_for_factor_selection": False,
        },
        "parameter_report_after_restore": qwen3_codebook_parameter_report(
            runtime
        ).as_dict(),
        "transaction": {
            "steps_rerun": 16,
            "optimizer_saved": False,
            "parent_tensors_saved": False,
            "base_model_tensors_saved": False,
            "full_model_checkpoint_saved": False,
            "factor_tensors_saved": 8,
            "incumbent_p35_runtime_restored_exactly": True,
            "candidate_installed_after_return": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    report["report_sha256"] = base._sha256(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--fit-prompts-json", type=Path, default=qkvo_runner.DEFAULT_FIT_PROMPTS
    )
    parser.add_argument(
        "--calibration-selection-prompts-json",
        type=Path,
        default=qkvo_runner.DEFAULT_CALIBRATION_PROMPTS,
    )
    parser.add_argument(
        "--heldout-evaluation-prompts-json",
        type=Path,
        default=qkvo_runner.DEFAULT_HELDOUT_PROMPTS,
    )
    parser.add_argument(
        "--fit-prompt-manifest",
        type=Path,
        default=qkvo_runner.DEFAULT_FIT_MANIFEST,
    )
    parser.add_argument(
        "--audited-prompt-manifest",
        type=Path,
        default=qkvo_runner.DEFAULT_AUDITED_MANIFEST,
    )
    parser.add_argument(
        "--transition-inventory",
        type=Path,
        default=qkvo_runner.DEFAULT_TRANSITION_INVENTORY,
    )
    parser.add_argument(
        "--source-success-report",
        type=Path,
        default=DEFAULT_SOURCE_SUCCESS_REPORT,
    )
    parser.add_argument("--factor-pack", type=Path, required=True)
    parser.add_argument("--factor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-vocabulary", type=int, default=1_000_000)
    parser.add_argument("--max-logit-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.set_defaults(model=None)
    return parser


def _prepare_replay(args: argparse.Namespace) -> tuple[Any, ...]:
    source_success = load_exact_source_success_report(args.source_success_report)
    source_success_proof = {
        "path": str(args.source_success_report.resolve()),
        "file_sha256": EXPECTED_SOURCE_REPORT_FILE_SHA256,
        "report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "best_step": source_success["calibration_winner_lock"]["best_step"],
        "selected_adapter_state_sha256": source_success["calibration_winner_lock"][
            "adapter_state_sha256"
        ],
    }
    fixtures = qkvo_runner.load_verified_prompt_splits(
        args.fit_prompts_json,
        args.calibration_selection_prompts_json,
        args.heldout_evaluation_prompts_json,
        fit_manifest_path=args.fit_prompt_manifest,
        audited_manifest_path=args.audited_prompt_manifest,
        transition_inventory_path=args.transition_inventory,
    )
    if fixtures.provenance != source_success.get("prompt_fixture_provenance"):
        raise QwenP35QKVOSelectedReplayPackError(
            "Replay fixtures differ from the successful run's exact provenance."
        )
    runtime, tokenizer, source = base._load_runtime_and_tokenizer(args)
    if source.get("manifest_sha256") != EXPECTED_CODEBOOK_MANIFEST_SHA256:
        raise QwenP35QKVOSelectedReplayPackError(
            "Loaded p35 codebook manifest differs from the successful run."
        )
    exact_environment = assert_exact_execution_environment(
        runtime, device=args.device, dtype=args.dtype
    )
    fit_prompts = qkvo_runner._encode_prompts(
        tokenizer, fixtures.fit, max_prompt_tokens=args.max_prompt_tokens
    )
    calibration_prompts = qkvo_runner._encode_prompts(
        tokenizer,
        fixtures.calibration,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    heldout_prompts = qkvo_runner._encode_prompts(
        tokenizer, fixtures.heldout, max_prompt_tokens=args.max_prompt_tokens
    )
    limits = FunctionalDecodeLimits(
        max_horizon=4,
        max_prompt_tokens=args.max_prompt_tokens,
        max_vocabulary=args.max_vocabulary,
        max_logit_bytes=args.max_logit_bytes,
        max_model_devices=1,
    )
    return (
        source_success_proof,
        fixtures,
        runtime,
        tokenizer,
        source,
        exact_environment,
        fit_prompts,
        calibration_prompts,
        heldout_prompts,
        limits,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_paths = {
        args.factor_pack.resolve(),
        args.factor_manifest.resolve(),
        args.output.resolve(),
    }
    output_parents = {path.parent for path in output_paths}
    output_parent = next(iter(output_parents)) if len(output_parents) == 1 else None
    if len(output_paths) != 3 or output_parent is None or output_parent.exists():
        raise QwenP35QKVOSelectedReplayPackError(
            "Factor, manifest, and report must be distinct in one absent fresh dir."
        )
    try:
        output_parent.mkdir(parents=False, exist_ok=False)
    except FileExistsError as error:
        raise QwenP35QKVOSelectedReplayPackError(
            "Fresh output directory was concurrently reserved."
        ) from error
    output_directory_owned = True
    pack_owned = False
    report_owned = False
    pack_hashes: dict[Path, str] = {}
    try:
        (
            source_success_proof,
            fixtures,
            runtime,
            tokenizer,
            source,
            exact_environment,
            fit_prompts,
            calibration_prompts,
            heldout_prompts,
            limits,
        ) = _prepare_replay(args)
    except Exception:
        try:
            output_parent.rmdir()
        except OSError:
            # A non-owned path appeared inside our reserved directory.
            pass
        raise
    try:
        report = reproduce_pack_and_verify(
            runtime,
            tokenizer,
            fit_prompts,
            calibration_prompts,
            heldout_prompts,
            split_provenance=fixtures.provenance,
            source_success_proof=source_success_proof,
            source_runtime=source,
            factor_path=args.factor_pack,
            manifest_path=args.factor_manifest,
            trace_limits=limits,
            device=args.device,
        )
        pack_owned = True
        pack_hashes = {
            args.factor_pack.resolve(): report["factor_pack"]["factor_file_sha256"],
            args.factor_manifest.resolve(): file_sha256(args.factor_manifest.resolve()),
        }
        report["environment"] = exact_environment
        report["report_sha256"] = base._sha256(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
        _exclusive_atomic_json(args.output, report)
        report_owned = True
    except Exception:
        if pack_owned:
            for path, digest in pack_hashes.items():
                _unlink_if_owned(path, digest)
        if report_owned:
            args.output.resolve().unlink(missing_ok=True)
        if output_directory_owned:
            try:
                output_parent.rmdir()
            except OSError:
                # A path not owned by this invocation appeared; never delete it.
                pass
        raise
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "factor_pack": str(args.factor_pack.resolve()),
                "factor_manifest": str(args.factor_manifest.resolve()),
                "report_sha256": report["report_sha256"],
                "selected_adapter_state_sha256": (
                    EXPECTED_SELECTED_ADAPTER_STATE_SHA256
                ),
                "calibration_top1": 0.9375,
                "heldout_top1": 1.0,
                "runtime_restored": True,
                "checkpoint_saved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
