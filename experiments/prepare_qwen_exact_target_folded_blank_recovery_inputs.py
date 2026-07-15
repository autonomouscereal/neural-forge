#!/usr/bin/env python3
"""Freeze teacher replay and the strongest exact-target recovery A/B plan.

This is the only preparation step that loads the parent model or performs
teacher generation.  It publishes no accepted model and never reads a sealed
evaluation split.  The resulting JSON files are independently verifiable
without loading a model, tokenizer, or GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_qwen_exact_target_native_stage as stage
from neural_forge import hf_recovery, hf_teacher_token_replay
from neural_forge.hf_guided_checkpoint_recovery import (
    canonical_json_bytes,
    canonical_sha256,
)


REPLAY_WRAPPER_SCHEMA = "neural-forge-frozen-recovery-replay-bank-v2"
RECOVERY_PLAN_SCHEMA = "neural-forge-folded-blank-recovery-plan-v2"
BEHAVIOR_SUITE_BINDING_SCHEMA = (
    "neural-forge-candidate-recovery-behavior-suite-binding-v1"
)
TEACHER_SOURCE_BINDING_SCHEMA = "neural-forge-teacher-source-artifact-binding-v1"
BUILD_REPORT_SCHEMA = "neural-forge-folded-blank-recovery-input-build-v1"

REPLAY_FILENAME = "replay_bank.json"
PLAN_FILENAME = "recovery_plan.json"
BUILD_REPORT_FILENAME = "build_report.json"

STEPS = 180
BATCH_SIZE = 1
MAX_LENGTH = 128
CHECKPOINT_STEPS = (0, 1, 4, 8, 16, 32, 64, 96, 128, 180)
RETAINED_CHECKPOINT_STEPS: tuple[int, ...] = ()
CALIBRATION_ORDER = "original"
RESPONSE_LOSS_MODE = "auto"
STEPWISE_DONOR_MAX_NEW_TOKENS = 16
ON_POLICY_MAX_NEW_TOKENS = 24
TEACHER_GENERATE_MAX_NEW_TOKENS = 48
ON_POLICY_INTERVAL = 8
RECOVERY_SEEDS = {
    "python": 20_260_715,
    "torch": 20_260_715,
    "cuda": 20_260_715,
    "data_order": 20_260_715,
}

_SHA256_HEX = frozenset("0123456789abcdef")
_REPLAY_FIELDS = frozenset(
    {
        "schema",
        "status",
        "teacher_token_replay",
        "teacher_source_artifact",
        "behavior_eval_suite",
        "frozen_before_preregistration",
        "contains_sealed_data",
        "publication_authorized",
        "replay_payload_sha256",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "schema",
        "status",
        "recovery_kwargs",
        "seeds",
        "checkpoint_steps",
        "retained_checkpoint_steps",
        "teacher_token_replay_payload_sha256",
        "zero_step_before_optimizer",
        "reset_rng_before_each_arm",
        "publication_authorized",
        "sealed_data_accessed",
        "plan_payload_sha256",
    }
)
_BEHAVIOR_FIELDS = frozenset(
    {
        "schema",
        "status",
        "split",
        "source_filename",
        "source_file_sha256",
        "suite",
        "suite_sha256",
        "sealed_data_accessed",
        "contains_sealed_data",
        "publication_authorized",
        "payload_sha256",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "schema",
        "expected_parent_artifact_tree_sha256",
        "actual_parent_artifact_tree_sha256",
        "artifact_tree_file_count",
        "artifact_tree_total_bytes",
        "tree_match_verified_before_model_load",
        "source_model_state_dict_sha256",
        "sha256",
    }
)
_BUILD_FIELDS = frozenset(
    {
        "schema",
        "status",
        "outputs",
        "teacher_token_replay_schema",
        "teacher_token_replay_payload_sha256",
        "teacher_source_artifact",
        "behavior_eval_suite_payload_sha256",
        "calibration_text_count",
        "calibration_texts_sha256",
        "recovery_plan_payload_sha256",
        "recovery_steps",
        "checkpoint_steps",
        "retained_checkpoint_steps",
        "source_torch_dtype",
        "candidate_recovery_torch_dtype",
        "teacher_generation_device",
        "recovery_device",
        "local_files_only",
        "trust_remote_code",
        "model_weights_loaded",
        "teacher_generation_completed",
        "gpu_work_launched",
        "training_launched",
        "optimizer_created",
        "sealed_data_accessed",
        "publication_authorized",
        "accepted_artifact",
        "model_free_verification_passed",
        "atomic_publication",
        "build_report_sha256",
    }
)


class RecoveryInputPreparationError(RuntimeError):
    """Frozen recovery inputs could not be authenticated and published."""


@dataclass(frozen=True, slots=True)
class TeacherComponents:
    """The only live objects used during the explicit teacher-generation phase."""

    torch: Any
    source_model: Any
    source_tokenizer: Any
    candidate_tokenizer: Any


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise RecoveryInputPreparationError(f"{name} must be a lowercase SHA-256.")
    return value


def _strict_fields(
    value: Any,
    fields: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise RecoveryInputPreparationError(
            f"{name} fields changed; expected={sorted(fields)!r}, actual={actual!r}."
        )
    return value


def _json_value(value: Any, *, name: str = "value") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RecoveryInputPreparationError(f"{name} has a non-string key.")
            if key in result:
                raise RecoveryInputPreparationError(
                    f"{name} has duplicate key {key!r}."
                )
            result[key] = _json_value(item, name=f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise RecoveryInputPreparationError(f"{name} is not finite canonical JSON.")


def _strict_json_file(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryInputPreparationError(
                    f"Duplicate JSON key {key!r} in {path.name}."
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                RecoveryInputPreparationError(
                    f"Non-finite JSON constant {constant!r} in {path.name}."
                )
            ),
        )
    except RecoveryInputPreparationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryInputPreparationError(
            f"Cannot read strict JSON from {path}."
        ) from error
    if not isinstance(value, dict):
        raise RecoveryInputPreparationError(
            f"{path.name} must contain one JSON object."
        )
    canonical_json_bytes(value)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _existing_local_directory(value: Path | str, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RecoveryInputPreparationError(
            f"{name} must be an existing local directory."
        )
    return path


def _existing_real_file(value: Path | str, name: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise RecoveryInputPreparationError(f"{name} cannot be a symbolic link.")
    path = raw.resolve()
    if not path.is_file():
        raise RecoveryInputPreparationError(f"{name} must be an existing real file.")
    return path


def _absent_output_directory(value: Path | str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() or path.name in {"", ".", ".."}:
        raise RecoveryInputPreparationError(
            "output_dir must be one absent, explicitly named directory."
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RecoveryInputPreparationError(
            "output_dir parent must be an existing real directory."
        )
    return path


def load_teacher_components(
    *,
    source_model_path: Path,
    source_tokenizer_path: Path,
    candidate_tokenizer_path: Path,
    device: str,
    local_files_only: bool,
    trust_remote_code: bool,
    source_torch_dtype: str,
) -> TeacherComponents:
    """Load exactly one local FP16 source and two local trusted tokenizers."""

    if local_files_only is not True or trust_remote_code is not False:
        raise RecoveryInputPreparationError(
            "Teacher preparation requires local_files_only=True and trust_remote_code=False."
        )
    if source_torch_dtype != "float16":
        raise RecoveryInputPreparationError("Teacher source dtype must be float16.")
    torch, transformers = hf_recovery._torch_and_transformers()
    tokenizer_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    source_tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(source_tokenizer_path),
        **tokenizer_kwargs,
    )
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(candidate_tokenizer_path),
        **tokenizer_kwargs,
    )
    source_model = transformers.AutoModelForCausalLM.from_pretrained(
        str(source_model_path),
        torch_dtype=torch.float16,
        device_map=None,
        local_files_only=True,
        trust_remote_code=False,
    )
    source_model.to(device)
    source_model.eval()
    return TeacherComponents(
        torch=torch,
        source_model=source_model,
        source_tokenizer=source_tokenizer,
        candidate_tokenizer=candidate_tokenizer,
    )


def strongest_recovery_kwargs(device: str) -> dict[str, Any]:
    """Return the pinned strongest known full-model 180-step recovery recipe."""

    if not isinstance(device, str) or not device.strip():
        raise RecoveryInputPreparationError("device must be a nonblank string.")
    return {
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "lr": 5e-7,
        "anchor_kd_weight": 0.04,
        "lm_weight": 0.08,
        "first_token_lm_weight": 0.05,
        "response_end_lm_weight": 0.05,
        "focused_lm_weight": 0.04,
        "dual_space_kd_weight": 0.08,
        "attention_influence_weight": 0.02,
        "attention_influence_layer": "middle",
        "hidden_kd_weight": 0.06,
        "projected_vocab_kd_weight": 0.04,
        "focused_projected_vocab_kd_weight": 0.04,
        "span_hidden_kd_weight": 0.10,
        "token_kd_weight": 0.05,
        "token_focus": "numbers",
        "token_teacher_confidence_floor": 0.05,
        "answer_contrastive_weight": 0.10,
        "answer_first_token_margin_weight": 0.08,
        "answer_first_token_margin": 1.5,
        "teacher_rank_weight": 0.08,
        "teacher_rank_temperature": 0.7,
        "teacher_generate_lm_weight": 0.015,
        "teacher_generate_max_new_tokens": TEACHER_GENERATE_MAX_NEW_TOKENS,
        "teacher_generate_loss_mode": "none",
        "stepwise_donor_kd_weight": 0.025,
        "stepwise_donor_max_new_tokens": STEPWISE_DONOR_MAX_NEW_TOKENS,
        "on_policy_anchor_kd_weight": 0.01,
        "on_policy_unlikelihood_weight": 0.002,
        "on_policy_hidden_kd_weight": 0.02,
        "on_policy_max_new_tokens": ON_POLICY_MAX_NEW_TOKENS,
        "on_policy_interval": ON_POLICY_INTERVAL,
        "response_loss_mode": RESPONSE_LOSS_MODE,
        "calibration_order": CALIBRATION_ORDER,
        "train_scope": "all",
        "torch_dtype": "float16",
        "candidate_torch_dtype": "float32",
        "device": device,
        "local_files_only": True,
        "trust_remote_code": False,
        "save_best_loss": False,
        "save_best_behavior": False,
        "behavior_eval_checkpoint_steps": list(CHECKPOINT_STEPS),
        "adamw_foreach": False,
    }


def _behavior_suite_binding(path: Path) -> dict[str, Any]:
    raw = _strict_json_file(path)
    if (
        raw.get("split") != "candidate"
        or raw.get("sealed_data_accessed") is not False
        or raw.get("contains_sealed_data", False) is not False
        or raw.get("publication_authorized", False) is not False
        or raw.get("sealed_before_model_access", False) is not False
    ):
        raise RecoveryInputPreparationError(
            "Behavior suite must explicitly be candidate-only, sealed-free, and unpublished."
        )
    suite_input = raw.get("suite", raw)
    if not isinstance(suite_input, dict) or not isinstance(
        suite_input.get("behavior_probes"), list
    ):
        raise RecoveryInputPreparationError(
            "Candidate behavior suite must explicitly contain behavior_probes."
        )
    loaded, behavior_probes, generation_prompts = (
        hf_recovery._recovery_behavior_eval_suite(raw)
    )
    if (
        loaded is None
        or not behavior_probes
        or not generation_prompts
        or any(
            not isinstance(item.get("prompt"), str) or not item["prompt"].strip()
            for item in (*behavior_probes, *generation_prompts)
        )
    ):
        raise RecoveryInputPreparationError(
            "Candidate behavior suite needs nonblank behavior and generation prompts."
        )
    behavior_ids = [str(item.get("id")) for item in behavior_probes]
    generation_ids = [str(item.get("id")) for item in generation_prompts]
    if len(set(behavior_ids)) != len(behavior_ids) or len(set(generation_ids)) != len(
        generation_ids
    ):
        raise RecoveryInputPreparationError(
            "Candidate behavior and generation prompt IDs must be unique."
        )
    normalized = _json_value(loaded, name="behavior_eval_suite.suite")
    normalized["source"] = path.name
    body = {
        "schema": BEHAVIOR_SUITE_BINDING_SCHEMA,
        "status": "frozen",
        "split": "candidate",
        "source_filename": path.name,
        "source_file_sha256": _file_sha256(path),
        "suite": normalized,
        "suite_sha256": canonical_sha256(normalized),
        "sealed_data_accessed": False,
        "contains_sealed_data": False,
        "publication_authorized": False,
    }
    return {**body, "payload_sha256": canonical_sha256(body)}


def verify_behavior_suite_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_value(payload, name="behavior_eval_suite")
    row = _strict_fields(value, _BEHAVIOR_FIELDS, "behavior_eval_suite")
    body = dict(row)
    claimed = _require_sha256(body.pop("payload_sha256"), "behavior payload SHA")
    suite = row["suite"]
    if not isinstance(suite, Mapping):
        raise RecoveryInputPreparationError(
            "behavior_eval_suite.suite must be an object."
        )
    behavior_probes = suite.get("behavior_probes")
    generation_prompts = suite.get("generation_prompts")
    if (
        row.get("schema") != BEHAVIOR_SUITE_BINDING_SCHEMA
        or row.get("status") != "frozen"
        or row.get("split") != "candidate"
        or row.get("sealed_data_accessed") is not False
        or row.get("contains_sealed_data") is not False
        or row.get("publication_authorized") is not False
        or not isinstance(behavior_probes, list)
        or not behavior_probes
        or not isinstance(generation_prompts, list)
        or not generation_prompts
        or canonical_sha256(suite) != row.get("suite_sha256")
        or canonical_sha256(body) != claimed
    ):
        raise RecoveryInputPreparationError(
            "Candidate behavior suite binding failed authentication."
        )
    _require_sha256(row["source_file_sha256"], "behavior source file SHA")
    _require_sha256(row["suite_sha256"], "behavior suite SHA")
    return value


def _source_binding(
    *,
    expected_tree_sha256: str,
    actual_tree: Mapping[str, Any],
    state_dict_sha256: str,
) -> dict[str, Any]:
    actual = _require_sha256(actual_tree.get("tree_sha256"), "actual parent tree SHA")
    expected = _require_sha256(expected_tree_sha256, "expected parent tree SHA")
    if actual != expected:
        raise RecoveryInputPreparationError(
            "Teacher source artifact tree differs from the canonical parent lineage."
        )
    state = _require_sha256(state_dict_sha256, "teacher source state_dict SHA")
    body = {
        "schema": TEACHER_SOURCE_BINDING_SCHEMA,
        "expected_parent_artifact_tree_sha256": expected,
        "actual_parent_artifact_tree_sha256": actual,
        "artifact_tree_file_count": int(actual_tree["file_count"]),
        "artifact_tree_total_bytes": int(actual_tree["total_bytes"]),
        "tree_match_verified_before_model_load": True,
        "source_model_state_dict_sha256": state,
    }
    return {**body, "sha256": canonical_sha256(body)}


def _verify_source_binding(
    payload: Mapping[str, Any],
    *,
    teacher_state_dict_sha256: str,
) -> dict[str, Any]:
    value = _json_value(payload, name="teacher_source_artifact")
    row = _strict_fields(value, _SOURCE_FIELDS, "teacher_source_artifact")
    body = dict(row)
    claimed = _require_sha256(body.pop("sha256"), "teacher source binding SHA")
    expected = _require_sha256(
        row["expected_parent_artifact_tree_sha256"],
        "expected parent tree SHA",
    )
    actual = _require_sha256(
        row["actual_parent_artifact_tree_sha256"],
        "actual parent tree SHA",
    )
    state = _require_sha256(
        row["source_model_state_dict_sha256"],
        "teacher source state_dict SHA",
    )
    if (
        row.get("schema") != TEACHER_SOURCE_BINDING_SCHEMA
        or expected != actual
        or state != teacher_state_dict_sha256
        or row.get("tree_match_verified_before_model_load") is not True
        or isinstance(row.get("artifact_tree_file_count"), bool)
        or not isinstance(row.get("artifact_tree_file_count"), int)
        or row["artifact_tree_file_count"] <= 0
        or isinstance(row.get("artifact_tree_total_bytes"), bool)
        or not isinstance(row.get("artifact_tree_total_bytes"), int)
        or row["artifact_tree_total_bytes"] <= 0
        or canonical_sha256(body) != claimed
    ):
        raise RecoveryInputPreparationError(
            "Teacher source artifact binding failed authentication."
        )
    return value


def _teacher_parameters(device: str) -> dict[str, Any]:
    return {
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "response_loss_mode": RESPONSE_LOSS_MODE,
        "calibration_order": CALIBRATION_ORDER,
        "stepwise_donor_max_new_tokens": STEPWISE_DONOR_MAX_NEW_TOKENS,
        "on_policy_max_new_tokens": ON_POLICY_MAX_NEW_TOKENS,
        "teacher_generate_max_new_tokens": TEACHER_GENERATE_MAX_NEW_TOKENS,
        "on_policy_interval": ON_POLICY_INTERVAL,
        "device": device,
    }


def _replay_wrapper(
    *,
    teacher_payload: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    behavior_binding: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": REPLAY_WRAPPER_SCHEMA,
        "status": "frozen",
        "teacher_token_replay": _json_value(
            teacher_payload,
            name="teacher_token_replay",
        ),
        "teacher_source_artifact": _json_value(
            source_binding,
            name="teacher_source_artifact",
        ),
        "behavior_eval_suite": _json_value(
            behavior_binding,
            name="behavior_eval_suite",
        ),
        "frozen_before_preregistration": True,
        "contains_sealed_data": False,
        "publication_authorized": False,
    }
    return {**body, "replay_payload_sha256": canonical_sha256(body)}


def verify_frozen_replay_wrapper(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_value(payload, name="replay_bank")
    row = _strict_fields(value, _REPLAY_FIELDS, "replay_bank")
    body = dict(row)
    claimed = _require_sha256(body.pop("replay_payload_sha256"), "replay payload SHA")
    try:
        verified_teacher = hf_teacher_token_replay.verify_teacher_token_replay_payload(
            row["teacher_token_replay"]
        )
    except (TypeError, ValueError) as error:
        raise RecoveryInputPreparationError(
            "Embedded teacher token replay failed authentication."
        ) from error
    teacher = _json_value(verified_teacher, name="verified_teacher_token_replay")
    state_sha = teacher["source_model_fingerprint"]["model_state_fingerprint"][
        "state_dict_sha256"
    ]
    _verify_source_binding(
        row["teacher_source_artifact"],
        teacher_state_dict_sha256=state_sha,
    )
    verify_behavior_suite_binding(row["behavior_eval_suite"])
    if (
        row.get("schema") != REPLAY_WRAPPER_SCHEMA
        or row.get("status") != "frozen"
        or row.get("frozen_before_preregistration") is not True
        or row.get("contains_sealed_data") is not False
        or row.get("publication_authorized") is not False
        or canonical_sha256(body) != claimed
    ):
        raise RecoveryInputPreparationError(
            "Frozen replay wrapper failed self-authentication."
        )
    return value


def _recovery_plan(
    teacher_payload: Mapping[str, Any], recovery_device: str
) -> dict[str, Any]:
    body = {
        "schema": RECOVERY_PLAN_SCHEMA,
        "status": "preregistered",
        "recovery_kwargs": strongest_recovery_kwargs(recovery_device),
        "seeds": dict(RECOVERY_SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "retained_checkpoint_steps": list(RETAINED_CHECKPOINT_STEPS),
        "teacher_token_replay_payload_sha256": _require_sha256(
            teacher_payload.get("payload_sha256"),
            "teacher token replay payload SHA",
        ),
        "zero_step_before_optimizer": True,
        "reset_rng_before_each_arm": True,
        "publication_authorized": False,
        "sealed_data_accessed": False,
    }
    return {**body, "plan_payload_sha256": canonical_sha256(body)}


def verify_recovery_plan(
    payload: Mapping[str, Any],
    *,
    teacher_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = _json_value(payload, name="recovery_plan")
    row = _strict_fields(value, _PLAN_FIELDS, "recovery_plan")
    body = dict(row)
    claimed = _require_sha256(body.pop("plan_payload_sha256"), "plan payload SHA")
    kwargs = row["recovery_kwargs"]
    if not isinstance(kwargs, Mapping):
        raise RecoveryInputPreparationError("recovery_kwargs must be an object.")
    recovery_device = kwargs.get("device")
    if kwargs != strongest_recovery_kwargs(recovery_device):
        raise RecoveryInputPreparationError(
            "Recovery kwargs differ from the pinned strongest known recipe."
        )
    teacher_sha = _require_sha256(
        row["teacher_token_replay_payload_sha256"],
        "teacher token replay payload SHA",
    )
    if teacher_payload is not None:
        try:
            verified_teacher = (
                hf_teacher_token_replay.verify_teacher_token_replay_payload(
                    teacher_payload
                )
            )
        except (TypeError, ValueError) as error:
            raise RecoveryInputPreparationError(
                "Teacher token replay failed plan binding verification."
            ) from error
        teacher = _json_value(verified_teacher, name="teacher_token_replay")
        if teacher_sha != teacher["payload_sha256"]:
            raise RecoveryInputPreparationError(
                "Recovery plan is bound to a different teacher replay payload."
            )
        parameters = teacher["parameters"]
        teacher_device = parameters.get("device")
        if not isinstance(teacher_device, str) or not teacher_device.strip():
            raise RecoveryInputPreparationError(
                "Teacher replay device must be a nonblank string."
            )
        for name, expected in _teacher_parameters(teacher_device).items():
            if parameters.get(name) != expected or (
                name != "device" and kwargs.get(name) != expected
            ):
                raise RecoveryInputPreparationError(
                    f"Recovery plan and teacher replay disagree on {name}."
                )
    if (
        row.get("schema") != RECOVERY_PLAN_SCHEMA
        or row.get("status") != "preregistered"
        or row.get("seeds") != RECOVERY_SEEDS
        or row.get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or row.get("retained_checkpoint_steps") != list(RETAINED_CHECKPOINT_STEPS)
        or row.get("zero_step_before_optimizer") is not True
        or row.get("reset_rng_before_each_arm") is not True
        or row.get("publication_authorized") is not False
        or row.get("sealed_data_accessed") is not False
        or canonical_sha256(body) != claimed
    ):
        raise RecoveryInputPreparationError(
            "Recovery plan failed sealed-free deterministic authentication."
        )
    return value


def _build_report(
    *,
    replay_path: Path,
    plan_path: Path,
    teacher_payload: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    behavior_binding: Mapping[str, Any],
    plan_payload: Mapping[str, Any],
    teacher_generation_device: str,
    recovery_device: str,
) -> dict[str, Any]:
    body = {
        "schema": BUILD_REPORT_SCHEMA,
        "status": "ready-experimental-unaccepted",
        "outputs": {
            "replay_bank": {
                "filename": REPLAY_FILENAME,
                "file_sha256": _file_sha256(replay_path),
            },
            "recovery_plan": {
                "filename": PLAN_FILENAME,
                "file_sha256": _file_sha256(plan_path),
            },
            "build_report": {"filename": BUILD_REPORT_FILENAME},
        },
        "teacher_token_replay_schema": teacher_payload["schema"],
        "teacher_token_replay_payload_sha256": teacher_payload["payload_sha256"],
        "teacher_source_artifact": _json_value(source_binding),
        "behavior_eval_suite_payload_sha256": behavior_binding["payload_sha256"],
        "calibration_text_count": len(teacher_payload["calibration_texts"]),
        "calibration_texts_sha256": teacher_payload["provenance_sha256"][
            "calibration_texts"
        ],
        "recovery_plan_payload_sha256": plan_payload["plan_payload_sha256"],
        "recovery_steps": STEPS,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "retained_checkpoint_steps": list(RETAINED_CHECKPOINT_STEPS),
        "source_torch_dtype": "float16",
        "candidate_recovery_torch_dtype": "float32",
        "teacher_generation_device": teacher_generation_device,
        "recovery_device": recovery_device,
        "local_files_only": True,
        "trust_remote_code": False,
        "model_weights_loaded": True,
        "teacher_generation_completed": True,
        "gpu_work_launched": teacher_generation_device.casefold().startswith("cuda"),
        "training_launched": False,
        "optimizer_created": False,
        "sealed_data_accessed": False,
        "publication_authorized": False,
        "accepted_artifact": False,
        "model_free_verification_passed": True,
        "atomic_publication": True,
    }
    return {**body, "build_report_sha256": canonical_sha256(body)}


def verify_build_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_value(payload, name="build_report")
    row = _strict_fields(value, _BUILD_FIELDS, "build_report")
    body = dict(row)
    claimed = _require_sha256(body.pop("build_report_sha256"), "build report SHA")
    outputs = row.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "replay_bank",
        "recovery_plan",
        "build_report",
    }:
        raise RecoveryInputPreparationError("Build report output bindings changed.")
    _require_sha256(
        outputs["replay_bank"].get("file_sha256"),
        "replay bank file SHA",
    )
    _require_sha256(
        outputs["recovery_plan"].get("file_sha256"),
        "recovery plan file SHA",
    )
    _verify_source_binding(
        row["teacher_source_artifact"],
        teacher_state_dict_sha256=row["teacher_source_artifact"][
            "source_model_state_dict_sha256"
        ],
    )
    if (
        row.get("schema") != BUILD_REPORT_SCHEMA
        or row.get("status") != "ready-experimental-unaccepted"
        or outputs["replay_bank"].get("filename") != REPLAY_FILENAME
        or outputs["recovery_plan"].get("filename") != PLAN_FILENAME
        or outputs["build_report"] != {"filename": BUILD_REPORT_FILENAME}
        or row.get("recovery_steps") != STEPS
        or row.get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or row.get("retained_checkpoint_steps") != []
        or row.get("source_torch_dtype") != "float16"
        or row.get("candidate_recovery_torch_dtype") != "float32"
        or row.get("local_files_only") is not True
        or row.get("trust_remote_code") is not False
        or row.get("model_weights_loaded") is not True
        or row.get("teacher_generation_completed") is not True
        or row.get("training_launched") is not False
        or row.get("optimizer_created") is not False
        or row.get("sealed_data_accessed") is not False
        or row.get("publication_authorized") is not False
        or row.get("accepted_artifact") is not False
        or row.get("model_free_verification_passed") is not True
        or row.get("atomic_publication") is not True
        or canonical_sha256(body) != claimed
    ):
        raise RecoveryInputPreparationError("Build report failed authentication.")
    return value


def _verify_published_tree(output: Path) -> None:
    names = sorted(path.name for path in output.iterdir())
    if names != sorted((REPLAY_FILENAME, PLAN_FILENAME, BUILD_REPORT_FILENAME)):
        raise RecoveryInputPreparationError(
            "Published recovery-input tree contains unexpected files."
        )
    replay = _strict_json_file(output / REPLAY_FILENAME)
    plan = _strict_json_file(output / PLAN_FILENAME)
    report = _strict_json_file(output / BUILD_REPORT_FILENAME)
    verified_replay = verify_frozen_replay_wrapper(replay)
    teacher = verified_replay["teacher_token_replay"]
    verify_recovery_plan(plan, teacher_payload=teacher)
    verify_build_report(report)
    if report["outputs"]["replay_bank"]["file_sha256"] != _file_sha256(
        output / REPLAY_FILENAME
    ) or report["outputs"]["recovery_plan"]["file_sha256"] != _file_sha256(
        output / PLAN_FILENAME
    ):
        raise RecoveryInputPreparationError(
            "Published recovery-input file hashes differ from the build report."
        )


def prepare_recovery_inputs(
    *,
    output_dir: Path | str,
    source_model: Path | str,
    source_tokenizer: Path | str,
    candidate_tokenizer: Path | str,
    calibration_file: Path | str,
    behavior_eval_suite: Path | str,
    expected_parent_artifact_tree_sha256: str,
    teacher_device: str = "cuda:0",
    recovery_device: str = "cuda:0",
    max_calibration_texts: int = 640,
    component_loader: Callable[..., TeacherComponents] = load_teacher_components,
    teacher_replay_builder: Callable[..., Mapping[str, Any]] = (
        hf_teacher_token_replay.build_teacher_token_replay_payload
    ),
) -> dict[str, Any]:
    """Generate, authenticate, then atomically publish one frozen A/B input set."""

    output = _absent_output_directory(output_dir)
    source_root = _existing_local_directory(source_model, "source_model")
    source_tokenizer_root = _existing_local_directory(
        source_tokenizer,
        "source_tokenizer",
    )
    candidate_tokenizer_root = _existing_local_directory(
        candidate_tokenizer,
        "candidate_tokenizer",
    )
    calibration_path = _existing_real_file(calibration_file, "calibration_file")
    behavior_path = _existing_real_file(
        behavior_eval_suite,
        "behavior_eval_suite",
    )
    expected_tree_sha = _require_sha256(
        expected_parent_artifact_tree_sha256,
        "expected_parent_artifact_tree_sha256",
    )
    if (
        not isinstance(max_calibration_texts, int)
        or isinstance(max_calibration_texts, bool)
        or max_calibration_texts <= 0
    ):
        raise RecoveryInputPreparationError(
            "max_calibration_texts must be a positive integer."
        )
    for name, value in (
        ("teacher_device", teacher_device),
        ("recovery_device", recovery_device),
    ):
        if not isinstance(value, str) or not value.strip():
            raise RecoveryInputPreparationError(f"{name} must be a nonblank string.")

    # Authenticate parent lineage before loading any model or allocating a GPU.
    actual_parent_tree = stage._artifact_tree(source_root)
    if actual_parent_tree["tree_sha256"] != expected_tree_sha:
        raise RecoveryInputPreparationError(
            "Teacher source artifact tree differs from expected canonical parent lineage."
        )
    behavior_binding = _behavior_suite_binding(behavior_path)
    verify_behavior_suite_binding(behavior_binding)
    calibration_texts = hf_recovery.load_calibration_texts(
        calibration_path,
        max_texts=max_calibration_texts,
        min_chars=32,
        max_chars=4096,
        preserve_whitespace=True,
    )
    ordered_calibration = hf_recovery._order_recovery_texts(
        calibration_texts,
        CALIBRATION_ORDER,
    )
    if ordered_calibration != calibration_texts:
        raise RecoveryInputPreparationError(
            "Original calibration ordering changed unexpectedly."
        )

    components = component_loader(
        source_model_path=source_root,
        source_tokenizer_path=source_tokenizer_root,
        candidate_tokenizer_path=candidate_tokenizer_root,
        device=teacher_device,
        local_files_only=True,
        trust_remote_code=False,
        source_torch_dtype="float16",
    )
    try:
        teacher_payload_raw = teacher_replay_builder(
            torch=components.torch,
            source_model=components.source_model,
            source_tokenizer=components.source_tokenizer,
            candidate_tokenizer=components.candidate_tokenizer,
            calibration_texts=ordered_calibration,
            **_teacher_parameters(teacher_device),
        )
        try:
            verified_teacher = (
                hf_teacher_token_replay.verify_teacher_token_replay_payload(
                    teacher_payload_raw
                )
            )
        except (TypeError, ValueError) as error:
            raise RecoveryInputPreparationError(
                "Generated teacher replay failed model-free verification."
            ) from error
        teacher_payload = _json_value(
            verified_teacher,
            name="teacher_token_replay",
        )
    finally:
        torch_module = components.torch
        del components
        gc.collect()
        empty_cache = getattr(getattr(torch_module, "cuda", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    state_sha = teacher_payload["source_model_fingerprint"]["model_state_fingerprint"][
        "state_dict_sha256"
    ]
    source_binding = _source_binding(
        expected_tree_sha256=expected_tree_sha,
        actual_tree=actual_parent_tree,
        state_dict_sha256=state_sha,
    )
    replay_payload = _replay_wrapper(
        teacher_payload=teacher_payload,
        source_binding=source_binding,
        behavior_binding=behavior_binding,
    )
    verify_frozen_replay_wrapper(replay_payload)
    plan_payload = _recovery_plan(teacher_payload, recovery_device)
    verify_recovery_plan(plan_payload, teacher_payload=teacher_payload)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=output.parent,
        )
    ).resolve()
    renamed = False
    try:
        replay_path = staging / REPLAY_FILENAME
        plan_path = staging / PLAN_FILENAME
        _write_json_exclusive(replay_path, replay_payload)
        _write_json_exclusive(plan_path, plan_payload)
        report = _build_report(
            replay_path=replay_path,
            plan_path=plan_path,
            teacher_payload=teacher_payload,
            source_binding=source_binding,
            behavior_binding=behavior_binding,
            plan_payload=plan_payload,
            teacher_generation_device=teacher_device,
            recovery_device=recovery_device,
        )
        verify_build_report(report)
        _write_json_exclusive(staging / BUILD_REPORT_FILENAME, report)
        _verify_published_tree(staging)
        staged_tree = stage._artifact_tree(staging)
        os.replace(staging, output)
        renamed = True
        published_tree = stage._artifact_tree(output)
        if any(
            staged_tree[name] != published_tree[name]
            for name in ("file_count", "total_bytes", "tree_sha256")
        ):
            raise RecoveryInputPreparationError(
                "Atomic recovery-input tree changed during publication."
            )
        _verify_published_tree(output)
        return _strict_json_file(output / BUILD_REPORT_FILENAME)
    except BaseException:
        owned = output if renamed else staging
        if (
            owned.exists()
            and owned.parent == output.parent
            and not owned.is_symlink()
            and (
                (renamed and owned == output)
                or (
                    not renamed
                    and owned.name.startswith(f".{output.name}.")
                    and owned.name.endswith(".staging")
                )
            )
        ):
            shutil.rmtree(owned)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-tokenizer", type=Path, required=True)
    parser.add_argument("--candidate-tokenizer", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--behavior-eval-suite", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-artifact-tree-sha256",
        required=True,
    )
    parser.add_argument("--teacher-device", default="cuda:0")
    parser.add_argument("--recovery-device", required=True)
    parser.add_argument("--max-calibration-texts", type=int, default=640)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_recovery_inputs(
        output_dir=args.output_dir,
        source_model=args.source_model,
        source_tokenizer=args.source_tokenizer,
        candidate_tokenizer=args.candidate_tokenizer,
        calibration_file=args.calibration_file,
        behavior_eval_suite=args.behavior_eval_suite,
        expected_parent_artifact_tree_sha256=(
            args.expected_parent_artifact_tree_sha256
        ),
        teacher_device=args.teacher_device,
        recovery_device=args.recovery_device,
        max_calibration_texts=args.max_calibration_texts,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
