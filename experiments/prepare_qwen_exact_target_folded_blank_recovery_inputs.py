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
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import prepare_qwen_exact_target_native_continuation_splits as split_tools
import run_qwen_exact_target_native_stage as stage
from neural_forge import (
    hf_recovery,
    hf_teacher_replay_cache,
    hf_teacher_token_replay,
    hf_tools,
)
from neural_forge.hf_guided_checkpoint_recovery import (
    canonical_json_bytes,
    canonical_sha256,
)


REPLAY_WRAPPER_SCHEMA = "neural-forge-frozen-recovery-replay-bank-v5"
RECOVERY_PLAN_SCHEMA = "neural-forge-folded-blank-recovery-plan-v2"
BEHAVIOR_SUITE_BINDING_SCHEMA = (
    "neural-forge-candidate-recovery-behavior-suite-binding-v5"
)
FRESH_SELECTION_SCHEMA = "neural-forge-fresh-recovery-candidate-selection-v3"
FRESH_REPLAY_EXCLUSION_DERIVATION_SCHEMA = (
    "neural-forge-fresh-replay-exclusion-first-survivors-v1"
)
TOKEN_DOMAIN_ADAPTER_SCHEMA = "neural-forge-token-domain-recovery-behavior-adapter-v5"
RECOVERY_REPLAY_DISJOINTNESS_AUDIT_SCHEMA = (
    "neural-forge-fresh-candidate-replay-disjointness-audit-v1"
)
TEACHER_SOURCE_BINDING_SCHEMA = "neural-forge-teacher-source-artifact-binding-v1"
BUILD_REPORT_SCHEMA = "neural-forge-folded-blank-recovery-input-build-v3"
TEACHER_REPLAY_ACQUISITION_SCHEMA = (
    "neural-forge-teacher-replay-acquisition-v2"
)
TEACHER_REPLAY_CACHE_IMPORT_RECEIPT_SCHEMA = (
    "neural-forge-teacher-replay-compatible-import-v1"
)
PARENT_MODEL_USAGE_SCHEMA = "neural-forge-parent-model-usage-v1"

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
BEHAVIOR_MODE_FRESH = "fresh-token-domain"
BEHAVIOR_MODE_EXPLICIT = "explicit-json-diagnostic"
TEACHER_REPLAY_CACHE_MODES = frozenset({"auto", "require", "refresh", "off"})
_CANONICAL_TEACHER_REPLAY_BUILDER = (
    hf_teacher_token_replay.build_teacher_token_replay_payload
)
_CANONICAL_SOURCE_MODEL_FINGERPRINTER = (
    hf_teacher_token_replay._model_functional_fingerprint
)
_CACHE_TEST_INJECTION_AUTHORITY = object()
FRESH_CANDIDATE_COUNT = 24
FIRST_ACTION_PROBE_COUNT = FRESH_CANDIDATE_COUNT
LCP4_PROBE_COUNT = FRESH_CANDIDATE_COUNT
BEHAVIOR_PROBE_COUNT = FIRST_ACTION_PROBE_COUNT + LCP4_PROBE_COUNT
GENERATION_PROMPT_COUNT = FRESH_CANDIDATE_COUNT
BEHAVIOR_MAX_NEW_TOKENS = 16
MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW = 8
EXTERNAL_FRESH_RESERVOIR_SCHEMA = (
    "neural-forge.exact-target-fresh-recovery-reservoir-seed.v1"
)
EXTERNAL_FRESH_RESERVOIR_RECEIPT_SCHEMA = (
    "neural-forge.exact-target-fresh-recovery-reservoir-receipt.v1"
)
EXTERNAL_FRESH_RESERVOIR_EXPANSION_ALGORITHM = (
    "round-robin-topic-template-category-format-v1"
)
PINNED_EXTERNAL_FRESH_RESERVOIR_PATH = Path(__file__).with_name(
    "qwen_exact_target_fresh_recovery_reservoir_v1.json"
)
PINNED_EXTERNAL_FRESH_RESERVOIR_FILE_SHA256 = (
    "db955d62b35c6d221de71a254b82956056c58a7f0a5d7340f84e9cb587f335bb"
)
PINNED_EXTERNAL_FRESH_RESERVOIR_BYTE_COUNT = 13_322
PINNED_EXTERNAL_FRESH_RESERVOIR_PAYLOAD_SHA256 = (
    "d2e9f9af068cce6b0769ebb24f6bdf94985cc20c090c32b35415a9db1f7364e3"
)
PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_COUNT = 384
PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_MANIFEST_SHA256 = (
    "855949e15df34ee515aa7d4092d96353321207939e507215f8e963738e664a6f"
)
PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_TEXT_MANIFEST_SHA256 = (
    "fd2db9aa77e190f0923c77e4acfcdf5289167300b3273ae4fdcac5331fb23f75"
)
LEGACY_TEACHER_CACHE_IMPORT_SOURCE_COMMIT = (
    "6166562db25f6a9e938be45d580381fc9d81d104"
)
LEGACY_TEACHER_CACHE_IMPORT_PREP_PRODUCER = {
    "role": "prepare_recovery_inputs",
    "filename": "prepare_qwen_exact_target_folded_blank_recovery_inputs.py",
    "byte_count": 165_948,
    "sha256": "9b69c663880e6a4c891af8473159e4ef4ce641c705542da134212ac5d4b1f3d2",
}
PINNED_RECOVERY_CORPUS_SHA256 = (
    "ba2dd1202752e2dd2b9b981c52501060ee367e220e5fb73ce7f8c320e2bbdd90"
)
PINNED_RECOVERY_CORPUS_LINE_COUNT = 2_906
PINNED_RECOVERY_CORPUS_DUPLICATE_SOURCE_ROW_COUNT = 328
PINNED_V4_PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-gud-candidate-tournament-"
    "preregistration-v1"
)
PINNED_V4_CANDIDATE_SELECTION_SCHEMA = (
    "neural-forge-guided-gud-fresh-candidate-selection-v1"
)
PINNED_V4_SPLIT_COUNTS = {"fit": 192, "candidate": 24, "sealed": 32}
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
        "mode",
        "causal_ab_eligible",
        "split",
        "source_filename",
        "source_file_sha256",
        "suite",
        "suite_sha256",
        "adapter_provenance",
        "behavior_probe_count",
        "first_action_probe_count",
        "lcp4_probe_count",
        "generation_prompt_count",
        "sealed_exclusion_manifest_used_model_free",
        "sealed_prompts_supplied_to_model_or_trainer",
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
        "behavior_mode",
        "causal_ab_eligible",
        "behavior_probe_count",
        "generation_prompt_count",
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
        "teacher_replay_acquisition",
        "parent_model_usage",
        "fresh_candidate_reservoir",
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


@dataclass(frozen=True, slots=True)
class TokenizerComponents:
    """Model-free identity-tokenizer objects used to freeze a fresh domain."""

    torch: Any
    transformers: Any
    source_tokenizer: Any
    candidate_tokenizer: Any


@dataclass(frozen=True, slots=True)
class FreshRecoverySelection:
    """A fresh 24-prompt candidate frozen after replay exclusion."""

    prompts: tuple[split_tools._PromptRecord, ...]
    decoded_prompts: tuple[str, ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FreshRecoveryCandidateReservoir:
    """Ordered tokenized and decoded candidate reservoir frozen pre-parent."""

    prompts: tuple[split_tools._PromptRecord, ...]
    decoded_prompts: tuple[str, ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ExternalFreshReservoirSeed:
    """Authenticated independent prompt seed expanded without model access."""

    prompts: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CompatibleTeacherCacheImport:
    """One explicitly pinned legacy replay authorized for compatible migration."""

    request: Mapping[str, Any]
    envelope: Mapping[str, Any]
    pre_model_receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FreshCandidateReplayExclusions:
    """Frozen recovery fields against which final heldout prompts are filtered."""

    calibration: tuple[str, ...]
    teacher_generated: tuple[str, ...]
    on_policy: tuple[tuple[str, ...], ...]
    stepwise: tuple[Any, ...]
    indexed_training_texts: tuple[tuple[str, int, int | None, str], ...]
    training_text_rows: tuple[Mapping[str, Any], ...]
    prompt_windows: frozenset[tuple[int, ...]]
    rollout_windows: frozenset[tuple[int, ...]]
    prompt_window_total: int
    rollout_window_total: int
    stepwise_row_count: int
    stepwise_sha256: str


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


def _load_external_fresh_reservoir_seed() -> ExternalFreshReservoirSeed:
    """Authenticate and expand the committed prompt seed without model access."""

    raw_path = Path(PINNED_EXTERNAL_FRESH_RESERVOIR_PATH).expanduser()
    try:
        info = raw_path.lstat()
    except OSError as error:
        raise RecoveryInputPreparationError(
            "Pinned external fresh-reservoir seed is unavailable."
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RecoveryInputPreparationError(
            "Pinned external fresh-reservoir seed must be one real file."
        )
    if (
        info.st_size != PINNED_EXTERNAL_FRESH_RESERVOIR_BYTE_COUNT
        or _file_sha256(raw_path) != PINNED_EXTERNAL_FRESH_RESERVOIR_FILE_SHA256
    ):
        raise RecoveryInputPreparationError(
            "Pinned external fresh-reservoir seed bytes changed."
        )
    payload = _strict_json_file(raw_path)
    expected_fields = {
        "schema",
        "status",
        "seed",
        "expansion_algorithm",
        "categories",
        "expected_expanded_prompt_count",
        "contains_cached_prompt_or_target_content",
        "contains_parent_or_child_outputs",
        "payload_sha256",
    }
    if set(payload) != expected_fields:
        raise RecoveryInputPreparationError(
            "External fresh-reservoir seed fields changed."
        )
    body = dict(payload)
    claimed = _require_sha256(
        body.pop("payload_sha256"), "external fresh-reservoir payload SHA"
    )
    categories = payload.get("categories")
    if (
        payload.get("schema") != EXTERNAL_FRESH_RESERVOIR_SCHEMA
        or payload.get("status") != "static-independent-pre-model-seed"
        or payload.get("seed") != 20_260_715
        or payload.get("expansion_algorithm")
        != EXTERNAL_FRESH_RESERVOIR_EXPANSION_ALGORITHM
        or payload.get("expected_expanded_prompt_count")
        != PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_COUNT
        or payload.get("contains_cached_prompt_or_target_content") is not False
        or payload.get("contains_parent_or_child_outputs") is not False
        or claimed != PINNED_EXTERNAL_FRESH_RESERVOIR_PAYLOAD_SHA256
        or canonical_sha256(body) != claimed
        or not isinstance(categories, list)
        or len(categories) != 8
    ):
        raise RecoveryInputPreparationError(
            "External fresh-reservoir seed authentication failed."
        )

    normalized_categories: list[dict[str, Any]] = []
    category_ids: set[str] = set()
    candidate_ids: set[str] = set()
    source_texts: set[str] = set()
    for category_index, raw_category in enumerate(categories):
        if not isinstance(raw_category, Mapping) or set(raw_category) != {
            "category_id",
            "topics",
            "templates",
        }:
            raise RecoveryInputPreparationError(
                f"External reservoir category {category_index} fields changed."
            )
        category_id = raw_category.get("category_id")
        topics = raw_category.get("topics")
        templates = raw_category.get("templates")
        if (
            not isinstance(category_id, str)
            or not category_id
            or not category_id.replace("-", "").isalnum()
            or category_id in category_ids
            or not isinstance(topics, list)
            or len(topics) != 12
            or not isinstance(templates, list)
            or len(templates) != 4
        ):
            raise RecoveryInputPreparationError(
                f"External reservoir category {category_index} is invalid."
            )
        category_ids.add(category_id)
        normalized_topics: list[dict[str, str]] = []
        topic_ids: set[str] = set()
        for topic_index, topic in enumerate(topics):
            if not isinstance(topic, Mapping) or set(topic) != {"topic_id", "label"}:
                raise RecoveryInputPreparationError(
                    f"External reservoir topic {category_index}:{topic_index} changed."
                )
            topic_id = topic.get("topic_id")
            label = topic.get("label")
            if (
                not isinstance(topic_id, str)
                or not topic_id
                or not topic_id.replace("-", "").isalnum()
                or topic_id in topic_ids
                or not isinstance(label, str)
                or not label.strip()
                or label != label.strip()
            ):
                raise RecoveryInputPreparationError(
                    f"External reservoir topic {category_index}:{topic_index} is invalid."
                )
            topic_ids.add(topic_id)
            normalized_topics.append({"topic_id": topic_id, "label": label})
        normalized_templates: list[dict[str, str]] = []
        template_ids: set[str] = set()
        for template_index, template in enumerate(templates):
            if not isinstance(template, Mapping) or set(template) != {
                "template_id",
                "text",
            }:
                raise RecoveryInputPreparationError(
                    "External reservoir template fields changed."
                )
            template_id = template.get("template_id")
            text = template.get("text")
            if (
                not isinstance(template_id, str)
                or not template_id
                or not template_id.replace("-", "").isalnum()
                or template_id in template_ids
                or not isinstance(text, str)
                or text.count("{topic}") != 1
                or "{" in text.replace("{topic}", "")
                or "}" in text.replace("{topic}", "")
            ):
                raise RecoveryInputPreparationError(
                    f"External reservoir template {category_index}:{template_index} is invalid."
                )
            template_ids.add(template_id)
            normalized_templates.append({"template_id": template_id, "text": text})
        normalized_categories.append(
            {
                "category_id": category_id,
                "topics": normalized_topics,
                "templates": normalized_templates,
            }
        )

    expanded: list[dict[str, Any]] = []
    # The category loop is innermost: each consecutive block of eight spans
    # every domain, independent of any model, cache, or output.
    for topic_index in range(12):
        for template_index in range(4):
            for category in normalized_categories:
                topic = category["topics"][topic_index]
                template = category["templates"][template_index]
                candidate_id = (
                    f"{category['category_id']}.{topic['topic_id']}."
                    f"{template['template_id']}"
                )
                text = template["text"].format(topic=topic["label"])
                if (
                    candidate_id in candidate_ids
                    or text in source_texts
                    or not text.strip()
                    or text != text.strip()
                ):
                    raise RecoveryInputPreparationError(
                        "External reservoir expansion produced a duplicate or blank prompt."
                    )
                candidate_ids.add(candidate_id)
                source_texts.add(text)
                source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                expanded.append(
                    {
                        "source_line_number": len(expanded) + 1,
                        "source_candidate_id": candidate_id,
                        "category_id": category["category_id"],
                        "topic_id": topic["topic_id"],
                        "template_id": template["template_id"],
                        "source_prompt_sha256": source_sha,
                        "source_prompt_utf8_byte_count": len(text.encode("utf-8")),
                        "text": text,
                    }
                )
    if len(expanded) != PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_COUNT:
        raise RecoveryInputPreparationError(
            "External reservoir expansion count changed."
        )
    manifest = [
        {key: value for key, value in row.items() if key != "text"}
        for row in expanded
    ]
    expanded_manifest_sha = canonical_sha256(manifest)
    expanded_text_manifest_sha = canonical_sha256(
        [row["text"] for row in expanded]
    )
    if (
        expanded_manifest_sha
        != PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_MANIFEST_SHA256
        or expanded_text_manifest_sha
        != PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_TEXT_MANIFEST_SHA256
    ):
        raise RecoveryInputPreparationError(
            "External reservoir expanded prompt manifests changed."
        )
    receipt = {
        "schema": EXTERNAL_FRESH_RESERVOIR_RECEIPT_SCHEMA,
        "status": "authenticated-expanded-before-cache-or-model-access",
        "source_filename": raw_path.name,
        "source_file_sha256": PINNED_EXTERNAL_FRESH_RESERVOIR_FILE_SHA256,
        "source_file_byte_count": PINNED_EXTERNAL_FRESH_RESERVOIR_BYTE_COUNT,
        "source_payload_sha256": claimed,
        "seed": payload["seed"],
        "expansion_algorithm": payload["expansion_algorithm"],
        "category_count": len(normalized_categories),
        "topic_count": sum(len(row["topics"]) for row in normalized_categories),
        "template_count": sum(
            len(row["templates"]) for row in normalized_categories
        ),
        "expanded_prompt_count": len(expanded),
        "expanded_prompt_manifest_sha256": expanded_manifest_sha,
        "expanded_prompt_text_manifest_sha256": expanded_text_manifest_sha,
        "first_24_cover_all_categories": (
            len({row["category_id"] for row in expanded[:24]})
            == len(normalized_categories)
        ),
        "contains_cached_prompt_or_target_content": False,
        "contains_parent_or_child_outputs": False,
    }
    if receipt["first_24_cover_all_categories"] is not True:
        raise RecoveryInputPreparationError(
            "External reservoir round-robin coverage changed."
        )
    return ExternalFreshReservoirSeed(
        prompts=tuple(expanded),
        receipt=receipt,
    )


def _verify_external_fresh_reservoir_receipt(
    receipt: Mapping[str, Any],
) -> ExternalFreshReservoirSeed:
    expected = _load_external_fresh_reservoir_seed()
    if not isinstance(receipt, Mapping) or dict(receipt) != dict(expected.receipt):
        raise RecoveryInputPreparationError(
            "External fresh-reservoir provenance changed."
        )
    return expected


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


def load_local_tokenizers(
    *,
    source_tokenizer_path: Path,
    candidate_tokenizer_path: Path,
    local_files_only: bool,
    trust_remote_code: bool,
) -> TokenizerComponents:
    """Load two local tokenizers without constructing or touching a model."""

    if local_files_only is not True or trust_remote_code is not False:
        raise RecoveryInputPreparationError(
            "Tokenizer preparation requires local_files_only=True and "
            "trust_remote_code=False."
        )
    torch, transformers = hf_recovery._torch_and_transformers()
    kwargs = {"local_files_only": True, "trust_remote_code": False}
    return TokenizerComponents(
        torch=torch,
        transformers=transformers,
        source_tokenizer=transformers.AutoTokenizer.from_pretrained(
            str(source_tokenizer_path), **kwargs
        ),
        candidate_tokenizer=transformers.AutoTokenizer.from_pretrained(
            str(candidate_tokenizer_path), **kwargs
        ),
    )


def load_teacher_source_model(
    *,
    torch: Any,
    transformers: Any,
    source_model_path: Path,
    device: str,
    local_files_only: bool,
    trust_remote_code: bool,
    source_torch_dtype: str,
) -> Any:
    """Load the source once, after every model-free domain check has passed."""

    if local_files_only is not True or trust_remote_code is not False:
        raise RecoveryInputPreparationError(
            "Teacher preparation requires local_files_only=True and "
            "trust_remote_code=False."
        )
    if source_torch_dtype != "float16":
        raise RecoveryInputPreparationError("Teacher source dtype must be float16.")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(source_model_path),
        torch_dtype=torch.float16,
        device_map=None,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()
    return model


_CANONICAL_TOKENIZER_LOADER = load_local_tokenizers
_CANONICAL_SOURCE_MODEL_LOADER = load_teacher_source_model


def _authenticate_v4_preregistration(path: Path, expected_sha256: str) -> Any:
    # Imported lazily so diagnostic input preparation does not load the larger
    # historical/current tournament authentication boundary.
    import authenticate_qwen_exact_target_native_guided_gud_candidate_tournament_history as history

    return history.authenticate_guided_gud_candidate_preregistration_with_history(
        path, expected_sha256
    ).contract


def _decode_identity_prompt(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception as error:
        raise RecoveryInputPreparationError(
            "Identity tokenizer could not decode a fresh token-domain prompt."
        ) from error
    if not isinstance(decoded, str) or not decoded.strip():
        raise RecoveryInputPreparationError(
            "A fresh token-domain prompt decoded to blank text."
        )
    try:
        roundtrip = split_tools._tokenize(tokenizer, decoded)
    except split_tools.ContinuationSplitPreparationError as error:
        raise RecoveryInputPreparationError(
            "Identity tokenizer could not round-trip a fresh prompt."
        ) from error
    if tuple(roundtrip) != tuple(token_ids):
        raise RecoveryInputPreparationError(
            "A decoded fresh prompt does not round-trip to its authenticated token IDs."
        )
    return decoded


def _prompt_manifest(prompts: Sequence[split_tools._PromptRecord]) -> str:
    return canonical_sha256(
        [
            {
                "prompt_id": prompt.prompt_id,
                "token_ids_sha256": split_tools._token_sequence_sha256(
                    prompt.token_ids
                ),
                "token_count": len(prompt.token_ids),
            }
            for prompt in prompts
        ]
    )


def _calibration_exclusion_records(
    tokenizer: Any,
    calibration_texts: Sequence[str],
) -> tuple[tuple[split_tools._PromptRecord, ...], dict[str, Any]]:
    records: list[split_tools._PromptRecord] = []
    full_token_rows: list[dict[str, Any]] = []
    bounded_rows: list[dict[str, Any]] = []
    for index, text in enumerate(calibration_texts):
        try:
            full_tokens = split_tools._tokenize(tokenizer, text)
        except split_tools.ContinuationSplitPreparationError as error:
            raise RecoveryInputPreparationError(
                f"Cannot tokenize recovery calibration row {index}."
            ) from error
        bounded = tuple(full_tokens[:MAX_LENGTH])
        if len(bounded) < split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS:
            raise RecoveryInputPreparationError(
                f"Recovery calibration row {index} has fewer than eight tokens."
            )
        digest = split_tools._token_sequence_sha256(bounded)
        payload = {
            "token_ids": list(bounded),
            "prompt_id": f"prompt-{digest}",
            "token_ids_sha256": digest,
        }
        records.append(
            split_tools._PromptRecord(
                payload=payload,
                token_ids=bounded,
                prompt_id=payload["prompt_id"],
                # For candidates constrained to 8..128 tokens, any shared
                # meaningful prefix exists iff the first eight tokens match.
                # Retaining one tuple avoids O(n^2) prefix materialization for
                # long recovery rows while preserving collision equivalence.
                prefixes=frozenset(
                    {bounded[: split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS]}
                )
                if len(bounded) >= split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS
                else frozenset(),
            )
        )
        full_token_rows.append(
            {
                "index": index,
                "token_count": len(full_tokens),
                "token_ids_sha256": split_tools._token_sequence_sha256(full_tokens),
            }
        )
        bounded_rows.append(
            {
                "index": index,
                "token_count": len(bounded),
                "token_ids_sha256": digest,
            }
        )
    return tuple(records), {
        "text_count": len(calibration_texts),
        "ordered_text_manifest_sha256": canonical_sha256(list(calibration_texts)),
        "full_token_manifest_sha256": canonical_sha256(full_token_rows),
        "bounded_token_manifest_sha256": canonical_sha256(bounded_rows),
        "prefix_bookkeeping_max_tokens": MAX_LENGTH,
        "prefix_bookkeeping_tokens_per_row": (
            split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS
        ),
        "full_tokenization_bound_separately": True,
    }


def _authenticate_fresh_recovery_candidate_reservoir(
    *,
    v4_preregistration: Path,
    v4_preregistration_sha256: str,
    corpus_path: Path,
    calibration_texts: Sequence[str],
    tokenizer: Any,
    preregistration_authenticator: Callable[[Path, str], Any],
) -> FreshRecoveryCandidateReservoir:
    """Freeze the ordered tokenized/decoded reservoir before model access."""

    if len(calibration_texts) != 640:
        raise RecoveryInputPreparationError(
            "Fresh causal recovery selection requires exactly 640 calibration rows."
        )
    expected_v4_sha = _require_sha256(
        v4_preregistration_sha256, "v4_preregistration_sha256"
    )
    if _file_sha256(v4_preregistration) != expected_v4_sha:
        raise RecoveryInputPreparationError("V4 preregistration hash changed.")
    prereg_raw = _strict_json_file(v4_preregistration)
    if (
        prereg_raw.get("schema") != PINNED_V4_PREREGISTRATION_SCHEMA
        or prereg_raw.get("sealed_before_model_access") is not True
    ):
        raise RecoveryInputPreparationError(
            "V4 preregistration schema or pre-model seal changed."
        )
    try:
        contract = preregistration_authenticator(v4_preregistration, expected_v4_sha)
    except Exception as error:
        raise RecoveryInputPreparationError(
            "V4 preregistration failed full model-free authentication."
        ) from error
    if (
        Path(contract.path).resolve() != v4_preregistration
        or contract.file_sha256 != expected_v4_sha
    ):
        raise RecoveryInputPreparationError(
            "V4 authenticator returned a different preregistration identity."
        )

    split_prompts: dict[str, tuple[split_tools._PromptRecord, ...]] = {}
    split_provenance: dict[str, Any] = {}
    for role, expected_count in PINNED_V4_SPLIT_COUNTS.items():
        try:
            binding = contract.splits[role]
        except (AttributeError, KeyError, TypeError) as error:
            raise RecoveryInputPreparationError(
                f"V4 preregistration lacks the {role} split binding."
            ) from error
        split_path = Path(binding.path).resolve()
        split_sha = _require_sha256(binding.sha256, f"V4 {role} split SHA")
        if (
            not split_path.is_file()
            or _file_sha256(split_path) != split_sha
            or binding.expected_count != expected_count
        ):
            raise RecoveryInputPreparationError(
                f"V4 {role} split file/hash/count changed."
            )
        payload = _strict_json_file(split_path)
        raw_prompts = payload.get("prompts")
        if (
            payload != binding.payload
            or payload.get("schema") != split_tools.SPLIT_SCHEMA
            or payload.get("split") != role
            or payload.get("sealed_before_model_access") is not (role == "sealed")
            or payload.get("model_accessed") is not False
            or payload.get("raw_prompt_text_retained") is not False
            or not isinstance(raw_prompts, list)
            or len(raw_prompts) != expected_count
        ):
            raise RecoveryInputPreparationError(
                f"V4 {role} split role/seal/model flags changed."
            )
        selection = payload.get("selection")
        if role == "candidate" and (
            not isinstance(selection, Mapping)
            or selection.get("schema") != PINNED_V4_CANDIDATE_SELECTION_SCHEMA
        ):
            raise RecoveryInputPreparationError(
                "V4 candidate selection schema changed."
            )
        content_sha = _require_sha256(
            payload.get("split_content_sha256"),
            f"V4 {role} split content SHA",
        )
        content_body = dict(payload)
        content_body.pop("split_content_sha256")
        if split_tools._canonical_sha256(content_body) != content_sha:
            raise RecoveryInputPreparationError(
                f"V4 {role} split content binding failed."
            )
        try:
            prompts = tuple(
                split_tools._prompt_record(item, f"V4.{role}.prompts[{index}]")
                for index, item in enumerate(raw_prompts)
            )
        except split_tools.ContinuationSplitPreparationError as error:
            raise RecoveryInputPreparationError(
                f"V4 {role} split prompt authentication failed."
            ) from error
        if len({prompt.prompt_id for prompt in prompts}) != len(prompts):
            raise RecoveryInputPreparationError(
                f"V4 {role} split contains duplicate prompt identities."
            )
        split_prompts[role] = prompts
        split_provenance[role] = {
            "artifact_file_sha256": split_sha,
            "split_content_sha256": content_sha,
            "prompt_count": len(prompts),
            "prompt_manifest_sha256": _prompt_manifest(prompts),
            "sealed_before_model_access": role == "sealed",
            "model_accessed": False,
            "raw_prompt_text_retained": False,
        }
    try:
        split_tools._assert_globally_disjoint(
            split_prompts, label="Authenticated V4 exclusion domains"
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise RecoveryInputPreparationError(
            "V4 exclusion domains are not globally disjoint."
        ) from error

    external_seed = _load_external_fresh_reservoir_seed()
    try:
        corpus_records = split_tools._load_corpus(
            corpus_path,
            expected_sha256=PINNED_RECOVERY_CORPUS_SHA256,
            expected_line_count=PINNED_RECOVERY_CORPUS_LINE_COUNT,
        )
        calibration_records, calibration_provenance = _calibration_exclusion_records(
            tokenizer, calibration_texts
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise RecoveryInputPreparationError(
            "Pinned recovery corpus authentication failed."
        ) from error

    duplicate_source_row_count = (
        PINNED_RECOVERY_CORPUS_LINE_COUNT - len(corpus_records)
    )
    if (
        duplicate_source_row_count
        != PINNED_RECOVERY_CORPUS_DUPLICATE_SOURCE_ROW_COUNT
    ):
        raise RecoveryInputPreparationError(
            "Pinned recovery corpus unique-prompt count changed."
        )

    tokenized_records: list[tuple[Mapping[str, Any], tuple[int, ...]]] = []
    skipped_short = 0
    skipped_calibration_text = 0
    for record in external_seed.prompts:
        text = str(record["text"])
        if any(
            text in calibration_text or calibration_text in text
            for calibration_text in calibration_texts
        ):
            skipped_calibration_text += 1
            continue
        try:
            tokens = split_tools._tokenize(tokenizer, text)
        except split_tools.ContinuationSplitPreparationError as error:
            raise RecoveryInputPreparationError(
                "External candidate tokenization failed during fresh selection."
            ) from error
        if len(tokens) < split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS:
            skipped_short += 1
            continue
        tokenized_records.append((record, tokens))

    excluded = (
        tuple(
            prompt
            for role in ("fit", "candidate", "sealed")
            for prompt in split_prompts[role]
        )
        + calibration_records
    )
    statically_excluded_prompt_ids = {prompt.prompt_id for prompt in excluded}
    statically_excluded_prefixes = {
        prefix for prompt in excluded for prefix in prompt.prefixes
    }
    reservoir_prompts: list[split_tools._PromptRecord] = []
    decoded_reservoir: list[str] = []
    reservoir_stats = {
        "records_considered": len(external_seed.prompts),
        "unique_source_prompt_count": len(external_seed.prompts),
        "duplicate_source_row_count": 0,
        "skipped_candidate_text_present_in_calibration": skipped_calibration_text,
        "skipped_under_minimum_prompt_tokens": skipped_short,
        "skipped_over_token_limit": 0,
        "skipped_exact_prompt_collision": 0,
        "skipped_meaningful_prefix_collision": 0,
        "skipped_decoded_calibration_collision": 0,
        "skipped_duplicate_decoded_prompt": 0,
        "accepted_into_reservoir": 0,
    }
    decoded_seen: set[str] = set()
    for record, tokens in tokenized_records:
        if len(tokens) > MAX_LENGTH:
            reservoir_stats["skipped_over_token_limit"] += 1
            continue
        digest = split_tools._token_sequence_sha256(tokens)
        prompt_id = f"prompt-{digest}"
        prefixes = split_tools._meaningful_prefixes(tokens)
        if prompt_id in statically_excluded_prompt_ids:
            reservoir_stats["skipped_exact_prompt_collision"] += 1
            continue
        if prefixes & statically_excluded_prefixes:
            reservoir_stats["skipped_meaningful_prefix_collision"] += 1
            continue
        decoded = _decode_identity_prompt(tokenizer, tokens)
        if decoded != record["text"]:
            raise RecoveryInputPreparationError(
                "An external candidate did not round-trip to its exact source text."
            )
        if any(
            decoded in calibration_text or calibration_text in decoded
            for calibration_text in calibration_texts
        ):
            reservoir_stats["skipped_decoded_calibration_collision"] += 1
            continue
        if decoded in decoded_seen:
            reservoir_stats["skipped_duplicate_decoded_prompt"] += 1
            continue
        prompt = split_tools._PromptRecord(
            payload={
                "source_line_number": int(record["source_line_number"]),
                "source_prompt_sha256": str(record["source_prompt_sha256"]),
                "token_ids": list(tokens),
                "prompt_id": prompt_id,
                "token_ids_sha256": digest,
            },
            token_ids=tokens,
            prompt_id=prompt_id,
            prefixes=prefixes,
        )
        reservoir_prompts.append(prompt)
        decoded_reservoir.append(decoded)
        decoded_seen.add(decoded)
        reservoir_stats["accepted_into_reservoir"] += 1

    if len(reservoir_prompts) < FRESH_CANDIDATE_COUNT:
        raise RecoveryInputPreparationError(
            "External seed has fewer than 24 pre-model reservoir prompts after "
            f"static exclusions; accepted={len(reservoir_prompts)}, "
            f"expanded={len(external_seed.prompts)}."
        )
    reservoir_manifest = [
        {
            "reservoir_index": index,
            "prompt_id": prompt.prompt_id,
            "token_ids_sha256": split_tools._token_sequence_sha256(
                prompt.token_ids
            ),
            "decoded_prompt_sha256": hashlib.sha256(
                decoded_reservoir[index].encode("utf-8")
            ).hexdigest(),
            "source_line_number": int(prompt.payload["source_line_number"]),
            "source_prompt_sha256": str(prompt.payload["source_prompt_sha256"]),
        }
        for index, prompt in enumerate(reservoir_prompts)
    ]
    reservoir_body = {
        "status": "candidate-reservoir-frozen-before-parent-load",
        "v4_preregistration": {
            "schema": PINNED_V4_PREREGISTRATION_SCHEMA,
            "source_filename": v4_preregistration.name,
            "file_sha256": expected_v4_sha,
            "evaluation_contract_sha256": _require_sha256(
                contract.evaluation_contract_sha256,
                "V4 evaluation contract SHA",
            ),
            "sealed_before_model_access": True,
        },
        "v4_exclusion_splits": split_provenance,
        "external_candidate_source": dict(external_seed.receipt),
        "corpus": {
            "file_sha256": PINNED_RECOVERY_CORPUS_SHA256,
            "nonblank_line_count": PINNED_RECOVERY_CORPUS_LINE_COUNT,
            "duplicate_source_row_count": duplicate_source_row_count,
            "purpose": "teacher-recovery-calibration-lineage-only",
        },
        "recovery_calibration": calibration_provenance,
        "reservoir_prompt_count": len(reservoir_prompts),
        "reservoir_manifest_sha256": canonical_sha256(reservoir_manifest),
        "reservoir_order": (
            "external-round-robin-source-line-order-after-static-exclusions"
        ),
        "reservoir_rows_hold_token_ids_and_decoded_text": True,
        "preselection_statistics": reservoir_stats,
        "sealed_exclusion_manifest_used_model_free": True,
        "sealed_prompts_supplied_to_model_or_trainer": False,
        "sealed_outputs_or_selection_scores_computed": False,
        "contains_sealed_data": False,
        "parent_model_loaded_during_reservoir_freeze": False,
        "parent_forward_or_generate_during_reservoir_freeze": False,
        "identity_tokenizer_used_during_reservoir_freeze": True,
    }
    reservoir_provenance = {
        **reservoir_body,
        "candidate_reservoir_payload_sha256": canonical_sha256(reservoir_body),
    }
    return FreshRecoveryCandidateReservoir(
        prompts=tuple(reservoir_prompts),
        decoded_prompts=tuple(decoded_reservoir),
        provenance=reservoir_provenance,
    )


def _select_fresh_recovery_candidate(
    *,
    reservoir: FreshRecoveryCandidateReservoir,
    teacher_payload: Mapping[str, Any],
) -> FreshRecoverySelection:
    """Choose the first 24 replay-disjoint rows from the pre-model reservoir."""

    reservoir_rows = _fresh_candidate_reservoir_rows(reservoir)
    derivation = _derive_fresh_candidate_selection(
        reservoir_rows=reservoir_rows,
        teacher_payload=teacher_payload,
    )
    selected_indices = tuple(derivation["selected_reservoir_indices"])
    prompts = tuple(reservoir.prompts[index] for index in selected_indices)
    decoded = tuple(reservoir.decoded_prompts[index] for index in selected_indices)
    external_seed = _verify_external_fresh_reservoir_receipt(
        reservoir.provenance["external_candidate_source"]
    )
    source_categories = {
        int(row["source_line_number"]): str(row["category_id"])
        for row in external_seed.prompts
    }
    selected_categories = [
        source_categories[int(prompt.payload["source_line_number"])]
        for prompt in prompts
    ]
    all_categories = sorted({str(row["category_id"]) for row in external_seed.prompts})
    category_histogram = {
        category: selected_categories.count(category) for category in all_categories
    }
    if any(count == 0 for count in category_histogram.values()):
        raise RecoveryInputPreparationError(
            "The first 24 replay-disjoint survivors do not cover all external "
            f"candidate categories; category_counts={json.dumps(category_histogram, sort_keys=True)}."
        )
    token_counts = [len(prompt.token_ids) for prompt in prompts]
    if (
        len(prompts) != FRESH_CANDIDATE_COUNT
        or min(token_counts) < split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS
        or max(token_counts) > MAX_LENGTH
    ):
        raise RecoveryInputPreparationError(
            "Fresh recovery candidate prompt length contract failed."
        )
    if len(set(decoded)) != len(decoded):
        raise RecoveryInputPreparationError(
            "Fresh recovery candidate is not disjoint from recovery training data."
        )
    selected_rows = [
        {
            "reservoir_index": selected_indices[index],
            "prompt_id": prompt.prompt_id,
            "token_ids_sha256": split_tools._token_sequence_sha256(prompt.token_ids),
            "token_count": len(prompt.token_ids),
            "source_line_number": int(prompt.payload["source_line_number"]),
            "source_prompt_sha256": str(prompt.payload["source_prompt_sha256"]),
            "decoded_prompt_sha256": hashlib.sha256(
                decoded[index].encode("utf-8")
            ).hexdigest(),
        }
        for index, prompt in enumerate(prompts)
    ]
    body = {
        "schema": FRESH_SELECTION_SCHEMA,
        "status": "frozen-after-teacher-replay-exclusion",
        "mode": BEHAVIOR_MODE_FRESH,
        "causal_ab_eligible": True,
        "v4_preregistration": reservoir.provenance["v4_preregistration"],
        "v4_exclusion_splits": reservoir.provenance["v4_exclusion_splits"],
        "external_candidate_source": reservoir.provenance[
            "external_candidate_source"
        ],
        "corpus": reservoir.provenance["corpus"],
        "recovery_calibration": reservoir.provenance["recovery_calibration"],
        "candidate_reservoir": {
            "provenance": reservoir.provenance,
            "rows": reservoir_rows,
            "rows_sha256": canonical_sha256(reservoir_rows),
        },
        "replay_exclusion_derivation": derivation,
        "selected_candidate": {
            "prompt_count": len(prompts),
            "minimum_token_count": min(token_counts),
            "maximum_token_count": max(token_counts),
            "minimum_required_token_count": (
                split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS
            ),
            "maximum_allowed_token_count": MAX_LENGTH,
            "prompt_rows": selected_rows,
            "prompt_manifest_sha256": canonical_sha256(selected_rows),
            "decoded_prompt_manifest_sha256": canonical_sha256(list(decoded)),
            "category_ids_in_selected_order": selected_categories,
            "category_histogram": category_histogram,
            "all_external_categories_represented": True,
        },
        "selection_statistics": {
            "preselection": reservoir.provenance["preselection_statistics"],
            "replay_exclusion_reason_counts": derivation["reason_counts"],
            "replay_disjoint_survivor_count": derivation["survivor_count"],
            "replay_disjoint_survivor_margin": derivation["survivor_margin"],
        },
        "disjointness": {
            "exact_token_prompt_overlap_count": 0,
            "meaningful_prefix_overlap_count": 0,
            "decoded_candidate_text_in_recovery_calibration_count": 0,
            "minimum_meaningful_prefix_tokens": (
                split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS
            ),
            "recovery_calibration_prefixes_bounded_to_candidate_max_tokens": True,
        },
        "sealed_exclusion_manifest_used_model_free": True,
        "sealed_prompts_supplied_to_model_or_trainer": False,
        "sealed_outputs_or_selection_scores_computed": False,
        "contains_sealed_data": False,
        "candidate_reservoir_frozen_before_parent_load": True,
        "final_candidate_frozen_before_parent_load": False,
        "parent_model_loaded_before_final_selection": True,
        "teacher_replay_authenticated_before_final_selection": True,
        "teacher_replay_used_for_exclusion_only": True,
        "parent_forward_or_score_during_final_selection": False,
        "tokenizer_called_during_final_selection": False,
        "child_or_arm_outputs_accessed_during_final_selection": False,
    }
    replay_audit = _fresh_candidate_replay_disjointness_audit(
        decoded_prompts=decoded,
        prompt_token_ids=[prompt.token_ids for prompt in prompts],
        teacher_payload=teacher_payload,
    )
    body["final_replay_disjointness_audit_sha256"] = replay_audit["payload_sha256"]
    provenance = {**body, "payload_sha256": canonical_sha256(body)}
    return FreshRecoverySelection(
        prompts=prompts,
        decoded_prompts=decoded,
        provenance=provenance,
    )


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
        "behavior_max_probes": BEHAVIOR_PROBE_COUNT,
        "behavior_max_generation_prompts": GENERATION_PROMPT_COUNT,
        "behavior_max_new_tokens": BEHAVIOR_MAX_NEW_TOKENS,
        "adamw_foreach": False,
    }


def _binding_from_suite(
    *,
    suite: Mapping[str, Any],
    mode: str,
    source_filename: str,
    source_file_sha256: str,
    adapter_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = _json_value(suite, name="behavior_eval_suite.suite")
    behavior_probes = normalized.get("behavior_probes")
    generation_prompts = normalized.get("generation_prompts")
    if not isinstance(behavior_probes, list) or not isinstance(
        generation_prompts, list
    ):
        raise RecoveryInputPreparationError(
            "Behavior suite must contain behavior_probes and generation_prompts."
        )
    first_count = sum(
        probe.get("category") == "parent-first-action-decoded-prefix"
        for probe in behavior_probes
        if isinstance(probe, Mapping)
    )
    lcp4_count = sum(
        probe.get("category") == "parent-lcp4-decoded-prefix"
        for probe in behavior_probes
        if isinstance(probe, Mapping)
    )
    fresh = mode == BEHAVIOR_MODE_FRESH
    body = {
        "schema": BEHAVIOR_SUITE_BINDING_SCHEMA,
        "status": "frozen",
        "mode": mode,
        "causal_ab_eligible": fresh,
        "split": "candidate",
        "source_filename": source_filename,
        "source_file_sha256": _require_sha256(
            source_file_sha256, "behavior source file SHA"
        ),
        "suite": normalized,
        "suite_sha256": canonical_sha256(normalized),
        "adapter_provenance": (
            None
            if adapter_provenance is None
            else _json_value(adapter_provenance, name="adapter_provenance")
        ),
        "behavior_probe_count": len(behavior_probes),
        "first_action_probe_count": first_count,
        "lcp4_probe_count": lcp4_count,
        "generation_prompt_count": len(generation_prompts),
        "sealed_exclusion_manifest_used_model_free": fresh,
        "sealed_prompts_supplied_to_model_or_trainer": False,
        # Historical field semantics: sealed prompt content does not enter the
        # suite/replay/trainer.  Fresh provenance separately and honestly says
        # the sealed token manifest was read model-free for exclusion hygiene.
        "sealed_data_accessed": False,
        "contains_sealed_data": False,
        "publication_authorized": False,
    }
    return {**body, "payload_sha256": canonical_sha256(body)}


def _behavior_suite_binding(path: Path) -> dict[str, Any]:
    """Load legacy explicit JSON as a diagnostic-only, non-causal suite."""

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
    return _binding_from_suite(
        suite=normalized,
        mode=BEHAVIOR_MODE_EXPLICIT,
        source_filename=path.name,
        source_file_sha256=_file_sha256(path),
        adapter_provenance=None,
    )


def _decoded_prefix(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    continuation_ids: Sequence[int],
    canonical_prompt: str,
) -> str:
    decoded = tokenizer.decode(
        [*prompt_ids, *continuation_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str) or not decoded.startswith(canonical_prompt):
        raise RecoveryInputPreparationError(
            "Parent prefix decode does not start with the canonical suite prompt."
        )
    prefix = decoded[len(canonical_prompt) :].strip()
    if not prefix:
        raise RecoveryInputPreparationError(
            "Parent decoded continuation prefix is blank after scorer stripping."
        )
    return prefix


def _build_fresh_behavior_binding(
    *,
    torch: Any,
    source_model: Any,
    tokenizer: Any,
    selection: FreshRecoverySelection,
    teacher_payload: Mapping[str, Any],
    device: str,
    teacher_replay_origin: str = "generated",
) -> dict[str, Any]:
    """Generate and self-score parent decoded-prefix targets for the fresh 24."""

    if teacher_replay_origin not in {"generated", "cache-hit", "cache-import"}:
        raise RecoveryInputPreparationError("teacher_replay_origin is unsupported.")

    tokenizer_fingerprints = teacher_payload.get("tokenizer_fingerprints")
    if not isinstance(tokenizer_fingerprints, Mapping) or tokenizer_fingerprints.get(
        "source"
    ) != tokenizer_fingerprints.get("candidate"):
        raise RecoveryInputPreparationError(
            "Fresh token-domain adapter requires one identical tokenizer fingerprint."
        )
    replay_disjointness_audit = _fresh_candidate_replay_disjointness_audit(
        decoded_prompts=selection.decoded_prompts,
        prompt_token_ids=[prompt.token_ids for prompt in selection.prompts],
        teacher_payload=teacher_payload,
    )
    behavior_probes: list[dict[str, Any]] = []
    generation_prompts: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    generated_outputs: dict[str, str] = {}
    generated_token_ids: dict[str, tuple[int, ...]] = {}
    eos_value = getattr(tokenizer, "eos_token_id", None)
    eos_ids = (
        {int(eos_value)}
        if isinstance(eos_value, int) and not isinstance(eos_value, bool)
        else set()
    )
    pad_value = getattr(tokenizer, "pad_token_id", None)
    pad_token_id = pad_value if pad_value is not None else eos_value
    source_model.eval()
    with torch.no_grad():
        for index, (prompt, canonical_prompt) in enumerate(
            zip(selection.prompts, selection.decoded_prompts, strict=True)
        ):
            input_ids = torch.tensor(
                [list(prompt.token_ids)], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids)
            generated_tensor = source_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=BEHAVIOR_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_value,
            )
            output_ids = tuple(
                int(value) for value in generated_tensor[0].detach().cpu().tolist()
            )
            if output_ids[: len(prompt.token_ids)] != prompt.token_ids:
                raise RecoveryInputPreparationError(
                    "Parent generation did not preserve the exact prompt token prefix."
                )
            continuation = output_ids[len(prompt.token_ids) :]
            if len(continuation) > BEHAVIOR_MAX_NEW_TOKENS:
                raise RecoveryInputPreparationError(
                    "Parent exceeded the pinned 16-token adapter generation limit."
                )
            pre_eos: list[int] = []
            for token_id in continuation:
                if token_id in eos_ids:
                    break
                pre_eos.append(token_id)
            if len(pre_eos) < 4:
                raise RecoveryInputPreparationError(
                    "Parent produced fewer than four real pre-EOS continuation tokens."
                )
            full_decoded = tokenizer.decode(
                list(output_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(full_decoded, str) or not full_decoded.startswith(
                canonical_prompt
            ):
                raise RecoveryInputPreparationError(
                    "Parent full decode does not start with the canonical suite prompt."
                )
            first_prefix = _decoded_prefix(
                tokenizer, prompt.token_ids, pre_eos[:1], canonical_prompt
            )
            lcp4_prefix = _decoded_prefix(
                tokenizer, prompt.token_ids, pre_eos[:4], canonical_prompt
            )
            scored_continuation = full_decoded[len(canonical_prompt) :].strip()
            if not scored_continuation.startswith(first_prefix) or not (
                scored_continuation.startswith(lcp4_prefix)
            ):
                raise RecoveryInputPreparationError(
                    "Parent target prefix disagrees with scorer continuation slicing."
                )
            first_id = f"parent-first-action-{prompt.prompt_id}"
            lcp4_id = f"parent-lcp4-{prompt.prompt_id}"
            generation_id = f"parent-generation-{prompt.prompt_id}"
            first_probe = {
                "id": first_id,
                "category": "parent-first-action-decoded-prefix",
                "prompt": canonical_prompt,
                "kind": "literal_prefix",
                "expected_prefix": first_prefix,
            }
            lcp4_probe = {
                "id": lcp4_id,
                "category": "parent-lcp4-decoded-prefix",
                "prompt": canonical_prompt,
                "kind": "literal_prefix",
                "expected_prefix": lcp4_prefix,
            }
            behavior_probes.extend((first_probe, lcp4_probe))
            generation_prompts.append(
                {
                    "id": generation_id,
                    "category": "parent-lcp4-decoded-prefix",
                    "prompt": canonical_prompt,
                    "kind": "literal_prefix",
                    "expected_prefix": lcp4_prefix,
                    "expected_prefix_generated_token_count": 4,
                    "relevance_rules": [],
                    "require_generation_health": True,
                }
            )
            generated_outputs[canonical_prompt] = full_decoded
            generated_token_ids[canonical_prompt] = tuple(pre_eos)
            target_rows.append(
                {
                    "index": index,
                    "prompt_id": prompt.prompt_id,
                    "prompt_token_ids": list(prompt.token_ids),
                    "prompt_token_ids_sha256": (
                        split_tools._token_sequence_sha256(prompt.token_ids)
                    ),
                    "prompt_token_count": len(prompt.token_ids),
                    "decoded_prompt_sha256": hashlib.sha256(
                        canonical_prompt.encode("utf-8")
                    ).hexdigest(),
                    "generated_continuation_token_ids": list(continuation),
                    "generated_continuation_token_ids_sha256": (
                        split_tools._token_sequence_sha256(continuation)
                    ),
                    "generated_continuation_token_count": len(continuation),
                    "generated_continuation_decoded_text": scored_continuation,
                    "generated_continuation_decoded_text_sha256": hashlib.sha256(
                        scored_continuation.encode("utf-8")
                    ).hexdigest(),
                    "pre_eos_token_count": len(pre_eos),
                    "first_action_token_ids": pre_eos[:1],
                    "first_action_token_ids_sha256": (
                        split_tools._token_sequence_sha256(pre_eos[:1])
                    ),
                    "first_action_decoded_prefix": first_prefix,
                    "first_action_decoded_prefix_sha256": hashlib.sha256(
                        first_prefix.encode("utf-8")
                    ).hexdigest(),
                    "lcp4_token_ids": pre_eos[:4],
                    "lcp4_token_ids_sha256": split_tools._token_sequence_sha256(
                        pre_eos[:4]
                    ),
                    "lcp4_decoded_prefix": lcp4_prefix,
                    "lcp4_decoded_prefix_sha256": hashlib.sha256(
                        lcp4_prefix.encode("utf-8")
                    ).hexdigest(),
                    "first_action_probe_id": first_id,
                    "lcp4_probe_id": lcp4_id,
                    "generation_prompt_id": generation_id,
                }
            )

    if len(generated_outputs) != FRESH_CANDIDATE_COUNT:
        raise RecoveryInputPreparationError(
            "Fresh adapter generation cache lost a unique canonical prompt."
        )
    for probe in behavior_probes:
        scored = hf_tools._score_behavior_probe(
            probe,
            {
                "output": generated_outputs[probe["prompt"]],
                "prompt": probe["prompt"],
            },
        )
        if scored.get("passed") is not True:
            raise RecoveryInputPreparationError(
                "Parent failed its own frozen decoded-prefix behavior probe."
            )
    parent_generation_health_pass_count = 0
    parent_generation_health_receipts: list[dict[str, Any]] = []
    for generation in generation_prompts:
        scored = hf_tools._generation_health(
            {
                **generation,
                "output": generated_outputs[generation["prompt"]],
                "diagnostics": hf_tools._generation_diagnostics(
                    generated_outputs[generation["prompt"]][
                        len(generation["prompt"]) :
                    ].strip(),
                    generated_token_ids[generation["prompt"]],
                ),
            },
            min_continuation_chars=24,
            min_unique_token_ratio=0.20,
        )
        if scored.get("passed") is not True:
            raise RecoveryInputPreparationError(
                "Parent failed its own frozen prefix-plus-coherence generation probe."
            )
        parent_generation_health_pass_count += 1
        parent_generation_health_receipts.append(
            {
                "generation_prompt_id": generation["id"],
                "health": _json_value(
                    scored,
                    name="parent_generation_health_receipt",
                ),
            }
        )
    suite = {
        "source": "fresh-token-domain-parent-greedy-v1",
        "text_evals": [],
        "choice_probes": [],
        "generation_prompts": generation_prompts,
        "behavior_probes": behavior_probes,
    }
    source_fingerprint = teacher_payload["source_model_fingerprint"]
    source_state_sha = source_fingerprint["model_state_fingerprint"][
        "state_dict_sha256"
    ]
    adapter_body = {
        "schema": TOKEN_DOMAIN_ADAPTER_SCHEMA,
        "status": "frozen-parent-targets",
        "mode": BEHAVIOR_MODE_FRESH,
        "causal_ab_eligible": True,
        "selection": _json_value(selection.provenance, name="fresh_selection"),
        "source_model_fingerprint_sha256": _require_sha256(
            source_fingerprint["sha256"], "source model fingerprint SHA"
        ),
        "source_model_state_dict_sha256": _require_sha256(
            source_state_sha, "source state_dict SHA"
        ),
        "identity_tokenizer_fingerprint_sha256": _require_sha256(
            tokenizer_fingerprints["source"]["sha256"],
            "identity tokenizer fingerprint SHA",
        ),
        "generation": {
            "do_sample": False,
            "max_new_tokens": BEHAVIOR_MAX_NEW_TOKENS,
            "minimum_real_pre_eos_tokens": 4,
            "batch_size": 1,
            "parent_model_target_generation_performed": True,
            "teacher_replay_origin": teacher_replay_origin,
            "parent_model_generated_teacher_replay_this_run": (
                teacher_replay_origin == "generated"
            ),
        },
        "recovery_replay_disjointness_audit": replay_disjointness_audit,
        "recovery_replay_disjointness_audit_sha256": (
            replay_disjointness_audit["payload_sha256"]
        ),
        "targets": target_rows,
        "target_manifest_sha256": canonical_sha256(target_rows),
        "suite_sha256": canonical_sha256(suite),
        "behavior_probe_count": len(behavior_probes),
        "first_action_probe_count": FIRST_ACTION_PROBE_COUNT,
        "lcp4_probe_count": LCP4_PROBE_COUNT,
        "generation_prompt_count": len(generation_prompts),
        "suite_prompt_exact_token_roundtrip_count": FRESH_CANDIDATE_COUNT,
        "parent_full_decode_starts_canonical_prompt_count": FRESH_CANDIDATE_COUNT,
        "parent_self_score_pass_count": len(behavior_probes),
        "parent_self_score_probe_count": len(behavior_probes),
        "parent_generation_health_pass_count": parent_generation_health_pass_count,
        "parent_generation_health_probe_count": len(generation_prompts),
        "parent_generation_health_receipts": parent_generation_health_receipts,
        "parent_generation_health_receipts_sha256": canonical_sha256(
            parent_generation_health_receipts
        ),
        "sealed_exclusion_manifest_used_model_free": True,
        "sealed_prompt_decode_count": 0,
        "sealed_prompt_forward_count": 0,
        "sealed_prompt_output_count": 0,
        "sealed_prompts_supplied_to_model_or_trainer": False,
        "contains_sealed_data": False,
        "publication_authorized": False,
    }
    adapter = {**adapter_body, "payload_sha256": canonical_sha256(adapter_body)}
    return _binding_from_suite(
        suite=suite,
        mode=BEHAVIOR_MODE_FRESH,
        source_filename=selection.provenance["v4_preregistration"]["source_filename"],
        source_file_sha256=selection.provenance["v4_preregistration"]["file_sha256"],
        adapter_provenance=adapter,
    )


def _fresh_selection_structure_is_valid(selection: Mapping[str, Any]) -> bool:
    """Verify reservoir/order/first-survivor bindings without replay access."""

    try:
        reservoir_rows = _verify_fresh_candidate_reservoir_payload(
            selection["candidate_reservoir"]
        )
        derivation = selection["replay_exclusion_derivation"]
        if not isinstance(derivation, Mapping):
            return False
        derivation_fields = {
            "schema",
            "status",
            "algorithm",
            "reason_precedence",
            "minimum_common_token_window",
            "teacher_token_replay_payload_sha256",
            "training_text_manifest_sha256",
            "stepwise_replay_bank_sha256",
            "reservoir_rows_sha256",
            "reservoir_prompt_count",
            "classification_rows",
            "classification_rows_sha256",
            "reason_counts",
            "survivor_count",
            "survivor_margin",
            "selected_reservoir_indices",
            "selected_prompt_ids",
            "selected_count",
            "full_reservoir_classified",
            "selection_is_first_survivors_in_frozen_order",
            "teacher_replay_used_for_exclusion_only",
            "candidate_behavior_or_child_arm_scores_accessed",
            "parent_forward_or_score_executed_by_derivation",
            "tokenizer_called_by_derivation",
            "payload_sha256",
        }
        if set(derivation) != derivation_fields:
            return False
        derivation_body = dict(derivation)
        claimed_derivation_sha = _require_sha256(
            derivation_body.pop("payload_sha256"),
            "fresh replay exclusion derivation SHA",
        )
        classifications = derivation["classification_rows"]
        reason_precedence = [
            "bidirectional_recovery_text_collision",
            "stepwise_prompt_window_collision",
            "stepwise_rollout_window_collision",
            "prior_survivor_meaningful_prefix_collision",
            "replay_disjoint_survivor",
        ]
        if (
            not isinstance(classifications, list)
            or len(classifications) != len(reservoir_rows)
            or any(
                row
                != {
                    "reservoir_index": index,
                    "classification": row.get("classification"),
                }
                or row.get("classification") not in reason_precedence
                for index, row in enumerate(classifications)
                if isinstance(row, Mapping)
            )
            or any(not isinstance(row, Mapping) for row in classifications)
        ):
            return False
        recomputed_counts = {
            reason: sum(
                row["classification"] == reason for row in classifications
            )
            for reason in reason_precedence
        }
        survivor_indices = [
            row["reservoir_index"]
            for row in classifications
            if row["classification"] == "replay_disjoint_survivor"
        ]
        selected_indices = survivor_indices[:FRESH_CANDIDATE_COUNT]
        selected_prompt_ids = [
            reservoir_rows[index]["prompt_id"] for index in selected_indices
        ]
        selected_rows = selection["selected_candidate"]["prompt_rows"]
        if (
            derivation.get("schema")
            != FRESH_REPLAY_EXCLUSION_DERIVATION_SCHEMA
            or derivation.get("status")
            != "authenticated-first-24-replay-disjoint-survivors"
            or derivation.get("algorithm")
            != "full-reservoir-fixed-reason-order-first-24-v1"
            or derivation.get("reason_precedence") != reason_precedence
            or derivation.get("minimum_common_token_window")
            != MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW
            or derivation.get("reservoir_rows_sha256")
            != canonical_sha256(reservoir_rows)
            or derivation.get("reservoir_prompt_count") != len(reservoir_rows)
            or derivation.get("classification_rows_sha256")
            != canonical_sha256(classifications)
            or derivation.get("reason_counts") != recomputed_counts
            or derivation.get("survivor_count") != len(survivor_indices)
            or derivation.get("survivor_margin")
            != len(survivor_indices) - FRESH_CANDIDATE_COUNT
            or derivation.get("selected_reservoir_indices") != selected_indices
            or derivation.get("selected_prompt_ids") != selected_prompt_ids
            or derivation.get("selected_count") != FRESH_CANDIDATE_COUNT
            or derivation.get("full_reservoir_classified") is not True
            or derivation.get("selection_is_first_survivors_in_frozen_order")
            is not True
            or derivation.get("teacher_replay_used_for_exclusion_only") is not True
            or derivation.get("candidate_behavior_or_child_arm_scores_accessed")
            is not False
            or derivation.get("parent_forward_or_score_executed_by_derivation")
            is not False
            or derivation.get("tokenizer_called_by_derivation") is not False
            or canonical_sha256(derivation_body) != claimed_derivation_sha
            or not isinstance(selected_rows, list)
            or len(selected_rows) != FRESH_CANDIDATE_COUNT
        ):
            return False
        for offset, selected_row in enumerate(selected_rows):
            source = reservoir_rows[selected_indices[offset]]
            if (
                not isinstance(selected_row, Mapping)
                or selected_row.get("reservoir_index") != selected_indices[offset]
                or selected_row.get("prompt_id") != source["prompt_id"]
                or selected_row.get("token_ids_sha256")
                != source["token_ids_sha256"]
                or selected_row.get("token_count") != len(source["token_ids"])
                or selected_row.get("decoded_prompt_sha256")
                != source["decoded_prompt_sha256"]
                or selected_row.get("source_line_number")
                != source["source_line_number"]
                or selected_row.get("source_prompt_sha256")
                != source["source_prompt_sha256"]
            ):
                return False
        external_seed = _verify_external_fresh_reservoir_receipt(
            selection["external_candidate_source"]
        )
        source_categories = {
            int(row["source_line_number"]): str(row["category_id"])
            for row in external_seed.prompts
        }
        selected_categories = [
            source_categories[int(row["source_line_number"])] for row in selected_rows
        ]
        all_categories = sorted(
            {str(row["category_id"]) for row in external_seed.prompts}
        )
        category_histogram = {
            category: selected_categories.count(category)
            for category in all_categories
        }
        selected_candidate = selection["selected_candidate"]
        if (
            selection.get("external_candidate_source")
            != selection["candidate_reservoir"]["provenance"].get(
                "external_candidate_source"
            )
            or selected_candidate.get("category_ids_in_selected_order")
            != selected_categories
            or selected_candidate.get("category_histogram") != category_histogram
            or selected_candidate.get("all_external_categories_represented") is not True
            or any(count == 0 for count in category_histogram.values())
        ):
            return False
        return (
            selection.get("status") == "frozen-after-teacher-replay-exclusion"
            and selection.get("candidate_reservoir_frozen_before_parent_load")
            is True
            and selection.get("final_candidate_frozen_before_parent_load") is False
            and selection.get("parent_model_loaded_before_final_selection") is True
            and selection.get("teacher_replay_authenticated_before_final_selection")
            is True
            and selection.get("teacher_replay_used_for_exclusion_only") is True
            and selection.get("parent_forward_or_score_during_final_selection")
            is False
            and selection.get("tokenizer_called_during_final_selection") is False
            and selection.get(
                "child_or_arm_outputs_accessed_during_final_selection"
            )
            is False
            and _require_sha256(
                selection.get("final_replay_disjointness_audit_sha256"),
                "final replay disjointness audit SHA",
            )
            == selection.get("final_replay_disjointness_audit_sha256")
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        RecoveryInputPreparationError,
    ):
        return False


def _fresh_adapter_cross_bindings_valid(
    adapter: Mapping[str, Any], suite: Mapping[str, Any]
) -> bool:
    try:
        targets = adapter["targets"]
        selection_rows = adapter["selection"]["selected_candidate"]["prompt_rows"]
        behavior_probes = suite["behavior_probes"]
        generation_prompts = suite["generation_prompts"]
        if (
            suite.get("text_evals") != []
            or suite.get("choice_probes") != []
            or not isinstance(targets, list)
            or not isinstance(selection_rows, list)
            or not isinstance(behavior_probes, list)
            or not isinstance(generation_prompts, list)
        ):
            return False
        selection_by_id = {row["prompt_id"]: row for row in selection_rows}
        target_by_id = {row["prompt_id"]: row for row in targets}
        behavior_by_id = {probe["id"]: probe for probe in behavior_probes}
        generation_by_id = {prompt["id"]: prompt for prompt in generation_prompts}
        if not (
            len(selection_by_id) == len(target_by_id) == FRESH_CANDIDATE_COUNT
            and len(behavior_by_id) == BEHAVIOR_PROBE_COUNT
            and len(generation_by_id) == GENERATION_PROMPT_COUNT
        ):
            return False
        for prompt_id, target in target_by_id.items():
            selection_row = selection_by_id[prompt_id]
            prompt_ids = target["prompt_token_ids"]
            continuation = target["generated_continuation_token_ids"]
            decoded_continuation = target["generated_continuation_decoded_text"]
            first_ids = target["first_action_token_ids"]
            lcp4_ids = target["lcp4_token_ids"]
            first_prefix = target["first_action_decoded_prefix"]
            lcp4_prefix = target["lcp4_decoded_prefix"]
            if (
                not isinstance(prompt_ids, list)
                or not prompt_ids
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in prompt_ids
                )
                or len(prompt_ids) != target["prompt_token_count"]
                or split_tools._token_sequence_sha256(prompt_ids)
                != target["prompt_token_ids_sha256"]
                or not isinstance(continuation, list)
                or not continuation
                or not isinstance(first_ids, list)
                or len(first_ids) != 1
                or not isinstance(lcp4_ids, list)
                or len(lcp4_ids) != 4
                or continuation[:1] != first_ids
                or continuation[:4] != lcp4_ids
                or target["pre_eos_token_count"] < 6
                or target["generated_continuation_token_count"] != len(continuation)
                or not isinstance(decoded_continuation, str)
                or not decoded_continuation.strip()
                or hashlib.sha256(decoded_continuation.encode("utf-8")).hexdigest()
                != target["generated_continuation_decoded_text_sha256"]
                or target["prompt_token_ids_sha256"]
                != selection_row["token_ids_sha256"]
                or target["prompt_token_count"] != selection_row["token_count"]
                or split_tools._token_sequence_sha256(continuation)
                != target["generated_continuation_token_ids_sha256"]
                or split_tools._token_sequence_sha256(first_ids)
                != target["first_action_token_ids_sha256"]
                or split_tools._token_sequence_sha256(lcp4_ids)
                != target["lcp4_token_ids_sha256"]
                or not isinstance(first_prefix, str)
                or not first_prefix.strip()
                or not isinstance(lcp4_prefix, str)
                or not lcp4_prefix.strip()
                or hashlib.sha256(first_prefix.encode("utf-8")).hexdigest()
                != target["first_action_decoded_prefix_sha256"]
                or hashlib.sha256(lcp4_prefix.encode("utf-8")).hexdigest()
                != target["lcp4_decoded_prefix_sha256"]
            ):
                return False
            first_probe = behavior_by_id[target["first_action_probe_id"]]
            lcp4_probe = behavior_by_id[target["lcp4_probe_id"]]
            generation = generation_by_id[target["generation_prompt_id"]]
            prompt_text = first_probe["prompt"]
            if (
                hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                != target["decoded_prompt_sha256"]
                or lcp4_probe.get("prompt") != prompt_text
                or generation.get("prompt") != prompt_text
                or first_probe.get("category") != "parent-first-action-decoded-prefix"
                or lcp4_probe.get("category") != "parent-lcp4-decoded-prefix"
                or generation.get("category") != "parent-lcp4-decoded-prefix"
                or first_probe.get("kind") != "literal_prefix"
                or lcp4_probe.get("kind") != "literal_prefix"
                or generation.get("kind") != "literal_prefix"
                or generation.get("expected_prefix_generated_token_count") != 4
                or generation.get("require_generation_health") is not True
                or first_probe.get("expected_prefix") != first_prefix
                or lcp4_probe.get("expected_prefix") != lcp4_prefix
                or generation.get("expected_prefix") != lcp4_prefix
            ):
                return False
        if [target.get("index") for target in targets] != list(
            range(FRESH_CANDIDATE_COUNT)
        ):
            return False
        ordered_prompts = [
            generation_by_id[target["generation_prompt_id"]]["prompt"]
            for target in targets
        ]
        ordered_prompt_ids = [target["prompt_token_ids"] for target in targets]
        expected_health_receipts: list[dict[str, Any]] = []
        for target in targets:
            generation = generation_by_id[target["generation_prompt_id"]]
            decoded_continuation = target["generated_continuation_decoded_text"]
            pre_eos_ids = target["generated_continuation_token_ids"][
                : target["pre_eos_token_count"]
            ]
            health = hf_tools._generation_health(
                {
                    **generation,
                    "output": generation["prompt"] + decoded_continuation,
                    "diagnostics": hf_tools._generation_diagnostics(
                        decoded_continuation,
                        pre_eos_ids,
                    ),
                },
                min_continuation_chars=24,
                min_unique_token_ratio=0.20,
            )
            if health.get("passed") is not True:
                return False
            expected_health_receipts.append(
                {
                    "generation_prompt_id": generation["id"],
                    "health": _json_value(
                        health,
                        name="verified_parent_generation_health_receipt",
                    ),
                }
            )
        if adapter.get(
            "parent_generation_health_receipts"
        ) != expected_health_receipts or adapter.get(
            "parent_generation_health_receipts_sha256"
        ) != canonical_sha256(expected_health_receipts):
            return False
        audit = adapter.get("recovery_replay_disjointness_audit")
        if not isinstance(audit, Mapping):
            return False
        audit_fields = {
            "schema",
            "status",
            "minimum_common_token_window",
            "candidate_prompt_count",
            "candidate_decoded_prompt_manifest_sha256",
            "candidate_token_id_manifest_sha256",
            "teacher_token_replay_payload_sha256",
            "training_text_manifest_sha256",
            "training_text_counts",
            "text_scan",
            "stepwise_scan",
            "model_or_tokenizer_accessed",
            "sealed_prompts_decoded",
            "sealed_prompts_supplied_to_model_or_trainer",
            "payload_sha256",
        }
        if set(audit) != audit_fields:
            return False
        audit_body = dict(audit)
        audit_sha = _require_sha256(
            audit_body.pop("payload_sha256"), "replay disjointness audit SHA"
        )
        training_counts = audit.get("training_text_counts")
        text_scan = audit.get("text_scan")
        stepwise_scan = audit.get("stepwise_scan")
        if not (
            isinstance(training_counts, Mapping)
            and set(training_counts)
            == {
                "calibration_texts",
                "precomputed_teacher_generated_texts",
                "on_policy_events",
                "on_policy_texts",
                "total",
            }
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in training_counts.values()
            )
            and training_counts["calibration_texts"] == 640
            and training_counts["total"]
            == training_counts["calibration_texts"]
            + training_counts["precomputed_teacher_generated_texts"]
            + training_counts["on_policy_texts"]
            and isinstance(text_scan, Mapping)
            and set(text_scan)
            == {
                "candidate_in_training_overlap_count",
                "training_in_candidate_overlap_count",
                "all_training_text_fields_scanned",
                "bidirectional_substring_scan",
                "outer_whitespace_stripped_before_scan",
            }
            and text_scan["candidate_in_training_overlap_count"] == 0
            and text_scan["training_in_candidate_overlap_count"] == 0
            and text_scan["all_training_text_fields_scanned"] is True
            and text_scan["bidirectional_substring_scan"] is True
            and text_scan["outer_whitespace_stripped_before_scan"] is True
            and isinstance(stepwise_scan, Mapping)
            and set(stepwise_scan)
            == {
                "stepwise_replay_bank_sha256",
                "event_count",
                "row_count",
                "candidate_window_total_count",
                "candidate_window_unique_count",
                "prompt_window_total_count",
                "rollout_window_total_count",
                "candidate_prompt_window_overlap_count",
                "candidate_rollout_window_overlap_count",
                "all_candidate_offsets_scanned",
                "all_stepwise_prompt_offsets_scanned",
                "all_stepwise_rollout_offsets_scanned",
            }
            and all(
                isinstance(stepwise_scan[name], int)
                and not isinstance(stepwise_scan[name], bool)
                and stepwise_scan[name] >= 0
                for name in (
                    "event_count",
                    "row_count",
                    "candidate_window_total_count",
                    "candidate_window_unique_count",
                    "prompt_window_total_count",
                    "rollout_window_total_count",
                )
            )
            and stepwise_scan["candidate_window_total_count"] >= FRESH_CANDIDATE_COUNT
            and 0
            < stepwise_scan["candidate_window_unique_count"]
            <= stepwise_scan["candidate_window_total_count"]
            and stepwise_scan["candidate_prompt_window_overlap_count"] == 0
            and stepwise_scan["candidate_rollout_window_overlap_count"] == 0
            and stepwise_scan["all_candidate_offsets_scanned"] is True
            and stepwise_scan["all_stepwise_prompt_offsets_scanned"] is True
            and stepwise_scan["all_stepwise_rollout_offsets_scanned"] is True
            and _require_sha256(
                stepwise_scan["stepwise_replay_bank_sha256"],
                "stepwise replay audit SHA",
            )
            == stepwise_scan["stepwise_replay_bank_sha256"]
            and audit.get("schema") == RECOVERY_REPLAY_DISJOINTNESS_AUDIT_SCHEMA
            and audit.get("status") == "passed"
            and audit.get("minimum_common_token_window")
            == MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW
            and audit.get("candidate_prompt_count") == FRESH_CANDIDATE_COUNT
            and audit.get("candidate_decoded_prompt_manifest_sha256")
            == canonical_sha256(ordered_prompts)
            and audit.get("candidate_token_id_manifest_sha256")
            == canonical_sha256(ordered_prompt_ids)
            and _require_sha256(
                audit.get("teacher_token_replay_payload_sha256"),
                "audited teacher replay SHA",
            )
            == audit.get("teacher_token_replay_payload_sha256")
            and _require_sha256(
                audit.get("training_text_manifest_sha256"),
                "training text manifest SHA",
            )
            == audit.get("training_text_manifest_sha256")
            and audit.get("model_or_tokenizer_accessed") is False
            and audit.get("sealed_prompts_decoded") is False
            and audit.get("sealed_prompts_supplied_to_model_or_trainer") is False
            and canonical_sha256(audit_body) == audit_sha
            and adapter.get("recovery_replay_disjointness_audit_sha256") == audit_sha
        ):
            return False
        generation_contract = adapter.get("generation")
        if not isinstance(generation_contract, Mapping):
            return False
        replay_origin = generation_contract.get("teacher_replay_origin")
        return generation_contract == {
            "do_sample": False,
            "max_new_tokens": BEHAVIOR_MAX_NEW_TOKENS,
            "minimum_real_pre_eos_tokens": 4,
            "batch_size": 1,
            "parent_model_target_generation_performed": True,
            "teacher_replay_origin": replay_origin,
            "parent_model_generated_teacher_replay_this_run": (
                replay_origin == "generated"
            ),
        } and replay_origin in {"generated", "cache-hit", "cache-import"}
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        RecoveryInputPreparationError,
        split_tools.ContinuationSplitPreparationError,
    ):
        return False


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
    mode = row.get("mode")
    fresh = mode == BEHAVIOR_MODE_FRESH
    explicit = mode == BEHAVIOR_MODE_EXPLICIT
    adapter = row.get("adapter_provenance")
    adapter_valid = adapter is None
    if fresh and isinstance(adapter, Mapping):
        adapter_body = dict(adapter)
        adapter_claimed = _require_sha256(
            adapter_body.pop("payload_sha256"), "adapter payload SHA"
        )
        selection = adapter.get("selection")
        selection_valid = False
        if isinstance(selection, Mapping):
            selection_body = dict(selection)
            selection_claimed = _require_sha256(
                selection_body.pop("payload_sha256"), "fresh selection payload SHA"
            )
            selection_valid = (
                selection.get("schema") == FRESH_SELECTION_SCHEMA
                and selection.get("mode") == BEHAVIOR_MODE_FRESH
                and selection.get("causal_ab_eligible") is True
                and _fresh_selection_structure_is_valid(selection)
                and selection.get("sealed_exclusion_manifest_used_model_free") is True
                and selection.get("sealed_prompts_supplied_to_model_or_trainer")
                is False
                and selection.get("contains_sealed_data") is False
                and selection.get("disjointness", {}).get(
                    "exact_token_prompt_overlap_count"
                )
                == 0
                and selection.get("disjointness", {}).get(
                    "meaningful_prefix_overlap_count"
                )
                == 0
                and selection.get("disjointness", {}).get(
                    "decoded_candidate_text_in_recovery_calibration_count"
                )
                == 0
                and selection.get("selected_candidate", {}).get("prompt_count")
                == FRESH_CANDIDATE_COUNT
                and isinstance(
                    selection.get("selected_candidate", {}).get("prompt_rows"),
                    list,
                )
                and len(selection.get("selected_candidate", {}).get("prompt_rows"))
                == FRESH_CANDIDATE_COUNT
                and canonical_sha256(
                    selection.get("selected_candidate", {}).get("prompt_rows")
                )
                == selection.get("selected_candidate", {}).get("prompt_manifest_sha256")
                and selection.get("recovery_calibration", {}).get("text_count") == 640
                and selection.get("corpus", {}).get("file_sha256")
                == PINNED_RECOVERY_CORPUS_SHA256
                and selection.get("corpus", {}).get("nonblank_line_count")
                == PINNED_RECOVERY_CORPUS_LINE_COUNT
                and canonical_sha256(selection_body) == selection_claimed
            )
        targets = adapter.get("targets")
        adapter_valid = (
            adapter.get("schema") == TOKEN_DOMAIN_ADAPTER_SCHEMA
            and adapter.get("status") == "frozen-parent-targets"
            and adapter.get("mode") == BEHAVIOR_MODE_FRESH
            and adapter.get("causal_ab_eligible") is True
            and selection_valid
            and isinstance(targets, list)
            and len(targets) == FRESH_CANDIDATE_COUNT
            and canonical_sha256(targets) == adapter.get("target_manifest_sha256")
            and adapter.get("suite_sha256") == row.get("suite_sha256")
            and adapter.get("behavior_probe_count") == BEHAVIOR_PROBE_COUNT
            and adapter.get("first_action_probe_count") == FIRST_ACTION_PROBE_COUNT
            and adapter.get("lcp4_probe_count") == LCP4_PROBE_COUNT
            and adapter.get("generation_prompt_count") == GENERATION_PROMPT_COUNT
            and adapter.get("suite_prompt_exact_token_roundtrip_count")
            == FRESH_CANDIDATE_COUNT
            and adapter.get("parent_full_decode_starts_canonical_prompt_count")
            == FRESH_CANDIDATE_COUNT
            and adapter.get("parent_self_score_pass_count") == BEHAVIOR_PROBE_COUNT
            and adapter.get("parent_self_score_probe_count") == BEHAVIOR_PROBE_COUNT
            and adapter.get("parent_generation_health_pass_count")
            == GENERATION_PROMPT_COUNT
            and adapter.get("parent_generation_health_probe_count")
            == GENERATION_PROMPT_COUNT
            and adapter.get("sealed_prompt_decode_count") == 0
            and adapter.get("sealed_prompt_forward_count") == 0
            and adapter.get("sealed_prompt_output_count") == 0
            and adapter.get("sealed_prompts_supplied_to_model_or_trainer") is False
            and adapter.get("contains_sealed_data") is False
            and adapter.get("publication_authorized") is False
            and _fresh_adapter_cross_bindings_valid(adapter, suite)
            and canonical_sha256(adapter_body) == adapter_claimed
        )
    behavior_ids = (
        [str(probe.get("id")) for probe in behavior_probes]
        if isinstance(behavior_probes, list)
        and all(isinstance(probe, Mapping) for probe in behavior_probes)
        else []
    )
    generation_ids = (
        [str(prompt.get("id")) for prompt in generation_prompts]
        if isinstance(generation_prompts, list)
        and all(isinstance(prompt, Mapping) for prompt in generation_prompts)
        else []
    )
    fresh_probe_contract = not fresh or (
        len(behavior_probes) == BEHAVIOR_PROBE_COUNT
        and len(generation_prompts) == GENERATION_PROMPT_COUNT
        and len(set(behavior_ids)) == len(behavior_ids)
        and len(set(generation_ids)) == len(generation_ids)
        and all(
            probe.get("kind") == "literal_prefix"
            and isinstance(probe.get("expected_prefix"), str)
            and bool(probe["expected_prefix"].strip())
            and isinstance(probe.get("prompt"), str)
            and bool(probe["prompt"].strip())
            for probe in behavior_probes
        )
        and all(
            prompt.get("kind") == "literal_prefix"
            and prompt.get("require_generation_health") is True
            and prompt.get("expected_prefix_generated_token_count") == 4
            and isinstance(prompt.get("expected_prefix"), str)
            and bool(prompt["expected_prefix"].strip())
            and isinstance(prompt.get("prompt"), str)
            and bool(prompt["prompt"].strip())
            for prompt in generation_prompts
        )
        and sum(
            probe.get("category") == "parent-first-action-decoded-prefix"
            for probe in behavior_probes
        )
        == FIRST_ACTION_PROBE_COUNT
        and sum(
            probe.get("category") == "parent-lcp4-decoded-prefix"
            for probe in behavior_probes
        )
        == LCP4_PROBE_COUNT
    )
    if (
        row.get("schema") != BEHAVIOR_SUITE_BINDING_SCHEMA
        or row.get("status") != "frozen"
        or not (fresh or explicit)
        or row.get("causal_ab_eligible") is not fresh
        or row.get("split") != "candidate"
        or row.get("sealed_exclusion_manifest_used_model_free") is not fresh
        or row.get("sealed_prompts_supplied_to_model_or_trainer") is not False
        or row.get("sealed_data_accessed") is not False
        or row.get("contains_sealed_data") is not False
        or row.get("publication_authorized") is not False
        or not isinstance(behavior_probes, list)
        or not behavior_probes
        or not isinstance(generation_prompts, list)
        or not generation_prompts
        or row.get("behavior_probe_count") != len(behavior_probes)
        or row.get("generation_prompt_count") != len(generation_prompts)
        or row.get("first_action_probe_count")
        != sum(
            isinstance(probe, Mapping)
            and probe.get("category") == "parent-first-action-decoded-prefix"
            for probe in behavior_probes
        )
        or row.get("lcp4_probe_count")
        != sum(
            isinstance(probe, Mapping)
            and probe.get("category") == "parent-lcp4-decoded-prefix"
            for probe in behavior_probes
        )
        or not adapter_valid
        or not fresh_probe_contract
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
    behavior = verify_behavior_suite_binding(row["behavior_eval_suite"])
    if behavior.get("causal_ab_eligible") is True:
        adapter = behavior["adapter_provenance"]
        tokenizers = teacher["tokenizer_fingerprints"]
        generation_by_id = {
            prompt["id"]: prompt for prompt in behavior["suite"]["generation_prompts"]
        }
        targets = adapter["targets"]
        selection = adapter["selection"]
        reservoir_rows = _verify_fresh_candidate_reservoir_payload(
            selection["candidate_reservoir"]
        )
        recomputed_derivation = _derive_fresh_candidate_selection(
            reservoir_rows=reservoir_rows,
            teacher_payload=teacher,
        )
        recomputed_disjointness = _fresh_candidate_replay_disjointness_audit(
            decoded_prompts=[
                generation_by_id[target["generation_prompt_id"]]["prompt"]
                for target in targets
            ],
            prompt_token_ids=[target["prompt_token_ids"] for target in targets],
            teacher_payload=teacher,
        )
        if (
            tokenizers.get("source") != tokenizers.get("candidate")
            or adapter.get("source_model_fingerprint_sha256")
            != teacher["source_model_fingerprint"]["sha256"]
            or adapter.get("source_model_state_dict_sha256") != state_sha
            or adapter.get("identity_tokenizer_fingerprint_sha256")
            != tokenizers["source"]["sha256"]
            or adapter.get("recovery_replay_disjointness_audit")
            != recomputed_disjointness
            or adapter.get("recovery_replay_disjointness_audit_sha256")
            != recomputed_disjointness["payload_sha256"]
            or selection.get("replay_exclusion_derivation")
            != recomputed_derivation
            or selection.get("final_replay_disjointness_audit_sha256")
            != recomputed_disjointness["payload_sha256"]
            or [target.get("prompt_id") for target in targets]
            != recomputed_derivation["selected_prompt_ids"]
            or [
                row.get("reservoir_index")
                for row in selection["selected_candidate"]["prompt_rows"]
            ]
            != recomputed_derivation["selected_reservoir_indices"]
        ):
            raise RecoveryInputPreparationError(
                "Fresh behavior adapter is bound to different replay/model/tokenizer "
                "evidence."
            )
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
    teacher_replay_acquisition: Mapping[str, Any],
    parent_model_usage: Mapping[str, Any],
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
        "behavior_mode": behavior_binding["mode"],
        "causal_ab_eligible": behavior_binding["causal_ab_eligible"],
        "behavior_probe_count": behavior_binding["behavior_probe_count"],
        "generation_prompt_count": behavior_binding["generation_prompt_count"],
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
        "teacher_generation_completed": (
            teacher_replay_acquisition.get("origin") == "generated"
        ),
        "teacher_replay_acquisition": _json_value(
            teacher_replay_acquisition,
            name="teacher_replay_acquisition",
        ),
        "parent_model_usage": _json_value(
            parent_model_usage,
            name="parent_model_usage",
        ),
        "fresh_candidate_reservoir": (
            _json_value(
                behavior_binding["adapter_provenance"]["selection"][
                    "external_candidate_source"
                ],
                name="fresh_candidate_reservoir",
            )
            if behavior_binding["mode"] == BEHAVIOR_MODE_FRESH
            else None
        ),
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
    acquisition = row.get("teacher_replay_acquisition")
    usage = row.get("parent_model_usage")
    fresh_reservoir = row.get("fresh_candidate_reservoir")
    if not isinstance(acquisition, Mapping) or set(acquisition) != {
        "schema",
        "cache_mode",
        "origin",
        "cache_result",
        "cache_request_sha256",
        "cache_envelope_sha256",
        "cache_entry_relative_path",
        "cache_verified_before_final_selection",
        "cache_import",
    }:
        raise RecoveryInputPreparationError(
            "Build report teacher replay acquisition changed."
        )
    if not isinstance(usage, Mapping) or set(usage) != {
        "schema",
        "model_load_count",
        "load_purpose",
        "teacher_replay_generated_this_run",
        "teacher_replay_reused_from_cache",
        "fresh_target_generation_performed",
    }:
        raise RecoveryInputPreparationError("Build report parent model usage changed.")
    if row.get("behavior_mode") == BEHAVIOR_MODE_FRESH:
        _verify_external_fresh_reservoir_receipt(fresh_reservoir)
    elif fresh_reservoir is not None:
        raise RecoveryInputPreparationError(
            "Diagnostic build report cannot bind a fresh candidate reservoir."
        )
    cache_mode = acquisition.get("cache_mode")
    origin = acquisition.get("origin")
    cache_result = acquisition.get("cache_result")
    cache_enabled = cache_mode != "off"
    if cache_enabled:
        _require_sha256(
            acquisition.get("cache_request_sha256"), "cache request SHA"
        )
        _require_sha256(
            acquisition.get("cache_envelope_sha256"), "cache envelope SHA"
        )
        relative = acquisition.get("cache_entry_relative_path")
        if relative != (
            f"{acquisition['cache_request_sha256']}/"
            f"{hf_teacher_replay_cache.CACHE_FILENAME}"
        ):
            raise RecoveryInputPreparationError("Cache entry relative path changed.")
    elif any(
        acquisition.get(name) is not None
        for name in (
            "cache_request_sha256",
            "cache_envelope_sha256",
            "cache_entry_relative_path",
        )
    ):
        raise RecoveryInputPreparationError("Disabled cache cannot bind an entry.")
    expected_reused = origin in {"cache-hit", "cache-import"}
    generated_results = {
        "published",
        "existing-byte-identical",
        "concurrent-byte-identical",
    }
    valid_mode_result = (
        (cache_mode == "off" and origin == "generated" and cache_result == "disabled")
        or (
            cache_mode in {"auto", "require"}
            and origin == "cache-hit"
            and cache_result == "hit"
        )
        or (
            cache_mode in {"auto", "require"}
            and origin == "cache-import"
            and cache_result in generated_results
        )
        or (
            cache_mode == "auto"
            and origin == "generated"
            and cache_result in generated_results
        )
        or (
            cache_mode == "refresh"
            and origin == "generated"
            and cache_result in generated_results
        )
    )
    import_receipt = acquisition.get("cache_import")
    if origin == "cache-import":
        _verify_teacher_cache_import_receipt(
            import_receipt,
            current_request_sha256=acquisition["cache_request_sha256"],
            current_envelope_sha256=acquisition["cache_envelope_sha256"],
        )
    elif import_receipt is not None:
        raise RecoveryInputPreparationError(
            "Non-import replay acquisition cannot bind an import receipt."
        )
    fresh_targets = usage.get("fresh_target_generation_performed")
    expected_load_purpose = (
        "fresh-targets"
        if expected_reused
        else (
            "teacher-replay-and-fresh-targets"
            if fresh_targets is True
            else "teacher-replay"
        )
    )
    if (
        acquisition.get("schema") != TEACHER_REPLAY_ACQUISITION_SCHEMA
        or usage.get("schema") != PARENT_MODEL_USAGE_SCHEMA
        or cache_mode not in TEACHER_REPLAY_CACHE_MODES
        or origin not in {"generated", "cache-hit", "cache-import"}
        or not valid_mode_result
        or acquisition.get("cache_verified_before_final_selection")
        is not cache_enabled
        or usage.get("model_load_count") != 1
        or usage.get("load_purpose") != expected_load_purpose
        or usage.get("teacher_replay_generated_this_run") is expected_reused
        or usage.get("teacher_replay_reused_from_cache") is not expected_reused
        or not isinstance(fresh_targets, bool)
        or fresh_targets is not (row.get("behavior_mode") == BEHAVIOR_MODE_FRESH)
        or (not fresh_targets and (cache_enabled or expected_reused))
    ):
        raise RecoveryInputPreparationError(
            "Build report replay acquisition or parent usage is inconsistent."
        )
    if (
        row.get("schema") != BUILD_REPORT_SCHEMA
        or row.get("status") != "ready-experimental-unaccepted"
        or outputs["replay_bank"].get("filename") != REPLAY_FILENAME
        or outputs["recovery_plan"].get("filename") != PLAN_FILENAME
        or outputs["build_report"] != {"filename": BUILD_REPORT_FILENAME}
        or row.get("recovery_steps") != STEPS
        or row.get("behavior_mode") not in {BEHAVIOR_MODE_FRESH, BEHAVIOR_MODE_EXPLICIT}
        or row.get("causal_ab_eligible")
        is not (row.get("behavior_mode") == BEHAVIOR_MODE_FRESH)
        or isinstance(row.get("behavior_probe_count"), bool)
        or not isinstance(row.get("behavior_probe_count"), int)
        or row.get("behavior_probe_count", 0) <= 0
        or isinstance(row.get("generation_prompt_count"), bool)
        or not isinstance(row.get("generation_prompt_count"), int)
        or row.get("generation_prompt_count", 0) <= 0
        or row.get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or row.get("retained_checkpoint_steps") != []
        or row.get("source_torch_dtype") != "float16"
        or row.get("candidate_recovery_torch_dtype") != "float32"
        or row.get("local_files_only") is not True
        or row.get("trust_remote_code") is not False
        or row.get("model_weights_loaded") is not True
        or row.get("teacher_generation_completed")
        is not usage.get("teacher_replay_generated_this_run")
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


def _token_windows(
    token_ids: Sequence[int], window_size: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(token_ids[start : start + window_size])
        for start in range(0, max(0, len(token_ids) - window_size + 1))
    )


def _freeze_fresh_candidate_replay_exclusions(
    teacher_payload: Mapping[str, Any],
) -> FreshCandidateReplayExclusions:
    """Freeze every recovery text and stepwise token window used for filtering."""

    if not isinstance(teacher_payload, Mapping):
        raise RecoveryInputPreparationError("Teacher replay payload must be an object.")
    try:
        calibration = hf_recovery._freeze_ordered_recovery_texts(
            teacher_payload.get("calibration_texts"),
            field_name="calibration_texts",
        )
        teacher_generated = hf_recovery._freeze_ordered_recovery_texts(
            teacher_payload.get("precomputed_teacher_generated_texts"),
            field_name="precomputed_teacher_generated_texts",
        )
        on_policy = hf_recovery._freeze_on_policy_replay_bank(
            teacher_payload.get("on_policy_replay_bank")
        )
        stepwise = hf_recovery._freeze_stepwise_replay_bank(
            teacher_payload.get("stepwise_replay_bank")
        )
    except (TypeError, ValueError) as error:
        raise RecoveryInputPreparationError(
            "Fresh causal mode requires fully formed frozen text and token replay banks."
        ) from error

    training_text_rows: list[Mapping[str, Any]] = []
    indexed_training_texts: list[tuple[str, int, int | None, str]] = []
    for source, texts in (
        ("calibration_texts", calibration),
        ("precomputed_teacher_generated_texts", teacher_generated),
    ):
        for index, text in enumerate(texts):
            training_text_rows.append(
                {
                    "source": source,
                    "event_index": None,
                    "text_index": index,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
            indexed_training_texts.append((source, index, None, text))
    for event_index, event in enumerate(on_policy):
        for text_index, text in enumerate(event):
            training_text_rows.append(
                {
                    "source": "on_policy_replay_bank",
                    "event_index": event_index,
                    "text_index": text_index,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
            indexed_training_texts.append(
                ("on_policy_replay_bank", text_index, event_index, text)
            )

    prompt_windows: set[tuple[int, ...]] = set()
    rollout_windows: set[tuple[int, ...]] = set()
    prompt_window_total = 0
    rollout_window_total = 0
    stepwise_row_count = 0
    for event in stepwise:
        for row in event.rows:
            stepwise_row_count += 1
            row_prompt_windows = _token_windows(
                row.prompt_input_ids, MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW
            )
            row_rollout_windows = _token_windows(
                row.rollout_input_ids, MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW
            )
            prompt_window_total += len(row_prompt_windows)
            rollout_window_total += len(row_rollout_windows)
            prompt_windows.update(row_prompt_windows)
            rollout_windows.update(row_rollout_windows)

    return FreshCandidateReplayExclusions(
        calibration=tuple(calibration),
        teacher_generated=tuple(teacher_generated),
        on_policy=tuple(tuple(event) for event in on_policy),
        stepwise=tuple(stepwise),
        indexed_training_texts=tuple(indexed_training_texts),
        training_text_rows=tuple(training_text_rows),
        prompt_windows=frozenset(prompt_windows),
        rollout_windows=frozenset(rollout_windows),
        prompt_window_total=prompt_window_total,
        rollout_window_total=rollout_window_total,
        stepwise_row_count=stepwise_row_count,
        stepwise_sha256=hf_recovery._stepwise_replay_bank_sha256(stepwise),
    )


def _freeze_fresh_candidate_reservoir_rows(
    reservoir_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Authenticate the complete ordered reservoir without tokenizer access."""

    if (
        isinstance(reservoir_rows, (str, bytes, Mapping))
        or not isinstance(reservoir_rows, Sequence)
        or len(reservoir_rows) < FRESH_CANDIDATE_COUNT
    ):
        raise RecoveryInputPreparationError(
            "Fresh candidate reservoir must contain at least 24 ordered rows."
        )
    expected_fields = {
        "reservoir_index",
        "prompt_id",
        "token_ids",
        "token_ids_sha256",
        "decoded_prompt",
        "decoded_prompt_sha256",
        "source_line_number",
        "source_prompt_sha256",
    }
    frozen: list[dict[str, Any]] = []
    prompt_ids: set[str] = set()
    decoded_prompts: set[str] = set()
    previous_source_line = 0
    external_seed = _load_external_fresh_reservoir_seed()
    external_by_line = {
        int(row["source_line_number"]): row for row in external_seed.prompts
    }
    for index, row in enumerate(reservoir_rows):
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise RecoveryInputPreparationError(
                f"Fresh reservoir row {index} fields changed."
            )
        raw_tokens = row.get("token_ids")
        if isinstance(raw_tokens, (str, bytes, Mapping)) or not isinstance(
            raw_tokens, Sequence
        ):
            raise RecoveryInputPreparationError(
                f"Fresh reservoir row {index} token_ids must be a sequence."
            )
        tokens = tuple(raw_tokens)
        if (
            not (
                split_tools.MINIMUM_MEANINGFUL_PREFIX_TOKENS
                <= len(tokens)
                <= MAX_LENGTH
            )
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in tokens
            )
        ):
            raise RecoveryInputPreparationError(
                f"Fresh reservoir row {index} has invalid token IDs."
            )
        digest = split_tools._token_sequence_sha256(tokens)
        decoded = row.get("decoded_prompt")
        source_line = row.get("source_line_number")
        if (
            row.get("reservoir_index") != index
            or row.get("prompt_id") != f"prompt-{digest}"
            or row.get("token_ids_sha256") != digest
            or not isinstance(decoded, str)
            or not decoded.strip()
            or row.get("decoded_prompt_sha256")
            != hashlib.sha256(decoded.encode("utf-8")).hexdigest()
            or isinstance(source_line, bool)
            or not isinstance(source_line, int)
            or source_line <= 0
        ):
            raise RecoveryInputPreparationError(
                f"Fresh reservoir row {index} failed content authentication."
            )
        _require_sha256(
            row.get("source_prompt_sha256"),
            f"fresh reservoir row {index} source prompt SHA",
        )
        expected_source = external_by_line.get(source_line)
        if (
            source_line <= previous_source_line
            or expected_source is None
            or row["source_prompt_sha256"]
            != expected_source["source_prompt_sha256"]
            or decoded != expected_source["text"]
        ):
            raise RecoveryInputPreparationError(
                "Fresh reservoir row differs from strict external source order/content."
            )
        previous_source_line = source_line
        if row["prompt_id"] in prompt_ids or decoded in decoded_prompts:
            raise RecoveryInputPreparationError(
                "Fresh reservoir contains duplicate prompts or decoded text."
            )
        prompt_ids.add(str(row["prompt_id"]))
        decoded_prompts.add(decoded)
        frozen.append(
            {
                "reservoir_index": index,
                "prompt_id": str(row["prompt_id"]),
                "token_ids": [int(token_id) for token_id in tokens],
                "token_ids_sha256": digest,
                "decoded_prompt": decoded,
                "decoded_prompt_sha256": str(row["decoded_prompt_sha256"]),
                "source_line_number": source_line,
                "source_prompt_sha256": str(row["source_prompt_sha256"]),
            }
        )
    return frozen


def _teacher_replay_cache_request(
    *,
    actual_parent_tree: Mapping[str, Any],
    expected_parent_tree_sha256: str,
    calibration_path: Path,
    ordered_calibration: Sequence[str],
    source_tokenizer_fingerprint: Mapping[str, Any],
    candidate_tokenizer_fingerprint: Mapping[str, Any],
    torch: Any,
    transformers: Any,
    teacher_device: str,
) -> dict[str, Any]:
    """Build cache identity without accepting any candidate or held-out input."""

    try:
        producers = hf_teacher_replay_cache.producer_byte_hash_manifest(
            (
                ("prepare_recovery_inputs", Path(__file__)),
                (
                    "hf_teacher_token_replay",
                    Path(hf_teacher_token_replay.__file__),
                ),
                ("hf_recovery", Path(hf_recovery.__file__)),
                (
                    "hf_teacher_replay_cache",
                    Path(hf_teacher_replay_cache.__file__),
                ),
            )
        )
        runtime = hf_teacher_replay_cache.runtime_identity(
            torch,
            transformers,
            teacher_device,
        )
        return hf_teacher_replay_cache.build_request(
            expected_parent_artifact_tree_sha256=expected_parent_tree_sha256,
            actual_parent_artifact_tree=actual_parent_tree,
            calibration_source_file_sha256=_file_sha256(calibration_path),
            ordered_calibration_texts=ordered_calibration,
            source_tokenizer_fingerprint=source_tokenizer_fingerprint,
            candidate_tokenizer_fingerprint=candidate_tokenizer_fingerprint,
            teacher_parameters=_teacher_parameters(teacher_device),
            producer_manifest=producers,
            runtime=runtime,
        )
    except hf_teacher_replay_cache.TeacherReplayCacheError as error:
        raise RecoveryInputPreparationError(
            "Teacher replay cache request failed authentication."
        ) from error


def _read_exact_compatible_teacher_cache_import(
    *,
    cache_dir: Path | str,
    current_request: Mapping[str, Any],
    source_request_sha256: str,
    source_envelope_sha256: str,
) -> CompatibleTeacherCacheImport:
    """Read only one explicitly named, semantically compatible legacy entry."""

    source_request_sha = _require_sha256(
        source_request_sha256, "cache import source request SHA"
    )
    source_envelope_sha = _require_sha256(
        source_envelope_sha256, "cache import source envelope SHA"
    )
    try:
        verified_current = hf_teacher_replay_cache.verify_request(current_request)
        root = hf_teacher_replay_cache._absolute_path(cache_dir)
        if not hf_teacher_replay_cache._lexists(root):
            raise hf_teacher_replay_cache.TeacherReplayCacheError(
                "Explicit cache import root does not exist."
            )
        hf_teacher_replay_cache._require_real_directory(
            root, "teacher replay cache import directory"
        )
        entry = root / source_request_sha
        if entry.parent != root or not hf_teacher_replay_cache._lexists(entry):
            raise hf_teacher_replay_cache.TeacherReplayCacheError(
                "Explicit cache import request entry is missing."
            )
        hf_teacher_replay_cache._require_real_directory(
            entry, "teacher replay cache import request path"
        )
        children = list(entry.iterdir())
        if len(children) != 1 or children[0].name != hf_teacher_replay_cache.CACHE_FILENAME:
            raise hf_teacher_replay_cache.TeacherReplayCacheError(
                "Explicit cache import entry has unexpected contents."
            )
        envelope_path = children[0]
        envelope_info = hf_teacher_replay_cache._lstat_or_none(envelope_path)
        if (
            envelope_info is None
            or stat.S_ISLNK(envelope_info.st_mode)
            or not stat.S_ISREG(envelope_info.st_mode)
        ):
            raise hf_teacher_replay_cache.TeacherReplayCacheError(
                "Explicit cache import envelope must be one real file."
            )
        raw_envelope = hf_teacher_replay_cache._strict_json_file(envelope_path)
        if raw_envelope.get("envelope_sha256") != source_envelope_sha:
            raise hf_teacher_replay_cache.TeacherReplayCacheError(
                "Explicit cache import envelope SHA differs from authorization."
            )
        source_request = hf_teacher_replay_cache.verify_request(
            raw_envelope.get("request")
        )
        if source_request["request_sha256"] != source_request_sha:
            raise hf_teacher_replay_cache.TeacherReplayCacheError(
                "Explicit cache import request SHA differs from authorization."
            )
        imported = hf_teacher_replay_cache.read_entry(root, source_request)
        if imported is None or imported["envelope_sha256"] != source_envelope_sha:
            raise hf_teacher_replay_cache.TeacherReplayCacheError(
                "Explicit cache import disappeared or changed during authentication."
            )
        imported = hf_teacher_replay_cache.verify_envelope(
            imported, expected_request=source_request
        )
    except (OSError, hf_teacher_replay_cache.TeacherReplayCacheError) as error:
        raise RecoveryInputPreparationError(
            "Explicit teacher replay cache import failed strict path/request/envelope "
            "authentication."
        ) from error

    old_semantics = dict(source_request)
    old_semantics.pop("request_sha256")
    old_producers = old_semantics.pop("producer_byte_hash_manifest")
    current_semantics = dict(verified_current)
    current_semantics.pop("request_sha256")
    current_producers = current_semantics.pop("producer_byte_hash_manifest")
    old_rows = {row["role"]: dict(row) for row in old_producers["files"]}
    current_rows = {
        row["role"]: dict(row) for row in current_producers["files"]
    }
    nonprep_roles = {
        "hf_teacher_token_replay",
        "hf_recovery",
        "hf_teacher_replay_cache",
    }
    if (
        old_semantics != current_semantics
        or old_rows.get("prepare_recovery_inputs")
        != LEGACY_TEACHER_CACHE_IMPORT_PREP_PRODUCER
        or set(old_rows) != {*nonprep_roles, "prepare_recovery_inputs"}
        or set(current_rows) != set(old_rows)
        or any(old_rows[role] != current_rows[role] for role in nonprep_roles)
        or current_rows["prepare_recovery_inputs"]
        == old_rows["prepare_recovery_inputs"]
    ):
        raise RecoveryInputPreparationError(
            "Explicit teacher replay cache import is not semantically compatible "
            "with the current producer request."
        )
    receipt_body = {
        "schema": TEACHER_REPLAY_CACHE_IMPORT_RECEIPT_SCHEMA,
        "status": "compatible-source-verified-before-model-load",
        "authorized_by_explicit_request_and_envelope_cli": True,
        "source_producer_commit": LEGACY_TEACHER_CACHE_IMPORT_SOURCE_COMMIT,
        "source_request_sha256": source_request_sha,
        "source_envelope_sha256": source_envelope_sha,
        "source_teacher_payload_sha256": _require_sha256(
            imported.get("teacher_payload_sha256"),
            "cache import source teacher payload SHA",
        ),
        "source_prepare_producer_allowlist": dict(
            LEGACY_TEACHER_CACHE_IMPORT_PREP_PRODUCER
        ),
        "current_prepare_producer": current_rows["prepare_recovery_inputs"],
        "current_request_sha256": verified_current["request_sha256"],
        "request_fields_equal_except_producer_manifest_and_request_self_hash": True,
        "only_prepare_producer_row_changed": True,
        "other_producer_rows_byte_identical": True,
        "reservoir_frozen_before_import_read": True,
        "source_verified_before_model_load": True,
        "contains_heldout_or_reservoir_text": False,
        "live_source_fingerprint_verified_after_model_load": False,
        "teacher_replay_generated_during_import": False,
    }
    pre_model_receipt = {
        **receipt_body,
        "pre_model_compatibility_sha256": canonical_sha256(receipt_body),
    }
    return CompatibleTeacherCacheImport(
        request=source_request,
        envelope=imported,
        pre_model_receipt=pre_model_receipt,
    )


def _finalize_teacher_cache_import_receipt(
    *,
    compatible_import: CompatibleTeacherCacheImport,
    current_envelope: Mapping[str, Any],
    publication_result: str,
) -> dict[str, Any]:
    pre_model = _json_value(
        compatible_import.pre_model_receipt,
        name="teacher cache import pre-model receipt",
    )
    pre_body = dict(pre_model)
    pre_claimed = _require_sha256(
        pre_body.pop("pre_model_compatibility_sha256"),
        "cache import pre-model compatibility SHA",
    )
    if canonical_sha256(pre_body) != pre_claimed:
        raise RecoveryInputPreparationError(
            "Teacher cache import pre-model receipt changed."
        )
    body = {
        **pre_model,
        "status": "compatible-replay-republished-under-current-request",
        "live_source_fingerprint_verified_after_model_load": True,
        "current_envelope_sha256": _require_sha256(
            current_envelope.get("envelope_sha256"),
            "cache import current envelope SHA",
        ),
        "current_teacher_payload_sha256": _require_sha256(
            current_envelope.get("teacher_payload_sha256"),
            "cache import current teacher payload SHA",
        ),
        "current_cache_publication_result": publication_result,
        "teacher_payload_byte_identical": (
            current_envelope.get("teacher_token_replay")
            == compatible_import.envelope.get("teacher_token_replay")
        ),
    }
    if (
        body["teacher_payload_byte_identical"] is not True
        or body["current_teacher_payload_sha256"]
        != body["source_teacher_payload_sha256"]
    ):
        raise RecoveryInputPreparationError(
            "Teacher cache import changed replay bytes during republication."
        )
    return {**body, "payload_sha256": canonical_sha256(body)}


def _verify_teacher_cache_import_receipt(
    receipt: Mapping[str, Any],
    *,
    current_request_sha256: str,
    current_envelope_sha256: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "authorized_by_explicit_request_and_envelope_cli",
        "source_producer_commit",
        "source_request_sha256",
        "source_envelope_sha256",
        "source_teacher_payload_sha256",
        "source_prepare_producer_allowlist",
        "current_prepare_producer",
        "current_request_sha256",
        "request_fields_equal_except_producer_manifest_and_request_self_hash",
        "only_prepare_producer_row_changed",
        "other_producer_rows_byte_identical",
        "reservoir_frozen_before_import_read",
        "source_verified_before_model_load",
        "contains_heldout_or_reservoir_text",
        "live_source_fingerprint_verified_after_model_load",
        "teacher_replay_generated_during_import",
        "pre_model_compatibility_sha256",
        "current_envelope_sha256",
        "current_teacher_payload_sha256",
        "current_cache_publication_result",
        "teacher_payload_byte_identical",
        "payload_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != fields:
        raise RecoveryInputPreparationError(
            "Teacher cache import receipt fields changed."
        )
    body = dict(receipt)
    claimed = _require_sha256(body.pop("payload_sha256"), "cache import receipt SHA")
    pre_body = dict(receipt)
    for name in (
        "current_envelope_sha256",
        "current_teacher_payload_sha256",
        "current_cache_publication_result",
        "teacher_payload_byte_identical",
        "payload_sha256",
    ):
        pre_body.pop(name)
    pre_claimed = _require_sha256(
        pre_body.pop("pre_model_compatibility_sha256"),
        "cache import pre-model compatibility SHA",
    )
    pre_body["status"] = "compatible-source-verified-before-model-load"
    pre_body["live_source_fingerprint_verified_after_model_load"] = False
    source_prepare = receipt.get("source_prepare_producer_allowlist")
    current_prepare = receipt.get("current_prepare_producer")
    result = receipt.get("current_cache_publication_result")
    if (
        receipt.get("schema") != TEACHER_REPLAY_CACHE_IMPORT_RECEIPT_SCHEMA
        or receipt.get("status")
        != "compatible-replay-republished-under-current-request"
        or receipt.get("authorized_by_explicit_request_and_envelope_cli") is not True
        or receipt.get("source_producer_commit")
        != LEGACY_TEACHER_CACHE_IMPORT_SOURCE_COMMIT
        or source_prepare != LEGACY_TEACHER_CACHE_IMPORT_PREP_PRODUCER
        or not isinstance(current_prepare, Mapping)
        or current_prepare.get("role") != "prepare_recovery_inputs"
        or set(current_prepare) != {"role", "filename", "byte_count", "sha256"}
        or current_prepare == source_prepare
        or receipt.get("current_request_sha256") != current_request_sha256
        or receipt.get("current_envelope_sha256") != current_envelope_sha256
        or receipt.get("current_teacher_payload_sha256")
        != receipt.get("source_teacher_payload_sha256")
        or result
        not in {"published", "existing-byte-identical", "concurrent-byte-identical"}
        or any(
            receipt.get(name) is not True
            for name in (
                "request_fields_equal_except_producer_manifest_and_request_self_hash",
                "only_prepare_producer_row_changed",
                "other_producer_rows_byte_identical",
                "reservoir_frozen_before_import_read",
                "source_verified_before_model_load",
                "live_source_fingerprint_verified_after_model_load",
                "teacher_payload_byte_identical",
            )
        )
        or receipt.get("contains_heldout_or_reservoir_text") is not False
        or receipt.get("teacher_replay_generated_during_import") is not False
        or canonical_sha256(pre_body) != pre_claimed
        or canonical_sha256(body) != claimed
    ):
        raise RecoveryInputPreparationError(
            "Teacher cache import receipt failed authentication."
        )
    _require_sha256(receipt.get("source_request_sha256"), "import source request SHA")
    _require_sha256(receipt.get("source_envelope_sha256"), "import source envelope SHA")
    _require_sha256(
        receipt.get("source_teacher_payload_sha256"), "import teacher payload SHA"
    )
    _require_sha256(current_prepare.get("sha256"), "current prep producer SHA")
    if (
        isinstance(current_prepare.get("byte_count"), bool)
        or not isinstance(current_prepare.get("byte_count"), int)
        or current_prepare["byte_count"] <= 0
    ):
        raise RecoveryInputPreparationError(
            "Current prep producer byte count is invalid."
        )
    return _json_value(receipt, name="teacher cache import receipt")


def _verify_fresh_candidate_reservoir_payload(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "provenance",
        "rows",
        "rows_sha256",
    }:
        raise RecoveryInputPreparationError(
            "Fresh candidate reservoir payload fields changed."
        )
    frozen = _freeze_fresh_candidate_reservoir_rows(payload["rows"])
    if payload.get("rows_sha256") != canonical_sha256(frozen):
        raise RecoveryInputPreparationError(
            "Fresh candidate reservoir row binding changed."
        )
    manifest = [
        {key: row[key] for key in row if key not in {"token_ids", "decoded_prompt"}}
        for row in frozen
    ]
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RecoveryInputPreparationError(
            "Fresh candidate reservoir provenance must be an object."
        )
    provenance_body = dict(provenance)
    claimed = _require_sha256(
        provenance_body.pop("candidate_reservoir_payload_sha256"),
        "candidate reservoir payload SHA",
    )
    statistics = provenance.get("preselection_statistics")
    statistic_fields = {
        "records_considered",
        "unique_source_prompt_count",
        "duplicate_source_row_count",
        "skipped_candidate_text_present_in_calibration",
        "skipped_under_minimum_prompt_tokens",
        "skipped_over_token_limit",
        "skipped_exact_prompt_collision",
        "skipped_meaningful_prefix_collision",
        "skipped_decoded_calibration_collision",
        "skipped_duplicate_decoded_prompt",
        "accepted_into_reservoir",
    }
    statistics_valid = (
        isinstance(statistics, Mapping)
        and set(statistics) == statistic_fields
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in statistics.values()
        )
        and statistics["records_considered"]
        == statistics["unique_source_prompt_count"]
        and statistics["unique_source_prompt_count"]
        == PINNED_EXTERNAL_FRESH_RESERVOIR_EXPANDED_COUNT
        and statistics["duplicate_source_row_count"] == 0
        and statistics["accepted_into_reservoir"] == len(frozen)
        and sum(
            value
            for name, value in statistics.items()
            if name
            not in {
                "records_considered",
                "unique_source_prompt_count",
                "duplicate_source_row_count",
            }
        )
        == statistics["records_considered"]
    )
    try:
        _verify_external_fresh_reservoir_receipt(
            provenance.get("external_candidate_source")
        )
    except RecoveryInputPreparationError:
        raise
    corpus = provenance.get("corpus")
    corpus_valid = (
        isinstance(corpus, Mapping)
        and corpus
        == {
            "file_sha256": PINNED_RECOVERY_CORPUS_SHA256,
            "nonblank_line_count": PINNED_RECOVERY_CORPUS_LINE_COUNT,
            "duplicate_source_row_count": (
                PINNED_RECOVERY_CORPUS_DUPLICATE_SOURCE_ROW_COUNT
            ),
            "purpose": "teacher-recovery-calibration-lineage-only",
        }
    )
    if (
        provenance.get("status")
        != "candidate-reservoir-frozen-before-parent-load"
        or provenance.get("reservoir_prompt_count") != len(frozen)
        or provenance.get("reservoir_manifest_sha256")
        != canonical_sha256(manifest)
        or provenance.get("reservoir_rows_hold_token_ids_and_decoded_text") is not True
        or provenance.get("parent_model_loaded_during_reservoir_freeze") is not False
        or provenance.get("parent_forward_or_generate_during_reservoir_freeze")
        is not False
        or provenance.get("identity_tokenizer_used_during_reservoir_freeze")
        is not True
        or provenance.get("reservoir_order")
        != "external-round-robin-source-line-order-after-static-exclusions"
        or not corpus_valid
        or not statistics_valid
        or canonical_sha256(provenance_body) != claimed
    ):
        raise RecoveryInputPreparationError(
            "Fresh candidate reservoir provenance failed authentication."
        )
    return frozen


def _fresh_candidate_reservoir_rows(
    reservoir: FreshRecoveryCandidateReservoir,
) -> list[dict[str, Any]]:
    if len(reservoir.prompts) != len(reservoir.decoded_prompts):
        raise RecoveryInputPreparationError(
            "Fresh candidate reservoir prompt/text alignment changed."
        )
    rows = [
        {
            "reservoir_index": index,
            "prompt_id": prompt.prompt_id,
            "token_ids": list(prompt.token_ids),
            "token_ids_sha256": split_tools._token_sequence_sha256(
                prompt.token_ids
            ),
            "decoded_prompt": reservoir.decoded_prompts[index],
            "decoded_prompt_sha256": hashlib.sha256(
                reservoir.decoded_prompts[index].encode("utf-8")
            ).hexdigest(),
            "source_line_number": int(prompt.payload["source_line_number"]),
            "source_prompt_sha256": str(prompt.payload["source_prompt_sha256"]),
        }
        for index, prompt in enumerate(reservoir.prompts)
    ]
    return _verify_fresh_candidate_reservoir_payload(
        {
            "provenance": reservoir.provenance,
            "rows": rows,
            "rows_sha256": canonical_sha256(rows),
        }
    )


def _derive_fresh_candidate_selection(
    *,
    reservoir_rows: Sequence[Mapping[str, Any]],
    teacher_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically classify all rows and select the first 24 survivors."""

    frozen_rows = _freeze_fresh_candidate_reservoir_rows(reservoir_rows)
    replay = _freeze_fresh_candidate_replay_exclusions(teacher_payload)
    classifications: list[dict[str, Any]] = []
    survivors: list[int] = []
    reason_counts = {
        "bidirectional_recovery_text_collision": 0,
        "stepwise_prompt_window_collision": 0,
        "stepwise_rollout_window_collision": 0,
        "prior_survivor_meaningful_prefix_collision": 0,
        "replay_disjoint_survivor": 0,
    }
    survivor_prefixes: set[tuple[int, ...]] = set()
    for row in frozen_rows:
        decoded = row["decoded_prompt"].strip()
        if any(
            decoded in training_text.strip() or training_text.strip() in decoded
            for _source, _text_index, _event_index, training_text in (
                replay.indexed_training_texts
            )
        ):
            reason = "bidirectional_recovery_text_collision"
        else:
            candidate_windows = frozenset(
                _token_windows(
                    row["token_ids"], MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW
                )
            )
            if candidate_windows & replay.prompt_windows:
                reason = "stepwise_prompt_window_collision"
            elif candidate_windows & replay.rollout_windows:
                reason = "stepwise_rollout_window_collision"
            elif split_tools._meaningful_prefixes(
                tuple(row["token_ids"])
            ) & survivor_prefixes:
                reason = "prior_survivor_meaningful_prefix_collision"
            else:
                reason = "replay_disjoint_survivor"
                survivors.append(row["reservoir_index"])
                survivor_prefixes.update(
                    split_tools._meaningful_prefixes(tuple(row["token_ids"]))
                )
        reason_counts[reason] += 1
        classifications.append(
            {"reservoir_index": row["reservoir_index"], "classification": reason}
        )
    if len(survivors) < FRESH_CANDIDATE_COUNT:
        raise RecoveryInputPreparationError(
            "Replay exclusion left fewer than 24 fresh candidate prompts; "
            f"reservoir={len(frozen_rows)}, survivors={len(survivors)}, "
            f"required={FRESH_CANDIDATE_COUNT}, reason_counts="
            f"{json.dumps(reason_counts, sort_keys=True)}."
        )
    selected_indices = survivors[:FRESH_CANDIDATE_COUNT]
    selected_prompt_ids = [frozen_rows[index]["prompt_id"] for index in selected_indices]
    body = {
        "schema": FRESH_REPLAY_EXCLUSION_DERIVATION_SCHEMA,
        "status": "authenticated-first-24-replay-disjoint-survivors",
        "algorithm": "full-reservoir-fixed-reason-order-first-24-v1",
        "reason_precedence": [
            "bidirectional_recovery_text_collision",
            "stepwise_prompt_window_collision",
            "stepwise_rollout_window_collision",
            "prior_survivor_meaningful_prefix_collision",
            "replay_disjoint_survivor",
        ],
        "minimum_common_token_window": MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW,
        "teacher_token_replay_payload_sha256": _require_sha256(
            teacher_payload.get("payload_sha256"),
            "teacher replay payload SHA",
        ),
        "training_text_manifest_sha256": canonical_sha256(
            list(replay.training_text_rows)
        ),
        "stepwise_replay_bank_sha256": replay.stepwise_sha256,
        "reservoir_rows_sha256": canonical_sha256(frozen_rows),
        "reservoir_prompt_count": len(frozen_rows),
        "classification_rows": classifications,
        "classification_rows_sha256": canonical_sha256(classifications),
        "reason_counts": reason_counts,
        "survivor_count": len(survivors),
        "survivor_margin": len(survivors) - FRESH_CANDIDATE_COUNT,
        "selected_reservoir_indices": selected_indices,
        "selected_prompt_ids": selected_prompt_ids,
        "selected_count": FRESH_CANDIDATE_COUNT,
        "full_reservoir_classified": True,
        "selection_is_first_survivors_in_frozen_order": True,
        "teacher_replay_used_for_exclusion_only": True,
        "candidate_behavior_or_child_arm_scores_accessed": False,
        "parent_forward_or_score_executed_by_derivation": False,
        "tokenizer_called_by_derivation": False,
    }
    return {**body, "payload_sha256": canonical_sha256(body)}


def _fresh_candidate_replay_disjointness_audit(
    *,
    decoded_prompts: Sequence[str],
    prompt_token_ids: Sequence[Sequence[int]],
    teacher_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate that fresh evaluation prompts never occur in recovery input."""

    if not isinstance(teacher_payload, Mapping):
        raise RecoveryInputPreparationError("Teacher replay payload must be an object.")
    if (
        isinstance(decoded_prompts, (str, bytes, Mapping))
        or not isinstance(decoded_prompts, Sequence)
        or isinstance(prompt_token_ids, (str, bytes, Mapping))
        or not isinstance(prompt_token_ids, Sequence)
        or len(decoded_prompts) != FRESH_CANDIDATE_COUNT
        or len(prompt_token_ids) != FRESH_CANDIDATE_COUNT
    ):
        raise RecoveryInputPreparationError(
            "Replay disjointness requires exactly 24 aligned fresh prompts."
        )
    frozen_decoded: list[str] = []
    frozen_candidate_ids: list[tuple[int, ...]] = []
    for index, (decoded, token_row) in enumerate(
        zip(decoded_prompts, prompt_token_ids, strict=True)
    ):
        if not isinstance(decoded, str) or not decoded.strip():
            raise RecoveryInputPreparationError(
                f"Fresh decoded prompt {index} must be nonblank text."
            )
        if isinstance(token_row, (str, bytes, Mapping)) or not isinstance(
            token_row, Sequence
        ):
            raise RecoveryInputPreparationError(
                f"Fresh prompt token row {index} must be a token sequence."
            )
        frozen_row: list[int] = []
        for token_index, token_id in enumerate(token_row):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise RecoveryInputPreparationError(
                    "Fresh prompt token IDs must be non-negative integers; "
                    f"row {index}, offset {token_index} is invalid."
                )
            frozen_row.append(int(token_id))
        if not (
            MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW
            <= len(frozen_row)
            <= split_tools.MAX_PROMPT_TOKENS
        ):
            raise RecoveryInputPreparationError(
                "Fresh prompt token rows must remain within the authenticated "
                "8..128-token candidate range."
            )
        frozen_decoded.append(decoded)
        frozen_candidate_ids.append(tuple(frozen_row))
    if len(set(frozen_decoded)) != len(frozen_decoded):
        raise RecoveryInputPreparationError("Fresh decoded prompts must be unique.")
    tokenizer_fingerprints = teacher_payload.get("tokenizer_fingerprints")
    candidate_tokenizer_fingerprint = (
        tokenizer_fingerprints.get("candidate")
        if isinstance(tokenizer_fingerprints, Mapping)
        else None
    )
    tokenizer_length = (
        candidate_tokenizer_fingerprint.get("tokenizer_length")
        if isinstance(candidate_tokenizer_fingerprint, Mapping)
        else None
    )
    if (
        isinstance(tokenizer_length, bool)
        or not isinstance(tokenizer_length, int)
        or tokenizer_length <= 0
        or any(
            token_id >= tokenizer_length
            for token_row in frozen_candidate_ids
            for token_id in token_row
        )
    ):
        raise RecoveryInputPreparationError(
            "Fresh prompt token IDs exceed the authenticated candidate vocabulary."
        )

    replay = _freeze_fresh_candidate_replay_exclusions(teacher_payload)

    for candidate_index, candidate in enumerate(frozen_decoded):
        for source, text_index, event_index, training_text in (
            replay.indexed_training_texts
        ):
            candidate_view = candidate.strip()
            training_view = training_text.strip()
            location = (
                f"{source}[{text_index}]"
                if event_index is None
                else f"{source}[{event_index}][{text_index}]"
            )
            if candidate_view in training_view:
                raise RecoveryInputPreparationError(
                    "Fresh decoded candidate prompt "
                    f"{candidate_index} is contained in recovery text {location}."
                )
            if training_view in candidate_view:
                raise RecoveryInputPreparationError(
                    f"Recovery text {location} is contained in fresh decoded "
                    f"candidate prompt {candidate_index}."
                )

    candidate_windows: set[tuple[int, ...]] = set()
    candidate_window_total = 0
    for token_row in frozen_candidate_ids:
        windows = _token_windows(token_row, MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW)
        candidate_window_total += len(windows)
        candidate_windows.update(windows)

    for event_index, event in enumerate(replay.stepwise):
        for row_index, row in enumerate(event.rows):
            prompt_windows = _token_windows(
                row.prompt_input_ids, MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW
            )
            rollout_windows = _token_windows(
                row.rollout_input_ids, MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW
            )
            if candidate_windows.intersection(prompt_windows):
                raise RecoveryInputPreparationError(
                    "A fresh candidate shares an eight-token window with stepwise "
                    f"prompt replay event {event_index}, row {row_index}."
                )
            if candidate_windows.intersection(rollout_windows):
                raise RecoveryInputPreparationError(
                    "A fresh candidate shares an eight-token window with stepwise "
                    f"rollout replay event {event_index}, row {row_index}."
                )

    audit_body = {
        "schema": RECOVERY_REPLAY_DISJOINTNESS_AUDIT_SCHEMA,
        "status": "passed",
        "minimum_common_token_window": MINIMUM_REPLAY_DISJOINT_TOKEN_WINDOW,
        "candidate_prompt_count": len(frozen_decoded),
        "candidate_decoded_prompt_manifest_sha256": canonical_sha256(frozen_decoded),
        "candidate_token_id_manifest_sha256": canonical_sha256(
            [list(row) for row in frozen_candidate_ids]
        ),
        "teacher_token_replay_payload_sha256": _require_sha256(
            teacher_payload.get("payload_sha256"),
            "teacher replay payload SHA",
        ),
        "training_text_manifest_sha256": canonical_sha256(
            list(replay.training_text_rows)
        ),
        "training_text_counts": {
            "calibration_texts": len(replay.calibration),
            "precomputed_teacher_generated_texts": len(replay.teacher_generated),
            "on_policy_events": len(replay.on_policy),
            "on_policy_texts": sum(len(event) for event in replay.on_policy),
            "total": len(replay.indexed_training_texts),
        },
        "text_scan": {
            "candidate_in_training_overlap_count": 0,
            "training_in_candidate_overlap_count": 0,
            "all_training_text_fields_scanned": True,
            "bidirectional_substring_scan": True,
            "outer_whitespace_stripped_before_scan": True,
        },
        "stepwise_scan": {
            "stepwise_replay_bank_sha256": replay.stepwise_sha256,
            "event_count": len(replay.stepwise),
            "row_count": replay.stepwise_row_count,
            "candidate_window_total_count": candidate_window_total,
            "candidate_window_unique_count": len(candidate_windows),
            "prompt_window_total_count": replay.prompt_window_total,
            "rollout_window_total_count": replay.rollout_window_total,
            "candidate_prompt_window_overlap_count": 0,
            "candidate_rollout_window_overlap_count": 0,
            "all_candidate_offsets_scanned": True,
            "all_stepwise_prompt_offsets_scanned": True,
            "all_stepwise_rollout_offsets_scanned": True,
        },
        "model_or_tokenizer_accessed": False,
        "sealed_prompts_decoded": False,
        "sealed_prompts_supplied_to_model_or_trainer": False,
    }
    return {**audit_body, "payload_sha256": canonical_sha256(audit_body)}


def _assert_fresh_prompts_absent_from_teacher_replay(
    selection: FreshRecoverySelection,
    teacher_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Backward-compatible selection wrapper for the full replay audit."""

    return _fresh_candidate_replay_disjointness_audit(
        decoded_prompts=selection.decoded_prompts,
        prompt_token_ids=[prompt.token_ids for prompt in selection.prompts],
        teacher_payload=teacher_payload,
    )


def prepare_recovery_inputs(
    *,
    output_dir: Path | str,
    source_model: Path | str,
    source_tokenizer: Path | str,
    candidate_tokenizer: Path | str,
    calibration_file: Path | str,
    behavior_eval_suite: Path | str | None,
    expected_parent_artifact_tree_sha256: str,
    behavior_mode: str = BEHAVIOR_MODE_EXPLICIT,
    v4_preregistration: Path | str | None = None,
    v4_preregistration_sha256: str | None = None,
    teacher_device: str = "cuda:0",
    recovery_device: str = "cuda:0",
    max_calibration_texts: int = 640,
    teacher_replay_cache_dir: Path | str | None = None,
    teacher_replay_cache_mode: str = "off",
    teacher_replay_cache_import_request_sha256: str | None = None,
    teacher_replay_cache_import_envelope_sha256: str | None = None,
    component_loader: Callable[..., TeacherComponents] | None = load_teacher_components,
    tokenizer_loader: Callable[..., TokenizerComponents] = _CANONICAL_TOKENIZER_LOADER,
    source_model_loader: Callable[..., Any] = _CANONICAL_SOURCE_MODEL_LOADER,
    preregistration_authenticator: Callable[[Path, str], Any] = (
        _authenticate_v4_preregistration
    ),
    teacher_replay_builder: Callable[..., Mapping[str, Any]] = (
        _CANONICAL_TEACHER_REPLAY_BUILDER
    ),
    source_model_fingerprinter: Callable[[Any], Mapping[str, Any]] = (
        _CANONICAL_SOURCE_MODEL_FINGERPRINTER
    ),
    _test_only_cache_injection_authority: object | None = None,
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
    if behavior_mode not in {BEHAVIOR_MODE_FRESH, BEHAVIOR_MODE_EXPLICIT}:
        raise RecoveryInputPreparationError(
            "behavior_mode must be fresh-token-domain or explicit-json-diagnostic."
        )
    fresh_mode = behavior_mode == BEHAVIOR_MODE_FRESH
    if (
        _test_only_cache_injection_authority is not None
        and _test_only_cache_injection_authority
        is not _CACHE_TEST_INJECTION_AUTHORITY
    ):
        raise RecoveryInputPreparationError(
            "Invalid internal cache test-injection authority."
        )
    allow_test_cache_injections = (
        _test_only_cache_injection_authority is _CACHE_TEST_INJECTION_AUTHORITY
    )
    import_argument_count = sum(
        value is not None
        for value in (
            teacher_replay_cache_import_request_sha256,
            teacher_replay_cache_import_envelope_sha256,
        )
    )
    if import_argument_count not in {0, 2}:
        raise RecoveryInputPreparationError(
            "Teacher replay cache import request/envelope SHAs must be supplied together."
        )
    cache_import_authorized = import_argument_count == 2
    if cache_import_authorized:
        assert teacher_replay_cache_import_request_sha256 is not None
        assert teacher_replay_cache_import_envelope_sha256 is not None
        _require_sha256(
            teacher_replay_cache_import_request_sha256,
            "teacher_replay_cache_import_request_sha256",
        )
        _require_sha256(
            teacher_replay_cache_import_envelope_sha256,
            "teacher_replay_cache_import_envelope_sha256",
        )
    if teacher_replay_cache_mode not in TEACHER_REPLAY_CACHE_MODES:
        raise RecoveryInputPreparationError(
            "teacher_replay_cache_mode must be auto, require, refresh, or off."
        )
    if teacher_replay_cache_mode == "off":
        if teacher_replay_cache_dir is not None or cache_import_authorized:
            raise RecoveryInputPreparationError(
                "Teacher cache directory/import authorization requires an enabled "
                "cache mode."
            )
    elif teacher_replay_cache_dir is None:
        raise RecoveryInputPreparationError(
            "Enabled teacher replay caching requires teacher_replay_cache_dir."
        )
    elif not allow_test_cache_injections and (
        tokenizer_loader is not _CANONICAL_TOKENIZER_LOADER
        or source_model_loader is not _CANONICAL_SOURCE_MODEL_LOADER
        or teacher_replay_builder is not _CANONICAL_TEACHER_REPLAY_BUILDER
        or source_model_fingerprinter is not _CANONICAL_SOURCE_MODEL_FINGERPRINTER
    ):
        raise RecoveryInputPreparationError(
            "Cache-enabled production runs require the canonical tokenizer loader, "
            "source-model loader, teacher replay builder, and source-model "
            "fingerprinter."
        )
    if cache_import_authorized and teacher_replay_cache_mode not in {
        "auto",
        "require",
    }:
        raise RecoveryInputPreparationError(
            "Explicit compatible cache import is allowed only with auto or require."
        )
    if not fresh_mode and teacher_replay_cache_mode != "off":
        raise RecoveryInputPreparationError(
            "Teacher replay caching is available only in fresh causal mode."
        )
    behavior_path = (
        None
        if behavior_eval_suite is None
        else _existing_real_file(behavior_eval_suite, "behavior_eval_suite")
    )
    v4_path = (
        None
        if v4_preregistration is None
        else _existing_real_file(v4_preregistration, "v4_preregistration")
    )
    if fresh_mode:
        if behavior_path is not None:
            raise RecoveryInputPreparationError(
                "Fresh causal mode generates its suite and rejects behavior_eval_suite."
            )
        if v4_path is None or v4_preregistration_sha256 is None:
            raise RecoveryInputPreparationError(
                "Fresh causal mode requires the authenticated V4 preregistration."
            )
        if (
            source_tokenizer_root != source_root
            or candidate_tokenizer_root != source_root
        ):
            raise RecoveryInputPreparationError(
                "Fresh causal mode requires both tokenizer roots to be the exact "
                "contract parent identity path."
            )
        if max_calibration_texts != 640:
            raise RecoveryInputPreparationError(
                "Fresh causal mode requires max_calibration_texts exactly 640."
            )
        if component_loader not in (None, load_teacher_components):
            raise RecoveryInputPreparationError(
                "Fresh causal mode forbids the legacy combined component loader."
            )
    elif behavior_path is None:
        raise RecoveryInputPreparationError(
            "Diagnostic explicit mode requires behavior_eval_suite."
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
    behavior_binding: dict[str, Any] | None = None
    if not fresh_mode:
        assert behavior_path is not None
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
    if fresh_mode and len(ordered_calibration) != 640:
        raise RecoveryInputPreparationError(
            "Pinned recovery corpus did not yield exactly 640 frozen calibration rows."
        )

    candidate_reservoir: FreshRecoveryCandidateReservoir | None = None
    selection: FreshRecoverySelection | None = None
    cache_request: dict[str, Any] | None = None
    cached_envelope: dict[str, Any] | None = None
    cache_envelope: dict[str, Any] | None = None
    compatible_cache_import: CompatibleTeacherCacheImport | None = None
    cache_import_receipt: dict[str, Any] | None = None
    teacher_replay_origin = "generated"
    cache_result = "disabled"
    if fresh_mode:
        tokenizers = tokenizer_loader(
            source_tokenizer_path=source_tokenizer_root,
            candidate_tokenizer_path=candidate_tokenizer_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        source_tokenizer_fingerprint = (
            hf_teacher_token_replay.tokenizer_functional_fingerprint(
                tokenizers.source_tokenizer
            )
        )
        candidate_tokenizer_fingerprint = (
            hf_teacher_token_replay.tokenizer_functional_fingerprint(
                tokenizers.candidate_tokenizer
            )
        )
        if source_tokenizer_fingerprint != candidate_tokenizer_fingerprint:
            raise RecoveryInputPreparationError(
                "Fresh causal mode identity tokenizers differ functionally."
            )
        assert v4_path is not None
        assert v4_preregistration_sha256 is not None
        candidate_reservoir = _authenticate_fresh_recovery_candidate_reservoir(
            v4_preregistration=v4_path,
            v4_preregistration_sha256=v4_preregistration_sha256,
            corpus_path=calibration_path,
            calibration_texts=ordered_calibration,
            tokenizer=tokenizers.candidate_tokenizer,
            preregistration_authenticator=preregistration_authenticator,
        )
        # This is the first cache access.  The full candidate reservoir above is
        # authenticated and immutable before any cached parent-derived content
        # can be read.  The cache request API cannot receive that reservoir.
        if teacher_replay_cache_mode != "off":
            assert teacher_replay_cache_dir is not None
            cache_request = _teacher_replay_cache_request(
                actual_parent_tree=actual_parent_tree,
                expected_parent_tree_sha256=expected_tree_sha,
                calibration_path=calibration_path,
                ordered_calibration=ordered_calibration,
                source_tokenizer_fingerprint=source_tokenizer_fingerprint,
                candidate_tokenizer_fingerprint=candidate_tokenizer_fingerprint,
                torch=tokenizers.torch,
                transformers=tokenizers.transformers,
                teacher_device=teacher_device,
            )
            try:
                cached_envelope = hf_teacher_replay_cache.read_entry(
                    teacher_replay_cache_dir,
                    cache_request,
                )
            except hf_teacher_replay_cache.TeacherReplayCacheError as error:
                raise RecoveryInputPreparationError(
                    "Existing teacher replay cache entry failed authentication."
                ) from error
            if cached_envelope is None and cache_import_authorized:
                assert teacher_replay_cache_import_request_sha256 is not None
                assert teacher_replay_cache_import_envelope_sha256 is not None
                compatible_cache_import = (
                    _read_exact_compatible_teacher_cache_import(
                        cache_dir=teacher_replay_cache_dir,
                        current_request=cache_request,
                        source_request_sha256=(
                            teacher_replay_cache_import_request_sha256
                        ),
                        source_envelope_sha256=(
                            teacher_replay_cache_import_envelope_sha256
                        ),
                    )
                )
                teacher_replay_origin = "cache-import"
            if cached_envelope is None and teacher_replay_cache_mode == "require":
                if compatible_cache_import is None:
                    raise RecoveryInputPreparationError(
                        "Required teacher replay cache entry is missing."
                    )
            if (
                cached_envelope is not None
                and teacher_replay_cache_mode in {"auto", "require"}
            ):
                teacher_replay_origin = "cache-hit"
                cache_result = "hit"
        source_model_object = source_model_loader(
            torch=tokenizers.torch,
            transformers=tokenizers.transformers,
            source_model_path=source_root,
            device=teacher_device,
            local_files_only=True,
            trust_remote_code=False,
            source_torch_dtype="float16",
        )
        components = TeacherComponents(
            torch=tokenizers.torch,
            source_model=source_model_object,
            source_tokenizer=tokenizers.source_tokenizer,
            candidate_tokenizer=tokenizers.candidate_tokenizer,
        )
    else:
        if component_loader is None:
            raise RecoveryInputPreparationError(
                "Diagnostic explicit mode requires a component loader."
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
        live_source_fingerprint: dict[str, Any] | None = None
        if teacher_replay_origin in {"cache-hit", "cache-import"}:
            assert cache_request is not None
            try:
                live_source_fingerprint = _json_value(
                    source_model_fingerprinter(components.source_model),
                    name="live_source_model_fingerprint",
                )
                source_envelope = (
                    cached_envelope
                    if teacher_replay_origin == "cache-hit"
                    else (
                        compatible_cache_import.envelope
                        if compatible_cache_import is not None
                        else None
                    )
                )
                source_request = (
                    cache_request
                    if teacher_replay_origin == "cache-hit"
                    else (
                        compatible_cache_import.request
                        if compatible_cache_import is not None
                        else None
                    )
                )
                assert source_envelope is not None
                assert source_request is not None
                verified_source_envelope = hf_teacher_replay_cache.verify_envelope(
                    source_envelope,
                    expected_request=source_request,
                    live_source_model_fingerprint=live_source_fingerprint,
                )
            except (
                TypeError,
                ValueError,
                hf_teacher_replay_cache.TeacherReplayCacheError,
            ) as error:
                raise RecoveryInputPreparationError(
                    "Teacher replay cache source-model binding failed authentication."
                ) from error
        if teacher_replay_origin in {"cache-hit", "cache-import"}:
            teacher_payload = _json_value(
                verified_source_envelope["teacher_token_replay"],
                name="cached_teacher_token_replay",
            )
            if teacher_replay_origin == "cache-hit":
                cache_envelope = verified_source_envelope
            else:
                assert teacher_replay_cache_dir is not None
                assert cache_request is not None
                assert live_source_fingerprint is not None
                assert compatible_cache_import is not None
                try:
                    cache_envelope, cache_result = (
                        hf_teacher_replay_cache.publish_entry(
                            cache_dir=teacher_replay_cache_dir,
                            request=cache_request,
                            teacher_payload=teacher_payload,
                            live_source_model_fingerprint=live_source_fingerprint,
                        )
                    )
                except hf_teacher_replay_cache.TeacherReplayCacheError as error:
                    raise RecoveryInputPreparationError(
                        "Compatible teacher replay cache republication failed closed."
                    ) from error
                cache_import_receipt = _finalize_teacher_cache_import_receipt(
                    compatible_import=compatible_cache_import,
                    current_envelope=cache_envelope,
                    publication_result=cache_result,
                )
        else:
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
            if fresh_mode and teacher_replay_cache_mode != "off":
                assert teacher_replay_cache_dir is not None
                assert cache_request is not None
                try:
                    live_source_fingerprint = _json_value(
                        source_model_fingerprinter(components.source_model),
                        name="live_source_model_fingerprint",
                    )
                except (TypeError, ValueError) as error:
                    raise RecoveryInputPreparationError(
                        "Teacher replay cache source-model fingerprinting failed."
                    ) from error
                assert live_source_fingerprint is not None
                try:
                    cache_envelope, cache_result = (
                        hf_teacher_replay_cache.publish_entry(
                            cache_dir=teacher_replay_cache_dir,
                            request=cache_request,
                            teacher_payload=teacher_payload,
                            live_source_model_fingerprint=live_source_fingerprint,
                        )
                    )
                except hf_teacher_replay_cache.TeacherReplayCacheError as error:
                    raise RecoveryInputPreparationError(
                        "Generated teacher replay cache publication failed closed."
                    ) from error
        if fresh_mode:
            assert candidate_reservoir is not None
            selection = _select_fresh_recovery_candidate(
                reservoir=candidate_reservoir,
                teacher_payload=teacher_payload,
            )
            behavior_binding = _build_fresh_behavior_binding(
                torch=components.torch,
                source_model=components.source_model,
                tokenizer=components.candidate_tokenizer,
                selection=selection,
                teacher_payload=teacher_payload,
                device=teacher_device,
                teacher_replay_origin=teacher_replay_origin,
            )
            verify_behavior_suite_binding(behavior_binding)
    finally:
        torch_module = components.torch
        del components
        gc.collect()
        empty_cache = getattr(getattr(torch_module, "cuda", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    if behavior_binding is None:
        raise RecoveryInputPreparationError(
            "Behavior suite binding was not frozen before publication."
        )
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

    if teacher_replay_cache_mode == "off":
        cache_request_sha: str | None = None
        cache_envelope_sha: str | None = None
        cache_relative_path: str | None = None
    else:
        assert cache_request is not None
        assert cache_envelope is not None
        cache_request_sha = cache_request["request_sha256"]
        cache_envelope_sha = cache_envelope["envelope_sha256"]
        cache_relative_path = (
            f"{cache_request_sha}/{hf_teacher_replay_cache.CACHE_FILENAME}"
        )
    teacher_replay_acquisition = {
        "schema": TEACHER_REPLAY_ACQUISITION_SCHEMA,
        "cache_mode": teacher_replay_cache_mode,
        "origin": teacher_replay_origin,
        "cache_result": cache_result,
        "cache_request_sha256": cache_request_sha,
        "cache_envelope_sha256": cache_envelope_sha,
        "cache_entry_relative_path": cache_relative_path,
        "cache_verified_before_final_selection": (
            teacher_replay_cache_mode != "off"
        ),
        "cache_import": cache_import_receipt,
    }
    parent_model_usage = {
        "schema": PARENT_MODEL_USAGE_SCHEMA,
        "model_load_count": 1,
        "load_purpose": (
            "fresh-targets"
            if teacher_replay_origin in {"cache-hit", "cache-import"}
            else (
                "teacher-replay-and-fresh-targets"
                if fresh_mode
                else "teacher-replay"
            )
        ),
        "teacher_replay_generated_this_run": (
            teacher_replay_origin == "generated"
        ),
        "teacher_replay_reused_from_cache": teacher_replay_origin
        in {"cache-hit", "cache-import"},
        "fresh_target_generation_performed": fresh_mode,
    }

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
            teacher_replay_acquisition=teacher_replay_acquisition,
            parent_model_usage=parent_model_usage,
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
    parser.add_argument(
        "--behavior-mode",
        choices=(BEHAVIOR_MODE_FRESH, BEHAVIOR_MODE_EXPLICIT),
        default=BEHAVIOR_MODE_FRESH,
    )
    parser.add_argument("--behavior-eval-suite", type=Path)
    parser.add_argument("--v4-preregistration", type=Path)
    parser.add_argument("--v4-preregistration-sha256")
    parser.add_argument(
        "--expected-parent-artifact-tree-sha256",
        required=True,
    )
    parser.add_argument("--teacher-device", default="cuda:0")
    parser.add_argument("--recovery-device", required=True)
    parser.add_argument("--max-calibration-texts", type=int, default=640)
    parser.add_argument("--teacher-replay-cache-dir", type=Path)
    parser.add_argument(
        "--teacher-replay-cache-mode",
        choices=tuple(sorted(TEACHER_REPLAY_CACHE_MODES)),
        default="off",
    )
    parser.add_argument("--teacher-replay-cache-import-request-sha256")
    parser.add_argument("--teacher-replay-cache-import-envelope-sha256")
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
        behavior_mode=args.behavior_mode,
        v4_preregistration=args.v4_preregistration,
        v4_preregistration_sha256=args.v4_preregistration_sha256,
        expected_parent_artifact_tree_sha256=(
            args.expected_parent_artifact_tree_sha256
        ),
        teacher_device=args.teacher_device,
        recovery_device=args.recovery_device,
        max_calibration_texts=args.max_calibration_texts,
        teacher_replay_cache_dir=args.teacher_replay_cache_dir,
        teacher_replay_cache_mode=args.teacher_replay_cache_mode,
        teacher_replay_cache_import_request_sha256=(
            args.teacher_replay_cache_import_request_sha256
        ),
        teacher_replay_cache_import_envelope_sha256=(
            args.teacher_replay_cache_import_envelope_sha256
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
