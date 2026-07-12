#!/usr/bin/env python3
"""Run the first exact target-native Qwen compiler stage.

This is the production transaction runner for the already-materialized
``4,022,468,096 -> 1,410,567,168`` stage.  It is deliberately not a model
builder and never creates another random target.  Before importing model
weights it authenticates a preregistration file, the pinned parent and blank
target artifacts, and three disjoint behavior splits.  The live transaction
contains all 120 target Q/K/V/O physical-group sites and all 30 target
Gate/Up/Down layers at rank eight.

Only causal output actions and free-running rollouts can select a checkpoint.
Weight, activation, hidden-state, residual, attention, or projection matching
is never consulted by the promotion gate.  A rejection writes only a compact
factor pack and JSON report, then proves exact rollback.  An acceptance bakes
the factors into native dense weights, saves the child once, freshly reloads
those bytes, and certifies the exact count, state-key topology, and absence of
runtime sidecars.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from neural_forge.hf_arch_layout import build_decoder_layout
from neural_forge.hf_blank_checkpoint import _state_dict_sha256
from neural_forge.hf_canonical_graph import (
    CanonicalChildSpec,
    CanonicalParentLogits,
    CanonicalTokenContract,
)
from neural_forge.hf_canonical_kernel import (
    CanonicalParentTrace,
    CanonicalRowIdentity,
    build_proportional_depth_cover,
)
from neural_forge.hf_compiled_child_cache import (
    CompiledChildCacheIntegrityError,
    CompilerIdentity,
    load_compiled_child_cache,
    publish_compiled_child_cache,
    verify_compiled_child_cache,
)
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import DenseGQASite
from neural_forge.hf_factorized_vocab import exact_parameter_report
from neural_forge.hf_parent_compiler import (
    HFParentCompilerResult,
    compile_hf_parents_to_blank_child,
    fingerprint_hf_child_structure,
)
from neural_forge.hf_progressive_target_bridge import (
    ExternalBehaviorGateReceipt,
    QWEN3_1P4B_EXACT_PARAMETERS,
    QWEN3_4B_EXACT_PARAMETERS,
    certify_native_dense_gqa_target,
    default_qwen4b_progressive_target_plan,
    install_exact_target_compiler_transaction,
    install_parent_compiler_result_target_native_transaction,
)


RUN_SCHEMA = "neural-forge-qwen-exact-target-native-stage-v1"
PREREGISTRATION_SCHEMA = "neural-forge-qwen-exact-target-native-preregistration-v1"
SPLIT_SCHEMA = "neural-forge-target-native-behavior-split-v1"
FACTOR_PACK_SCHEMA = "neural-forge-rejected-target-native-factor-pack-v1"
FACTOR_PACK_V2_SCHEMA = "neural-forge-rejected-target-native-factor-pack-v2"
TRAINER_FACTOR_PACK_SCHEMA = "neural-forge-target-native-factor-pack-v1"
TRAINER_FACTOR_PACK_V2_SCHEMA = "neural-forge-target-native-factor-pack-v2"
CANONICAL_TRAINER_MODULE = "neural_forge.hf_target_native_behavior_distillation"
OUTPUT_JACOBIAN_BASIS_MODULE = "neural_forge.hf_output_jacobian_orthogonal_basis"
OUTPUT_JACOBIAN_SKETCH_MODULE = "neural_forge.hf_output_jacobian_sketch"
QKVO_OUTPUT_JACOBIAN_AUDIT_MODULE = "neural_forge.hf_qkvo_output_jacobian_audit"
OUTPUT_JACOBIAN_AUDIT_SCHEMA = "neural-forge-qkvo-output-jacobian-audit-v1"
GUIDED_BASIS_PACK_SCHEMA = "neural-forge-guided-qkvo-basis-pack-v1"
GUIDED_QKVO_RIGHT_BASIS_SCHEMA = "neural-forge-guided-qkvo-right-basis-v1"

TARGET_LAYER_COUNT = 30
TARGET_PHYSICAL_GROUPS = 4
QKVO_SITE_COUNT = TARGET_LAYER_COUNT * TARGET_PHYSICAL_GROUPS
QKVO_RANK = 8
TARGET_HIDDEN_SIZE = 1_536
TARGET_QUERY_GROUP_WIDTH = 512
TARGET_HEAD_DIM = 128
SWIGLU_LAYER_COUNT = TARGET_LAYER_COUNT
SWIGLU_RANK = 8
TARGET_INTERMEDIATE_SIZE = 5_120
TARGET_NATIVE_FACTOR_COUNT = 14
TARGET_NATIVE_FACTOR_PARAMETERS = 11_919_360
LEXICAL_FACTOR_COUNT = 4
LEXICAL_RANK = 32
LEXICAL_VOCAB_SIZE = 151_936
LEXICAL_HIDDEN_SIZE = 1_536
LEXICAL_FACTOR_PARAMETERS = (
    2 * LEXICAL_RANK * (LEXICAL_VOCAB_SIZE + LEXICAL_HIDDEN_SIZE)
)
LEXICAL_TARGET_NATIVE_FACTOR_COUNT = TARGET_NATIVE_FACTOR_COUNT + LEXICAL_FACTOR_COUNT
LEXICAL_TARGET_NATIVE_FACTOR_PARAMETERS = (
    TARGET_NATIVE_FACTOR_PARAMETERS + LEXICAL_FACTOR_PARAMETERS
)
CANONICAL_ATTENTION_IMPLEMENTATION = "eager"

DEFAULT_BLANK_TARGET = Path(
    "/mnt/ai_data/neural-forge/canonical_blank_targets/"
    "qwen3_stage1_blank_1410567168_fp16_seed0"
)
DEFAULT_BLANK_WEIGHT_NAME = "model.safetensors"
DEFAULT_BLANK_REPORT_NAME = "neural_forge_qwen_target_report.json"
DEFAULT_BLANK_CONFIG_SHA256 = (
    "8ff1783dc4c4985db254d53e4b4ce1e5fdf9a541cff711253892244ac84a61d7"
)
DEFAULT_BLANK_TOKENIZER_SHA256 = (
    "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
)
DEFAULT_BLANK_TOKENIZER_CONFIG_SHA256 = (
    "5f007d04324afa93876e2a484190e868136c69dfe0e1a29a1063eae583df9d32"
)
DEFAULT_BLANK_WEIGHT_SHA256 = (
    "4147674b29180d75e8f5bb518b5fc69349d475bb5c8bd49a8f5e7e0ecf364a45"
)
DEFAULT_BLANK_MATERIALIZATION_REPORT_SHA256 = (
    "e3c2e19f2698d28541c10b0a3086bedab428a477ccb693fcc310802f8b692db5"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BANNED_SELECTION_WORDS = (
    "weight_similarity",
    "weight_matching",
    "activation_matching",
    "hidden_state_matching",
    "attention_matching",
    "residual_matching",
    "projection_matching",
)


class ExactTargetNativeStageError(RuntimeError):
    """A provenance, transaction, behavior, or publication gate failed."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ExactTargetNativeStageError(
            "Provenance is not finite canonical JSON."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ExactTargetNativeStageError(f"{name} must be a SHA-256 string.")
    normalized = value.removeprefix("sha256:").lower()
    if not _SHA256.fullmatch(normalized):
        raise ExactTargetNativeStageError(f"{name} must be a SHA-256 string.")
    return normalized


def _strict_json(path: Path) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExactTargetNativeStageError(
                    f"Duplicate JSON key {key!r} in {path}."
                )
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExactTargetNativeStageError(
                    f"Non-finite JSON constant {value!r} in {path}."
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExactTargetNativeStageError(f"Cannot read strict JSON {path}.") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ExactTargetNativeStageError(f"Cannot hash artifact {path}.") from error
    return digest.hexdigest()


def _artifact_tree(root: Path) -> dict[str, Any]:
    """Content-address every regular file, following HF snapshot symlinks."""

    if not root.is_dir():
        raise ExactTargetNativeStageError(f"Artifact root is not a directory: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_dir():
            continue
        if not path.is_file():
            raise ExactTargetNativeStageError(
                f"Artifact contains a non-file entry: {path}"
            )
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    if not records:
        raise ExactTargetNativeStageError(f"Artifact directory is empty: {root}")
    return {
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "tree_sha256": _canonical_sha256(records),
        "files": records,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if resolved.exists():
            raise ExactTargetNativeStageError(
                f"Refusing to overwrite existing report {resolved}."
            )
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise ExactTargetNativeStageError(
            f"Refusing to overwrite factor pack {resolved}."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(resolved)


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    name: str
    path: Path
    exact_unique_parameters: int
    required_files: tuple[tuple[str, str], ...]
    tree_sha256: str | None


@dataclass(frozen=True, slots=True)
class SplitBinding:
    name: Literal["fit", "candidate", "sealed"]
    path: Path
    sha256: str
    expected_count: int
    payload: Mapping[str, Any]
    row_ids: tuple[str, ...]
    prompt_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContinuationFactorBinding:
    factor_pack_path: Path
    factor_pack_sha256: str
    source_run_report_path: Path
    source_run_report_sha256: str
    factor_state_sha256: str
    compiler_report_sha256: str
    compiled_state_sha256: str
    source_preregistration_sha256: str
    source_evaluation_contract_sha256: str
    factor_payload: Mapping[str, Any]
    factor_pack_schema: str
    trainer_factor_pack_schema: str
    lexical_rank: int | None

    def as_report(self) -> dict[str, Any]:
        return {
            "factor_pack_path": str(self.factor_pack_path),
            "factor_pack_sha256": self.factor_pack_sha256,
            "source_run_report_path": str(self.source_run_report_path),
            "source_run_report_sha256": self.source_run_report_sha256,
            "factor_state_sha256": self.factor_state_sha256,
            "compiler_report_sha256": self.compiler_report_sha256,
            "compiled_state_sha256": self.compiled_state_sha256,
            "source_preregistration_sha256": self.source_preregistration_sha256,
            "source_evaluation_contract_sha256": (
                self.source_evaluation_contract_sha256
            ),
            "factor_pack_schema": self.factor_pack_schema,
            "trainer_factor_pack_schema": self.trainer_factor_pack_schema,
            "lexical_rank": self.lexical_rank,
            "factor_only": True,
            "authenticated_before_model_access": True,
        }


@dataclass(frozen=True, slots=True)
class CompilerCacheBinding:
    root: Path
    require_hit: bool = False
    expected_compiler_identity_sha256: str | None = None
    expected_receipt_sha256: str | None = None
    expected_compiler_report_sha256: str | None = None
    expected_state_dict_sha256: str | None = None
    expected_artifact_tree_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LexicalResidualBinding:
    rank: int
    vocab_size: int
    hidden_size: int
    untied_required: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class OutputJacobianAuditBinding:
    schema: str
    source_exposed_candidate_as_fit: bool
    basis_split_role: str
    prompt_count: int
    horizon: int
    lanes: tuple[str, str]
    rank: int
    sketch_width: int
    min_normal_fraction: float
    min_guided_over_random_ratio: float
    max_bytes: int

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["lanes"] = list(self.lanes)
        return payload


@dataclass(frozen=True, slots=True)
class OutputJacobianAuditExecution:
    result: Any
    restore_mode: Literal["full"]
    restored_factor_state_sha256: str
    restored_factor_schema: str


@dataclass(frozen=True, slots=True)
class CompiledBase:
    child_model: nn.Module
    report: Mapping[str, Any]
    compiler_report_sha256: str
    compiled_state_sha256: str
    cache_evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BehaviorThresholds:
    minimum_action_accuracy: float
    minimum_exact_rollout_fraction: float
    minimum_token_match_fraction: float
    minimum_lcp_fraction: float
    minimum_action_count: int
    minimum_rollout_count: int
    maximum_repeat_fraction_excess: float = 1.0
    maximum_adjacent_repeat_fraction_excess: float = 1.0
    maximum_longest_run_excess: int = 2_147_483_647

    @classmethod
    def from_mapping(cls, value: Any, name: str) -> BehaviorThresholds:
        if not isinstance(value, Mapping):
            raise ExactTargetNativeStageError(f"{name} thresholds must be an object.")
        expected = {field.name for field in dataclasses.fields(cls)}
        legacy = {
            "minimum_action_accuracy",
            "minimum_exact_rollout_fraction",
            "minimum_token_match_fraction",
            "minimum_lcp_fraction",
            "minimum_action_count",
            "minimum_rollout_count",
        }
        if frozenset(value) not in {frozenset(expected), frozenset(legacy)}:
            raise ExactTargetNativeStageError(
                f"{name} thresholds must contain either {sorted(legacy)} or "
                f"{sorted(expected)}."
            )
        fractions: dict[str, float] = {}
        for key in (
            "minimum_action_accuracy",
            "minimum_exact_rollout_fraction",
            "minimum_token_match_fraction",
            "minimum_lcp_fraction",
            "maximum_repeat_fraction_excess",
            "maximum_adjacent_repeat_fraction_excess",
        ):
            raw = value.get(key, 1.0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ExactTargetNativeStageError(f"{name}.{key} must be numeric.")
            number = float(raw)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ExactTargetNativeStageError(f"{name}.{key} must be in [0, 1].")
            fractions[key] = number
        counts: dict[str, int] = {}
        for key in (
            "minimum_action_count",
            "minimum_rollout_count",
            "maximum_longest_run_excess",
        ):
            raw = value.get(key, 2_147_483_647)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, int)
                or (raw < 0 if key == "maximum_longest_run_excess" else raw <= 0)
            ):
                raise ExactTargetNativeStageError(
                    f"{name}.{key} has an invalid integer bound."
                )
            counts[key] = raw
        return cls(**fractions, **counts)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class AuthenticatedPreRegistration:
    path: Path
    file_sha256: str
    parent: ArtifactBinding
    target: ArtifactBinding
    splits: Mapping[str, SplitBinding]
    compiler_split: SplitBinding
    thresholds: Mapping[str, BehaviorThresholds]
    compiler: Mapping[str, Any]
    training: Mapping[str, Any]
    seed: int
    evaluation_contract_sha256: str
    artifact_authentication: Mapping[str, Any]
    implementation_authentication: Mapping[str, Any]
    continuation: ContinuationFactorBinding | None
    compiler_cache: CompilerCacheBinding | None
    lexical_residual: LexicalResidualBinding | None
    output_jacobian_audit: OutputJacobianAuditBinding | None

    def preflight_report(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "status": "preflight-passed",
            "preregistration": str(self.path),
            "preregistration_sha256": self.file_sha256,
            "parent": {
                "path": str(self.parent.path),
                "exact_unique_parameters": self.parent.exact_unique_parameters,
            },
            "target": {
                "path": str(self.target.path),
                "exact_unique_parameters": self.target.exact_unique_parameters,
                "reuses_existing_staging_artifact": (
                    self.target.path == DEFAULT_BLANK_TARGET
                ),
            },
            "splits": {
                name: {
                    "path": str(split.path),
                    "sha256": split.sha256,
                    "row_count": len(split.row_ids),
                }
                for name, split in self.splits.items()
            },
            "compiler_split": {
                "path": str(self.compiler_split.path),
                "sha256": self.compiler_split.sha256,
                "row_count": len(self.compiler_split.row_ids),
                "reuses_behavior_fit": self.compiler_split is self.splits["fit"],
            },
            "thresholds": {
                name: threshold.as_dict() for name, threshold in self.thresholds.items()
            },
            "canonical_parent_evidence": _compiler_partition_report(self),
            "qkvo": {
                "site_count": QKVO_SITE_COUNT,
                "rank": QKVO_RANK,
                "all_target_layers": True,
                "all_physical_groups": True,
            },
            "gate_up_down": {
                "layer_count": SWIGLU_LAYER_COUNT,
                "rank": SWIGLU_RANK,
                "role_complete": True,
            },
            "lexical_residual": (
                None
                if self.lexical_residual is None
                else self.lexical_residual.as_dict()
            ),
            **(
                {}
                if self.output_jacobian_audit is None
                else {"output_jacobian_audit": self.output_jacobian_audit.as_dict()}
            ),
            "selection_domain": "causal-output-actions-and-free-rollouts-only",
            "runtime_dtype": "float16",
            "runtime_attention_implementation": (CANONICAL_ATTENTION_IMPLEMENTATION),
            "internal_numeric_matching_can_promote": False,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "artifact_authentication": dict(self.artifact_authentication),
            "implementation_authentication": dict(self.implementation_authentication),
            "continuation": (
                None if self.continuation is None else self.continuation.as_report()
            ),
            "compiler_cache": (
                None
                if self.compiler_cache is None
                else {
                    "root": str(self.compiler_cache.root),
                    "require_hit": self.compiler_cache.require_hit,
                    "expected_compiler_identity_sha256": (
                        self.compiler_cache.expected_compiler_identity_sha256
                    ),
                    "expected_receipt_sha256": (
                        self.compiler_cache.expected_receipt_sha256
                    ),
                    "expected_compiler_report_sha256": (
                        self.compiler_cache.expected_compiler_report_sha256
                    ),
                    "expected_state_dict_sha256": (
                        self.compiler_cache.expected_state_dict_sha256
                    ),
                    "expected_artifact_tree_sha256": (
                        self.compiler_cache.expected_artifact_tree_sha256
                    ),
                    "content_addressed": True,
                    "copied_into_run_output": False,
                }
            ),
            "models_loaded": False,
            "gpu_work_launched": False,
        }


def _resolve_path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExactTargetNativeStageError(f"{name} must be a nonempty path string.")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _artifact_binding(
    value: Any,
    *,
    name: str,
    base: Path,
    expected_parameters: int,
    default_path: Path | None = None,
    default_required: Mapping[str, str] | None = None,
) -> ArtifactBinding:
    if not isinstance(value, Mapping):
        raise ExactTargetNativeStageError(f"{name} artifact must be an object.")
    raw_path = value.get("path")
    path = (
        default_path.resolve()
        if raw_path is None and default_path is not None
        else _resolve_path(base, raw_path, f"{name}.path")
    )
    raw_count = value.get("exact_unique_parameters")
    if raw_count != expected_parameters:
        raise ExactTargetNativeStageError(
            f"{name} exact_unique_parameters must be {expected_parameters}."
        )
    required_value = value.get("required_files", default_required or {})
    if not isinstance(required_value, Mapping):
        raise ExactTargetNativeStageError(f"{name}.required_files must be an object.")
    required = tuple(
        sorted(
            (
                str(relative),
                _require_sha256(digest, f"{name}.required_files[{relative!r}]"),
            )
            for relative, digest in required_value.items()
        )
    )
    tree = value.get("tree_sha256")
    tree_sha = None if tree is None else _require_sha256(tree, f"{name}.tree_sha256")
    if not required and tree_sha is None:
        raise ExactTargetNativeStageError(
            f"{name} must pin required_files or a complete tree_sha256."
        )
    return ArtifactBinding(
        name=name,
        path=path,
        exact_unique_parameters=expected_parameters,
        required_files=required,
        tree_sha256=tree_sha,
    )


def _authenticate_artifact(binding: ArtifactBinding) -> dict[str, Any]:
    root = binding.path
    if not root.is_dir():
        raise ExactTargetNativeStageError(
            f"Authenticated {binding.name} directory does not exist: {root}"
        )
    required: list[dict[str, Any]] = []
    for relative, expected in binding.required_files:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            # Hugging Face snapshots intentionally point into the sibling blob
            # store.  The relative name and resolved content hash remain bound.
            if not (root / relative).is_symlink():
                raise ExactTargetNativeStageError(
                    f"{binding.name} required file escapes its artifact root."
                )
        if not candidate.is_file():
            raise ExactTargetNativeStageError(
                f"Missing {binding.name} required file {relative}."
            )
        actual = _file_sha256(candidate)
        if actual != expected:
            raise ExactTargetNativeStageError(
                f"{binding.name} required file hash mismatch for {relative}."
            )
        required.append(
            {
                "path": relative,
                "sha256": actual,
                "size_bytes": int(candidate.stat().st_size),
            }
        )
    tree = None
    if binding.tree_sha256 is not None:
        tree = _artifact_tree(root)
        if tree["tree_sha256"] != binding.tree_sha256:
            raise ExactTargetNativeStageError(
                f"{binding.name} artifact tree hash mismatch."
            )
    return {
        "path": str(root),
        "required_files": required,
        "tree": None
        if tree is None
        else {key: tree[key] for key in ("file_count", "total_bytes", "tree_sha256")},
        "authenticated_before_model_access": True,
    }


def _split_identity(item: Any, split_name: str, index: int) -> tuple[str, str]:
    if not isinstance(item, Mapping):
        raise ExactTargetNativeStageError(
            f"{split_name}.items[{index}] must be an object."
        )
    token_values = item.get("token_ids", item.get("input_ids"))
    if (
        isinstance(token_values, (str, bytes))
        or not isinstance(token_values, Sequence)
        or not token_values
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_values
        )
    ):
        raise ExactTargetNativeStageError(
            f"{split_name}.prompts[{index}] must contain nonnegative token_ids."
        )
    prompt_sha = _canonical_sha256(list(token_values))
    row_id = item.get("id", item.get("prompt_id", f"prompt-{prompt_sha}"))
    if not isinstance(row_id, str) or not row_id:
        raise ExactTargetNativeStageError(
            f"{split_name}.prompts[{index}] ID must be nonempty."
        )
    supplied_hash = item.get("token_ids_sha256")
    if supplied_hash is not None:
        # The trainer uses its own typed token-sequence digest.  At this outer
        # boundary we only require a valid immutable digest; the trainer then
        # recomputes and checks its exact convention before optimization.
        _require_sha256(
            supplied_hash, f"{split_name}.prompts[{index}].token_ids_sha256"
        )
    return row_id, prompt_sha


def _load_split(
    value: Any,
    *,
    name: Literal["fit", "candidate", "sealed"],
    base: Path,
) -> SplitBinding:
    if not isinstance(value, Mapping):
        raise ExactTargetNativeStageError(f"{name} split binding must be an object.")
    path = _resolve_path(base, value.get("path"), f"splits.{name}.path")
    expected_sha = _require_sha256(value.get("sha256"), f"splits.{name}.sha256")
    if _file_sha256(path) != expected_sha:
        raise ExactTargetNativeStageError(f"{name} split artifact hash mismatch.")
    payload = _strict_json(path)
    if not isinstance(payload, Mapping):
        raise ExactTargetNativeStageError(f"{name} split must contain a JSON object.")
    if payload.get("schema") != SPLIT_SCHEMA or payload.get("split") != name:
        raise ExactTargetNativeStageError(f"{name} split schema/name mismatch.")
    if name == "sealed" and payload.get("sealed_before_model_access") is not True:
        raise ExactTargetNativeStageError(
            "The sealed split must attest sealed_before_model_access=true."
        )
    items = payload.get("prompts")
    if not isinstance(items, list) or not items:
        raise ExactTargetNativeStageError(f"{name} split prompts must be nonempty.")
    expected_count = value.get("expected_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or expected_count != len(items)
    ):
        raise ExactTargetNativeStageError(
            f"{name}.expected_count must equal the authenticated item count."
        )
    identities = tuple(
        _split_identity(item, name, index) for index, item in enumerate(items)
    )
    row_ids = tuple(value[0] for value in identities)
    hashes = tuple(value[1] for value in identities)
    if len(set(row_ids)) != len(row_ids) or len(set(hashes)) != len(hashes):
        raise ExactTargetNativeStageError(
            f"{name} contains duplicate row IDs or prompt identities."
        )
    return SplitBinding(
        name=name,
        path=path,
        sha256=expected_sha,
        expected_count=expected_count,
        payload=payload,
        row_ids=row_ids,
        prompt_sha256s=hashes,
    )


def _factor_tensor_sha256(value: Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(detached.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _output_jacobian_tensor_sha256(value: Tensor) -> str:
    """Match the source-pinned Jacobian sketch tensor digest convention."""

    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    flat = tensor.reshape(-1).contiguous().view(torch.uint8)
    chunk_bytes = 1 << 20
    for start in range(0, int(flat.numel()), chunk_bytes):
        digest.update(memoryview(flat[start : start + chunk_bytes].numpy()))
    return digest.hexdigest()


def _expected_continuation_factor_shapes(
    *, lexical_rank: int | None
) -> dict[str, tuple[int, ...]]:
    """Return the pinned exact-stage-1 factor tensor topology."""

    shapes = {
        "qkvo.q_left": (QKVO_SITE_COUNT, TARGET_QUERY_GROUP_WIDTH, QKVO_RANK),
        "qkvo.q_right": (QKVO_SITE_COUNT, QKVO_RANK, TARGET_HIDDEN_SIZE),
        "qkvo.k_left": (QKVO_SITE_COUNT, TARGET_HEAD_DIM, QKVO_RANK),
        "qkvo.k_right": (QKVO_SITE_COUNT, QKVO_RANK, TARGET_HIDDEN_SIZE),
        "qkvo.v_left": (QKVO_SITE_COUNT, TARGET_HEAD_DIM, QKVO_RANK),
        "qkvo.v_right": (QKVO_SITE_COUNT, QKVO_RANK, TARGET_HIDDEN_SIZE),
        "qkvo.o_left": (QKVO_SITE_COUNT, TARGET_HIDDEN_SIZE, QKVO_RANK),
        "qkvo.o_right": (QKVO_SITE_COUNT, QKVO_RANK, TARGET_QUERY_GROUP_WIDTH),
        "gud.gate_left": (
            SWIGLU_LAYER_COUNT,
            TARGET_INTERMEDIATE_SIZE,
            SWIGLU_RANK,
        ),
        "gud.gate_right": (
            SWIGLU_LAYER_COUNT,
            SWIGLU_RANK,
            TARGET_HIDDEN_SIZE,
        ),
        "gud.up_left": (
            SWIGLU_LAYER_COUNT,
            TARGET_INTERMEDIATE_SIZE,
            SWIGLU_RANK,
        ),
        "gud.up_right": (
            SWIGLU_LAYER_COUNT,
            SWIGLU_RANK,
            TARGET_HIDDEN_SIZE,
        ),
        "gud.down_left": (
            SWIGLU_LAYER_COUNT,
            TARGET_HIDDEN_SIZE,
            SWIGLU_RANK,
        ),
        "gud.down_right": (
            SWIGLU_LAYER_COUNT,
            SWIGLU_RANK,
            TARGET_INTERMEDIATE_SIZE,
        ),
    }
    if lexical_rank is None:
        return shapes
    if lexical_rank != LEXICAL_RANK:
        raise ExactTargetNativeStageError(
            "Continuation lexical factors do not use the pinned stage-1 rank."
        )
    shapes.update(
        {
            "lexical.embedding_left": (LEXICAL_VOCAB_SIZE, LEXICAL_RANK),
            "lexical.embedding_right": (LEXICAL_RANK, LEXICAL_HIDDEN_SIZE),
            "lexical.lm_head_left": (LEXICAL_VOCAB_SIZE, LEXICAL_RANK),
            "lexical.lm_head_right": (LEXICAL_RANK, LEXICAL_HIDDEN_SIZE),
        }
    )
    return shapes


def _authenticate_continuation_factor_payload(
    value: Any,
    *,
    lexical_rank: int | None,
) -> tuple[Mapping[str, Any], str, Mapping[str, str], str, str, int | None]:
    if not isinstance(value, Mapping):
        raise ExactTargetNativeStageError("Continuation factor file must be a mapping.")
    base_envelope = {
        "schema",
        "reason",
        "preregistration_sha256",
        "evaluation_contract_sha256",
        "qkvo_site_count",
        "qkvo_rank",
        "swiglu_layer_count",
        "swiglu_rank",
        "factor_pack",
        "contains_model_checkpoint",
        "contains_parent_parameters",
        "contains_activations_or_hidden_states",
    }
    envelope_schema = value.get("schema")
    envelope_lexical_rank: int | None = None
    expected_envelope = set(base_envelope)
    if envelope_schema == FACTOR_PACK_V2_SCHEMA:
        expected_envelope.add("lexical_rank")
        envelope_lexical_rank = value.get("lexical_rank")
        if (
            isinstance(envelope_lexical_rank, bool)
            or not isinstance(envelope_lexical_rank, int)
            or envelope_lexical_rank <= 0
        ):
            raise ExactTargetNativeStageError(
                "Continuation factor envelope lexical_rank is malformed."
            )
    elif envelope_schema != FACTOR_PACK_SCHEMA:
        raise ExactTargetNativeStageError(
            "Continuation factor envelope has an unsupported schema or fields."
        )
    if set(value) != expected_envelope:
        raise ExactTargetNativeStageError(
            "Continuation factor envelope has an unsupported schema or fields."
        )
    if (
        value.get("qkvo_site_count") != QKVO_SITE_COUNT
        or value.get("qkvo_rank") != QKVO_RANK
        or value.get("swiglu_layer_count") != SWIGLU_LAYER_COUNT
        or value.get("swiglu_rank") != SWIGLU_RANK
    ):
        raise ExactTargetNativeStageError(
            "Continuation factor envelope targets a different transaction topology."
        )
    if any(
        value.get(name) is not False
        for name in (
            "contains_model_checkpoint",
            "contains_parent_parameters",
            "contains_activations_or_hidden_states",
        )
    ):
        raise ExactTargetNativeStageError(
            "Continuation artifact must contain target-native factors only."
        )
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ExactTargetNativeStageError(
            "Continuation factor envelope reason must be nonempty."
        )
    envelope_provenance = {
        "reason": reason,
        "preregistration_sha256": _require_sha256(
            value.get("preregistration_sha256"),
            "continuation.factor_envelope.preregistration_sha256",
        ),
        "evaluation_contract_sha256": _require_sha256(
            value.get("evaluation_contract_sha256"),
            "continuation.factor_envelope.evaluation_contract_sha256",
        ),
    }
    payload = value.get("factor_pack")
    expected_payload = {
        "schema",
        "factor_count",
        "factors",
        "factor_state_sha256",
        "parent_state_included",
        "native_child_state_included",
        "optimizer_state_included",
        "factor_tensors",
    }
    trainer_schema = payload.get("schema") if isinstance(payload, Mapping) else None
    payload_lexical_rank: int | None = None
    if trainer_schema == TRAINER_FACTOR_PACK_V2_SCHEMA:
        expected_payload.add("lexical_rank")
        payload_lexical_rank = payload.get("lexical_rank")
    expected_factor_count = (
        LEXICAL_TARGET_NATIVE_FACTOR_COUNT
        if trainer_schema == TRAINER_FACTOR_PACK_V2_SCHEMA
        else TARGET_NATIVE_FACTOR_COUNT
    )
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_payload
        or trainer_schema
        not in {TRAINER_FACTOR_PACK_SCHEMA, TRAINER_FACTOR_PACK_V2_SCHEMA}
        or payload.get("factor_count") != expected_factor_count
        or any(
            payload.get(name) is not False
            for name in (
                "parent_state_included",
                "native_child_state_included",
                "optimizer_state_included",
            )
        )
    ):
        raise ExactTargetNativeStageError(
            "Continuation payload is not the exact factor-only trainer artifact."
        )
    rows = payload.get("factors")
    tensors = payload.get("factor_tensors")
    if not isinstance(rows, list) or not isinstance(tensors, Mapping):
        raise ExactTargetNativeStageError("Continuation factor manifest is malformed.")
    expected_name_set = {
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
    if trainer_schema == TRAINER_FACTOR_PACK_V2_SCHEMA:
        if (
            lexical_rank is None
            or payload_lexical_rank != lexical_rank
            or envelope_schema != FACTOR_PACK_V2_SCHEMA
            or envelope_lexical_rank != lexical_rank
        ):
            raise ExactTargetNativeStageError(
                "Continuation v2 factor pack does not match the active lexical rank."
            )
        expected_name_set.update(
            f"lexical.{role}_{side}"
            for role in ("embedding", "lm_head")
            for side in ("left", "right")
        )
    elif envelope_schema != FACTOR_PACK_SCHEMA:
        raise ExactTargetNativeStageError(
            "A legacy v1 trainer factor pack requires the legacy v1 envelope."
        )
    expected_shapes = _expected_continuation_factor_shapes(
        lexical_rank=(
            payload_lexical_rank
            if trainer_schema == TRAINER_FACTOR_PACK_V2_SCHEMA
            else None
        )
    )
    if set(expected_shapes) != expected_name_set:
        raise ExactTargetNativeStageError(
            "Continuation factor shape topology does not match its schema."
        )
    expected_names = tuple(sorted(expected_name_set))
    if len(rows) != len(expected_names) or set(tensors) != set(expected_names):
        raise ExactTargetNativeStageError(
            "Continuation factor names do not match the complete QKVO/GUD topology."
        )
    canonical_rows: list[dict[str, Any]] = []
    total_parameters = 0
    for index, (expected_name, row) in enumerate(
        zip(expected_names, rows, strict=True)
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "shape",
            "dtype",
            "tensor_sha256",
        }:
            raise ExactTargetNativeStageError(
                f"Continuation factor manifest row {index} is malformed."
            )
        if row.get("name") != expected_name:
            raise ExactTargetNativeStageError(
                "Continuation factor manifest must use canonical sorted names."
            )
        tensor = tensors[expected_name]
        if (
            not isinstance(tensor, Tensor)
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ExactTargetNativeStageError(
                f"Continuation factor {expected_name!r} is not finite CPU FP32."
            )
        detached = tensor.detach().contiguous()
        expected_shape = expected_shapes[expected_name]
        if tuple(detached.shape) != expected_shape or row.get("shape") != list(
            expected_shape
        ):
            raise ExactTargetNativeStageError(
                f"Continuation factor {expected_name!r} does not match its exact "
                "stage-1 shape."
            )
        digest = _factor_tensor_sha256(detached)
        expected_row = {
            "name": expected_name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "tensor_sha256": digest,
        }
        if dict(row) != expected_row:
            raise ExactTargetNativeStageError(
                f"Continuation factor {expected_name!r} failed manifest authentication."
            )
        total_parameters += int(detached.numel())
        canonical_rows.append(expected_row)
    expected_parameters = (
        LEXICAL_TARGET_NATIVE_FACTOR_PARAMETERS
        if trainer_schema == TRAINER_FACTOR_PACK_V2_SCHEMA
        else TARGET_NATIVE_FACTOR_PARAMETERS
    )
    if total_parameters != expected_parameters:
        raise ExactTargetNativeStageError(
            "Continuation factor parameter count does not match exact stage 1."
        )
    state_sha = _require_sha256(
        payload.get("factor_state_sha256"), "continuation.factor_state_sha256"
    )
    state_preimage: Any = canonical_rows
    if trainer_schema == TRAINER_FACTOR_PACK_V2_SCHEMA:
        state_preimage = {
            "schema": trainer_schema,
            "lexical_rank": lexical_rank,
            "factors": canonical_rows,
        }
    if _canonical_sha256(state_preimage) != state_sha:
        raise ExactTargetNativeStageError(
            "Continuation factor-state hash does not authenticate its manifest."
        )
    return (
        dict(payload),
        state_sha,
        envelope_provenance,
        str(envelope_schema),
        str(trainer_schema),
        payload_lexical_rank,
    )


def _authenticate_continuation(
    value: Any,
    *,
    base: Path,
    lexical_rank: int | None,
) -> ContinuationFactorBinding:
    if not isinstance(value, Mapping) or set(value) != {
        "factor_pack",
        "source_run_report",
        "expected_factor_state_sha256",
        "expected_compiler_report_sha256",
        "expected_compiled_state_sha256",
    }:
        raise ExactTargetNativeStageError(
            "continuation must exactly bind the factor pack, source report, and states."
        )

    def file_binding(raw: Any, name: str) -> tuple[Path, str]:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ExactTargetNativeStageError(f"continuation.{name} is malformed.")
        path = _resolve_path(base, raw.get("path"), f"continuation.{name}.path")
        expected_sha = _require_sha256(raw.get("sha256"), f"continuation.{name}.sha256")
        if not path.is_file() or _file_sha256(path) != expected_sha:
            raise ExactTargetNativeStageError(
                f"Authenticated continuation {name} hash mismatch."
            )
        return path, expected_sha

    factor_path, factor_sha = file_binding(value["factor_pack"], "factor_pack")
    report_path, report_sha = file_binding(
        value["source_run_report"], "source_run_report"
    )
    try:
        envelope = torch.load(factor_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ExactTargetNativeStageError(
            "Cannot safely load the authenticated continuation factor pack."
        ) from error
    (
        factor_payload,
        factor_state_sha,
        envelope_provenance,
        factor_pack_schema,
        trainer_factor_pack_schema,
        factor_lexical_rank,
    ) = _authenticate_continuation_factor_payload(
        envelope,
        lexical_rank=lexical_rank,
    )
    expected_factor_state_sha = _require_sha256(
        value.get("expected_factor_state_sha256"),
        "continuation.expected_factor_state_sha256",
    )
    if factor_state_sha != expected_factor_state_sha:
        raise ExactTargetNativeStageError(
            "Continuation factor state differs from the preregistered state."
        )
    report = _strict_json(report_path)
    if not isinstance(report, Mapping):
        raise ExactTargetNativeStageError("Continuation source report is malformed.")
    if (
        report.get("status") != "rejected"
        or report.get("rollback_exact") is not True
        or report.get("model_checkpoint_written") is not False
        or report.get("standalone_compaction_passed") is not False
    ):
        raise ExactTargetNativeStageError(
            "Continuation source must be a rejected run with exact rollback."
        )
    if any(
        envelope_provenance[name] != report.get(name)
        for name in (
            "reason",
            "preregistration_sha256",
            "evaluation_contract_sha256",
        )
    ):
        raise ExactTargetNativeStageError(
            "Continuation factor envelope provenance disagrees with its source report."
        )
    report_factor = report.get("factor_pack")
    if (
        not isinstance(report_factor, Mapping)
        or report_factor.get("sha256") != factor_sha
    ):
        raise ExactTargetNativeStageError(
            "Continuation source report does not authenticate the factor file."
        )
    trainer_evidence = report.get("trainer_evidence")
    if not isinstance(trainer_evidence, Mapping) or _canonical_sha256(
        trainer_evidence
    ) != _require_sha256(
        report.get("trainer_evidence_sha256"),
        "continuation.source_run_report.trainer_evidence_sha256",
    ):
        raise ExactTargetNativeStageError(
            "Continuation source trainer evidence hash does not verify."
        )
    manifest = (
        trainer_evidence.get("factor_pack_manifest")
        if isinstance(trainer_evidence, Mapping)
        else None
    )
    acquisition = report.get("factor_acquisition")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("factor_state_sha256") != factor_state_sha
        or not isinstance(acquisition, Mapping)
        or acquisition.get("selected_factor_state_sha256") != factor_state_sha
        or acquisition.get("target_native_factors_acquired_behavior") is not True
    ):
        raise ExactTargetNativeStageError(
            "Continuation source does not prove behavior-acquired selected factors."
        )
    compiler_report = report.get("compiler_report")
    if not isinstance(compiler_report, Mapping):
        raise ExactTargetNativeStageError(
            "Continuation source report omits structural compiler evidence."
        )
    compiler_sha = _canonical_sha256(compiler_report)
    expected_compiler_sha = _require_sha256(
        value.get("expected_compiler_report_sha256"),
        "continuation.expected_compiler_report_sha256",
    )
    if compiler_sha != expected_compiler_sha:
        raise ExactTargetNativeStageError(
            "Continuation compiler report differs from its preregistered digest."
        )
    compiled_state_sha = _require_sha256(
        report.get("rollback_state_sha256"),
        "continuation.source_run_report.rollback_state_sha256",
    )
    expected_compiled_sha = _require_sha256(
        value.get("expected_compiled_state_sha256"),
        "continuation.expected_compiled_state_sha256",
    )
    if compiled_state_sha != expected_compiled_sha:
        raise ExactTargetNativeStageError(
            "Continuation compiled-child state differs from its preregistered digest."
        )
    return ContinuationFactorBinding(
        factor_pack_path=factor_path,
        factor_pack_sha256=factor_sha,
        source_run_report_path=report_path,
        source_run_report_sha256=report_sha,
        factor_state_sha256=factor_state_sha,
        compiler_report_sha256=compiler_sha,
        compiled_state_sha256=compiled_state_sha,
        source_preregistration_sha256=_require_sha256(
            report.get("preregistration_sha256"),
            "continuation.source_run_report.preregistration_sha256",
        ),
        source_evaluation_contract_sha256=_require_sha256(
            report.get("evaluation_contract_sha256"),
            "continuation.source_run_report.evaluation_contract_sha256",
        ),
        factor_payload=factor_payload,
        factor_pack_schema=factor_pack_schema,
        trainer_factor_pack_schema=trainer_factor_pack_schema,
        lexical_rank=factor_lexical_rank,
    )


def _output_jacobian_audit_binding(
    value: Any,
) -> OutputJacobianAuditBinding:
    expected: dict[str, Any] = {
        "schema": OUTPUT_JACOBIAN_AUDIT_SCHEMA,
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
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ExactTargetNativeStageError(
            "output_jacobian_audit must contain exactly the authenticated "
            "audit-only contract fields."
        )
    if _canonical_json(dict(value)) != _canonical_json(expected):
        raise ExactTargetNativeStageError(
            "output_jacobian_audit must use the pinned source-candidate, "
            "two-lane rank-8 sketch contract."
        )
    return OutputJacobianAuditBinding(
        schema=OUTPUT_JACOBIAN_AUDIT_SCHEMA,
        source_exposed_candidate_as_fit=True,
        basis_split_role="candidate",
        prompt_count=24,
        horizon=8,
        lanes=("parent-greedy", "child-greedy"),
        rank=8,
        sketch_width=16,
        min_normal_fraction=0.20,
        min_guided_over_random_ratio=1.50,
        max_bytes=268_435_456,
    )


def authenticate_preregistration(
    path: Path | str,
    expected_sha256: str,
) -> AuthenticatedPreRegistration:
    """Authenticate the complete run contract before loading any model."""

    prereg_path = Path(path).expanduser().resolve()
    expected = _require_sha256(expected_sha256, "preregistration_sha256")
    actual = _file_sha256(prereg_path)
    if actual != expected:
        raise ExactTargetNativeStageError("Preregistration file hash mismatch.")
    raw = _strict_json(prereg_path)
    if not isinstance(raw, Mapping) or raw.get("schema") != PREREGISTRATION_SCHEMA:
        raise ExactTargetNativeStageError("Unsupported preregistration schema.")
    if raw.get("sealed_before_model_access") is not True:
        raise ExactTargetNativeStageError(
            "Preregistration must attest sealed_before_model_access=true."
        )
    lexical_value = raw.get("lexical_residual")
    lexical_residual: LexicalResidualBinding | None = None
    if lexical_value is not None:
        if not isinstance(lexical_value, Mapping) or set(lexical_value) != {
            "rank",
            "vocab_size",
            "hidden_size",
            "untied_required",
        }:
            raise ExactTargetNativeStageError(
                "lexical_residual must exactly bind rank, vocabulary, hidden size, "
                "and untied ownership."
            )
        if dict(lexical_value) != {
            "rank": LEXICAL_RANK,
            "vocab_size": LEXICAL_VOCAB_SIZE,
            "hidden_size": LEXICAL_HIDDEN_SIZE,
            "untied_required": True,
        }:
            raise ExactTargetNativeStageError(
                "The lexical residual contract must use the pinned exact-stage-1 "
                "rank, vocabulary, hidden size, and untied boundary."
            )
        lexical_residual = LexicalResidualBinding(**dict(lexical_value))
    output_jacobian_audit = (
        None
        if "output_jacobian_audit" not in raw
        else _output_jacobian_audit_binding(raw["output_jacobian_audit"])
    )
    if output_jacobian_audit is not None and lexical_residual is None:
        raise ExactTargetNativeStageError(
            "The output-Jacobian audit requires the exact lexical residual transaction."
        )
    implementation = raw.get("implementation")
    required_implementation = {
        "runner_sha256",
        "trainer_module",
        "trainer_sha256",
    }
    if raw.get("compiler_cache") is not None:
        required_implementation.add("compiler_cache_sha256")
    if lexical_residual is not None:
        required_implementation.update(
            {
                "lexical_adapter_sha256",
                "progressive_bridge_sha256",
                "qkvo_adapter_sha256",
                "swiglu_adapter_sha256",
            }
        )
    if output_jacobian_audit is not None:
        required_implementation.update(
            {
                "output_jacobian_basis_sha256",
                "output_jacobian_sketch_sha256",
                "qkvo_output_jacobian_audit_sha256",
            }
        )
    if not isinstance(implementation, Mapping) or set(implementation) != (
        required_implementation
    ):
        raise ExactTargetNativeStageError(
            "implementation does not contain the exact required runtime source pins."
        )
    if implementation.get("trainer_module") != CANONICAL_TRAINER_MODULE:
        raise ExactTargetNativeStageError(
            "Only the canonical output/rollout trainer module is permitted."
        )
    runner_sha = _require_sha256(
        implementation.get("runner_sha256"), "implementation.runner_sha256"
    )
    trainer_sha = _require_sha256(
        implementation.get("trainer_sha256"), "implementation.trainer_sha256"
    )
    runner_path = Path(__file__).resolve()
    trainer_spec = importlib.util.find_spec(CANONICAL_TRAINER_MODULE)
    if trainer_spec is None or trainer_spec.origin is None:
        raise ExactTargetNativeStageError(
            "Canonical output/rollout trainer module is unavailable."
        )
    trainer_path = Path(trainer_spec.origin).resolve()
    actual_runner_sha = _file_sha256(runner_path)
    actual_trainer_sha = _file_sha256(trainer_path)
    if runner_sha != actual_runner_sha or trainer_sha != actual_trainer_sha:
        raise ExactTargetNativeStageError(
            "Preregistered runner/trainer source hash does not match the live code."
        )
    cache_path: Path | None = None
    cache_sha: str | None = None
    if raw.get("compiler_cache") is not None:
        cache_spec = importlib.util.find_spec("neural_forge.hf_compiled_child_cache")
        if cache_spec is None or cache_spec.origin is None:
            raise ExactTargetNativeStageError(
                "Compiled-child cache implementation is unavailable."
            )
        cache_path = Path(cache_spec.origin).resolve()
        cache_sha = _file_sha256(cache_path)
        if cache_sha != _require_sha256(
            implementation.get("compiler_cache_sha256"),
            "implementation.compiler_cache_sha256",
        ):
            raise ExactTargetNativeStageError(
                "Preregistered compiler-cache source hash does not match live code."
            )
    implementation_authentication = {
        "runner": {"path": str(runner_path), "sha256": actual_runner_sha},
        "trainer": {
            "module": CANONICAL_TRAINER_MODULE,
            "path": str(trainer_path),
            "sha256": actual_trainer_sha,
        },
        "authenticated_before_model_access": True,
    }
    if cache_path is not None and cache_sha is not None:
        implementation_authentication["compiler_cache"] = {
            "path": str(cache_path),
            "sha256": cache_sha,
        }
    if lexical_residual is not None:
        for field_name, module_name in (
            (
                "lexical_adapter_sha256",
                "neural_forge.hf_dense_lexical_residual_adapter",
            ),
            ("progressive_bridge_sha256", "neural_forge.hf_progressive_target_bridge"),
            (
                "qkvo_adapter_sha256",
                "neural_forge.hf_dense_gqa_qkvo_residual_adapter",
            ),
            (
                "swiglu_adapter_sha256",
                "neural_forge.hf_dense_swiglu_projection_residual_adapter",
            ),
        ):
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise ExactTargetNativeStageError(
                    f"Lexical continuation implementation {module_name!r} is unavailable."
                )
            source_path = Path(spec.origin).resolve()
            source_sha = _file_sha256(source_path)
            if source_sha != _require_sha256(
                implementation.get(field_name), f"implementation.{field_name}"
            ):
                raise ExactTargetNativeStageError(
                    "Preregistered lexical transaction source hash does not match "
                    "the live code."
                )
            implementation_authentication[field_name.removesuffix("_sha256")] = {
                "path": str(source_path),
                "sha256": source_sha,
            }
    if output_jacobian_audit is not None:
        for field_name, module_name in (
            (
                "output_jacobian_basis_sha256",
                OUTPUT_JACOBIAN_BASIS_MODULE,
            ),
            (
                "output_jacobian_sketch_sha256",
                OUTPUT_JACOBIAN_SKETCH_MODULE,
            ),
            (
                "qkvo_output_jacobian_audit_sha256",
                QKVO_OUTPUT_JACOBIAN_AUDIT_MODULE,
            ),
        ):
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise ExactTargetNativeStageError(
                    f"Output-Jacobian audit implementation {module_name!r} is "
                    "unavailable."
                )
            source_path = Path(spec.origin).resolve()
            source_sha = _file_sha256(source_path)
            if source_sha != _require_sha256(
                implementation.get(field_name), f"implementation.{field_name}"
            ):
                raise ExactTargetNativeStageError(
                    "Preregistered output-Jacobian source hash does not match "
                    "the live code."
                )
            implementation_authentication[field_name.removesuffix("_sha256")] = {
                "module": module_name,
                "path": str(source_path),
                "sha256": source_sha,
            }
    base = prereg_path.parent
    parent = _artifact_binding(
        raw.get("parent"),
        name="parent",
        base=base,
        expected_parameters=QWEN3_4B_EXACT_PARAMETERS,
    )
    target = _artifact_binding(
        raw.get("blank_target"),
        name="blank_target",
        base=base,
        expected_parameters=QWEN3_1P4B_EXACT_PARAMETERS,
        default_path=DEFAULT_BLANK_TARGET,
        default_required={
            DEFAULT_BLANK_WEIGHT_NAME: DEFAULT_BLANK_WEIGHT_SHA256,
            DEFAULT_BLANK_REPORT_NAME: DEFAULT_BLANK_MATERIALIZATION_REPORT_SHA256,
            "config.json": DEFAULT_BLANK_CONFIG_SHA256,
            "tokenizer.json": DEFAULT_BLANK_TOKENIZER_SHA256,
            "tokenizer_config.json": DEFAULT_BLANK_TOKENIZER_CONFIG_SHA256,
        },
    )
    if lexical_residual is not None:
        config_path = target.path / "config.json"
        config = _strict_json(config_path)
        if not isinstance(config, Mapping) or (
            config.get("vocab_size") != lexical_residual.vocab_size
            or config.get("hidden_size") != lexical_residual.hidden_size
            or config.get("tie_word_embeddings") is not False
        ):
            raise ExactTargetNativeStageError(
                "The authenticated target config does not expose the pinned untied "
                "151936 x 1536 lexical boundary."
            )
    split_values = raw.get("splits")
    if not isinstance(split_values, Mapping) or set(split_values) != {
        "fit",
        "candidate",
        "sealed",
    }:
        raise ExactTargetNativeStageError(
            "splits must contain exactly fit, candidate, and sealed."
        )
    splits = {
        name: _load_split(split_values[name], name=name, base=base)
        for name in ("fit", "candidate", "sealed")
    }
    if (
        output_jacobian_audit is not None
        and len(splits[output_jacobian_audit.basis_split_role].row_ids)
        != output_jacobian_audit.prompt_count
    ):
        raise ExactTargetNativeStageError(
            "The output-Jacobian audit source candidate must contain exactly "
            f"{output_jacobian_audit.prompt_count} prompts."
        )
    for left_index, left in enumerate(("fit", "candidate", "sealed")):
        for right in ("fit", "candidate", "sealed")[left_index + 1 :]:
            if set(splits[left].row_ids) & set(splits[right].row_ids):
                raise ExactTargetNativeStageError(
                    f"Row IDs overlap between {left} and {right}."
                )
            if set(splits[left].prompt_sha256s) & set(splits[right].prompt_sha256s):
                raise ExactTargetNativeStageError(
                    f"Prompt identities overlap between {left} and {right}."
                )
    compiler_split_value = raw.get("compiler_split")
    compiler_split = (
        splits["fit"]
        if compiler_split_value is None
        else _load_split(compiler_split_value, name="fit", base=base)
    )
    for heldout_role in ("candidate", "sealed"):
        heldout = splits[heldout_role]
        if set(compiler_split.row_ids) & set(heldout.row_ids):
            raise ExactTargetNativeStageError(
                f"Compiler split row IDs overlap behavior {heldout_role}."
            )
        if set(compiler_split.prompt_sha256s) & set(heldout.prompt_sha256s):
            raise ExactTargetNativeStageError(
                f"Compiler split prompt identities overlap behavior {heldout_role}."
            )
    threshold_values = raw.get("thresholds")
    if not isinstance(threshold_values, Mapping) or set(threshold_values) != {
        "fit",
        "candidate",
        "sealed",
    }:
        raise ExactTargetNativeStageError(
            "thresholds must contain exactly fit, candidate, and sealed."
        )
    thresholds = {
        name: BehaviorThresholds.from_mapping(
            threshold_values[name], f"thresholds.{name}"
        )
        for name in ("fit", "candidate", "sealed")
    }
    compiler = raw.get("compiler")
    training = raw.get("training")
    if not isinstance(compiler, Mapping) or not isinstance(training, Mapping):
        raise ExactTargetNativeStageError(
            "compiler and training preregistrations must be objects."
        )
    if training.get("selection_policy") != "rollout-first-acquisition":
        raise ExactTargetNativeStageError(
            "This blank-target stage requires the preregistered "
            "rollout-first-acquisition policy; strict step-zero non-regression is "
            "reserved for already-functional folds."
        )
    serialized_training = json.dumps(training, sort_keys=True).lower()
    if any(word in serialized_training for word in _BANNED_SELECTION_WORDS):
        raise ExactTargetNativeStageError(
            "Training contract contains a forbidden internal-matching objective."
        )
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExactTargetNativeStageError("seed must be a nonnegative integer.")
    authentication = {
        "parent": _authenticate_artifact(parent),
        "blank_target": _authenticate_artifact(target),
        "split_sha256s": {name: split.sha256 for name, split in splits.items()},
        "compiler_split_sha256": compiler_split.sha256,
    }
    continuation_value = raw.get("continuation")
    continuation = (
        None
        if continuation_value is None
        else _authenticate_continuation(
            continuation_value,
            base=base,
            lexical_rank=(None if lexical_residual is None else lexical_residual.rank),
        )
    )
    cache_value = raw.get("compiler_cache")
    compiler_cache: CompilerCacheBinding | None = None
    if cache_value is not None:
        legacy_cache_fields = {"root"}
        bound_cache_fields = {
            "root",
            "require_hit",
            "expected_compiler_identity_sha256",
            "expected_receipt_sha256",
            "expected_compiler_report_sha256",
            "expected_state_dict_sha256",
            "expected_artifact_tree_sha256",
        }
        if not isinstance(cache_value, Mapping) or set(cache_value) not in {
            frozenset(legacy_cache_fields),
            frozenset(bound_cache_fields),
        }:
            raise ExactTargetNativeStageError(
                "compiler_cache must use either the legacy root-only contract or "
                "the complete authenticated-hit contract."
            )
        cache_root = _resolve_path(base, cache_value.get("root"), "compiler_cache.root")
        if cache_root.exists() and (cache_root.is_symlink() or not cache_root.is_dir()):
            raise ExactTargetNativeStageError(
                "compiler_cache.root must be an ordinary directory when present."
            )
        require_hit = cache_value.get("require_hit", False)
        if not isinstance(require_hit, bool):
            raise ExactTargetNativeStageError(
                "compiler_cache.require_hit must be bool."
            )
        expected_cache_hashes = {
            name: (
                None
                if name not in cache_value
                else _require_sha256(cache_value.get(name), f"compiler_cache.{name}")
            )
            for name in (
                "expected_compiler_identity_sha256",
                "expected_receipt_sha256",
                "expected_compiler_report_sha256",
                "expected_state_dict_sha256",
                "expected_artifact_tree_sha256",
            )
        }
        if require_hit and any(
            value is None for value in expected_cache_hashes.values()
        ):
            raise ExactTargetNativeStageError(
                "A required compiler-cache hit must bind every immutable cache hash."
            )
        compiler_cache = CompilerCacheBinding(
            root=cache_root,
            require_hit=require_hit,
            **expected_cache_hashes,
        )
    if continuation is not None and compiler_cache is None:
        raise ExactTargetNativeStageError(
            "Authenticated continuation runs must use the shared structural cache."
        )
    if output_jacobian_audit is not None and (
        continuation is None
        or continuation.trainer_factor_pack_schema != TRAINER_FACTOR_PACK_V2_SCHEMA
        or continuation.factor_pack_schema != FACTOR_PACK_V2_SCHEMA
        or continuation.lexical_rank != LEXICAL_RANK
    ):
        raise ExactTargetNativeStageError(
            "The output-Jacobian audit requires an authenticated full lexical v2 "
            "continuation payload."
        )
    if output_jacobian_audit is not None and (
        compiler_cache is None or compiler_cache.require_hit is not True
    ):
        raise ExactTargetNativeStageError(
            "The output-Jacobian audit requires a fully authenticated compiler-cache "
            "hit."
        )
    evaluation_contract = {
        "split_sha256s": authentication["split_sha256s"],
        "compiler_split_sha256": compiler_split.sha256,
        "thresholds": {
            name: threshold.as_dict() for name, threshold in thresholds.items()
        },
        "selection_domain": "causal-output-actions-and-free-rollouts-only",
        "qkvo_sites": QKVO_SITE_COUNT,
        "qkvo_rank": QKVO_RANK,
        "swiglu_layers": SWIGLU_LAYER_COUNT,
        "swiglu_rank": SWIGLU_RANK,
        "lexical_residual": (
            None if lexical_residual is None else lexical_residual.as_dict()
        ),
        **(
            {}
            if output_jacobian_audit is None
            else {"output_jacobian_audit": output_jacobian_audit.as_dict()}
        ),
        "implementation": implementation_authentication,
        "runtime_dtype": "float16",
        "runtime_attention_implementation": CANONICAL_ATTENTION_IMPLEMENTATION,
        "canonical_parent_evidence": {
            "source_split": "authenticated-compiler-split-deterministic-partition",
            "candidate_split_used": False,
            "sealed_split_used": False,
            "compiler_options": dict(compiler),
        },
    }
    return AuthenticatedPreRegistration(
        path=prereg_path,
        file_sha256=actual,
        parent=parent,
        target=target,
        splits=splits,
        compiler_split=compiler_split,
        thresholds=thresholds,
        compiler=dict(compiler),
        training=dict(training),
        seed=seed,
        evaluation_contract_sha256=_canonical_sha256(evaluation_contract),
        artifact_authentication=authentication,
        implementation_authentication=implementation_authentication,
        continuation=continuation,
        compiler_cache=compiler_cache,
        lexical_residual=lexical_residual,
        output_jacobian_audit=output_jacobian_audit,
    )


def _model_input_device(model: nn.Module) -> torch.device:
    embedding = model.get_input_embeddings()
    return embedding.weight.device


def _move_inputs(inputs: Mapping[str, Tensor], model: nn.Module) -> dict[str, Tensor]:
    device = _model_input_device(model)
    return {name: value.to(device=device) for name, value in inputs.items()}


def _prompt_token_ids(split: SplitBinding) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for item in split.payload["prompts"]:
        values = item.get("token_ids", item.get("input_ids"))
        result.append(tuple(int(value) for value in values))
    return tuple(result)


def _compiler_fit_partitions(
    contract: AuthenticatedPreRegistration,
) -> tuple[SplitBinding, SplitBinding]:
    """Partition only fit evidence; candidate and sealed remain untouched."""

    source = contract.compiler_split
    count = len(source.row_ids)
    validation_count = contract.compiler.get(
        "canonical_validation_count", max(1, count // 5)
    )
    if (
        isinstance(validation_count, bool)
        or not isinstance(validation_count, int)
        or validation_count <= 0
        or validation_count >= count
    ):
        raise ExactTargetNativeStageError(
            "compiler.canonical_validation_count must leave nonempty disjoint "
            "calibration and validation partitions inside fit."
        )
    boundary = count - validation_count

    def subset(name: str, indices: range) -> SplitBinding:
        selected = tuple(indices)
        payload = dict(source.payload)
        payload["split"] = name
        payload["prompts"] = [source.payload["prompts"][index] for index in selected]
        subset_sha = _canonical_sha256(
            {
                "source_fit_artifact_sha256": source.sha256,
                "role": name,
                "ordered_indices": list(selected),
            }
        )
        return SplitBinding(
            name=name,  # type: ignore[arg-type]
            path=source.path,
            sha256=subset_sha,
            expected_count=len(selected),
            payload=payload,
            row_ids=tuple(source.row_ids[index] for index in selected),
            prompt_sha256s=tuple(source.prompt_sha256s[index] for index in selected),
        )

    return (
        subset("compiler-fit-calibration", range(0, boundary)),
        subset("compiler-fit-validation", range(boundary, count)),
    )


def _compiler_partition_report(
    contract: AuthenticatedPreRegistration,
) -> dict[str, Any]:
    calibration, validation = _compiler_fit_partitions(contract)
    return {
        "source_split": "compiler_split",
        "source_fit_artifact_sha256": contract.compiler_split.sha256,
        "calibration_row_count": len(calibration.row_ids),
        "calibration_row_ids_sha256": _canonical_sha256(list(calibration.row_ids)),
        "validation_row_count": len(validation.row_ids),
        "validation_row_ids_sha256": _canonical_sha256(list(validation.row_ids)),
        "candidate_split_used": False,
        "sealed_split_used": False,
        "partition_fixed_before_model_access": True,
    }


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ExactTargetNativeStageError("Tokenizer has no stable vocabulary map.")
    return _canonical_sha256(
        {
            "vocabulary": sorted(
                (str(token), int(index)) for token, index in vocab.items()
            ),
            "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
            "length": len(tokenizer),
        }
    )


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class _CapturedSplit:
    trace: CanonicalParentTrace
    logits: Tensor
    model_inputs: Mapping[str, Tensor]


def _capture_canonical_parent_split(
    parent_model: nn.Module,
    tokenizer: Any,
    split: SplitBinding,
    *,
    target_layer_count: int,
    batch_size: int,
    max_prompt_tokens: int,
    tokenizer_fingerprint: str,
) -> _CapturedSplit:
    prompts = _prompt_token_ids(split)
    layer_chunks: list[list[Tensor]] | None = None
    logit_chunks: list[Tensor] = []
    if any(len(prompt) > max_prompt_tokens for prompt in prompts):
        raise ExactTargetNativeStageError(
            "An authenticated canonical prompt exceeds max_prompt_tokens; refusing "
            "to silently truncate it."
        )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ExactTargetNativeStageError("Tokenizer must expose an integer pad token.")
    vocabulary_size = int(getattr(parent_model.config, "vocab_size", 0))
    if vocabulary_size <= 0 or any(
        token >= vocabulary_size for prompt in prompts for token in prompt
    ):
        raise ExactTargetNativeStageError(
            "An authenticated prompt token lies outside the parent vocabulary."
        )
    width = max(len(prompt) for prompt in prompts)
    input_ids = torch.full((len(prompts), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
    left_padding = getattr(tokenizer, "padding_side", "right") == "left"
    for row, prompt in enumerate(prompts):
        start = width - len(prompt) if left_padding else 0
        input_ids[row, start : start + len(prompt)] = torch.tensor(
            prompt, dtype=torch.long
        )
        attention_mask[row, start : start + len(prompt)] = 1
    complete_inputs: dict[str, Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for start in range(0, len(prompts), batch_size):
        stop = min(start + batch_size, len(prompts))
        batch = {
            name: value[start:stop]
            for name, value in complete_inputs.items()
            if isinstance(value, Tensor)
        }
        device_batch = _move_inputs(batch, parent_model)
        with torch.inference_mode():
            output = parent_model(
                **device_batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = getattr(output, "hidden_states", None)
        logits = getattr(output, "logits", None)
        if not isinstance(hidden_states, tuple) or len(hidden_states) < 2:
            raise ExactTargetNativeStageError(
                "Parent did not return a complete hidden-state trajectory."
            )
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ExactTargetNativeStageError("Parent did not return causal logits.")
        mask = device_batch["attention_mask"].to(dtype=torch.bool)
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        final_positions = positions.masked_fill(~mask, -1).max(dim=1).values
        if bool((final_positions < 0).any()):
            raise ExactTargetNativeStageError("A canonical prompt has no active token.")
        row = torch.arange(mask.shape[0], device=mask.device)
        if layer_chunks is None:
            layer_chunks = [[] for _ in hidden_states]
        if len(layer_chunks) != len(hidden_states):
            raise ExactTargetNativeStageError(
                "Parent hidden-state depth changed between capture batches."
            )
        for destination, state in zip(layer_chunks, hidden_states, strict=True):
            destination.append(
                state[row, final_positions]
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
        logit_chunks.append(
            logits[row, final_positions].detach().to(device="cpu", dtype=torch.float64)
        )
        del output, hidden_states, logits
    assert layer_chunks is not None
    layers = tuple(torch.cat(chunks, dim=0) for chunks in layer_chunks)
    captured_logits = torch.cat(logit_chunks, dim=0)
    source_layers = len(layers) - 1
    cache_fingerprint = _canonical_sha256(
        {
            "split_sha256": split.sha256,
            "layer_sha256s": [_tensor_sha256(value) for value in layers],
            "logits_sha256": _tensor_sha256(captured_logits),
            "pooling": "last_valid",
        }
    )
    identity = CanonicalRowIdentity(
        split_name=split.name,
        pooling="last_valid",
        row_ids=split.row_ids,
        raw_prompt_sha256=split.prompt_sha256s,
        corpus_sha256=_canonical_sha256([list(prompt) for prompt in prompts]),
        tokenizer_fingerprint=tokenizer_fingerprint,
        cache_fingerprint=cache_fingerprint,
    )
    trace = CanonicalParentTrace(
        name="Qwen3-4B-Base",
        layer_states=layers,
        row_identity=identity,
        depth_cover=build_proportional_depth_cover(source_layers, target_layer_count),
        weight=1.0,
    )
    return _CapturedSplit(
        trace=trace,
        logits=captured_logits,
        model_inputs={
            name: value
            for name, value in complete_inputs.items()
            if isinstance(value, Tensor)
        },
    )


def _compile_real_parent_evidence(
    parent_model: nn.Module,
    target_model: nn.Module,
    tokenizer: Any,
    contract: AuthenticatedPreRegistration,
) -> HFParentCompilerResult:
    stage = default_qwen4b_progressive_target_plan().stages[0]
    batch_size = int(contract.compiler.get("capture_batch_size", 1))
    max_tokens = int(contract.compiler.get("max_prompt_tokens", 256))
    if batch_size <= 0 or max_tokens <= 0:
        raise ExactTargetNativeStageError(
            "Compiler capture_batch_size and max_prompt_tokens must be positive."
        )
    tokenizer_sha = _tokenizer_fingerprint(tokenizer)
    compiler_calibration, compiler_validation = _compiler_fit_partitions(contract)
    fit = _capture_canonical_parent_split(
        parent_model,
        tokenizer,
        compiler_calibration,
        target_layer_count=TARGET_LAYER_COUNT,
        batch_size=batch_size,
        max_prompt_tokens=max_tokens,
        tokenizer_fingerprint=tokenizer_sha,
    )
    validation = _capture_canonical_parent_split(
        parent_model,
        tokenizer,
        compiler_validation,
        target_layer_count=TARGET_LAYER_COUNT,
        batch_size=batch_size,
        max_prompt_tokens=max_tokens,
        tokenizer_fingerprint=tokenizer_sha,
    )
    token_contract = CanonicalTokenContract(
        parent_name="Qwen3-4B-Base",
        parent_tokenizer_fingerprint=tokenizer_sha,
        child_tokenizer_fingerprint=tokenizer_sha,
        method="identity",
        mapping_fingerprint=tokenizer_sha,
        coverage_fraction=1.0,
        exact=True,
    )
    parent_logits = CanonicalParentLogits(
        parent_name="Qwen3-4B-Base",
        calibration_logits=fit.logits,
        validation_logits=validation.logits,
        token_contract=token_contract,
    )
    layout = build_decoder_layout(target_model)
    child_spec = CanonicalChildSpec(
        family="qwen3",
        layer_count=TARGET_LAYER_COUNT,
        hidden_size=stage.shape.hidden_size,
        parameter_budget=QWEN3_1P4B_EXACT_PARAMETERS,
        vocabulary_size=stage.shape.vocab_size,
        tokenizer_fingerprint=tokenizer_sha,
        tie_embeddings=False,
        position_policy=str(layout.position.kind),
        architecture_fingerprint=fingerprint_hf_child_structure(
            target_model, layout=layout
        ),
    )
    bootstrap_kwargs = contract.compiler.get("bootstrap_kwargs", {})
    if not isinstance(bootstrap_kwargs, Mapping):
        raise ExactTargetNativeStageError(
            "compiler.bootstrap_kwargs must be an object."
        )
    return compile_hf_parents_to_blank_child(
        (fit.trace,),
        (validation.trace,),
        (parent_logits,),
        structural_carrier_model=parent_model,
        blank_target_model=target_model,
        carrier_calibration_inputs=_move_inputs(fit.model_inputs, parent_model),
        carrier_validation_inputs=_move_inputs(validation.model_inputs, parent_model),
        child_calibration_inputs=_move_inputs(fit.model_inputs, target_model),
        child_validation_inputs=_move_inputs(validation.model_inputs, target_model),
        child_spec=child_spec,
        exact_parameter_budget=QWEN3_1P4B_EXACT_PARAMETERS,
        bootstrap_kwargs=dict(bootstrap_kwargs),
    )


def _metrics_dict(value: Any) -> dict[str, Any]:
    if value is None:
        raise ExactTargetNativeStageError("Required behavior metrics are absent.")
    if hasattr(value, "as_dict"):
        payload = value.as_dict()
    elif dataclasses.is_dataclass(value):
        payload = dataclasses.asdict(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ExactTargetNativeStageError("Behavior metrics have no auditable mapping.")
    if not isinstance(payload, Mapping):
        raise ExactTargetNativeStageError("Behavior metrics mapping is invalid.")
    return dict(payload)


def _behavior_gate(
    metrics: Any,
    thresholds: BehaviorThresholds,
) -> tuple[bool, dict[str, Any]]:
    """Decide exclusively from public causal output behavior."""

    payload = _metrics_dict(metrics)
    required = {
        "action_count",
        "action_accuracy",
        "rollout_count",
        "exact_rollout_fraction",
        "token_match_fraction",
        "lcp_fraction",
    }
    repetition_required = (
        thresholds.maximum_repeat_fraction_excess < 1.0
        or thresholds.maximum_adjacent_repeat_fraction_excess < 1.0
        or thresholds.maximum_longest_run_excess < 2_147_483_647
    )
    if repetition_required:
        required.update(
            {
                "repeat_fraction_excess",
                "adjacent_repeat_fraction_excess",
                "longest_run_excess",
            }
        )
    if not required.issubset(payload):
        raise ExactTargetNativeStageError(
            f"Behavior metrics are missing {sorted(required - set(payload))}."
        )
    for key in ("action_count", "rollout_count"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExactTargetNativeStageError(
                f"Behavior metric {key} must be a nonnegative integer."
            )
    for key in (
        "action_accuracy",
        "exact_rollout_fraction",
        "token_match_fraction",
        "lcp_fraction",
        "repeat_fraction_excess",
        "adjacent_repeat_fraction_excess",
    ):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExactTargetNativeStageError(
                f"Behavior metric {key} must be a finite fraction."
            )
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ExactTargetNativeStageError(
                f"Behavior metric {key} must be in [0, 1]."
            )
    if "longest_run_excess" in payload and (
        isinstance(payload["longest_run_excess"], bool)
        or not isinstance(payload["longest_run_excess"], int)
        or payload["longest_run_excess"] < 0
    ):
        raise ExactTargetNativeStageError(
            "Behavior metric longest_run_excess must be a nonnegative integer."
        )

    def consistent_fraction(
        numerator_name: str,
        denominator_name: str,
        fraction_name: str,
    ) -> None:
        if numerator_name not in payload:
            return
        numerator = payload[numerator_name]
        denominator = payload[denominator_name]
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or numerator < 0
            or numerator > denominator
        ):
            raise ExactTargetNativeStageError(
                f"Behavior metric {numerator_name} is inconsistent."
            )
        expected = 0.0 if denominator == 0 else numerator / denominator
        if not math.isclose(
            float(payload[fraction_name]), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ExactTargetNativeStageError(
                f"Behavior metric {fraction_name} disagrees with its counts."
            )

    consistent_fraction("action_agreement_count", "action_count", "action_accuracy")
    consistent_fraction(
        "exact_rollout_count", "rollout_count", "exact_rollout_fraction"
    )
    if "rollout_token_count" in payload:
        token_count = payload["rollout_token_count"]
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ExactTargetNativeStageError(
                "Behavior metric rollout_token_count must be nonnegative integer."
            )
        consistent_fraction(
            "token_match_count", "rollout_token_count", "token_match_fraction"
        )
        consistent_fraction("lcp_token_count", "rollout_token_count", "lcp_fraction")
    checks = {
        "minimum_action_count": int(payload["action_count"])
        >= thresholds.minimum_action_count,
        "minimum_action_accuracy": float(payload["action_accuracy"])
        >= thresholds.minimum_action_accuracy,
        "minimum_rollout_count": int(payload["rollout_count"])
        >= thresholds.minimum_rollout_count,
        "minimum_exact_rollout_fraction": float(payload["exact_rollout_fraction"])
        >= thresholds.minimum_exact_rollout_fraction,
        "minimum_token_match_fraction": float(payload["token_match_fraction"])
        >= thresholds.minimum_token_match_fraction,
        "minimum_lcp_fraction": float(payload["lcp_fraction"])
        >= thresholds.minimum_lcp_fraction,
        "maximum_repeat_fraction_excess": float(
            payload.get("repeat_fraction_excess", 0.0)
        )
        <= thresholds.maximum_repeat_fraction_excess,
        "maximum_adjacent_repeat_fraction_excess": float(
            payload.get("adjacent_repeat_fraction_excess", 0.0)
        )
        <= thresholds.maximum_adjacent_repeat_fraction_excess,
        "maximum_longest_run_excess": int(payload.get("longest_run_excess", 0))
        <= thresholds.maximum_longest_run_excess,
    }
    decision = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": payload,
        "thresholds": thresholds.as_dict(),
        "selection_inputs": sorted(required),
        "internal_numeric_matching_consulted": False,
    }
    decision["decision_sha256"] = _canonical_sha256(decision)
    return bool(decision["passed"]), decision


class ProductionStageRuntime:
    """Heavy runtime boundary; tests replace this object with a strict fake."""

    def __init__(
        self,
        *,
        dtype: str = "float16",
        parent_device_map: Any = "auto",
        target_device_map: Any = "auto",
        trainer_module: str = "neural_forge.hf_target_native_behavior_distillation",
    ) -> None:
        if dtype != "float16":
            raise ExactTargetNativeStageError(
                "The authenticated canonical stage-1 blank is FP16; production "
                "runtime dtype is pinned to float16."
            )
        if trainer_module != CANONICAL_TRAINER_MODULE:
            raise ExactTargetNativeStageError(
                "Only the source-pinned canonical output/rollout trainer is allowed."
            )
        self.dtype = dtype
        self.parent_device_map = parent_device_map
        self.target_device_map = target_device_map
        self.trainer_module_name = trainer_module
        self._compiler_identity: CompilerIdentity | None = None
        self._compiler_cache_path: Path | None = None
        self._verified_compiler_cache: Any | None = None
        self._loaded_compiler_cache: Any | None = None
        self._lexical_rank: int | None = None

    def _torch_dtype(self) -> torch.dtype:
        return torch.float16

    def _compiler_identity_for_contract(
        self, contract: AuthenticatedPreRegistration
    ) -> CompilerIdentity:
        parent_tree = contract.artifact_authentication["parent"].get("tree")
        target_tree = contract.artifact_authentication["blank_target"].get("tree")
        if not isinstance(parent_tree, Mapping) or not isinstance(target_tree, Mapping):
            raise ExactTargetNativeStageError(
                "Shared compiler caching requires complete parent and blank-target "
                "artifact tree bindings."
            )
        parent_tokenizer_path = contract.parent.path / "tokenizer.json"
        if not parent_tokenizer_path.is_file():
            raise ExactTargetNativeStageError(
                "Compiler cache identity cannot bind the actual parent tokenizer."
            )
        tokenizer_sha = _file_sha256(parent_tokenizer_path)
        calibration, validation = _compiler_fit_partitions(contract)
        compiler_modules = (
            "neural_forge.hf_arch_layout",
            "neural_forge.hf_blank_bootstrap",
            "neural_forge.hf_canonical_graph",
            "neural_forge.hf_canonical_kernel",
            "neural_forge.hf_parent_compiler",
        )
        implementation_hashes: dict[str, str] = {}
        for module_name in compiler_modules:
            module = importlib.import_module(module_name)
            module_path = Path(str(module.__file__)).resolve()
            implementation_hashes[module_name] = _file_sha256(module_path)
        glue_functions = (
            _prompt_token_ids,
            _compiler_fit_partitions,
            _tokenizer_fingerprint,
            _capture_canonical_parent_split,
            _compile_real_parent_evidence,
        )
        implementation_hashes["exact_stage_compiler_glue"] = _sha256_text(
            "\n\n".join(inspect.getsource(value) for value in glue_functions)
        )

        def device_record(value: Any) -> dict[str, Any]:
            result: dict[str, Any] = {"device_map": str(value)}
            if str(value) == "auto" and torch.cuda.is_available():
                result["visible_devices"] = [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_properties(index).name,
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory": int(
                            torch.cuda.get_device_properties(index).total_memory
                        ),
                    }
                    for index in range(torch.cuda.device_count())
                ]
                return result
            try:
                device = torch.device(str(value))
            except (RuntimeError, TypeError, ValueError):
                return result
            if device.type == "cuda" and torch.cuda.is_available():
                index = (
                    torch.cuda.current_device()
                    if device.index is None
                    else device.index
                )
                properties = torch.cuda.get_device_properties(index)
                result.update(
                    {
                        "name": properties.name,
                        "capability": [properties.major, properties.minor],
                        "total_memory": int(properties.total_memory),
                    }
                )
            return result

        return CompilerIdentity(
            parent_artifact_tree_sha256=str(parent_tree["tree_sha256"]),
            blank_target_artifact_tree_sha256=str(target_tree["tree_sha256"]),
            tokenizer_sha256=tokenizer_sha,
            compiler_input_sha256s={
                "compiler_split_artifact": contract.compiler_split.sha256,
                "calibration_row_ids": _canonical_sha256(list(calibration.row_ids)),
                "validation_row_ids": _canonical_sha256(list(validation.row_ids)),
                "calibration_prompt_ids": _canonical_sha256(
                    list(calibration.prompt_sha256s)
                ),
                "validation_prompt_ids": _canonical_sha256(
                    list(validation.prompt_sha256s)
                ),
            },
            compiler_implementation_sha256s=implementation_hashes,
            compiler_options={
                **dict(contract.compiler),
                "target_exact_unique_parameters": QWEN3_1P4B_EXACT_PARAMETERS,
                "target_layer_count": TARGET_LAYER_COUNT,
                "trace_pooling": "last_valid",
                "token_contract": "identity",
            },
            runtime={
                "dtype": self.dtype,
                "attention_implementation": CANONICAL_ATTENTION_IMPLEMENTATION,
                "torch_version": torch.__version__,
                "transformers_version": importlib.metadata.version("transformers"),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "torch_build_config_sha256": _sha256_text(torch.__config__.show()),
                "parent_device": device_record(self.parent_device_map),
                "target_device": device_record(self.target_device_map),
            },
            seed=int(contract.compiler.get("compiler_seed", 0)),
        )

    def prepare_compiler_cache(
        self, contract: AuthenticatedPreRegistration
    ) -> Mapping[str, Any] | None:
        self._lexical_rank = (
            None
            if contract.lexical_residual is None
            else contract.lexical_residual.rank
        )
        binding = contract.compiler_cache
        if binding is None:
            return None
        identity = self._compiler_identity_for_contract(contract)
        if (
            binding.expected_compiler_identity_sha256 is not None
            and identity.compiler_identity_sha256
            != binding.expected_compiler_identity_sha256
        ):
            raise ExactTargetNativeStageError(
                "The computed compiler identity differs from the required cache hit."
            )
        cache_path = (binding.root / identity.compiler_identity_sha256).resolve()
        try:
            cache_path.relative_to(binding.root.resolve())
        except ValueError as error:
            raise ExactTargetNativeStageError(
                "Derived compiler cache path escapes its authenticated root."
            ) from error
        self._compiler_identity = identity
        self._compiler_cache_path = cache_path
        if cache_path.exists():
            try:
                verified = verify_compiled_child_cache(
                    cache_path,
                    expected_identity=identity,
                    expected_receipt_sha256=binding.expected_receipt_sha256,
                    expected_compiler_report_sha256=(
                        binding.expected_compiler_report_sha256
                    ),
                )
            except CompiledChildCacheIntegrityError as error:
                raise ExactTargetNativeStageError(
                    "Existing shared compiler cache failed authentication."
                ) from error
            self._verified_compiler_cache = verified
            receipt_payload = verified.receipt.as_dict()
            if (
                binding.expected_state_dict_sha256 is not None
                and receipt_payload["model_state"]["state_dict_sha256"]
                != binding.expected_state_dict_sha256
            ):
                raise ExactTargetNativeStageError(
                    "The required compiler cache state-dict hash changed."
                )
            if (
                binding.expected_artifact_tree_sha256 is not None
                and receipt_payload["artifact_tree"]["tree_sha256"]
                != binding.expected_artifact_tree_sha256
            ):
                raise ExactTargetNativeStageError(
                    "The required compiler cache artifact-tree hash changed."
                )
            disposition = "hit"
            receipt_sha = verified.receipt.receipt_sha256
        else:
            if binding.require_hit:
                raise ExactTargetNativeStageError(
                    "The preregistered compiler cache requires an exact hit, but "
                    "the authenticated identity is absent."
                )
            self._verified_compiler_cache = None
            disposition = "miss"
            receipt_sha = None
        return {
            "disposition": disposition,
            "path": str(cache_path),
            "compiler_identity_sha256": identity.compiler_identity_sha256,
            "receipt_sha256": receipt_sha,
            "required_hit": binding.require_hit,
            "verified_before_model_access": disposition == "hit",
            "copied_into_run_output": False,
        }

    def load_models(
        self, contract: AuthenticatedPreRegistration
    ) -> tuple[nn.Module, nn.Module, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError(
                "Transformers is required for the production stage."
            ) from error
        dtype = self._torch_dtype()
        parent = AutoModelForCausalLM.from_pretrained(
            contract.parent.path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            device_map=self.parent_device_map,
            attn_implementation=CANONICAL_ATTENTION_IMPLEMENTATION,
        ).eval()
        if self._verified_compiler_cache is None:
            target = AutoModelForCausalLM.from_pretrained(
                contract.target.path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
                device_map=self.target_device_map,
                attn_implementation=CANONICAL_ATTENTION_IMPLEMENTATION,
            ).eval()
        else:
            assert self._compiler_identity is not None
            assert self._compiler_cache_path is not None

            def load_cached(path: Path) -> nn.Module:
                return AutoModelForCausalLM.from_pretrained(
                    path,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=dtype,
                    device_map=self.target_device_map,
                    attn_implementation=CANONICAL_ATTENTION_IMPLEMENTATION,
                ).eval()

            self._loaded_compiler_cache = load_compiled_child_cache(
                self._compiler_cache_path,
                expected_identity=self._compiler_identity,
                model_loader=load_cached,
                expected_receipt_sha256=(
                    self._verified_compiler_cache.receipt.receipt_sha256
                ),
            )
            target = self._loaded_compiler_cache.model
        tokenizer = AutoTokenizer.from_pretrained(
            contract.parent.path,
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ExactTargetNativeStageError("Tokenizer has no pad/eos token.")
            tokenizer.pad_token = tokenizer.eos_token
        parent_count = int(exact_parameter_report(parent)["unique_parameter_count"])
        target_count = int(exact_parameter_report(target)["unique_parameter_count"])
        if parent_count != QWEN3_4B_EXACT_PARAMETERS:
            raise ExactTargetNativeStageError(
                f"Live parent contains {parent_count}, not {QWEN3_4B_EXACT_PARAMETERS}, "
                "unique parameters."
            )
        if target_count != QWEN3_1P4B_EXACT_PARAMETERS:
            raise ExactTargetNativeStageError(
                f"Live target contains {target_count}, not "
                f"{QWEN3_1P4B_EXACT_PARAMETERS}, unique parameters."
            )
        stage = default_qwen4b_progressive_target_plan().stages[0]
        certify_native_dense_gqa_target(
            target,
            exact_unique_parameters=QWEN3_1P4B_EXACT_PARAMETERS,
            stage=stage,
            require_dense_swiglu=True,
        )
        return parent, target, tokenizer

    def compile(
        self,
        parent: nn.Module,
        target: nn.Module,
        tokenizer: Any,
        contract: AuthenticatedPreRegistration,
    ) -> HFParentCompilerResult | CompiledBase:
        if self._loaded_compiler_cache is not None:
            report = dict(self._loaded_compiler_cache.compiler_report)
            state_sha = self.state_sha256(target)
            receipt = self._loaded_compiler_cache.receipt.as_dict()
            return CompiledBase(
                child_model=target,
                report=report,
                compiler_report_sha256=_canonical_sha256(report),
                compiled_state_sha256=state_sha,
                cache_evidence={
                    "disposition": "hit",
                    "path": str(self._loaded_compiler_cache.path),
                    "compiler_identity_sha256": (
                        self._loaded_compiler_cache.compiler_identity.compiler_identity_sha256
                    ),
                    "receipt_sha256": self._loaded_compiler_cache.receipt.receipt_sha256,
                    "compiler_report_sha256": _canonical_sha256(report),
                    "state_dict_sha256": receipt["model_state"]["state_dict_sha256"],
                    "artifact_tree_sha256": receipt["artifact_tree"]["tree_sha256"],
                    "verified_before_factor_install": True,
                    "copied_into_run_output": False,
                },
            )
        result = _compile_real_parent_evidence(parent, target, tokenizer, contract)
        report = result.report.as_dict()
        return CompiledBase(
            child_model=result.child_model,
            report=report,
            compiler_report_sha256=_canonical_sha256(report),
            compiled_state_sha256=self.state_sha256(result.child_model),
            cache_evidence={
                "disposition": "uncached-compile",
                "copied_into_run_output": False,
            },
        )

    def finalize_compiled_base(
        self,
        compiled: HFParentCompilerResult | CompiledBase,
        contract: AuthenticatedPreRegistration,
    ) -> HFParentCompilerResult | CompiledBase:
        if not isinstance(compiled, CompiledBase):
            return compiled
        if compiled.cache_evidence.get("disposition") == "hit":
            return compiled
        if contract.compiler_cache is not None and contract.compiler_cache.require_hit:
            raise ExactTargetNativeStageError(
                "A required immutable compiler-cache hit cannot be republished."
            )
        if self._compiler_identity is None or self._compiler_cache_path is None:
            return compiled
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover
            raise RuntimeError(
                "Transformers is required for compiler caching."
            ) from error

        def save_model(model: nn.Module, path: Path) -> None:
            model.save_pretrained(
                path,
                safe_serialization=True,
                max_shard_size="10GB",
            )

        def load_cpu(path: Path) -> nn.Module:
            return AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=self._torch_dtype(),
                device_map="cpu",
                attn_implementation=CANONICAL_ATTENTION_IMPLEMENTATION,
            ).eval()

        receipt = publish_compiled_child_cache(
            self._compiler_cache_path,
            model=compiled.child_model,
            compiler_report=compiled.report,
            compiler_identity=self._compiler_identity,
            model_saver=save_model,
            model_loader=load_cpu,
        )
        verified = verify_compiled_child_cache(
            self._compiler_cache_path,
            expected_identity=self._compiler_identity,
            expected_receipt_sha256=receipt.receipt_sha256,
            expected_compiler_report_sha256=compiled.compiler_report_sha256,
        )
        receipt_payload = verified.receipt.as_dict()
        return dataclasses.replace(
            compiled,
            cache_evidence={
                "disposition": "published",
                "path": str(verified.path),
                "compiler_identity_sha256": (
                    verified.compiler_identity.compiler_identity_sha256
                ),
                "receipt_sha256": verified.receipt.receipt_sha256,
                "compiler_report_sha256": compiled.compiler_report_sha256,
                "state_dict_sha256": receipt_payload["model_state"][
                    "state_dict_sha256"
                ],
                "artifact_tree_sha256": receipt_payload["artifact_tree"]["tree_sha256"],
                "verified_before_factor_install": True,
                "copied_into_run_output": False,
            },
        )

    def state_sha256(self, model: nn.Module) -> str:
        return _state_dict_sha256(torch, model)

    def install_transaction(self, compiled: HFParentCompilerResult) -> Any:
        stage = default_qwen4b_progressive_target_plan().stages[0]
        sites = tuple(
            DenseGQASite(layer, group)
            for layer in range(TARGET_LAYER_COUNT)
            for group in range(TARGET_PHYSICAL_GROUPS)
        )
        if len(sites) != QKVO_SITE_COUNT:
            raise AssertionError("The exact QKVO site topology changed.")
        if isinstance(compiled, CompiledBase):
            return install_exact_target_compiler_transaction(
                compiled.child_model,
                exact_unique_parameters=QWEN3_1P4B_EXACT_PARAMETERS,
                qkvo_sites=sites,
                qkvo_rank=QKVO_RANK,
                swiglu_layer_indices=tuple(range(TARGET_LAYER_COUNT)),
                swiglu_rank=SWIGLU_RANK,
                lexical_rank=self._lexical_rank,
                stage=stage,
                initialization_seed=0,
                parent_compiler_report_sha256=compiled.compiler_report_sha256,
            )
        return install_parent_compiler_result_target_native_transaction(
            compiled,
            qkvo_sites=sites,
            qkvo_rank=QKVO_RANK,
            swiglu_layer_indices=tuple(range(TARGET_LAYER_COUNT)),
            swiglu_rank=SWIGLU_RANK,
            lexical_rank=self._lexical_rank,
            stage=stage,
            initialization_seed=0,
        )

    def train(
        self,
        transaction: Any,
        parent: nn.Module,
        contract: AuthenticatedPreRegistration,
    ) -> Any:
        module = importlib.import_module(self.trainer_module_name)
        loader = getattr(module, "load_behavior_domain_suite")
        suites = {
            "fit": loader(
                "fit",
                contract.splits["fit"].payload,
                artifact_sha256=contract.splits["fit"].sha256,
            ),
            "candidate": loader(
                "candidate",
                contract.splits["candidate"].payload,
                artifact_sha256=contract.splits["candidate"].sha256,
            ),
            "sealed": loader(
                "sealed",
                contract.splits["sealed"].payload,
                artifact_sha256=contract.splits["sealed"].sha256,
            ),
        }
        schedule_type = getattr(module, "BehaviorDistillationSchedule")
        schedule = schedule_type(**dict(contract.training))
        threshold_type = getattr(module, "CandidateBehaviorThresholds")
        candidate_thresholds = threshold_type(
            **contract.thresholds["candidate"].as_dict()
        )
        trainer = getattr(module, "train_target_native_behavior_distillation")
        kwargs = {
            "transaction": transaction,
            "parent_model": parent,
            "domains": suites,
            "schedule": schedule,
            "candidate_thresholds": candidate_thresholds,
            "seed": contract.seed,
        }
        if contract.continuation is not None:
            kwargs["initial_factor_payload"] = contract.continuation.factor_payload
        result = trainer(**kwargs)
        result_type = getattr(module, "BehaviorDistillationResult")
        if type(result) is not result_type:
            raise ExactTargetNativeStageError(
                "Canonical trainer returned a noncanonical result type."
            )
        return result

    def run_output_jacobian_audit(
        self,
        transaction: Any,
        parent: nn.Module,
        contract: AuthenticatedPreRegistration,
    ) -> OutputJacobianAuditExecution:
        """Restore the authenticated v2 state and run the audit-only diagnostic."""

        binding = contract.output_jacobian_audit
        continuation = contract.continuation
        if binding is None or continuation is None:
            raise ExactTargetNativeStageError(
                "The production output-Jacobian runtime requires its authenticated "
                "audit and continuation bindings."
            )
        if (
            continuation.trainer_factor_pack_schema != TRAINER_FACTOR_PACK_V2_SCHEMA
            or continuation.lexical_rank != LEXICAL_RANK
        ):
            raise ExactTargetNativeStageError(
                "The output-Jacobian runtime can restore only a rank-bound v2 pack."
            )
        trainer_module = importlib.import_module(self.trainer_module_name)
        restore = getattr(trainer_module, "restore_target_native_factor_payload")
        restored_state = restore(transaction, continuation.factor_payload)
        restored_sha = getattr(restored_state, "factor_state_sha256", None)
        restored_schema = getattr(restored_state, "schema", None)
        restored_lexical_rank = getattr(restored_state, "lexical_rank", None)
        if (
            restored_sha != continuation.factor_state_sha256
            or restored_schema != TRAINER_FACTOR_PACK_V2_SCHEMA
            or restored_lexical_rank != LEXICAL_RANK
        ):
            raise ExactTargetNativeStageError(
                "The public trainer restore did not prove the full authenticated v2 "
                "factor state."
            )
        loader = getattr(trainer_module, "load_behavior_domain_suite")
        source = contract.splits[binding.basis_split_role]
        basis_domain = loader(
            "fit",
            source.payload,
            artifact_sha256=source.sha256,
        )
        if (
            getattr(basis_domain, "split_name", None) != "fit"
            or len(getattr(basis_domain, "prompts", ())) != binding.prompt_count
        ):
            raise ExactTargetNativeStageError(
                "The source candidate was not exposed as the exact 24-prompt fit "
                "basis domain."
            )
        schedule_type = getattr(trainer_module, "BehaviorDistillationSchedule")
        schedule = schedule_type(**dict(contract.training))
        audit_module = importlib.import_module(QKVO_OUTPUT_JACOBIAN_AUDIT_MODULE)
        config_type = getattr(audit_module, "QKVOOutputJacobianAuditConfig")
        config = config_type(
            prompt_count=binding.prompt_count,
            horizon=binding.horizon,
            lanes=binding.lanes,
            rank=binding.rank,
            sketch_width=binding.sketch_width,
            seed=contract.seed,
            min_normal_fraction=binding.min_normal_fraction,
            min_guided_over_random_ratio=(binding.min_guided_over_random_ratio),
            max_bytes=binding.max_bytes,
        )
        audit = getattr(audit_module, "run_qkvo_output_jacobian_audit")
        result = audit(
            transaction,
            parent,
            basis_domain,
            schedule,
            config,
        )
        expected_type = getattr(audit_module, "QKVOOutputJacobianAuditResult")
        if type(result) is not expected_type:
            raise ExactTargetNativeStageError(
                "The source-pinned output-Jacobian audit returned a noncanonical "
                "result."
            )
        capture = getattr(trainer_module, "capture_target_native_factor_state")
        state_after_audit = capture(transaction)
        if (
            getattr(state_after_audit, "factor_state_sha256", None)
            != continuation.factor_state_sha256
            or getattr(state_after_audit, "schema", None)
            != TRAINER_FACTOR_PACK_V2_SCHEMA
            or getattr(state_after_audit, "lexical_rank", None) != LEXICAL_RANK
        ):
            raise ExactTargetNativeStageError(
                "Output-Jacobian audit changed the authenticated live factor state."
            )
        return OutputJacobianAuditExecution(
            result=result,
            restore_mode="full",
            restored_factor_state_sha256=restored_sha,
            restored_factor_schema=restored_schema,
        )

    def evaluate_baked_behavior(
        self,
        model: nn.Module,
        parent: nn.Module,
        contract: AuthenticatedPreRegistration,
        split_name: Literal["fit", "candidate", "sealed"],
    ) -> Any:
        """Re-evaluate the native baked bytes under the pinned output gate."""

        if split_name not in {"fit", "candidate", "sealed"}:
            raise ExactTargetNativeStageError(
                f"Unsupported post-bake behavior split {split_name!r}."
            )
        module = importlib.import_module(self.trainer_module_name)
        loader = getattr(module, "load_behavior_domain_suite")
        evaluator = getattr(module, "evaluate_behavior_domain")
        schedule_type = getattr(module, "BehaviorDistillationSchedule")
        schedule = schedule_type(**dict(contract.training))
        binding = contract.splits[split_name]
        suite = loader(
            split_name,
            binding.payload,
            artifact_sha256=binding.sha256,
        )
        evaluation_split = {
            "fit": "calibration-exploratory",
            "candidate": "calibration-selection",
            "sealed": "heldout-evaluation",
        }[split_name]
        return evaluator(
            parent,
            model,
            suite,
            horizon=schedule.evaluation_horizon,
            top_k=schedule.top_k,
            split=evaluation_split,
        )

    def factor_pack_payload(self, training_result: Any) -> Mapping[str, Any]:
        method = getattr(training_result, "factor_pack_payload", None)
        if not callable(method):
            raise ExactTargetNativeStageError(
                "Trainer result did not expose a factor-only pack."
            )
        payload = method()
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("factor_tensors"), Mapping
        ):
            raise ExactTargetNativeStageError("Trainer factor pack is malformed.")
        tensors = payload["factor_tensors"]
        if not tensors or any(
            not isinstance(value, Tensor) for value in tensors.values()
        ):
            raise ExactTargetNativeStageError(
                "Trainer factor pack must contain only named tensors."
            )
        manifest = payload.get("factors")
        if not isinstance(manifest, list) or {
            str(value.get("name")) for value in manifest if isinstance(value, Mapping)
        } != set(tensors):
            raise ExactTargetNativeStageError(
                "Trainer factor tensor names disagree with the factor manifest."
            )
        return payload

    def save_reload_certify(
        self,
        model: nn.Module,
        tokenizer: Any,
        output_root: Path,
    ) -> dict[str, Any]:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Transformers is required for fresh reload.") from error
        staging = (output_root / ".accepted_child.staging").resolve()
        final = (output_root / "accepted_child").resolve()
        if staging.exists() or final.exists():
            raise ExactTargetNativeStageError(
                "Accepted child staging/final path already exists."
            )
        before_keys = tuple(sorted(model.state_dict()))
        before_key_sha = _canonical_sha256(list(before_keys))
        before_state_sha = _state_dict_sha256(torch, model)
        stage = default_qwen4b_progressive_target_plan().stages[0]
        model.save_pretrained(
            staging,
            safe_serialization=True,
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(staging)
        try:
            reloaded = AutoModelForCausalLM.from_pretrained(
                staging,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=self._torch_dtype(),
                device_map="cpu",
                attn_implementation=CANONICAL_ATTENTION_IMPLEMENTATION,
            ).eval()
            certificate = certify_native_dense_gqa_target(
                reloaded,
                exact_unique_parameters=QWEN3_1P4B_EXACT_PARAMETERS,
                stage=stage,
                require_dense_swiglu=True,
            )
            after_keys = tuple(sorted(reloaded.state_dict()))
            after_key_sha = _canonical_sha256(list(after_keys))
            if after_keys != before_keys or after_key_sha != before_key_sha:
                raise ExactTargetNativeStageError(
                    "Fresh-reloaded state keys differ from the baked child."
                )
            after_state_sha = _state_dict_sha256(torch, reloaded)
            if after_state_sha != before_state_sha:
                raise ExactTargetNativeStageError(
                    "Fresh-reloaded tensor bytes differ from the baked child."
                )
            banned_modules = (
                "ResidualProjection",
                "ResidualRegistry",
                "Adapter",
            )
            if any(
                any(word in type(module).__name__ for word in banned_modules)
                for module in reloaded.modules()
            ):
                raise ExactTargetNativeStageError(
                    "Fresh-reloaded child retains an adapter/registry wrapper."
                )
            banned_names = ("adapter", "factor", "optimizer", "parent", "sidecar")
            bad_files = [
                path.relative_to(staging).as_posix()
                for path in staging.rglob("*")
                if path.is_file()
                and any(word in path.name.lower() for word in banned_names)
            ]
            if bad_files:
                raise ExactTargetNativeStageError(
                    f"Published child contains sidecar-like files: {bad_files}"
                )
            tree = _artifact_tree(staging)
            os.replace(staging, final)
            return {
                "fresh_reload_performed": True,
                "exact_unique_parameters": certificate.unique_parameter_count,
                "exact_budget_match": True,
                "state_key_count": len(after_keys),
                "state_key_sha256": after_key_sha,
                "state_keys_match_pre_save": True,
                "state_dict_sha256": after_state_sha,
                "state_bytes_match_pre_save": True,
                "native_certificate": certificate.as_dict(),
                "published_adapter_parameter_count": 0,
                "published_parent_runtime_reference_count": 0,
                "published_source_sized_sidecar_parameter_count": 0,
                "artifact_tree": {
                    key: tree[key]
                    for key in ("file_count", "total_bytes", "tree_sha256")
                },
                "path": str(final),
                "model_save_pretrained_calls": 1,
            }
        except BaseException:
            _remove_rejected_candidate_tree(staging, output_root)
            raise


def _remove_rejected_candidate_tree(path: Path, output_root: Path) -> None:
    """Remove only this runner's known staging tree after a failed publication."""

    resolved = path.resolve()
    root = output_root.resolve()
    if resolved.parent != root or resolved.name != ".accepted_child.staging":
        raise ExactTargetNativeStageError(
            "Refusing to remove a candidate tree outside the run output root."
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def _training_metric(training_result: Any, name: str) -> Any:
    value = getattr(training_result, name, None)
    if value is None and name == "candidate_metrics":
        value = getattr(training_result, "calibration_metrics", None)
    return value


def _compiled_report_dict(compiled: Any) -> dict[str, Any]:
    value = getattr(compiled, "report", None)
    if isinstance(value, Mapping):
        return dict(value)
    method = getattr(value, "as_dict", None)
    if not callable(method):
        raise ExactTargetNativeStageError(
            "Structural compiler result has no auditable report mapping."
        )
    payload = method()
    if not isinstance(payload, Mapping):
        raise ExactTargetNativeStageError(
            "Structural compiler report method returned a non-mapping."
        )
    return dict(payload)


def _factor_envelope(
    factor_payload: Mapping[str, Any],
    contract: AuthenticatedPreRegistration,
    reason: str,
) -> dict[str, Any]:
    forbidden = ("parent_state", "child_state", "activation", "hidden", "attention")
    if any(key in factor_payload for key in forbidden):
        raise ExactTargetNativeStageError(
            "Rejected factor pack contains forbidden parent/internal state."
        )
    lexical_rank = (
        None if contract.lexical_residual is None else contract.lexical_residual.rank
    )
    expected_trainer_schema = (
        TRAINER_FACTOR_PACK_SCHEMA
        if lexical_rank is None
        else TRAINER_FACTOR_PACK_V2_SCHEMA
    )
    if (
        lexical_rank is not None
        and factor_payload.get("schema") != expected_trainer_schema
    ):
        raise ExactTargetNativeStageError(
            "Trainer factor schema does not match the active transaction topology."
        )
    payload = {
        "schema": FACTOR_PACK_SCHEMA if lexical_rank is None else FACTOR_PACK_V2_SCHEMA,
        "reason": reason,
        "preregistration_sha256": contract.file_sha256,
        "evaluation_contract_sha256": contract.evaluation_contract_sha256,
        "qkvo_site_count": QKVO_SITE_COUNT,
        "qkvo_rank": QKVO_RANK,
        "swiglu_layer_count": SWIGLU_LAYER_COUNT,
        "swiglu_rank": SWIGLU_RANK,
        "factor_pack": dict(factor_payload),
        "contains_model_checkpoint": False,
        "contains_parent_parameters": False,
        "contains_activations_or_hidden_states": False,
    }
    if lexical_rank is not None:
        payload["lexical_rank"] = lexical_rank
    return payload


def _trainer_evidence(training_result: Any) -> tuple[str, dict[str, Any]]:
    """Authenticate and return the exact JSON evidence hash preimage."""

    method = getattr(training_result, "evidence_payload", None)
    if not callable(method):
        raise ExactTargetNativeStageError(
            "Trainer result did not expose its canonical evidence preimage."
        )
    raw = method()
    if not isinstance(raw, Mapping):
        raise ExactTargetNativeStageError(
            "Trainer evidence preimage must be a JSON mapping."
        )
    payload = dict(raw)
    actual = _canonical_sha256(payload)
    declared = _require_sha256(
        getattr(training_result, "evidence_sha256", None),
        "training_result.evidence_sha256",
    )
    if actual != declared:
        raise ExactTargetNativeStageError(
            "Trainer evidence SHA does not authenticate its persisted preimage."
        )
    required = {
        "checkpoints",
        "selected_step",
        "selection_policy",
        "fit_metrics",
        "candidate_metrics",
        "sealed_metrics",
        "training_audit",
        "factor_pack_manifest",
        "domain_binding_sha256",
        "schedule_sha256",
        "candidate_thresholds",
        "candidate_gate_passed",
        "sealed_evaluation_unlocked",
    }
    if not required.issubset(payload):
        raise ExactTargetNativeStageError(
            f"Trainer evidence omits {sorted(required - set(payload))}."
        )
    if payload.get("internal_numeric_matching_used") is not False:
        raise ExactTargetNativeStageError(
            "Trainer evidence does not prove output-only selection."
        )
    return declared, payload


def _factor_acquisition_claim(evidence: Mapping[str, Any]) -> dict[str, Any]:
    checkpoints = evidence.get("checkpoints")
    selected_step = evidence.get("selected_step")
    manifest = evidence.get("factor_pack_manifest")
    if (
        not isinstance(checkpoints, list)
        or isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or not isinstance(manifest, Mapping)
    ):
        raise ExactTargetNativeStageError(
            "Trainer evidence has invalid checkpoint/factor acquisition records."
        )
    by_step = {
        value.get("step"): value
        for value in checkpoints
        if isinstance(value, Mapping) and isinstance(value.get("step"), int)
    }
    if 0 not in by_step or selected_step not in by_step:
        raise ExactTargetNativeStageError(
            "Trainer evidence omits step zero or the selected checkpoint."
        )
    step0_sha = _require_sha256(
        by_step[0].get("factor_state_sha256"),
        "trainer_evidence.step0.factor_state_sha256",
    )
    selected_sha = _require_sha256(
        manifest.get("factor_state_sha256"),
        "trainer_evidence.factor_pack_manifest.factor_state_sha256",
    )
    checkpoint_selected_sha = _require_sha256(
        by_step[selected_step].get("factor_state_sha256"),
        "trainer_evidence.selected.factor_state_sha256",
    )
    if checkpoint_selected_sha != selected_sha:
        raise ExactTargetNativeStageError(
            "Selected checkpoint and factor-pack evidence hashes disagree."
        )
    acquired = selected_step > 0 and selected_sha != step0_sha
    return {
        "selected_step": selected_step,
        "step0_factor_state_sha256": step0_sha,
        "selected_factor_state_sha256": selected_sha,
        "target_native_factors_acquired_behavior": acquired,
        "residual_learning_claimed": acquired,
        "step0_bootstrap_may_pass_standalone_gate": True,
        "selected_factor_pack_schema": manifest.get("schema"),
        "selected_factor_count": manifest.get("factor_count"),
        "selected_lexical_rank": manifest.get("lexical_rank"),
    }


def _validate_continuation_compilation(
    contract: AuthenticatedPreRegistration,
    compiler_report: Mapping[str, Any],
    compiled_state_sha256: str,
) -> None:
    continuation = contract.continuation
    if continuation is None:
        return
    if _canonical_sha256(compiler_report) != continuation.compiler_report_sha256:
        raise ExactTargetNativeStageError(
            "Current structural compiler report differs from the continuation source."
        )
    if compiled_state_sha256 != continuation.compiled_state_sha256:
        raise ExactTargetNativeStageError(
            "Current compiled-child bytes differ from the continuation source."
        )


def _remove_published_child_tree(path: Path, output_root: Path) -> None:
    """Remove only this run's exact promoted path after report failure."""

    resolved = path.resolve()
    root = output_root.resolve()
    if resolved.parent != root or resolved.name != "accepted_child":
        raise ExactTargetNativeStageError(
            "Refusing to remove a published tree outside the run output root."
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def _output_jacobian_audit_artifacts(
    execution: Any,
    contract: AuthenticatedPreRegistration,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(execution, OutputJacobianAuditExecution):
        raise ExactTargetNativeStageError(
            "Runtime did not return the authenticated output-Jacobian execution "
            "envelope."
        )
    continuation = contract.continuation
    if (
        continuation is None
        or execution.restore_mode != "full"
        or execution.restored_factor_schema != TRAINER_FACTOR_PACK_V2_SCHEMA
        or execution.restored_factor_state_sha256 != continuation.factor_state_sha256
    ):
        raise ExactTargetNativeStageError(
            "Output-Jacobian execution did not prove a full authenticated v2 "
            "factor restore."
        )
    report_object = getattr(execution.result, "report", None)
    report_method = getattr(report_object, "to_dict", None)
    if not callable(report_method):
        raise ExactTargetNativeStageError(
            "Output-Jacobian result did not expose its canonical gate report."
        )
    audit_report = report_method()
    if not isinstance(audit_report, Mapping):
        raise ExactTargetNativeStageError("Output-Jacobian gate report is malformed.")
    canonical_report = dict(audit_report)
    _canonical_json(canonical_report)
    binding = contract.output_jacobian_audit
    if binding is None:
        raise ExactTargetNativeStageError(
            "Output-Jacobian report is not bound to an authenticated audit."
        )
    required_report_fields = {
        "schema",
        "passed",
        "config",
        "config_sha256",
        "site_count",
        "role_count",
        "sketch_count",
        "basis_prompt_count",
        "basis_source_role",
        "source_exposed_candidate_as_fit",
        "new_candidate_accessed",
        "sealed_accessed",
        "optimizer_created",
        "optimizer_used",
        "optimizer_steps",
        "hooks_registered",
        "hooks_removed",
        "active_hook_count_after",
        "transaction_live_after",
        "native_state_sha256_before",
        "native_state_sha256_after",
        "inherited_qkvo_state_sha256_before",
        "inherited_qkvo_state_sha256_after",
        "factor_state_sha256_before",
        "factor_state_sha256_after",
        "factor_parameter_versions_sha256_before",
        "factor_parameter_versions_sha256_after",
        "factor_gradients_cleared",
        "pass1_replay_sha256",
        "pass2_replay_sha256",
        "replay_match",
        "accounted_global_peak_bytes",
        "max_bytes",
        "aggregate",
        "per_role",
        "basis_manifest_sha256",
        "report_sha256",
    }
    if not required_report_fields <= set(canonical_report):
        raise ExactTargetNativeStageError(
            "Output-Jacobian gate report omits required audit attestations."
        )
    expected_config = {
        "prompt_count": binding.prompt_count,
        "horizon": binding.horizon,
        "lanes": list(binding.lanes),
        "rank": binding.rank,
        "sketch_width": binding.sketch_width,
        "seed": contract.seed,
        "min_normal_fraction": binding.min_normal_fraction,
        "min_guided_over_random_ratio": binding.min_guided_over_random_ratio,
        "max_bytes": binding.max_bytes,
    }
    config = canonical_report.get("config")
    if (
        not isinstance(config, Mapping)
        or _canonical_json(dict(config)) != _canonical_json(expected_config)
        or canonical_report.get("config_sha256") != _canonical_sha256(config)
    ):
        raise ExactTargetNativeStageError(
            "Output-Jacobian report configuration does not match its authenticated "
            "contract/hash."
        )
    fixed_attestations = {
        "schema": OUTPUT_JACOBIAN_AUDIT_SCHEMA,
        "site_count": QKVO_SITE_COUNT,
        "role_count": 4,
        "sketch_count": QKVO_SITE_COUNT * 4,
        "basis_prompt_count": binding.prompt_count,
        "basis_source_role": "source-exposed-candidate",
        "source_exposed_candidate_as_fit": True,
        "new_candidate_accessed": False,
        "sealed_accessed": False,
        "optimizer_created": False,
        "optimizer_used": False,
        "optimizer_steps": 0,
        "hooks_registered": QKVO_SITE_COUNT,
        "hooks_removed": True,
        "active_hook_count_after": 0,
        "transaction_live_after": True,
        "factor_gradients_cleared": True,
        "replay_match": True,
        "max_bytes": binding.max_bytes,
    }
    if any(
        _canonical_json(canonical_report.get(name)) != _canonical_json(value)
        for name, value in fixed_attestations.items()
    ):
        raise ExactTargetNativeStageError(
            "Output-Jacobian report failed an audit-only lifecycle attestation."
        )
    if not isinstance(canonical_report.get("passed"), bool):
        raise ExactTargetNativeStageError(
            "Output-Jacobian report lacks a boolean diagnostic decision."
        )
    hash_pairs = (
        ("native_state_sha256_before", "native_state_sha256_after"),
        (
            "inherited_qkvo_state_sha256_before",
            "inherited_qkvo_state_sha256_after",
        ),
        ("factor_state_sha256_before", "factor_state_sha256_after"),
        (
            "factor_parameter_versions_sha256_before",
            "factor_parameter_versions_sha256_after",
        ),
        ("pass1_replay_sha256", "pass2_replay_sha256"),
    )
    for before_name, after_name in hash_pairs:
        before = _require_sha256(
            canonical_report.get(before_name),
            f"output_jacobian_audit_report.{before_name}",
        )
        after = _require_sha256(
            canonical_report.get(after_name),
            f"output_jacobian_audit_report.{after_name}",
        )
        if before != after:
            raise ExactTargetNativeStageError(
                "Output-Jacobian audit changed authenticated state or replay inputs."
            )
    if canonical_report["factor_state_sha256_before"] != (
        continuation.factor_state_sha256
    ):
        raise ExactTargetNativeStageError(
            "Output-Jacobian audit did not retain the restored continuation state."
        )
    peak = canonical_report.get("accounted_global_peak_bytes")
    if (
        isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak <= 0
        or peak > binding.max_bytes
    ):
        raise ExactTargetNativeStageError(
            "Output-Jacobian audit exceeded or malformed its global byte ceiling."
        )
    aggregate = canonical_report.get("aggregate")
    per_role = canonical_report.get("per_role")
    if (
        not isinstance(aggregate, Mapping)
        or not isinstance(per_role, Mapping)
        or set(per_role) != {"q", "k", "v", "o"}
    ):
        raise ExactTargetNativeStageError(
            "Output-Jacobian report omits aggregate or per-role gate evidence."
        )
    normal_fraction = aggregate.get("normal_fraction")
    capture_ratio = aggregate.get("guided_over_random_ratio")
    if (
        isinstance(normal_fraction, bool)
        or not isinstance(normal_fraction, (int, float))
        or not math.isfinite(float(normal_fraction))
        or float(normal_fraction) < 0.0
        or isinstance(capture_ratio, bool)
        or not isinstance(capture_ratio, (int, float))
        or not math.isfinite(float(capture_ratio))
        or float(capture_ratio) < 0.0
    ):
        raise ExactTargetNativeStageError(
            "Output-Jacobian aggregate gate metrics are malformed or non-finite."
        )
    expected_normal_gate = float(normal_fraction) >= binding.min_normal_fraction
    expected_ratio_gate = (
        float(capture_ratio) >= binding.min_guided_over_random_ratio
    )
    expected_passed = expected_normal_gate and expected_ratio_gate
    if (
        aggregate.get("normal_fraction_gate_passed") is not expected_normal_gate
        or aggregate.get("guided_over_random_gate_passed") is not expected_ratio_gate
        or aggregate.get("passed") is not expected_passed
        or canonical_report.get("passed") is not expected_passed
    ):
        raise ExactTargetNativeStageError(
            "Output-Jacobian decision disagrees with the authenticated aggregate "
            "thresholds."
        )
    _require_sha256(
        canonical_report.get("basis_manifest_sha256"),
        "output_jacobian_audit_report.basis_manifest_sha256",
    )
    declared_report_sha = _require_sha256(
        canonical_report.get("report_sha256"),
        "output_jacobian_audit_report.report_sha256",
    )
    report_preimage = {
        name: value
        for name, value in canonical_report.items()
        if name != "report_sha256"
    }
    if _canonical_sha256(report_preimage) != declared_report_sha:
        raise ExactTargetNativeStageError(
            "Output-Jacobian report self-hash does not authenticate its evidence."
        )
    payload_method = getattr(execution.result, "basis_pack_payload", None)
    if not callable(payload_method):
        raise ExactTargetNativeStageError(
            "Output-Jacobian result did not expose a guided basis pack."
        )
    basis_payload = payload_method()
    if not isinstance(basis_payload, Mapping) or not basis_payload:
        raise ExactTargetNativeStageError(
            "Output-Jacobian guided basis pack is malformed."
        )
    expected_pack_fields = {
        "schema",
        "factor_state_sha256",
        "basis_manifest_sha256",
        "tensor_count",
        "total_numel",
        "total_nbytes",
        "factors",
        "factor_tensors",
        "contains_model_state",
        "contains_parent_state",
        "contains_optimizer_state",
        "contains_gradients",
        "contains_activations",
        "contains_hidden_states",
        "authorized_for_training",
    }
    if set(basis_payload) != expected_pack_fields:
        raise ExactTargetNativeStageError(
            "Guided basis pack does not contain exactly the factor-only schema."
        )
    if basis_payload.get("schema") != GUIDED_QKVO_RIGHT_BASIS_SCHEMA:
        raise ExactTargetNativeStageError(
            "Guided basis pack schema is not the source-pinned QKVO basis schema."
        )
    if any(
        basis_payload.get(name) is not False
        for name in (
            "contains_model_state",
            "contains_parent_state",
            "contains_optimizer_state",
            "contains_gradients",
            "contains_activations",
            "contains_hidden_states",
        )
    ):
        raise ExactTargetNativeStageError(
            "Guided basis pack contains banned model, parent, optimizer, gradient, "
            "activation, or hidden state."
        )
    if basis_payload.get("authorized_for_training") is not canonical_report.get(
        "passed"
    ):
        raise ExactTargetNativeStageError(
            "Guided basis training authorization disagrees with the diagnostic gate."
        )
    if basis_payload.get("factor_state_sha256") != (continuation.factor_state_sha256):
        raise ExactTargetNativeStageError(
            "Guided basis pack is not bound to the restored continuation state."
        )
    tensor_specs = (
        ("q_orthogonal_right", (QKVO_SITE_COUNT, QKVO_RANK, TARGET_HIDDEN_SIZE)),
        ("k_orthogonal_right", (QKVO_SITE_COUNT, QKVO_RANK, TARGET_HIDDEN_SIZE)),
        ("v_orthogonal_right", (QKVO_SITE_COUNT, QKVO_RANK, TARGET_HIDDEN_SIZE)),
        (
            "o_orthogonal_right",
            (QKVO_SITE_COUNT, QKVO_RANK, TARGET_QUERY_GROUP_WIDTH),
        ),
    )
    tensors = basis_payload.get("factor_tensors")
    rows = basis_payload.get("factors")
    expected_names = tuple(name for name, _shape in tensor_specs)
    if (
        not isinstance(tensors, Mapping)
        or set(tensors) != set(expected_names)
        or not isinstance(rows, list)
        or len(rows) != len(tensor_specs)
    ):
        raise ExactTargetNativeStageError(
            "Guided basis pack does not contain the ordered Q/K/V/O tensors."
        )
    canonical_rows: list[dict[str, Any]] = []
    total_numel = 0
    total_nbytes = 0
    for (name, shape), row in zip(tensor_specs, rows, strict=True):
        value = tensors[name]
        if (
            not isinstance(value, Tensor)
            or value.layout != torch.strided
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad is not None
            or tuple(value.shape) != shape
            or not bool(torch.isfinite(value).all())
        ):
            raise ExactTargetNativeStageError(
                f"Guided basis tensor {name!r} is not exact finite contiguous CPU FP32."
            )
        tensor_numel = int(value.numel())
        tensor_nbytes = tensor_numel * int(value.element_size())
        expected_row = {
            "name": name,
            "shape": list(shape),
            "dtype": "torch.float32",
            "tensor_sha256": _output_jacobian_tensor_sha256(value),
            "numel": tensor_numel,
            "nbytes": tensor_nbytes,
        }
        if not isinstance(row, Mapping) or _canonical_json(
            dict(row)
        ) != _canonical_json(expected_row):
            raise ExactTargetNativeStageError(
                f"Guided basis manifest row {name!r} failed shape/hash binding."
            )
        canonical_rows.append(expected_row)
        total_numel += tensor_numel
        total_nbytes += tensor_nbytes
    declared_totals = (
        basis_payload.get("tensor_count"),
        basis_payload.get("total_numel"),
        basis_payload.get("total_nbytes"),
    )
    if (
        total_numel != 4_915_200
        or total_nbytes != 19_660_800
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in declared_totals
        )
        or declared_totals != (4, total_numel, total_nbytes)
    ):
        raise ExactTargetNativeStageError(
            "Guided QKVO basis tensor totals differ from exact stage 1."
        )
    basis_manifest_sha = _canonical_sha256(canonical_rows)
    if (
        basis_payload.get("basis_manifest_sha256") != basis_manifest_sha
        or canonical_report.get("basis_manifest_sha256") != basis_manifest_sha
    ):
        raise ExactTargetNativeStageError(
            "Guided basis manifest hash disagrees with its tensors or audit report."
        )
    return canonical_report, basis_payload


def _guided_basis_envelope(
    basis_payload: Mapping[str, Any],
    audit_report: Mapping[str, Any],
    contract: AuthenticatedPreRegistration,
) -> dict[str, Any]:
    binding = contract.output_jacobian_audit
    continuation = contract.continuation
    if binding is None or continuation is None:
        raise ExactTargetNativeStageError(
            "Cannot bind a guided basis pack outside an authenticated audit."
        )
    return {
        "schema": GUIDED_BASIS_PACK_SCHEMA,
        "preregistration_sha256": contract.file_sha256,
        "evaluation_contract_sha256": contract.evaluation_contract_sha256,
        "output_jacobian_audit": binding.as_dict(),
        "output_jacobian_audit_report_sha256": _canonical_sha256(audit_report),
        "source_factor_state_sha256": continuation.factor_state_sha256,
        "source_factor_pack_schema": continuation.trainer_factor_pack_schema,
        "basis_pack": dict(basis_payload),
        "contains_model_checkpoint": False,
        "contains_parent_parameters": False,
        "contains_optimizer_state": False,
        "contains_activations_or_hidden_states": False,
    }


def execute_authenticated_stage(
    contract: AuthenticatedPreRegistration,
    output_root: Path | str,
    *,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Execute one authenticated transaction; never launch from preflight."""

    runner = ProductionStageRuntime() if runtime is None else runtime
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise ExactTargetNativeStageError("output_root must not already exist.")
    output.mkdir(parents=True)
    parent: Any = None
    target: Any = None
    tokenizer: Any = None
    transaction: Any = None
    compiled: Any = None
    compiled_state_sha: str | None = None
    training_result: Any = None
    published_child = False
    report: dict[str, Any] = {
        **contract.preflight_report(),
        "status": "running",
        "output_root": str(output),
    }
    try:
        prepare_cache = getattr(runner, "prepare_compiler_cache", None)
        if callable(prepare_cache):
            cache_plan = prepare_cache(contract)
            if cache_plan is not None:
                report["compiled_child_cache"] = dict(cache_plan)
                report["last_completed_phase"] = "compiler-cache-authentication"
        parent, target, tokenizer = runner.load_models(contract)
        report.update(
            {
                "models_loaded": True,
                "gpu_work_launched": True,
                "last_completed_phase": "authenticated-model-load",
            }
        )
        compiled = runner.compile(parent, target, tokenizer, contract)
        compiler_report = _compiled_report_dict(compiled)
        parent_evidence_partition = _compiler_partition_report(contract)
        if (
            compiler_report.get("bootstrap_accepted") is not True
            or compiler_report.get("target_budget_exact") is not True
            or compiler_report.get("target_unique_parameter_count")
            != QWEN3_1P4B_EXACT_PARAMETERS
        ):
            raise ExactTargetNativeStageError(
                "Canonical parent compilation did not produce an accepted exact child."
            )
        compiled_state_sha = runner.state_sha256(compiled.child_model)
        declared_compiled_state = getattr(compiled, "compiled_state_sha256", None)
        if (
            declared_compiled_state is not None
            and declared_compiled_state != compiled_state_sha
        ):
            raise ExactTargetNativeStageError(
                "Compiled-base state changed before factor installation."
            )
        _validate_continuation_compilation(
            contract,
            compiler_report,
            compiled_state_sha,
        )
        # An immutable cache miss may be published only after the live compile
        # has passed every structural and continuation-source compatibility gate.
        finalize_compiled = getattr(runner, "finalize_compiled_base", None)
        if callable(finalize_compiled):
            finalized = finalize_compiled(compiled, contract)
            finalized_report = _compiled_report_dict(finalized)
            finalized_state_sha = runner.state_sha256(finalized.child_model)
            if (
                finalized_report != compiler_report
                or finalized_state_sha != compiled_state_sha
            ):
                raise ExactTargetNativeStageError(
                    "Compiler cache finalization changed the validated compiled base."
                )
            compiled = finalized
        cache_evidence = getattr(compiled, "cache_evidence", None)
        report.update(
            {
                "compiler_report": compiler_report,
                "canonical_parent_evidence": parent_evidence_partition,
                **(
                    {}
                    if not isinstance(cache_evidence, Mapping)
                    else {"compiled_child_cache": dict(cache_evidence)}
                ),
                "last_completed_phase": "structural-compiler-bootstrap",
            }
        )
        transaction = runner.install_transaction(compiled)
        installation = transaction.installation_report()
        report.update(
            {
                "installation": installation,
                "last_completed_phase": "target-native-factor-installation",
            }
        )
        if (
            installation.get("zero_step_identity") is not True
            or installation.get("temporary_qkvo_factor_parameters", 0) <= 0
            or installation.get("temporary_gate_up_down_factor_parameters", 0) <= 0
            or (
                contract.lexical_residual is not None
                and (
                    installation.get("temporary_lexical_factor_parameters", 0)
                    != LEXICAL_FACTOR_PARAMETERS
                    or installation.get("role_complete_lexical_boundary") is not True
                    or installation.get("lexical_embedding_and_lm_head_untied")
                    is not True
                )
            )
        ):
            raise ExactTargetNativeStageError(
                "Combined target-native transaction did not install as exact identity."
            )
        if contract.output_jacobian_audit is not None:
            audit_method = getattr(runner, "run_output_jacobian_audit", None)
            if not callable(audit_method):
                raise ExactTargetNativeStageError(
                    "Runtime does not implement the authenticated output-Jacobian "
                    "audit branch."
                )
            report["last_completed_phase"] = "output-jacobian-audit-started"
            audit_execution = audit_method(transaction, parent, contract)
            audit_report, basis_payload = _output_jacobian_audit_artifacts(
                audit_execution,
                contract,
            )
            audit_report_sha = _canonical_sha256(audit_report)
            report["last_completed_phase"] = "output-jacobian-audit-complete"
            transaction.rollback()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_state_sha:
                raise ExactTargetNativeStageError(
                    "Output-Jacobian audit did not restore the exact compiled child "
                    "bytes."
                )
            basis_path = output / "guided_basis_pack.pt"
            try:
                basis_sha = _atomic_torch_save(
                    basis_path,
                    _guided_basis_envelope(
                        basis_payload,
                        audit_report,
                        contract,
                    ),
                )
                basis_size = int(basis_path.stat().st_size)
            except BaseException:
                basis_path.unlink(missing_ok=True)
                raise
            maximum_bytes = contract.output_jacobian_audit.max_bytes
            if basis_size > maximum_bytes:
                basis_path.unlink(missing_ok=True)
                raise ExactTargetNativeStageError(
                    "Guided basis pack exceeds the authenticated byte ceiling."
                )
            diagnostic_passed = bool(audit_report["passed"])
            diagnostic_status = (
                "diagnostic-passed" if diagnostic_passed else "diagnostic-rejected"
            )
            report.update(
                {
                    "status": diagnostic_status,
                    "reason": (
                        "output-jacobian-audit-passed"
                        if diagnostic_passed
                        else "output-jacobian-audit-rejected"
                    ),
                    "output_jacobian_audit_report": audit_report,
                    "output_jacobian_audit_report_sha256": audit_report_sha,
                    "output_jacobian_restore": {
                        "mode": audit_execution.restore_mode,
                        "factor_pack_schema": (audit_execution.restored_factor_schema),
                        "factor_state_sha256": (
                            audit_execution.restored_factor_state_sha256
                        ),
                        "matches_authenticated_continuation": True,
                    },
                    "guided_basis_pack": {
                        "path": str(basis_path),
                        "sha256": basis_sha,
                        "size_bytes": basis_size,
                        "maximum_bytes": maximum_bytes,
                    },
                    "source_candidate_exposed_as_fit": True,
                    "sealed_not_accessed": True,
                    "training_started": False,
                    "optimizer_created": False,
                    "optimizer_steps": 0,
                    "training_checkpoints_written": 0,
                    "native_factor_bake_performed": False,
                    "model_checkpoint_written": False,
                    "standalone_compaction_passed": False,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "last_completed_phase": "output-jacobian-audit-exact-rollback",
                    "retained_artifacts": [
                        "guided_basis_pack.pt",
                        "run_report.json",
                    ],
                }
            )
            try:
                _atomic_json(output / "run_report.json", report)
            except BaseException:
                basis_path.unlink(missing_ok=True)
                raise
            return report
        report["last_completed_phase"] = "behavior-training-started"
        training_result = runner.train(transaction, parent, contract)
        evidence_sha, trainer_evidence = _trainer_evidence(training_result)
        if contract.continuation is not None and (
            trainer_evidence.get("continued_from_factor_pack") is not True
            or trainer_evidence.get("initial_factor_state_sha256")
            != contract.continuation.factor_state_sha256
        ):
            raise ExactTargetNativeStageError(
                "Trainer did not prove restoration of the authenticated continuation."
            )
        if contract.continuation is not None and contract.lexical_residual is not None:
            selected_manifest = trainer_evidence.get("factor_pack_manifest")
            if (
                not isinstance(selected_manifest, Mapping)
                or selected_manifest.get("schema") != TRAINER_FACTOR_PACK_V2_SCHEMA
                or selected_manifest.get("factor_count")
                != LEXICAL_TARGET_NATIVE_FACTOR_COUNT
                or selected_manifest.get("lexical_rank")
                != contract.lexical_residual.rank
            ):
                raise ExactTargetNativeStageError(
                    "Trainer did not emit the complete rank-bound 18-factor v2 pack."
                )
            checkpoints = trainer_evidence.get("checkpoints")
            step0 = (
                next(
                    (
                        value
                        for value in checkpoints
                        if isinstance(value, Mapping) and value.get("step") == 0
                    ),
                    None,
                )
                if isinstance(checkpoints, list)
                else None
            )
            if not isinstance(step0, Mapping) or trainer_evidence.get(
                "combined_step0_factor_state_sha256"
            ) != step0.get("factor_state_sha256"):
                raise ExactTargetNativeStageError(
                    "Trainer did not bind the combined lexical step-zero factor state."
                )
            source_schema = contract.continuation.trainer_factor_pack_schema
            if source_schema == TRAINER_FACTOR_PACK_SCHEMA:
                lexical_initialization_sha = trainer_evidence.get(
                    "lexical_initialization_state_sha256"
                )
                if (
                    trainer_evidence.get("initial_factor_payload_schema")
                    != TRAINER_FACTOR_PACK_SCHEMA
                    or trainer_evidence.get("initial_factor_restore_mode")
                    != "qkvo-gud-partial"
                    or trainer_evidence.get("lexical_initialization_unchanged")
                    is not True
                    or not isinstance(lexical_initialization_sha, str)
                    or not _SHA256.fullmatch(lexical_initialization_sha)
                ):
                    raise ExactTargetNativeStageError(
                        "Trainer did not prove an unchanged lexical initialization "
                        "during partial v1 continuation restore."
                    )
            elif (
                source_schema != TRAINER_FACTOR_PACK_V2_SCHEMA
                or contract.continuation.lexical_rank != contract.lexical_residual.rank
                or trainer_evidence.get("initial_factor_payload_schema")
                != TRAINER_FACTOR_PACK_V2_SCHEMA
                or trainer_evidence.get("initial_factor_restore_mode") != "full"
            ):
                raise ExactTargetNativeStageError(
                    "Trainer did not prove a full rank-bound lexical v2 restore."
                )
        factor_acquisition = _factor_acquisition_claim(trainer_evidence)
        fit_passed, fit_decision = _behavior_gate(
            _training_metric(training_result, "fit_metrics"),
            contract.thresholds["fit"],
        )
        candidate_passed, candidate_decision = _behavior_gate(
            _training_metric(training_result, "candidate_metrics"),
            contract.thresholds["candidate"],
        )
        trainer_candidate = getattr(training_result, "candidate_gate_passed", None)
        if (
            trainer_candidate is not None
            and bool(trainer_candidate) != candidate_passed
        ):
            raise ExactTargetNativeStageError(
                "Trainer and runner disagree on the preregistered candidate gate."
            )
        sealed_metrics = _training_metric(training_result, "sealed_metrics")
        sealed_unlocked = bool(
            getattr(
                training_result,
                "sealed_evaluation_unlocked",
                sealed_metrics is not None,
            )
        )
        if candidate_passed and not sealed_unlocked:
            raise ExactTargetNativeStageError(
                "Passing candidate metrics did not unlock the sealed evaluation."
            )
        if not candidate_passed and (sealed_unlocked or sealed_metrics is not None):
            raise ExactTargetNativeStageError(
                "Sealed evaluation was accessed before the candidate gate passed."
            )
        sealed_passed = False
        sealed_decision: dict[str, Any] | None = None
        if candidate_passed:
            sealed_passed, sealed_decision = _behavior_gate(
                sealed_metrics,
                contract.thresholds["sealed"],
            )
        pre_bake_decisions = {
            "fit": fit_decision,
            "candidate": candidate_decision,
            "sealed": sealed_decision,
        }
        report.update(
            {
                # Keep the legacy field as the trainer/pre-bake decision while
                # making the two publication gates independently auditable.
                "behavior_decisions": pre_bake_decisions,
                "pre_bake_behavior_decisions": pre_bake_decisions,
                "post_bake_behavior_decisions": None,
            }
        )
        accepted = fit_passed and candidate_passed and sealed_passed
        if not accepted:
            reason = next(
                name
                for name, passed in (
                    ("fit-gate-rejected", fit_passed),
                    ("candidate-gate-rejected", candidate_passed),
                    ("sealed-gate-rejected", sealed_passed),
                )
                if not passed
            )
            factor_payload = runner.factor_pack_payload(training_result)
            factor_path = output / "rejected_factor_pack.pt"
            factor_sha = _atomic_torch_save(
                factor_path,
                _factor_envelope(factor_payload, contract, reason),
            )
            transaction.rollback()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_state_sha:
                raise ExactTargetNativeStageError(
                    "Rejected transaction did not restore the compiled child bytes."
                )
            report.update(
                {
                    "status": "rejected",
                    "reason": reason,
                    "compiler_report": compiler_report,
                    "canonical_parent_evidence": parent_evidence_partition,
                    "installation": installation,
                    "trainer_evidence_sha256": evidence_sha,
                    "trainer_evidence": trainer_evidence,
                    "factor_acquisition": factor_acquisition,
                    "standalone_compaction_passed": False,
                    "factor_pack": {
                        "path": str(factor_path),
                        "sha256": factor_sha,
                    },
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "model_checkpoint_written": False,
                    "retained_artifacts": [
                        "rejected_factor_pack.pt",
                        "run_report.json",
                    ],
                }
            )
            _atomic_json(output / "run_report.json", report)
            return report
        sealed_payload = _metrics_dict(sealed_metrics)
        behavior_receipt = ExternalBehaviorGateReceipt(
            evidence_sha256=evidence_sha,
            evaluation_contract_sha256=contract.evaluation_contract_sha256,
            heldout_action_count=int(sealed_payload["action_count"]),
            free_running_rollout_count=int(sealed_payload["rollout_count"]),
            passed=True,
            internal_similarity_only=False,
        )
        factor_payload = runner.factor_pack_payload(training_result)
        publication = transaction.bake_for_publication(behavior_receipt)
        post_bake_decisions: dict[str, dict[str, Any] | None] = {
            "fit": None,
            "candidate": None,
            "sealed": None,
        }
        report.update(
            {
                "publication": publication,
                "post_bake_behavior_decisions": post_bake_decisions,
                "last_completed_phase": "native-factor-bake",
            }
        )
        baked_evaluator = getattr(runner, "evaluate_baked_behavior", None)
        if not callable(baked_evaluator):
            raise ExactTargetNativeStageError(
                "Runtime does not implement the mandatory post-bake behavior gate."
            )
        post_bake_rejection_reason: str | None = None
        for split_name in ("fit", "candidate", "sealed"):
            # This loop is reachable only after the preregistered candidate gate
            # unlocked and passed the sealed evaluation.  On the baked bytes, a
            # candidate failure again keeps sealed evidence inaccessible.
            if split_name == "sealed" and (
                post_bake_decisions["candidate"] is None
                or post_bake_decisions["candidate"]["passed"] is not True
            ):
                break
            baked_metrics = baked_evaluator(
                compiled.child_model,
                parent,
                contract,
                split_name,
            )
            baked_passed, baked_decision = _behavior_gate(
                baked_metrics,
                contract.thresholds[split_name],
            )
            post_bake_decisions[split_name] = baked_decision
            report["last_completed_phase"] = f"post-bake-{split_name}-evaluation"
            if not baked_passed:
                post_bake_rejection_reason = f"post-bake-{split_name}-gate-rejected"
                break
        if post_bake_rejection_reason is not None:
            factor_path = output / "rejected_factor_pack.pt"
            factor_sha = _atomic_torch_save(
                factor_path,
                _factor_envelope(
                    factor_payload,
                    contract,
                    post_bake_rejection_reason,
                ),
            )
            restore_prepublication = getattr(
                transaction, "restore_prepublication_native", None
            )
            if not callable(restore_prepublication):
                raise ExactTargetNativeStageError(
                    "A rejected post-bake gate cannot restore native child bytes."
                )
            restore_prepublication()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_state_sha:
                raise ExactTargetNativeStageError(
                    "Post-bake rejection did not restore the compiled child bytes."
                )
            report.update(
                {
                    "status": "rejected",
                    "reason": post_bake_rejection_reason,
                    "compiler_report": compiler_report,
                    "canonical_parent_evidence": parent_evidence_partition,
                    "installation": installation,
                    "trainer_evidence_sha256": evidence_sha,
                    "trainer_evidence": trainer_evidence,
                    "factor_acquisition": factor_acquisition,
                    "standalone_compaction_passed": False,
                    "factor_pack": {
                        "path": str(factor_path),
                        "sha256": factor_sha,
                    },
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "prepublication_native_restored": True,
                    "model_checkpoint_written": False,
                    "retained_artifacts": [
                        "rejected_factor_pack.pt",
                        "run_report.json",
                    ],
                    "last_completed_phase": "post-bake-rejection-restoration",
                }
            )
            _atomic_json(output / "run_report.json", report)
            return report
        certificate = runner.save_reload_certify(
            compiled.child_model, tokenizer, output
        )
        published_child = True
        report.update(
            {
                "status": "accepted",
                "compiler_report": compiler_report,
                "canonical_parent_evidence": parent_evidence_partition,
                "installation": installation,
                "pre_bake_behavior_decisions": pre_bake_decisions,
                "post_bake_behavior_decisions": post_bake_decisions,
                "behavior_evidence_sha256": evidence_sha,
                "trainer_evidence_sha256": evidence_sha,
                "trainer_evidence": trainer_evidence,
                "factor_acquisition": factor_acquisition,
                "standalone_compaction_passed": True,
                "publication": publication,
                "fresh_reload_certificate": certificate,
                "rollback_exact": None,
                "model_checkpoint_written": True,
                "model_save_count": 1,
                "selection_used_internal_numeric_matching": False,
            }
        )
        _atomic_json(output / "run_report.json", report)
        return report
    except BaseException as error:
        rollback_error: str | None = None
        restored_prepublication_native = False
        restore_prepublication = (
            None
            if transaction is None
            else getattr(transaction, "restore_prepublication_native", None)
        )
        if callable(restore_prepublication):
            try:
                restore_prepublication()
                if (
                    compiled is not None
                    and compiled_state_sha is not None
                    and runner.state_sha256(compiled.child_model) != compiled_state_sha
                ):
                    raise ExactTargetNativeStageError(
                        "Prepublication restoration did not recover compiled bytes."
                    )
                restored_prepublication_native = True
            except BaseException as rollback_failure:  # pragma: no cover - hard failure
                rollback_error = repr(rollback_failure)
        elif transaction is not None and getattr(transaction, "is_live", False):
            try:
                transaction.rollback()
            except BaseException as rollback_failure:  # pragma: no cover - hard failure
                rollback_error = repr(rollback_failure)
        if published_child or (output / "accepted_child").exists():
            try:
                _remove_published_child_tree(output / "accepted_child", output)
                published_child = False
            except BaseException as cleanup_failure:  # pragma: no cover - hard failure
                prefix = "" if rollback_error is None else f"{rollback_error}; "
                rollback_error = (
                    f"{prefix}published-child cleanup failed: {cleanup_failure!r}"
                )
        failure = {
            **report,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "rollback_error": rollback_error,
            "prepublication_native_restored": restored_prepublication_native,
            "model_checkpoint_promoted": False,
        }
        try:
            if not (output / "run_report.json").exists():
                _atomic_json(output / "run_report.json", failure)
        except BaseException:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--preflight",
        "--dry-run",
        dest="preflight",
        action="store_true",
        help="Authenticate and print the exact plan without loading models or GPUs.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--parent-device-map", default="auto")
    parser.add_argument("--target-device-map", default="auto")
    parser.add_argument(
        "--trainer-module",
        default=CANONICAL_TRAINER_MODULE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = authenticate_preregistration(
        args.preregistration,
        args.preregistration_sha256,
    )
    if args.preflight:
        print(json.dumps(contract.preflight_report(), indent=2, sort_keys=True))
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root is required outside --preflight/--dry-run.")
    runtime = ProductionStageRuntime(
        dtype=args.dtype,
        parent_device_map=args.parent_device_map,
        target_device_map=args.target_device_map,
        trainer_module=args.trainer_module,
    )
    result = execute_authenticated_stage(
        contract,
        args.output_root,
        runtime=runtime,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if result["status"]
        in {
            "accepted",
            "rejected",
            "diagnostic-passed",
            "diagnostic-rejected",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
