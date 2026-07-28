"""Configure-once LFM2 exact-4x backend for whole-model semantic folding.

The backend binds the generic semantic transaction ABI to the existing
exact-4x composite analytic kernel.  It consumes one facade-synthesized dense
blank in place and reuses one resolver-opened local tokenizer bridge and cache
bundle.  Component receipts are diagnostic partitions of that complete child;
they are never independent partial-model objectives or acceptance gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .dynamic_behavior_execution import BehaviorInferenceExecutionContract
from .dynamic_behavior_gate import BehaviorGateBinding, expose_behavior_gate_binding
from .dynamic_circuit_ir import canonical_sha256
from .dynamic_fold_executor import FoldDecision, fingerprint_model_state
from .dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
    validate_lfm2_dense_budgeted_child,
)
from .dynamic_lfm2_semantic_system import (
    LFM2_BOUND_SEMANTIC_WHOLE_MODEL_STACK_ADAPTER_ID,
)
from .dynamic_semantic_fold import (
    DynamicSemanticFoldError,
    SemanticComponentFoldReceipt,
    SemanticWholeModelCompileRequest,
    SemanticWholeModelCompileResult,
    SemanticWholeModelFoldEvidence,
    SemanticWholeModelFoldResources,
    SemanticWholeModelTargetBinding,
)
from .dynamic_target_synthesis import (
    DynamicTargetSynthesisResult,
    verify_compiler_authenticated_target,
)
from .hf_lfm2_circuit_trace_cache import open_lfm2_circuit_trace_cache
from .hf_lfm2_dense_budget_model_fold import DENSE_BUDGET_MODEL_METHOD
from .hf_lfm2_direct_local import (
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
from .hf_lfm2_semantic_postfold import (
    LFM2_LIVE_CHILD_PREFIX_RECAPTURE_DISABLED,
    LFM2_LIVE_CHILD_PREFIX_RECAPTURE_MODES,
)
from .hf_lfm2_semantic_short_conv_refold_adapter import (
    LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS,
)
from .hf_tokenizer_bridge import (
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)
from .lfm2_decoded_review import generate_review_rows, load_review_prompts
from .lfm2_exact4x_contract import (
    EXPECTED_CHILD_TOPOLOGY,
    EXPECTED_CHILD_UNIQUE_PARAMETERS,
    EXPECTED_PARENT_UNIQUE_PARAMETERS,
)
from .lfm2_exact4x_kernel_bridge import (
    TRANSITIONAL_REPOSITORY_RUNTIME_DEPENDENCY,
    run_existing_target as _run_transitional_exact4x_target,
)


SCHEMA = "neural-forge-lfm2-exact4x-semantic-backend-v1"
LFM2_EXACT4X_SEMANTIC_COMPILER_ID = (
    "neural_forge.lfm2.exact4x_composite_semantic_compiler"
)
LFM2_EXACT4X_SEMANTIC_RESOLVER_ID = (
    "neural_forge.lfm2.exact4x_composite_semantic_resolver"
)
LFM2_MANUAL_REVIEW_READY_GATE_ID = (
    "neural_forge.lfm2.decoded_manual_review_ready"
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise DynamicSemanticFoldError(
            "PyTorch is required for the LFM2 semantic backend."
        ) from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicSemanticFoldError(f"{name} must be a lowercase SHA-256.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicSemanticFoldError(f"{name} must be nonempty.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicSemanticFoldError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicSemanticFoldError(f"{name} must be non-negative.")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicSemanticFoldError(f"{name} must be finite numeric data.")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicSemanticFoldError(f"{name} must be finite numeric data.")
    return result


def _absolute_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or "://" in str(value):
        raise DynamicSemanticFoldError(f"{name} must be one ordinary local path.")
    return Path(value).expanduser().absolute()


def _ordinary_directory(path: Path, name: str) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise DynamicSemanticFoldError(f"{name} is missing or unsafe: {path}.")
    return path


def _ordinary_file(path: Path, name: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise DynamicSemanticFoldError(f"{name} is missing or unsafe: {path}.")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise DynamicSemanticFoldError(
            f"Refusing to replace decoded review evidence: {path}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="ascii")


class Lfm2Exact4xSemanticBackendConfig:
    """Immutable local asset and analytic-recipe registration."""

    __slots__ = (
        "parent_model_directory",
        "donor_tokenizer_directory",
        "target_tokenizer_directory",
        "bridge_directory",
        "expected_bridge_artifact_sha256",
        "compiler_cache_directory",
        "expected_compiler_cache_key",
        "semantic_cache_directory",
        "expected_semantic_cache_key",
        "trajectory_corpus",
        "selective_catalog_store_directory",
        "selective_catalog_receipt_json",
        "expected_selective_catalog_receipt_sha256",
        "expected_parent_identity_sha256",
        "expected_parent_unique_parameters",
        "expected_parent_graph_fingerprint",
        "source_parent_name",
        "prompts_json",
        "expected_prompts_file_sha256",
        "parent_envelope",
        "expected_parent_envelope_sha256",
        "expected_parent_envelope_file_sha256",
        "body_recipe",
        "expected_body_recipe_sha256",
        "short_conv_construction",
        "model_device",
        "fold_compute_device",
        "analytic_solve_device",
        "global_residual_backend",
        "global_residual_basis_rank",
        "correction_scale",
        "target_dtype_representability_level",
        "child_dtype",
        "maximum_prompt_tokens",
        "maximum_new_tokens",
        "coordinate_completion_recipe",
        "normalized_input_correction_rho",
        "normalized_input_request_capture_directory",
        "live_child_prefix_recapture_mode",
        "compile_artifact_root",
        "priority",
        "fingerprint",
        "resource_fingerprint",
    )

    def __init__(
        self,
        *,
        parent_model_directory: Path,
        donor_tokenizer_directory: Path,
        target_tokenizer_directory: Path,
        bridge_directory: Path,
        expected_bridge_artifact_sha256: str,
        compiler_cache_directory: Path,
        expected_compiler_cache_key: str,
        semantic_cache_directory: Path,
        expected_semantic_cache_key: str,
        trajectory_corpus: Path,
        selective_catalog_store_directory: Path,
        selective_catalog_receipt_json: Path,
        expected_selective_catalog_receipt_sha256: str,
        expected_parent_identity_sha256: str,
        expected_parent_unique_parameters: int,
        prompts_json: Path,
        expected_prompts_file_sha256: str,
        parent_envelope: Path,
        expected_parent_envelope_sha256: str,
        expected_parent_envelope_file_sha256: str,
        body_recipe: str,
        expected_body_recipe_sha256: str,
        short_conv_construction: str,
        model_device: str,
        fold_compute_device: str,
        analytic_solve_device: str,
        global_residual_backend: str,
        global_residual_basis_rank: int | None,
        correction_scale: float,
        target_dtype_representability_level: str,
        child_dtype: str,
        maximum_prompt_tokens: int,
        maximum_new_tokens: int,
        compile_artifact_root: Path,
        expected_parent_graph_fingerprint: str | None = None,
        source_parent_name: str | None = None,
        coordinate_completion_recipe: str = "bridge-procrustes-isometric",
        normalized_input_correction_rho: float = 0.25,
        normalized_input_request_capture_directory: Path | None = None,
        live_child_prefix_recapture_mode: str = (
            LFM2_LIVE_CHILD_PREFIX_RECAPTURE_DISABLED
        ),
        priority: int = 500,
    ) -> None:
        path_fields = {
            "parent_model_directory": parent_model_directory,
            "donor_tokenizer_directory": donor_tokenizer_directory,
            "target_tokenizer_directory": target_tokenizer_directory,
            "bridge_directory": bridge_directory,
            "compiler_cache_directory": compiler_cache_directory,
            "semantic_cache_directory": semantic_cache_directory,
            "trajectory_corpus": trajectory_corpus,
            "selective_catalog_store_directory": selective_catalog_store_directory,
            "selective_catalog_receipt_json": selective_catalog_receipt_json,
            "prompts_json": prompts_json,
            "parent_envelope": parent_envelope,
            "compile_artifact_root": compile_artifact_root,
        }
        for name, value in path_fields.items():
            setattr(self, name, _absolute_path(value, name))
        self.normalized_input_request_capture_directory = (
            None
            if normalized_input_request_capture_directory is None
            else _absolute_path(
                normalized_input_request_capture_directory,
                "normalized_input_request_capture_directory",
            )
        )
        for name, value in {
            "expected_bridge_artifact_sha256": expected_bridge_artifact_sha256,
            "expected_compiler_cache_key": expected_compiler_cache_key,
            "expected_semantic_cache_key": expected_semantic_cache_key,
            "expected_selective_catalog_receipt_sha256": (
                expected_selective_catalog_receipt_sha256
            ),
            "expected_parent_identity_sha256": expected_parent_identity_sha256,
            "expected_prompts_file_sha256": expected_prompts_file_sha256,
            "expected_parent_envelope_sha256": expected_parent_envelope_sha256,
            "expected_parent_envelope_file_sha256": (
                expected_parent_envelope_file_sha256
            ),
            "expected_body_recipe_sha256": expected_body_recipe_sha256,
        }.items():
            setattr(self, name, _digest(value, name))
        self.expected_parent_graph_fingerprint = (
            None
            if expected_parent_graph_fingerprint is None
            else _digest(
                expected_parent_graph_fingerprint,
                "expected_parent_graph_fingerprint",
            )
        )
        self.source_parent_name = (
            None
            if source_parent_name is None
            else _text(source_parent_name, "source_parent_name")
        )
        self.expected_parent_unique_parameters = _positive_int(
            expected_parent_unique_parameters,
            "expected_parent_unique_parameters",
        )
        if (
            self.expected_parent_unique_parameters
            != EXPECTED_PARENT_UNIQUE_PARAMETERS
        ):
            raise DynamicSemanticFoldError(
                "The exact-4x backend requires its authenticated parent count."
            )
        self.body_recipe = _text(body_recipe, "body_recipe")
        self.short_conv_construction = _text(
            short_conv_construction,
            "short_conv_construction",
        )
        if self.short_conv_construction not in (
            LFM2_SEMANTIC_SHORT_CONV_CONSTRUCTION_IDS
        ):
            raise DynamicSemanticFoldError("Unknown ShortConv construction.")
        self.model_device = _text(model_device, "model_device")
        self.fold_compute_device = _text(
            fold_compute_device,
            "fold_compute_device",
        )
        self.analytic_solve_device = _text(
            analytic_solve_device,
            "analytic_solve_device",
        )
        self.global_residual_backend = _text(
            global_residual_backend,
            "global_residual_backend",
        )
        self.global_residual_basis_rank = (
            None
            if global_residual_basis_rank is None
            else _positive_int(
                global_residual_basis_rank,
                "global_residual_basis_rank",
            )
        )
        self.correction_scale = _finite(correction_scale, "correction_scale")
        self.target_dtype_representability_level = _text(
            target_dtype_representability_level,
            "target_dtype_representability_level",
        )
        self.child_dtype = _text(child_dtype, "child_dtype").lower()
        if self.child_dtype not in {"bfloat16", "float32"}:
            raise DynamicSemanticFoldError("child_dtype must be bfloat16 or float32.")
        self.maximum_prompt_tokens = _positive_int(
            maximum_prompt_tokens,
            "maximum_prompt_tokens",
        )
        self.maximum_new_tokens = _positive_int(
            maximum_new_tokens,
            "maximum_new_tokens",
        )
        self.coordinate_completion_recipe = _text(
            coordinate_completion_recipe,
            "coordinate_completion_recipe",
        )
        self.normalized_input_correction_rho = _finite(
            normalized_input_correction_rho,
            "normalized_input_correction_rho",
        )
        self.live_child_prefix_recapture_mode = _text(
            live_child_prefix_recapture_mode,
            "live_child_prefix_recapture_mode",
        )
        if self.live_child_prefix_recapture_mode not in (
            LFM2_LIVE_CHILD_PREFIX_RECAPTURE_MODES
        ):
            raise DynamicSemanticFoldError("Unknown live-child recapture mode.")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise DynamicSemanticFoldError("priority must be an integer.")
        self.priority = priority
        body = self.as_dict(include_fingerprint=False)
        self.fingerprint = canonical_sha256(body)
        self.resource_fingerprint = self.fingerprint

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "kind": "exact4x_semantic_backend_config",
            "parent_model_directory": str(self.parent_model_directory),
            "donor_tokenizer_directory": str(self.donor_tokenizer_directory),
            "target_tokenizer_directory": str(self.target_tokenizer_directory),
            "bridge_directory": str(self.bridge_directory),
            "expected_bridge_artifact_sha256": (
                self.expected_bridge_artifact_sha256
            ),
            "compiler_cache_directory": str(self.compiler_cache_directory),
            "expected_compiler_cache_key": self.expected_compiler_cache_key,
            "semantic_cache_directory": str(self.semantic_cache_directory),
            "expected_semantic_cache_key": self.expected_semantic_cache_key,
            "trajectory_corpus": str(self.trajectory_corpus),
            "selective_catalog_store_directory": str(
                self.selective_catalog_store_directory
            ),
            "selective_catalog_receipt_json": str(
                self.selective_catalog_receipt_json
            ),
            "expected_selective_catalog_receipt_sha256": (
                self.expected_selective_catalog_receipt_sha256
            ),
            "expected_parent_identity_sha256": (
                self.expected_parent_identity_sha256
            ),
            "expected_parent_unique_parameters": (
                self.expected_parent_unique_parameters
            ),
            "expected_parent_graph_fingerprint": (
                self.expected_parent_graph_fingerprint
            ),
            "source_parent_name": self.source_parent_name,
            "prompts_json": str(self.prompts_json),
            "expected_prompts_file_sha256": self.expected_prompts_file_sha256,
            "parent_envelope": str(self.parent_envelope),
            "expected_parent_envelope_sha256": (
                self.expected_parent_envelope_sha256
            ),
            "expected_parent_envelope_file_sha256": (
                self.expected_parent_envelope_file_sha256
            ),
            "body_recipe": self.body_recipe,
            "expected_body_recipe_sha256": self.expected_body_recipe_sha256,
            "short_conv_construction": self.short_conv_construction,
            "model_device": self.model_device,
            "fold_compute_device": self.fold_compute_device,
            "analytic_solve_device": self.analytic_solve_device,
            "global_residual_backend": self.global_residual_backend,
            "global_residual_basis_rank": self.global_residual_basis_rank,
            "correction_scale": self.correction_scale,
            "target_dtype_representability_level": (
                self.target_dtype_representability_level
            ),
            "child_dtype": self.child_dtype,
            "maximum_prompt_tokens": self.maximum_prompt_tokens,
            "maximum_new_tokens": self.maximum_new_tokens,
            "coordinate_completion_recipe": self.coordinate_completion_recipe,
            "normalized_input_correction_rho": (
                self.normalized_input_correction_rho
            ),
            "normalized_input_request_capture_directory": (
                None
                if self.normalized_input_request_capture_directory is None
                else str(self.normalized_input_request_capture_directory)
            ),
            "live_child_prefix_recapture_mode": (
                self.live_child_prefix_recapture_mode
            ),
            "compile_artifact_root": str(self.compile_artifact_root),
            "priority": self.priority,
            "network_allowed": False,
            "transitional_repository_kernel_runtime_dependency": (
                TRANSITIONAL_REPOSITORY_RUNTIME_DEPENDENCY
            ),
            "fully_packaged_production_kernel": False,
        }
        if include_fingerprint:
            body["fingerprint"] = self.fingerprint
        return body

    def verify(self) -> "Lfm2Exact4xSemanticBackendConfig":
        for path, name in (
            (self.parent_model_directory, "parent model directory"),
            (self.donor_tokenizer_directory, "donor tokenizer directory"),
            (self.target_tokenizer_directory, "target tokenizer directory"),
            (self.bridge_directory, "bridge directory"),
            (self.compiler_cache_directory, "compiler cache directory"),
            (self.semantic_cache_directory, "semantic cache directory"),
            (
                self.selective_catalog_store_directory,
                "selective catalog store directory",
            ),
        ):
            _ordinary_directory(path, name)
        for path, name in (
            (self.trajectory_corpus, "trajectory corpus"),
            (self.selective_catalog_receipt_json, "selective catalog receipt"),
            (self.prompts_json, "review prompts"),
            (self.parent_envelope, "parent response envelope"),
        ):
            _ordinary_file(path, name)
        if _file_sha256(self.prompts_json) != self.expected_prompts_file_sha256:
            raise DynamicSemanticFoldError("Review prompt bytes changed.")
        if (
            _file_sha256(self.parent_envelope)
            != self.expected_parent_envelope_file_sha256
        ):
            raise DynamicSemanticFoldError("Parent response envelope bytes changed.")
        root = self.compile_artifact_root
        if root.exists() and (not root.is_dir() or root.is_symlink()):
            raise DynamicSemanticFoldError("compile_artifact_root is unsafe.")
        if not root.exists():
            _ordinary_directory(root.parent, "compile artifact parent directory")
        if canonical_sha256(self.as_dict(include_fingerprint=False)) != self.fingerprint:
            raise DynamicSemanticFoldError("Semantic backend config changed.")
        return self

    def kernel_arguments(self) -> dict[str, Any]:
        return {
            "parent_model_directory": self.parent_model_directory,
            "donor_tokenizer_directory": self.donor_tokenizer_directory,
            "target_tokenizer_directory": self.target_tokenizer_directory,
            "bridge_directory": self.bridge_directory,
            "expected_bridge_artifact_sha256": (
                self.expected_bridge_artifact_sha256
            ),
            "compiler_cache_directory": self.compiler_cache_directory,
            "expected_compiler_cache_key": self.expected_compiler_cache_key,
            "semantic_cache_directory": self.semantic_cache_directory,
            "expected_semantic_cache_key": self.expected_semantic_cache_key,
            "trajectory_corpus": self.trajectory_corpus,
            "selective_catalog_store_directory": (
                self.selective_catalog_store_directory
            ),
            "selective_catalog_receipt_json": self.selective_catalog_receipt_json,
            "expected_selective_catalog_receipt_sha256": (
                self.expected_selective_catalog_receipt_sha256
            ),
            "expected_parent_identity_sha256": (
                self.expected_parent_identity_sha256
            ),
            "expected_parent_unique_parameters": (
                self.expected_parent_unique_parameters
            ),
            "prompts_json": self.prompts_json,
            "expected_prompts_file_sha256": self.expected_prompts_file_sha256,
            "parent_envelope": self.parent_envelope,
            "expected_parent_envelope_sha256": (
                self.expected_parent_envelope_sha256
            ),
            "expected_parent_envelope_file_sha256": (
                self.expected_parent_envelope_file_sha256
            ),
            "body_recipe": self.body_recipe,
            "expected_body_recipe_sha256": self.expected_body_recipe_sha256,
            "short_conv_construction": self.short_conv_construction,
            "model_device": self.model_device,
            "fold_compute_device": self.fold_compute_device,
            "analytic_solve_device": self.analytic_solve_device,
            "global_residual_backend": self.global_residual_backend,
            "global_residual_basis_rank": self.global_residual_basis_rank,
            "correction_scale": self.correction_scale,
            "target_dtype_representability_level": (
                self.target_dtype_representability_level
            ),
            "child_dtype": self.child_dtype,
            "maximum_prompt_tokens": self.maximum_prompt_tokens,
            "maximum_new_tokens": self.maximum_new_tokens,
            "coordinate_completion_recipe": self.coordinate_completion_recipe,
            "normalized_input_correction_rho": (
                self.normalized_input_correction_rho
            ),
            "normalized_input_request_capture_directory": (
                self.normalized_input_request_capture_directory
            ),
            "live_child_prefix_recapture_mode": (
                self.live_child_prefix_recapture_mode
            ),
        }


class Lfm2Exact4xSemanticAssetBundle:
    """Resolver-opened immutable handles reused by the compile transaction."""

    __slots__ = (
        "config",
        "donor_tokenizer",
        "target_tokenizer",
        "bridge",
        "compiler_cache",
        "semantic_cache",
        "fingerprint",
        "resource_fingerprint",
    )

    def __init__(
        self,
        *,
        config: Lfm2Exact4xSemanticBackendConfig,
        donor_tokenizer: Any,
        target_tokenizer: Any,
        bridge: Any,
        compiler_cache: Any,
        semantic_cache: Any,
    ) -> None:
        self.config = config.verify()
        self.donor_tokenizer = donor_tokenizer
        self.target_tokenizer = target_tokenizer
        self.bridge = bridge
        self.compiler_cache = compiler_cache
        self.semantic_cache = semantic_cache
        self.fingerprint = canonical_sha256(self._identity_body())
        self.resource_fingerprint = self.fingerprint
        self.verify()

    def _identity_body(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "resolver_opened_semantic_asset_bundle",
            "config_fingerprint": self.config.fingerprint,
            "bridge_artifact_sha256": self.bridge.artifact_sha256,
            "bridge_profile_sha256": self.bridge.profile_fingerprint,
            "donor_tokenizer_semantics_sha256": (
                fingerprint_hf_tokenizer_semantics(self.donor_tokenizer)
            ),
            "target_tokenizer_semantics_sha256": (
                fingerprint_hf_tokenizer_semantics(self.target_tokenizer)
            ),
            "compiler_cache_key": self.compiler_cache.cache_key,
            "semantic_cache_key": self.semantic_cache.cache_key,
            "parent_identity_sha256": self.compiler_cache.parent_fingerprint,
            "cache_payloads_opened_read_only": True,
            "full_cache_tensor_sweep_at_resolve": False,
            "network_allowed": False,
        }

    def verify(self) -> "Lfm2Exact4xSemanticAssetBundle":
        self.config.verify()
        self.bridge.verify_files()
        body = self._identity_body()
        if (
            body["bridge_artifact_sha256"]
            != self.config.expected_bridge_artifact_sha256
            or body["compiler_cache_key"]
            != self.config.expected_compiler_cache_key
            or body["semantic_cache_key"]
            != self.config.expected_semantic_cache_key
            or body["parent_identity_sha256"]
            != self.config.expected_parent_identity_sha256
            or self.semantic_cache.parent_fingerprint
            != self.config.expected_parent_identity_sha256
            or body["donor_tokenizer_semantics_sha256"]
            != self.bridge.donor_tokenizer_semantics_sha256
            or body["target_tokenizer_semantics_sha256"]
            != self.bridge.target_tokenizer_semantics_sha256
            or canonical_sha256(body) != self.fingerprint
        ):
            raise DynamicSemanticFoldError(
                "Resolver-opened bridge or cache resources changed."
            )
        return self


def _run_existing_exact4x_target(**kwargs: Any) -> dict[str, Any]:
    return _run_transitional_exact4x_target(**kwargs)


def _receipt_digest(value: Mapping[str, Any], name: str) -> str:
    if not isinstance(value, Mapping):
        raise DynamicSemanticFoldError(f"Kernel omitted {name} receipt.")
    result = value.get("result_sha256") or value.get("receipt_sha256")
    if result is None:
        return canonical_sha256(dict(value))
    return _digest(result, f"{name} receipt SHA-256")


def _component_parameter_paths(model: Any) -> dict[str, tuple[str, ...]]:
    try:
        paths = tuple(
            name
            for name, _parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        )
    except TypeError:  # pragma: no cover - old Torch compatibility
        paths = tuple(name for name, _parameter in model.named_parameters())
    if not paths or len(set(paths)) != len(paths):
        raise DynamicSemanticFoldError(
            "LFM2 semantic target has empty or duplicate parameter paths."
        )
    groups: dict[str, list[str]] = {
        "token-interface-and-causal-head": [],
        "residual-attention": [],
        "short-convolution": [],
        "gated-mlp": [],
        "normalization-and-global-residual": [],
    }
    for path in paths:
        if path in {"model.embed_tokens.weight", "lm_head.weight"}:
            group = "token-interface-and-causal-head"
        elif ".self_attn." in path:
            group = "residual-attention"
        elif ".conv." in path:
            group = "short-convolution"
        elif ".feed_forward." in path:
            group = "gated-mlp"
        elif "norm" in path:
            group = "normalization-and-global-residual"
        else:
            raise DynamicSemanticFoldError(
                f"Unclassified LFM2 target parameter: {path}."
            )
        groups[group].append(path)
    if any(not values for values in groups.values()):
        raise DynamicSemanticFoldError(
            "Exact-4x LFM2 component partition is structurally incomplete."
        )
    return {
        name: tuple(sorted(values)) for name, values in groups.items()
    }


class Lfm2Exact4xSemanticCompiler:
    """Call the real composite kernel once against the facade-owned blank."""

    compiler_id = LFM2_EXACT4X_SEMANTIC_COMPILER_ID
    version = 1

    def __init__(self, resources: Lfm2Exact4xSemanticAssetBundle) -> None:
        self.resources = resources.verify()
        self.config = resources.config
        self.resource_fingerprint = canonical_sha256(
            {
                "schema": SCHEMA,
                "kind": "exact4x_semantic_compiler",
                "asset_bundle_fingerprint": resources.fingerprint,
                "compiler_id": self.compiler_id,
                "version": self.version,
            }
        )

    def _next_run_directory(
        self,
        request: SemanticWholeModelCompileRequest,
    ) -> Path:
        root = self.config.compile_artifact_root
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise DynamicSemanticFoldError("Compile artifact root became unsafe.")
        stem = canonical_sha256(
            {
                "target": request.target.fingerprint,
                "action": request.action.action_sha256,
                "evidence": request.evidence.fingerprint,
                "compiler": self.resource_fingerprint,
            }
        )[:20]
        for index in range(10_000):
            candidate = root / f"semantic-{stem}-{index:04d}"
            if not candidate.exists() and not candidate.is_symlink():
                return candidate
        raise DynamicSemanticFoldError("Compile artifact namespace is exhausted.")

    def compile(
        self,
        request: SemanticWholeModelCompileRequest,
    ) -> SemanticWholeModelCompileResult:
        if not isinstance(request, SemanticWholeModelCompileRequest):
            raise DynamicSemanticFoldError("Invalid semantic compile request.")
        resources = self.resources.verify()
        target = request.target
        child = request.context.child_model
        if target.model is not child:
            raise DynamicSemanticFoldError(
                "Exact-4x compiler did not receive the facade-owned blank."
            )
        verify_compiler_authenticated_target(target)
        selected = target.adapter_payload
        if not isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
            raise DynamicSemanticFoldError("Target is not a dense LFM2 selection.")
        validate_lfm2_dense_budgeted_child(selected, child, target.graph)
        if (
            request.evidence.source is not resources
            or request.evidence.root_fingerprint
            != self.config.expected_semantic_cache_key
            or fingerprint_model_state(child) != target.blank_state_fingerprint
            or bool(getattr(child, "training", True))
        ):
            raise DynamicSemanticFoldError(
                "Semantic evidence or facade blank changed before compilation."
            )
        parent = request.context.parent_models.get(selected.source_parent)
        parent_graph = request.context.parent_graphs.get(selected.source_parent)
        if (
            parent is None
            or parent_graph is None
            or parent_graph.fingerprint != selected.source_graph_fingerprint
            or bool(getattr(parent, "training", True))
        ):
            raise DynamicSemanticFoldError(
                "The selected LFM2 source parent is missing or changed."
            )
        run_directory = self._next_run_directory(request)
        result = _run_existing_exact4x_target(
            run_directory=run_directory,
            target=target,
            parent_model=parent,
            blank_seed=target.request.blank_seed,
            existing_donor_tokenizer=resources.donor_tokenizer,
            existing_target_tokenizer=resources.target_tokenizer,
            existing_bridge=resources.bridge,
            existing_compiler_cache=resources.compiler_cache,
            existing_semantic_cache=resources.semantic_cache,
            **self.config.kernel_arguments(),
        )
        if not isinstance(result, Mapping):
            raise DynamicSemanticFoldError("Exact-4x kernel returned no receipt.")
        installed_state = _digest(
            result.get("complete_child_model_state_sha256"),
            "installed child state",
        )
        audit = result.get("operation_audit")
        if not isinstance(audit, Mapping):
            raise DynamicSemanticFoldError("Exact-4x kernel omitted its audit.")
        required_zero = (
            "training_steps",
            "optimizer_constructions",
            "optimizer_steps",
            "autograd_grad_calls",
            "backward_calls",
            "parent_forward_calls",
            "parent_generate_calls",
            "parent_trace_recaptures",
            "parameters_pruned",
            "parameters_quantized",
            "parameters_sparsified",
            "external_adapter_parameters",
            "parent_sidecar_parameters",
            "runtime_shared_parent_parameters",
            "parent_model_loads_inside_kernel",
            "target_model_materializations_inside_kernel",
            "verification_reload_materializations",
        )
        if (
            installed_state != fingerprint_model_state(child)
            or result.get("existing_compiler_blank_used") is not True
            or result.get("publication_deferred_to_outer_compiler") is not True
            or result.get("preopened_semantic_resources_reused") is not True
            or any(audit.get(name) != 0 for name in required_zero)
            or audit.get("network_access") is not False
        ):
            raise DynamicSemanticFoldError(
                "Exact-4x kernel violated its in-place cache-only boundary."
            )
        groups = _component_parameter_paths(child)
        layer_stack = result.get("semantic_layer_stack")
        structural_interface = result.get("structural_interface_install")
        causal_bridge = result.get("causal_token_bridge")
        causal_output = result.get("causal_output_install")
        deferred_w2 = result.get("deferred_final_w2_install")
        if any(
            not isinstance(value, Mapping)
            for value in (
                layer_stack,
                structural_interface,
                causal_bridge,
                causal_output,
                deferred_w2,
            )
        ):
            raise DynamicSemanticFoldError(
                "Exact-4x kernel omitted a required component installation receipt."
            )
        layer_stack_sha256 = _receipt_digest(layer_stack, "semantic layer stack")
        components = (
            SemanticComponentFoldReceipt(
                component_id="token-interface-and-causal-head",
                covered_parameter_paths=groups[
                    "token-interface-and-causal-head"
                ],
                receipt={
                    "structural_interface_sha256": _receipt_digest(
                        structural_interface,
                        "structural interface",
                    ),
                    "causal_token_bridge_sha256": _receipt_digest(
                        causal_bridge,
                        "causal token bridge",
                    ),
                    "causal_output_install_sha256": _receipt_digest(
                        causal_output,
                        "causal output install",
                    ),
                    "decoded_behavior_gate_authority": False,
                },
            ),
            SemanticComponentFoldReceipt(
                component_id="residual-attention",
                covered_parameter_paths=groups["residual-attention"],
                receipt={
                    "semantic_layer_stack_sha256": layer_stack_sha256,
                    "attention_lineage_sha256": _digest(
                        result.get("attention_lineage_sha256"),
                        "attention lineage SHA-256",
                    ),
                    "structural_candidate_archive_sha256": _digest(
                        result["structural_attention_candidate_archive"].get(
                            "manifest_sha256"
                        ),
                        "attention archive SHA-256",
                    ),
                    "diagnostic_only": True,
                },
            ),
            SemanticComponentFoldReceipt(
                component_id="short-convolution",
                covered_parameter_paths=groups["short-convolution"],
                receipt={
                    "semantic_layer_stack_sha256": layer_stack_sha256,
                    "construction": result.get("short_conv_construction"),
                    "coordinate_completion_recipe": result.get(
                        "body_global_coordinate_completion_recipe"
                    ),
                    "diagnostic_only": True,
                },
            ),
            SemanticComponentFoldReceipt(
                component_id="gated-mlp",
                covered_parameter_paths=groups["gated-mlp"],
                receipt={
                    "semantic_layer_stack_sha256": layer_stack_sha256,
                    "deferred_final_w2_install_sha256": _receipt_digest(
                        deferred_w2,
                        "deferred final W2",
                    ),
                    "diagnostic_only": True,
                },
            ),
            SemanticComponentFoldReceipt(
                component_id="normalization-and-global-residual",
                covered_parameter_paths=groups[
                    "normalization-and-global-residual"
                ],
                receipt={
                    "neutral_norm_install_sha256": canonical_sha256(
                        dict(result.get("neutral_norm_install", {}))
                    ),
                    "global_residual_geometry_sha256": _digest(
                        result.get("body_global_parent_geometry_sha256"),
                        "global residual geometry SHA-256",
                    ),
                    "semantic_layer_stack_sha256": layer_stack_sha256,
                    "diagnostic_only": True,
                },
            ),
        )
        paths = tuple(sorted(path for values in groups.values() for path in values))
        normalized_audit = {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "cached_parent_evidence_only": True,
            "complete_model_compaction": True,
            "component_diagnostics_have_acceptance_authority": False,
            "numeric_diagnostics_have_acceptance_authority": False,
            "parent_model_loads_inside_kernel": 0,
            "target_model_materializations_inside_kernel": 0,
            "verification_reload_materializations_inside_kernel": 0,
            "parent_sidecar_parameters": 0,
            "runtime_shared_parent_parameters": 0,
            "network_access": False,
        }
        whole_receipt = {
            "status": "complete-in-place-exact4x-child-installed",
            "kernel_run_directory": str(run_directory),
            "kernel_run_receipt": str(run_directory / "run-receipt.json"),
            "kernel_run_receipt_sha256": _digest(
                result.get("result_sha256"),
                "kernel result SHA-256",
            ),
            "complete_child_model_state_sha256": installed_state,
            "parent_identity_sha256": result.get("parent_identity_sha256"),
            "parent_unique_parameters": result.get("parent_unique_parameters"),
            "target_unique_parameters": result.get("target_unique_parameters"),
            "realized_parameter_compaction_ratio": result.get(
                "realized_parameter_compaction_ratio"
            ),
            "component_count": len(components),
            "complete_target_parameter_count": len(paths),
            "outer_publication_and_decoded_review_required": True,
            "automatic_behavior_verdict_performed": False,
            "manual_intelligence_review_required": True,
            "transitional_repository_kernel_runtime_dependency": (
                TRANSITIONAL_REPOSITORY_RUNTIME_DEPENDENCY
            ),
            "fully_packaged_production_kernel": False,
        }
        return SemanticWholeModelCompileResult(
            covered_target_parameter_paths=paths,
            component_receipts=components,
            whole_model_receipt=whole_receipt,
            operation_audit=normalized_audit,
        )


class Lfm2DecodedManualReviewReadyGate:
    """Publish complete decoded outputs, without deciding intelligence."""

    gate_id = LFM2_MANUAL_REVIEW_READY_GATE_ID
    gate_version = 1

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompts: Sequence[Mapping[str, Any]],
        output_root: Path,
        maximum_prompt_tokens: int,
        maximum_new_tokens: int,
        baseline_fingerprint: str,
        prompt_set_fingerprint: str,
        tokenizer_fingerprint: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompts = tuple(dict(value) for value in prompts)
        if not self.prompts:
            raise DynamicSemanticFoldError("Manual review gate needs prompts.")
        self.output_root = _absolute_path(output_root, "manual review output root")
        self.maximum_prompt_tokens = _positive_int(
            maximum_prompt_tokens,
            "maximum_prompt_tokens",
        )
        self.maximum_new_tokens = _positive_int(
            maximum_new_tokens,
            "maximum_new_tokens",
        )
        self.prompt_set_fingerprint = _digest(
            prompt_set_fingerprint,
            "prompt_set_fingerprint",
        )
        self.tokenizer_fingerprint = _digest(
            tokenizer_fingerprint,
            "tokenizer_fingerprint",
        )
        self.resource_fingerprint = canonical_sha256(
            {
                "schema": SCHEMA,
                "kind": "decoded_manual_review_ready_gate",
                "output_root": str(self.output_root),
                "prompt_set_fingerprint": self.prompt_set_fingerprint,
                "tokenizer_fingerprint": self.tokenizer_fingerprint,
                "maximum_prompt_tokens": self.maximum_prompt_tokens,
                "maximum_new_tokens": self.maximum_new_tokens,
                "automatic_behavior_verdict_performed": False,
            }
        )
        contract = BehaviorInferenceExecutionContract(
            entrypoint="generate",
            required_module_path="",
        )
        manifest_body = {
            "schema": "neural-forge-lfm2-manual-review-ready-gate-v1",
            "gate_id": self.gate_id,
            "gate_version": self.gate_version,
            "baseline_fingerprint": _digest(
                baseline_fingerprint,
                "baseline_fingerprint",
            ),
            "prompt_set_fingerprint": self.prompt_set_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "automatic_behavior_verdict_performed": False,
            "manual_intelligence_review_required": True,
            "numeric_diagnostics_have_acceptance_authority": False,
            "accepted_means_ready_for_manual_review": True,
            "inference_execution": contract.as_dict(),
        }
        manifest_fingerprint = canonical_sha256(manifest_body)
        binding = BehaviorGateBinding(
            gate_id=self.gate_id,
            gate_version=self.gate_version,
            baseline_fingerprint=manifest_body["baseline_fingerprint"],
            manifest_fingerprint=manifest_fingerprint,
            manifest={
                **manifest_body,
                "manifest_fingerprint": manifest_fingerprint,
            },
        )
        expose_behavior_gate_binding(self, binding)
        self._lock = threading.Lock()
        self._review_count = 0

    def _review(self, model: Any, phase: str) -> FoldDecision:
        candidate_state = fingerprint_model_state(model)
        responses = generate_review_rows(
            model,
            self.tokenizer,
            self.prompts,
            maximum_prompt_tokens=self.maximum_prompt_tokens,
            maximum_new_tokens=self.maximum_new_tokens,
        )
        if len(responses) != len(self.prompts):
            raise DynamicSemanticFoldError(
                "Decoded review did not preserve every requested response."
            )
        body = {
            "schema": "neural-forge-lfm2-decoded-manual-review-dossier-v1",
            "status": "candidate-ready-for-manual-intelligence-review",
            "phase": phase,
            "candidate_state_sha256": candidate_state,
            "prompt_set_fingerprint": self.prompt_set_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "response_count": len(responses),
            "responses": responses,
            "decoded_whole_responses_are_primary_evidence": True,
            "complete_decoded_responses_preserved": True,
            "diagnostic_metrics_are_advisory_only": True,
            "numeric_diagnostics_have_acceptance_authority": False,
            "automatic_behavior_verdict_performed": False,
            "manual_intelligence_review_required": True,
            "accepted_means_ready_for_manual_review": True,
            "intelligence_success_claimed": False,
        }
        dossier = {**body, "dossier_sha256": canonical_sha256(body)}
        with self._lock:
            self.output_root.mkdir(parents=True, exist_ok=True)
            if self.output_root.is_symlink():
                raise DynamicSemanticFoldError("Manual review output root is unsafe.")
            index = self._review_count
            self._review_count += 1
            path = self.output_root / (
                f"{index:04d}-{phase}-{candidate_state[:16]}.json"
            )
            _write_json(path, dossier)
        return FoldDecision.accept(
            "Decoded outputs published; candidate is structurally ready for "
            "manual intelligence review. No automatic intelligence verdict was made.",
            {
                "decoded_response_count": len(responses),
                "manual_intelligence_review_required": True,
                "automatic_behavior_verdict_performed": False,
                "accepted_means_ready_for_manual_review": True,
                "dossier_sha256": dossier["dossier_sha256"],
            },
        )

    def evaluate(self, context: Any) -> FoldDecision:
        model = context.execution_context.child_model
        return self._review(model, "in-memory")

    def __call__(self, model: Any) -> FoldDecision:
        return self._review(model, "fresh-reload")


class Lfm2Exact4xSemanticResourceResolver:
    """Select and open one exact-4x resource bundle per synthesized target."""

    resolver_id = LFM2_EXACT4X_SEMANTIC_RESOLVER_ID
    version = 1
    target_families = ("lfm2",)
    target_modalities = ("text",)
    method_ids = (DENSE_BUDGET_MODEL_METHOD,)

    def __init__(
        self,
        configs: Sequence[Lfm2Exact4xSemanticBackendConfig],
    ) -> None:
        self.configs = tuple(configs)
        if not self.configs or any(
            not isinstance(value, Lfm2Exact4xSemanticBackendConfig)
            for value in self.configs
        ):
            raise DynamicSemanticFoldError(
                "Exact-4x resolver needs typed backend configs."
            )
        fingerprints = [value.fingerprint for value in self.configs]
        if len(set(fingerprints)) != len(fingerprints):
            raise DynamicSemanticFoldError("Resolver configs must be unique.")
        self.priority = max(value.priority for value in self.configs)
        self.resource_fingerprint = canonical_sha256(
            {
                "schema": SCHEMA,
                "kind": "exact4x_semantic_resource_resolver",
                "resolver_id": self.resolver_id,
                "version": self.version,
                "config_fingerprints": sorted(fingerprints),
            }
        )
        self.resolve_count = 0

    @staticmethod
    def _matches(
        config: Lfm2Exact4xSemanticBackendConfig,
        target: DynamicTargetSynthesisResult,
    ) -> bool:
        if not isinstance(target, DynamicTargetSynthesisResult):
            return False
        selected = target.adapter_payload
        if not isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
            return False
        try:
            verify_compiler_authenticated_target(target)
            validate_lfm2_dense_budgeted_child(selected, target.model, target.graph)
        except Exception:
            return False
        topology = {
            "vocab_size": selected.vocabulary_size,
            "hidden_size": selected.hidden_size,
            "intermediate_size": selected.intermediate_size,
            "num_hidden_layers": selected.target_depth,
            "num_attention_heads": selected.query_heads,
            "num_key_value_heads": selected.key_value_heads,
            "head_dim": selected.head_dim,
            "tie_word_embeddings": selected.tie_word_embeddings,
        }
        parent_graphs = dict(target.parent_graph_fingerprints)
        return (
            target.adapter_id == Lfm2TargetSynthesisAdapter.adapter_id
            and target.adapter_version == Lfm2TargetSynthesisAdapter.version
            and target.resource_profile.family == "lfm2"
            and target.resource_profile.modalities == ("text",)
            and target.request.fold_method_id in {None, DENSE_BUDGET_MODEL_METHOD}
            and target.request.exact_unique_parameters
            == EXPECTED_CHILD_UNIQUE_PARAMETERS
            and target.resource_profile.unique_parameters
            == EXPECTED_CHILD_UNIQUE_PARAMETERS
            and topology == EXPECTED_CHILD_TOPOLOGY
            and selected.ordered_source_spans
            == tuple((index,) for index in range(selected.target_depth))
            and selected.tokenizer_bridge_required
            and selected.target_token_interface_mode == "untied"
            and selected.tokenizer_bridge_artifact_sha256
            == config.expected_bridge_artifact_sha256
            and parent_graphs.get(selected.source_parent)
            == selected.source_graph_fingerprint
            and (
                config.source_parent_name is None
                or selected.source_parent == config.source_parent_name
            )
            and (
                config.expected_parent_graph_fingerprint is None
                or selected.source_graph_fingerprint
                == config.expected_parent_graph_fingerprint
            )
            and target.blank_provenance is not None
            and target.blank_provenance.compiler_created_blank
            and target.blank_provenance.all_parameters_zero
            and not bool(getattr(target.model, "training", True))
        )

    def _matching_configs(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> tuple[Lfm2Exact4xSemanticBackendConfig, ...]:
        return tuple(
            config for config in self.configs if self._matches(config, target)
        )

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        return len(self._matching_configs(target)) == 1

    def resolve(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> SemanticWholeModelFoldResources:
        matches = self._matching_configs(target)
        if len(matches) != 1:
            raise DynamicSemanticFoldError(
                "Exact-4x resolver found no unique backend registration."
            )
        config = matches[0].verify()
        donor_tokenizer = load_lfm2_direct_local_tokenizer(
            config.donor_tokenizer_directory
        )
        target_tokenizer = load_lfm2_direct_local_tokenizer(
            config.target_tokenizer_directory
        )
        bridge = load_strict_tokenizer_bridge_artifact(
            config.bridge_directory,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            expected_artifact_sha256=config.expected_bridge_artifact_sha256,
        )
        compiler_cache = open_lfm2_circuit_trace_cache(
            config.compiler_cache_directory,
            verify=False,
        )
        semantic_cache = open_lfm2_circuit_trace_cache(
            config.semantic_cache_directory,
            verify=False,
        )
        assets = Lfm2Exact4xSemanticAssetBundle(
            config=config,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            bridge=bridge,
            compiler_cache=compiler_cache,
            semantic_cache=semantic_cache,
        )
        prompts = load_review_prompts(config.prompts_json)
        prompt_set_fingerprint = canonical_sha256(
            [dict(value) for value in prompts]
        )
        gate = Lfm2DecodedManualReviewReadyGate(
            tokenizer=target_tokenizer,
            prompts=prompts,
            output_root=config.compile_artifact_root / "decoded-manual-review",
            maximum_prompt_tokens=config.maximum_prompt_tokens,
            maximum_new_tokens=config.maximum_new_tokens,
            baseline_fingerprint=config.expected_parent_envelope_sha256,
            prompt_set_fingerprint=prompt_set_fingerprint,
            tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        )
        checkpoint_io = Lfm2DirectLocalCheckpointIO(
            tokenizer_bridge=bridge,
            device=_torch().device(config.model_device),
            dtype=getattr(_torch(), config.child_dtype),
        )
        binding = SemanticWholeModelTargetBinding(
            binding_id=(
                f"lfm2-exact4x-{config.fingerprint[:16]}-"
                f"{target.topology_fingerprint[:16]}"
            ),
            target_family="lfm2",
            target_modalities=("text",),
            target_adapter_id=target.adapter_id,
            target_adapter_version=target.adapter_version,
            method_id=DENSE_BUDGET_MODEL_METHOD,
            exact_target_unique_parameters=(
                target.resource_profile.unique_parameters
            ),
            parent_graph_fingerprints=target.parent_graph_fingerprints,
            child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
            target_configuration_sha256=canonical_sha256(
                dict(target.request.configuration)
            ),
            topology_fingerprint=target.topology_fingerprint,
        )
        compiler = Lfm2Exact4xSemanticCompiler(assets)
        evidence = SemanticWholeModelFoldEvidence(
            root_fingerprint=config.expected_semantic_cache_key,
            source=assets,
        )
        self.resolve_count += 1
        return SemanticWholeModelFoldResources(
            binding=binding,
            evidence=evidence,
            compiler=compiler,
            final_evaluator=gate,
            model_saver=checkpoint_io.save,
            model_loader=checkpoint_io.load,
            fresh_reload_evaluator=gate,
            saver_id=checkpoint_io.saver_id,
            loader_id=checkpoint_io.loader_id,
            stack_adapter_id=LFM2_BOUND_SEMANTIC_WHOLE_MODEL_STACK_ADAPTER_ID,
            stack_adapter_version=1,
            priority=config.priority,
        )


__all__ = [
    "LFM2_EXACT4X_SEMANTIC_COMPILER_ID",
    "LFM2_EXACT4X_SEMANTIC_RESOLVER_ID",
    "LFM2_MANUAL_REVIEW_READY_GATE_ID",
    "Lfm2DecodedManualReviewReadyGate",
    "Lfm2Exact4xSemanticAssetBundle",
    "Lfm2Exact4xSemanticBackendConfig",
    "Lfm2Exact4xSemanticCompiler",
    "Lfm2Exact4xSemanticResourceResolver",
]
