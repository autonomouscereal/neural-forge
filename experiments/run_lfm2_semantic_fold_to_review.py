#!/usr/bin/env python3
"""Fold one LFM2 body lane directly into complete decoded behavior evidence.

The transaction publishes a preterminal capsule, exports the terminal W2 bank
from the same live cache session, materializes every structurally eligible
writer as a standalone dense child, and generates whole responses.  Historical
recipes retain the v1 qualitative-review handoff.  Opted-in coupled-attention
recipes use a separately sealed v2 flow in which metrics and optional human
observations guide investigation but cannot gate continuation or publication.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# CUDA folding is analytic but still uses cuBLAS.  Establish PyTorch's
# deterministic workspace contract before any imported module can initialize
# CUDA; an explicitly supplied value remains authoritative and is validated by
# the CUDA folding stages themselves.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from experiments.prepare_lfm2_semantic_qualitative_review import (  # noqa: E402
    prepare_qualitative_review,
)
from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    Lfm2SemanticFrontierReviewError,
    _load_parent_envelope,
    load_review_prompts,
    review_frontier,
)
from experiments.run_lfm2_dynamic_catalog import (  # noqa: E402
    fingerprint_local_tokenizer,
)
from experiments.run_lfm2_residual_attention_fold import (  # noqa: E402
    load_local_tokenizer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_global_residual_closure import (  # noqa: E402
    Lfm2GlobalResidualClosurePolicy,
)
from neural_forge.hf_lfm2_producer_aware_semantic_integration import (  # noqa: E402
    Lfm2ProducerAwareCorrectionScalePolicy,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (  # noqa: E402
    LFM2_SEMANTIC_BODY_RECIPE_IDS,
    Lfm2SemanticBodyRecipe,
    Lfm2SemanticNonblockingBehaviorFlow,
    resolve_lfm2_semantic_body_recipe,
    resolve_lfm2_semantic_nonblocking_behavior_flow,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    TIED_RESIDUAL_CONGRUENCE_MODES,
    TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    run_lfm2_semantic_postfold,
)
from neural_forge.hf_lfm2_semantic_postfold_frontier import (  # noqa: E402
    Lfm2SemanticTerminalFrontierBankReference,
    Lfm2SemanticTerminalFrontierSourceRoots,
    load_lfm2_semantic_terminal_frontier_bank,
    materialize_lfm2_semantic_terminal_frontier_candidate,
    publish_lfm2_semantic_terminal_frontier_bank,
)
from neural_forge.hf_lfm2_semantic_preterminal_capsule import (  # noqa: E402
    Lfm2SemanticPreterminalCapsuleReference,
)


PIPELINE_SCHEMA = "neural-forge-lfm2-semantic-fold-to-review-v1"
NONBLOCKING_PIPELINE_SCHEMA = "neural-forge-lfm2-semantic-fold-to-decoded-behavior-v2"
OPTIONAL_BEHAVIOR_OBSERVATIONS_SCHEMA = (
    "neural-forge-lfm2-semantic-optional-behavior-observations-v1"
)
CONTINUOUS_COMPILATION_SCHEMA = (
    "neural-forge-lfm2-semantic-continuous-terminal-frontier-compilation-v1"
)
FRONTIER_RUN_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-run-v1"
FRONTIER_PROGRESS_SCHEMA = "neural-forge-lfm2-semantic-terminal-frontier-progress-v1"
CORRECTION_SCALE_FRONTIER_SCHEMA = (
    "neural-forge-lfm2-producer-aware-correction-scale-frontier-v1"
)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_ZERO_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "parent_forward_passes": 0,
    "parent_trace_recaptures": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "parameters_sparsified": 0,
    "sidecar_parameters": 0,
}


class Lfm2SemanticFoldToReviewError(RuntimeError):
    """The no-training fold-to-review transaction could not be completed."""


@dataclass(frozen=True, slots=True)
class _AuthenticatedReviewPreflight:
    prompt_set_sha256: str
    prompt_ids: tuple[str, ...]
    envelope_sha256: str
    envelope_file_sha256: str
    parent_state_sha256: str
    parent_identity_sha256: str
    parent_unique_parameters: int
    tokenizer_sha256: str
    response_bank_sha256: str
    cache_key: str
    cache_manifest_sha256: str
    maximum_prompt_tokens: int
    maximum_new_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_set_sha256": self.prompt_set_sha256,
            "prompt_ids": list(self.prompt_ids),
            "envelope_sha256": self.envelope_sha256,
            "envelope_file_sha256": self.envelope_file_sha256,
            "parent_state_sha256": self.parent_state_sha256,
            "parent_identity_sha256": self.parent_identity_sha256,
            "parent_unique_parameters": self.parent_unique_parameters,
            "tokenizer_sha256": self.tokenizer_sha256,
            "response_bank_sha256": self.response_bank_sha256,
            "cache_key": self.cache_key,
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "maximum_prompt_tokens": self.maximum_prompt_tokens,
            "maximum_new_tokens": self.maximum_new_tokens,
        }


def _authenticated_local_parent_parameter_inventory(
    postfold: Mapping[str, Any],
    *,
    expected_parent_identity_sha256: str,
    expected_parent_unique_parameters: int,
) -> Mapping[str, Any]:
    """Count the exact local parent tensors already authenticated by postfold."""

    stack_value = postfold.get("semantic_layer_stack")
    if not isinstance(stack_value, Mapping):
        raise Lfm2SemanticFoldToReviewError(
            "Postfold omitted its sealed semantic layer-stack receipt."
        )
    stack = dict(stack_value)
    stack_result_sha256 = stack.pop("result_sha256", None)
    session_value = (
        stack_value.get("parent_checkpoint_session")
        if isinstance(stack_value, Mapping)
        else None
    )
    if not isinstance(session_value, Mapping):
        raise Lfm2SemanticFoldToReviewError(
            "Postfold omitted its authenticated direct-local parent tensor inventory."
        )
    session = dict(session_value)
    session_sha256 = session.pop("session_sha256", None)
    identity_value = session.get("parent_identity_reconstruction")
    if not isinstance(identity_value, Mapping):
        raise Lfm2SemanticFoldToReviewError(
            "Direct-local parent session omitted exact identity reconstruction."
        )
    identity = dict(identity_value)
    identity_sha256 = identity.pop("result_sha256", None)
    records = identity.get("checkpoint_tensor_records")
    expected_identity = str(expected_parent_identity_sha256).strip().lower()
    if (
        stack_result_sha256 != canonical_sha256(stack)
        or session_sha256 != canonical_sha256(session)
        or stack_value.get("parent_checkpoint_session_sha256") != session_sha256
        or session_value.get("schema") != "neural-forge-direct-parent-session-v2"
        or session_value.get("parent_model_identity_sha256") != expected_identity
        or session_value.get("parent_identity_reconstruction_sha256") != identity_sha256
        or identity_sha256 != canonical_sha256(identity)
        or identity.get("schema")
        != "neural-forge-direct-parent-exact-identity-reconstruction-v1"
        or identity.get("cache_parent_identity_sha256") != expected_identity
        or identity.get("reconstructed_parent_identity_sha256") != expected_identity
        or identity.get("exact_parent_identity_authenticated") is not True
        or identity.get("parent_model_objects_constructed") != 0
        or identity.get("parent_forward_calls") != 0
        or identity.get("network_access_performed") is not False
        or session_value.get("exact_parent_identity_authenticated") is not True
        or session_value.get("checkpoint_files_rehashed_after_materialization")
        is not True
        or session_value.get("shards_hashed_once") is not True
        or session_value.get("shards_opened_once") is not True
        or session_value.get("parent_model_objects_constructed") != 0
        or session_value.get("parent_forward_calls") != 0
        or session_value.get("network_access_performed") is not False
        or not isinstance(records, list)
        or not records
        or identity.get("checkpoint_tensor_count") != len(records)
        or identity.get("checkpoint_tensor_inventory_sha256")
        != canonical_sha256(records)
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Direct-local parent tensor inventory failed authentication."
        )
    names: set[str] = set()
    parameter_count = 0
    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise Lfm2SemanticFoldToReviewError(
                "Direct-local parent tensor inventory contains a malformed record."
            )
        name = record.get("name")
        shape = record.get("shape")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(shape, list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            )
        ):
            raise Lfm2SemanticFoldToReviewError(
                f"Direct-local parent tensor record {ordinal} has invalid geometry."
            )
        names.add(name)
        parameter_count += math.prod(shape)
    if (
        isinstance(expected_parent_unique_parameters, bool)
        or not isinstance(expected_parent_unique_parameters, int)
        or expected_parent_unique_parameters <= 0
        or parameter_count != expected_parent_unique_parameters
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Fresh local parent parameter count differs from the authenticated "
            "parent response envelope."
        )
    body = {
        "parent_identity_sha256": expected_identity,
        "semantic_layer_stack_result_sha256": stack_result_sha256,
        "parent_checkpoint_session_sha256": session_sha256,
        "parent_identity_reconstruction_sha256": identity_sha256,
        "checkpoint_tensor_inventory_sha256": identity[
            "checkpoint_tensor_inventory_sha256"
        ],
        "checkpoint_tensor_count": len(records),
        "unique_stored_parameter_elements": parameter_count,
        "envelope_parent_unique_parameters": expected_parent_unique_parameters,
        "fresh_local_count_matches_authenticated_envelope": True,
        "count_method": "sum_product_of_authenticated_local_safetensor_shapes",
        "parent_model_objects_constructed_for_count": 0,
        "parent_forward_calls_for_count": 0,
        "network_access_performed_for_count": False,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _digest(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Lfm2SemanticFoldToReviewError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _authenticate_parent_envelope_payload(
    envelope: Mapping[str, Any],
    *,
    prompt_ids: Sequence[str],
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> tuple[str, int, str]:
    """Validate every sealed parent-envelope field consumed by review."""

    parent_state_sha256 = _digest(
        envelope.get("parent_state_sha256"),
        label="parent envelope state SHA-256",
    )
    parent_unique_parameters = envelope.get("parent_unique_parameters")
    response_bank_sha256 = _digest(
        envelope.get("response_bank_sha256"),
        label="parent envelope response bank SHA-256",
    )
    responses = envelope.get("responses")
    prompt_count = envelope.get("prompt_count")
    if (
        isinstance(parent_unique_parameters, bool)
        or not isinstance(parent_unique_parameters, int)
        or parent_unique_parameters <= 0
        or isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or prompt_count != len(prompt_ids)
        or isinstance(envelope.get("maximum_prompt_tokens"), bool)
        or envelope.get("maximum_prompt_tokens") != maximum_prompt_tokens
        or isinstance(envelope.get("maximum_new_tokens"), bool)
        or envelope.get("maximum_new_tokens") != maximum_new_tokens
        or not isinstance(responses, list)
        or len(responses) != len(prompt_ids)
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Parent response envelope has malformed review metadata."
        )
    for prompt_id, response in zip(prompt_ids, responses, strict=True):
        if not isinstance(response, Mapping):
            raise Lfm2SemanticFoldToReviewError(
                "Parent response envelope contains a non-object response row."
            )
        response_token_ids = response.get("response_token_ids")
        if (
            response.get("id") != prompt_id
            or not isinstance(response.get("response"), str)
            or not isinstance(response_token_ids, list)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in response_token_ids
            )
            or not isinstance(response.get("diagnostics"), Mapping)
        ):
            raise Lfm2SemanticFoldToReviewError(
                f"Parent response envelope row {prompt_id!r} is malformed."
            )
    if canonical_sha256(responses) != response_bank_sha256:
        raise Lfm2SemanticFoldToReviewError(
            "Parent response envelope response-bank identity is invalid."
        )
    return (
        parent_state_sha256,
        int(parent_unique_parameters),
        response_bank_sha256,
    )


def _authenticate_review_preflight(
    *,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    trace_cache: Path,
    expected_cache_key: str,
    donor_tokenizer_directory: Path,
    parent_model_directory: Path,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> _AuthenticatedReviewPreflight:
    """Authenticate review/cache ancestry without opening trace tensor payloads."""

    prompt_file_pin = _digest(
        expected_prompts_file_sha256,
        label="expected prompts file SHA-256",
    )
    if _file_sha256(prompts_json) != prompt_file_pin:
        raise Lfm2SemanticFoldToReviewError("Whole-response prompt bytes changed.")
    prompts = load_review_prompts(prompts_json)
    prompt_rows = [dict(row) for row in prompts]
    prompt_set_sha256 = canonical_sha256(prompt_rows)
    prompt_ids = tuple(str(row["id"]) for row in prompts)
    try:
        envelope = _load_parent_envelope(
            parent_envelope,
            expected_envelope_sha256=expected_parent_envelope_sha256,
            expected_file_sha256=expected_parent_envelope_file_sha256,
            expected_prompt_set_sha256=prompt_set_sha256,
            expected_prompt_ids=prompt_ids,
        )
    except (
        AttributeError,
        Lfm2SemanticFrontierReviewError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise Lfm2SemanticFoldToReviewError(
            "Parent response envelope failed preflight authentication."
        ) from exc
    if (
        envelope.get("maximum_prompt_tokens") != maximum_prompt_tokens
        or envelope.get("maximum_new_tokens") != maximum_new_tokens
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Parent response envelope was captured at a different review horizon."
        )
    (
        parent_state_sha256,
        parent_unique_parameters,
        response_bank_sha256,
    ) = _authenticate_parent_envelope_payload(
        envelope,
        prompt_ids=prompt_ids,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    parent_identity_sha256 = _digest(
        envelope.get("parent_identity_sha256"),
        label="parent envelope model identity SHA-256",
    )
    tokenizer_sha256 = _digest(
        envelope.get("tokenizer_sha256"),
        label="parent envelope tokenizer SHA-256",
    )
    donor_tokenizer_sha256 = fingerprint_local_tokenizer(donor_tokenizer_directory)
    if donor_tokenizer_sha256 != tokenizer_sha256:
        raise Lfm2SemanticFoldToReviewError(
            "Parent response envelope belongs to a different donor tokenizer."
        )
    parent_tokenizer_json = parent_model_directory / "tokenizer.json"
    if parent_tokenizer_json.exists() or parent_tokenizer_json.is_symlink():
        if parent_tokenizer_json.is_symlink() or not parent_tokenizer_json.is_file():
            raise Lfm2SemanticFoldToReviewError(
                "Co-located parent tokenizer.json must be one regular local file."
            )
        if fingerprint_local_tokenizer(parent_model_directory) != tokenizer_sha256:
            raise Lfm2SemanticFoldToReviewError(
                "Parent response envelope belongs to a different co-located parent "
                "tokenizer."
            )

    expected_cache = _digest(expected_cache_key, label="expected cache key")
    cache = open_lfm2_circuit_trace_cache(trace_cache, verify=False)
    manifest = cache.manifest
    manifest_parent = manifest.get("parent")
    cache_manifest_sha256 = _digest(
        manifest.get("manifest_sha256"),
        label="trace cache manifest SHA-256",
    )
    if (
        cache.cache_key != expected_cache
        or manifest.get("cache_key") != expected_cache
        or not isinstance(manifest_parent, Mapping)
        or manifest_parent.get("identity_sha256") != parent_identity_sha256
        or cache.parent_fingerprint != parent_identity_sha256
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Trace-cache manifest, expected cache key, and parent response envelope "
            "do not share one parent identity."
        )
    return _AuthenticatedReviewPreflight(
        prompt_set_sha256=prompt_set_sha256,
        prompt_ids=prompt_ids,
        envelope_sha256=_digest(
            envelope.get("envelope_sha256"),
            label="parent envelope SHA-256",
        ),
        envelope_file_sha256=_file_sha256(parent_envelope),
        parent_state_sha256=parent_state_sha256,
        parent_identity_sha256=parent_identity_sha256,
        parent_unique_parameters=parent_unique_parameters,
        tokenizer_sha256=tokenizer_sha256,
        response_bank_sha256=response_bank_sha256,
        cache_key=expected_cache,
        cache_manifest_sha256=cache_manifest_sha256,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )


def _existing_directory(value: Path, *, label: str) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2SemanticFoldToReviewError(f"{label} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticFoldToReviewError(f"{label} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2SemanticFoldToReviewError(
            f"{label} must be one canonical real directory."
        )
    return resolved


def _existing_file(value: Path, *, label: str) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2SemanticFoldToReviewError(f"{label} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2SemanticFoldToReviewError(f"{label} is missing.") from exc
    if resolved != lexical or not resolved.is_file():
        raise Lfm2SemanticFoldToReviewError(f"{label} must be one canonical real file.")
    return resolved


def _new_run_directory(value: Path, *, readonly: Sequence[Path]) -> Path:
    output = value.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise Lfm2SemanticFoldToReviewError(
            f"Refusing to replace an existing run directory: {output}."
        )
    parent = _existing_directory(output.parent, label="run directory parent")
    output = parent / output.name
    for source in readonly:
        if output == source or source in output.parents or output in source.parents:
            raise Lfm2SemanticFoldToReviewError(
                f"Run directory overlaps read-only source {source}."
            )
    output.mkdir(mode=0o700)
    return output


def _atomic_json(
    path: Path, value: Mapping[str, Any], *, replace: bool = False
) -> None:
    payload = (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare_optional_behavior_observations(
    *,
    output_json: Path,
    recipe: Lfm2SemanticBodyRecipe,
    behavior_flow: Lfm2SemanticNonblockingBehaviorFlow,
    dossier_path: Path,
    dossier: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an optional freeform notebook, never a verdict or release gate."""

    body = {
        "schema": OPTIONAL_BEHAVIOR_OBSERVATIONS_SCHEMA,
        "status": "optional-nonbinary-observations-empty",
        "body_recipe_id": recipe.recipe_id,
        "body_recipe_sha256": recipe.recipe_sha256,
        "behavior_flow_policy_sha256": behavior_flow.policy_sha256,
        "whole_response_dossier": {
            "path": str(dossier_path),
            "file_sha256": _file_sha256(dossier_path),
            "dossier_sha256": dossier["dossier_sha256"],
        },
        "candidate_count": dossier["reviewed_candidate_count"],
        "prompt_count": dossier["prompt_count"],
        "observation_format": "optional-nonbinary-freeform",
        "manual_input_required": False,
        "binary_verdict_requested": False,
        "score_cutoff_requested": False,
        "accept_veto_publish_delete_stop_authority": False,
        "may_guide_next_hypothesis": True,
        "observations": [],
    }
    document = {
        **body,
        "observations_sha256": canonical_sha256(body),
    }
    _atomic_json(output_json, document)
    return {
        "path": str(output_json),
        "file_sha256": _file_sha256(output_json),
        **document,
    }


def _candidate_component(index: int, candidate_id: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("-", candidate_id).strip(".-")[:56] or "candidate"
    identity = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"{index:03d}-{cleaned}-{identity}"


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _assert_zero_mutation_policy(value: Mapping[str, Any], *, label: str) -> None:
    operation = value.get("operation_audit")
    if not isinstance(operation, Mapping):
        raise Lfm2SemanticFoldToReviewError(f"{label} omitted its operation audit.")
    forbidden = {
        "training_steps",
        "optimizer_constructions",
        "optimizer_steps",
        "autograd_grad_calls",
        "backward_calls",
        "parent_model_objects_constructed",
        "parent_model_forward_calls",
        "parent_model_generate_calls",
        "parent_forward_passes",
        "parent_cache_generation_calls",
        "parent_trace_recaptures",
        "trace_recaptures",
        "child_model_forward_calls",
        "child_model_generate_calls",
        "model_generate_calls",
        "new_parameters",
        "parameters_pruned",
        "parameters_quantized",
        "parameters_sparsified",
        "sidecar_parameters",
    }
    if any(operation.get(name, 0) != 0 for name in forbidden):
        raise Lfm2SemanticFoldToReviewError(
            f"{label} reports a forbidden training, recapture, or compression operation."
        )


def _continuous_compilation_receipt(
    *,
    postfold: Mapping[str, Any],
    capsule: Lfm2SemanticPreterminalCapsuleReference,
    bank: Lfm2SemanticTerminalFrontierBankReference,
) -> dict[str, Any]:
    fused = postfold.get("fused_semantic_terminal")
    if not isinstance(fused, Mapping):
        raise Lfm2SemanticFoldToReviewError(
            "Review-pending postfold omitted its fused terminal receipt."
        )
    body = {
        "schema": CONTINUOUS_COMPILATION_SCHEMA,
        "postfold_result_sha256": postfold["result_sha256"],
        "fused_terminal_receipt_sha256": fused["result_sha256"],
        "preterminal_capsule": capsule.as_dict(),
        "terminal_frontier_bank": bank.as_dict(),
        "terminal_result_sha256": bank.terminal_result_sha256,
        "preterminal_resume_mode": False,
        "same_live_cache_session_as_layer_stack": True,
        "cache_content_identity_matches_layer_stack": True,
        "frontier_exported_before_final_w2_installation": True,
        "terminal_solve_count": 1,
        "terminal_frontier_export_count": 1,
        "candidate_metric_threshold_used_for_export": False,
        "numeric_metrics_have_final_intelligence_authority": False,
        "holistic_whole_response_review_required": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    return {**body, "compilation_sha256": canonical_sha256(body)}


def _materialize_complete_structural_frontier(
    *,
    capsule: Lfm2SemanticPreterminalCapsuleReference,
    bank_reference: Lfm2SemanticTerminalFrontierBankReference,
    source_roots: Lfm2SemanticTerminalFrontierSourceRoots,
    compilation_receipt: Mapping[str, Any],
    materializations_directory: Path,
    receipt_json: Path,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    bank = load_lfm2_semantic_terminal_frontier_bank(
        bank_reference.path,
        expected_receipt_file_sha256=bank_reference.receipt_file_sha256,
        expected_receipt_sha256=bank_reference.receipt_sha256,
        expected_preterminal_capsule_receipt_sha256=(capsule.receipt_sha256),
        expected_terminal_result_sha256=bank_reference.terminal_result_sha256,
    )
    candidate_ids = tuple(bank_reference.candidate_ids)
    manifest_eligible = bank.manifest.get("structurally_eligible_candidate_ids")
    if (
        bank.reference.as_dict() != bank_reference.as_dict()
        or tuple(manifest_eligible or ()) != candidate_ids
        or bank.manifest.get("structurally_eligible_candidate_count")
        != len(candidate_ids)
        or bank.manifest.get("all_structurally_eligible_candidates_exported")
        is not True
        or bank.manifest.get("structurally_ineligible_candidates_exported") is not False
        or bank.manifest.get("structural_safety_is_only_candidate_veto") is not True
        or bank.manifest.get("candidate_metric_threshold_used_for_export") is not False
        or not candidate_ids
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Terminal bank does not expose exactly its structurally eligible review "
            "inventory."
        )
    if materializations_directory.exists() or materializations_directory.is_symlink():
        raise Lfm2SemanticFoldToReviewError(
            "Frontier materializations destination already exists."
        )
    if receipt_json.exists() or receipt_json.is_symlink():
        raise Lfm2SemanticFoldToReviewError("Frontier run receipt already exists.")
    materializations_directory.mkdir(mode=0o700)
    progress_path = materializations_directory / "progress.json"
    completed: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(candidate_ids, start=1):
        destination = materializations_directory / _candidate_component(
            index,
            candidate_id,
        )
        reference = materialize_lfm2_semantic_terminal_frontier_candidate(
            preterminal_capsule_directory=capsule.path,
            expected_preterminal_receipt_file_sha256=(capsule.receipt_file_sha256),
            expected_preterminal_receipt_sha256=capsule.receipt_sha256,
            expected_preterminal_bindings_sha256=capsule.bindings_sha256,
            terminal_frontier_bank_directory=bank_reference.path,
            expected_terminal_frontier_bank_receipt_file_sha256=(
                bank_reference.receipt_file_sha256
            ),
            expected_terminal_frontier_bank_receipt_sha256=(
                bank_reference.receipt_sha256
            ),
            expected_terminal_result_sha256=bank_reference.terminal_result_sha256,
            expected_source_roots=source_roots,
            candidate_id=candidate_id,
            output_directory=destination,
            device=device,
            dtype=dtype,
        )
        completed.append(reference.as_dict())
        progress_body = {
            "schema": FRONTIER_PROGRESS_SCHEMA,
            "status": "running" if index < len(candidate_ids) else "complete",
            "terminal_frontier_bank": bank_reference.as_dict(),
            "candidate_count": len(candidate_ids),
            "materialized_count": len(completed),
            "remaining_count": len(candidate_ids) - len(completed),
            "candidate_ids": list(candidate_ids),
            "materializations": list(completed),
            "all_candidates_retained_without_numeric_veto": True,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        _atomic_json(
            progress_path,
            {
                **progress_body,
                "progress_sha256": canonical_sha256(progress_body),
            },
            replace=True,
        )
    body = {
        "schema": FRONTIER_RUN_SCHEMA,
        "status": "complete",
        "preterminal_capsule": capsule.as_dict(),
        "terminal_frontier_bank": bank_reference.as_dict(),
        "terminal_frontier_compilation": dict(compilation_receipt),
        "source_roots": source_roots.as_dict(),
        "materializations_directory": str(materializations_directory),
        "progress_json": str(progress_path),
        "candidate_count": len(candidate_ids),
        "materialized_count": len(completed),
        "candidate_ids": list(candidate_ids),
        "materializations": completed,
        "all_retained_candidates_materialized": len(completed) == len(candidate_ids),
        "all_and_only_structurally_eligible_candidates_materialized": True,
        "candidate_metric_threshold_used_for_materialization": False,
        "numeric_metrics_are_diagnostics_only": True,
        "whole_response_intelligence_review_is_authoritative": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    document = {**body, "run_sha256": canonical_sha256(body)}
    _atomic_json(receipt_json, document)
    return document


def _retained_candidate_checkpoint_ids(
    frontier: Mapping[str, Any],
) -> tuple[str, ...]:
    """Prove that every decoded candidate checkpoint remains locally retained."""

    candidate_ids = frontier.get("candidate_ids")
    materializations = frontier.get("materializations")
    raw_materializations_directory = frontier.get("materializations_directory")
    materializations_directory = (
        Path(raw_materializations_directory).absolute()
        if isinstance(raw_materializations_directory, str)
        and raw_materializations_directory
        else None
    )
    if (
        not isinstance(candidate_ids, Sequence)
        or isinstance(candidate_ids, (str, bytes, bytearray))
        or not isinstance(materializations, Sequence)
        or isinstance(materializations, (str, bytes, bytearray))
        or len(candidate_ids) != len(materializations)
        or frontier.get("candidate_count") != len(candidate_ids)
        or frontier.get("materialized_count") != len(candidate_ids)
        or frontier.get("all_retained_candidates_materialized") is not True
        or materializations_directory is None
        or not materializations_directory.is_dir()
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Decoded behavior flow cannot prove complete checkpoint retention."
        )
    retained: list[str] = []
    for expected_id, materialization in zip(
        candidate_ids, materializations, strict=True
    ):
        if (
            not isinstance(expected_id, str)
            or not isinstance(materialization, Mapping)
            or materialization.get("candidate_id") != expected_id
        ):
            raise Lfm2SemanticFoldToReviewError(
                "Decoded behavior flow found a foreign materialized checkpoint."
            )
        checkpoint = Path(str(materialization.get("checkpoint_path", ""))).absolute()
        if (
            not checkpoint.is_dir()
            or checkpoint.is_symlink()
            or materializations_directory not in checkpoint.parents
        ):
            raise Lfm2SemanticFoldToReviewError(
                "Decoded behavior flow found an unretained candidate checkpoint."
            )
        retained.append(expected_id)
    return tuple(retained)


def _validate_pending_postfold(
    *,
    postfold: Mapping[str, Any],
    capsule: Lfm2SemanticPreterminalCapsuleReference,
    bank: Lfm2SemanticTerminalFrontierBankReference,
    unpublished_checkpoint: Path,
    preflight: _AuthenticatedReviewPreflight,
) -> None:
    fused = postfold.get("fused_semantic_terminal")
    reuse = postfold.get("pipeline_cache_session_reuse")
    context = (
        fused.get("terminal_frontier_export_context")
        if isinstance(
            fused,
            Mapping,
        )
        else None
    )
    if (
        postfold.get("status") != "terminal-frontier-exported-review-pending"
        or postfold.get("final_selection_status")
        != "deferred-to-manual-whole-response-review"
        or postfold.get("final_selected_candidate_id") is not None
        or postfold.get("final_selection_performed") is not False
        or postfold.get("automatic_candidate_selection_performed") is not False
        or postfold.get("automatic_candidate_acceptance_performed") is not False
        or postfold.get("candidate_published_before_manual_review") is not False
        or postfold.get("checkpoint_published") is not False
        or postfold.get("semantic_evidence_cache_key") != preflight.cache_key
        or postfold.get("cache_parent_fingerprint") != preflight.parent_identity_sha256
        or postfold.get("terminal_writer_install_count") != 0
        or postfold.get("terminal_writer_mutation_count") != 0
        or postfold.get("preterminal_capsule") != capsule.as_dict()
        or postfold.get("terminal_frontier_export") != bank.as_dict()
        or not isinstance(fused, Mapping)
        or fused.get("terminal_frontier_export") != bank.as_dict()
        or fused.get("terminal_writer_install_count") != 0
        or fused.get("terminal_writer_mutation_count") != 0
        or fused.get("final_selection_performed") is not False
        or not isinstance(context, Mapping)
        or context.get("preterminal_resume_mode") is not False
        or context.get("same_live_cache_session_as_layer_stack") is not True
        or context.get("frontier_export_occurs_before_final_w2_installation")
        is not True
        or not isinstance(reuse, Mapping)
        or reuse.get("same_pipeline_session_reused_by_layer_stack_and_terminal")
        is not True
        or unpublished_checkpoint.exists()
        or unpublished_checkpoint.is_symlink()
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Continuous postfold did not stop at one exported, review-pending "
            "frontier without selecting or publishing a child."
        )
    _assert_zero_mutation_policy(postfold, label="continuous postfold")
    _assert_zero_mutation_policy(fused, label="fused terminal")


def _run_fold_to_review_scale(
    *,
    run_directory: Path,
    input_checkpoint: Path,
    expected_input_artifact_sha256: str,
    trace_cache: Path,
    expected_cache_key: str,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    parent_model_directory: Path,
    compiler_diagnostic_json: Path,
    expected_compiler_diagnostic_sha256: str,
    interface_statistics_directory: Path,
    expected_interface_statistics_sha256: str,
    interface_frontier_companion_json: Path,
    expected_interface_frontier_companion_file_sha256: str,
    expected_interface_frontier_companion_sha256: str,
    anchor_candidate_index: int,
    body_recipe: str,
    expected_body_recipe_sha256: str,
    tied_interface_mode: str,
    device: str,
    fold_compute_device: str,
    dtype: str,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    global_residual_closure_version: str = "producer_aware_v2a",
    correction_scale_frontier: Sequence[float] = (1.0,),
    correction_scale: float = 1.0,
    global_residual_backend: str = "auto",
    global_residual_basis_rank: int | None = None,
) -> dict[str, Any]:
    """Run one pinned recipe through its sealed decoded-evidence flow."""

    if (
        isinstance(maximum_prompt_tokens, bool)
        or not isinstance(maximum_prompt_tokens, int)
        or not 1 <= maximum_prompt_tokens <= 8192
    ):
        raise Lfm2SemanticFoldToReviewError(
            "maximum_prompt_tokens must be between 1 and 8192."
        )
    if (
        isinstance(maximum_new_tokens, bool)
        or not isinstance(maximum_new_tokens, int)
        or not 16 <= maximum_new_tokens <= 2048
    ):
        raise Lfm2SemanticFoldToReviewError(
            "maximum_new_tokens must be between 16 and 2048."
        )
    if global_residual_basis_rank is not None and (
        isinstance(global_residual_basis_rank, bool)
        or not isinstance(global_residual_basis_rank, int)
        or global_residual_basis_rank <= 0
    ):
        raise Lfm2SemanticFoldToReviewError(
            "global_residual_basis_rank must be a positive integer or None."
        )
    directories = {
        "input checkpoint": input_checkpoint,
        "trace cache": trace_cache,
        "tokenizer bridge": bridge_directory,
        "donor tokenizer": donor_tokenizer_directory,
        "target tokenizer": target_tokenizer_directory,
        "parent model": parent_model_directory,
        "interface statistics": interface_statistics_directory,
    }
    files = {
        "compiler diagnostic": compiler_diagnostic_json,
        "interface frontier companion": interface_frontier_companion_json,
        "whole-response prompts": prompts_json,
        "cached parent response envelope": parent_envelope,
    }
    normalized_directories = {
        label: _existing_directory(path, label=label)
        for label, path in directories.items()
    }
    normalized_files = {
        label: _existing_file(path, label=label) for label, path in files.items()
    }
    recipe = resolve_lfm2_semantic_body_recipe(body_recipe)
    closure_version = str(global_residual_closure_version).strip().lower()
    if closure_version not in {
        "producer_aware_v2a",
        "output_writer_v1_diagnostic",
    }:
        raise Lfm2SemanticFoldToReviewError(
            "global_residual_closure_version must be producer_aware_v2a or "
            "output_writer_v1_diagnostic."
        )
    closure_policy = Lfm2GlobalResidualClosurePolicy(
        backend=str(global_residual_backend).strip().lower()
    )
    scale_policy = Lfm2ProducerAwareCorrectionScalePolicy(
        materialization_frontier=tuple(
            float(value) for value in correction_scale_frontier
        ),
        materialization_scale=float(correction_scale),
    )
    if (
        closure_version == "output_writer_v1_diagnostic"
        and global_residual_basis_rank is not None
    ):
        raise Lfm2SemanticFoldToReviewError(
            "The output-writer v1 diagnostic does not support an explicit "
            "residual basis rank."
        )
    if recipe.recipe_sha256 != _digest(
        expected_body_recipe_sha256,
        label="expected body recipe SHA-256",
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Selected semantic body recipe differs from its external pin."
        )
    behavior_flow = resolve_lfm2_semantic_nonblocking_behavior_flow(recipe)
    if isinstance(anchor_candidate_index, bool) or anchor_candidate_index < 0:
        raise Lfm2SemanticFoldToReviewError(
            "anchor_candidate_index must be a nonnegative integer."
        )
    preflight = _authenticate_review_preflight(
        prompts_json=normalized_files["whole-response prompts"],
        expected_prompts_file_sha256=expected_prompts_file_sha256,
        parent_envelope=normalized_files["cached parent response envelope"],
        expected_parent_envelope_sha256=expected_parent_envelope_sha256,
        expected_parent_envelope_file_sha256=(expected_parent_envelope_file_sha256),
        trace_cache=normalized_directories["trace cache"],
        expected_cache_key=expected_cache_key,
        donor_tokenizer_directory=normalized_directories["donor tokenizer"],
        parent_model_directory=normalized_directories["parent model"],
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    readonly = tuple(normalized_directories.values()) + tuple(normalized_files.values())
    run_root = _new_run_directory(run_directory, readonly=readonly)
    unpublished_checkpoint = run_root / "unselected-checkpoint"
    capsule_output = run_root / "preterminal-capsule"
    bank_output = run_root / "terminal-frontier-bank"
    postfold_receipt = run_root / "postfold-receipt.json"
    materializations = run_root / "materializations"
    frontier_receipt = run_root / "frontier-receipt.json"
    dossier_path = run_root / "whole-response-dossier.json"
    merged_path = run_root / "qualitative-merged.json"
    markdown_path = run_root / "qualitative-review.md"
    verdict_path = run_root / "manual-verdict-template.json"
    observations_path = run_root / "optional-behavior-observations.json"
    pipeline_receipt = run_root / "pipeline-receipt.json"

    donor_tokenizer = load_local_tokenizer(normalized_directories["donor tokenizer"])
    target_tokenizer = load_local_tokenizer(normalized_directories["target tokenizer"])
    capsules: list[Lfm2SemanticPreterminalCapsuleReference] = []
    banks: list[Lfm2SemanticTerminalFrontierBankReference] = []
    source_root_records: list[Lfm2SemanticTerminalFrontierSourceRoots] = []

    def exporter_factory(capsule_reference: Any):
        if not isinstance(
            capsule_reference,
            Lfm2SemanticPreterminalCapsuleReference,
        ):
            raise Lfm2SemanticFoldToReviewError(
                "Postfold exporter factory received no typed capsule reference."
            )
        if capsules:
            raise Lfm2SemanticFoldToReviewError(
                "Postfold attempted to construct more than one frontier exporter."
            )
        capsules.append(capsule_reference)
        source_roots = Lfm2SemanticTerminalFrontierSourceRoots(
            preterminal_capsule_directory=capsule_reference.path,
            semantic_cache_directory=normalized_directories["trace cache"],
            tokenizer_bridge_directory=normalized_directories["tokenizer bridge"],
            parent_model_directory=normalized_directories["parent model"],
            donor_tokenizer_directories=(normalized_directories["donor tokenizer"],),
            target_tokenizer_directories=(normalized_directories["target tokenizer"],),
        )
        source_root_records.append(source_roots)

        def exporter(result: Any, export_context: Mapping[str, Any]):
            if banks:
                raise Lfm2SemanticFoldToReviewError(
                    "Fused terminal attempted to export more than one frontier bank."
                )
            reference = publish_lfm2_semantic_terminal_frontier_bank(
                result,
                export_context,
                preterminal_capsule_reference=capsule_reference,
                source_roots=source_roots,
                output_directory=bank_output,
            )
            banks.append(reference)
            return reference

        return exporter

    postfold = dict(
        run_lfm2_semantic_postfold(
            input_checkpoint_directory=normalized_directories["input checkpoint"],
            expected_input_artifact_sha256=expected_input_artifact_sha256,
            output_checkpoint_directory=unpublished_checkpoint,
            cache_directory=normalized_directories["trace cache"],
            expected_cache_key=expected_cache_key,
            bridge_directory=normalized_directories["tokenizer bridge"],
            expected_bridge_artifact_sha256=expected_bridge_artifact_sha256,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            parent_model_directory=normalized_directories["parent model"],
            compiler_diagnostic_json=normalized_files["compiler diagnostic"],
            expected_compiler_diagnostic_sha256=(expected_compiler_diagnostic_sha256),
            interface_statistics_directory=(
                normalized_directories["interface statistics"]
            ),
            expected_interface_statistics_sha256=(expected_interface_statistics_sha256),
            interface_frontier_companion_json=(
                normalized_files["interface frontier companion"]
            ),
            expected_interface_frontier_companion_file_sha256=(
                expected_interface_frontier_companion_file_sha256
            ),
            expected_interface_frontier_companion_sha256=(
                expected_interface_frontier_companion_sha256
            ),
            anchor_candidate_index=anchor_candidate_index,
            device=device,
            fold_compute_device=fold_compute_device,
            global_residual_closure_version=closure_version,
            correction_scale_frontier=scale_policy.materialization_frontier,
            correction_scale=scale_policy.materialization_scale,
            global_residual_backend=closure_policy.backend,
            global_residual_basis_rank=global_residual_basis_rank,
            body_recipe=recipe,
            dtype=dtype,
            tied_interface_mode=tied_interface_mode,
            preterminal_capsule_directory=capsule_output,
            terminal_frontier_exporter_factory=exporter_factory,
        )
    )
    if len(capsules) != 1 or len(banks) != 1 or len(source_root_records) != 1:
        raise Lfm2SemanticFoldToReviewError(
            "Continuous postfold did not produce exactly one capsule and frontier."
        )
    capsule = capsules[0]
    bank = banks[0]
    source_roots = source_root_records[0]
    _validate_pending_postfold(
        postfold=postfold,
        capsule=capsule,
        bank=bank,
        unpublished_checkpoint=unpublished_checkpoint,
        preflight=preflight,
    )
    _atomic_json(postfold_receipt, postfold)
    compilation = _continuous_compilation_receipt(
        postfold=postfold,
        capsule=capsule,
        bank=bank,
    )
    _release_cuda()
    frontier = _materialize_complete_structural_frontier(
        capsule=capsule,
        bank_reference=bank,
        source_roots=source_roots,
        compilation_receipt=compilation,
        materializations_directory=materializations,
        receipt_json=frontier_receipt,
        device=device,
        dtype=dtype,
    )
    _release_cuda()
    dossier = review_frontier(
        frontier_run_receipt=frontier_receipt,
        expected_frontier_run_sha256=frontier["run_sha256"],
        expected_frontier_run_file_sha256=_file_sha256(frontier_receipt),
        prompts_json=normalized_files["whole-response prompts"],
        output_json=dossier_path,
        candidate_ids=(),
        device=device,
        dtype=dtype,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
        parent_envelope=normalized_files["cached parent response envelope"],
        expected_parent_envelope_sha256=preflight.envelope_sha256,
        expected_parent_envelope_file_sha256=preflight.envelope_file_sha256,
    )
    dossier_parent = dossier.get("parent_response_envelope")
    if (
        dossier.get("reviewed_candidate_count") != frontier["candidate_count"]
        or dossier.get("requested_candidate_ids") != frontier["candidate_ids"]
        or dossier.get("prompt_set_sha256") != preflight.prompt_set_sha256
        or dossier.get("maximum_prompt_tokens") != preflight.maximum_prompt_tokens
        or dossier.get("maximum_new_tokens") != preflight.maximum_new_tokens
        or not isinstance(dossier_parent, Mapping)
        or dossier_parent.get("envelope_sha256") != preflight.envelope_sha256
        or dossier_parent.get("parent_state_sha256") != preflight.parent_state_sha256
        or dossier_parent.get("parent_identity_sha256")
        != preflight.parent_identity_sha256
        or dossier_parent.get("parent_unique_parameters")
        != preflight.parent_unique_parameters
        or dossier_parent.get("tokenizer_sha256") != preflight.tokenizer_sha256
        or dossier_parent.get("maximum_new_tokens") != preflight.maximum_new_tokens
        or dossier_parent.get("response_bank_sha256") != preflight.response_bank_sha256
        or dossier.get("evaluation_policy", {}).get(
            "decoded_whole_responses_are_authoritative"
        )
        is not True
        or dossier.get("evaluation_policy", {}).get(
            "diagnostic_numbers_have_decision_authority"
        )
        is not False
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Whole-response dossier did not review the complete structural bank."
        )
    _release_cuda()
    layer_stack_receipt = postfold.get("semantic_layer_stack")
    local_parent_inventory = _authenticated_local_parent_parameter_inventory(
        postfold,
        expected_parent_identity_sha256=preflight.parent_identity_sha256,
        expected_parent_unique_parameters=preflight.parent_unique_parameters,
    )
    parent_parameter_count = local_parent_inventory["unique_stored_parameter_elements"]
    child_parameter_count = (
        layer_stack_receipt.get("parameter_element_count")
        if isinstance(layer_stack_receipt, Mapping)
        else None
    )
    if (
        type(child_parameter_count) is not int
        or child_parameter_count <= 0
        or postfold.get("global_residual_closure_version") != closure_version
        or postfold.get("global_residual_backend") != closure_policy.backend
        or postfold.get("global_residual_correction_scale_frontier")
        != list(scale_policy.materialization_frontier)
        or postfold.get("global_residual_correction_scale")
        != scale_policy.materialization_scale
        or postfold.get("global_residual_frontier_member_selected_as_winner")
        is not False
    ):
        raise Lfm2SemanticFoldToReviewError(
            "Postfold dropped the authenticated residual-closure controls or child "
            "parameter topology."
        )
    parameter_reduction = parent_parameter_count - child_parameter_count
    parameter_compaction = {
        "parent_unique_parameters": parent_parameter_count,
        "child_parameter_element_count": child_parameter_count,
        "parameter_reduction": parameter_reduction,
        "parent_to_child_parameter_ratio": (
            parent_parameter_count / child_parameter_count
        ),
        "child_parameter_fraction_of_parent": (
            child_parameter_count / parent_parameter_count
        ),
        "parameter_reduction_fraction": (parameter_reduction / parent_parameter_count),
        "parent_count_source": (
            "authenticated_direct_local_safetensor_tensor_inventory"
        ),
        "child_count_source": "authenticated_semantic_layer_stack_receipt",
        "local_parent_parameter_inventory": dict(local_parent_inventory),
        "envelope_parent_count_cross_checked": True,
        "counts_recomputed_from_authenticated_values": True,
        "dense_parameter_compaction_is_topology_evidence": True,
        "dense_parameter_compaction_is_quality_gate": False,
    }
    common_body = {
        "run_directory": str(run_root),
        "body_recipe": recipe.as_dict(),
        "body_recipe_sha256": recipe.recipe_sha256,
        "body_recipe_id": recipe.recipe_id,
        "global_residual_closure_version": closure_version,
        "global_residual_backend": closure_policy.backend,
        "global_residual_requested_basis_rank": global_residual_basis_rank,
        "global_residual_closure_policy_sha256": (closure_policy.authentication_sha256),
        "global_residual_correction_scale_frontier": list(
            scale_policy.materialization_frontier
        ),
        "global_residual_correction_scale": scale_policy.materialization_scale,
        "global_residual_correction_scale_policy_sha256": (
            scale_policy.authentication_sha256
        ),
        "global_residual_frontier_member_selected_as_winner": False,
        "global_residual_materialization_scale_is_execution_variant_not_winner": (
            closure_version == "producer_aware_v2a"
        ),
        "dense_parameter_compaction": parameter_compaction,
        "authenticated_review_preflight": preflight.as_dict(),
        "postfold_receipt": {
            "path": str(postfold_receipt),
            "file_sha256": _file_sha256(postfold_receipt),
            "result_sha256": postfold["result_sha256"],
        },
        "preterminal_capsule": capsule.as_dict(),
        "terminal_frontier_bank": bank.as_dict(),
        "continuous_terminal_frontier_compilation": compilation,
        "frontier_run_receipt": {
            "path": str(frontier_receipt),
            "file_sha256": _file_sha256(frontier_receipt),
            "run_sha256": frontier["run_sha256"],
        },
        "whole_response_dossier": {
            "path": str(dossier_path),
            "file_sha256": _file_sha256(dossier_path),
            "dossier_sha256": dossier["dossier_sha256"],
            "candidate_count": dossier["reviewed_candidate_count"],
            "prompt_count": dossier["prompt_count"],
            "maximum_prompt_tokens": maximum_prompt_tokens,
            "maximum_new_tokens": maximum_new_tokens,
        },
        "candidate_count": frontier["candidate_count"],
        "all_and_only_structurally_eligible_candidates_materialized": True,
        "all_materialized_candidates_received_whole_response_review": True,
        "automatic_candidate_selection_performed": False,
        "automatic_candidate_acceptance_performed": False,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    if behavior_flow is None:
        qualitative = prepare_qualitative_review(
            dossier_paths=(dossier_path,),
            expected_dossier_file_sha256s=(_file_sha256(dossier_path),),
            expected_dossier_sha256s=(dossier["dossier_sha256"],),
            lane_labels=(recipe.recipe_id,),
            merged_json=merged_path,
            markdown_path=markdown_path,
            verdict_template_json=verdict_path,
        )
        body = {
            "schema": PIPELINE_SCHEMA,
            "status": "ready-for-manual-whole-response-intelligence-review",
            **common_body,
            "qualitative_review": dict(qualitative),
            "numeric_metrics_are_diagnostics_only": True,
            "numeric_metrics_have_final_intelligence_authority": False,
            "candidate_published_before_manual_review": False,
            "blank_manual_verdict_prepared": True,
            "manual_whole_response_intelligence_review_is_authoritative": True,
        }
    else:
        retained_candidate_ids = _retained_candidate_checkpoint_ids(frontier)
        observations = _prepare_optional_behavior_observations(
            output_json=observations_path,
            recipe=recipe,
            behavior_flow=behavior_flow,
            dossier_path=dossier_path,
            dossier=dossier,
        )
        body = {
            "schema": NONBLOCKING_PIPELINE_SCHEMA,
            "status": "decoded-behavior-evidence-complete-continuation-unblocked",
            **common_body,
            "behavior_flow": behavior_flow.as_dict(),
            "behavior_flow_policy_sha256": behavior_flow.policy_sha256,
            "legacy_v1_pending_terminal_receipt_interpretation": {
                "source_status": postfold["status"],
                "source_final_selection_status": postfold["final_selection_status"],
                "source_postfold_result_sha256": postfold["result_sha256"],
                "interpreted_as_frontier_handoff_only": True,
                "source_manual_review_pending_label_has_no_v2_continuation_authority": True,
            },
            "legacy_v1_artifact_interpretation": {
                "historical_artifact_roles": [
                    "body-recipe-v1-authentication-provenance",
                    "postfold-frontier-handoff",
                    "continuous-frontier-compilation-provenance",
                    "frontier-materialization-provenance",
                ],
                "nested_manual_review_required_or_authoritative_labels_are_historical_only": True,
                "unchanged_v1_body_recipe_manual_authority_label_is_historical_only": True,
                "nested_v1_quality_labels_have_v2_accept_veto_publish_delete_stop_authority": False,
                "nested_v1_quality_labels_have_v2_continuation_authority": False,
            },
            "optional_behavior_observations": observations,
            "structural_authentication_is_only_materialization_veto": True,
            "all_structurally_safe_candidates_must_be_materialized": True,
            "all_materialized_candidates_must_receive_decoded_review": True,
            "decoded_behavior_evidence_complete": True,
            "all_materialized_candidate_checkpoints_retained": True,
            "retained_candidate_checkpoint_count": len(retained_candidate_ids),
            "retained_candidate_checkpoint_ids": list(retained_candidate_ids),
            "candidate_checkpoint_retention_depends_on_manual_verdict": False,
            "candidate_checkpoints_are_evidence_not_accepted_or_rejected": True,
            "decoded_complete_behavior_is_primary_intelligence_evidence": True,
            "retention_target_is_objective_not_cutoff": True,
            "numeric_metrics_may_prioritize_investigation": True,
            "numeric_accept_veto_publish_delete_stop_authority": False,
            "manual_annotations_may_guide_next_hypothesis": True,
            "manual_or_binary_accept_veto_publish_delete_stop_authority": False,
            "manual_verdict_required_for_continuation": False,
            "continuation_requires_manual_review_or_verdict": False,
            "manual_verdict_artifact_created": False,
            "blank_manual_verdict_prepared": False,
            "quality_evidence_can_stop_iteration": False,
            "continuation_unblocked": True,
        }
    document = {**body, "pipeline_sha256": canonical_sha256(body)}
    _atomic_json(pipeline_receipt, document)
    return document


def run_fold_to_review(
    *,
    run_directory: Path,
    input_checkpoint: Path,
    expected_input_artifact_sha256: str,
    trace_cache: Path,
    expected_cache_key: str,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    parent_model_directory: Path,
    compiler_diagnostic_json: Path,
    expected_compiler_diagnostic_sha256: str,
    interface_statistics_directory: Path,
    expected_interface_statistics_sha256: str,
    interface_frontier_companion_json: Path,
    expected_interface_frontier_companion_file_sha256: str,
    expected_interface_frontier_companion_sha256: str,
    anchor_candidate_index: int,
    body_recipe: str,
    expected_body_recipe_sha256: str,
    tied_interface_mode: str,
    device: str,
    fold_compute_device: str,
    dtype: str,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    global_residual_closure_version: str = "producer_aware_v2a",
    correction_scale_frontier: Sequence[float] = (1.0,),
    correction_scale: float | None = None,
    global_residual_backend: str = "auto",
    global_residual_basis_rank: int | None = None,
) -> dict[str, Any]:
    """Fold, decode, and retain every requested correction-scale member."""

    frontier = tuple(float(value) for value in correction_scale_frontier)
    reference_scale = (
        frontier[0] if correction_scale is None and frontier else correction_scale
    )
    try:
        scale_policy = Lfm2ProducerAwareCorrectionScalePolicy(
            materialization_frontier=frontier,
            materialization_scale=float(reference_scale),
        )
        closure_policy = Lfm2GlobalResidualClosurePolicy(
            backend=str(global_residual_backend).strip().lower()
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2SemanticFoldToReviewError(
            "Correction-scale frontier or global residual backend is invalid."
        ) from exc
    closure_version = str(global_residual_closure_version).strip().lower()
    if global_residual_basis_rank is not None and (
        isinstance(global_residual_basis_rank, bool)
        or not isinstance(global_residual_basis_rank, int)
        or global_residual_basis_rank <= 0
    ):
        raise Lfm2SemanticFoldToReviewError(
            "global_residual_basis_rank must be a positive integer or None."
        )
    if closure_version not in {
        "producer_aware_v2a",
        "output_writer_v1_diagnostic",
    }:
        raise Lfm2SemanticFoldToReviewError(
            "global_residual_closure_version must be producer_aware_v2a or "
            "output_writer_v1_diagnostic."
        )
    if (
        closure_version == "output_writer_v1_diagnostic"
        and (
            scale_policy.materialization_frontier != (1.0,)
            or global_residual_basis_rank is not None
        )
    ):
        raise Lfm2SemanticFoldToReviewError(
            "The output-writer v1 diagnostic supports only its canonical scale "
            "1.0 and legacy topology-derived residual basis rank."
        )
    arguments = {
        "input_checkpoint": input_checkpoint,
        "expected_input_artifact_sha256": expected_input_artifact_sha256,
        "trace_cache": trace_cache,
        "expected_cache_key": expected_cache_key,
        "bridge_directory": bridge_directory,
        "expected_bridge_artifact_sha256": expected_bridge_artifact_sha256,
        "donor_tokenizer_directory": donor_tokenizer_directory,
        "target_tokenizer_directory": target_tokenizer_directory,
        "parent_model_directory": parent_model_directory,
        "compiler_diagnostic_json": compiler_diagnostic_json,
        "expected_compiler_diagnostic_sha256": expected_compiler_diagnostic_sha256,
        "interface_statistics_directory": interface_statistics_directory,
        "expected_interface_statistics_sha256": expected_interface_statistics_sha256,
        "interface_frontier_companion_json": interface_frontier_companion_json,
        "expected_interface_frontier_companion_file_sha256": (
            expected_interface_frontier_companion_file_sha256
        ),
        "expected_interface_frontier_companion_sha256": (
            expected_interface_frontier_companion_sha256
        ),
        "anchor_candidate_index": anchor_candidate_index,
        "body_recipe": body_recipe,
        "expected_body_recipe_sha256": expected_body_recipe_sha256,
        "tied_interface_mode": tied_interface_mode,
        "device": device,
        "fold_compute_device": fold_compute_device,
        "dtype": dtype,
        "prompts_json": prompts_json,
        "expected_prompts_file_sha256": expected_prompts_file_sha256,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "parent_envelope": parent_envelope,
        "expected_parent_envelope_sha256": expected_parent_envelope_sha256,
        "expected_parent_envelope_file_sha256": expected_parent_envelope_file_sha256,
        "global_residual_closure_version": closure_version,
        "correction_scale_frontier": scale_policy.materialization_frontier,
        "global_residual_backend": closure_policy.backend,
        "global_residual_basis_rank": global_residual_basis_rank,
    }
    if len(scale_policy.materialization_frontier) == 1:
        return _run_fold_to_review_scale(
            run_directory=run_directory,
            correction_scale=scale_policy.materialization_frontier[0],
            **arguments,
        )

    readonly = (
        input_checkpoint,
        trace_cache,
        bridge_directory,
        donor_tokenizer_directory,
        target_tokenizer_directory,
        parent_model_directory,
        compiler_diagnostic_json,
        interface_statistics_directory,
        interface_frontier_companion_json,
        prompts_json,
        parent_envelope,
    )
    outer = _new_run_directory(
        run_directory,
        readonly=tuple(path.expanduser().absolute() for path in readonly),
    )
    scale_runs = []
    compaction_evidence = None
    for index, scale in enumerate(scale_policy.materialization_frontier):
        scale_label = _SAFE_COMPONENT.sub(
            "-",
            format(scale, ".12g").replace("-", "neg-").replace(".", "p"),
        )
        member_directory = outer / f"scale-{index:02d}-{scale_label}"
        member = _run_fold_to_review_scale(
            run_directory=member_directory,
            correction_scale=scale,
            **arguments,
        )
        observed_compaction = member.get("dense_parameter_compaction")
        if (
            not isinstance(observed_compaction, Mapping)
            or type(member.get("candidate_count")) is not int
            or member["candidate_count"] <= 0
            or (
                compaction_evidence is not None
                and dict(observed_compaction) != compaction_evidence
            )
            or (
                member.get("all_materialized_candidates_received_whole_response_review")
                is not True
                or member.get("all_materialized_candidate_checkpoints_retained")
                is not True
            )
        ):
            raise Lfm2SemanticFoldToReviewError(
                "Correction-scale members must share one dense topology and each "
                "complete decoded review with every checkpoint retained."
            )
        compaction_evidence = dict(observed_compaction)
        member_receipt = member_directory / "pipeline-receipt.json"
        scale_runs.append(
            {
                "correction_scale": scale,
                "run_directory": str(member_directory),
                "pipeline_receipt": str(member_receipt),
                "pipeline_receipt_file_sha256": _file_sha256(member_receipt),
                "pipeline_sha256": member["pipeline_sha256"],
                "status": member["status"],
                "candidate_count": member["candidate_count"],
                "all_materialized_candidates_received_whole_response_review": member[
                    "all_materialized_candidates_received_whole_response_review"
                ],
                "all_materialized_candidate_checkpoints_retained": member.get(
                    "all_materialized_candidate_checkpoints_retained",
                    False,
                ),
                "selected_as_winner": False,
            }
        )
        _release_cuda()
    body = {
        "schema": CORRECTION_SCALE_FRONTIER_SCHEMA,
        "status": "all-correction-scales-folded-decoded-retained",
        "run_directory": str(outer),
        "global_residual_closure_version": closure_version,
        "global_residual_backend": closure_policy.backend,
        "global_residual_closure_policy_sha256": (closure_policy.authentication_sha256),
        "correction_scale_frontier": list(scale_policy.materialization_frontier),
        "correction_scale_policy_reference_member": float(reference_scale),
        "correction_scale_policy_sha256_by_member": [
            Lfm2ProducerAwareCorrectionScalePolicy(
                materialization_frontier=scale_policy.materialization_frontier,
                materialization_scale=scale,
            ).authentication_sha256
            for scale in scale_policy.materialization_frontier
        ],
        "scale_run_count": len(scale_runs),
        "scale_runs": scale_runs,
        "dense_parameter_compaction": compaction_evidence,
        "every_requested_scale_separately_folded": True,
        "every_requested_scale_separately_decoded": True,
        "every_requested_scale_retained": True,
        "automatic_scale_winner_selected": False,
        "numeric_or_manual_quality_gate": False,
        "decoded_outputs_and_metrics_are_evidence_for_iteration": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    document = {**body, "pipeline_sha256": canonical_sha256(body)}
    _atomic_json(outer / "pipeline-receipt.json", document)
    return document


def _parse_correction_scale_frontier(value: str) -> tuple[float, ...]:
    try:
        frontier = tuple(float(item.strip()) for item in value.split(","))
        selected = frontier[0]
        return Lfm2ProducerAwareCorrectionScalePolicy(
            materialization_frontier=frontier,
            materialization_scale=selected,
        ).materialization_frontier
    except (TypeError, ValueError, RuntimeError, IndexError) as exc:
        raise argparse.ArgumentTypeError(
            "correction-scale frontier must be sorted unique nonnegative CSV floats"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fold one pinned LFM2 body recipe into a complete structural terminal "
            "frontier and decoded behavior dossier without training or automatic "
            "quality gating. Coupled Q-DEIM recipes continue under the sealed "
            "nonblocking v2 behavior policy."
        )
    )
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--input-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-input-artifact-sha256", required=True)
    parser.add_argument("--trace-cache", required=True, type=Path)
    parser.add_argument("--expected-cache-key", required=True)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument("--compiler-diagnostic-json", required=True, type=Path)
    parser.add_argument("--expected-compiler-diagnostic-sha256", required=True)
    parser.add_argument(
        "--interface-statistics-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--expected-interface-statistics-sha256", required=True)
    parser.add_argument(
        "--interface-frontier-companion-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-interface-frontier-companion-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-interface-frontier-companion-sha256",
        required=True,
    )
    parser.add_argument("--anchor-candidate-index", required=True, type=int)
    parser.add_argument(
        "--body-recipe",
        required=True,
        choices=LFM2_SEMANTIC_BODY_RECIPE_IDS,
    )
    parser.add_argument("--expected-body-recipe-sha256", required=True)
    parser.add_argument(
        "--tied-interface-mode",
        choices=TIED_RESIDUAL_CONGRUENCE_MODES,
        default=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold-compute-device", default="cuda:0")
    parser.add_argument(
        "--global-residual-closure-version",
        choices=("producer_aware_v2a", "output_writer_v1_diagnostic"),
        default="producer_aware_v2a",
    )
    parser.add_argument(
        "--correction-scale-frontier",
        type=_parse_correction_scale_frontier,
        default=(1.0,),
        help=(
            "Sorted unique CSV scales. Every member is independently folded, "
            "decoded, and retained; no winner is selected."
        ),
    )
    parser.add_argument(
        "--correction-scale",
        type=float,
        default=None,
        help="Optional reference member for receipt ordering; it is not a winner.",
    )
    parser.add_argument(
        "--global-residual-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--global-residual-basis-rank",
        type=int,
        default=None,
        help=(
            "Optional explicit whole-stack residual basis rank. Omit to retain "
            "the legacy 2*attention-head-dimension rank; use the child hidden "
            "width for full residual coverage."
        ),
    )
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_fold_to_review(
        run_directory=args.run_directory,
        input_checkpoint=args.input_checkpoint,
        expected_input_artifact_sha256=args.expected_input_artifact_sha256,
        trace_cache=args.trace_cache,
        expected_cache_key=args.expected_cache_key,
        bridge_directory=args.bridge_directory,
        expected_bridge_artifact_sha256=args.expected_bridge_artifact_sha256,
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        parent_model_directory=args.parent_model_directory,
        compiler_diagnostic_json=args.compiler_diagnostic_json,
        expected_compiler_diagnostic_sha256=(args.expected_compiler_diagnostic_sha256),
        interface_statistics_directory=args.interface_statistics_directory,
        expected_interface_statistics_sha256=(
            args.expected_interface_statistics_sha256
        ),
        interface_frontier_companion_json=(args.interface_frontier_companion_json),
        expected_interface_frontier_companion_file_sha256=(
            args.expected_interface_frontier_companion_file_sha256
        ),
        expected_interface_frontier_companion_sha256=(
            args.expected_interface_frontier_companion_sha256
        ),
        anchor_candidate_index=args.anchor_candidate_index,
        body_recipe=args.body_recipe,
        expected_body_recipe_sha256=args.expected_body_recipe_sha256,
        tied_interface_mode=args.tied_interface_mode,
        device=args.device,
        fold_compute_device=args.fold_compute_device,
        global_residual_closure_version=args.global_residual_closure_version,
        correction_scale_frontier=args.correction_scale_frontier,
        correction_scale=args.correction_scale,
        global_residual_backend=args.global_residual_backend,
        global_residual_basis_rank=args.global_residual_basis_rank,
        dtype=args.dtype,
        prompts_json=args.prompts_json,
        expected_prompts_file_sha256=args.expected_prompts_file_sha256,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
        maximum_new_tokens=args.maximum_new_tokens,
        parent_envelope=args.parent_envelope,
        expected_parent_envelope_sha256=args.expected_parent_envelope_sha256,
        expected_parent_envelope_file_sha256=(
            args.expected_parent_envelope_file_sha256
        ),
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
