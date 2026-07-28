#!/usr/bin/env python3
"""Decode component, subsystem-grouped, and cumulative non-final LFM2 hybrids.

This runner consumes two existing local checkpoints: the original parent shell
and an already-folded compact child.  A sealed plan names corresponding
attention, MLP, short-convolution, or terminal-readout modules.  Dense bridge
tensors come from one independently pinned local safetensors archive; normally
the parent-to-child map is the exact4x foundation ``producer_encoder`` and the
child-to-parent map is the layer candidate ``normalized_input_decoder`` (or an
explicitly authenticated analytic decoder derived during fold preparation).

For every planned component, the runner materializes an isolated hybrid.  It
also groups every repeated attention, MLP, and short-convolution subsystem,
tests topology-ordered cumulative subsystem groups, and preserves the original
ordered component prefixes.  Duplicate component sets are omitted.  Every
hybrid is saved, freshly reconstructed through the custom LFM2 diagnostic
loader, and decoded on the complete prompt bank.  Cached parent text is
attached for human comparison.
No parent generation, trace recapture, training, gradient, optimizer, pruning,
quantization, network loader, or automatic intelligence gate is used.

These artifacts retain the parent shell and bridges.  They are evidence about
component folding and composition only, never standalone compression results.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
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

from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _generate_rows,
    _load_parent_envelope,
    load_review_prompts,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    Lfm2ProgressiveComponentAblationError,
    Lfm2ProgressiveComponentCapsule,
    _module_state_sha256,
    apply_lfm2_component_ablation_candidate,
    build_lfm2_progressive_component_ablation_schedule,
    load_lfm2_progressive_hybrid_checkpoint,
    schedule_receipt,
    write_lfm2_progressive_hybrid_checkpoint,
)

PLAN_SCHEMA = "neural-forge-lfm2-progressive-component-ablation-plan-v1"
RUN_SCHEMA = "neural-forge-lfm2-progressive-component-ablation-run-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-progressive-component-ablation-progress-v1"
BRIDGE_ARCHIVE_SCHEMA = "neural-forge-lfm2-component-bridge-archive-reference-v1"
PLAN_PRODUCER_SCHEMA = "neural-forge-lfm2-component-ablation-plan-producer-v1"
PRODUCER_RECEIPT_FILENAME = "producer-receipt.json"
PLAN_FILENAME = "component-plan.json"
BRIDGE_FILENAME = "component-bridges.safetensors"
AUTOMATIC_PLAN_STATUS = "NON-FINAL-COMPONENT-ABLATION-PLAN-AUTOMATICALLY-DERIVED"
PRODUCER_RECEIPT_STATUS = "component-ablation-bridge-and-plan-publication-complete"


class Lfm2ProgressiveComponentAblationRunError(RuntimeError):
    """A plan, local asset, hybrid, or decoded-evidence invariant failed."""


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_file(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ProgressiveComponentAblationRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Lfm2ProgressiveComponentAblationRunError(f"{label} is missing or unsafe.")
    return path


def _ordinary_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ProgressiveComponentAblationRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2ProgressiveComponentAblationRunError(f"{label} is missing or unsafe.")
    return path


def _new_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ProgressiveComponentAblationRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2ProgressiveComponentAblationRunError(
            f"{label} must not already exist."
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2ProgressiveComponentAblationRunError(
            f"{label} parent must be one ordinary directory."
        )
    path.mkdir()
    return path


def _checkpoint_inventory(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise Lfm2ProgressiveComponentAblationRunError(
            "Checkpoint inventory cannot be empty."
        )
    return rows


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    if path.is_symlink() or (path.exists() and not replace):
        raise Lfm2ProgressiveComponentAblationRunError(
            f"Refusing to replace evidence file: {path.name!r}."
        )
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise Lfm2ProgressiveComponentAblationRunError(
            "A stale progress temporary file already exists."
        )
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    temporary.write_text(payload + "\n", encoding="ascii")
    if replace and path.exists():
        path.unlink()
    temporary.replace(path)


def _emit(stage: str, **values: Any) -> None:
    print(
        json.dumps({"stage": stage, **values}, allow_nan=False, sort_keys=True),
        flush=True,
    )


def _load_plan(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_file_sha256: str,
) -> dict[str, Any]:
    ordinary = _ordinary_file(path, label="component plan")
    if _file_sha256(ordinary) != _digest(
        expected_file_sha256,
        label="expected component-plan file",
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Component-plan file differs from its external pin."
        )
    try:
        document = json.loads(ordinary.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2ProgressiveComponentAblationRunError(
            "Component plan is not valid JSON."
        ) from exc
    if not isinstance(document, dict):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Component plan must be one JSON object."
        )
    body = dict(document)
    supplied = body.pop("plan_sha256", None)
    components = document.get("components")
    bridge = document.get("bridge_archive")
    source_authentication = document.get("source_authentication")
    if (
        document.get("schema") != PLAN_SCHEMA
        or document.get("status") != AUTOMATIC_PLAN_STATUS
        or document.get("producer_schema") != PLAN_PRODUCER_SCHEMA
        or supplied != canonical_sha256(body)
        or supplied != _digest(expected_plan_sha256, label="expected component plan")
        or not isinstance(components, list)
        or not components
        or not isinstance(bridge, Mapping)
        or bridge.get("schema") != BRIDGE_ARCHIVE_SCHEMA
        or not isinstance(source_authentication, Mapping)
        or not isinstance(
            source_authentication.get("component_fold_receipts"),
            Mapping,
        )
        or document.get("component_granularity") != "whole-subsystem"
        or document.get("q_k_v_o_or_individual_head_isolation_claimed") is not False
        or document.get(
            "all_included_components_require_literal_parameter_reduction_true"
        )
        is not True
        or document.get("diagnostic_hybrid_only") is not True
        or document.get("standalone_compression_claimed") is not False
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Component plan seal, schema, or non-final policy is invalid."
        )
    return document


def _load_producer_receipt(
    path: Path,
    *,
    expected_receipt_sha256: str,
    expected_file_sha256: str,
) -> dict[str, Any]:
    ordinary = _ordinary_file(path, label="automatic producer receipt")
    if ordinary.name != PRODUCER_RECEIPT_FILENAME:
        raise Lfm2ProgressiveComponentAblationRunError(
            f"Automatic producer receipt must be named {PRODUCER_RECEIPT_FILENAME!r}."
        )
    observed_file_sha256 = _file_sha256(ordinary)
    if observed_file_sha256 != _digest(
        expected_file_sha256,
        label="expected producer-receipt file",
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Producer-receipt file differs from its external pin."
        )
    try:
        document = json.loads(ordinary.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2ProgressiveComponentAblationRunError(
            "Automatic producer receipt is not valid JSON."
        ) from exc
    if not isinstance(document, dict):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Automatic producer receipt must be one JSON object."
        )
    body = dict(document)
    supplied = body.pop("receipt_sha256", None)
    expected_logical = _digest(
        expected_receipt_sha256,
        label="expected producer receipt",
    )
    component_count = document.get("component_count")
    excluded_count = document.get("excluded_component_count")
    if (
        document.get("schema") != PLAN_PRODUCER_SCHEMA
        or document.get("status") != PRODUCER_RECEIPT_STATUS
        or supplied != canonical_sha256(body)
        or supplied != expected_logical
        or document.get("plan_filename") != PLAN_FILENAME
        or document.get("bridge_filename") != BRIDGE_FILENAME
        or document.get("component_granularity") != "whole-subsystem"
        or document.get("automatic_topology_and_component_derivation_performed")
        is not True
        or document.get("fresh_source_receipt_authentication_performed") is not True
        or document.get("fresh_archive_authentication_performed") is not True
        or document.get("human_authored_module_or_receipt_pins_accepted") is not False
        or isinstance(component_count, bool)
        or not isinstance(component_count, int)
        or component_count <= 0
        or isinstance(excluded_count, bool)
        or not isinstance(excluded_count, int)
        or excluded_count < 0
        or document.get("training_steps") != 0
        or document.get("model_forward_calls") != 0
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Automatic producer receipt seal, schema, derivation policy, or counts are invalid."
        )
    for field in (
        "plan_sha256",
        "plan_file_sha256",
        "bridge_file_sha256",
        "source_run_result_sha256",
        "parent_checkpoint_inventory_sha256",
        "compact_checkpoint_inventory_sha256",
    ):
        _digest(document.get(field), label=f"producer receipt {field}")
    return document


def _authenticate_producer_publication(
    *,
    producer_receipt: Mapping[str, Any],
    producer_receipt_path: Path,
    plan: Mapping[str, Any],
    component_plan: Path,
    bridge_tensors: Path,
    expected_component_plan_sha256: str,
    expected_component_plan_file_sha256: str,
    expected_bridge_tensors_file_sha256: str,
) -> None:
    receipt_path = _ordinary_file(
        producer_receipt_path,
        label="automatic producer receipt",
    )
    plan_path = _ordinary_file(component_plan, label="component plan")
    bridge_path = _ordinary_file(bridge_tensors, label="component bridge archive")
    source_authentication = plan.get("source_authentication")
    excluded_components = plan.get("excluded_components")
    if not isinstance(source_authentication, Mapping) or not isinstance(
        excluded_components,
        list,
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Automatic component plan lacks source authentication or exclusions."
        )
    expected_plan = _digest(
        expected_component_plan_sha256,
        label="expected component plan",
    )
    expected_plan_file = _digest(
        expected_component_plan_file_sha256,
        label="expected component-plan file",
    )
    expected_bridge_file = _digest(
        expected_bridge_tensors_file_sha256,
        label="expected bridge archive file",
    )
    parent_inventory = producer_receipt["parent_checkpoint_inventory_sha256"]
    compact_inventory = producer_receipt["compact_checkpoint_inventory_sha256"]
    if (
        receipt_path.parent != plan_path.parent
        or receipt_path.parent != bridge_path.parent
        or plan_path.name != producer_receipt.get("plan_filename")
        or bridge_path.name != producer_receipt.get("bridge_filename")
        or producer_receipt.get("plan_sha256") != expected_plan
        or plan.get("plan_sha256") != expected_plan
        or producer_receipt.get("plan_file_sha256") != expected_plan_file
        or _file_sha256(plan_path) != expected_plan_file
        or producer_receipt.get("bridge_file_sha256") != expected_bridge_file
        or _file_sha256(bridge_path) != expected_bridge_file
        or plan.get("bridge_archive", {}).get("filename")
        != producer_receipt.get("bridge_filename")
        or plan.get("bridge_archive", {}).get("file_sha256") != expected_bridge_file
        or producer_receipt.get("source_run_result_sha256")
        != source_authentication.get("source_run_result_sha256")
        or parent_inventory != plan.get("parent_checkpoint_inventory_sha256")
        or parent_inventory
        != source_authentication.get("parent_checkpoint_inventory_sha256")
        or compact_inventory != plan.get("compact_checkpoint_inventory_sha256")
        or compact_inventory
        != source_authentication.get("compact_checkpoint_inventory_sha256")
        or producer_receipt.get("component_count") != len(plan.get("components", ()))
        or producer_receipt.get("excluded_component_count") != len(excluded_components)
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Plan or bridge inputs are not the publication bound to the externally "
            "pinned automatic producer receipt."
        )


def _load_bridge_tensors(
    path: Path,
    *,
    expected_file_sha256: str,
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - required runtime
        raise Lfm2ProgressiveComponentAblationRunError(
            "Bridge archive loading requires safetensors."
        ) from exc
    ordinary = _ordinary_file(path, label="component bridge archive")
    observed_file = _file_sha256(ordinary)
    expected_file = _digest(
        expected_file_sha256,
        label="expected bridge archive file",
    )
    reference = plan["bridge_archive"]
    if observed_file != expected_file or reference.get("file_sha256") != expected_file:
        raise Lfm2ProgressiveComponentAblationRunError(
            "Bridge archive differs from its plan and external pin."
        )
    required_names: set[str] = set()
    for component in plan["components"]:
        if not isinstance(component, Mapping):
            raise Lfm2ProgressiveComponentAblationRunError(
                "Component-plan row is malformed."
            )
        for field in (
            "input_encoder_tensor",
            "output_decoder_tensor",
            "output_domain_decoder_tensor",
        ):
            value = component.get(field)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise Lfm2ProgressiveComponentAblationRunError(
                        f"{field} must name one bridge tensor."
                    )
                required_names.add(value)
    tensors = {}
    with safe_open(str(ordinary), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = sorted(required_names - available)
        if missing:
            raise Lfm2ProgressiveComponentAblationRunError(
                f"Bridge archive lacks required tensors: {missing!r}."
            )
        for name in sorted(required_names):
            tensor = handle.get_tensor(name).detach().cpu().contiguous()
            tensor.requires_grad_(False)
            tensors[name] = tensor
    return tensors


def _optional_digest(value: Any, *, label: str) -> str | None:
    return None if value is None else _digest(value, label=label)


def _build_capsules(
    *,
    plan: Mapping[str, Any],
    bridge_tensors: Mapping[str, Any],
    parent_model: Any,
    compact_model: Any,
) -> tuple[Lfm2ProgressiveComponentCapsule, ...]:
    capsules = []
    source_authentication = plan.get("source_authentication")
    if not isinstance(source_authentication, Mapping):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Automatically derived plan lacks source authentication."
        )
    verified_fold_receipts = source_authentication.get("component_fold_receipts")
    if not isinstance(verified_fold_receipts, Mapping):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Automatically derived plan lacks verified component fold receipts."
        )
    planned_compact_checkpoint = _digest(
        plan.get("compact_checkpoint_inventory_sha256"),
        label="planned compact checkpoint inventory",
    )
    for index, raw in enumerate(plan["components"]):
        if not isinstance(raw, Mapping):
            raise Lfm2ProgressiveComponentAblationRunError(
                "Component-plan row is malformed."
            )
        component_id = raw.get("component_id")
        kind = raw.get("kind")
        parent_path = raw.get("parent_module_path")
        compact_path = raw.get("compact_module_path")
        input_name = raw.get("input_encoder_tensor")
        output_name = raw.get("output_decoder_tensor")
        domain_name = raw.get("output_domain_decoder_tensor")
        if (
            not isinstance(component_id, str)
            or not isinstance(kind, str)
            or not isinstance(parent_path, str)
            or not isinstance(compact_path, str)
            or not isinstance(input_name, str)
        ):
            raise Lfm2ProgressiveComponentAblationRunError(
                f"Component row {index} lacks required canonical fields."
            )
        try:
            parent_module = parent_model.get_submodule(parent_path)
            compact_module = compact_model.get_submodule(compact_path)
        except (AttributeError, KeyError) as exc:
            raise Lfm2ProgressiveComponentAblationRunError(
                f"Component row {component_id!r} names an absent module."
            ) from exc
        observed_parent_module = _module_state_sha256(parent_module)
        observed_compact_module = _module_state_sha256(compact_module)
        expected_parent_module = _digest(
            raw.get("parent_module_sha256"),
            label=f"{component_id} parent module",
        )
        expected_compact_module = _digest(
            raw.get("compact_module_sha256"),
            label=f"{component_id} compact module",
        )
        if (
            observed_parent_module != expected_parent_module
            or observed_compact_module != expected_compact_module
        ):
            raise Lfm2ProgressiveComponentAblationRunError(
                f"Component {component_id!r} module state differs from its plan."
            )
        input_tensor = bridge_tensors[input_name]
        output_tensor = None if output_name is None else bridge_tensors[output_name]
        domain_tensor = None if domain_name is None else bridge_tensors[domain_name]
        require_parameter_reduction = raw.get("require_parameter_reduction")
        if require_parameter_reduction is not True:
            raise Lfm2ProgressiveComponentAblationRunError(
                f"Component {component_id!r} must declare the literal JSON boolean "
                "true for require_parameter_reduction; coercion is forbidden."
            )
        source_fold_receipt = _digest(
            raw.get("source_fold_receipt_sha256"),
            label=f"{component_id} source fold receipt",
        )
        verified_fold = verified_fold_receipts.get(component_id)
        source_compact_checkpoint = _digest(
            raw.get("source_compact_checkpoint_sha256"),
            label=f"{component_id} compact checkpoint",
        )
        if (
            not isinstance(verified_fold, Mapping)
            or verified_fold.get("receipt_sha256") != source_fold_receipt
            or verified_fold.get("receipt_seal_freshly_verified") is not True
            or source_compact_checkpoint != planned_compact_checkpoint
            or source_authentication.get("compact_checkpoint_inventory_sha256")
            != planned_compact_checkpoint
        ):
            raise Lfm2ProgressiveComponentAblationRunError(
                f"Component {component_id!r} is not bound to freshly verified fold "
                "and compact-checkpoint provenance."
            )
        capsules.append(
            Lfm2ProgressiveComponentCapsule(
                component_id=component_id,
                kind=kind,
                parent_module_path=parent_path,
                compact_module_path=compact_path,
                input_encoder=input_tensor,
                output_decoder=output_tensor,
                output_domain_decoder=domain_tensor,
                expected_parent_module_sha256=expected_parent_module,
                expected_compact_module_sha256=expected_compact_module,
                expected_input_encoder_sha256=_digest(
                    raw.get("input_encoder_sha256"),
                    label=f"{component_id} input encoder",
                ),
                expected_output_decoder_sha256=_optional_digest(
                    raw.get("output_decoder_sha256"),
                    label=f"{component_id} output decoder",
                ),
                expected_output_domain_decoder_sha256=_optional_digest(
                    raw.get("output_domain_decoder_sha256"),
                    label=f"{component_id} output-domain decoder",
                ),
                source_fold_receipt_sha256=source_fold_receipt,
                source_compact_checkpoint_sha256=source_compact_checkpoint,
                require_parameter_reduction=require_parameter_reduction,
            )
        )
    return tuple(capsules)


def _dtype(value: str) -> Any:
    import torch

    normalized = value.strip().lower()
    if normalized not in {"float32", "bfloat16"}:
        raise Lfm2ProgressiveComponentAblationRunError(
            "dtype must be float32 or bfloat16."
        )
    return getattr(torch, normalized)


def _attach_cached_parent_text(
    responses: list[dict[str, Any]],
    *,
    parent_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    for response in responses:
        parent = parent_rows[str(response["id"])]
        response["parent_response"] = parent["response"]
        response["parent_response_token_ids"] = parent["response_token_ids"]
        response["parent_diagnostics"] = parent["diagnostics"]


def _candidate_progress(
    *,
    producer_receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
    completed: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    body = {
        "schema": PROGRESS_SCHEMA,
        "status": "complete" if len(completed) == len(candidate_ids) else "running",
        "producer_receipt_sha256": producer_receipt["receipt_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "candidate_ids": list(candidate_ids),
        "completed_candidate_ids": [row["candidate_id"] for row in completed],
        "remaining_candidate_ids": list(candidate_ids[len(completed) :]),
        "candidate_reviews": list(completed),
        "raw_decoded_responses_preserved": True,
        "automatic_acceptance_or_rejection_performed": False,
        "standalone_child_compression_claimed": False,
    }
    return {**body, "progress_sha256": canonical_sha256(body)}


def _candidate_window(
    *,
    candidate_count: int,
    start_index: int,
    stop_index: int | None,
) -> tuple[int, int]:
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or start_index < 0
        or isinstance(stop_index, bool)
        or (stop_index is not None and not isinstance(stop_index, int))
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Candidate window indices must be non-negative integers."
        )
    stop = candidate_count if stop_index is None else stop_index
    if not 0 <= start_index < stop <= candidate_count:
        raise Lfm2ProgressiveComponentAblationRunError(
            "Candidate window must satisfy 0 <= start < stop <= schedule count."
        )
    return start_index, stop


def _candidate_checkpoint_name(*, candidate_index: int, candidate_id: str) -> str:
    """Return a short, deterministic filesystem key for one sealed candidate.

    Cumulative candidate identities deliberately retain every installed
    component and can exceed a filesystem's single-component name limit.  The
    complete identity remains in the checkpoint manifest and review receipt;
    only the disposable checkpoint directory uses this bounded digest key.
    """

    if (
        isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
        or candidate_index < 0
        or not isinstance(candidate_id, str)
        or not candidate_id
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Candidate checkpoint identity is invalid."
        )
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:20]
    return f"candidate-{candidate_index:04d}-{digest}"


def run(
    *,
    run_directory: Path,
    parent_checkpoint: Path,
    compact_checkpoint: Path,
    producer_receipt: Path,
    expected_producer_receipt_sha256: str,
    expected_producer_receipt_file_sha256: str,
    component_plan: Path,
    expected_component_plan_sha256: str,
    expected_component_plan_file_sha256: str,
    bridge_tensors: Path,
    expected_bridge_tensors_file_sha256: str,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    device: str,
    dtype: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    checkpoint_retention: str,
    candidate_start_index: int = 0,
    candidate_stop_index: int | None = None,
) -> Mapping[str, Any]:
    import torch

    if checkpoint_retention not in {"keep-all", "keep-last", "receipts-only"}:
        raise Lfm2ProgressiveComponentAblationRunError(
            "checkpoint_retention must be keep-all, keep-last, or receipts-only."
        )
    if not 1 <= maximum_prompt_tokens <= 8192:
        raise Lfm2ProgressiveComponentAblationRunError(
            "maximum_prompt_tokens must be between 1 and 8192."
        )
    if not 16 <= maximum_new_tokens <= 2048:
        raise Lfm2ProgressiveComponentAblationRunError(
            "maximum_new_tokens must be between 16 and 2048."
        )
    run_root = _new_directory(run_directory, label="run directory")
    checkpoints_root = run_root / "non-final-hybrids"
    checkpoints_root.mkdir()
    parent_path = _ordinary_directory(parent_checkpoint, label="parent checkpoint")
    compact_path = _ordinary_directory(compact_checkpoint, label="compact checkpoint")
    prompt_path = _ordinary_file(prompts_json, label="prompt bank")
    envelope_path = _ordinary_file(parent_envelope, label="cached parent envelope")
    if _file_sha256(prompt_path) != _digest(
        expected_prompts_file_sha256,
        label="expected prompt-bank file",
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Prompt bank differs from its external pin."
        )
    publication = _load_producer_receipt(
        producer_receipt,
        expected_receipt_sha256=expected_producer_receipt_sha256,
        expected_file_sha256=expected_producer_receipt_file_sha256,
    )
    plan = _load_plan(
        component_plan,
        expected_plan_sha256=expected_component_plan_sha256,
        expected_file_sha256=expected_component_plan_file_sha256,
    )
    _authenticate_producer_publication(
        producer_receipt=publication,
        producer_receipt_path=producer_receipt,
        plan=plan,
        component_plan=component_plan,
        bridge_tensors=bridge_tensors,
        expected_component_plan_sha256=expected_component_plan_sha256,
        expected_component_plan_file_sha256=expected_component_plan_file_sha256,
        expected_bridge_tensors_file_sha256=(expected_bridge_tensors_file_sha256),
    )
    parent_checkpoint_inventory_sha256 = canonical_sha256(
        _checkpoint_inventory(parent_path)
    )
    compact_checkpoint_inventory_sha256 = canonical_sha256(
        _checkpoint_inventory(compact_path)
    )
    if (
        parent_checkpoint_inventory_sha256
        != plan.get("parent_checkpoint_inventory_sha256")
        or compact_checkpoint_inventory_sha256
        != plan.get("compact_checkpoint_inventory_sha256")
        or plan.get("source_authentication", {}).get(
            "parent_checkpoint_inventory_sha256"
        )
        != parent_checkpoint_inventory_sha256
        or plan.get("source_authentication", {}).get(
            "compact_checkpoint_inventory_sha256"
        )
        != compact_checkpoint_inventory_sha256
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Parent or compact checkpoint inventory differs from the automatic plan."
        )
    bridges = _load_bridge_tensors(
        bridge_tensors,
        expected_file_sha256=expected_bridge_tensors_file_sha256,
        plan=plan,
    )
    torch_dtype = _dtype(dtype)
    resolved_device = torch.device(device)

    _emit("load-and-authenticate-existing-local-parent-and-compact-child")
    parent_template = load_lfm2_direct_local_model(
        parent_path,
        device=resolved_device,
        dtype=torch_dtype,
    )
    compact_model = load_lfm2_direct_local_model(
        compact_path,
        device=resolved_device,
        dtype=torch_dtype,
    )
    parent_state = fingerprint_lfm2_direct_local_state(parent_template)
    compact_state = fingerprint_lfm2_direct_local_state(compact_model)
    if parent_state != _digest(
        plan.get("parent_model_state_sha256"), label="planned parent state"
    ) or compact_state != _digest(
        plan.get("compact_model_state_sha256"),
        label="planned compact state",
    ):
        raise Lfm2ProgressiveComponentAblationRunError(
            "Parent or compact checkpoint differs from the sealed component plan."
        )
    capsules = _build_capsules(
        plan=plan,
        bridge_tensors=bridges,
        parent_model=parent_template,
        compact_model=compact_model,
    )
    schedule = build_lfm2_progressive_component_ablation_schedule(capsules)
    schedule_document = schedule_receipt(schedule)
    _write_json(run_root / "schedule.json", schedule_document)
    start_index, stop_index = _candidate_window(
        candidate_count=len(schedule),
        start_index=candidate_start_index,
        stop_index=candidate_stop_index,
    )
    selected_schedule = schedule[start_index:stop_index]
    full_candidate_ids = [value.candidate_id for value in schedule]
    candidate_ids = [value.candidate_id for value in selected_schedule]
    selection_body = {
        "schema": "neural-forge-lfm2-progressive-component-window-v1",
        "schedule_sha256": schedule_document["schedule_sha256"],
        "full_candidate_count": len(schedule),
        "candidate_start_index": start_index,
        "candidate_stop_index_exclusive": stop_index,
        "selected_candidate_count": len(selected_schedule),
        "selected_original_indices": list(range(start_index, stop_index)),
        "selected_candidate_ids": candidate_ids,
        "completed_prior_candidates_rerun": False,
    }
    selection_document = {
        **selection_body,
        "selection_sha256": canonical_sha256(selection_body),
    }
    _write_json(run_root / "schedule-selection.json", selection_document)
    del parent_template
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt_rows = [dict(value) for value in load_review_prompts(prompt_path)]
    prompt_set_sha256 = canonical_sha256(prompt_rows)
    cached_parent = _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=_digest(
            expected_parent_envelope_sha256,
            label="expected cached parent envelope",
        ),
        expected_file_sha256=_digest(
            expected_parent_envelope_file_sha256,
            label="expected cached parent-envelope file",
        ),
        expected_prompt_set_sha256=prompt_set_sha256,
        expected_prompt_ids=[str(value["id"]) for value in prompt_rows],
    )
    parent_rows = {str(value["id"]): value for value in cached_parent["responses"]}
    tokenizer = load_lfm2_direct_local_tokenizer(parent_path)
    completed = []
    progress_path = run_root / "progress.json"
    retained_paths = []
    previous_retained: Path | None = None
    try:
        for candidate_index, candidate in enumerate(
            selected_schedule,
            start=start_index,
        ):
            _emit(
                "materialize-component-hybrid",
                candidate_id=candidate.candidate_id,
                candidate_index=candidate_index,
                candidate_count=len(schedule),
            )
            hybrid_model = load_lfm2_direct_local_model(
                parent_path,
                device=resolved_device,
                dtype=torch_dtype,
            )
            if fingerprint_lfm2_direct_local_state(hybrid_model) != parent_state:
                raise Lfm2ProgressiveComponentAblationRunError(
                    "Fresh parent shell changed between ablation candidates."
                )
            installations = apply_lfm2_component_ablation_candidate(
                parent_model=hybrid_model,
                compact_model=compact_model,
                candidate=candidate,
            )
            checkpoint_storage_key = _candidate_checkpoint_name(
                candidate_index=candidate_index,
                candidate_id=candidate.candidate_id,
            )
            checkpoint = checkpoints_root / checkpoint_storage_key
            manifest = write_lfm2_progressive_hybrid_checkpoint(
                model=hybrid_model,
                destination=checkpoint,
                tokenizer_source_directory=parent_path,
                candidate=candidate,
                installations=installations,
            )
            del hybrid_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _emit(
                "fresh-reload-and-decode-whole-behavior",
                candidate_id=candidate.candidate_id,
            )
            reloaded, authenticated_manifest = load_lfm2_progressive_hybrid_checkpoint(
                checkpoint,
                expected_manifest_sha256=manifest["manifest_sha256"],
                device=resolved_device,
                dtype=torch_dtype,
            )
            responses = _generate_rows(
                reloaded,
                tokenizer,
                prompt_rows,
                maximum_prompt_tokens=maximum_prompt_tokens,
                maximum_new_tokens=maximum_new_tokens,
            )
            _attach_cached_parent_text(responses, parent_rows=parent_rows)
            record = {
                "candidate_id": candidate.candidate_id,
                "checkpoint_storage_key": checkpoint_storage_key,
                "ablation_mode": candidate.mode,
                "component_ids": [value.component_id for value in candidate.capsules],
                "component_installations": [
                    dict(value.receipt) for value in installations
                ],
                "hybrid_manifest": dict(authenticated_manifest),
                "hybrid_manifest_sha256": authenticated_manifest["manifest_sha256"],
                "physical_parameter_elements": authenticated_manifest[
                    "unique_parameter_elements"
                ],
                "responses": responses,
                "response_count": len(responses),
                "fresh_reload_verified_before_decode": True,
                "diagnostic_hybrid_only": True,
                "standalone_child_compression_claimed": False,
            }
            completed.append(record)
            _write_json(
                progress_path,
                _candidate_progress(
                    producer_receipt=publication,
                    plan=plan,
                    schedule=schedule_document,
                    completed=completed,
                    candidate_ids=candidate_ids,
                ),
                replace=progress_path.exists(),
            )
            del reloaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            keep = checkpoint_retention == "keep-all" or (
                checkpoint_retention == "keep-last"
                and candidate_index == stop_index - 1
            )
            if keep:
                retained_paths.append(str(checkpoint))
                previous_retained = checkpoint
            else:
                shutil.rmtree(checkpoint)
        if checkpoint_retention == "keep-last" and previous_retained is None:
            raise Lfm2ProgressiveComponentAblationRunError(
                "keep-last retention did not preserve a checkpoint."
            )
    finally:
        del compact_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    body = {
        "schema": RUN_SCHEMA,
        "status": "NON-FINAL-COMPONENT-ABLATION-WHOLE-BEHAVIOR-DECODED",
        "producer_receipt": publication,
        "producer_receipt_sha256": publication["receipt_sha256"],
        "producer_receipt_file_sha256": _file_sha256(
            _ordinary_file(producer_receipt, label="automatic producer receipt")
        ),
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
        "schedule": dict(schedule_document),
        "schedule_sha256": schedule_document["schedule_sha256"],
        "schedule_selection": selection_document,
        "schedule_selection_sha256": selection_document["selection_sha256"],
        "parent_model_state_sha256": parent_state,
        "compact_model_state_sha256": compact_state,
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": len(prompt_rows),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "candidate_ids": candidate_ids,
        "full_candidate_ids": full_candidate_ids,
        "reviewed_candidate_count": len(completed),
        "candidate_reviews": completed,
        "checkpoint_retention": checkpoint_retention,
        "retained_checkpoint_paths": retained_paths,
        "evaluation_policy": {
            "decoded_whole_responses_are_primary_evidence": True,
            "cached_parent_text_comparison_only": True,
            "exact_token_match_required": False,
            "numeric_metrics_have_decision_authority": False,
            "manual_intelligence_review_required_for_interpretation": True,
            "automatic_acceptance_or_rejection_performed": False,
        },
        "interpretation": {
            "parent_shell_retained": True,
            "interface_bridges_retained_and_counted": True,
            "isolated_results_localize_component_folds": True,
            "whole_subsystem_groups_localize_distributed_fold_failures": True,
            "cumulative_subsystem_groups_localize_subsystem_composition": True,
            "cumulative_prefixes_localize_composition_failures": True,
            "completed_prior_candidates_rerun": False,
            "diagnostic_hybrid_only": True,
            "standalone_child_compression_claimed": False,
            "does_not_replace_main_exact4x_compiler": True,
        },
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_model_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "new_parent_trace_rows": 0,
            "hybrid_model_generate_passes": len(completed),
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "network_access": False,
        },
    }
    result = {**body, "run_sha256": canonical_sha256(body)}
    _write_json(run_root / "run.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--compact-checkpoint", type=Path, required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--expected-producer-receipt-sha256", required=True)
    parser.add_argument("--expected-producer-receipt-file-sha256", required=True)
    parser.add_argument("--component-plan", type=Path, required=True)
    parser.add_argument("--expected-component-plan-sha256", required=True)
    parser.add_argument("--expected-component-plan-file-sha256", required=True)
    parser.add_argument("--bridge-tensors", type=Path, required=True)
    parser.add_argument("--expected-bridge-tensors-file-sha256", required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--maximum-prompt-tokens", type=int, default=2048)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--candidate-start-index", type=int, default=0)
    parser.add_argument("--candidate-stop-index", type=int)
    parser.add_argument(
        "--checkpoint-retention",
        choices=("keep-all", "keep-last", "receipts-only"),
        default="keep-last",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(
            run_directory=args.run_directory,
            parent_checkpoint=args.parent_checkpoint,
            compact_checkpoint=args.compact_checkpoint,
            producer_receipt=args.producer_receipt,
            expected_producer_receipt_sha256=(args.expected_producer_receipt_sha256),
            expected_producer_receipt_file_sha256=(
                args.expected_producer_receipt_file_sha256
            ),
            component_plan=args.component_plan,
            expected_component_plan_sha256=(args.expected_component_plan_sha256),
            expected_component_plan_file_sha256=(
                args.expected_component_plan_file_sha256
            ),
            bridge_tensors=args.bridge_tensors,
            expected_bridge_tensors_file_sha256=(
                args.expected_bridge_tensors_file_sha256
            ),
            prompts_json=args.prompts_json,
            expected_prompts_file_sha256=args.expected_prompts_file_sha256,
            parent_envelope=args.parent_envelope,
            expected_parent_envelope_sha256=(args.expected_parent_envelope_sha256),
            expected_parent_envelope_file_sha256=(
                args.expected_parent_envelope_file_sha256
            ),
            device=args.device,
            dtype=args.dtype,
            maximum_prompt_tokens=args.maximum_prompt_tokens,
            maximum_new_tokens=args.maximum_new_tokens,
            checkpoint_retention=args.checkpoint_retention,
            candidate_start_index=args.candidate_start_index,
            candidate_stop_index=args.candidate_stop_index,
        )
    except (
        Lfm2ProgressiveComponentAblationError,
        Lfm2ProgressiveComponentAblationRunError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRIDGE_ARCHIVE_SCHEMA",
    "Lfm2ProgressiveComponentAblationRunError",
    "PLAN_SCHEMA",
    "PLAN_PRODUCER_SCHEMA",
    "PRODUCER_RECEIPT_FILENAME",
    "PROGRESS_SCHEMA",
    "RUN_SCHEMA",
    "main",
    "run",
]
