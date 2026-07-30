#!/usr/bin/env python3
"""Run one sealed, generic subset of authenticated fused-body components.

The wrapper does not produce new folds.  It selects any non-empty canonical
subset already present in an immutable fused-body publication, installs those
components into a fresh parent shell, differentially checkpoints the mixed
model, fresh-reloads it, and publishes decode evidence for manual review.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
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

from experiments import run_lfm2_fused_body_diagnostic as base  # noqa: E402
from experiments.run_lfm2_dynamic_catalog import (  # noqa: E402
    fingerprint_local_tokenizer,
)
from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _load_parent_envelope,
    load_review_prompts,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _model_state_sha256,
    _module_at,
    _stored_tensor_elements,
    _unique_parameter_bytes,
    _unique_parameter_elements,
)


SELECTION_SCHEMA = "neural-forge-lfm2-fused-body-component-set-selection-v1"
SELECTION_STATUS = "SEALED-GENERIC-FUSED-BODY-COMPONENT-SET"
RUN_SCHEMA = "neural-forge-lfm2-fused-body-selected-component-set-run-v1"
RUN_STATUS = "NON-FINAL-SELECTED-COMPONENT-SET-REVIEW-EVIDENCE-COMPLETE"
SELECTION_FILENAME = "selection-receipt.json"
RUN_FILENAME = "run.json"
REVIEW_FILENAME = "manual-review.json"
CHECKPOINT_DIRECTORY = "mixed-differential-checkpoint"
PRIOR_REVIEW_OUTCOME = "successful-manual-intelligence-review"


class Lfm2SelectedComponentSetError(base.Lfm2FusedBodyDiagnosticRunError):
    """A selected-set source, seal, runtime, or count invariant failed."""


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Lfm2SelectedComponentSetError(f"{label} is invalid.")
    return value


def _selected_component_ids(
    selected: Sequence[str],
    *,
    plan: Mapping[str, Any],
) -> tuple[str, ...]:
    if isinstance(selected, (str, bytes)):
        raise Lfm2SelectedComponentSetError(
            "Selected component IDs must be one non-empty sequence."
        )
    identifiers = tuple(selected)
    if (
        not identifiers
        or any(not isinstance(value, str) or not value for value in identifiers)
        or len(identifiers) != len(set(identifiers))
    ):
        raise Lfm2SelectedComponentSetError(
            "Selected component IDs must be non-empty and unique."
        )
    components = base._component_map(plan)
    unknown = [value for value in identifiers if value not in components]
    if unknown:
        raise Lfm2SelectedComponentSetError(
            f"Selected component set contains unknown IDs: {unknown!r}."
        )
    schedule = plan.get("schedule")
    base.producer._authenticate_schedule(schedule)
    component_order = tuple(schedule["component_order"])
    canonical_subset = tuple(
        identifier for identifier in component_order if identifier in identifiers
    )
    if identifiers != canonical_subset:
        raise Lfm2SelectedComponentSetError(
            "Selected component IDs must follow the authenticated schedule order."
        )
    return identifiers


def _prior_review_pins(
    raw_pins: Sequence[Mapping[str, Any]],
    *,
    selected_component_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(raw_pins, (str, bytes)):
        raise Lfm2SelectedComponentSetError(
            "Prior successful review pins must be one sequence."
        )
    by_component: dict[str, Mapping[str, Any]] = {}
    required = {
        "component_id",
        "candidate_id",
        "manual_review_sha256",
        "manual_review_file_sha256",
        "mixed_checkpoint_manifest_sha256",
        "outcome",
    }
    for raw in raw_pins:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise Lfm2SelectedComponentSetError(
                "Each prior review pin must contain only the required pin fields."
            )
        component_id = raw.get("component_id")
        candidate_id = raw.get("candidate_id")
        if (
            not isinstance(component_id, str)
            or component_id not in selected_component_ids
            or component_id in by_component
            or not isinstance(candidate_id, str)
            or not candidate_id
            or raw.get("outcome") != PRIOR_REVIEW_OUTCOME
        ):
            raise Lfm2SelectedComponentSetError(
                "Prior review pins must be unique successful reviews of selected IDs."
            )
        normalized = dict(raw)
        for field in (
            "manual_review_sha256",
            "manual_review_file_sha256",
            "mixed_checkpoint_manifest_sha256",
        ):
            normalized[field] = base._digest(
                raw.get(field),
                label=f"prior review {component_id} {field}",
            )
        by_component[component_id] = MappingProxyType(normalized)
    return tuple(
        by_component[identifier]
        for identifier in selected_component_ids
        if identifier in by_component
    )


def _selection_receipt(
    *,
    publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    producer_receipt_file_sha256: str,
    plan_file_sha256: str,
    archive_file_sha256: str,
    selected_component_ids: Sequence[str],
    prior_successful_review_pins: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    """Build a deterministic receipt for any authenticated component subset."""

    identifiers = _selected_component_ids(selected_component_ids, plan=plan)
    prior_pins = _prior_review_pins(
        prior_successful_review_pins,
        selected_component_ids=identifiers,
    )
    receipt_file_sha = base._digest(
        producer_receipt_file_sha256,
        label="fused-body producer receipt file",
    )
    fused_plan_file_sha = base._digest(
        plan_file_sha256,
        label="fused-body plan file",
    )
    fused_archive_file_sha = base._digest(
        archive_file_sha256,
        label="fused-body archive file",
    )
    if (
        publication.get("plan_sha256") != plan.get("plan_sha256")
        or publication.get("plan_file_sha256") != fused_plan_file_sha
        or publication.get("archive_file_sha256") != fused_archive_file_sha
    ):
        raise Lfm2SelectedComponentSetError(
            "Selection ancestry differs from the authenticated publication."
        )
    components = base._component_map(plan)
    archive_records = plan.get("archive", {}).get("tensor_records")
    if not isinstance(archive_records, Mapping):
        raise Lfm2SelectedComponentSetError(
            "Fused-body plan lacks authenticated archive records."
        )
    selected_rows = []
    original_sum = 0
    fused_sum = 0
    reduction_sum = 0
    tensor_count = 0
    state_pins = {}
    for identifier in identifiers:
        row = components[identifier]
        physical = row.get("physical_parameter_accounting")
        tensor_names = row.get("tensor_names")
        if not isinstance(physical, Mapping) or not isinstance(tensor_names, Mapping):
            raise Lfm2SelectedComponentSetError(
                f"Component {identifier!r} lacks physical or tensor accounting."
            )
        original = _integer(
            physical.get("original_parent_component_parameter_elements"),
            label=f"{identifier} original parameter count",
            minimum=1,
        )
        fused = _integer(
            physical.get("fused_candidate_parameter_elements"),
            label=f"{identifier} fused parameter count",
            minimum=1,
        )
        reduction = _integer(
            physical.get("physical_parameter_reduction_from_parent"),
            label=f"{identifier} parameter reduction",
            minimum=1,
        )
        if original - fused != reduction:
            raise Lfm2SelectedComponentSetError(
                f"Component {identifier!r} has inconsistent physical counts."
            )
        tensor_pins = {}
        for state_name, archive_name in sorted(tensor_names.items()):
            record = archive_records.get(archive_name)
            if (
                not isinstance(state_name, str)
                or not isinstance(archive_name, str)
                or not isinstance(record, Mapping)
            ):
                raise Lfm2SelectedComponentSetError(
                    f"Component {identifier!r} has an invalid archive tensor pin."
                )
            tensor_pins[state_name] = {
                "archive_name": archive_name,
                "shape": list(record.get("shape", ())),
                "dtype": record.get("dtype"),
                "sha256": base._digest(
                    record.get("sha256"),
                    label=f"{identifier} {state_name} tensor",
                ),
            }
        state_pin = base._digest(
            row.get("candidate_state_sha256"),
            label=f"{identifier} candidate state",
        )
        state_pins[identifier] = state_pin
        selected_rows.append(
            {
                "component_id": identifier,
                "kind": row["kind"],
                "layer_index": row["layer_index"],
                "parent_module_path": row["parent_module_path"],
                "parent_module_sha256": base._digest(
                    row.get("parent_module_sha256"),
                    label=f"{identifier} parent module",
                ),
                "candidate_result_sha256": base._digest(
                    row.get("candidate_result_sha256"),
                    label=f"{identifier} candidate result",
                ),
                "candidate_state_sha256": state_pin,
                "reconstruction_descriptor_sha256": base._digest(
                    row.get("reconstruction_descriptor_sha256"),
                    label=f"{identifier} reconstruction descriptor",
                ),
                "physical_parameter_accounting": base._plain(physical),
                "tensor_pins": tensor_pins,
            }
        )
        original_sum += original
        fused_sum += fused
        reduction_sum += reduction
        tensor_count += len(tensor_pins)
    parent_parameters = _integer(
        plan.get("parent_cache_binding", {}).get("parent_unique_parameters"),
        label="parent unique parameter count",
        minimum=1,
    )
    candidate_key = canonical_sha256({"component_ids": list(identifiers)})
    candidate = {
        "candidate_id": f"selected-set--{candidate_key[:24]}",
        "mode": "selected-component-set",
        "source_stage": "final-s2",
        "component_ids": list(identifiers),
        "component_state_sha256": state_pins,
    }
    body = {
        "schema": SELECTION_SCHEMA,
        "status": SELECTION_STATUS,
        "producer_receipt_sha256": publication["receipt_sha256"],
        "producer_receipt_file_sha256": receipt_file_sha,
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": fused_plan_file_sha,
        "schedule_sha256": plan["schedule"]["schedule_sha256"],
        "archive_file_sha256": fused_archive_file_sha,
        "component_order_policy": plan["schedule"]["component_order_policy"],
        "selected_component_ids": list(identifiers),
        "selected_component_count": len(identifiers),
        "candidate": candidate,
        "selected_components": selected_rows,
        "prior_successful_review_pins": base._plain(prior_pins),
        "prior_successful_review_pin_count": len(prior_pins),
        "physical_parameter_accounting": {
            "parent_unique_parameter_elements": parent_parameters,
            "selected_original_component_parameter_elements": original_sum,
            "selected_fused_component_parameter_elements": fused_sum,
            "expected_parameter_reduction": reduction_sum,
            "expected_parameter_byte_reduction_fp32": 4 * reduction_sum,
            "expected_mixed_unique_parameter_elements": (
                parent_parameters - reduction_sum
            ),
            "expected_mixed_unique_parameter_bytes_fp32": (
                4 * (parent_parameters - reduction_sum)
            ),
            "selected_differential_tensor_count": tensor_count,
            "bridge_sidecar_parameter_elements": 0,
            "shared_parameter_elements": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
        "operation_audit": {
            "selection_only": True,
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "model_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "new_parent_trace_rows": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "bridge_sidecars_materialized": 0,
            "parameter_sharing_introduced": False,
        },
    }
    return MappingProxyType(
        {**body, "selection_sha256": canonical_sha256(body)}
    )


def _authenticate_selection_receipt(
    receipt: Mapping[str, Any],
    *,
    publication: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping):
        raise Lfm2SelectedComponentSetError("Selection receipt must be one mapping.")
    body = dict(receipt)
    supplied = body.pop("selection_sha256", None)
    if (
        receipt.get("schema") != SELECTION_SCHEMA
        or receipt.get("status") != SELECTION_STATUS
        or supplied != canonical_sha256(body)
    ):
        raise Lfm2SelectedComponentSetError("Selection receipt seal is invalid.")
    expected = _selection_receipt(
        publication=publication,
        plan=plan,
        producer_receipt_file_sha256=receipt.get(
            "producer_receipt_file_sha256"
        ),
        plan_file_sha256=receipt.get("plan_file_sha256"),
        archive_file_sha256=receipt.get("archive_file_sha256"),
        selected_component_ids=receipt.get("selected_component_ids", ()),
        prior_successful_review_pins=receipt.get(
            "prior_successful_review_pins", ()
        ),
    )
    if dict(receipt) != dict(expected):
        raise Lfm2SelectedComponentSetError(
            "Selection receipt differs from its authenticated plan-derived form."
        )
    return receipt


def _runtime_integrity_audit(
    model: Any,
    *,
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Prove exact counts and independent frozen selected-component tensors."""

    identifiers = tuple(selection["selected_component_ids"])
    components = base._component_map(plan)
    named = dict(model.named_parameters(remove_duplicate=False))
    expected_selected_names = set()
    component_tensor_counts = {}
    for identifier in identifiers:
        row = components[identifier]
        path = row["parent_module_path"]
        module = _module_at(model, path)
        expected_state_names = set(row["tensor_names"])
        if not base._no_sidecars(module, expected_state_names=expected_state_names):
            raise Lfm2SelectedComponentSetError(
                f"Fresh component {identifier!r} failed its no-sidecar state audit."
            )
        component_names = {f"{path}.{name}" for name in expected_state_names}
        if not component_names <= set(named):
            raise Lfm2SelectedComponentSetError(
                f"Fresh component {identifier!r} lacks expected parameters."
            )
        expected_selected_names.update(component_names)
        component_tensor_counts[identifier] = len(component_names)

    def storage_identity(tensor: Any) -> tuple[str, int, int]:
        storage = tensor.untyped_storage()
        return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))

    selected_parameters = [named[name] for name in sorted(expected_selected_names)]
    selected_storages = [storage_identity(value) for value in selected_parameters]
    outside_storages = {
        storage_identity(value)
        for name, value in named.items()
        if name not in expected_selected_names
    }
    physical = selection["physical_parameter_accounting"]
    actual = {
        "model_state_sha256": _model_state_sha256(model),
        "unique_parameter_elements": _unique_parameter_elements(model),
        "unique_parameter_bytes": _unique_parameter_bytes(model),
        "stored_tensor_elements": _stored_tensor_elements(model),
    }
    if (
        model.training
        or any(module.training for module in model.modules())
        or any(parameter.requires_grad for parameter in model.parameters())
        or len(selected_storages) != len(set(selected_storages))
        or bool(set(selected_storages) & outside_storages)
        or actual["model_state_sha256"] != manifest["model_state_sha256"]
        or actual["unique_parameter_elements"]
        != manifest["unique_parameter_elements"]
        or actual["unique_parameter_bytes"] != manifest["unique_parameter_bytes"]
        or actual["stored_tensor_elements"] != manifest["stored_tensor_elements"]
        or actual["unique_parameter_elements"]
        != physical["expected_mixed_unique_parameter_elements"]
        or actual["unique_parameter_bytes"]
        != physical["expected_mixed_unique_parameter_bytes_fp32"]
        or manifest["differential_tensor_count"]
        != physical["selected_differential_tensor_count"]
        or manifest["installation_receipt"]["realized_parameter_reduction"]
        != physical["expected_parameter_reduction"]
        or manifest["installation_receipt"]["realized_parameter_byte_reduction"]
        != physical["expected_parameter_byte_reduction_fp32"]
    ):
        raise Lfm2SelectedComponentSetError(
            "Fresh mixed model failed frozen/eval/independence/count verification."
        )
    body = {
        "status": "fresh-reload-frozen-eval-independent-counts-exact",
        **actual,
        "selected_component_ids": list(identifiers),
        "selected_component_tensor_counts": component_tensor_counts,
        "selected_parameter_tensor_count": len(selected_parameters),
        "selected_parameter_storage_count": len(set(selected_storages)),
        "selected_internal_shared_storage_count": 0,
        "selected_to_outside_shared_storage_count": 0,
        "all_parameters_frozen": True,
        "all_modules_eval": True,
        "parameter_alias_topology_unchanged": manifest[
            "installation_receipt"
        ]["parameter_alias_topology_unchanged"],
        "bridge_sidecar_parameter_elements": 0,
        "new_parameter_sharing_introduced": False,
        "fresh_differential_reload_verified": True,
    }
    return MappingProxyType(
        {**body, "runtime_integrity_sha256": canonical_sha256(body)}
    )


def run(
    *,
    run_directory: Path,
    parent_checkpoint: Path,
    expected_parent_checkpoint_inventory_sha256: str,
    producer_receipt: Path,
    expected_producer_receipt_sha256: str,
    expected_producer_receipt_file_sha256: str,
    fused_plan: Path,
    expected_fused_plan_sha256: str,
    expected_fused_plan_file_sha256: str,
    fused_archive: Path,
    expected_fused_archive_file_sha256: str,
    selected_component_ids: Sequence[str],
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    device: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    checkpoint_retention: str = "keep-last",
    prior_successful_review_pins: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    """Run one authenticated generic selected-component-set experiment."""

    import torch

    if checkpoint_retention not in {"receipts-only", "keep-last"}:
        raise Lfm2SelectedComponentSetError(
            "checkpoint_retention must be receipts-only or keep-last."
        )
    if not 256 <= maximum_prompt_tokens <= 8192:
        raise Lfm2SelectedComponentSetError(
            "maximum_prompt_tokens must be between 256 and 8192."
        )
    if not 256 <= maximum_new_tokens <= 2048:
        raise Lfm2SelectedComponentSetError(
            "maximum_new_tokens must be between 256 and 2048."
        )
    run_root = base._new_directory(run_directory, label="run directory")
    parent_path = base._ordinary_directory(
        parent_checkpoint,
        label="parent checkpoint",
    )
    receipt_path = base._ordinary_file(
        producer_receipt,
        label="fused-body producer receipt",
    )
    plan_path = base._ordinary_file(fused_plan, label="fused-body plan")
    archive_path = base._ordinary_file(fused_archive, label="fused-body archive")
    prompt_path = base._ordinary_file(prompts_json, label="prompt bank")
    envelope_path = base._ordinary_file(
        parent_envelope,
        label="cached parent envelope",
    )
    base._forbid_run_inside_authenticated_inputs(
        run_root,
        input_directories=(
            parent_path,
            receipt_path.parent,
            plan_path.parent,
            archive_path.parent,
            prompt_path.parent,
            envelope_path.parent,
        ),
    )
    if base._file_sha256(prompt_path) != base._digest(
        expected_prompts_file_sha256,
        label="expected prompt bank",
    ):
        raise Lfm2SelectedComponentSetError(
            "Prompt bank differs from its external pin."
        )
    publication = base._load_publication(
        receipt_path,
        expected_receipt_sha256=expected_producer_receipt_sha256,
        expected_file_sha256=expected_producer_receipt_file_sha256,
    )
    plan = base._load_plan(
        plan_path,
        expected_plan_sha256=expected_fused_plan_sha256,
        expected_file_sha256=expected_fused_plan_file_sha256,
    )
    base._authenticate_publication(
        receipt=publication,
        receipt_path=receipt_path,
        plan=plan,
        plan_path=plan_path,
        archive_path=archive_path,
        expected_plan_sha256=expected_fused_plan_sha256,
        expected_plan_file_sha256=expected_fused_plan_file_sha256,
        expected_archive_file_sha256=expected_fused_archive_file_sha256,
    )
    parent_inventory_sha = canonical_sha256(base._checkpoint_inventory(parent_path))
    if (
        parent_inventory_sha
        != base._digest(
            expected_parent_checkpoint_inventory_sha256,
            label="expected parent checkpoint inventory",
        )
        or parent_inventory_sha != plan["parent_checkpoint_inventory_sha256"]
    ):
        raise Lfm2SelectedComponentSetError(
            "Parent checkpoint inventory differs from fused-body ancestry."
        )
    archive_tensors = base._load_tensor_archive(
        archive_path,
        expected_file_sha256=expected_fused_archive_file_sha256,
        records=plan["archive"]["tensor_records"],
    )
    selection = _selection_receipt(
        publication=publication,
        plan=plan,
        producer_receipt_file_sha256=expected_producer_receipt_file_sha256,
        plan_file_sha256=expected_fused_plan_file_sha256,
        archive_file_sha256=expected_fused_archive_file_sha256,
        selected_component_ids=selected_component_ids,
        prior_successful_review_pins=prior_successful_review_pins,
    )
    _authenticate_selection_receipt(
        selection,
        publication=publication,
        plan=plan,
    )

    binding = plan["parent_cache_binding"]
    parent_cache_source = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    if (
        fingerprint_lfm2_direct_local_state(parent_cache_source)
        != binding["parent_state_sha256"]
        or fingerprint_lfm2_parent(parent_cache_source).identity_sha256
        != binding["parent_identity_sha256"]
        or _unique_parameter_elements(parent_cache_source)
        != binding["parent_unique_parameters"]
        or fingerprint_local_tokenizer(parent_path) != binding["tokenizer_sha256"]
    ):
        raise Lfm2SelectedComponentSetError(
            "Fresh BF16 parent/tokenizer differs from the cache binding."
        )
    del parent_cache_source

    prompt_rows = [dict(value) for value in load_review_prompts(prompt_path)]
    prompt_set_sha = canonical_sha256(prompt_rows)
    parent_cache = _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=base._digest(
            expected_parent_envelope_sha256,
            label="expected parent envelope",
        ),
        expected_file_sha256=base._digest(
            expected_parent_envelope_file_sha256,
            label="expected parent envelope file",
        ),
        expected_prompt_set_sha256=prompt_set_sha,
        expected_prompt_ids=[str(row["id"]) for row in prompt_rows],
    )
    parent_cache_authentication = base._authenticate_parent_cache_binding(
        parent_cache,
        plan=plan,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    parent_rows = {str(row["id"]): row for row in parent_cache["responses"]}
    tokenizer = load_lfm2_direct_local_tokenizer(parent_path)
    resolved_device = torch.device(device)

    run_root.mkdir()
    base._write_json(run_root / SELECTION_FILENAME, selection)
    candidate = selection["candidate"]
    model = load_lfm2_direct_local_model(
        parent_path,
        device=resolved_device,
        dtype=torch.float32,
    )
    if (
        fingerprint_lfm2_direct_local_state(model)
        != plan["parent_model_state_sha256"]
        or _unique_parameter_elements(model)
        != selection["physical_parameter_accounting"][
            "parent_unique_parameter_elements"
        ]
    ):
        raise Lfm2SelectedComponentSetError(
            "Fresh FP32 parent shell differs before selected-set installation."
        )
    installation = base._install_schedule_candidate(
        model=model,
        plan=plan,
        schedule_candidate=candidate,
        tensors=archive_tensors,
        device=resolved_device,
    )
    physical = selection["physical_parameter_accounting"]
    if (
        installation["installation_count"] != selection["selected_component_count"]
        or installation["realized_parameter_reduction"]
        != physical["expected_parameter_reduction"]
        or installation["realized_parameter_byte_reduction"]
        != physical["expected_parameter_byte_reduction_fp32"]
        or installation["after"]["unique_parameter_elements"]
        != physical["expected_mixed_unique_parameter_elements"]
        or installation["after"]["unique_parameter_bytes"]
        != physical["expected_mixed_unique_parameter_bytes_fp32"]
    ):
        raise Lfm2SelectedComponentSetError(
            "Selected-set installation differs from exact receipt counts."
        )
    checkpoint = run_root / CHECKPOINT_DIRECTORY
    manifest = base._write_mixed_checkpoint(
        destination=checkpoint,
        model=model,
        plan=plan,
        producer_receipt=publication,
        schedule_candidate=candidate,
        installation_receipt=installation,
        archive_tensors=archive_tensors,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded, authenticated_manifest = base._fresh_load_mixed_checkpoint(
        source=checkpoint,
        expected_manifest_sha256=manifest["manifest_sha256"],
        parent_checkpoint=parent_path,
        plan=plan,
        producer_receipt=publication,
        device=resolved_device,
    )
    runtime_integrity = _runtime_integrity_audit(
        reloaded,
        plan=plan,
        selection=selection,
        manifest=authenticated_manifest,
    )
    responses = base._generate_rows(
        reloaded,
        tokenizer,
        prompt_rows,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    base._attach_parent(responses, parent_rows=parent_rows)
    review = base._manual_review_document(
        candidate=candidate,
        manifest=authenticated_manifest,
        responses=responses,
    )
    review_path = run_root / REVIEW_FILENAME
    base._write_json(review_path, review)
    del reloaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if checkpoint_retention == "receipts-only":
        shutil.rmtree(checkpoint)

    body = {
        "schema": RUN_SCHEMA,
        "status": RUN_STATUS,
        "selection_receipt": base._plain(selection),
        "selection_sha256": selection["selection_sha256"],
        "selection_receipt_file": SELECTION_FILENAME,
        "selection_receipt_file_sha256": base._file_sha256(
            run_root / SELECTION_FILENAME
        ),
        "producer_receipt_sha256": publication["receipt_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": plan["schedule"]["schedule_sha256"],
        "archive_file_sha256": expected_fused_archive_file_sha256,
        "parent_checkpoint_inventory_sha256": parent_inventory_sha,
        "parent_cache_authentication": base._plain(parent_cache_authentication),
        "prompt_set_sha256": prompt_set_sha,
        "prompt_count": len(prompt_rows),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "candidate": base._plain(candidate),
        "installation_receipt": base._plain(installation),
        "mixed_checkpoint_manifest_sha256": authenticated_manifest[
            "manifest_sha256"
        ],
        "runtime_integrity_audit": base._plain(runtime_integrity),
        "manual_review_file": REVIEW_FILENAME,
        "manual_review_file_sha256": base._file_sha256(review_path),
        "manual_review_sha256": review["review_sha256"],
        "responses": responses,
        "response_count": len(responses),
        "checkpoint_retention": checkpoint_retention,
        "retained_checkpoint_paths": (
            [CHECKPOINT_DIRECTORY] if checkpoint_retention == "keep-last" else []
        ),
        "evaluation_policy": {
            "complete_prompt_bank_required": True,
            "fresh_executed_decode_is_primary_candidate_evidence": True,
            "cached_parent_text_is_reference_evidence_only": True,
            "exact_token_match_required": False,
            "numeric_metrics_have_decision_authority": False,
            "manual_intelligence_review_required": True,
            "automatic_acceptance_or_rejection_performed": False,
        },
        "interpretation": {
            "generic_plan_derived_component_subset": True,
            "hardcoded_component_ids": False,
            "fresh_parent_install_checkpoint_reload_decode_verified": True,
            "diagnostic_parent_shell_only": True,
            "standalone_child_compression_claimed": False,
            "bridge_sidecars_present": False,
            "new_parameter_sharing_introduced": False,
            "does_not_modify_fused_body_producer_or_runner": True,
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
            "hybrid_model_generate_passes": 1,
            "cached_parent_references_reused": len(prompt_rows),
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "bridge_sidecar_parameter_elements": 0,
            "shared_parameter_elements": 0,
            "network_access": False,
        },
    }
    result = MappingProxyType({**body, "run_sha256": canonical_sha256(body)})
    base._write_json(run_root / RUN_FILENAME, result)
    return result


def _load_prior_pin_files(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    pins = []
    for index, path in enumerate(paths):
        pins.append(base._load_json(path, label=f"prior review pin {index}"))
    return pins


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--expected-producer-receipt-sha256", required=True)
    parser.add_argument("--expected-producer-receipt-file-sha256", required=True)
    parser.add_argument("--fused-plan", type=Path, required=True)
    parser.add_argument("--expected-fused-plan-sha256", required=True)
    parser.add_argument("--expected-fused-plan-file-sha256", required=True)
    parser.add_argument("--fused-archive", type=Path, required=True)
    parser.add_argument("--expected-fused-archive-file-sha256", required=True)
    parser.add_argument(
        "--component-id",
        dest="selected_component_ids",
        action="append",
        required=True,
    )
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument(
        "--checkpoint-retention",
        choices=("receipts-only", "keep-last"),
        default="keep-last",
    )
    parser.add_argument(
        "--prior-successful-review-pin-file",
        type=Path,
        action="append",
        default=[],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = vars(_parser().parse_args(argv))
    pin_files = arguments.pop("prior_successful_review_pin_file")
    arguments["prior_successful_review_pins"] = _load_prior_pin_files(pin_files)
    result = run(**arguments)
    print(
        json.dumps(base._plain(result), allow_nan=False, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
