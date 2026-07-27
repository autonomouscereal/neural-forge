#!/usr/bin/env python3
"""Publish authenticated protected-channel LFM2 short-conv diagnostics.

The producer consumes the sealed A0019/B038 lineage, the direct-local parent
and exact4x checkpoints, and B038's authenticated residual encoder/decoder.
It creates three layer-0 parent-shell round-trip candidates, a counted
standalone-child-stem adapter control, a layer-1 parent-both control, and the
existing dense compact-channel layer-0 fold as a negative control.  It
performs no model forward, generation, trace capture, training, pruning,
quantization, sparsification, or network loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
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

from experiments import run_lfm2_progressive_component_ablation as b038  # noqa: E402
from experiments.run_lfm2_dynamic_catalog import (  # noqa: E402
    fingerprint_local_tokenizer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_fused_body_diagnostic import (  # noqa: E402
    Lfm2FusedShortConvDiagnosticInput,
    authenticate_lfm2_fused_body_candidate,
    compile_lfm2_fused_short_conv_diagnostic,
    fingerprint_lfm2_fused_body_diagnostic_input,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _module_state_sha256,
    _unique_parameter_elements,
)
from neural_forge.hf_lfm2_protected_short_conv_diagnostic import (  # noqa: E402
    Lfm2ProtectedShortConvInput,
    authenticate_lfm2_protected_short_conv_candidate,
    compile_lfm2_protected_short_conv_diagnostic,
    fingerprint_lfm2_protected_short_conv_input,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


PLAN_SCHEMA = "neural-forge-lfm2-protected-short-conv-diagnostic-plan-v1"
PRODUCER_SCHEMA = "neural-forge-lfm2-protected-short-conv-plan-producer-v1"
ARCHIVE_SCHEMA = "neural-forge-lfm2-protected-short-conv-candidate-archive-v1"
SCHEDULE_SCHEMA = "neural-forge-lfm2-protected-short-conv-schedule-v1"
ARCHIVE_FILENAME = "protected-short-conv-candidates.safetensors"
PLAN_FILENAME = "protected-short-conv-plan.json"
PRODUCER_RECEIPT_FILENAME = "protected-short-conv-producer-receipt.json"
PLAN_STATUS = "NON-FINAL-PROTECTED-SHORT-CONV-DIAGNOSTIC-PLAN"
PRODUCER_STATUS = "protected-short-conv-candidates-and-plan-publication-complete"
EXPECTED_SCHEDULE_CANDIDATE_IDS = (
    "cached-parent-reference",
    "layer-000-parent-input-roundtrip",
    "layer-000-parent-output-roundtrip",
    "layer-000-parent-both-roundtrip",
    "layer-000-dense-compact-negative-control",
    "layer-000-standalone-child-stem-adapter-control",
    "layer-001-parent-both-roundtrip",
)


class Lfm2ProtectedShortConvPlanProducerError(RuntimeError):
    """An authenticated source, candidate, or publication is invalid."""


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ProtectedShortConvPlanProducerError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_plain(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _new_output(value: Path) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ProtectedShortConvPlanProducerError("Output directory must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2ProtectedShortConvPlanProducerError(
            "Output directory must not already exist."
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2ProtectedShortConvPlanProducerError(
            "Output parent must be one ordinary directory."
        )
    return path


def _forbid_output_inside_authenticated_inputs(
    output: Path,
    *,
    input_directories: Sequence[Path],
) -> None:
    target = output.resolve(strict=False)
    for raw in input_directories:
        source = raw.resolve(strict=True)
        if target == source or source in target.parents:
            raise Lfm2ProtectedShortConvPlanProducerError(
                "Output directory must be lexically separate from every "
                "authenticated input directory."
            )


def _tensor_record(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(size) for size in value.shape],
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": lfm2_authenticated_tensor_sha256(value),
    }


def _component_rows(plan: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    rows = plan.get("components")
    if not isinstance(rows, list):
        raise Lfm2ProtectedShortConvPlanProducerError(
            "B038 component plan has no component rows."
        )
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("component_id"), str):
            raise Lfm2ProtectedShortConvPlanProducerError(
                "B038 component row is malformed."
            )
        identifier = str(row["component_id"])
        if identifier in result:
            raise Lfm2ProtectedShortConvPlanProducerError(
                "B038 component identifiers are duplicated."
            )
        result[identifier] = row
    return result


def _parent_cache_binding(
    *,
    parent_path: Path,
    parent_inventory_sha256: str,
    expected_identity_sha256: str,
) -> Mapping[str, Any]:
    import torch

    model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    identity = fingerprint_lfm2_parent(model).identity_sha256
    if identity != expected_identity_sha256:
        raise Lfm2ProtectedShortConvPlanProducerError(
            "Fresh BF16 parent identity differs from B038 provenance."
        )
    body = {
        "schema": "neural-forge-lfm2-protected-short-conv-parent-cache-binding-v1",
        "status": "exact-local-parent-cache-source-authenticated",
        "parent_checkpoint_inventory_sha256": parent_inventory_sha256,
        "parent_runtime_materialization_dtype": "bfloat16",
        "parent_state_sha256": fingerprint_lfm2_direct_local_state(model),
        "parent_identity_sha256": identity,
        "parent_unique_parameters": _unique_parameter_elements(model),
        "tokenizer_sha256": fingerprint_local_tokenizer(parent_path),
        "direct_local_checkpoint_source_authenticated": True,
        "hub_loader_permitted": False,
        "network_access_permitted": False,
    }
    del model
    return {**body, "binding_sha256": canonical_sha256(body)}


def _protected_source(
    *,
    module: Any,
    layer_index: int,
    parent_hidden: int,
    child_hidden: int,
    encoder: Any,
    decoder: Any,
) -> Lfm2ProtectedShortConvInput:
    return Lfm2ProtectedShortConvInput(
        layer_index=layer_index,
        parent_hidden_size=parent_hidden,
        child_hidden_size=child_hidden,
        kernel_size=int(module.conv.weight.shape[-1]),
        parent_conv_weight=module.conv.weight.detach().cpu().contiguous(),
        parent_in_proj_weight=module.in_proj.weight.detach().cpu().contiguous(),
        parent_out_proj_weight=module.out_proj.weight.detach().cpu().contiguous(),
        input_encoder=encoder,
        output_decoder=decoder,
        parent_conv_bias=module.conv.bias,
        parent_in_proj_bias=module.in_proj.bias,
        parent_out_proj_bias=module.out_proj.bias,
    )


def _dense_source(
    *,
    module: Any,
    layer_index: int,
    parent_hidden: int,
    child_hidden: int,
    encoder: Any,
    decoder: Any,
) -> Lfm2FusedShortConvDiagnosticInput:
    return Lfm2FusedShortConvDiagnosticInput(
        layer_index=layer_index,
        parent_hidden_size=parent_hidden,
        compact_hidden_size=child_hidden,
        kernel_size=int(module.conv.weight.shape[-1]),
        compact_conv_weight=module.conv.weight.detach().cpu().contiguous(),
        compact_in_proj_weight=module.in_proj.weight.detach().cpu().contiguous(),
        compact_out_proj_weight=module.out_proj.weight.detach().cpu().contiguous(),
        input_encoder=encoder,
        output_decoder=decoder,
        compact_conv_bias=module.conv.bias,
        compact_in_proj_bias=module.in_proj.bias,
        compact_out_proj_bias=module.out_proj.bias,
    )


def _archive_candidate(
    *,
    candidate_id: str,
    candidate_type: str,
    layer_index: int,
    module_path: str,
    candidate: Any,
    archive: dict[str, Any],
) -> Mapping[str, Any]:
    if candidate_type == "protected-parent-shell":
        receipt = authenticate_lfm2_protected_short_conv_candidate(candidate)
    elif candidate_type == "dense-compact-negative-control":
        receipt = authenticate_lfm2_fused_body_candidate(candidate)
    else:  # pragma: no cover - internal construction invariant
        raise AssertionError(candidate_type)
    state = candidate.state_dict()
    tensor_names = {}
    for state_name, tensor in state.items():
        archive_name = f"candidate.{candidate_id}.{state_name}"
        if archive_name in archive:
            raise Lfm2ProtectedShortConvPlanProducerError(
                "Candidate archive tensor names are duplicated."
            )
        archive[archive_name] = tensor.detach().cpu().contiguous()
        tensor_names[state_name] = archive_name
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "layer_index": layer_index,
        "parent_module_path": module_path,
        "variant": receipt.get("variant"),
        "candidate_state_sha256": receipt["candidate_state_sha256"],
        "candidate_result_sha256": receipt["result_sha256"],
        "reconstruction_descriptor_sha256": candidate.reconstruction_descriptor[
            "descriptor_sha256"
        ],
        "receipt": _plain(candidate.receipt),
        "reconstruction_descriptor": _plain(candidate.reconstruction_descriptor),
        "tensor_names": tensor_names,
        "physical_parameter_accounting": _plain(
            receipt["physical_parameter_accounting"]
        ),
        "explicit_interface_adapter_parameter_elements": 0,
        "hidden_sidecar_parameter_elements": 0,
        "runtime_parameter_sharing_used": False,
    }


def _schedule(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    rows = [
        {
            "candidate_id": "cached-parent-reference",
            "mode": "cached-parent-reference",
            "candidate_type": "cached-parent-reference",
            "candidate_state_sha256": None,
        }
    ]
    rows.extend(
        {
            "candidate_id": row["candidate_id"],
            "mode": (
                "dense-compact-negative-control"
                if row["candidate_type"] == "dense-compact-negative-control"
                else "protected-standalone-adapter-control"
                if row["candidate_type"] == "protected-standalone-adapter-control"
                else "protected-parent-shell"
            ),
            "candidate_type": row["candidate_type"],
            "candidate_state_sha256": row["candidate_state_sha256"],
        }
        for row in candidates
    )
    body = {
        "schema": SCHEDULE_SCHEMA,
        "status": "NON-FINAL-PROTECTED-SHORT-CONV-DIAGNOSTIC-SCHEDULE",
        "candidate_ids": [row["candidate_id"] for row in rows],
        "candidates": rows,
        "candidate_count": len(rows),
        "cached_parent_reference_count": 1,
        "protected_parent_shell_count": sum(
            row["candidate_type"] == "protected-parent-shell" for row in rows
        ),
        "dense_compact_negative_control_count": sum(
            row["candidate_type"] == "dense-compact-negative-control" for row in rows
        ),
        "protected_standalone_adapter_control_count": sum(
            row["candidate_type"] == "protected-standalone-adapter-control"
            for row in rows
        ),
        "numeric_acceptance_gate_defined": False,
    }
    return {**body, "schedule_sha256": canonical_sha256(body)}


def authenticate_schedule(schedule: Mapping[str, Any]) -> Mapping[str, Any]:
    """Authenticate the fixed diagnostic schedule."""

    if not isinstance(schedule, Mapping):
        raise Lfm2ProtectedShortConvPlanProducerError(
            "Protected short-conv schedule must be one mapping."
        )
    value = dict(schedule)
    supplied = value.pop("schedule_sha256", None)
    rows = schedule.get("candidates")
    if (
        schedule.get("schema") != SCHEDULE_SCHEMA
        or supplied != canonical_sha256(value)
        or not isinstance(rows, list)
        or schedule.get("candidate_count") != len(rows)
        or schedule.get("candidate_ids") != [row.get("candidate_id") for row in rows]
        or tuple(schedule.get("candidate_ids", ())) != EXPECTED_SCHEDULE_CANDIDATE_IDS
        or len(set(schedule.get("candidate_ids", ()))) != len(rows)
        or schedule.get("cached_parent_reference_count") != 1
        or schedule.get("protected_parent_shell_count") != 4
        or schedule.get("dense_compact_negative_control_count") != 1
        or schedule.get("protected_standalone_adapter_control_count") != 1
        or schedule.get("numeric_acceptance_gate_defined") is not False
    ):
        raise Lfm2ProtectedShortConvPlanProducerError(
            "Protected short-conv schedule seal or composition is invalid."
        )
    return schedule


def prepare(
    *,
    output_directory: Path,
    component_publication_directory: Path,
    expected_component_producer_receipt_sha256: str,
    expected_component_producer_receipt_file_sha256: str,
    expected_component_plan_sha256: str,
    expected_component_plan_file_sha256: str,
    expected_component_bridges_file_sha256: str,
    expected_a0019_source_run_result_sha256: str,
    expected_a0019_source_run_file_sha256: str,
    parent_checkpoint: Path,
    expected_parent_checkpoint_inventory_sha256: str,
    final_checkpoint: Path,
    expected_final_checkpoint_inventory_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate A0019/B038 and atomically publish six candidates."""

    import torch
    from safetensors.torch import save_file

    target = _new_output(output_directory)
    component_root = b038._ordinary_directory(
        component_publication_directory,
        label="component publication directory",
    )
    parent_path = b038._ordinary_directory(parent_checkpoint, label="parent checkpoint")
    final_path = b038._ordinary_directory(final_checkpoint, label="final checkpoint")
    _forbid_output_inside_authenticated_inputs(
        target,
        input_directories=(component_root, parent_path, final_path),
    )
    receipt_path = component_root / b038.PRODUCER_RECEIPT_FILENAME
    component_plan_path = component_root / b038.PLAN_FILENAME
    bridge_path = component_root / b038.BRIDGE_FILENAME
    component_receipt = b038._load_producer_receipt(
        receipt_path,
        expected_receipt_sha256=expected_component_producer_receipt_sha256,
        expected_file_sha256=expected_component_producer_receipt_file_sha256,
    )
    component_plan = b038._load_plan(
        component_plan_path,
        expected_plan_sha256=expected_component_plan_sha256,
        expected_file_sha256=expected_component_plan_file_sha256,
    )
    b038._authenticate_producer_publication(
        producer_receipt=component_receipt,
        producer_receipt_path=receipt_path,
        plan=component_plan,
        component_plan=component_plan_path,
        bridge_tensors=bridge_path,
        expected_component_plan_sha256=expected_component_plan_sha256,
        expected_component_plan_file_sha256=expected_component_plan_file_sha256,
        expected_bridge_tensors_file_sha256=expected_component_bridges_file_sha256,
    )
    source_auth = component_plan.get("source_authentication", {})
    if (
        source_auth.get("source_run_result_sha256")
        != _digest(
            expected_a0019_source_run_result_sha256,
            label="expected A0019 source run result",
        )
        or source_auth.get("source_run_receipt_file_sha256")
        != _digest(
            expected_a0019_source_run_file_sha256,
            label="expected A0019 source run receipt file",
        )
        or component_plan.get("materialization_dtype") != "float32"
    ):
        raise Lfm2ProtectedShortConvPlanProducerError(
            "B038 does not bind the externally pinned FP32 A0019 source."
        )
    bridges = b038._load_bridge_tensors(
        bridge_path,
        expected_file_sha256=expected_component_bridges_file_sha256,
        plan=component_plan,
    )
    encoder_source = bridges.get("foundation.producer_encoder")
    decoder_source = bridges.get("foundation.analytic_output_decoder")
    if encoder_source is None or decoder_source is None:
        raise Lfm2ProtectedShortConvPlanProducerError(
            "B038 publication lacks the foundation bridge pair."
        )
    encoder = encoder_source.detach().to(dtype=torch.float32).cpu().contiguous()
    decoder = decoder_source.detach().to(dtype=torch.float32).cpu().contiguous()
    parent_inventory_sha = canonical_sha256(b038._checkpoint_inventory(parent_path))
    final_inventory_sha = canonical_sha256(b038._checkpoint_inventory(final_path))
    if (
        parent_inventory_sha
        != _digest(
            expected_parent_checkpoint_inventory_sha256,
            label="expected parent checkpoint inventory",
        )
        or final_inventory_sha
        != _digest(
            expected_final_checkpoint_inventory_sha256,
            label="expected final checkpoint inventory",
        )
        or parent_inventory_sha != component_plan["parent_checkpoint_inventory_sha256"]
        or final_inventory_sha != component_plan["compact_checkpoint_inventory_sha256"]
    ):
        raise Lfm2ProtectedShortConvPlanProducerError(
            "Parent/final checkpoint inventory differs from B038 ancestry."
        )
    parent_model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    final_model = load_lfm2_direct_local_model(
        final_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    parent_state_sha = fingerprint_lfm2_direct_local_state(parent_model)
    final_state_sha = fingerprint_lfm2_direct_local_state(final_model)
    if (
        parent_state_sha != component_plan["parent_model_state_sha256"]
        or final_state_sha != component_plan["compact_model_state_sha256"]
    ):
        raise Lfm2ProtectedShortConvPlanProducerError(
            "Direct-local checkpoint state differs from B038 provenance."
        )
    parent_hidden = int(parent_model.config.hidden_size)
    child_hidden = int(final_model.config.hidden_size)
    if (
        tuple(encoder.shape) != (parent_hidden, child_hidden)
        or tuple(decoder.shape) != (child_hidden, parent_hidden)
        or child_hidden >= parent_hidden
    ):
        raise Lfm2ProtectedShortConvPlanProducerError(
            "Materialized residual bridge pair has stale parent/child geometry."
        )
    components = _component_rows(component_plan)
    source_rows = {}
    for layer_index in (0, 1):
        identifier = f"layer-{layer_index:03d}-short_conv"
        row = components.get(identifier)
        path = f"model.layers.{layer_index}.conv"
        if (
            row is None
            or row.get("kind") != "short_conv"
            or row.get("parent_module_path") != path
            or row.get("compact_module_path") != path
        ):
            raise Lfm2ProtectedShortConvPlanProducerError(
                f"B038 lacks canonical short-conv component {identifier!r}."
            )
        parent_module = parent_model.get_submodule(path)
        compact_module = final_model.get_submodule(path)
        if (
            _module_state_sha256(parent_module) != row["parent_module_sha256"]
            or _module_state_sha256(compact_module) != row["compact_module_sha256"]
        ):
            raise Lfm2ProtectedShortConvPlanProducerError(
                f"Live short-conv {identifier!r} differs from B038 module pins."
            )
        source_rows[layer_index] = (row, parent_module, compact_module)

    archive: dict[str, Any] = {}
    candidates = []
    for layer_index, variant in (
        (0, "parent-input-roundtrip"),
        (0, "parent-output-roundtrip"),
        (0, "parent-both-roundtrip"),
        (1, "parent-both-roundtrip"),
    ):
        _, parent_module, _ = source_rows[layer_index]
        source = _protected_source(
            module=parent_module,
            layer_index=layer_index,
            parent_hidden=parent_hidden,
            child_hidden=child_hidden,
            encoder=encoder,
            decoder=decoder,
        )
        source_fingerprint = fingerprint_lfm2_protected_short_conv_input(source)
        candidate = compile_lfm2_protected_short_conv_diagnostic(
            source,
            variant=variant,
            expected_input_bundle_sha256=source_fingerprint["input_bundle_sha256"],
            expected_tensor_sha256={
                name: record["sha256"]
                for name, record in source_fingerprint["tensor_records"].items()
            },
        )
        candidate_id = f"layer-{layer_index:03d}-{variant}"
        candidates.append(
            _archive_candidate(
                candidate_id=candidate_id,
                candidate_type="protected-parent-shell",
                layer_index=layer_index,
                module_path=f"model.layers.{layer_index}.conv",
                candidate=candidate,
                archive=archive,
            )
        )

    _, _, compact_layer_zero = source_rows[0]
    dense_source = _dense_source(
        module=compact_layer_zero,
        layer_index=0,
        parent_hidden=parent_hidden,
        child_hidden=child_hidden,
        encoder=encoder_source,
        decoder=decoder_source,
    )
    dense_fingerprint = fingerprint_lfm2_fused_body_diagnostic_input(dense_source)
    dense_candidate = compile_lfm2_fused_short_conv_diagnostic(
        dense_source,
        expected_input_bundle_sha256=dense_fingerprint["input_bundle_sha256"],
        expected_tensor_sha256={
            name: record["sha256"]
            for name, record in dense_fingerprint["tensor_records"].items()
        },
    )
    candidates.insert(
        3,
        _archive_candidate(
            candidate_id="layer-000-dense-compact-negative-control",
            candidate_type="dense-compact-negative-control",
            layer_index=0,
            module_path="model.layers.0.conv",
            candidate=dense_candidate,
            archive=archive,
        ),
    )
    _, parent_layer_zero, _ = source_rows[0]
    standalone_source = _protected_source(
        module=parent_layer_zero,
        layer_index=0,
        parent_hidden=parent_hidden,
        child_hidden=child_hidden,
        encoder=encoder,
        decoder=decoder,
    )
    standalone_fingerprint = fingerprint_lfm2_protected_short_conv_input(
        standalone_source
    )
    standalone_candidate = compile_lfm2_protected_short_conv_diagnostic(
        standalone_source,
        variant="standalone-child-stem",
        expected_input_bundle_sha256=standalone_fingerprint["input_bundle_sha256"],
        expected_tensor_sha256={
            name: record["sha256"]
            for name, record in standalone_fingerprint["tensor_records"].items()
        },
    )
    standalone_row = dict(
        _archive_candidate(
            candidate_id="layer-000-standalone-child-stem-adapter-control",
            candidate_type="protected-parent-shell",
            layer_index=0,
            module_path="model.layers.0.conv",
            candidate=standalone_candidate,
            archive=archive,
        )
    )
    standalone_row["candidate_type"] = "protected-standalone-adapter-control"
    input_adapter_name = (
        "candidate.layer-000-standalone-child-stem-adapter-control.input_bridge.weight"
    )
    output_adapter_name = (
        "candidate.layer-000-standalone-child-stem-adapter-control.output_bridge.weight"
    )
    archive[input_adapter_name] = encoder.T.contiguous().clone()
    archive[output_adapter_name] = decoder.T.contiguous().clone()
    core_elements = int(
        standalone_row["physical_parameter_accounting"]["candidate_parameter_elements"]
    )
    bridge_elements = 2 * parent_hidden * child_hidden
    parent_component_elements = int(
        standalone_row["physical_parameter_accounting"][
            "original_parent_component_parameter_elements"
        ]
    )
    standalone_row["wrapper_adapter_tensor_names"] = {
        "input_bridge.weight": input_adapter_name,
        "output_bridge.weight": output_adapter_name,
    }
    standalone_row["parent_shell_wrapper_physical_parameter_accounting"] = {
        "protected_standalone_core_parameter_elements": core_elements,
        "input_bridge_parameter_elements": parent_hidden * child_hidden,
        "output_bridge_parameter_elements": child_hidden * parent_hidden,
        "explicitly_counted_bridge_parameter_elements": bridge_elements,
        "installed_parent_shell_parameter_elements": core_elements + bridge_elements,
        "original_parent_component_parameter_elements": parent_component_elements,
        "physical_parameter_reduction_from_parent": (
            parent_component_elements - core_elements - bridge_elements
        ),
        "bridge_adapters_are_diagnostic_runtime_parameters": True,
        "bridge_parameters_hidden_or_omitted_from_accounting": False,
        "parameter_sharing_used": False,
    }
    standalone_row["explicit_interface_adapter_parameter_elements"] = bridge_elements
    candidates.insert(4, standalone_row)
    schedule = authenticate_schedule(_schedule(candidates))
    parent_cache_binding = _parent_cache_binding(
        parent_path=parent_path,
        parent_inventory_sha256=parent_inventory_sha,
        expected_identity_sha256=source_auth["parent_lfm2_identity_sha256"],
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        archive_path = stage / ARCHIVE_FILENAME
        save_file(
            dict(sorted(archive.items())),
            str(archive_path),
            metadata={
                "format": "pt",
                "neural_forge_status": "NON-FINAL",
                "neural_forge_role": "protected-short-conv-diagnostics",
            },
        )
        archive_records = {
            name: _tensor_record(tensor) for name, tensor in sorted(archive.items())
        }
        archive_file_sha = _file_sha256(archive_path)
        plan_body = {
            "schema": PLAN_SCHEMA,
            "status": PLAN_STATUS,
            "producer_schema": PRODUCER_SCHEMA,
            "source_authentication": {
                "a0019_source_run_result_sha256": source_auth[
                    "source_run_result_sha256"
                ],
                "a0019_source_run_receipt_file_sha256": source_auth[
                    "source_run_receipt_file_sha256"
                ],
                "component_producer_receipt_sha256": component_receipt[
                    "receipt_sha256"
                ],
                "component_producer_receipt_file_sha256": _file_sha256(receipt_path),
                "component_plan_sha256": component_plan["plan_sha256"],
                "component_plan_file_sha256": _file_sha256(component_plan_path),
                "component_bridges_file_sha256": _file_sha256(bridge_path),
                "parent_checkpoint_inventory_sha256": parent_inventory_sha,
                "final_checkpoint_inventory_sha256": final_inventory_sha,
                "foundation_encoder_source_record": _tensor_record(encoder_source),
                "analytic_decoder_source_record": _tensor_record(decoder_source),
                "foundation_encoder_runtime_fp32_record": _tensor_record(encoder),
                "analytic_decoder_runtime_fp32_record": _tensor_record(decoder),
                "prepare_time_fp32_bridge_materialization_performed": True,
            },
            "parent_checkpoint_inventory_sha256": parent_inventory_sha,
            "final_checkpoint_inventory_sha256": final_inventory_sha,
            "parent_model_state_sha256": parent_state_sha,
            "final_model_state_sha256": final_state_sha,
            "parent_unique_parameters": _unique_parameter_elements(parent_model),
            "final_unique_parameters": _unique_parameter_elements(final_model),
            "parent_hidden_size": parent_hidden,
            "child_hidden_size": child_hidden,
            "materialization_dtype": "float32",
            "parent_cache_binding": _plain(parent_cache_binding),
            "candidates": candidates,
            "candidate_count": len(candidates),
            "schedule": _plain(schedule),
            "archive": {
                "schema": ARCHIVE_SCHEMA,
                "filename": ARCHIVE_FILENAME,
                "file_sha256": archive_file_sha,
                "tensor_records": archive_records,
                "tensor_count": len(archive_records),
            },
            "physical_accounting_policy": {
                "all_runtime_parameter_storage_counted": True,
                "source_bridge_factors_retained_only_as_explicit_counted_standalone_control_adapters": True,
                "shared_parameter_elements": 0,
                "hidden_sidecar_parameter_elements": 0,
                "protected_parent_shell_candidates_reduce_component_parameters": False,
                "dense_compact_negative_control_reduces_component_parameters": True,
                "standalone_adapter_control_bridges_explicitly_counted": True,
                "standalone_adapter_control_may_expand_parent_component": True,
            },
            "evaluation_policy": {
                "complete_sealed_prompt_bank_required": True,
                "maximum_prompt_tokens": 1024,
                "maximum_new_tokens": 256,
                "exact_token_match_required": False,
                "numeric_metrics_have_decision_authority": False,
                "manual_intelligence_review_required": True,
            },
            "operation_audit": {
                "parent_model_forward_calls": 0,
                "final_model_forward_calls": 0,
                "model_generate_calls": 0,
                "parent_trace_recaptures": 0,
                "new_parent_trace_rows": 0,
                "training_steps": 0,
                "optimizer_constructions": 0,
                "optimizer_steps": 0,
                "autograd_grad_calls": 0,
                "backward_calls": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "parameters_sparsified": 0,
                "runtime_parameter_aliases_created": 0,
                "network_access": False,
            },
            "interpretation": {
                "diagnostic_parent_shell_only": True,
                "standalone_child_compression_claimed": False,
                "does_not_modify_main_exact4x_compiler_or_b038": True,
                "protected_channel_algebra_hypothesis_isolated": True,
                "layer_zero_stem_is_primary_test": True,
                "layer_one_both_roundtrip_is_control": True,
                "dense_compact_layer_zero_is_negative_control": True,
                "standalone_child_stem_with_counted_adapters_is_direct_replacement_control": True,
                "standalone_adapter_control_is_algebraically_compared_to_parent_both": True,
            },
        }
        plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
        _json_write(stage / PLAN_FILENAME, plan)
        producer_body = {
            "schema": PRODUCER_SCHEMA,
            "status": PRODUCER_STATUS,
            "plan_filename": PLAN_FILENAME,
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": _file_sha256(stage / PLAN_FILENAME),
            "archive_filename": ARCHIVE_FILENAME,
            "archive_file_sha256": archive_file_sha,
            "schedule_sha256": schedule["schedule_sha256"],
            "candidate_count": len(candidates),
            "schedule_candidate_count": schedule["candidate_count"],
            "a0019_source_run_result_sha256": source_auth["source_run_result_sha256"],
            "component_plan_sha256": component_plan["plan_sha256"],
            "fresh_direct_local_source_authentication_performed": True,
            "fresh_candidate_authentication_performed": True,
            "training_steps": 0,
            "model_forward_calls": 0,
            "network_access": False,
        }
        producer_receipt = {
            **producer_body,
            "receipt_sha256": canonical_sha256(producer_body),
        }
        _json_write(stage / PRODUCER_RECEIPT_FILENAME, producer_receipt)
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        del parent_model, final_model
    return producer_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--component-publication-directory", type=Path, required=True)
    parser.add_argument("--expected-component-producer-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-component-producer-receipt-file-sha256", required=True
    )
    parser.add_argument("--expected-component-plan-sha256", required=True)
    parser.add_argument("--expected-component-plan-file-sha256", required=True)
    parser.add_argument("--expected-component-bridges-file-sha256", required=True)
    parser.add_argument("--expected-a0019-source-run-result-sha256", required=True)
    parser.add_argument("--expected-a0019-source-run-file-sha256", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-final-checkpoint-inventory-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = prepare(**vars(arguments))
    print(json.dumps(_plain(receipt), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
