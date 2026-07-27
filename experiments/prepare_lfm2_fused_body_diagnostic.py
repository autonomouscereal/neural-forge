#!/usr/bin/env python3
"""Publish authenticated fold-only LFM2 fused-body diagnostic candidates.

This additive producer consumes the existing progressive component-ablation
publication (b038) and the already-materialized parent/final checkpoints.  It
absorbs the authenticated residual encoder/decoder into each compact MLP and
short-convolution operator using the committed fused-body core.  It performs
no model forward, generation, trace capture, training, pruning, quantization,
or network loading.
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
    Lfm2FusedMlpDiagnosticInput,
    Lfm2FusedShortConvDiagnosticInput,
    authenticate_lfm2_fused_body_candidate,
    compile_lfm2_fused_mlp_diagnostic,
    compile_lfm2_fused_short_conv_diagnostic,
    fingerprint_lfm2_fused_body_diagnostic_input,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _module_state_sha256,
    _unique_parameter_elements,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


PLAN_SCHEMA = "neural-forge-lfm2-fused-body-diagnostic-plan-v1"
PRODUCER_SCHEMA = "neural-forge-lfm2-fused-body-plan-producer-v1"
ARCHIVE_SCHEMA = "neural-forge-lfm2-fused-body-candidate-archive-v1"
SCHEDULE_SCHEMA = "neural-forge-lfm2-fused-body-diagnostic-schedule-v1"
ARCHIVE_FILENAME = "fused-body-candidates.safetensors"
PLAN_FILENAME = "fused-body-plan.json"
PRODUCER_RECEIPT_FILENAME = "fused-body-producer-receipt.json"
PLAN_STATUS = "NON-FINAL-FUSED-BODY-DIAGNOSTIC-PLAN-AUTOMATICALLY-DERIVED"
PRODUCER_STATUS = "fused-body-candidates-and-plan-publication-complete"


class Lfm2FusedBodyPlanProducerError(RuntimeError):
    """An authenticated source, fold result, schedule, or publication is invalid."""


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2FusedBodyPlanProducerError(
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
        raise Lfm2FusedBodyPlanProducerError("output directory must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2FusedBodyPlanProducerError("output directory must not already exist.")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2FusedBodyPlanProducerError(
            "output parent must be one ordinary directory."
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
            raise Lfm2FusedBodyPlanProducerError(
                "Output directory must be lexically separate from every "
                "authenticated input checkpoint/publication directory."
            )


def _tensor_record(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(size) for size in value.shape],
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": lfm2_authenticated_tensor_sha256(value),
    }


def _layer_index(path: Any, *, suffix: str) -> int:
    if not isinstance(path, str):
        raise Lfm2FusedBodyPlanProducerError("Component module path is not text.")
    parts = path.split(".")
    if (
        len(parts) != 4
        or parts[:2] != ["model", "layers"]
        or not parts[2].isdigit()
        or parts[3] != suffix
    ):
        raise Lfm2FusedBodyPlanProducerError(
            f"Component module path {path!r} is not one canonical {suffix!r} path."
        )
    return int(parts[2])


def _ordered_components(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    parsed = []
    identifiers: set[str] = set()
    locations: set[tuple[int, str]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise Lfm2FusedBodyPlanProducerError(
                "Fused-body component row is malformed."
            )
        kind = raw.get("kind")
        component_id = raw.get("component_id")
        if kind not in {"mlp", "short_conv"} or not isinstance(component_id, str):
            raise Lfm2FusedBodyPlanProducerError(
                "Fused-body rows must name canonical MLP or short-conv components."
            )
        suffix = "feed_forward" if kind == "mlp" else "conv"
        index = _layer_index(raw.get("parent_module_path"), suffix=suffix)
        if (
            component_id in identifiers
            or (index, kind) in locations
            or raw.get("compact_module_path") != raw.get("parent_module_path")
        ):
            raise Lfm2FusedBodyPlanProducerError(
                "Fused-body component identifiers/locations are duplicated or divergent."
            )
        identifiers.add(component_id)
        locations.add((index, kind))
        parsed.append((index, 0 if kind == "short_conv" else 1, raw))
    if not parsed or not any(row[2].get("kind") == "mlp" for row in parsed):
        raise Lfm2FusedBodyPlanProducerError(
            "Fused-body production requires at least one authenticated MLP."
        )
    return tuple(value[2] for value in sorted(parsed, key=lambda value: value[:2]))


def _schedule_document(
    component_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build and seal all requested lanes, deduplicated by installed component set."""

    rows = _ordered_components(component_rows)
    candidates: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    seen: dict[frozenset[str], str] = {}
    by_id = {str(row["component_id"]): row for row in rows}

    def add(candidate_id: str, mode: str, component_ids: Sequence[str]) -> None:
        identifiers = tuple(component_ids)
        signature = frozenset(identifiers)
        if len(signature) != len(identifiers):
            raise Lfm2FusedBodyPlanProducerError(
                "Schedule candidates must contain one unique component set."
            )
        if any(identifier not in by_id for identifier in identifiers):
            raise Lfm2FusedBodyPlanProducerError(
                "Schedule candidate names an unavailable fused-body component."
            )
        if signature in seen:
            aliases.append(
                {
                    "omitted_candidate_id": candidate_id,
                    "retained_candidate_id": seen[signature],
                    "requested_mode": mode,
                    "reason": "identical-installed-component-set",
                }
            )
            return
        seen[signature] = candidate_id
        candidates.append(
            {
                "candidate_id": candidate_id,
                "mode": mode,
                "source_stage": (
                    "cached-parent-bfloat16-reference"
                    if not identifiers
                    else "final-s2"
                ),
                "component_ids": list(identifiers),
                "component_state_sha256": {
                    identifier: by_id[identifier]["candidate_state_sha256"]
                    for identifier in identifiers
                },
            }
        )

    add("cached-parent-reference", "cached-parent-reference", ())
    for row in rows:
        add(
            f"isolated--{row['component_id']}",
            "isolated-component",
            (str(row["component_id"]),),
        )
    mlp_ids = tuple(str(row["component_id"]) for row in rows if row["kind"] == "mlp")
    conv_ids = tuple(
        str(row["component_id"]) for row in rows if row["kind"] == "short_conv"
    )
    add("all-mlp", "all-mlp", mlp_ids)
    if conv_ids:
        add("all-short-conv", "all-short-conv", conv_ids)
    by_layer_kind = {
        (int(row["layer_index"]), str(row["kind"])): str(row["component_id"])
        for row in rows
    }
    for layer_index in sorted({int(row["layer_index"]) for row in rows}):
        conv_id = by_layer_kind.get((layer_index, "short_conv"))
        mlp_id = by_layer_kind.get((layer_index, "mlp"))
        if conv_id is not None and mlp_id is not None:
            add(
                f"paired-layer-{layer_index:03d}-conv-mlp",
                "paired-same-layer-conv-mlp",
                (conv_id, mlp_id),
            )
    ordered_ids = tuple(str(row["component_id"]) for row in rows)
    if conv_ids:
        add("all-fused-body", "subsystem-combination-all-body", ordered_ids)
    for layer_index in sorted({int(row["layer_index"]) for row in rows}):
        cumulative = tuple(
            str(row["component_id"])
            for row in rows
            if int(row["layer_index"]) <= layer_index
        )
        add(
            f"cumulative-body-through-layer-{layer_index:03d}",
            "cumulative-mlp-conv-by-layer",
            cumulative,
        )
    for prefix_length in range(1, len(ordered_ids) + 1):
        add(
            f"ordered-body-prefix-{prefix_length:03d}",
            "ordered-body-prefix",
            ordered_ids[:prefix_length],
        )
    candidate_ids = [row["candidate_id"] for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)) or len(seen) != len(candidates):
        raise Lfm2FusedBodyPlanProducerError(
            "Fused-body schedule contains duplicate identifiers or component sets."
        )
    body = {
        "schema": SCHEDULE_SCHEMA,
        "status": "NON-FINAL-FUSED-BODY-DIAGNOSTIC-SCHEDULE",
        "component_order": list(ordered_ids),
        "component_order_policy": "layer-index-ascending-short-conv-before-mlp",
        "candidate_ids": candidate_ids,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "mode_counts": {
            mode: sum(row["mode"] == mode for row in candidates)
            for mode in (
                "isolated-component",
                "cached-parent-reference",
                "all-mlp",
                "all-short-conv",
                "paired-same-layer-conv-mlp",
                "subsystem-combination-all-body",
                "cumulative-mlp-conv-by-layer",
                "ordered-body-prefix",
            )
        },
        "deduplicated_candidate_aliases": aliases,
        "deduplication_key": "unordered-complete-installed-component-id-set",
        "all_requested_lanes_considered": True,
        "numeric_acceptance_gate_defined": False,
    }
    return {**body, "schedule_sha256": canonical_sha256(body)}


def _authenticate_schedule(schedule: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(schedule, Mapping):
        raise Lfm2FusedBodyPlanProducerError("Schedule must be one mapping.")
    value = dict(schedule)
    supplied = value.pop("schedule_sha256", None)
    candidates = schedule.get("candidates")
    if (
        schedule.get("schema") != SCHEDULE_SCHEMA
        or supplied != canonical_sha256(value)
        or not isinstance(candidates, list)
        or not candidates
        or schedule.get("candidate_count") != len(candidates)
        or schedule.get("candidate_ids")
        != [row.get("candidate_id") for row in candidates]
        or len({frozenset(row.get("component_ids", ())) for row in candidates})
        != len(candidates)
        or schedule.get("all_requested_lanes_considered") is not True
        or schedule.get("numeric_acceptance_gate_defined") is not False
    ):
        raise Lfm2FusedBodyPlanProducerError("Fused-body schedule seal is invalid.")
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
    parent_checkpoint: Path,
    expected_parent_checkpoint_inventory_sha256: str,
    final_checkpoint: Path,
    expected_final_checkpoint_inventory_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate b038 ancestry and atomically publish fused body candidates."""

    import torch
    from safetensors.torch import save_file

    target = _new_output(output_directory)
    publication_root = b038._ordinary_directory(
        component_publication_directory,
        label="component publication directory",
    )
    parent_path = b038._ordinary_directory(parent_checkpoint, label="parent checkpoint")
    final_path = b038._ordinary_directory(final_checkpoint, label="final checkpoint")
    _forbid_output_inside_authenticated_inputs(
        target,
        input_directories=(publication_root, parent_path, final_path),
    )
    receipt_path = publication_root / b038.PRODUCER_RECEIPT_FILENAME
    component_plan_path = publication_root / b038.PLAN_FILENAME
    bridge_path = publication_root / b038.BRIDGE_FILENAME
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
    bridges = b038._load_bridge_tensors(
        bridge_path,
        expected_file_sha256=expected_component_bridges_file_sha256,
        plan=component_plan,
    )
    if component_plan.get("materialization_dtype") != "float32":
        raise Lfm2FusedBodyPlanProducerError(
            "Fused-body production requires the actual runtime FP32 b038 state."
        )
    parent_inventory = b038._checkpoint_inventory(parent_path)
    final_inventory = b038._checkpoint_inventory(final_path)
    parent_inventory_sha = canonical_sha256(parent_inventory)
    final_inventory_sha = canonical_sha256(final_inventory)
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
        raise Lfm2FusedBodyPlanProducerError(
            "Parent/final checkpoint inventory differs from b038 ancestry."
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
        raise Lfm2FusedBodyPlanProducerError(
            "Direct-local parent/final state differs from the authenticated b038 plan."
        )
    parent_cache_model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    parent_cache_identity = fingerprint_lfm2_parent(parent_cache_model).identity_sha256
    parent_cache_state = fingerprint_lfm2_direct_local_state(parent_cache_model)
    parent_unique_parameters = _unique_parameter_elements(parent_cache_model)
    parent_tokenizer_sha = fingerprint_local_tokenizer(parent_path)
    if parent_cache_identity != component_plan["source_authentication"].get(
        "parent_lfm2_identity_sha256"
    ):
        raise Lfm2FusedBodyPlanProducerError(
            "Fresh BF16 parent identity differs from authenticated b038 provenance."
        )
    parent_cache_binding_body = {
        "schema": "neural-forge-lfm2-fused-body-parent-cache-binding-v1",
        "status": "exact-local-parent-cache-source-authenticated",
        "parent_checkpoint_inventory_sha256": parent_inventory_sha,
        "parent_runtime_materialization_dtype": "bfloat16",
        "parent_state_sha256": parent_cache_state,
        "parent_identity_sha256": parent_cache_identity,
        "parent_unique_parameters": parent_unique_parameters,
        "tokenizer_sha256": parent_tokenizer_sha,
        "direct_local_checkpoint_source_authenticated": True,
        "hub_loader_permitted": False,
        "network_access_permitted": False,
    }
    parent_cache_binding = {
        **parent_cache_binding_body,
        "binding_sha256": canonical_sha256(parent_cache_binding_body),
    }
    del parent_cache_model
    encoder_name = "foundation.producer_encoder"
    decoder_name = "foundation.analytic_output_decoder"
    encoder = bridges.get(encoder_name)
    decoder = bridges.get(decoder_name)
    bridge_records = component_plan.get("bridge_archive", {}).get("tensor_records", {})
    if (
        encoder is None
        or decoder is None
        or not isinstance(bridge_records, Mapping)
        or bridge_records.get(encoder_name, {}).get("sha256")
        != lfm2_authenticated_tensor_sha256(encoder)
        or bridge_records.get(decoder_name, {}).get("sha256")
        != lfm2_authenticated_tensor_sha256(decoder)
    ):
        raise Lfm2FusedBodyPlanProducerError(
            "b038 publication lacks authenticated foundation encoder/decoder tensors."
        )
    verified_fold_receipts = component_plan["source_authentication"].get(
        "component_fold_receipts"
    )
    if not isinstance(verified_fold_receipts, Mapping):
        raise Lfm2FusedBodyPlanProducerError(
            "b038 component fold-receipt provenance is absent."
        )

    source_rows = _ordered_components(
        [
            row
            for row in component_plan["components"]
            if isinstance(row, Mapping) and row.get("kind") in {"mlp", "short_conv"}
        ]
    )
    archive_tensors: dict[str, Any] = {}
    candidate_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        component_id = str(source_row["component_id"])
        kind = str(source_row["kind"])
        parent_module = parent_model.get_submodule(
            str(source_row["parent_module_path"])
        )
        compact_module = final_model.get_submodule(
            str(source_row["compact_module_path"])
        )
        fold_receipt = verified_fold_receipts.get(component_id)
        if (
            source_row.get("granularity") != "whole-subsystem"
            or source_row.get("input_encoder_tensor") != encoder_name
            or source_row.get("output_decoder_tensor") != decoder_name
            or source_row.get("output_domain_decoder_tensor") is not None
            or source_row.get("input_encoder_sha256")
            != lfm2_authenticated_tensor_sha256(encoder)
            or source_row.get("output_decoder_sha256")
            != lfm2_authenticated_tensor_sha256(decoder)
            or _module_state_sha256(parent_module)
            != _digest(
                source_row.get("parent_module_sha256"),
                label=f"{component_id} parent module",
            )
            or _module_state_sha256(compact_module)
            != _digest(
                source_row.get("compact_module_sha256"),
                label=f"{component_id} compact module",
            )
            or source_row.get("require_parameter_reduction") is not True
            or source_row.get("source_compact_checkpoint_sha256") != final_inventory_sha
            or not isinstance(fold_receipt, Mapping)
            or fold_receipt.get("receipt_sha256")
            != source_row.get("source_fold_receipt_sha256")
            or fold_receipt.get("receipt_seal_freshly_verified") is not True
        ):
            raise Lfm2FusedBodyPlanProducerError(
                f"Component {component_id!r} diverges from authenticated b038 provenance."
            )
        layer_index = _layer_index(
            source_row["parent_module_path"],
            suffix="feed_forward" if kind == "mlp" else "conv",
        )
        if kind == "mlp":
            source = Lfm2FusedMlpDiagnosticInput(
                parent_hidden_size=int(parent_module.w1.weight.shape[1]),
                parent_intermediate_size=int(parent_module.w1.weight.shape[0]),
                compact_hidden_size=int(compact_module.w1.weight.shape[1]),
                compact_intermediate_size=int(compact_module.w1.weight.shape[0]),
                compact_w1_weight=compact_module.w1.weight,
                compact_w2_weight=compact_module.w2.weight,
                compact_w3_weight=compact_module.w3.weight,
                input_encoder=encoder,
                output_decoder=decoder,
                compact_w1_bias=getattr(compact_module.w1, "bias", None),
                compact_w2_bias=getattr(compact_module.w2, "bias", None),
                compact_w3_bias=getattr(compact_module.w3, "bias", None),
            )
        else:
            source = Lfm2FusedShortConvDiagnosticInput(
                layer_index=layer_index,
                parent_hidden_size=int(parent_module.in_proj.weight.shape[1]),
                compact_hidden_size=int(compact_module.in_proj.weight.shape[1]),
                kernel_size=int(compact_module.conv.weight.shape[-1]),
                compact_conv_weight=compact_module.conv.weight,
                compact_in_proj_weight=compact_module.in_proj.weight,
                compact_out_proj_weight=compact_module.out_proj.weight,
                input_encoder=encoder,
                output_decoder=decoder,
                compact_conv_bias=getattr(compact_module.conv, "bias", None),
                compact_in_proj_bias=getattr(compact_module.in_proj, "bias", None),
                compact_out_proj_bias=getattr(compact_module.out_proj, "bias", None),
            )
        input_authentication = fingerprint_lfm2_fused_body_diagnostic_input(source)
        pins = {
            name: record["sha256"]
            for name, record in input_authentication["tensor_records"].items()
        }
        candidate = (
            compile_lfm2_fused_mlp_diagnostic(
                source,
                expected_input_bundle_sha256=input_authentication[
                    "input_bundle_sha256"
                ],
                expected_tensor_sha256=pins,
            )
            if kind == "mlp"
            else compile_lfm2_fused_short_conv_diagnostic(
                source,
                expected_input_bundle_sha256=input_authentication[
                    "input_bundle_sha256"
                ],
                expected_tensor_sha256=pins,
            )
        )
        authenticated = authenticate_lfm2_fused_body_candidate(candidate)
        tensor_names = {}
        for state_name, tensor in candidate.state_dict().items():
            archive_name = f"component.{component_id}.{state_name}"
            if archive_name in archive_tensors:
                raise Lfm2FusedBodyPlanProducerError(
                    "Fused-body candidate archive tensor names collided."
                )
            archive_tensors[archive_name] = tensor.detach().cpu().contiguous()
            tensor_names[state_name] = archive_name
        candidate_rows.append(
            {
                "component_id": component_id,
                "kind": kind,
                "source_stage": "final-s2",
                "source_stage_bytes_exactly_loaded": True,
                "layer_index": layer_index,
                "parent_module_path": source_row["parent_module_path"],
                "compact_module_path": source_row["compact_module_path"],
                "parent_module_sha256": source_row["parent_module_sha256"],
                "compact_module_sha256": source_row["compact_module_sha256"],
                "source_component_row_sha256": canonical_sha256(_plain(source_row)),
                "source_fold_receipt_sha256": source_row["source_fold_receipt_sha256"],
                "source_compact_checkpoint_sha256": final_inventory_sha,
                "input_authentication": _plain(input_authentication),
                "candidate_state_sha256": authenticated["candidate_state_sha256"],
                "candidate_result_sha256": authenticated["result_sha256"],
                "reconstruction_descriptor_sha256": candidate.reconstruction_descriptor[
                    "descriptor_sha256"
                ],
                "physical_parameter_accounting": _plain(
                    authenticated["physical_parameter_accounting"]
                ),
                "tensor_names": tensor_names,
                "receipt": _plain(candidate.receipt),
                "reconstruction_descriptor": _plain(
                    candidate.reconstruction_descriptor
                ),
            }
        )

    schedule = _authenticate_schedule(_schedule_document(candidate_rows))
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        archive_path = stage / ARCHIVE_FILENAME
        save_file(
            archive_tensors,
            str(archive_path),
            metadata={"format": "pt", "neural_forge_role": "fused-body-diagnostic"},
        )
        records = {
            name: _tensor_record(tensor)
            for name, tensor in sorted(archive_tensors.items())
        }
        plan_body = {
            "schema": PLAN_SCHEMA,
            "status": PLAN_STATUS,
            "producer_schema": PRODUCER_SCHEMA,
            "source_authentication": {
                "component_producer_receipt_sha256": component_receipt[
                    "receipt_sha256"
                ],
                "component_producer_receipt_file_sha256": _file_sha256(receipt_path),
                "component_plan_sha256": component_plan["plan_sha256"],
                "component_plan_file_sha256": _file_sha256(component_plan_path),
                "component_bridges_file_sha256": _file_sha256(bridge_path),
                "component_source_run_result_sha256": component_receipt[
                    "source_run_result_sha256"
                ],
                "parent_checkpoint_inventory_sha256": parent_inventory_sha,
                "final_checkpoint_inventory_sha256": final_inventory_sha,
                "parent_model_state_sha256": parent_state_sha,
                "final_model_state_sha256": final_state_sha,
                "parent_cache_state_sha256": parent_cache_state,
                "parent_lfm2_identity_sha256": parent_cache_identity,
                "parent_tokenizer_sha256": parent_tokenizer_sha,
                "foundation_encoder_sha256": lfm2_authenticated_tensor_sha256(encoder),
                "analytic_decoder_sha256": lfm2_authenticated_tensor_sha256(decoder),
                "all_source_artifacts_freshly_authenticated": True,
                "all_source_tensor_pins_derived_from_authenticated_artifacts": True,
            },
            "parent_checkpoint_inventory_sha256": parent_inventory_sha,
            "final_checkpoint_inventory_sha256": final_inventory_sha,
            "parent_model_state_sha256": parent_state_sha,
            "final_model_state_sha256": final_state_sha,
            "materialization_dtype": "float32",
            "parent_cache_binding": parent_cache_binding,
            "archive": {
                "schema": ARCHIVE_SCHEMA,
                "filename": ARCHIVE_FILENAME,
                "file_sha256": _file_sha256(archive_path),
                "tensor_records": records,
                "tensor_count": len(records),
                "stored_tensor_elements": sum(
                    int(tensor.numel()) for tensor in archive_tensors.values()
                ),
            },
            "components": candidate_rows,
            "component_ids": [row["component_id"] for row in candidate_rows],
            "component_count": len(candidate_rows),
            "mlp_component_count": sum(row["kind"] == "mlp" for row in candidate_rows),
            "short_conv_component_count": sum(
                row["kind"] == "short_conv" for row in candidate_rows
            ),
            "schedule": _plain(schedule),
            "fold_stage_controls": {
                "cached_parent_reference": {
                    "available": True,
                    "exact": True,
                    "source": "authenticated-bfloat16-parent-response-cache",
                    "response_runtime_dtype": "bfloat16",
                    "schedule_candidate_id": "cached-parent-reference",
                    "candidate_runtime_executed": False,
                    "response_attributed_to_fp32_structural_shell": False,
                },
                "fp32_unmodified_parent_structural_shell": {
                    "available": True,
                    "exact_structural_manifest": True,
                    "source": "externally-pinned-parent-checkpoint-runtime-cast-float32",
                    "runtime_dtype": "float32",
                    "candidate_runtime_executed": False,
                    "decoded_responses_claimed": False,
                    "purpose": "save-reload-structure-control-only",
                },
                "final_s2": {
                    "available": True,
                    "exact": True,
                    "source": "externally-pinned-final-checkpoint",
                    "candidate_count": len(candidate_rows),
                },
                "width_matched_s0": {
                    "available": False,
                    "exact": False,
                    "reason": (
                        "b038-authenticates-only-an-S0-structural-checkpoint-inventory-"
                        "digest;this-publication-input-does-not-expose-S0-operator-bytes"
                    ),
                    "b038_structural_checkpoint_inventory_sha256": (
                        component_plan["source_authentication"].get(
                            "structural_body_checkpoint_inventory_sha256"
                        )
                    ),
                    "bytes_invented_or_reconstructed": False,
                    "safe_extension_contract": {
                        "requires_external_local_checkpoint": True,
                        "requires_external_checkpoint_inventory_sha256": True,
                        "requires_external_model_state_sha256": True,
                        "requires_b038_structural_inventory_match": True,
                        "requires_same_compact_topology_and_widths": True,
                        "would_emit_stage_parallel_isolated_group_and_prefix_lanes": True,
                    },
                },
            },
            "diagnostic_parent_shell_only": True,
            "standalone_child_compression_claimed": False,
            "numeric_acceptance_gate_defined": False,
            "bridge_sidecar_parameters_materialized": False,
            "operation_audit": {
                "parent_model_forward_calls": 0,
                "compact_model_forward_calls": 0,
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
                "network_access": False,
            },
        }
        plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
        _json_write(stage / PLAN_FILENAME, plan)
        receipt_body = {
            "schema": PRODUCER_SCHEMA,
            "status": PRODUCER_STATUS,
            "plan_filename": PLAN_FILENAME,
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": _file_sha256(stage / PLAN_FILENAME),
            "archive_filename": ARCHIVE_FILENAME,
            "archive_file_sha256": _file_sha256(archive_path),
            "component_producer_receipt_sha256": component_receipt["receipt_sha256"],
            "parent_checkpoint_inventory_sha256": parent_inventory_sha,
            "final_checkpoint_inventory_sha256": final_inventory_sha,
            "parent_cache_binding_sha256": parent_cache_binding["binding_sha256"],
            "component_count": len(candidate_rows),
            "candidate_schedule_count": schedule["candidate_count"],
            "fresh_source_receipt_authentication_performed": True,
            "fresh_archive_authentication_performed": True,
            "direct_local_checkpoint_loading_used": True,
            "human_authored_module_or_tensor_pins_accepted": False,
            "training_steps": 0,
            "model_forward_calls": 0,
        }
        receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
        _json_write(stage / PRODUCER_RECEIPT_FILENAME, receipt)
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        del parent_model
        del final_model
    return receipt


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
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-final-checkpoint-inventory-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = prepare(**vars(arguments))
    print(json.dumps(_plain(result), allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
