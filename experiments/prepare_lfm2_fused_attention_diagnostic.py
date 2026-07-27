#!/usr/bin/env python3
"""Publish an authenticated, fold-only LFM2 fused-attention diagnostic bank.

This producer is an additive second stage over the committed exact4x and
progressive-component artifacts.  It authenticates the existing component
producer publication, the exact4x source run, the S0 structural candidate
archive, the E/D bridge sources, and local parent/S2 checkpoints.  It then
materializes ordinary parent-I/O reduced-head attention states with closed-form
FP64 bridge fusion.  It performs no model forward, trace capture, training,
optimization, pruning, quantization, or network loading.
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

from experiments import (  # noqa: E402
    prepare_lfm2_progressive_component_ablation as component_producer,
)
from experiments import run_lfm2_progressive_component_ablation as component_runner  # noqa: E402
from experiments.run_lfm2_dynamic_catalog import (  # noqa: E402
    fingerprint_local_tokenizer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_fused_attention_diagnostic import (  # noqa: E402
    Lfm2FusedAttentionDiagnosticInput,
    authenticate_lfm2_fused_attention_candidate,
    compile_lfm2_fused_attention_diagnostic_bank,
    fingerprint_lfm2_fused_attention_diagnostic_input,
    substitute_lfm2_fused_attention_gqa_group,
)
from neural_forge.hf_lfm2_postfold_reclosure_seed_archive import (  # noqa: E402
    load_lfm2_postfold_reclosure_seed_archive,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _module_state_sha256,
    _unique_parameter_elements,
)
from neural_forge.hf_lfm2_structural_attention_candidate_archive import (  # noqa: E402
    load_lfm2_structural_attention_candidate_archive,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


PLAN_SCHEMA = "neural-forge-lfm2-fused-attention-diagnostic-plan-v1"
PRODUCER_SCHEMA = "neural-forge-lfm2-fused-attention-plan-producer-v1"
SCHEDULE_SCHEMA = "neural-forge-lfm2-fused-attention-diagnostic-schedule-v1"
GROUP_SELECTION_SCHEMA = "neural-forge-lfm2-fused-attention-gqa-selections-v1"
BANK_FILENAME = "fused-attention-bank.safetensors"
PLAN_FILENAME = "fused-attention-plan.json"
PRODUCER_RECEIPT_FILENAME = "producer-receipt.json"
PLAN_STATUS = "NON-FINAL-FUSED-ATTENTION-DIAGNOSTIC-PLAN-AUTOMATICALLY-DERIVED"
PRODUCER_STATUS = "fused-attention-bank-and-plan-publication-complete"


class Lfm2FusedAttentionPlanProducerError(RuntimeError):
    """A source artifact, fold bank, selection, or publication is invalid."""


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2FusedAttentionPlanProducerError(
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
        raise Lfm2FusedAttentionPlanProducerError("output directory must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2FusedAttentionPlanProducerError(
            "output directory must not already exist."
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2FusedAttentionPlanProducerError(
            "output parent must be one ordinary directory."
        )
    return path


def _require_output_outside_sources(
    output: Path,
    *,
    source_directories: Sequence[Path],
) -> None:
    """Prevent publication staging from mutating an authenticated input tree."""

    for source in source_directories:
        root = source.expanduser().absolute()
        if output == root or root in output.parents:
            raise Lfm2FusedAttentionPlanProducerError(
                "output directory must be outside every authenticated input directory."
            )


def _tensor_record(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(size) for size in value.shape],
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": lfm2_authenticated_tensor_sha256(value),
    }


def _canonical_token(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in value
        )
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            f"{label} must be one lowercase canonical token."
        )
    return value


def _load_group_selections(
    path: Path | None,
    *,
    expected_file_sha256: str | None,
) -> Mapping[str, Any] | None:
    if path is None:
        if expected_file_sha256 is not None:
            raise Lfm2FusedAttentionPlanProducerError(
                "A group-selection pin was supplied without a selection file."
            )
        return None
    if expected_file_sha256 is None:
        raise Lfm2FusedAttentionPlanProducerError(
            "Group selections require an external file pin."
        )
    ordinary = component_producer._ordinary_file(path, label="GQA group selections")
    if _file_sha256(ordinary) != _digest(
        expected_file_sha256,
        label="expected GQA group-selection file",
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "GQA group-selection file differs from its external pin."
        )
    document = component_producer._load_json(
        ordinary,
        label="GQA group selections",
    )
    body = dict(document)
    supplied = body.pop("selection_sha256", None)
    rows = document.get("selections")
    if (
        document.get("schema") != GROUP_SELECTION_SCHEMA
        or document.get("status") != "explicit-gqa-group-selections"
        or supplied != canonical_sha256(body)
        or not isinstance(rows, list)
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "GQA group-selection document seal or schema is invalid."
        )
    identifiers = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Lfm2FusedAttentionPlanProducerError(
                f"GQA group selection {index} is malformed."
            )
        identifiers.append(
            _canonical_token(row.get("selection_id"), label="selection_id")
        )
    if len(set(identifiers)) != len(identifiers):
        raise Lfm2FusedAttentionPlanProducerError(
            "GQA group selection identifiers are duplicated."
        )
    return document


def _schedule_document(layer_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not layer_rows:
        raise Lfm2FusedAttentionPlanProducerError(
            "Fused attention schedule requires at least one attention layer."
        )
    rows = sorted(layer_rows, key=lambda value: int(value["layer_index"]))
    candidates: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    seen: dict[tuple[tuple[int, str], ...], str] = {}

    def add(
        *,
        candidate_id: str,
        mode: str,
        variant_id: str,
        installations: Sequence[tuple[Mapping[str, Any], str]],
    ) -> None:
        signature = tuple(
            sorted(
                (
                    int(layer["layer_index"]),
                    str(layer["candidates"][variant]["candidate_state_sha256"]),
                )
                for layer, variant in installations
            )
        )
        if signature in seen:
            omitted.append(
                {
                    "omitted_candidate_id": candidate_id,
                    "retained_candidate_id": seen[signature],
                    "reason": "byte-identical-installed-attention-state-set",
                }
            )
            return
        seen[signature] = candidate_id
        candidates.append(
            {
                "candidate_id": candidate_id,
                "mode": mode,
                "variant_id": variant_id,
                "installations": [
                    {
                        "layer_index": int(layer["layer_index"]),
                        "variant_id": variant,
                        "candidate_state_sha256": layer["candidates"][variant][
                            "candidate_state_sha256"
                        ],
                    }
                    for layer, variant in installations
                ],
            }
        )

    # Isolated variants always lead the schedule.
    for layer in rows:
        for variant in layer["base_variant_ids"]:
            add(
                candidate_id=f"isolated--layer-{int(layer['layer_index']):03d}--{variant}",
                mode="isolated-layer-variant",
                variant_id=variant,
                installations=((layer, variant),),
            )
        for variant in layer["explicit_group_variant_ids"]:
            add(
                candidate_id=f"explicit-group--layer-{int(layer['layer_index']):03d}--{variant}",
                mode="explicit-gqa-group-isolated",
                variant_id=variant,
                installations=((layer, variant),),
            )

    common_variants = tuple(rows[0]["base_variant_ids"])
    if any(tuple(row["base_variant_ids"]) != common_variants for row in rows):
        raise Lfm2FusedAttentionPlanProducerError(
            "Attention layers do not expose one canonical base variant order."
        )
    for variant in common_variants:
        add(
            candidate_id=f"all-attention--{variant}",
            mode="all-attention-same-variant",
            variant_id=variant,
            installations=tuple((layer, variant) for layer in rows),
        )
    for variant in common_variants:
        for prefix_length in range(2, len(rows)):
            add(
                candidate_id=f"attention-prefix-{prefix_length:03d}--{variant}",
                mode="ordered-attention-prefix",
                variant_id=variant,
                installations=tuple((layer, variant) for layer in rows[:prefix_length]),
            )
    body = {
        "schema": SCHEDULE_SCHEMA,
        "status": "NON-FINAL-FUSED-ATTENTION-DIAGNOSTIC-SCHEDULE",
        "candidate_ids": [value["candidate_id"] for value in candidates],
        "candidates": candidates,
        "candidate_count": len(candidates),
        "mode_counts": {
            mode: sum(value["mode"] == mode for value in candidates)
            for mode in (
                "isolated-layer-variant",
                "explicit-gqa-group-isolated",
                "all-attention-same-variant",
                "ordered-attention-prefix",
            )
        },
        "deduplicated_candidate_aliases": omitted,
        "deduplication_key": "ordered-layer-index-and-candidate-state-sha256",
        "isolated_layer_variants_scheduled_first": True,
        "group_substitutions_default_cartesian_expansion_performed": False,
    }
    return {**body, "schedule_sha256": canonical_sha256(body)}


def _attention_component_rows(
    plan: Mapping[str, Any],
) -> Mapping[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in plan["components"]:
        if row.get("kind") != "attention":
            continue
        path = row.get("parent_module_path")
        if not isinstance(path, str) or not path.startswith("model.layers."):
            raise Lfm2FusedAttentionPlanProducerError(
                "Component plan attention path is not canonical."
            )
        parts = path.split(".")
        if len(parts) != 4 or parts[3] != "self_attn" or not parts[2].isdigit():
            raise Lfm2FusedAttentionPlanProducerError(
                "Component plan attention path is not canonical."
            )
        layer_index = int(parts[2])
        if layer_index in result:
            raise Lfm2FusedAttentionPlanProducerError(
                "Component plan duplicates an attention layer."
            )
        result[layer_index] = row
    if not result:
        raise Lfm2FusedAttentionPlanProducerError(
            "Component plan contains no authenticated attention component."
        )
    return result


def prepare(
    *,
    output_directory: Path,
    source_run_directory: Path,
    expected_source_run_result_sha256: str,
    expected_source_run_file_sha256: str,
    parent_checkpoint: Path,
    expected_parent_checkpoint_inventory_sha256: str,
    final_checkpoint: Path,
    expected_final_checkpoint_inventory_sha256: str,
    component_producer_receipt: Path,
    expected_component_producer_receipt_sha256: str,
    expected_component_producer_receipt_file_sha256: str,
    component_plan: Path,
    expected_component_plan_sha256: str,
    expected_component_plan_file_sha256: str,
    component_bridges: Path,
    expected_component_bridges_file_sha256: str,
    expected_seed_archive_manifest_sha256: str,
    expected_structural_archive_manifest_sha256: str,
    group_substitutions_json: Path | None = None,
    expected_group_substitutions_file_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Authenticate all local sources and atomically publish the fused bank."""

    import torch
    from safetensors.torch import save_file

    target = _new_output(output_directory)
    source_root = component_producer._ordinary_directory(
        source_run_directory,
        label="source run directory",
    )
    parent_path = component_producer._ordinary_directory(
        parent_checkpoint,
        label="parent checkpoint",
    )
    final_path = component_producer._ordinary_directory(
        final_checkpoint,
        label="final S2 checkpoint",
    )
    component_receipt_path = component_producer._ordinary_file(
        component_producer_receipt,
        label="component producer receipt",
    )
    component_plan_path = component_producer._ordinary_file(
        component_plan,
        label="component plan",
    )
    component_bridges_path = component_producer._ordinary_file(
        component_bridges,
        label="component bridges",
    )
    protected_sources = {
        source_root,
        parent_path,
        final_path,
        component_receipt_path.parent,
        component_plan_path.parent,
        component_bridges_path.parent,
    }
    _require_output_outside_sources(
        target,
        source_directories=tuple(protected_sources),
    )
    selection_document = _load_group_selections(
        group_substitutions_json,
        expected_file_sha256=expected_group_substitutions_file_sha256,
    )

    component_receipt = component_runner._load_producer_receipt(
        component_receipt_path,
        expected_receipt_sha256=expected_component_producer_receipt_sha256,
        expected_file_sha256=expected_component_producer_receipt_file_sha256,
    )
    component_plan_document = component_runner._load_plan(
        component_plan_path,
        expected_plan_sha256=expected_component_plan_sha256,
        expected_file_sha256=expected_component_plan_file_sha256,
    )
    component_runner._authenticate_producer_publication(
        producer_receipt=component_receipt,
        producer_receipt_path=component_receipt_path,
        plan=component_plan_document,
        component_plan=component_plan_path,
        bridge_tensors=component_bridges_path,
        expected_component_plan_sha256=expected_component_plan_sha256,
        expected_component_plan_file_sha256=expected_component_plan_file_sha256,
        expected_bridge_tensors_file_sha256=expected_component_bridges_file_sha256,
    )
    bridge_tensors = component_runner._load_bridge_tensors(
        component_bridges_path,
        expected_file_sha256=expected_component_bridges_file_sha256,
        plan=component_plan_document,
    )
    attention_components = _attention_component_rows(component_plan_document)

    source_run = component_producer._load_source_run(
        source_root,
        expected_result_sha256=expected_source_run_result_sha256,
        expected_file_sha256=expected_source_run_file_sha256,
    )
    parent_inventory = component_producer._checkpoint_inventory(parent_path)
    final_inventory = component_producer._checkpoint_inventory(final_path)
    parent_inventory_sha = canonical_sha256(parent_inventory)
    final_inventory_sha = canonical_sha256(final_inventory)
    parent_tokenizer_sha = fingerprint_local_tokenizer(parent_path)
    if parent_inventory_sha != _digest(
        expected_parent_checkpoint_inventory_sha256,
        label="expected parent checkpoint inventory",
    ) or final_inventory_sha != _digest(
        expected_final_checkpoint_inventory_sha256,
        label="expected final checkpoint inventory",
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "Parent or final checkpoint inventory differs from its external pin."
        )
    if (
        source_run["result_sha256"] != component_receipt["source_run_result_sha256"]
        or parent_inventory_sha
        != component_receipt["parent_checkpoint_inventory_sha256"]
        or final_inventory_sha
        != component_receipt["compact_checkpoint_inventory_sha256"]
        or final_inventory != source_run["candidate_checkpoint_inventory"]
        or final_inventory_sha != source_run["candidate_checkpoint_inventory_sha256"]
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "Component publication, source run, and checkpoint ancestry diverged."
        )

    dtype = component_producer._torch_dtype(source_run.get("child_parameter_dtype"))
    parent_identity_model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    parent_identity_sha = fingerprint_lfm2_parent(parent_identity_model).identity_sha256
    if (
        parent_identity_sha != source_run["parent_identity_sha256"]
        or _unique_parameter_elements(parent_identity_model)
        != source_run["parent_unique_parameters"]
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "Parent checkpoint identity/count differs from authenticated provenance."
        )
    del parent_identity_model
    parent_model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    final_model = load_lfm2_direct_local_model(
        final_path,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    parent_state_sha = fingerprint_lfm2_direct_local_state(parent_model)
    final_state_sha = fingerprint_lfm2_direct_local_state(final_model)
    final_complete_state_sha = fingerprint_model_state(final_model)
    if (
        parent_state_sha != component_plan_document["parent_model_state_sha256"]
        or final_state_sha != component_plan_document["compact_model_state_sha256"]
        or final_complete_state_sha != source_run["complete_child_model_state_sha256"]
        or _unique_parameter_elements(parent_model)
        != source_run["parent_unique_parameters"]
        or _unique_parameter_elements(final_model)
        != source_run["target_unique_parameters"]
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "Direct-local parent or final model differs from authenticated provenance."
        )

    seed_manifest = _digest(
        expected_seed_archive_manifest_sha256,
        label="expected seed archive manifest",
    )
    structural_manifest = _digest(
        expected_structural_archive_manifest_sha256,
        label="expected structural archive manifest",
    )
    if (
        seed_manifest != source_run["postfold_reclosure_seed_archive_manifest_sha256"]
        or structural_manifest
        != source_run["structural_attention_candidate_archive"]["manifest_sha256"]
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "Source-run archive pins differ from independently supplied pins."
        )
    seeds, seed_authentication = load_lfm2_postfold_reclosure_seed_archive(
        source_root / "postfold-reclosure-seeds",
        foundation_interface_solve_receipt=source_run["foundation_interface_solve"],
        semantic_layer_stack_receipt=source_run["semantic_layer_stack"],
        deferred_final_w2_installation_receipt=source_run["deferred_final_w2_install"],
        expected_manifest_sha256=seed_manifest,
    )
    structural_candidates, structural_authentication = (
        load_lfm2_structural_attention_candidate_archive(
            source_root / "structural-attention-candidates",
            semantic_layer_stack_receipt=source_run["semantic_layer_stack"],
            expected_manifest_sha256=structural_manifest,
        )
    )
    if (
        dict(seed_authentication) != dict(source_run["postfold_reclosure_seed_archive"])
        or dict(structural_authentication)
        != dict(source_run["structural_attention_candidate_archive"])
        or set(structural_candidates) != set(attention_components)
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "Fresh seed/structural archive authentication differs from source lineage."
        )
    encoder = seeds.producer_encoder.detach().cpu().contiguous()
    component_encoder = bridge_tensors.get("foundation.producer_encoder")
    if component_encoder is None or lfm2_authenticated_tensor_sha256(
        component_encoder
    ) != lfm2_authenticated_tensor_sha256(encoder):
        raise Lfm2FusedAttentionPlanProducerError(
            "Component bridge archive does not contain the authenticated seed encoder."
        )

    parent_config = parent_model.config
    final_config = final_model.config
    if (
        len(parent_model.model.layers) != len(final_model.model.layers)
        or list(parent_config.layer_types) != list(final_config.layer_types)
        or float(parent_config.norm_eps) != float(final_config.norm_eps)
        or getattr(parent_config, "_attn_implementation", None)
        != getattr(final_config, "_attn_implementation", None)
    ):
        raise Lfm2FusedAttentionPlanProducerError(
            "Parent/final LFM2 topology is incompatible with role-local attention fusion."
        )

    raw_selections = (
        [] if selection_document is None else selection_document["selections"]
    )
    selections_by_layer: dict[int, list[Mapping[str, Any]]] = {}
    for row in raw_selections:
        layer_index = row.get("layer_index")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index not in attention_components
        ):
            raise Lfm2FusedAttentionPlanProducerError(
                "Explicit GQA selection names an unauthenticated attention layer."
            )
        selections_by_layer.setdefault(layer_index, []).append(row)

    bank_tensors: dict[str, Any] = {}
    layer_rows: list[dict[str, Any]] = []
    for layer_index in sorted(attention_components):
        component = attention_components[layer_index]
        structural = structural_candidates[layer_index]
        parent_attention = parent_model.model.layers[layer_index].self_attn
        final_attention = final_model.model.layers[layer_index].self_attn
        decoder_name = component.get("output_decoder_tensor")
        decoder = bridge_tensors.get(decoder_name)
        if (
            not isinstance(decoder_name, str)
            or decoder is None
            or lfm2_authenticated_tensor_sha256(decoder)
            != lfm2_authenticated_tensor_sha256(structural.normalized_input_decoder)
            or _module_state_sha256(parent_attention)
            != component["parent_module_sha256"]
            or _module_state_sha256(final_attention)
            != component["compact_module_sha256"]
        ):
            raise Lfm2FusedAttentionPlanProducerError(
                f"Layer {layer_index} component bridge/module lineage diverged."
            )
        source = Lfm2FusedAttentionDiagnosticInput(
            layer_index=layer_index,
            source_query_head_count=int(parent_config.num_attention_heads),
            source_key_value_head_count=int(parent_config.num_key_value_heads),
            target_query_head_count=int(final_config.num_attention_heads),
            target_key_value_head_count=int(final_config.num_key_value_heads),
            head_dim=int(parent_attention.head_dim),
            selected_source_key_value_groups=(
                structural.selected_source_key_value_groups
            ),
            selected_source_query_heads=structural.selected_source_query_heads,
            norm_eps=float(parent_config.norm_eps),
            attention_implementation=str(parent_config._attn_implementation),
            parent_query_weight=parent_attention.q_proj.weight,
            parent_key_weight=parent_attention.k_proj.weight,
            parent_value_weight=parent_attention.v_proj.weight,
            parent_out_weight=parent_attention.out_proj.weight,
            parent_query_norm_weight=parent_attention.q_layernorm.weight,
            parent_key_norm_weight=parent_attention.k_layernorm.weight,
            compact_pristine_query_weight=structural.query_weight,
            compact_pristine_key_weight=structural.key_weight,
            compact_pristine_value_weight=structural.value_weight,
            compact_pristine_out_weight=structural.out_weight,
            compact_final_query_weight=final_attention.q_proj.weight,
            compact_final_key_weight=final_attention.k_proj.weight,
            compact_final_value_weight=final_attention.v_proj.weight,
            compact_final_out_weight=final_attention.out_proj.weight,
            compact_query_norm_weight=final_attention.q_layernorm.weight,
            compact_key_norm_weight=final_attention.k_layernorm.weight,
            input_encoder=encoder,
            output_decoder=decoder,
            context_reconstruction=structural.context_reconstruction,
        )
        input_authentication = fingerprint_lfm2_fused_attention_diagnostic_input(source)
        pins = {
            name: record["sha256"]
            for name, record in input_authentication["tensor_records"].items()
        }
        bank = compile_lfm2_fused_attention_diagnostic_bank(
            source,
            expected_input_bundle_sha256=input_authentication["input_bundle_sha256"],
            expected_tensor_sha256=pins,
        )
        candidates = dict(bank.candidates)
        base_variant_ids = tuple(candidates)
        explicit_variant_ids: list[str] = []
        for selection in selections_by_layer.get(layer_index, []):
            base_id = selection.get("base_variant_id")
            donor_id = selection.get("donor_variant_id")
            role_bundle = selection.get("role_bundle")
            group_index = selection.get("target_group_index")
            if base_id not in candidates or donor_id not in candidates:
                raise Lfm2FusedAttentionPlanProducerError(
                    "Explicit GQA selection names an unavailable base/donor variant."
                )
            selection_id = _canonical_token(
                selection.get("selection_id"),
                label="selection_id",
            )
            variant_id = f"explicit-gqa-{selection_id}"
            candidate = substitute_lfm2_fused_attention_gqa_group(
                base=candidates[base_id],
                donor=candidates[donor_id],
                target_group_index=group_index,
                role_bundle=role_bundle,
                variant_id=variant_id,
            )
            candidates[variant_id] = candidate
            explicit_variant_ids.append(variant_id)

        variant_rows: dict[str, Any] = {}
        canonical_by_state: dict[str, str] = {}
        tensor_names_by_state: dict[str, dict[str, str]] = {}
        for variant_id, candidate in candidates.items():
            authenticated = authenticate_lfm2_fused_attention_candidate(candidate)
            state_sha = authenticated["candidate_state_sha256"]
            state = candidate.state_dict()
            if state_sha not in canonical_by_state:
                canonical_by_state[state_sha] = variant_id
                prefix = f"layer.{layer_index:03d}.state.{state_sha}"
                tensor_names = {}
                for state_name, tensor in state.items():
                    tensor_name = f"{prefix}.{state_name}"
                    bank_tensors[tensor_name] = tensor.detach().cpu().contiguous()
                    tensor_names[state_name] = tensor_name
                tensor_names_by_state[state_sha] = tensor_names
            variant_rows[variant_id] = {
                "variant_id": variant_id,
                "candidate_class": (
                    "explicit-gqa-group"
                    if variant_id in explicit_variant_ids
                    else "base-role-localized"
                ),
                "candidate_state_sha256": state_sha,
                "candidate_result_sha256": authenticated["result_sha256"],
                "reconstruction_descriptor_sha256": (
                    candidate.reconstruction_descriptor["descriptor_sha256"]
                ),
                "canonical_stored_variant_id": canonical_by_state[state_sha],
                "byte_identical_candidate_deduplicated": (
                    canonical_by_state[state_sha] != variant_id
                ),
                "tensor_names": tensor_names_by_state[state_sha],
                "receipt": _plain(candidate.receipt),
                "reconstruction_descriptor": _plain(
                    candidate.reconstruction_descriptor
                ),
            }
        layer_rows.append(
            {
                "layer_index": layer_index,
                "parent_module_path": f"model.layers.{layer_index}.self_attn",
                "parent_attention_module_sha256": _module_state_sha256(
                    parent_attention
                ),
                "final_attention_module_sha256": _module_state_sha256(final_attention),
                "input_authentication": _plain(input_authentication),
                "bank_receipt": _plain(bank.receipt),
                "base_variant_ids": list(base_variant_ids),
                "explicit_group_variant_ids": explicit_variant_ids,
                "variant_ids": list(candidates),
                "candidates": variant_rows,
                "unique_stored_candidate_state_count": len(canonical_by_state),
                "candidate_variant_count": len(candidates),
            }
        )

    schedule = _schedule_document(layer_rows)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        bank_path = stage / BANK_FILENAME
        save_file(
            bank_tensors,
            str(bank_path),
            metadata={
                "format": "pt",
                "neural_forge_role": "fused-attention-diagnostic-bank",
            },
        )
        bank_records = {
            name: _tensor_record(tensor)
            for name, tensor in sorted(bank_tensors.items())
        }
        plan_body = {
            "schema": PLAN_SCHEMA,
            "status": PLAN_STATUS,
            "producer_schema": PRODUCER_SCHEMA,
            "source_authentication": {
                "component_producer_receipt_sha256": component_receipt[
                    "receipt_sha256"
                ],
                "component_producer_receipt_file_sha256": _file_sha256(
                    component_runner._ordinary_file(
                        component_producer_receipt,
                        label="component producer receipt",
                    )
                ),
                "component_plan_sha256": component_plan_document["plan_sha256"],
                "component_plan_file_sha256": _file_sha256(
                    component_runner._ordinary_file(
                        component_plan,
                        label="component plan",
                    )
                ),
                "component_bridges_file_sha256": _file_sha256(
                    component_runner._ordinary_file(
                        component_bridges,
                        label="component bridges",
                    )
                ),
                "source_run_result_sha256": source_run["result_sha256"],
                "source_run_receipt_file_sha256": _file_sha256(
                    source_root / "run-receipt.json"
                ),
                "parent_checkpoint_inventory_sha256": parent_inventory_sha,
                "final_checkpoint_inventory_sha256": final_inventory_sha,
                "parent_lfm2_identity_sha256": parent_identity_sha,
                "parent_tokenizer_sha256": parent_tokenizer_sha,
                "parent_model_state_sha256": parent_state_sha,
                "final_model_state_sha256": final_state_sha,
                "final_complete_model_state_sha256": final_complete_state_sha,
                "seed_archive_manifest_sha256": seed_manifest,
                "structural_archive_manifest_sha256": structural_manifest,
                "group_selection_file_sha256": (
                    None
                    if group_substitutions_json is None
                    else _file_sha256(
                        component_producer._ordinary_file(
                            group_substitutions_json,
                            label="GQA group selections",
                        )
                    )
                ),
                "group_selection_sha256": (
                    None
                    if selection_document is None
                    else selection_document["selection_sha256"]
                ),
                "all_source_tensor_pins_embedded_per_layer": True,
                "all_source_artifacts_freshly_authenticated": True,
            },
            "parent_checkpoint_inventory_sha256": parent_inventory_sha,
            "final_checkpoint_inventory_sha256": final_inventory_sha,
            "parent_tokenizer_sha256": parent_tokenizer_sha,
            "parent_model_state_sha256": parent_state_sha,
            "final_model_state_sha256": final_state_sha,
            "materialization_dtype": str(dtype).removeprefix("torch."),
            "bank": {
                "filename": BANK_FILENAME,
                "file_sha256": _file_sha256(bank_path),
                "tensor_records": bank_records,
                "tensor_count": len(bank_records),
                "stored_tensor_elements": sum(
                    int(value.numel()) for value in bank_tensors.values()
                ),
            },
            "attention_layers": layer_rows,
            "attention_layer_indices": [
                int(value["layer_index"]) for value in layer_rows
            ],
            "attention_layer_count": len(layer_rows),
            "schedule": _plain(schedule),
            "explicit_group_selection_document": _plain(selection_document),
            "group_substitutions_included_only_when_explicitly_selected": True,
            "group_substitution_cartesian_expansion_performed": False,
            "byte_identical_candidate_states_stored_once": True,
            "diagnostic_parent_shell_only": True,
            "standalone_child_compression_claimed": False,
            "numeric_acceptance_gate_defined": False,
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
            "bank_filename": BANK_FILENAME,
            "bank_file_sha256": _file_sha256(bank_path),
            "component_producer_receipt_sha256": component_receipt["receipt_sha256"],
            "source_run_result_sha256": source_run["result_sha256"],
            "parent_checkpoint_inventory_sha256": parent_inventory_sha,
            "final_checkpoint_inventory_sha256": final_inventory_sha,
            "parent_tokenizer_sha256": parent_tokenizer_sha,
            "attention_layer_count": len(layer_rows),
            "candidate_schedule_count": schedule["candidate_count"],
            "unique_stored_candidate_state_count": sum(
                int(value["unique_stored_candidate_state_count"])
                for value in layer_rows
            ),
            "fresh_source_receipt_authentication_performed": True,
            "fresh_archive_authentication_performed": True,
            "direct_local_checkpoint_loading_used": True,
            "human_authored_module_or_tensor_pins_accepted": False,
            "training_steps": 0,
            "model_forward_calls": 0,
        }
        receipt = {
            **receipt_body,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
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
    parser.add_argument("--source-run-directory", type=Path, required=True)
    parser.add_argument("--expected-source-run-result-sha256", required=True)
    parser.add_argument("--expected-source-run-file-sha256", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-final-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--component-producer-receipt", type=Path, required=True)
    parser.add_argument("--expected-component-producer-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-component-producer-receipt-file-sha256", required=True
    )
    parser.add_argument("--component-plan", type=Path, required=True)
    parser.add_argument("--expected-component-plan-sha256", required=True)
    parser.add_argument("--expected-component-plan-file-sha256", required=True)
    parser.add_argument("--component-bridges", type=Path, required=True)
    parser.add_argument("--expected-component-bridges-file-sha256", required=True)
    parser.add_argument("--expected-seed-archive-manifest-sha256", required=True)
    parser.add_argument("--expected-structural-archive-manifest-sha256", required=True)
    parser.add_argument("--group-substitutions-json", type=Path)
    parser.add_argument("--expected-group-substitutions-file-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = prepare(**vars(arguments))
    print(json.dumps(_plain(receipt), allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
