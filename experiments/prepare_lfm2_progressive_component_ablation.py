#!/usr/bin/env python3
"""Derive a sealed component-ablation plan from one exact4x composite run.

The producer performs no folding.  It freshly authenticates a current-code
exact4x composite run, both local checkpoints, the seed archive, the structural
attention candidate archive, and every component fold receipt.  It then emits
one atomic three-file publication accepted by the progressive ablation runner:

* ``component-bridges.safetensors`` containing the authenticated foundation
  producer encoder, per-attention normalized-input decoders, and—where a
  component has no archived decoder—a sealed full-column-rank Moore-Penrose
  decoder derived solely from that authenticated encoder; and
* ``component-plan.json`` containing topology-derived module paths and actual
  module/checkpoint/fold-receipt digests; and
* ``producer-receipt.json`` binding both artifacts and their authenticated
  source-run/checkpoint provenance behind an independently pinnable seal.

No module hashes, bridge pins, or fold receipt digests are accepted from a
human-authored plan.  This first experiment replaces whole LFM2 subsystems.  It
does not claim separate Q/K/V/O or individual-head isolation.
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

from experiments import run_lfm2_progressive_component_ablation as runner  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_postfold_reclosure_seed_archive import (  # noqa: E402
    load_lfm2_postfold_reclosure_seed_archive,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _module_state_sha256,
    _parameter_alias_topology,
    _unique_parameter_elements,
)
from neural_forge.hf_lfm2_structural_attention_candidate_archive import (  # noqa: E402
    load_lfm2_structural_attention_candidate_archive,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


SOURCE_RUN_SCHEMA = "neural-forge-lfm2-exact4x-composite-fold-run-v1"
SOURCE_BODY_SCHEMA = "neural-forge-lfm2-exact4x-structural-body-checkpoint-v1"
PLAN_PRODUCER_SCHEMA = "neural-forge-lfm2-component-ablation-plan-producer-v1"
ANALYTIC_DECODER_SCHEMA = "neural-forge-lfm2-encoder-pseudoinverse-decoder-v1"
BRIDGE_FILENAME = "component-bridges.safetensors"
PLAN_FILENAME = "component-plan.json"
PRODUCER_RECEIPT_FILENAME = "producer-receipt.json"


class Lfm2ComponentAblationPlanProducerError(RuntimeError):
    """An exact4x source, archive, checkpoint, or derived plan is invalid."""


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ComponentAblationPlanProducerError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2ComponentAblationPlanProducerError(
            f"{label} is missing or unsafe."
        )
    return path


def _ordinary_file(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ComponentAblationPlanProducerError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Lfm2ComponentAblationPlanProducerError(
            f"{label} is missing or unsafe."
        )
    return path


def _new_output(value: Path) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ComponentAblationPlanProducerError(
            "output directory must be local."
        )
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2ComponentAblationPlanProducerError(
            "output directory must not already exist."
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2ComponentAblationPlanProducerError(
            "output parent must be one ordinary directory."
        )
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
        raise Lfm2ComponentAblationPlanProducerError(
            "Checkpoint inventory cannot be empty."
        )
    return rows


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    ordinary = _ordinary_file(path, label=label)
    try:
        value = json.loads(ordinary.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2ComponentAblationPlanProducerError(
            f"{label} is not valid JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2ComponentAblationPlanProducerError(
            f"{label} must be one JSON object."
        )
    return value


def _sealed_receipt(
    value: Any,
    *,
    label: str,
    seal_field: str = "result_sha256",
    schema: str | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2ComponentAblationPlanProducerError(
            f"{label} must be one sealed receipt."
        )
    body = dict(value)
    supplied = body.pop(seal_field, None)
    if (
        supplied != canonical_sha256(body)
        or (schema is not None and value.get("schema") != schema)
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            f"{label} seal or schema is invalid."
        )
    return value


def _same_disk_receipt(
    source_root: Path,
    filename: str,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if _load_json(source_root / filename, label=label) != dict(expected):
        raise Lfm2ComponentAblationPlanProducerError(
            f"{label} differs between source-run disk and sealed run receipt."
        )


def _load_source_run(
    source_root: Path,
    *,
    expected_result_sha256: str,
    expected_file_sha256: str,
) -> dict[str, Any]:
    path = _ordinary_file(source_root / "run-receipt.json", label="source run receipt")
    if _file_sha256(path) != _digest(
        expected_file_sha256,
        label="expected source-run receipt file",
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Source-run receipt file differs from its external pin."
        )
    run = _load_json(path, label="source run receipt")
    _sealed_receipt(run, label="source run receipt", schema=SOURCE_RUN_SCHEMA)
    if run.get("status") != "standalone-exact4x-composite-child-decoded":
        raise Lfm2ComponentAblationPlanProducerError(
            "Source run did not reach the successful decoded exact4x status."
        )
    if run.get("result_sha256") != _digest(
        expected_result_sha256,
        label="expected source-run result",
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Source-run result differs from its external pin."
        )
    required = (
        "foundation_interface_solve",
        "semantic_layer_stack",
        "deferred_final_w2_install",
        "structural_body_checkpoint_receipt",
        "structural_attention_candidate_archive",
        "postfold_reclosure_seed_archive",
        "causal_output_install",
        "candidate_checkpoint_inventory",
        "candidate_checkpoint_inventory_sha256",
        "complete_child_model_state_sha256",
    )
    if any(name not in run for name in required):
        raise Lfm2ComponentAblationPlanProducerError(
            "Source run predates the current exact4x archive/checkpoint contract."
        )
    foundation = _sealed_receipt(
        run["foundation_interface_solve"],
        label="foundation interface solve",
    )
    stack = _sealed_receipt(run["semantic_layer_stack"], label="semantic layer stack")
    deferred = _sealed_receipt(
        run["deferred_final_w2_install"],
        label="deferred final-W2 installation",
    )
    structural_body = _sealed_receipt(
        run["structural_body_checkpoint_receipt"],
        label="structural body receipt",
        seal_field="receipt_sha256",
        schema=SOURCE_BODY_SCHEMA,
    )
    if (
        structural_body.get("status")
        != "standalone-structural-body-saved-and-freshly-reloaded"
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Structural body receipt did not reach its successful reload status."
        )
    causal_output = _sealed_receipt(
        run["causal_output_install"],
        label="causal output installation",
    )
    _same_disk_receipt(
        source_root,
        "semantic-layer-stack.json",
        stack,
        label="semantic layer-stack receipt",
    )
    _same_disk_receipt(
        source_root,
        "deferred-final-w2-install.json",
        deferred,
        label="deferred final-W2 receipt",
    )
    _same_disk_receipt(
        source_root,
        "structural-body-receipt.json",
        structural_body,
        label="structural body receipt",
    )
    if (
        deferred.get("semantic_layer_stack_result_sha256")
        != stack.get("result_sha256")
        or structural_body.get("semantic_layer_stack_sha256")
        != stack.get("result_sha256")
        or structural_body.get("deferred_final_w2_install_sha256")
        != deferred.get("result_sha256")
        or run.get("structural_body_model_state_sha256")
        != structural_body.get("body_model_state_sha256")
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Source-run foundation/body/deferred receipt ancestry diverged."
        )
    # Keep references alive for archive authentication and make their validation
    # explicit to static/audit readers.
    if foundation.get("result_sha256") is None or causal_output.get("result_sha256") is None:
        raise Lfm2ComponentAblationPlanProducerError(
            "Source-run foundation or output receipt lost its seal."
        )
    return run


def _torch_dtype(name: Any) -> Any:
    import torch

    if name not in {"float32", "bfloat16"}:
        raise Lfm2ComponentAblationPlanProducerError(
            "Exact4x child declares an unsupported materialization dtype."
        )
    return getattr(torch, str(name))


def _analytic_decoder(encoder: Any) -> tuple[Any, Mapping[str, Any]]:
    import torch

    matrix = encoder.detach().to(device="cpu", dtype=torch.float64).contiguous()
    parent_hidden, child_hidden = map(int, matrix.shape)
    rank = int(torch.linalg.matrix_rank(matrix).item())
    if parent_hidden < child_hidden or rank != child_hidden:
        raise Lfm2ComponentAblationPlanProducerError(
            "Foundation producer encoder lacks full child-column rank; an analytic "
            "child-to-parent decoder is not geometrically valid."
        )
    decoder = torch.linalg.pinv(matrix, rcond=1e-12).contiguous()
    identity_error = float(
        torch.linalg.matrix_norm(
            decoder @ matrix - torch.eye(child_hidden, dtype=torch.float64)
        ).item()
    )
    if (
        tuple(decoder.shape) != (child_hidden, parent_hidden)
        or not bool(torch.isfinite(decoder).all().item())
        or not identity_error <= 1e-8
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Authenticated encoder pseudoinverse failed its child-coordinate identity."
        )
    body = {
        "schema": ANALYTIC_DECODER_SCHEMA,
        "method": "torch.linalg.pinv-fp64-cpu",
        "rcond": 1e-12,
        "source_encoder_sha256": lfm2_authenticated_tensor_sha256(matrix),
        "source_encoder_shape": [parent_hidden, child_hidden],
        "source_encoder_rank": rank,
        "decoder_sha256": lfm2_authenticated_tensor_sha256(decoder),
        "decoder_shape": [child_hidden, parent_hidden],
        "decoder_times_encoder_identity_frobenius_error": identity_error,
        "derived_only_from_authenticated_encoder": True,
        "training_steps": 0,
        "model_forward_calls": 0,
    }
    return decoder, {**body, "receipt_sha256": canonical_sha256(body)}


def _component_fold_receipt(
    layer: Mapping[str, Any],
    *,
    component_kind: str,
    layer_index: int,
) -> tuple[Mapping[str, Any], str]:
    if layer.get("layer_index") != layer_index:
        raise Lfm2ComponentAblationPlanProducerError(
            "Semantic layer receipt order/index is inconsistent."
        )
    _sealed_receipt(layer, label=f"layer {layer_index} transaction")
    field = (
        "mlp_complete_candidate_frontier_receipt"
        if component_kind == "mlp"
        else "operator_complete_candidate_frontier_receipt"
    )
    receipt = _sealed_receipt(
        layer.get(field),
        label=f"layer {layer_index} {component_kind} fold receipt",
    )
    expected = layer.get(
        "mlp_installation_sha256"
        if component_kind == "mlp"
        else "operator_installation_sha256"
    )
    if receipt.get("result_sha256") != expected:
        raise Lfm2ComponentAblationPlanProducerError(
            f"Layer {layer_index} {component_kind} fold receipt is not the installed decision."
        )
    return receipt, field


def _parameter_reduction(
    *,
    kind: str,
    parent_module: Any,
    compact_module: Any,
    parent_hidden: int,
    child_hidden: int,
    output_domain_bridge_elements: int = 0,
) -> dict[str, int]:
    original = _unique_parameter_elements(parent_module)
    compact = _unique_parameter_elements(compact_module)
    bridge = parent_hidden * child_hidden
    if kind != "terminal_readout":
        bridge *= 2
    if output_domain_bridge_elements:
        bridge += output_domain_bridge_elements
    installed = compact + bridge
    return {
        "original_component_parameter_elements": original,
        "compact_operator_parameter_elements": compact,
        "counted_bridge_parameter_elements": bridge,
        "installed_component_parameter_elements": installed,
        "component_parameter_reduction": original - installed,
    }


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def prepare(
    *,
    output_directory: Path,
    source_run_directory: Path,
    expected_source_run_result_sha256: str,
    expected_source_run_file_sha256: str,
    parent_checkpoint: Path,
    expected_parent_checkpoint_inventory_sha256: str,
    compact_checkpoint: Path,
    expected_compact_checkpoint_inventory_sha256: str,
    expected_seed_archive_manifest_sha256: str,
    expected_candidate_archive_manifest_sha256: str,
) -> Mapping[str, Any]:
    import torch
    from safetensors.torch import save_file

    target = _new_output(output_directory)
    source_root = _ordinary_directory(source_run_directory, label="source run directory")
    parent_path = _ordinary_directory(parent_checkpoint, label="parent checkpoint")
    compact_path = _ordinary_directory(compact_checkpoint, label="compact checkpoint")
    run = _load_source_run(
        source_root,
        expected_result_sha256=expected_source_run_result_sha256,
        expected_file_sha256=expected_source_run_file_sha256,
    )
    parent_inventory = _checkpoint_inventory(parent_path)
    compact_inventory = _checkpoint_inventory(compact_path)
    parent_inventory_sha = canonical_sha256(parent_inventory)
    compact_inventory_sha = canonical_sha256(compact_inventory)
    if parent_inventory_sha != _digest(
        expected_parent_checkpoint_inventory_sha256,
        label="expected parent checkpoint inventory",
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Parent checkpoint inventory differs from its external pin."
        )
    if compact_inventory_sha != _digest(
        expected_compact_checkpoint_inventory_sha256,
        label="expected compact checkpoint inventory",
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Compact checkpoint inventory differs from its external pin."
        )
    if (
        compact_inventory != run.get("candidate_checkpoint_inventory")
        or compact_inventory_sha != run.get("candidate_checkpoint_inventory_sha256")
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Compact checkpoint is not the final candidate bound to this exact4x run."
        )
    structural_body = run["structural_body_checkpoint_receipt"]
    structural_checkpoint = _ordinary_directory(
        source_root / "structural-body-checkpoint",
        label="source structural-body checkpoint",
    )
    structural_inventory = _checkpoint_inventory(structural_checkpoint)
    if (
        structural_inventory != structural_body.get("checkpoint_inventory")
        or canonical_sha256(structural_inventory)
        != structural_body.get("checkpoint_inventory_sha256")
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Source structural-body checkpoint inventory differs from its receipt."
        )

    child_dtype = _torch_dtype(run.get("child_parameter_dtype"))
    parent_identity_model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    if (
        fingerprint_lfm2_parent(parent_identity_model).identity_sha256
        != run.get("parent_identity_sha256")
        or _unique_parameter_elements(parent_identity_model)
        != run.get("parent_unique_parameters")
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Parent checkpoint identity/count differs from the exact4x source run."
        )
    parent_identity_sha = fingerprint_lfm2_parent(parent_identity_model).identity_sha256
    del parent_identity_model
    parent_model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=child_dtype,
    )
    compact_model = load_lfm2_direct_local_model(
        compact_path,
        device=torch.device("cpu"),
        dtype=child_dtype,
    )
    if (
        fingerprint_model_state(compact_model)
        != run.get("complete_child_model_state_sha256")
        or _unique_parameter_elements(compact_model) != run.get("target_unique_parameters")
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Compact checkpoint state/count differs from the exact4x source run."
        )
    realized_ratio = _unique_parameter_elements(parent_model) / _unique_parameter_elements(
        compact_model
    )
    if not 3.99 <= realized_ratio <= 4.01:
        raise Lfm2ComponentAblationPlanProducerError(
            "Source run/checkpoints do not realize the current exact4x parameter ratio."
        )

    foundation = run["foundation_interface_solve"]
    stack = run["semantic_layer_stack"]
    deferred = run["deferred_final_w2_install"]
    seed_manifest_expected = _digest(
        expected_seed_archive_manifest_sha256,
        label="expected seed archive manifest",
    )
    candidate_manifest_expected = _digest(
        expected_candidate_archive_manifest_sha256,
        label="expected candidate archive manifest",
    )
    if (
        seed_manifest_expected
        != run.get("postfold_reclosure_seed_archive_manifest_sha256")
        or seed_manifest_expected
        != structural_body.get("postfold_reclosure_seed_archive_manifest_sha256")
        or candidate_manifest_expected
        != run["structural_attention_candidate_archive"].get("manifest_sha256")
        or candidate_manifest_expected
        != structural_body.get(
            "structural_attention_candidate_archive_manifest_sha256"
        )
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Archive manifest pins differ across source-run receipts."
        )
    seeds, seed_authentication = load_lfm2_postfold_reclosure_seed_archive(
        source_root / "postfold-reclosure-seeds",
        foundation_interface_solve_receipt=foundation,
        semantic_layer_stack_receipt=stack,
        deferred_final_w2_installation_receipt=deferred,
        expected_manifest_sha256=seed_manifest_expected,
    )
    attention_candidates, candidate_authentication = (
        load_lfm2_structural_attention_candidate_archive(
            source_root / "structural-attention-candidates",
            semantic_layer_stack_receipt=stack,
            expected_manifest_sha256=candidate_manifest_expected,
        )
    )
    if (
        dict(seed_authentication) != dict(run["postfold_reclosure_seed_archive"])
        or dict(candidate_authentication)
        != dict(run["structural_attention_candidate_archive"])
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "Fresh archive authentication differs from the source-run receipt."
        )

    encoder = seeds.producer_encoder.detach().to(device="cpu").contiguous()
    parent_hidden = int(parent_model.config.hidden_size)
    child_hidden = int(compact_model.config.hidden_size)
    if tuple(encoder.shape) != (parent_hidden, child_hidden):
        raise Lfm2ComponentAblationPlanProducerError(
            "Foundation producer encoder does not bridge live parent/child widths."
        )
    analytic_decoder, analytic_decoder_receipt = _analytic_decoder(encoder)
    parent_layers = tuple(parent_model.model.layers)
    compact_layers = tuple(compact_model.model.layers)
    layer_rows = stack.get("layer_installations")
    if (
        not isinstance(layer_rows, list)
        or len(parent_layers) != len(compact_layers)
        or len(layer_rows) != len(compact_layers)
        or list(parent_model.config.layer_types) != list(compact_model.config.layer_types)
        or list(stack.get("operator_layer_kinds", ()))
        != [
            "full_attention" if value == "full_attention" else "short_conv"
            for value in compact_model.config.layer_types
        ]
    ):
        raise Lfm2ComponentAblationPlanProducerError(
            "First component experiment requires one-to-one, same-kind LFM2 layer topology."
        )

    bridge_tensors: dict[str, Any] = {
        "foundation.producer_encoder": encoder,
        "foundation.analytic_output_decoder": analytic_decoder,
    }
    components = []
    source_fold_receipts: dict[str, Any] = {}
    exclusions = []

    def add_component(
        *,
        component_id: str,
        kind: str,
        parent_path: str,
        compact_path: str,
        output_tensor_name: str | None,
        output_domain_tensor_name: str | None,
        fold_receipt: Mapping[str, Any],
        fold_locator: str,
    ) -> None:
        parent_module = parent_model.get_submodule(parent_path)
        compact_module = compact_model.get_submodule(compact_path)
        accounting = _parameter_reduction(
            kind=kind,
            parent_module=parent_module,
            compact_module=compact_module,
            parent_hidden=parent_hidden,
            child_hidden=child_hidden,
            output_domain_bridge_elements=(
                0
                if output_domain_tensor_name is None
                else int(bridge_tensors[output_domain_tensor_name].numel())
            ),
        )
        if accounting["component_parameter_reduction"] <= 0:
            exclusions.append(
                {
                    "component_id": component_id,
                    "reason": "compact-operator-plus-counted-bridges-is-not-smaller",
                    "physical_parameter_accounting": accounting,
                }
            )
            return
        fold_sha = _digest(
            fold_receipt.get("result_sha256"),
            label=f"{component_id} verified source fold receipt",
        )
        input_sha = lfm2_authenticated_tensor_sha256(encoder)
        output_sha = (
            None
            if output_tensor_name is None
            else lfm2_authenticated_tensor_sha256(bridge_tensors[output_tensor_name])
        )
        domain_sha = (
            None
            if output_domain_tensor_name is None
            else lfm2_authenticated_tensor_sha256(
                bridge_tensors[output_domain_tensor_name]
            )
        )
        source_fold_receipts[component_id] = {
            "receipt_sha256": fold_sha,
            "receipt_schema": fold_receipt.get("schema"),
            "source_run_locator": fold_locator,
            "receipt_seal_freshly_verified": True,
        }
        components.append(
            {
                "component_id": component_id,
                "kind": kind,
                "granularity": "whole-subsystem",
                "parent_module_path": parent_path,
                "compact_module_path": compact_path,
                "parent_module_sha256": _module_state_sha256(parent_module),
                "compact_module_sha256": _module_state_sha256(compact_module),
                "input_encoder_tensor": "foundation.producer_encoder",
                "output_decoder_tensor": output_tensor_name,
                "output_domain_decoder_tensor": output_domain_tensor_name,
                "input_encoder_sha256": input_sha,
                "output_decoder_sha256": output_sha,
                "output_domain_decoder_sha256": domain_sha,
                "source_fold_receipt_sha256": fold_sha,
                "source_compact_checkpoint_sha256": compact_inventory_sha,
                "physical_parameter_accounting": accounting,
                "require_parameter_reduction": True,
            }
        )

    for layer_index, (layer, layer_receipt, layer_type) in enumerate(
        zip(compact_layers, layer_rows, compact_model.config.layer_types, strict=True)
    ):
        operator_kind = "attention" if layer_type == "full_attention" else "short_conv"
        operator_attr = "self_attn" if operator_kind == "attention" else "conv"
        operator_receipt, operator_receipt_field = _component_fold_receipt(
            layer_receipt,
            component_kind=operator_kind,
            layer_index=layer_index,
        )
        if operator_kind == "attention":
            if layer_index not in attention_candidates:
                raise Lfm2ComponentAblationPlanProducerError(
                    f"Attention layer {layer_index} lacks its authenticated candidate decoder."
                )
            decoder_name = f"attention.layer.{layer_index:03d}.normalized_input_decoder"
            decoder = attention_candidates[layer_index].normalized_input_decoder.detach().cpu().contiguous()
            if tuple(decoder.shape) != (child_hidden, parent_hidden):
                raise Lfm2ComponentAblationPlanProducerError(
                    f"Attention layer {layer_index} decoder has stale geometry."
                )
            bridge_tensors[decoder_name] = decoder
        else:
            decoder_name = "foundation.analytic_output_decoder"
        add_component(
            component_id=f"layer-{layer_index:03d}-{operator_kind}",
            kind=operator_kind,
            parent_path=f"model.layers.{layer_index}.{operator_attr}",
            compact_path=f"model.layers.{layer_index}.{operator_attr}",
            output_tensor_name=decoder_name,
            output_domain_tensor_name=None,
            fold_receipt=operator_receipt,
            fold_locator=(
                f"semantic_layer_stack.layer_installations[{layer_index}]."
                f"{operator_receipt_field}"
            ),
        )
        mlp_receipt, mlp_receipt_field = _component_fold_receipt(
            layer_receipt,
            component_kind="mlp",
            layer_index=layer_index,
        )
        add_component(
            component_id=f"layer-{layer_index:03d}-mlp",
            kind="mlp",
            parent_path=f"model.layers.{layer_index}.feed_forward",
            compact_path=f"model.layers.{layer_index}.feed_forward",
            output_tensor_name="foundation.analytic_output_decoder",
            output_domain_tensor_name=None,
            fold_receipt=mlp_receipt,
            fold_locator=(
                f"semantic_layer_stack.layer_installations[{layer_index}]."
                f"{mlp_receipt_field}"
            ),
        )

    parent_readout = parent_model.get_output_embeddings()
    compact_readout = compact_model.get_output_embeddings()
    parent_alias_groups = _parameter_alias_topology(parent_model)
    parent_readout_is_native_tied_alias = any(
        {"model.embed_tokens.weight", "lm_head.weight"}.issubset(
            {member[0] for member in members}
        )
        for _, _, members in parent_alias_groups
    )
    if parent_readout_is_native_tied_alias:
        exclusions.append(
            {
                "component_id": "terminal-readout",
                "reason": "replacement-would-break-native-parent-input-output-alias",
                "native_parent_aliases_retained": True,
            }
        )
    elif int(parent_readout.weight.shape[0]) == int(compact_readout.weight.shape[0]):
        causal_output = _sealed_receipt(
            run["causal_output_install"],
            label="terminal causal-output fold receipt",
        )
        add_component(
            component_id="terminal-readout",
            kind="terminal_readout",
            parent_path="lm_head",
            compact_path="lm_head",
            output_tensor_name=None,
            output_domain_tensor_name=None,
            fold_receipt=causal_output,
            fold_locator="causal_output_install",
        )
    else:
        exclusions.append(
            {
                "component_id": "terminal-readout",
                "reason": "cross-vocabulary-output-domain-bridge-not-present-in-authenticated-inputs",
                "parent_vocabulary_width": int(parent_readout.weight.shape[0]),
                "compact_vocabulary_width": int(compact_readout.weight.shape[0]),
            }
        )
    if not components:
        raise Lfm2ComponentAblationPlanProducerError(
            "No topology-derived component is physically smaller after counted bridges."
        )

    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        bridge_path = stage / BRIDGE_FILENAME
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in bridge_tensors.items()},
            str(bridge_path),
            metadata={"format": "pt", "neural_forge_role": "component-ablation-bridges"},
        )
        bridge_file_sha = _file_sha256(bridge_path)
        bridge_records = {
            name: {
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype).removeprefix("torch."),
                "sha256": lfm2_authenticated_tensor_sha256(value),
            }
            for name, value in sorted(bridge_tensors.items())
        }
        plan_body = {
            "schema": runner.PLAN_SCHEMA,
            "status": "NON-FINAL-COMPONENT-ABLATION-PLAN-AUTOMATICALLY-DERIVED",
            "producer_schema": PLAN_PRODUCER_SCHEMA,
            "source_authentication": {
                "source_run_result_sha256": run["result_sha256"],
                "source_run_receipt_file_sha256": _file_sha256(
                    source_root / "run-receipt.json"
                ),
                "parent_checkpoint_inventory_sha256": parent_inventory_sha,
                "parent_lfm2_identity_sha256": parent_identity_sha,
                "compact_checkpoint_inventory_sha256": compact_inventory_sha,
                "compact_complete_model_state_sha256": run[
                    "complete_child_model_state_sha256"
                ],
                "structural_body_checkpoint_inventory_sha256": structural_body[
                    "checkpoint_inventory_sha256"
                ],
                "seed_archive_manifest_sha256": seed_manifest_expected,
                "candidate_archive_manifest_sha256": candidate_manifest_expected,
                "seed_archive_fresh_reload_verified": True,
                "candidate_archive_fresh_reload_verified": True,
                "component_fold_receipts": source_fold_receipts,
            },
            "parent_model_state_sha256": fingerprint_lfm2_direct_local_state(
                parent_model
            ),
            "compact_model_state_sha256": fingerprint_lfm2_direct_local_state(
                compact_model
            ),
            "parent_checkpoint_inventory_sha256": parent_inventory_sha,
            "compact_checkpoint_inventory_sha256": compact_inventory_sha,
            "materialization_dtype": str(child_dtype).removeprefix("torch."),
            "realized_source_parameter_ratio": realized_ratio,
            "bridge_archive": {
                "schema": runner.BRIDGE_ARCHIVE_SCHEMA,
                "filename": BRIDGE_FILENAME,
                "file_sha256": bridge_file_sha,
                "tensor_records": bridge_records,
                "producer_encoder_seed_archive_manifest_sha256": seed_manifest_expected,
                "attention_decoder_candidate_archive_manifest_sha256": (
                    candidate_manifest_expected
                ),
                "analytic_decoder_derivation": dict(analytic_decoder_receipt),
            },
            "component_granularity": "whole-subsystem",
            "q_k_v_o_or_individual_head_isolation_claimed": False,
            "components": components,
            "excluded_components": exclusions,
            "all_included_components_require_literal_parameter_reduction_true": True,
            "diagnostic_hybrid_only": True,
            "standalone_compression_claimed": False,
            "operation_audit": {
                "training_steps": 0,
                "optimizer_constructions": 0,
                "optimizer_steps": 0,
                "autograd_grad_calls": 0,
                "backward_calls": 0,
                "parent_model_forward_calls": 0,
                "compact_model_forward_calls": 0,
                "parent_trace_recaptures": 0,
                "network_access": False,
            },
        }
        plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
        _json_write(stage / PLAN_FILENAME, plan)
        producer_body = {
            "schema": PLAN_PRODUCER_SCHEMA,
            "status": "component-ablation-bridge-and-plan-publication-complete",
            "plan_filename": PLAN_FILENAME,
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": _file_sha256(stage / PLAN_FILENAME),
            "bridge_filename": BRIDGE_FILENAME,
            "bridge_file_sha256": bridge_file_sha,
            "source_run_result_sha256": run["result_sha256"],
            "parent_checkpoint_inventory_sha256": parent_inventory_sha,
            "compact_checkpoint_inventory_sha256": compact_inventory_sha,
            "component_count": len(components),
            "excluded_component_count": len(exclusions),
            "component_granularity": "whole-subsystem",
            "automatic_topology_and_component_derivation_performed": True,
            "fresh_source_receipt_authentication_performed": True,
            "fresh_archive_authentication_performed": True,
            "human_authored_module_or_receipt_pins_accepted": False,
            "training_steps": 0,
            "model_forward_calls": 0,
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
    return producer_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-run-directory", type=Path, required=True)
    parser.add_argument("--expected-source-run-result-sha256", required=True)
    parser.add_argument("--expected-source-run-file-sha256", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--compact-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-compact-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--expected-seed-archive-manifest-sha256", required=True)
    parser.add_argument("--expected-candidate-archive-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(
            output_directory=args.output_directory,
            source_run_directory=args.source_run_directory,
            expected_source_run_result_sha256=(args.expected_source_run_result_sha256),
            expected_source_run_file_sha256=args.expected_source_run_file_sha256,
            parent_checkpoint=args.parent_checkpoint,
            expected_parent_checkpoint_inventory_sha256=(
                args.expected_parent_checkpoint_inventory_sha256
            ),
            compact_checkpoint=args.compact_checkpoint,
            expected_compact_checkpoint_inventory_sha256=(
                args.expected_compact_checkpoint_inventory_sha256
            ),
            expected_seed_archive_manifest_sha256=(
                args.expected_seed_archive_manifest_sha256
            ),
            expected_candidate_archive_manifest_sha256=(
                args.expected_candidate_archive_manifest_sha256
            ),
        )
    except Lfm2ComponentAblationPlanProducerError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(dict(result), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYTIC_DECODER_SCHEMA",
    "BRIDGE_FILENAME",
    "Lfm2ComponentAblationPlanProducerError",
    "PLAN_FILENAME",
    "PLAN_PRODUCER_SCHEMA",
    "PRODUCER_RECEIPT_FILENAME",
    "SOURCE_RUN_SCHEMA",
    "main",
    "prepare",
]
