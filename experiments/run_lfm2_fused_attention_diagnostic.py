#!/usr/bin/env python3
"""Freshly reconstruct, persist, reload, and decode fused LFM2 attention seams.

Every scheduled candidate begins from a fresh direct-local parent shell.  Its
selected attention layers are replaced by ordinary parent-I/O reduced-head
``Lfm2Attention`` modules reconstructed from the authenticated producer bank.
The mixed-topology differential checkpoint is saved and custom-fresh-reloaded
before the complete cached prompt bank is decoded.  Cached parent responses
are attached for manual intelligence review; no numeric score accepts or
rejects a candidate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
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

from experiments.prepare_lfm2_fused_attention_diagnostic import (  # noqa: E402
    BANK_FILENAME,
    PLAN_FILENAME,
    PLAN_SCHEMA,
    PLAN_STATUS,
    PRODUCER_RECEIPT_FILENAME,
    PRODUCER_SCHEMA,
    PRODUCER_STATUS,
    SCHEDULE_SCHEMA,
)
from experiments import run_lfm2_residual_attention_fold as residual_runner  # noqa: E402
from experiments.run_lfm2_dynamic_catalog import (  # noqa: E402
    fingerprint_local_tokenizer,
)
from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _generate_rows,
    _load_parent_envelope,
    load_review_prompts,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_fused_attention_diagnostic import (  # noqa: E402
    Lfm2FusedAttentionDiagnosticCandidate,
    authenticate_lfm2_fused_attention_candidate,
    reconstruct_lfm2_fused_attention_candidate,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _model_state_sha256,
    _parameter_alias_topology,
    _stored_tensor_elements,
    _unique_parameter_bytes,
    _unique_parameter_elements,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


RUN_SCHEMA = "neural-forge-lfm2-fused-attention-diagnostic-run-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-fused-attention-diagnostic-progress-v1"
CHECKPOINT_SCHEMA = "neural-forge-lfm2-fused-attention-mixed-checkpoint-v1"
MANUAL_REVIEW_SCHEMA = "neural-forge-lfm2-fused-attention-manual-review-v1"
SHARD_SCHEMA = "neural-forge-lfm2-fused-attention-schedule-shard-v1"
PARENT_BINDING_SCHEMA = "neural-forge-lfm2-fused-attention-parent-envelope-binding-v1"
CHECKPOINT_TENSOR_FILENAME = "attention-overrides.safetensors"
CHECKPOINT_MANIFEST_FILENAME = "manifest.json"


class Lfm2FusedAttentionDiagnosticRunError(RuntimeError):
    """A producer artifact, reconstruction, reload, or review invariant failed."""


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
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


def _ordinary_file(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2FusedAttentionDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Lfm2FusedAttentionDiagnosticRunError(f"{label} is missing or unsafe.")
    return path


def _ordinary_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2FusedAttentionDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2FusedAttentionDiagnosticRunError(f"{label} is missing or unsafe.")
    return path


def _new_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2FusedAttentionDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2FusedAttentionDiagnosticRunError(f"{label} must not already exist.")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2FusedAttentionDiagnosticRunError(
            f"{label} parent must be one ordinary directory."
        )
    path.mkdir()
    return path


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    if path.is_symlink() or (path.exists() and not replace):
        raise Lfm2FusedAttentionDiagnosticRunError(
            f"Refusing to replace evidence file {path.name!r}."
        )
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise Lfm2FusedAttentionDiagnosticRunError(
            "A stale evidence temporary file already exists."
        )
    temporary.write_text(
        json.dumps(_plain(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    if replace and path.exists():
        path.unlink()
    temporary.replace(path)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    ordinary = _ordinary_file(path, label=label)
    try:
        value = json.loads(ordinary.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2FusedAttentionDiagnosticRunError(
            f"{label} is not canonical JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2FusedAttentionDiagnosticRunError(f"{label} must be one JSON object.")
    return value


def _checkpoint_inventory(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Checkpoint inventory contains a symlink."
            )
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _file_sha256(path),
                }
            )
    if not rows:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Checkpoint inventory cannot be empty."
        )
    return rows


def _artifact_inventory(
    directory: Path,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    return [
        row
        for row in _checkpoint_inventory(directory)
        if row["relative_path"] not in exclude
    ]


def _dtype(value: str) -> Any:
    import torch

    if value not in {"float32", "bfloat16"}:
        raise Lfm2FusedAttentionDiagnosticRunError("dtype must be float32 or bfloat16.")
    return getattr(torch, value)


def _schedule_shard(
    schedule: Mapping[str, Any],
    *,
    shard_count: int,
    shard_index: int,
) -> Mapping[str, Any]:
    candidates = schedule.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Schedule sharding requires the complete nonempty candidate order."
        )
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or shard_count > len(candidates)
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Shard count/index do not define one nonempty balanced schedule shard."
        )
    base, remainder = divmod(len(candidates), shard_count)
    start = shard_index * base + min(shard_index, remainder)
    length = base + int(shard_index < remainder)
    stop = start + length
    selected = candidates[start:stop]
    body = {
        "schema": SHARD_SCHEMA,
        "status": "authenticated-balanced-contiguous-schedule-shard",
        "full_schedule_sha256": schedule["schedule_sha256"],
        "full_candidate_count": len(candidates),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "start_index_inclusive": start,
        "stop_index_exclusive": stop,
        "selected_candidate_ids": [value["candidate_id"] for value in selected],
        "selected_candidate_count": len(selected),
        "balanced_contiguous_partition": True,
        "complete_full_schedule_sweep_claimed": shard_count == 1,
    }
    return {**body, "shard_sha256": canonical_sha256(body)}


def _authenticate_schedule_shard(
    schedule: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(shard, Mapping):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Schedule shard receipt must be one mapping."
        )
    expected = _schedule_shard(
        schedule,
        shard_count=shard.get("shard_count"),
        shard_index=shard.get("shard_index"),
    )
    if dict(shard) != dict(expected):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Schedule shard receipt differs from the deterministic balanced partition."
        )
    start = int(shard["start_index_inclusive"])
    stop = int(shard["stop_index_exclusive"])
    return tuple(schedule["candidates"][start:stop])


def _load_publication(
    path: Path,
    *,
    expected_receipt_sha256: str,
    expected_file_sha256: str,
) -> Mapping[str, Any]:
    ordinary = _ordinary_file(path, label="fused producer receipt")
    if ordinary.name != PRODUCER_RECEIPT_FILENAME or _file_sha256(ordinary) != _digest(
        expected_file_sha256,
        label="expected fused producer-receipt file",
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused producer-receipt filename or file pin is invalid."
        )
    value = _load_json(ordinary, label="fused producer receipt")
    body = dict(value)
    supplied = body.pop("receipt_sha256", None)
    if (
        value.get("schema") != PRODUCER_SCHEMA
        or value.get("status") != PRODUCER_STATUS
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_receipt_sha256, label="expected fused producer receipt")
        or value.get("plan_filename") != PLAN_FILENAME
        or value.get("bank_filename") != BANK_FILENAME
        or value.get("fresh_source_receipt_authentication_performed") is not True
        or value.get("fresh_archive_authentication_performed") is not True
        or value.get("direct_local_checkpoint_loading_used") is not True
        or value.get("human_authored_module_or_tensor_pins_accepted") is not False
        or value.get("training_steps") != 0
        or value.get("model_forward_calls") != 0
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused producer receipt seal or policy is invalid."
        )
    return value


def _load_plan(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_file_sha256: str,
) -> Mapping[str, Any]:
    ordinary = _ordinary_file(path, label="fused attention plan")
    if ordinary.name != PLAN_FILENAME or _file_sha256(ordinary) != _digest(
        expected_file_sha256,
        label="expected fused plan file",
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused attention plan filename or file pin is invalid."
        )
    value = _load_json(ordinary, label="fused attention plan")
    body = dict(value)
    supplied = body.pop("plan_sha256", None)
    schedule = value.get("schedule")
    schedule_body = dict(schedule) if isinstance(schedule, Mapping) else {}
    schedule_seal = schedule_body.pop("schedule_sha256", None)
    if (
        value.get("schema") != PLAN_SCHEMA
        or value.get("status") != PLAN_STATUS
        or value.get("producer_schema") != PRODUCER_SCHEMA
        or supplied != canonical_sha256(body)
        or supplied != _digest(expected_plan_sha256, label="expected fused plan")
        or not isinstance(value.get("attention_layers"), list)
        or not value["attention_layers"]
        or not isinstance(value.get("bank"), Mapping)
        or value["bank"].get("filename") != BANK_FILENAME
        or not isinstance(schedule, Mapping)
        or schedule.get("schema") != SCHEDULE_SCHEMA
        or schedule_seal != canonical_sha256(schedule_body)
        or schedule.get("candidate_ids")
        != [row.get("candidate_id") for row in schedule.get("candidates", ())]
        or value.get("group_substitutions_included_only_when_explicitly_selected")
        is not True
        or value.get("group_substitution_cartesian_expansion_performed") is not False
        or value.get("byte_identical_candidate_states_stored_once") is not True
        or value.get("diagnostic_parent_shell_only") is not True
        or value.get("standalone_child_compression_claimed") is not False
        or value.get("numeric_acceptance_gate_defined") is not False
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused attention plan seal, schema, schedule, or policy is invalid."
        )
    source_authentication = value.get("source_authentication")
    if not isinstance(source_authentication, Mapping) or _digest(
        value.get("parent_tokenizer_sha256"),
        label="planned parent tokenizer",
    ) != _digest(
        source_authentication.get("parent_tokenizer_sha256"),
        label="source-authenticated parent tokenizer",
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused attention plan does not bind one parent tokenizer identity."
        )
    _digest(
        source_authentication.get("parent_lfm2_identity_sha256"),
        label="source-authenticated parent identity",
    )
    return value


def _authenticate_publication(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    bank_path: Path,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
    expected_bank_file_sha256: str,
) -> None:
    receipt_file = _ordinary_file(receipt_path, label="fused producer receipt")
    plan_file = _ordinary_file(plan_path, label="fused plan")
    bank_file = _ordinary_file(bank_path, label="fused bank")
    if (
        receipt_file.parent != plan_file.parent
        or receipt_file.parent != bank_file.parent
        or receipt.get("plan_sha256") != expected_plan_sha256
        or plan.get("plan_sha256") != expected_plan_sha256
        or receipt.get("plan_file_sha256") != expected_plan_file_sha256
        or _file_sha256(plan_file) != expected_plan_file_sha256
        or receipt.get("bank_file_sha256") != expected_bank_file_sha256
        or plan.get("bank", {}).get("file_sha256") != expected_bank_file_sha256
        or _file_sha256(bank_file) != expected_bank_file_sha256
        or receipt.get("parent_checkpoint_inventory_sha256")
        != plan.get("parent_checkpoint_inventory_sha256")
        or receipt.get("final_checkpoint_inventory_sha256")
        != plan.get("final_checkpoint_inventory_sha256")
        or receipt.get("parent_tokenizer_sha256") != plan.get("parent_tokenizer_sha256")
        or receipt.get("attention_layer_count") != plan.get("attention_layer_count")
        or receipt.get("candidate_schedule_count")
        != plan.get("schedule", {}).get("candidate_count")
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused bank/plan files are not the pinned producer publication."
        )


def _load_tensor_archive(
    path: Path,
    *,
    expected_file_sha256: str,
    records: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused attention archives require safetensors."
        ) from exc
    ordinary = _ordinary_file(path, label="fused attention tensor archive")
    if _file_sha256(ordinary) != _digest(
        expected_file_sha256,
        label="expected fused attention tensor archive",
    ) or not isinstance(records, Mapping):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused attention tensor archive differs from its pin."
        )
    result = {}
    with safe_open(str(ordinary), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(records):
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Fused attention tensor inventory is incomplete."
            )
        for name in sorted(records):
            tensor = handle.get_tensor(name).detach().cpu().contiguous().clone()
            record = {
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sha256": lfm2_authenticated_tensor_sha256(tensor),
            }
            if record != records[name]:
                raise Lfm2FusedAttentionDiagnosticRunError(
                    f"Fused attention tensor {name!r} differs from its record."
                )
            result[name] = tensor
    return MappingProxyType(result)


def _layer_map(plan: Mapping[str, Any]) -> Mapping[int, Mapping[str, Any]]:
    result = {}
    for row in plan["attention_layers"]:
        if not isinstance(row, Mapping):
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Fused attention layer plan row is malformed."
            )
        index = row.get("layer_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in result
        ):
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Fused attention layer indices are invalid or duplicated."
            )
        result[index] = row
    return MappingProxyType(result)


def _candidate_from_tensors(
    row: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> Lfm2FusedAttentionDiagnosticCandidate:
    names = row.get("tensor_names")
    receipt = row.get("receipt")
    descriptor = row.get("reconstruction_descriptor")
    expected_keys = {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
    }
    if (
        not isinstance(names, Mapping)
        or set(names) != expected_keys
        or not isinstance(receipt, Mapping)
        or not isinstance(descriptor, Mapping)
        or any(name not in tensors for name in names.values())
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Candidate row lacks its complete authenticated tensor mapping."
        )
    candidate = Lfm2FusedAttentionDiagnosticCandidate(
        variant_id=str(row["variant_id"]),
        query_weight=tensors[names["q_proj.weight"]],
        key_weight=tensors[names["k_proj.weight"]],
        value_weight=tensors[names["v_proj.weight"]],
        out_weight=tensors[names["out_proj.weight"]],
        query_norm_weight=tensors[names["q_layernorm.weight"]],
        key_norm_weight=tensors[names["k_layernorm.weight"]],
        receipt=MappingProxyType(dict(receipt)),
        reconstruction_descriptor=MappingProxyType(dict(descriptor)),
    )
    authenticated = authenticate_lfm2_fused_attention_candidate(candidate)
    if (
        authenticated["result_sha256"] != row.get("candidate_result_sha256")
        or authenticated["candidate_state_sha256"] != row.get("candidate_state_sha256")
        or candidate.reconstruction_descriptor["descriptor_sha256"]
        != row.get("reconstruction_descriptor_sha256")
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Candidate tensor state differs from its producer plan."
        )
    return candidate


def _install_schedule_candidate(
    *,
    model: Any,
    plan: Mapping[str, Any],
    schedule_candidate: Mapping[str, Any],
    tensors: Mapping[str, Any],
    device: Any,
) -> tuple[Mapping[str, Any], ...]:
    layers = _layer_map(plan)
    before_model_elements = _unique_parameter_elements(model)
    expected_reduction = 0
    receipts = []
    installed_layers: set[int] = set()
    for raw in schedule_candidate.get("installations", ()):
        if not isinstance(raw, Mapping):
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Schedule installation row is malformed."
            )
        layer_index = raw.get("layer_index")
        variant_id = raw.get("variant_id")
        if layer_index not in layers or layer_index in installed_layers:
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Schedule installation layer is absent or duplicated."
            )
        layer = layers[layer_index]
        candidate_row = layer.get("candidates", {}).get(variant_id)
        if not isinstance(candidate_row, Mapping):
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Schedule names an unavailable fused attention variant."
            )
        if candidate_row.get("candidate_state_sha256") != raw.get(
            "candidate_state_sha256"
        ):
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Schedule candidate-state pin differs from the layer plan."
            )
        old_module = model.model.layers[layer_index].self_attn
        if _module_state_digest(old_module) != layer["parent_attention_module_sha256"]:
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Fresh parent attention differs from the producer plan."
            )
        candidate = _candidate_from_tensors(candidate_row, tensors)
        reconstruction = reconstruct_lfm2_fused_attention_candidate(
            candidate=candidate,
            parent_config=model.config,
            expected_candidate_result_sha256=candidate_row["candidate_result_sha256"],
            expected_descriptor_sha256=candidate_row[
                "reconstruction_descriptor_sha256"
            ],
            device=device,
        )
        physical = candidate.receipt["physical_parameter_accounting"]
        expected_reduction += int(physical["physical_parameter_reduction"])
        model.model.layers[layer_index].self_attn = reconstruction.module
        installed_layers.add(layer_index)
        body = {
            "layer_index": layer_index,
            "variant_id": variant_id,
            "candidate_state_sha256": candidate_row["candidate_state_sha256"],
            "candidate_result_sha256": candidate_row["candidate_result_sha256"],
            "reconstruction_receipt": _plain(reconstruction.receipt),
            "physical_parameter_accounting": _plain(physical),
            "ordinary_lfm2_attention_installed": True,
            "input_bridge_module_present": False,
            "output_bridge_module_present": False,
            "training_steps": 0,
            "model_forward_calls": 0,
        }
        receipts.append({**body, "installation_sha256": canonical_sha256(body)})
    after_model_elements = _unique_parameter_elements(model)
    if (
        not receipts
        or before_model_elements - after_model_elements != expected_reduction
        or any(
            name in type(model.model.layers[index].self_attn).__name__.lower()
            for index, name in ((value, "bridge") for value in installed_layers)
        )
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fused attention installation did not realize its physical reduction."
        )
    return tuple(MappingProxyType(value) for value in receipts)


def _module_state_digest(module: Any) -> str:
    records = []
    for name, tensor in sorted(module.state_dict().items()):
        records.append(
            {
                "name": name,
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sha256": lfm2_authenticated_tensor_sha256(tensor),
            }
        )
    return canonical_sha256(records)


def _checkpoint_tensor_subset(
    *,
    plan: Mapping[str, Any],
    schedule_candidate: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> Mapping[str, Any]:
    layers = _layer_map(plan)
    names = set()
    for installation in schedule_candidate["installations"]:
        candidate = layers[installation["layer_index"]]["candidates"][
            installation["variant_id"]
        ]
        names.update(candidate["tensor_names"].values())
    return MappingProxyType(
        {name: tensors[name].detach().cpu().contiguous() for name in sorted(names)}
    )


def _write_mixed_checkpoint(
    *,
    destination: Path,
    model: Any,
    plan: Mapping[str, Any],
    producer_receipt: Mapping[str, Any],
    schedule_candidate: Mapping[str, Any],
    installation_receipts: Sequence[Mapping[str, Any]],
    bank_tensors: Mapping[str, Any],
) -> Mapping[str, Any]:
    from safetensors.torch import save_file

    target = destination.expanduser().absolute()
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Mixed checkpoint destination is not new and local."
        )
    subset = _checkpoint_tensor_subset(
        plan=plan,
        schedule_candidate=schedule_candidate,
        tensors=bank_tensors,
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        tensor_path = temporary / CHECKPOINT_TENSOR_FILENAME
        save_file(
            dict(subset),
            str(tensor_path),
            metadata={"format": "pt", "neural_forge_status": "NON-FINAL"},
        )
        tensor_records = {
            name: {
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sha256": lfm2_authenticated_tensor_sha256(tensor),
            }
            for name, tensor in subset.items()
        }
        body = {
            "schema": CHECKPOINT_SCHEMA,
            "status": "NON-FINAL-FUSED-ATTENTION-MIXED-CHECKPOINT-SAVED",
            "producer_receipt_sha256": producer_receipt["receipt_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "candidate": _plain(schedule_candidate),
            "candidate_id": schedule_candidate["candidate_id"],
            "installation_receipts": _plain(installation_receipts),
            "tensor_filename": CHECKPOINT_TENSOR_FILENAME,
            "tensor_file_sha256": _file_sha256(tensor_path),
            "tensor_records": tensor_records,
            "model_state_sha256": _model_state_sha256(model),
            "unique_parameter_elements": _unique_parameter_elements(model),
            "unique_parameter_bytes": _unique_parameter_bytes(model),
            "stored_tensor_elements": _stored_tensor_elements(model),
            "parameter_alias_topology": _plain(_parameter_alias_topology(model)),
            "artifact_inventory": _artifact_inventory(temporary),
            "ordinary_fused_attention_only": True,
            "bridge_sidecars_present": False,
            "fresh_reload_required_before_decode": True,
            "diagnostic_parent_shell_only": True,
            "standalone_child_compression_claimed": False,
            "operation_audit": {
                "training_steps": 0,
                "model_forward_calls": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "parameters_sparsified": 0,
                "network_access": False,
            },
        }
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        _write_json(temporary / CHECKPOINT_MANIFEST_FILENAME, manifest)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return MappingProxyType(manifest)


def _load_checkpoint_manifest(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    value = _load_json(
        directory / CHECKPOINT_MANIFEST_FILENAME,
        label="mixed checkpoint manifest",
    )
    body = dict(value)
    supplied = body.pop("manifest_sha256", None)
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("status") != "NON-FINAL-FUSED-ATTENTION-MIXED-CHECKPOINT-SAVED"
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_manifest_sha256, label="expected mixed manifest")
        or value.get("ordinary_fused_attention_only") is not True
        or value.get("bridge_sidecars_present") is not False
        or value.get("fresh_reload_required_before_decode") is not True
        or value.get("standalone_child_compression_claimed") is not False
        or value.get("artifact_inventory")
        != _artifact_inventory(
            directory,
            exclude=frozenset({CHECKPOINT_MANIFEST_FILENAME}),
        )
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Mixed checkpoint manifest seal, inventory, or policy is invalid."
        )
    return value


def _fresh_load_mixed_checkpoint(
    *,
    source: Path,
    expected_manifest_sha256: str,
    parent_checkpoint: Path,
    plan: Mapping[str, Any],
    producer_receipt: Mapping[str, Any],
    device: Any,
    dtype: Any,
) -> tuple[Any, Mapping[str, Any]]:
    directory = _ordinary_directory(source, label="mixed checkpoint")
    manifest = _load_checkpoint_manifest(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if (
        manifest.get("producer_receipt_sha256") != producer_receipt["receipt_sha256"]
        or manifest.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Mixed checkpoint ancestry differs from the producer plan."
        )
    tensors = _load_tensor_archive(
        directory / CHECKPOINT_TENSOR_FILENAME,
        expected_file_sha256=manifest["tensor_file_sha256"],
        records=manifest["tensor_records"],
    )
    model = load_lfm2_direct_local_model(
        parent_checkpoint,
        device=device,
        dtype=dtype,
    )
    if fingerprint_lfm2_direct_local_state(model) != plan["parent_model_state_sha256"]:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fresh parent shell differs before mixed-topology reconstruction."
        )
    receipts = _install_schedule_candidate(
        model=model,
        plan=plan,
        schedule_candidate=manifest["candidate"],
        tensors=tensors,
        device=device,
    )
    if (
        _plain(receipts) != manifest["installation_receipts"]
        or _model_state_sha256(model) != manifest["model_state_sha256"]
        or _unique_parameter_elements(model) != manifest["unique_parameter_elements"]
        or _unique_parameter_bytes(model) != manifest["unique_parameter_bytes"]
        or _stored_tensor_elements(model) != manifest["stored_tensor_elements"]
        or _plain(_parameter_alias_topology(model))
        != manifest["parameter_alias_topology"]
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Fresh custom reload differs from its sealed mixed-topology checkpoint."
        )
    model.requires_grad_(False)
    model.eval()
    return model, manifest


def _attach_parent(
    responses: list[dict[str, Any]],
    *,
    parent_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    for response in responses:
        parent = parent_rows[str(response["id"])]
        response["parent_response"] = parent["response"]
        response["parent_response_token_ids"] = parent["response_token_ids"]
        response["parent_diagnostics"] = parent["diagnostics"]


def _authenticate_parent_envelope_binding(
    *,
    parent_cache: Mapping[str, Any],
    parent_checkpoint: Path,
    plan: Mapping[str, Any],
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> Mapping[str, Any]:
    """Bind cached text to this exact local parent, tokenizer, and decode budget."""

    import torch

    source_authentication = plan.get("source_authentication")
    audit = parent_cache.get("execution_audit")
    responses = parent_cache.get("responses")
    inference_dtype = parent_cache.get("inference_dtype")
    if (
        not isinstance(source_authentication, Mapping)
        or not isinstance(audit, Mapping)
        or not isinstance(responses, list)
        or inference_dtype not in {"float16", "float32", "bfloat16"}
        or parent_cache.get("maximum_prompt_tokens") != maximum_prompt_tokens
        or parent_cache.get("maximum_new_tokens") != maximum_new_tokens
        or audit.get("parent_model_loads") != 1
        or audit.get("parent_generate_invocations") != len(responses)
        or audit.get("response_cache_publications") != 1
        or any(
            audit.get(name) != 0
            for name in (
                "training_steps",
                "optimizer_steps",
                "backward_calls",
                "autograd_grad_calls",
            )
        )
        or audit.get("hub_loader_used") is not False
        or audit.get("network_access") is not False
        or audit.get("parent_state_unchanged") is not True
        or audit.get("parent_parameter_count_unchanged") is not True
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Cached parent envelope limits or execution audit are incompatible with this run."
        )

    envelope_state = _digest(
        parent_cache.get("parent_state_sha256"),
        label="cached parent state",
    )
    envelope_identity = _digest(
        parent_cache.get("parent_identity_sha256"),
        label="cached parent identity",
    )
    envelope_tokenizer = _digest(
        parent_cache.get("tokenizer_sha256"),
        label="cached parent tokenizer",
    )
    planned_tokenizer = _digest(
        plan.get("parent_tokenizer_sha256"),
        label="planned parent tokenizer",
    )
    planned_bfloat16_identity = _digest(
        source_authentication.get("parent_lfm2_identity_sha256"),
        label="planned bfloat16 parent identity",
    )
    parent_unique_parameters = parent_cache.get("parent_unique_parameters")
    if (
        isinstance(parent_unique_parameters, bool)
        or not isinstance(parent_unique_parameters, int)
        or parent_unique_parameters <= 0
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Cached parent envelope lacks a valid physical parameter count."
        )

    local_tokenizer = fingerprint_local_tokenizer(parent_checkpoint)
    if envelope_tokenizer != planned_tokenizer or local_tokenizer != planned_tokenizer:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Cached, planned, and local parent tokenizer identities diverged."
        )

    reference = load_lfm2_direct_local_model(
        parent_checkpoint,
        device=torch.device("cpu"),
        dtype={
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[inference_dtype],
    )
    try:
        observed_state = residual_runner.model_state_sha256(reference)
        observed_identity = fingerprint_lfm2_parent(reference).identity_sha256
        observed_parameters = _unique_parameter_elements(reference)
    finally:
        del reference
    if (
        observed_state != envelope_state
        or observed_identity != envelope_identity
        or observed_parameters != parent_unique_parameters
        or (
            inference_dtype == "bfloat16"
            and observed_identity != planned_bfloat16_identity
        )
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Cached parent state, identity, or count differs from the pinned local parent."
        )

    body = {
        "schema": PARENT_BINDING_SCHEMA,
        "status": "cached-parent-envelope-freshly-bound-to-local-parent",
        "parent_envelope_sha256": parent_cache["envelope_sha256"],
        "parent_checkpoint_inventory_sha256": plan[
            "parent_checkpoint_inventory_sha256"
        ],
        "parent_state_sha256": observed_state,
        "parent_identity_sha256": observed_identity,
        "planned_bfloat16_parent_identity_sha256": planned_bfloat16_identity,
        "planned_bfloat16_identity_comparison_performed": (
            inference_dtype == "bfloat16"
        ),
        "planned_bfloat16_identity_match": (
            True if inference_dtype == "bfloat16" else None
        ),
        "parent_unique_parameters": observed_parameters,
        "parent_tokenizer_sha256": local_tokenizer,
        "inference_dtype": inference_dtype,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "execution_audit": _plain(audit),
        "local_parent_reauthentication_loads": 1,
        "local_parent_reauthentication_forward_calls": 0,
        "local_parent_reauthentication_generate_calls": 0,
        "network_access": False,
    }
    return MappingProxyType({**body, "binding_sha256": canonical_sha256(body)})


def _manual_review_document(
    *,
    candidate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    parent_binding: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    body = {
        "schema": MANUAL_REVIEW_SCHEMA,
        "status": "awaiting-manual-intelligence-review",
        "candidate_id": candidate["candidate_id"],
        "mode": candidate["mode"],
        "variant_id": candidate["variant_id"],
        "installations": _plain(candidate["installations"]),
        "mixed_checkpoint_manifest_sha256": manifest["manifest_sha256"],
        "parent_envelope_binding_sha256": parent_binding["binding_sha256"],
        "responses": _plain(responses),
        "response_count": len(responses),
        "review_fields": {
            "coherent": None,
            "answers_the_prompt": None,
            "intelligence_retention_relative_to_parent": None,
            "notes": None,
        },
        "decoded_whole_responses_are_primary_evidence": True,
        "exact_token_match_required": False,
        "numeric_metrics_have_decision_authority": False,
        "automatic_acceptance_or_rejection_performed": False,
    }
    return {**body, "review_sha256": canonical_sha256(body)}


def _progress(
    *,
    publication: Mapping[str, Any],
    plan: Mapping[str, Any],
    completed: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    shard: Mapping[str, Any],
) -> Mapping[str, Any]:
    body = {
        "schema": PROGRESS_SCHEMA,
        "status": "complete" if len(completed) == len(candidate_ids) else "running",
        "producer_receipt_sha256": publication["receipt_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": plan["schedule"]["schedule_sha256"],
        "schedule_shard": _plain(shard),
        "schedule_shard_sha256": shard["shard_sha256"],
        "candidate_ids": list(candidate_ids),
        "completed_candidate_ids": [value["candidate_id"] for value in completed],
        "remaining_candidate_ids": list(candidate_ids[len(completed) :]),
        "candidate_reviews": _plain(completed),
        "automatic_acceptance_or_rejection_performed": False,
    }
    return {**body, "progress_sha256": canonical_sha256(body)}


def run(
    *,
    run_directory: Path,
    parent_checkpoint: Path,
    producer_receipt: Path,
    expected_producer_receipt_sha256: str,
    expected_producer_receipt_file_sha256: str,
    fused_plan: Path,
    expected_fused_plan_sha256: str,
    expected_fused_plan_file_sha256: str,
    fused_bank: Path,
    expected_fused_bank_file_sha256: str,
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
    shard_count: int = 1,
    shard_index: int = 0,
) -> Mapping[str, Any]:
    """Run the complete sealed schedule and emit manual-review evidence."""

    import torch

    if checkpoint_retention not in {"receipts-only", "keep-last"}:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "checkpoint_retention must be receipts-only or keep-last."
        )
    if not 256 <= maximum_prompt_tokens <= 8192:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "maximum_prompt_tokens must be between 256 and 8192; 64-token shortcuts are forbidden."
        )
    if not 256 <= maximum_new_tokens <= 2048:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "maximum_new_tokens must be between 256 and 2048; 64-token shortcuts are forbidden."
        )
    run_root = _new_directory(run_directory, label="run directory")
    checkpoints_root = run_root / "mixed-checkpoints"
    reviews_root = run_root / "manual-review"
    checkpoints_root.mkdir()
    reviews_root.mkdir()
    parent_path = _ordinary_directory(parent_checkpoint, label="parent checkpoint")
    prompt_path = _ordinary_file(prompts_json, label="prompt bank")
    envelope_path = _ordinary_file(parent_envelope, label="cached parent envelope")
    if _file_sha256(prompt_path) != _digest(
        expected_prompts_file_sha256,
        label="expected prompt bank",
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Prompt bank differs from its external pin."
        )
    publication = _load_publication(
        producer_receipt,
        expected_receipt_sha256=expected_producer_receipt_sha256,
        expected_file_sha256=expected_producer_receipt_file_sha256,
    )
    plan = _load_plan(
        fused_plan,
        expected_plan_sha256=expected_fused_plan_sha256,
        expected_file_sha256=expected_fused_plan_file_sha256,
    )
    _authenticate_publication(
        receipt=publication,
        receipt_path=producer_receipt,
        plan=plan,
        plan_path=fused_plan,
        bank_path=fused_bank,
        expected_plan_sha256=expected_fused_plan_sha256,
        expected_plan_file_sha256=expected_fused_plan_file_sha256,
        expected_bank_file_sha256=expected_fused_bank_file_sha256,
    )
    parent_inventory_sha = canonical_sha256(_checkpoint_inventory(parent_path))
    if parent_inventory_sha != plan["parent_checkpoint_inventory_sha256"]:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Parent checkpoint inventory differs from the producer plan."
        )
    torch_dtype = _dtype(dtype)
    if dtype != plan["materialization_dtype"]:
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Runner dtype must equal the authenticated producer materialization dtype."
        )
    resolved_device = torch.device(device)
    bank_tensors = _load_tensor_archive(
        fused_bank,
        expected_file_sha256=expected_fused_bank_file_sha256,
        records=plan["bank"]["tensor_records"],
    )
    parent_template = load_lfm2_direct_local_model(
        parent_path,
        device=resolved_device,
        dtype=torch_dtype,
    )
    if (
        fingerprint_lfm2_direct_local_state(parent_template)
        != plan["parent_model_state_sha256"]
    ):
        raise Lfm2FusedAttentionDiagnosticRunError(
            "Direct-local parent state differs from the fused plan."
        )
    del parent_template

    prompt_rows = [dict(value) for value in load_review_prompts(prompt_path)]
    prompt_set_sha = canonical_sha256(prompt_rows)
    parent_cache = _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=_digest(
            expected_parent_envelope_sha256,
            label="expected parent envelope",
        ),
        expected_file_sha256=_digest(
            expected_parent_envelope_file_sha256,
            label="expected parent envelope file",
        ),
        expected_prompt_set_sha256=prompt_set_sha,
        expected_prompt_ids=[str(value["id"]) for value in prompt_rows],
    )
    parent_binding = _authenticate_parent_envelope_binding(
        parent_cache=parent_cache,
        parent_checkpoint=parent_path,
        plan=plan,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    parent_rows = {str(value["id"]): value for value in parent_cache["responses"]}
    tokenizer = load_lfm2_direct_local_tokenizer(parent_path)
    shard_receipt = _schedule_shard(
        plan["schedule"],
        shard_count=shard_count,
        shard_index=shard_index,
    )
    schedule_rows = _authenticate_schedule_shard(
        plan["schedule"],
        shard_receipt,
    )
    candidate_ids = [value["candidate_id"] for value in schedule_rows]
    completed = []
    retained: Path | None = None
    progress_path = run_root / "progress.json"
    for candidate_index, schedule_candidate in enumerate(schedule_rows):
        model = load_lfm2_direct_local_model(
            parent_path,
            device=resolved_device,
            dtype=torch_dtype,
        )
        if (
            fingerprint_lfm2_direct_local_state(model)
            != plan["parent_model_state_sha256"]
        ):
            raise Lfm2FusedAttentionDiagnosticRunError(
                "Fresh parent shell changed between schedule candidates."
            )
        installations = _install_schedule_candidate(
            model=model,
            plan=plan,
            schedule_candidate=schedule_candidate,
            tensors=bank_tensors,
            device=resolved_device,
        )
        checkpoint = checkpoints_root / schedule_candidate["candidate_id"]
        manifest = _write_mixed_checkpoint(
            destination=checkpoint,
            model=model,
            plan=plan,
            producer_receipt=publication,
            schedule_candidate=schedule_candidate,
            installation_receipts=installations,
            bank_tensors=bank_tensors,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reloaded, authenticated_manifest = _fresh_load_mixed_checkpoint(
            source=checkpoint,
            expected_manifest_sha256=manifest["manifest_sha256"],
            parent_checkpoint=parent_path,
            plan=plan,
            producer_receipt=publication,
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
        _attach_parent(responses, parent_rows=parent_rows)
        review = _manual_review_document(
            candidate=schedule_candidate,
            manifest=authenticated_manifest,
            parent_binding=parent_binding,
            responses=responses,
        )
        review_path = reviews_root / f"{schedule_candidate['candidate_id']}.json"
        _write_json(review_path, review)
        record = {
            "candidate_id": schedule_candidate["candidate_id"],
            "mode": schedule_candidate["mode"],
            "variant_id": schedule_candidate["variant_id"],
            "installations": _plain(schedule_candidate["installations"]),
            "mixed_checkpoint_manifest_sha256": authenticated_manifest[
                "manifest_sha256"
            ],
            "fresh_reload_verified_before_decode": True,
            "manual_review_file": review_path.relative_to(run_root).as_posix(),
            "manual_review_file_sha256": _file_sha256(review_path),
            "manual_review_sha256": review["review_sha256"],
            "parent_envelope_binding_sha256": parent_binding["binding_sha256"],
            "responses": responses,
            "response_count": len(responses),
        }
        completed.append(record)
        _write_json(
            progress_path,
            _progress(
                publication=publication,
                plan=plan,
                completed=completed,
                candidate_ids=candidate_ids,
                shard=shard_receipt,
            ),
            replace=progress_path.exists(),
        )
        del reloaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if checkpoint_retention == "receipts-only":
            shutil.rmtree(checkpoint)
        else:
            if retained is not None:
                shutil.rmtree(retained)
            retained = checkpoint
        if (
            candidate_index == len(schedule_rows) - 1
            and checkpoint_retention == "keep-last"
        ):
            retained = checkpoint

    review_index_body = {
        "schema": MANUAL_REVIEW_SCHEMA,
        "status": "manual-review-bank-complete",
        "candidate_ids": candidate_ids,
        "review_files": [
            {
                "candidate_id": value["candidate_id"],
                "relative_path": value["manual_review_file"],
                "file_sha256": value["manual_review_file_sha256"],
                "review_sha256": value["manual_review_sha256"],
            }
            for value in completed
        ],
        "automatic_acceptance_or_rejection_performed": False,
    }
    review_index = {
        **review_index_body,
        "review_index_sha256": canonical_sha256(review_index_body),
    }
    _write_json(reviews_root / "index.json", review_index)
    body = {
        "schema": RUN_SCHEMA,
        "status": "NON-FINAL-FUSED-ATTENTION-WHOLE-BEHAVIOR-DECODED",
        "producer_receipt": _plain(publication),
        "producer_receipt_sha256": publication["receipt_sha256"],
        "plan": _plain(plan),
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": plan["schedule"]["schedule_sha256"],
        "schedule_shard": _plain(shard_receipt),
        "schedule_shard_sha256": shard_receipt["shard_sha256"],
        "complete_full_schedule_sweep_claimed": shard_count == 1,
        "parent_model_state_sha256": plan["parent_model_state_sha256"],
        "parent_envelope_binding": _plain(parent_binding),
        "parent_envelope_binding_sha256": parent_binding["binding_sha256"],
        "prompt_set_sha256": prompt_set_sha,
        "prompt_count": len(prompt_rows),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "candidate_ids": candidate_ids,
        "reviewed_candidate_count": len(completed),
        "candidate_reviews": completed,
        "manual_review_index": review_index,
        "checkpoint_retention": checkpoint_retention,
        "retained_checkpoint_paths": ([] if retained is None else [str(retained)]),
        "evaluation_policy": {
            "decoded_whole_responses_are_primary_evidence": True,
            "cached_parent_text_comparison_only": True,
            "exact_token_match_required": False,
            "numeric_metrics_have_decision_authority": False,
            "manual_intelligence_review_required": True,
            "automatic_acceptance_or_rejection_performed": False,
        },
        "interpretation": {
            "ordinary_fused_attention_installed": True,
            "bridge_sidecars_present": False,
            "mixed_attention_topology_custom_fresh_reload_verified": True,
            "isolated_layers_localize_role_fusion": True,
            "all_attention_groups_localize_composition": True,
            "ordered_attention_prefixes_localize_depth_composition": True,
            "diagnostic_parent_shell_only": True,
            "standalone_child_compression_claimed": False,
            "does_not_modify_main_exact4x_compiler_or_schedule": True,
        },
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_model_generate_calls": 0,
            "parent_identity_reauthentication_model_loads": 1,
            "parent_identity_reauthentication_forward_calls": 0,
            "parent_identity_reauthentication_generate_calls": 0,
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
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--expected-producer-receipt-sha256", required=True)
    parser.add_argument("--expected-producer-receipt-file-sha256", required=True)
    parser.add_argument("--fused-plan", type=Path, required=True)
    parser.add_argument("--expected-fused-plan-sha256", required=True)
    parser.add_argument("--expected-fused-plan-file-sha256", required=True)
    parser.add_argument("--fused-bank", type=Path, required=True)
    parser.add_argument("--expected-fused-bank-file-sha256", required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=8192)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument(
        "--checkpoint-retention",
        choices=("receipts-only", "keep-last"),
        default="receipts-only",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(**vars(arguments))
    print(json.dumps(_plain(result), allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
