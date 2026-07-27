#!/usr/bin/env python3
"""Run sealed fused-body diagnostics through fresh parent reconstruction.

Each selected schedule row starts from a newly loaded, authenticated local
parent shell.  Selected fused MLP/short-conv states are reconstructed from the
producer archive, installed, differentially saved, and custom-fresh-reloaded
before decoding the complete prompt bank.  Parent text is reused only from the
authenticated cache.  No training or numeric acceptance gate is performed.
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

from experiments import prepare_lfm2_fused_body_diagnostic as producer  # noqa: E402
from experiments import run_lfm2_progressive_component_ablation as b038  # noqa: E402
from experiments.run_lfm2_dynamic_catalog import (  # noqa: E402
    fingerprint_local_tokenizer,
)
from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _generate_rows,
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
from neural_forge.hf_lfm2_fused_body_diagnostic import (  # noqa: E402
    Lfm2FusedBodyDiagnosticCandidate,
    authenticate_lfm2_fused_body_candidate,
    reconstruct_lfm2_fused_body_candidate,
)
from neural_forge.hf_lfm2_progressive_component_ablation import (  # noqa: E402
    _model_state_sha256,
    _module_at,
    _module_state_sha256,
    _parameter_alias_topology,
    _replace_module,
    _stored_tensor_elements,
    _unique_parameter_bytes,
    _unique_parameter_elements,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


RUN_SCHEMA = "neural-forge-lfm2-fused-body-diagnostic-run-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-fused-body-diagnostic-progress-v1"
CHECKPOINT_SCHEMA = "neural-forge-lfm2-fused-body-differential-checkpoint-v1"
MANUAL_REVIEW_SCHEMA = "neural-forge-lfm2-fused-body-manual-review-v1"
SHARD_SCHEMA = "neural-forge-lfm2-fused-body-schedule-shard-v1"
CHECKPOINT_TENSOR_FILENAME = "fused-body-overrides.safetensors"
CHECKPOINT_MANIFEST_FILENAME = "manifest.json"


class Lfm2FusedBodyDiagnosticRunError(RuntimeError):
    """A source, reconstruction, persistence, cache, or review invariant failed."""


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
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
        raise Lfm2FusedBodyDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise Lfm2FusedBodyDiagnosticRunError(f"{label} is missing or unsafe.")
    return path


def _ordinary_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2FusedBodyDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise Lfm2FusedBodyDiagnosticRunError(f"{label} is missing or unsafe.")
    return path


def _new_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2FusedBodyDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2FusedBodyDiagnosticRunError(f"{label} must not already exist.")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2FusedBodyDiagnosticRunError(
            f"{label} parent must be one ordinary directory."
        )
    return path


def _forbid_run_inside_authenticated_inputs(
    run_directory: Path,
    *,
    input_directories: Sequence[Path],
) -> None:
    target = run_directory.resolve(strict=False)
    for raw in input_directories:
        source = raw.resolve(strict=True)
        if target == source or source in target.parents:
            raise Lfm2FusedBodyDiagnosticRunError(
                "Run directory must be lexically separate from every authenticated "
                "input checkpoint/publication/runtime-evidence directory."
            )


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    if path.is_symlink() or (path.exists() and not replace):
        raise Lfm2FusedBodyDiagnosticRunError(
            f"Refusing to replace evidence file {path.name!r}."
        )
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise Lfm2FusedBodyDiagnosticRunError(
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
        raise Lfm2FusedBodyDiagnosticRunError(
            f"{label} is not canonical JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2FusedBodyDiagnosticRunError(f"{label} must be one JSON object.")
    return value


def _checkpoint_inventory(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2FusedBodyDiagnosticRunError(
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
        raise Lfm2FusedBodyDiagnosticRunError("Checkpoint inventory cannot be empty.")
    return rows


def _artifact_inventory(
    directory: Path,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2FusedBodyDiagnosticRunError(
                "Differential checkpoint contains a symlink."
            )
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            if relative not in exclude:
                rows.append(
                    {
                        "relative_path": relative,
                        "size_bytes": int(path.stat().st_size),
                        "sha256": _file_sha256(path),
                    }
                )
    return rows


def _schedule_shard(
    schedule: Mapping[str, Any],
    *,
    shard_count: int,
    shard_index: int,
) -> Mapping[str, Any]:
    producer._authenticate_schedule(schedule)
    candidates = schedule.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise Lfm2FusedBodyDiagnosticRunError(
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
        raise Lfm2FusedBodyDiagnosticRunError(
            "Shard count/index do not define one nonempty deterministic shard."
        )
    selected_indices = [
        index for index in range(len(candidates)) if index % shard_count == shard_index
    ]
    selected_ids = [candidates[index]["candidate_id"] for index in selected_indices]
    body = {
        "schema": SHARD_SCHEMA,
        "status": "authenticated-position-modulo-schedule-shard",
        "full_schedule_sha256": schedule["schedule_sha256"],
        "partition_method": "canonical-position-modulo-shard-count-v1",
        "full_candidate_count": len(candidates),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "selected_candidate_indices": selected_indices,
        "selected_candidate_ids": selected_ids,
        "selected_candidate_ids_sha256": canonical_sha256(selected_ids),
        "selected_candidate_count": len(selected_ids),
        "partial_sweep": shard_count != 1,
        "complete_full_schedule_sweep_claimed": shard_count == 1,
    }
    return {**body, "shard_sha256": canonical_sha256(body)}


def _authenticate_schedule_shard(
    schedule: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(shard, Mapping):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Schedule shard receipt must be one mapping."
        )
    expected = _schedule_shard(
        schedule,
        shard_count=shard.get("shard_count"),
        shard_index=shard.get("shard_index"),
    )
    if dict(shard) != dict(expected):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Schedule shard differs from the authenticated modulo partition."
        )
    return tuple(
        schedule["candidates"][index] for index in shard["selected_candidate_indices"]
    )


def _load_publication(
    path: Path,
    *,
    expected_receipt_sha256: str,
    expected_file_sha256: str,
) -> Mapping[str, Any]:
    ordinary = _ordinary_file(path, label="fused-body producer receipt")
    if ordinary.name != producer.PRODUCER_RECEIPT_FILENAME or _file_sha256(
        ordinary
    ) != _digest(expected_file_sha256, label="expected fused producer receipt file"):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fused-body producer receipt filename or file pin is invalid."
        )
    value = _load_json(ordinary, label="fused-body producer receipt")
    body = dict(value)
    supplied = body.pop("receipt_sha256", None)
    if (
        value.get("schema") != producer.PRODUCER_SCHEMA
        or value.get("status") != producer.PRODUCER_STATUS
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_receipt_sha256, label="expected fused producer receipt")
        or value.get("plan_filename") != producer.PLAN_FILENAME
        or value.get("archive_filename") != producer.ARCHIVE_FILENAME
        or value.get("fresh_source_receipt_authentication_performed") is not True
        or value.get("fresh_archive_authentication_performed") is not True
        or value.get("direct_local_checkpoint_loading_used") is not True
        or value.get("human_authored_module_or_tensor_pins_accepted") is not False
        or value.get("training_steps") != 0
        or value.get("model_forward_calls") != 0
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fused-body producer receipt seal or policy is invalid."
        )
    return value


def _load_plan(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_file_sha256: str,
) -> Mapping[str, Any]:
    ordinary = _ordinary_file(path, label="fused-body plan")
    if ordinary.name != producer.PLAN_FILENAME or _file_sha256(ordinary) != _digest(
        expected_file_sha256, label="expected fused-body plan file"
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fused-body plan filename or file pin is invalid."
        )
    value = _load_json(ordinary, label="fused-body plan")
    body = dict(value)
    supplied = body.pop("plan_sha256", None)
    components = value.get("components")
    schedule = value.get("schedule")
    controls = value.get("fold_stage_controls")
    parent_cache_binding = value.get("parent_cache_binding")
    binding_body = (
        dict(parent_cache_binding) if isinstance(parent_cache_binding, Mapping) else {}
    )
    binding_seal = binding_body.pop("binding_sha256", None)
    cached_reference = (
        controls.get("cached_parent_reference")
        if isinstance(controls, Mapping)
        else None
    )
    fp32_structural_control = (
        controls.get("fp32_unmodified_parent_structural_shell")
        if isinstance(controls, Mapping)
        else None
    )
    final_control = controls.get("final_s2") if isinstance(controls, Mapping) else None
    s0_control = (
        controls.get("width_matched_s0") if isinstance(controls, Mapping) else None
    )
    if (
        value.get("schema") != producer.PLAN_SCHEMA
        or value.get("status") != producer.PLAN_STATUS
        or value.get("producer_schema") != producer.PRODUCER_SCHEMA
        or supplied != canonical_sha256(body)
        or supplied != _digest(expected_plan_sha256, label="expected fused-body plan")
        or not isinstance(components, list)
        or not components
        or value.get("component_ids") != [row.get("component_id") for row in components]
        or value.get("component_count") != len(components)
        or not isinstance(value.get("archive"), Mapping)
        or value["archive"].get("schema") != producer.ARCHIVE_SCHEMA
        or value["archive"].get("filename") != producer.ARCHIVE_FILENAME
        or value.get("materialization_dtype") != "float32"
        or value.get("diagnostic_parent_shell_only") is not True
        or value.get("standalone_child_compression_claimed") is not False
        or value.get("numeric_acceptance_gate_defined") is not False
        or value.get("bridge_sidecar_parameters_materialized") is not False
        or not isinstance(parent_cache_binding, Mapping)
        or parent_cache_binding.get("schema")
        != "neural-forge-lfm2-fused-body-parent-cache-binding-v1"
        or parent_cache_binding.get("status")
        != "exact-local-parent-cache-source-authenticated"
        or binding_seal != canonical_sha256(binding_body)
        or parent_cache_binding.get("parent_checkpoint_inventory_sha256")
        != value.get("parent_checkpoint_inventory_sha256")
        or parent_cache_binding.get("parent_runtime_materialization_dtype")
        != "bfloat16"
        or parent_cache_binding.get("direct_local_checkpoint_source_authenticated")
        is not True
        or parent_cache_binding.get("hub_loader_permitted") is not False
        or parent_cache_binding.get("network_access_permitted") is not False
        or not isinstance(cached_reference, Mapping)
        or cached_reference.get("available") is not True
        or cached_reference.get("exact") is not True
        or cached_reference.get("response_runtime_dtype") != "bfloat16"
        or cached_reference.get("schedule_candidate_id") != "cached-parent-reference"
        or cached_reference.get("candidate_runtime_executed") is not False
        or cached_reference.get("response_attributed_to_fp32_structural_shell")
        is not False
        or not isinstance(fp32_structural_control, Mapping)
        or fp32_structural_control.get("available") is not True
        or fp32_structural_control.get("exact_structural_manifest") is not True
        or fp32_structural_control.get("runtime_dtype") != "float32"
        or fp32_structural_control.get("candidate_runtime_executed") is not False
        or fp32_structural_control.get("decoded_responses_claimed") is not False
        or not isinstance(final_control, Mapping)
        or final_control.get("available") is not True
        or final_control.get("exact") is not True
        or final_control.get("candidate_count") != len(components)
        or not isinstance(s0_control, Mapping)
        or s0_control.get("available") is not False
        or s0_control.get("exact") is not False
        or s0_control.get("bytes_invented_or_reconstructed") is not False
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fused-body plan seal, schema, inventory, or policy is invalid."
        )
    for field in (
        "parent_state_sha256",
        "parent_identity_sha256",
        "tokenizer_sha256",
    ):
        _digest(
            parent_cache_binding.get(field),
            label=f"parent cache binding {field}",
        )
    parent_unique = parent_cache_binding.get("parent_unique_parameters")
    if (
        isinstance(parent_unique, bool)
        or not isinstance(parent_unique, int)
        or parent_unique <= 0
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Parent cache binding parameter count is invalid."
        )
    producer._authenticate_schedule(schedule)
    expected_schedule = producer._schedule_document(components)
    if dict(schedule) != dict(expected_schedule):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fused-body schedule is not the deterministic component-derived schedule."
        )
    return value


def _authenticate_publication(
    *,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    archive_path: Path,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
    expected_archive_file_sha256: str,
) -> None:
    receipt_file = _ordinary_file(receipt_path, label="fused-body producer receipt")
    plan_file = _ordinary_file(plan_path, label="fused-body plan")
    archive_file = _ordinary_file(archive_path, label="fused-body archive")
    if (
        receipt_file.parent != plan_file.parent
        or receipt_file.parent != archive_file.parent
        or receipt.get("plan_sha256") != expected_plan_sha256
        or plan.get("plan_sha256") != expected_plan_sha256
        or receipt.get("plan_file_sha256") != expected_plan_file_sha256
        or _file_sha256(plan_file) != expected_plan_file_sha256
        or receipt.get("archive_file_sha256") != expected_archive_file_sha256
        or plan.get("archive", {}).get("file_sha256") != expected_archive_file_sha256
        or _file_sha256(archive_file) != expected_archive_file_sha256
        or receipt.get("parent_checkpoint_inventory_sha256")
        != plan.get("parent_checkpoint_inventory_sha256")
        or receipt.get("final_checkpoint_inventory_sha256")
        != plan.get("final_checkpoint_inventory_sha256")
        or receipt.get("component_count") != plan.get("component_count")
        or receipt.get("candidate_schedule_count")
        != plan.get("schedule", {}).get("candidate_count")
        or receipt.get("parent_cache_binding_sha256")
        != plan.get("parent_cache_binding", {}).get("binding_sha256")
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fused-body plan/archive are not the externally pinned producer publication."
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
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fused-body archives require safetensors."
        ) from exc
    ordinary = _ordinary_file(path, label="fused-body tensor archive")
    if _file_sha256(ordinary) != _digest(
        expected_file_sha256, label="expected fused-body tensor archive"
    ) or not isinstance(records, Mapping):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fused-body tensor archive differs from its external pin."
        )
    result = {}
    with safe_open(str(ordinary), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(records):
            raise Lfm2FusedBodyDiagnosticRunError(
                "Fused-body tensor inventory is incomplete."
            )
        for name in sorted(records):
            tensor = handle.get_tensor(name).detach().cpu().contiguous()
            observed = {
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sha256": lfm2_authenticated_tensor_sha256(tensor),
            }
            if observed != records[name]:
                raise Lfm2FusedBodyDiagnosticRunError(
                    f"Fused-body tensor {name!r} differs from its record."
                )
            result[name] = tensor
    return MappingProxyType(result)


def _component_map(plan: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    result = {}
    paths = set()
    for row in plan["components"]:
        if not isinstance(row, Mapping):
            raise Lfm2FusedBodyDiagnosticRunError("Component plan row is malformed.")
        identifier = row.get("component_id")
        path = row.get("parent_module_path")
        if (
            not isinstance(identifier, str)
            or not isinstance(path, str)
            or identifier in result
            or path in paths
            or row.get("kind") not in {"mlp", "short_conv"}
        ):
            raise Lfm2FusedBodyDiagnosticRunError(
                "Fused-body component IDs/paths are invalid or duplicated."
            )
        result[identifier] = row
        paths.add(path)
    return MappingProxyType(result)


def _candidate_from_tensors(
    row: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> Lfm2FusedBodyDiagnosticCandidate:
    names = row.get("tensor_names")
    receipt = row.get("receipt")
    descriptor = row.get("reconstruction_descriptor")
    expected_keys = (
        {"w1.weight", "w2.weight", "w3.weight"}
        if row.get("kind") == "mlp"
        else {"conv.weight", "in_proj.weight", "out_proj.weight"}
    )
    if (
        not isinstance(names, Mapping)
        or set(names) != expected_keys
        or not isinstance(receipt, Mapping)
        or not isinstance(descriptor, Mapping)
        or any(name not in tensors for name in names.values())
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Component row lacks its complete candidate tensor mapping."
        )
    candidate = Lfm2FusedBodyDiagnosticCandidate(
        kind=str(row["kind"]),
        tensors=MappingProxyType(
            {
                state_name: tensors[archive_name]
                for state_name, archive_name in names.items()
            }
        ),
        receipt=MappingProxyType(dict(receipt)),
        reconstruction_descriptor=MappingProxyType(dict(descriptor)),
    )
    authenticated = authenticate_lfm2_fused_body_candidate(candidate)
    if (
        authenticated["result_sha256"] != row.get("candidate_result_sha256")
        or authenticated["candidate_state_sha256"] != row.get("candidate_state_sha256")
        or candidate.reconstruction_descriptor["descriptor_sha256"]
        != row.get("reconstruction_descriptor_sha256")
        or dict(authenticated["physical_parameter_accounting"])
        != dict(row.get("physical_parameter_accounting", {}))
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Candidate state/receipt differs from the producer plan."
        )
    return candidate


def _no_sidecars(module: Any, *, expected_state_names: set[str]) -> bool:
    parameter_names = {name for name, _ in module.named_parameters()}
    state_names = set(module.state_dict())
    return (
        parameter_names == expected_state_names
        and state_names == expected_state_names
        and not hasattr(module, "input_bridge")
        and not hasattr(module, "output_bridge")
        and "bridged" not in type(module).__name__.lower()
    )


def _install_schedule_candidate(
    *,
    model: Any,
    plan: Mapping[str, Any],
    schedule_candidate: Mapping[str, Any],
    tensors: Mapping[str, Any],
    device: Any,
) -> Mapping[str, Any]:
    components = _component_map(plan)
    identifiers = schedule_candidate.get("component_ids")
    state_pins = schedule_candidate.get("component_state_sha256")
    if (
        not isinstance(identifiers, list)
        or len(identifiers) != len(set(identifiers))
        or not isinstance(state_pins, Mapping)
        or set(state_pins) != set(identifiers)
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Schedule candidate installation set or state pins are malformed."
        )
    before = {
        "unique_parameter_elements": _unique_parameter_elements(model),
        "unique_parameter_bytes": _unique_parameter_bytes(model),
        "stored_tensor_elements": _stored_tensor_elements(model),
        "parameter_alias_topology": _plain(_parameter_alias_topology(model)),
        "model_state_sha256": _model_state_sha256(model),
    }
    expected_reduction = 0
    receipts = []
    installed_paths = set()
    for identifier in identifiers:
        row = components.get(identifier)
        if row is None or row["parent_module_path"] in installed_paths:
            raise Lfm2FusedBodyDiagnosticRunError(
                "Schedule names an absent or duplicate fused-body seam."
            )
        if state_pins[identifier] != row["candidate_state_sha256"]:
            raise Lfm2FusedBodyDiagnosticRunError(
                "Schedule component state pin differs from its plan row."
            )
        original = _module_at(model, row["parent_module_path"])
        if _module_state_sha256(original) != row["parent_module_sha256"]:
            raise Lfm2FusedBodyDiagnosticRunError(
                f"Fresh parent component {identifier!r} differs from the producer plan."
            )
        candidate = _candidate_from_tensors(row, tensors)
        reconstruction = reconstruct_lfm2_fused_body_candidate(
            candidate=candidate,
            parent_config=model.config,
            expected_candidate_result_sha256=row["candidate_result_sha256"],
            expected_descriptor_sha256=row["reconstruction_descriptor_sha256"],
            device=device,
        )
        physical = candidate.receipt["physical_parameter_accounting"]
        reduction = int(physical["physical_parameter_reduction_from_parent"])
        expected_state_names = set(row["tensor_names"])
        original_elements = _unique_parameter_elements(original)
        installed_elements = _unique_parameter_elements(reconstruction.module)
        if (
            original_elements
            != int(physical["original_parent_component_parameter_elements"])
            or installed_elements != int(physical["fused_candidate_parameter_elements"])
            or original_elements - installed_elements != reduction
            or _stored_tensor_elements(reconstruction.module) != installed_elements
            or not _no_sidecars(
                reconstruction.module,
                expected_state_names=expected_state_names,
            )
        ):
            raise Lfm2FusedBodyDiagnosticRunError(
                f"Fused component {identifier!r} failed physical/no-sidecar audit."
            )
        _replace_module(model, row["parent_module_path"], reconstruction.module)
        installed_paths.add(row["parent_module_path"])
        expected_reduction += reduction
        body = {
            "component_id": identifier,
            "kind": row["kind"],
            "layer_index": row["layer_index"],
            "parent_module_path": row["parent_module_path"],
            "candidate_state_sha256": row["candidate_state_sha256"],
            "candidate_result_sha256": row["candidate_result_sha256"],
            "reconstruction_receipt": _plain(reconstruction.receipt),
            "physical_parameter_accounting": _plain(physical),
            "original_component_parameter_elements": original_elements,
            "installed_component_parameter_elements": installed_elements,
            "realized_component_parameter_delta": reduction,
            "bridge_sidecar_parameter_elements": 0,
            "runtime_parameter_alias_group_count": 0,
            "ordinary_stock_mlp_installed": row["kind"] == "mlp",
            "custom_bottleneck_short_conv_installed": row["kind"] == "short_conv",
            "training_steps": 0,
            "model_forward_calls": 0,
        }
        receipts.append({**body, "installation_sha256": canonical_sha256(body)})
    after = {
        "unique_parameter_elements": _unique_parameter_elements(model),
        "unique_parameter_bytes": _unique_parameter_bytes(model),
        "stored_tensor_elements": _stored_tensor_elements(model),
        "parameter_alias_topology": _plain(_parameter_alias_topology(model)),
        "model_state_sha256": _model_state_sha256(model),
    }
    if (
        before["unique_parameter_elements"] - after["unique_parameter_elements"]
        != expected_reduction
        or before["unique_parameter_bytes"] - after["unique_parameter_bytes"]
        != 4 * expected_reduction
        or before["stored_tensor_elements"] - after["stored_tensor_elements"]
        != expected_reduction
        or before["parameter_alias_topology"] != after["parameter_alias_topology"]
        or (not identifiers and before != after)
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Mixed fused-body installation failed model-level count/alias audit."
        )
    body = {
        "candidate_id": schedule_candidate["candidate_id"],
        "component_ids": list(identifiers),
        "installation_receipts": receipts,
        "installation_count": len(receipts),
        "before": before,
        "after": after,
        "expected_parameter_reduction": expected_reduction,
        "realized_parameter_reduction": (
            before["unique_parameter_elements"] - after["unique_parameter_elements"]
        ),
        "realized_parameter_byte_reduction": (
            before["unique_parameter_bytes"] - after["unique_parameter_bytes"]
        ),
        "realized_stored_tensor_reduction": (
            before["stored_tensor_elements"] - after["stored_tensor_elements"]
        ),
        "parameter_alias_topology_unchanged": True,
        "bridge_sidecar_parameter_elements": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "training_steps": 0,
        "model_forward_calls": 0,
    }
    return MappingProxyType({**body, "installation_set_sha256": canonical_sha256(body)})


def _checkpoint_tensor_subset(
    *,
    plan: Mapping[str, Any],
    schedule_candidate: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> Mapping[str, Any]:
    components = _component_map(plan)
    names = set()
    for identifier in schedule_candidate["component_ids"]:
        names.update(components[identifier]["tensor_names"].values())
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
    installation_receipt: Mapping[str, Any],
    archive_tensors: Mapping[str, Any],
) -> Mapping[str, Any]:
    from safetensors.torch import save_file

    target = destination.expanduser().absolute()
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise Lfm2FusedBodyDiagnosticRunError(
            "Differential checkpoint destination is not new and local."
        )
    subset = _checkpoint_tensor_subset(
        plan=plan,
        schedule_candidate=schedule_candidate,
        tensors=archive_tensors,
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    try:
        tensor_file_sha = None
        tensor_records = {}
        tensor_filename = None
        if subset:
            tensor_path = temporary / CHECKPOINT_TENSOR_FILENAME
            save_file(
                dict(subset),
                str(tensor_path),
                metadata={"format": "pt", "neural_forge_status": "NON-FINAL"},
            )
            tensor_filename = CHECKPOINT_TENSOR_FILENAME
            tensor_file_sha = _file_sha256(tensor_path)
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
            "status": "NON-FINAL-FUSED-BODY-DIFFERENTIAL-CHECKPOINT-SAVED",
            "producer_receipt_sha256": producer_receipt["receipt_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "schedule_sha256": plan["schedule"]["schedule_sha256"],
            "candidate": _plain(schedule_candidate),
            "candidate_id": schedule_candidate["candidate_id"],
            "installation_receipt": _plain(installation_receipt),
            "tensor_filename": tensor_filename,
            "tensor_file_sha256": tensor_file_sha,
            "tensor_records": tensor_records,
            "differential_tensor_count": len(tensor_records),
            "model_state_sha256": _model_state_sha256(model),
            "unique_parameter_elements": _unique_parameter_elements(model),
            "unique_parameter_bytes": _unique_parameter_bytes(model),
            "stored_tensor_elements": _stored_tensor_elements(model),
            "parameter_alias_topology": _plain(_parameter_alias_topology(model)),
            "artifact_inventory": _artifact_inventory(temporary),
            "differential_checkpoint_only": True,
            "fresh_parent_reconstruction_required": True,
            "bridge_sidecars_present": False,
            "cached_parent_reference": not bool(schedule_candidate["component_ids"]),
            "fp32_structural_shell_manifest_only": not bool(
                schedule_candidate["component_ids"]
            ),
            "cached_reference_candidate_runtime_executed": False,
            "cached_reference_response_origin": (
                "authenticated-bfloat16-parent-cache"
                if not schedule_candidate["component_ids"]
                else None
            ),
            "cached_reference_response_attributed_to_this_fp32_shell": False,
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
        label="differential checkpoint manifest",
    )
    body = dict(value)
    supplied = body.pop("manifest_sha256", None)
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("status") != "NON-FINAL-FUSED-BODY-DIFFERENTIAL-CHECKPOINT-SAVED"
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_manifest_sha256, label="expected differential manifest")
        or value.get("differential_checkpoint_only") is not True
        or value.get("fresh_parent_reconstruction_required") is not True
        or value.get("bridge_sidecars_present") is not False
        or value.get("standalone_child_compression_claimed") is not False
        or value.get("artifact_inventory")
        != _artifact_inventory(
            directory,
            exclude=frozenset({CHECKPOINT_MANIFEST_FILENAME}),
        )
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Differential checkpoint manifest seal, inventory, or policy is invalid."
        )
    cached_reference = value.get("cached_parent_reference") is True
    if (
        cached_reference != (value.get("tensor_filename") is None)
        or value.get("fp32_structural_shell_manifest_only") != cached_reference
        or value.get("cached_reference_candidate_runtime_executed") is not False
        or value.get("cached_reference_response_attributed_to_this_fp32_shell")
        is not False
        or (
            cached_reference
            and value.get("cached_reference_response_origin")
            != "authenticated-bfloat16-parent-cache"
        )
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Cached-reference structural checkpoint policy is invalid."
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
) -> tuple[Any, Mapping[str, Any]]:
    directory = _ordinary_directory(source, label="differential checkpoint")
    manifest = _load_checkpoint_manifest(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if (
        manifest.get("producer_receipt_sha256") != producer_receipt["receipt_sha256"]
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("schedule_sha256") != plan["schedule"]["schedule_sha256"]
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Differential checkpoint ancestry differs from the producer plan."
        )
    if manifest["tensor_filename"] is None:
        tensors: Mapping[str, Any] = MappingProxyType({})
    else:
        tensors = _load_tensor_archive(
            directory / manifest["tensor_filename"],
            expected_file_sha256=manifest["tensor_file_sha256"],
            records=manifest["tensor_records"],
        )
    model = load_lfm2_direct_local_model(
        parent_checkpoint,
        device=device,
        dtype=__import__("torch").float32,
    )
    if fingerprint_lfm2_direct_local_state(model) != plan["parent_model_state_sha256"]:
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fresh parent shell differs before differential reconstruction."
        )
    installation = _install_schedule_candidate(
        model=model,
        plan=plan,
        schedule_candidate=manifest["candidate"],
        tensors=tensors,
        device=device,
    )
    if (
        _plain(installation) != manifest["installation_receipt"]
        or _model_state_sha256(model) != manifest["model_state_sha256"]
        or _unique_parameter_elements(model) != manifest["unique_parameter_elements"]
        or _unique_parameter_bytes(model) != manifest["unique_parameter_bytes"]
        or _stored_tensor_elements(model) != manifest["stored_tensor_elements"]
        or _plain(_parameter_alias_topology(model))
        != manifest["parameter_alias_topology"]
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fresh differential reconstruction differs from its sealed checkpoint."
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


def _authenticate_parent_cache_binding(
    parent_cache: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> Mapping[str, Any]:
    """Bind cached text to the exact producer parent, tokenizer, and horizon."""

    if not isinstance(parent_cache, Mapping):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Cached parent envelope must be one authenticated mapping."
        )
    binding = plan.get("parent_cache_binding")
    audit = parent_cache.get("execution_audit")
    cache_prompt_limit = parent_cache.get("maximum_prompt_tokens")
    cache_new_limit = parent_cache.get("maximum_new_tokens")
    parent_source = parent_cache.get("parent_model_directory")
    tokenizer_source = parent_cache.get("tokenizer_directory")
    if (
        not isinstance(binding, Mapping)
        or not isinstance(audit, Mapping)
        or isinstance(cache_prompt_limit, bool)
        or not isinstance(cache_prompt_limit, int)
        or cache_prompt_limit < 256
        or cache_prompt_limit != maximum_prompt_tokens
        or isinstance(cache_new_limit, bool)
        or not isinstance(cache_new_limit, int)
        or cache_new_limit < 256
        or cache_new_limit != maximum_new_tokens
        or parent_cache.get("parent_state_sha256") != binding.get("parent_state_sha256")
        or parent_cache.get("parent_identity_sha256")
        != binding.get("parent_identity_sha256")
        or parent_cache.get("parent_unique_parameters")
        != binding.get("parent_unique_parameters")
        or parent_cache.get("tokenizer_sha256") != binding.get("tokenizer_sha256")
        or parent_cache.get("inference_dtype")
        != binding.get("parent_runtime_materialization_dtype")
        or not isinstance(parent_cache.get("inference_device"), str)
        or not parent_cache["inference_device"]
        or not isinstance(parent_source, str)
        or not parent_source
        or not isinstance(tokenizer_source, str)
        or not tokenizer_source
        or audit.get("parent_model_loads") != 1
        or audit.get("parent_generate_invocations") != parent_cache.get("prompt_count")
        or audit.get("response_cache_publications") != 1
        or audit.get("training_steps") != 0
        or audit.get("optimizer_steps") != 0
        or audit.get("backward_calls") != 0
        or audit.get("autograd_grad_calls") != 0
        or audit.get("hub_loader_used") is not False
        or audit.get("network_access") is not False
        or audit.get("parent_state_unchanged") is not True
        or audit.get("parent_parameter_count_unchanged") is not True
        or binding.get("direct_local_checkpoint_source_authenticated") is not True
        or binding.get("hub_loader_permitted") is not False
        or binding.get("network_access_permitted") is not False
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Cached parent envelope is not bound to the exact direct-local parent, "
            "tokenizer, generation horizon, and zero-training provenance."
        )
    body = {
        "schema": "neural-forge-lfm2-fused-body-parent-cache-authentication-v1",
        "status": "cached-parent-exactly-bound-to-producer-plan",
        "envelope_sha256": parent_cache["envelope_sha256"],
        "producer_parent_cache_binding_sha256": binding["binding_sha256"],
        "parent_state_sha256": parent_cache["parent_state_sha256"],
        "parent_identity_sha256": parent_cache["parent_identity_sha256"],
        "tokenizer_sha256": parent_cache["tokenizer_sha256"],
        "cache_maximum_prompt_tokens": cache_prompt_limit,
        "requested_maximum_prompt_tokens": maximum_prompt_tokens,
        "cache_maximum_new_tokens": cache_new_limit,
        "requested_maximum_new_tokens": maximum_new_tokens,
        "envelope_parent_model_directory_provenance": parent_source,
        "envelope_tokenizer_directory_provenance": tokenizer_source,
        "parent_generation_performed_by_runner": False,
        "direct_local_source_verified": True,
        "zero_training_provenance_verified": True,
    }
    return MappingProxyType({**body, "authentication_sha256": canonical_sha256(body)})


def _cached_parent_reference_rows(
    prompts: Sequence[Mapping[str, Any]],
    *,
    parent_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        parent = parent_rows[str(prompt["id"])]
        rows.append(
            {
                **dict(prompt),
                "response": parent["response"],
                "response_token_ids": parent["response_token_ids"],
                "diagnostics": parent["diagnostics"],
                "response_origin": "authenticated-bfloat16-parent-cache",
                "response_runtime_dtype": "bfloat16",
                "candidate_runtime_executed": False,
                "response_attributed_to_fp32_structural_shell": False,
                "fp32_structural_shell_runtime_dtype": "float32",
                "reference_and_structural_shell_precision_differ": True,
            }
        )
    _attach_parent(rows, parent_rows=parent_rows)
    return rows


def _manual_review_document(
    *,
    candidate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    runtime_executed = bool(candidate["component_ids"])
    body = {
        "schema": MANUAL_REVIEW_SCHEMA,
        "status": "awaiting-manual-intelligence-review",
        "candidate_id": candidate["candidate_id"],
        "mode": candidate["mode"],
        "component_ids": list(candidate["component_ids"]),
        "mixed_checkpoint_manifest_sha256": manifest["manifest_sha256"],
        "installation_receipt": manifest["installation_receipt"],
        "responses": _plain(responses),
        "response_count": len(responses),
        "review_fields": (
            {
                "coherent": None,
                "answers_the_prompt": None,
                "intelligence_retention_relative_to_parent": None,
                "notes": None,
            }
            if runtime_executed
            else {
                "reference_integrity_notes": None,
            }
        ),
        "fresh_executed_decode_is_primary_candidate_evidence": runtime_executed,
        "cached_bfloat16_text_is_reference_evidence_only": not runtime_executed,
        "exact_token_match_required": False,
        "numeric_metrics_have_decision_authority": False,
        "automatic_acceptance_or_rejection_performed": False,
        "candidate_runtime_executed": runtime_executed,
        "candidate_intelligence_review_applicable": runtime_executed,
        "response_origin": (
            "fresh-fp32-fused-candidate-generation"
            if runtime_executed
            else "authenticated-bfloat16-parent-cache"
        ),
        "response_runtime_dtype": "float32" if runtime_executed else "bfloat16",
        "structural_manifest_runtime_dtype": "float32",
        "reference_and_structural_manifest_precision_differ": not runtime_executed,
        "response_attributed_to_structural_manifest_shell": runtime_executed,
        "fresh_cache_created_per_generated_candidate_and_prompt": runtime_executed,
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
        "full_schedule_sha256": plan["schedule"]["schedule_sha256"],
        "schedule_shard": _plain(shard),
        "schedule_shard_sha256": shard["shard_sha256"],
        "partial_sweep": shard["partial_sweep"],
        "candidate_ids": list(candidate_ids),
        "completed_candidate_ids": [row["candidate_id"] for row in completed],
        "remaining_candidate_ids": list(candidate_ids[len(completed) :]),
        "candidate_reviews": _plain(completed),
        "automatic_acceptance_or_rejection_performed": False,
    }
    return {**body, "progress_sha256": canonical_sha256(body)}


def run(
    *,
    run_directory: Path,
    parent_checkpoint: Path,
    expected_parent_checkpoint_inventory_sha256: str,
    final_checkpoint: Path,
    expected_final_checkpoint_inventory_sha256: str,
    component_publication_directory: Path,
    expected_component_producer_receipt_sha256: str,
    expected_component_producer_receipt_file_sha256: str,
    expected_component_plan_sha256: str,
    expected_component_plan_file_sha256: str,
    expected_component_bridges_file_sha256: str,
    producer_receipt: Path,
    expected_producer_receipt_sha256: str,
    expected_producer_receipt_file_sha256: str,
    fused_plan: Path,
    expected_fused_plan_sha256: str,
    expected_fused_plan_file_sha256: str,
    fused_archive: Path,
    expected_fused_archive_file_sha256: str,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    device: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    checkpoint_retention: str,
    shard_count: int = 1,
    shard_index: int = 0,
) -> Mapping[str, Any]:
    """Run one authenticated schedule shard and publish manual-review evidence."""

    import torch

    if checkpoint_retention not in {"receipts-only", "keep-last"}:
        raise Lfm2FusedBodyDiagnosticRunError(
            "checkpoint_retention must be receipts-only or keep-last."
        )
    if not 256 <= maximum_prompt_tokens <= 8192:
        raise Lfm2FusedBodyDiagnosticRunError(
            "maximum_prompt_tokens must be between 256 and 8192; short caps are forbidden."
        )
    if not 256 <= maximum_new_tokens <= 2048:
        raise Lfm2FusedBodyDiagnosticRunError(
            "maximum_new_tokens must be between 256 and 2048; short caps are forbidden."
        )
    run_root = _new_directory(run_directory, label="run directory")
    parent_path = _ordinary_directory(parent_checkpoint, label="parent checkpoint")
    final_path = _ordinary_directory(final_checkpoint, label="final checkpoint")
    prompt_path = _ordinary_file(prompts_json, label="prompt bank")
    envelope_path = _ordinary_file(parent_envelope, label="cached parent envelope")
    component_root = _ordinary_directory(
        component_publication_directory,
        label="component publication directory",
    )
    fused_receipt_path = _ordinary_file(
        producer_receipt,
        label="fused-body producer receipt",
    )
    fused_plan_path = _ordinary_file(fused_plan, label="fused-body plan")
    fused_archive_path = _ordinary_file(fused_archive, label="fused-body archive")
    _forbid_run_inside_authenticated_inputs(
        run_root,
        input_directories=(
            parent_path,
            final_path,
            component_root,
            fused_receipt_path.parent,
            fused_plan_path.parent,
            fused_archive_path.parent,
            prompt_path.parent,
            envelope_path.parent,
        ),
    )
    if _file_sha256(prompt_path) != _digest(
        expected_prompts_file_sha256,
        label="expected prompt bank",
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
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
        archive_path=fused_archive,
        expected_plan_sha256=expected_fused_plan_sha256,
        expected_plan_file_sha256=expected_fused_plan_file_sha256,
        expected_archive_file_sha256=expected_fused_archive_file_sha256,
    )

    component_receipt_path = component_root / b038.PRODUCER_RECEIPT_FILENAME
    component_plan_path = component_root / b038.PLAN_FILENAME
    component_bridge_path = component_root / b038.BRIDGE_FILENAME
    component_receipt = b038._load_producer_receipt(
        component_receipt_path,
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
        producer_receipt_path=component_receipt_path,
        plan=component_plan,
        component_plan=component_plan_path,
        bridge_tensors=component_bridge_path,
        expected_component_plan_sha256=expected_component_plan_sha256,
        expected_component_plan_file_sha256=expected_component_plan_file_sha256,
        expected_bridge_tensors_file_sha256=expected_component_bridges_file_sha256,
    )
    bridges = b038._load_bridge_tensors(
        component_bridge_path,
        expected_file_sha256=expected_component_bridges_file_sha256,
        plan=component_plan,
    )
    source_authentication = plan["source_authentication"]
    if (
        source_authentication["component_producer_receipt_sha256"]
        != component_receipt["receipt_sha256"]
        or source_authentication["component_plan_sha256"]
        != component_plan["plan_sha256"]
        or source_authentication["component_producer_receipt_file_sha256"]
        != _file_sha256(component_receipt_path)
        or source_authentication["component_plan_file_sha256"]
        != _file_sha256(component_plan_path)
        or source_authentication["component_bridges_file_sha256"]
        != _file_sha256(component_bridge_path)
        or source_authentication["foundation_encoder_sha256"]
        != lfm2_authenticated_tensor_sha256(bridges["foundation.producer_encoder"])
        or source_authentication["analytic_decoder_sha256"]
        != lfm2_authenticated_tensor_sha256(
            bridges["foundation.analytic_output_decoder"]
        )
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fresh b038 authentication differs from fused-body producer ancestry."
        )

    parent_inventory_sha = canonical_sha256(_checkpoint_inventory(parent_path))
    final_inventory_sha = canonical_sha256(_checkpoint_inventory(final_path))
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
        or parent_inventory_sha != plan["parent_checkpoint_inventory_sha256"]
        or final_inventory_sha != plan["final_checkpoint_inventory_sha256"]
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Parent/final checkpoint inventories differ from fused-body ancestry."
        )
    parent_cache_binding = plan["parent_cache_binding"]
    parent_cache_source_model = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    if (
        fingerprint_lfm2_direct_local_state(parent_cache_source_model)
        != parent_cache_binding["parent_state_sha256"]
        or fingerprint_lfm2_parent(parent_cache_source_model).identity_sha256
        != parent_cache_binding["parent_identity_sha256"]
        or _unique_parameter_elements(parent_cache_source_model)
        != parent_cache_binding["parent_unique_parameters"]
        or fingerprint_local_tokenizer(parent_path)
        != parent_cache_binding["tokenizer_sha256"]
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fresh BF16 parent/tokenizer differs from the producer cache binding."
        )
    del parent_cache_source_model
    final_model = load_lfm2_direct_local_model(
        final_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if (
        fingerprint_lfm2_direct_local_state(final_model)
        != plan["final_model_state_sha256"]
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Fresh final compact checkpoint differs from the fused-body plan."
        )
    del final_model
    resolved_device = torch.device(device)
    archive_tensors = _load_tensor_archive(
        fused_archive,
        expected_file_sha256=expected_fused_archive_file_sha256,
        records=plan["archive"]["tensor_records"],
    )
    parent_template = load_lfm2_direct_local_model(
        parent_path,
        device=resolved_device,
        dtype=torch.float32,
    )
    if (
        fingerprint_lfm2_direct_local_state(parent_template)
        != plan["parent_model_state_sha256"]
    ):
        raise Lfm2FusedBodyDiagnosticRunError(
            "Direct-local parent state differs from the fused-body plan."
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
        expected_prompt_ids=[str(row["id"]) for row in prompt_rows],
    )
    parent_cache_authentication = _authenticate_parent_cache_binding(
        parent_cache,
        plan=plan,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    parent_rows = {str(row["id"]): row for row in parent_cache["responses"]}
    tokenizer = load_lfm2_direct_local_tokenizer(parent_path)
    shard_receipt = _schedule_shard(
        plan["schedule"],
        shard_count=shard_count,
        shard_index=shard_index,
    )
    schedule_rows = _authenticate_schedule_shard(plan["schedule"], shard_receipt)
    candidate_ids = [row["candidate_id"] for row in schedule_rows]
    run_root.mkdir()
    checkpoints_root = run_root / "differential-checkpoints"
    reviews_root = run_root / "manual-review"
    checkpoints_root.mkdir()
    reviews_root.mkdir()
    completed = []
    retained: Path | None = None
    progress_path = run_root / "progress.json"
    generated_candidates = 0
    cached_parent_references = 0
    for schedule_candidate in schedule_rows:
        model = load_lfm2_direct_local_model(
            parent_path,
            device=resolved_device,
            dtype=torch.float32,
        )
        if (
            fingerprint_lfm2_direct_local_state(model)
            != plan["parent_model_state_sha256"]
        ):
            raise Lfm2FusedBodyDiagnosticRunError(
                "Fresh parent shell changed between schedule candidates."
            )
        installation = _install_schedule_candidate(
            model=model,
            plan=plan,
            schedule_candidate=schedule_candidate,
            tensors=archive_tensors,
            device=resolved_device,
        )
        checkpoint = checkpoints_root / schedule_candidate["candidate_id"]
        manifest = _write_mixed_checkpoint(
            destination=checkpoint,
            model=model,
            plan=plan,
            producer_receipt=publication,
            schedule_candidate=schedule_candidate,
            installation_receipt=installation,
            archive_tensors=archive_tensors,
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
        )
        if schedule_candidate["component_ids"]:
            responses = _generate_rows(
                reloaded,
                tokenizer,
                prompt_rows,
                maximum_prompt_tokens=maximum_prompt_tokens,
                maximum_new_tokens=maximum_new_tokens,
            )
            _attach_parent(responses, parent_rows=parent_rows)
            generated_candidates += 1
        else:
            responses = _cached_parent_reference_rows(
                prompt_rows,
                parent_rows=parent_rows,
            )
            cached_parent_references += 1
        review = _manual_review_document(
            candidate=schedule_candidate,
            manifest=authenticated_manifest,
            responses=responses,
        )
        review_path = reviews_root / f"{schedule_candidate['candidate_id']}.json"
        _write_json(review_path, review)
        record = {
            "candidate_id": schedule_candidate["candidate_id"],
            "mode": schedule_candidate["mode"],
            "component_ids": list(schedule_candidate["component_ids"]),
            "mixed_checkpoint_manifest_sha256": authenticated_manifest[
                "manifest_sha256"
            ],
            "installation_receipt": authenticated_manifest["installation_receipt"],
            "fresh_fp32_structural_reconstruction_verified": True,
            "candidate_runtime_executed": bool(schedule_candidate["component_ids"]),
            "runtime_decode_performed": bool(schedule_candidate["component_ids"]),
            "fresh_cache_per_candidate_and_prompt": bool(
                schedule_candidate["component_ids"]
            ),
            "cached_parent_reference_reused": not bool(
                schedule_candidate["component_ids"]
            ),
            "response_origin": (
                "fresh-fp32-fused-candidate-generation"
                if schedule_candidate["component_ids"]
                else "authenticated-bfloat16-parent-cache"
            ),
            "response_runtime_dtype": (
                "float32" if schedule_candidate["component_ids"] else "bfloat16"
            ),
            "structural_manifest_runtime_dtype": "float32",
            "reference_and_structural_manifest_precision_differ": not bool(
                schedule_candidate["component_ids"]
            ),
            "manual_review_file": review_path.relative_to(run_root).as_posix(),
            "manual_review_file_sha256": _file_sha256(review_path),
            "manual_review_sha256": review["review_sha256"],
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

    review_index_body = {
        "schema": MANUAL_REVIEW_SCHEMA,
        "status": "manual-review-bank-complete",
        "candidate_ids": candidate_ids,
        "review_files": [
            {
                "candidate_id": row["candidate_id"],
                "relative_path": row["manual_review_file"],
                "file_sha256": row["manual_review_file_sha256"],
                "review_sha256": row["manual_review_sha256"],
            }
            for row in completed
        ],
        "automatic_acceptance_or_rejection_performed": False,
    }
    review_index = {
        **review_index_body,
        "review_index_sha256": canonical_sha256(review_index_body),
    }
    _write_json(reviews_root / "index.json", review_index)
    selected_modes = {str(row["mode"]) for row in schedule_rows}
    selected_component_installation_count = sum(
        len(row["component_ids"]) for row in schedule_rows
    )
    cached_reference_selected = "cached-parent-reference" in selected_modes
    body = {
        "schema": RUN_SCHEMA,
        "status": (
            "NON-FINAL-FUSED-BODY-FULL-SCHEDULE-REVIEW-EVIDENCE-COMPLETE"
            if shard_count == 1
            else "NON-FINAL-FUSED-BODY-PARTIAL-SHARD-REVIEW-EVIDENCE-COMPLETE"
        ),
        "producer_receipt": _plain(publication),
        "producer_receipt_sha256": publication["receipt_sha256"],
        "plan": _plain(plan),
        "plan_sha256": plan["plan_sha256"],
        "full_schedule_sha256": plan["schedule"]["schedule_sha256"],
        "schedule_shard": _plain(shard_receipt),
        "schedule_shard_sha256": shard_receipt["shard_sha256"],
        "partial_sweep": shard_receipt["partial_sweep"],
        "complete_full_schedule_sweep_claimed": shard_count == 1,
        "parent_model_state_sha256": plan["parent_model_state_sha256"],
        "final_model_state_sha256": plan["final_model_state_sha256"],
        "prompt_set_sha256": prompt_set_sha,
        "parent_cache_authentication": _plain(parent_cache_authentication),
        "prompt_count": len(prompt_rows),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "candidate_ids": candidate_ids,
        "reviewed_candidate_count": len(completed),
        "candidate_reviews": completed,
        "manual_review_index": review_index,
        "checkpoint_retention": checkpoint_retention,
        "retained_checkpoint_paths": [] if retained is None else [str(retained)],
        "evaluation_policy": {
            "complete_prompt_bank_required": True,
            "fresh_executed_candidate_decodes_are_primary_candidate_evidence": True,
            "cached_bfloat16_parent_is_reference_evidence_only": True,
            "cached_parent_reference_is_not_an_executed_fp32_candidate": True,
            "cached_parent_text_comparison_only": True,
            "exact_token_match_required": False,
            "numeric_metrics_have_decision_authority": False,
            "manual_intelligence_review_required": True,
            "automatic_acceptance_or_rejection_performed": False,
        },
        "interpretation": {
            "full_plan_contains_cached_bfloat16_parent_reference": True,
            "cached_parent_reference_reviewed_in_this_shard": (
                cached_reference_selected
            ),
            "cached_parent_reference_is_fp32_shell_execution": False,
            "cached_parent_reference_response_runtime_dtype_in_this_shard": (
                "bfloat16" if cached_reference_selected else None
            ),
            "structural_manifest_runtime_dtype": "float32",
            "cached_reference_and_structural_manifest_precision_differ_in_this_shard": (
                cached_reference_selected
            ),
            "final_s2_fused_components_installed_in_this_shard": (
                selected_component_installation_count > 0
            ),
            "final_s2_executed_candidate_count_in_this_shard": generated_candidates,
            "differential_save_and_custom_fresh_parent_reconstruction_verified": True,
            "isolated_components_reviewed_in_this_shard": (
                "isolated-component" in selected_modes
            ),
            "paired_layers_reviewed_in_this_shard": (
                "paired-same-layer-conv-mlp" in selected_modes
            ),
            "subsystem_or_prefix_groups_reviewed_in_this_shard": bool(
                selected_modes
                & {
                    "all-mlp",
                    "all-short-conv",
                    "subsystem-combination-all-body",
                    "cumulative-mlp-conv-by-layer",
                    "ordered-body-prefix",
                }
            ),
            "bridge_sidecars_present": False,
            "diagnostic_parent_shell_only": True,
            "standalone_child_compression_claimed": False,
            "does_not_modify_main_exact4x_compiler_or_b038": True,
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
            "hybrid_model_generate_passes": generated_candidates,
            "cached_parent_references_reused": cached_parent_references,
            "cached_reference_candidate_runtime_executions": 0,
            "fresh_cache_created_for_each_executed_candidate_and_prompt": True,
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
    parser.add_argument("--expected-parent-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-final-checkpoint-inventory-sha256", required=True)
    parser.add_argument("--component-publication-directory", type=Path, required=True)
    parser.add_argument("--expected-component-producer-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-component-producer-receipt-file-sha256", required=True
    )
    parser.add_argument("--expected-component-plan-sha256", required=True)
    parser.add_argument("--expected-component-plan-file-sha256", required=True)
    parser.add_argument("--expected-component-bridges-file-sha256", required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--expected-producer-receipt-sha256", required=True)
    parser.add_argument("--expected-producer-receipt-file-sha256", required=True)
    parser.add_argument("--fused-plan", type=Path, required=True)
    parser.add_argument("--expected-fused-plan-sha256", required=True)
    parser.add_argument("--expected-fused-plan-file-sha256", required=True)
    parser.add_argument("--fused-archive", type=Path, required=True)
    parser.add_argument("--expected-fused-archive-file-sha256", required=True)
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
