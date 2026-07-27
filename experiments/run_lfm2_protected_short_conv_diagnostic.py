#!/usr/bin/env python3
"""Run protected short-conv candidates through fresh parent reconstruction.

Every executed candidate starts from a new authenticated direct-local FP32
parent, receives one protected or dense-control short-conv replacement, is
differentially saved, and is freshly reloaded before decoding the complete
sealed 16-prompt bank.  The native-parent response is reused only from the
authenticated BF16 cache and is explicitly labeled as reference evidence.
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

from experiments import (  # noqa: E402
    prepare_lfm2_protected_short_conv_diagnostic as producer,
)
from experiments import run_lfm2_progressive_component_ablation as b038  # noqa: E402
from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _generate_rows,
    _load_parent_envelope,
    load_review_prompts,
)
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
    _wrapper_classes,
)
from neural_forge.hf_lfm2_protected_short_conv_diagnostic import (  # noqa: E402
    Lfm2ProtectedShortConvCandidate,
    authenticate_lfm2_protected_short_conv_candidate,
    reconstruct_lfm2_protected_short_conv_candidate,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


RUN_SCHEMA = "neural-forge-lfm2-protected-short-conv-diagnostic-run-v1"
PROGRESS_SCHEMA = "neural-forge-lfm2-protected-short-conv-progress-v1"
CHECKPOINT_SCHEMA = "neural-forge-lfm2-protected-short-conv-checkpoint-v1"
MANUAL_REVIEW_SCHEMA = "neural-forge-lfm2-protected-short-conv-manual-review-v1"
SHARD_SCHEMA = "neural-forge-lfm2-protected-short-conv-schedule-shard-v1"
CHECKPOINT_TENSOR_FILENAME = "protected-short-conv-overrides.safetensors"
CHECKPOINT_MANIFEST_FILENAME = "manifest.json"


class Lfm2ProtectedShortConvDiagnosticRunError(RuntimeError):
    """A source, reconstruction, persistence, cache, or review invariant failed."""


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
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
        raise Lfm2ProtectedShortConvDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            f"{label} must be one ordinary file."
        )
    return path


def _ordinary_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ProtectedShortConvDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().resolve(strict=True)
    if not path.is_dir() or path.is_symlink():
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            f"{label} must be one ordinary directory."
        )
    return path


def _new_directory(value: Path, *, label: str) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ProtectedShortConvDiagnosticRunError(f"{label} must be local.")
    path = value.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise Lfm2ProtectedShortConvDiagnosticRunError(f"{label} must be new.")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            f"{label} parent must be one ordinary directory."
        )
    return path


def _forbid_run_inside_authenticated_inputs(
    run_root: Path,
    *,
    input_directories: Sequence[Path],
) -> None:
    target = run_root.resolve(strict=False)
    for raw in input_directories:
        source = raw.resolve(strict=True)
        if target == source or source in target.parents:
            raise Lfm2ProtectedShortConvDiagnosticRunError(
                "Run directory must be lexically separate from every authenticated "
                "input directory."
            )


def _write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    text = json.dumps(_plain(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    if replace:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="ascii")
        temporary.replace(path)
    else:
        path.write_text(text, encoding="ascii")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            f"Could not load {label}."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            f"{label} must be one JSON object."
        )
    return value


def _checkpoint_inventory(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2ProtectedShortConvDiagnosticRunError(
                "Checkpoint inventories reject symbolic links."
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
        raise Lfm2ProtectedShortConvDiagnosticRunError("Checkpoint inventory is empty.")
    return rows


def _artifact_inventory(
    directory: Path,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.name in exclude:
            continue
        if path.is_symlink() or not path.is_file():
            raise Lfm2ProtectedShortConvDiagnosticRunError(
                "Differential checkpoint contains a non-file artifact."
            )
        rows.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return rows


def _load_publication(
    path: Path,
    *,
    expected_receipt_sha256: str,
    expected_file_sha256: str,
) -> Mapping[str, Any]:
    receipt_path = _ordinary_file(path, label="protected producer receipt")
    if _file_sha256(receipt_path) != _digest(
        expected_file_sha256, label="expected protected producer receipt file"
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected producer receipt bytes changed."
        )
    value = _load_json(receipt_path, label="protected producer receipt")
    body = dict(value)
    supplied = body.pop("receipt_sha256", None)
    if (
        value.get("schema") != producer.PRODUCER_SCHEMA
        or value.get("status") != producer.PRODUCER_STATUS
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_receipt_sha256, label="expected protected receipt")
        or value.get("training_steps") != 0
        or value.get("model_forward_calls") != 0
        or value.get("network_access") is not False
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected producer receipt seal or zero-operation policy is invalid."
        )
    return MappingProxyType(value)


def _load_plan(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_file_sha256: str,
) -> Mapping[str, Any]:
    plan_path = _ordinary_file(path, label="protected plan")
    if _file_sha256(plan_path) != _digest(
        expected_file_sha256, label="expected protected plan file"
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError("Protected plan bytes changed.")
    value = _load_json(plan_path, label="protected plan")
    body = dict(value)
    supplied = body.pop("plan_sha256", None)
    candidates = value.get("candidates")
    if (
        value.get("schema") != producer.PLAN_SCHEMA
        or value.get("status") != producer.PLAN_STATUS
        or supplied != canonical_sha256(body)
        or supplied != _digest(expected_plan_sha256, label="expected protected plan")
        or not isinstance(candidates, list)
        or value.get("candidate_count") != len(candidates)
        or len({row.get("candidate_id") for row in candidates}) != len(candidates)
        or value.get("operation_audit", {}).get("training_steps") != 0
        or value.get("operation_audit", {}).get("network_access") is not False
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected plan seal, candidate set, or operation policy is invalid."
        )
    producer.authenticate_schedule(value.get("schedule"))
    return MappingProxyType(value)


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
    if (
        receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("plan_sha256") != expected_plan_sha256
        or receipt.get("plan_file_sha256") != expected_plan_file_sha256
        or receipt.get("plan_file_sha256") != _file_sha256(plan_path)
        or receipt.get("archive_file_sha256") != expected_archive_file_sha256
        or receipt.get("archive_file_sha256") != _file_sha256(archive_path)
        or plan.get("archive", {}).get("file_sha256") != expected_archive_file_sha256
        or receipt.get("schedule_sha256") != plan["schedule"]["schedule_sha256"]
        or receipt_path.name != producer.PRODUCER_RECEIPT_FILENAME
        or plan_path.name != producer.PLAN_FILENAME
        or archive_path.name != producer.ARCHIVE_FILENAME
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected publication files do not cross-authenticate."
        )


def _load_tensor_archive(
    path: Path,
    *,
    expected_file_sha256: str,
    records: Mapping[str, Any],
) -> Mapping[str, Any]:
    from safetensors import safe_open

    archive_path = _ordinary_file(path, label="protected candidate archive")
    if _file_sha256(archive_path) != _digest(
        expected_file_sha256, label="expected candidate archive"
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected candidate archive bytes changed."
        )
    tensors = {}
    with safe_open(str(archive_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(records):
            raise Lfm2ProtectedShortConvDiagnosticRunError(
                "Candidate archive namespace differs from its plan."
            )
        for name in sorted(handle.keys()):
            tensor = handle.get_tensor(name).detach().cpu().contiguous()
            record = records[name]
            if (
                record.get("shape") != [int(size) for size in tensor.shape]
                or record.get("dtype") != str(tensor.dtype).removeprefix("torch.")
                or record.get("sha256") != lfm2_authenticated_tensor_sha256(tensor)
            ):
                raise Lfm2ProtectedShortConvDiagnosticRunError(
                    f"Candidate archive tensor {name!r} differs from its record."
                )
            tensors[name] = tensor
    return MappingProxyType(tensors)


def _candidate_map(plan: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    result = {}
    paths = set()
    for row in plan["candidates"]:
        identifier = row.get("candidate_id")
        path = row.get("parent_module_path")
        if (
            not isinstance(identifier, str)
            or not isinstance(path, str)
            or identifier in result
            or row.get("candidate_type")
            not in {
                "protected-parent-shell",
                "protected-standalone-adapter-control",
                "dense-compact-negative-control",
            }
            or (identifier, path) in paths
        ):
            raise Lfm2ProtectedShortConvDiagnosticRunError(
                "Protected plan candidate identifiers or paths are invalid."
            )
        result[identifier] = row
        paths.add((identifier, path))
    return MappingProxyType(result)


def _candidate_from_tensors(
    row: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> Any:
    names = row.get("tensor_names")
    if (
        not isinstance(names, Mapping)
        or set(names) != {"conv.weight", "in_proj.weight", "out_proj.weight"}
        or any(name not in tensors for name in names.values())
        or not isinstance(row.get("receipt"), Mapping)
        or not isinstance(row.get("reconstruction_descriptor"), Mapping)
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected candidate row lacks complete archived state."
        )
    state = MappingProxyType(
        {
            state_name: tensors[archive_name]
            for state_name, archive_name in names.items()
        }
    )
    if row["candidate_type"] in {
        "protected-parent-shell",
        "protected-standalone-adapter-control",
    }:
        candidate = Lfm2ProtectedShortConvCandidate(
            variant=str(row["variant"]),
            tensors=state,
            receipt=MappingProxyType(dict(row["receipt"])),
            reconstruction_descriptor=MappingProxyType(
                dict(row["reconstruction_descriptor"])
            ),
        )
        authenticated = authenticate_lfm2_protected_short_conv_candidate(candidate)
    else:
        candidate = Lfm2FusedBodyDiagnosticCandidate(
            kind="short_conv",
            tensors=state,
            receipt=MappingProxyType(dict(row["receipt"])),
            reconstruction_descriptor=MappingProxyType(
                dict(row["reconstruction_descriptor"])
            ),
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
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected candidate differs from its sealed plan row."
        )
    return candidate


def _no_sidecars(module: Any) -> bool:
    expected = {"conv.weight", "in_proj.weight", "out_proj.weight"}
    return (
        {name for name, _ in module.named_parameters()} == expected
        and set(module.state_dict()) == expected
        and not hasattr(module, "input_bridge")
        and not hasattr(module, "output_bridge")
    )


def _standalone_parent_shell_wrapper(
    *,
    row: Mapping[str, Any],
    operator: Any,
    tensors: Mapping[str, Any],
    device: Any,
) -> Any:
    import torch

    names = row.get("wrapper_adapter_tensor_names")
    accounting = row.get("parent_shell_wrapper_physical_parameter_accounting")
    if (
        not isinstance(names, Mapping)
        or set(names) != {"input_bridge.weight", "output_bridge.weight"}
        or any(name not in tensors for name in names.values())
        or not isinstance(accounting, Mapping)
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Standalone adapter control lacks its counted bridge state."
        )
    input_weight = tensors[names["input_bridge.weight"]]
    output_weight = tensors[names["output_bridge.weight"]]
    child_hidden, parent_hidden = map(int, input_weight.shape)
    if (
        tuple(output_weight.shape) != (parent_hidden, child_hidden)
        or accounting.get("input_bridge_parameter_elements")
        != int(input_weight.numel())
        or accounting.get("output_bridge_parameter_elements")
        != int(output_weight.numel())
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Standalone adapter geometry differs from explicit accounting."
        )
    input_bridge = torch.nn.Linear(
        parent_hidden,
        child_hidden,
        bias=False,
        device=device,
        dtype=torch.float32,
    )
    output_bridge = torch.nn.Linear(
        child_hidden,
        parent_hidden,
        bias=False,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        input_bridge.weight.copy_(input_weight.to(device=device, dtype=torch.float32))
        output_bridge.weight.copy_(output_weight.to(device=device, dtype=torch.float32))
    wrapper = _wrapper_classes()["short_conv"](
        input_bridge,
        operator,
        output_bridge,
    )
    wrapper.requires_grad_(False)
    wrapper.eval()
    expected_names = {
        "input_bridge.weight",
        "compact_operator.conv.weight",
        "compact_operator.in_proj.weight",
        "compact_operator.out_proj.weight",
        "output_bridge.weight",
    }
    if (
        {name for name, _ in wrapper.named_parameters()} != expected_names
        or set(wrapper.state_dict()) != expected_names
        or _parameter_alias_topology(wrapper)
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Standalone adapter wrapper contains hidden state or aliases."
        )
    return wrapper


def _candidate_parameter_elements(row: Mapping[str, Any]) -> tuple[int, int]:
    physical = row["physical_parameter_accounting"]
    if row["candidate_type"] == "protected-parent-shell":
        return (
            int(physical["original_parent_component_parameter_elements"]),
            int(physical["candidate_parameter_elements"]),
        )
    if row["candidate_type"] == "protected-standalone-adapter-control":
        wrapper = row["parent_shell_wrapper_physical_parameter_accounting"]
        return (
            int(wrapper["original_parent_component_parameter_elements"]),
            int(wrapper["installed_parent_shell_parameter_elements"]),
        )
    return (
        int(physical["original_parent_component_parameter_elements"]),
        int(physical["fused_candidate_parameter_elements"]),
    )


def _install_candidate(
    *,
    model: Any,
    plan: Mapping[str, Any],
    schedule_candidate: Mapping[str, Any],
    tensors: Mapping[str, Any],
    device: Any,
) -> Mapping[str, Any]:
    before = {
        "unique_parameter_elements": _unique_parameter_elements(model),
        "unique_parameter_bytes": _unique_parameter_bytes(model),
        "stored_tensor_elements": _stored_tensor_elements(model),
        "parameter_alias_topology": _plain(_parameter_alias_topology(model)),
        "model_state_sha256": _model_state_sha256(model),
    }
    identifier = schedule_candidate["candidate_id"]
    if identifier == "cached-parent-reference":
        body = {
            "candidate_id": identifier,
            "candidate_type": "cached-parent-reference",
            "installation_count": 0,
            "before": before,
            "after": before,
            "expected_parameter_reduction": 0,
            "realized_parameter_reduction": 0,
            "realized_parameter_byte_reduction": 0,
            "realized_stored_tensor_reduction": 0,
            "parameter_alias_topology_unchanged": True,
            "explicit_interface_adapter_parameter_elements": 0,
            "hidden_sidecar_parameter_elements": 0,
            "training_steps": 0,
            "model_forward_calls": 0,
        }
        return MappingProxyType(
            {**body, "installation_set_sha256": canonical_sha256(body)}
        )
    row = _candidate_map(plan).get(identifier)
    if (
        row is None
        or schedule_candidate.get("candidate_type") != row["candidate_type"]
        or schedule_candidate.get("candidate_state_sha256")
        != row["candidate_state_sha256"]
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Schedule candidate differs from the protected plan."
        )
    original = _module_at(model, row["parent_module_path"])
    candidate = _candidate_from_tensors(row, tensors)
    if row["candidate_type"] in {
        "protected-parent-shell",
        "protected-standalone-adapter-control",
    }:
        reconstruction = reconstruct_lfm2_protected_short_conv_candidate(
            candidate=candidate,
            expected_candidate_result_sha256=row["candidate_result_sha256"],
            expected_descriptor_sha256=row["reconstruction_descriptor_sha256"],
            device=device,
        )
    else:
        reconstruction = reconstruct_lfm2_fused_body_candidate(
            candidate=candidate,
            parent_config=model.config,
            expected_candidate_result_sha256=row["candidate_result_sha256"],
            expected_descriptor_sha256=row["reconstruction_descriptor_sha256"],
            device=device,
        )
    reconstructed_core = reconstruction.module
    if row["candidate_type"] == "protected-standalone-adapter-control":
        replacement = _standalone_parent_shell_wrapper(
            row=row,
            operator=reconstructed_core,
            tensors=tensors,
            device=device,
        )
    else:
        replacement = reconstructed_core
    expected_original, expected_candidate = _candidate_parameter_elements(row)
    original_elements = _unique_parameter_elements(original)
    candidate_elements = _unique_parameter_elements(replacement)
    expected_reduction = expected_original - expected_candidate
    if (
        original_elements != expected_original
        or candidate_elements != expected_candidate
        or _stored_tensor_elements(replacement) != expected_candidate
        or not _no_sidecars(reconstructed_core)
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected replacement failed exact physical/no-sidecar accounting."
        )
    _replace_module(model, row["parent_module_path"], replacement)
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
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Model-level protected replacement count or alias audit failed."
        )
    installation_body = {
        "candidate_id": identifier,
        "candidate_type": row["candidate_type"],
        "layer_index": row["layer_index"],
        "parent_module_path": row["parent_module_path"],
        "candidate_state_sha256": row["candidate_state_sha256"],
        "candidate_result_sha256": row["candidate_result_sha256"],
        "original_component_sha256": _module_state_sha256(original),
        "installed_component_sha256": _module_state_sha256(replacement),
        "reconstruction_receipt": _plain(reconstruction.receipt),
        "physical_parameter_accounting": _plain(row["physical_parameter_accounting"]),
        "original_component_parameter_elements": original_elements,
        "installed_component_parameter_elements": candidate_elements,
        "realized_component_parameter_delta": expected_reduction,
        "explicit_interface_adapter_parameter_elements": row.get(
            "explicit_interface_adapter_parameter_elements", 0
        ),
        "hidden_sidecar_parameter_elements": 0,
        "runtime_parameter_alias_group_count": 0,
        "training_steps": 0,
        "model_forward_calls": 0,
    }
    installation = {
        **installation_body,
        "installation_sha256": canonical_sha256(installation_body),
    }
    body = {
        "candidate_id": identifier,
        "candidate_type": row["candidate_type"],
        "installation_count": 1,
        "installation_receipt": installation,
        "before": before,
        "after": after,
        "expected_parameter_reduction": expected_reduction,
        "realized_parameter_reduction": expected_reduction,
        "realized_parameter_byte_reduction": 4 * expected_reduction,
        "realized_stored_tensor_reduction": expected_reduction,
        "parameter_alias_topology_unchanged": True,
        "explicit_interface_adapter_parameter_elements": row.get(
            "explicit_interface_adapter_parameter_elements", 0
        ),
        "hidden_sidecar_parameter_elements": 0,
        "training_steps": 0,
        "model_forward_calls": 0,
    }
    return MappingProxyType({**body, "installation_set_sha256": canonical_sha256(body)})


def _tensor_subset(
    *,
    plan: Mapping[str, Any],
    schedule_candidate: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> Mapping[str, Any]:
    if schedule_candidate["candidate_id"] == "cached-parent-reference":
        return MappingProxyType({})
    row = _candidate_map(plan)[schedule_candidate["candidate_id"]]
    names = set(row["tensor_names"].values())
    names.update(row.get("wrapper_adapter_tensor_names", {}).values())
    return MappingProxyType(
        {name: tensors[name].detach().cpu().contiguous() for name in sorted(names)}
    )


def _write_differential_checkpoint(
    *,
    destination: Path,
    model: Any,
    plan: Mapping[str, Any],
    publication: Mapping[str, Any],
    schedule_candidate: Mapping[str, Any],
    installation: Mapping[str, Any],
    archive_tensors: Mapping[str, Any],
) -> Mapping[str, Any]:
    from safetensors.torch import save_file

    target = destination.expanduser().absolute()
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Differential checkpoint destination must be new and local."
        )
    subset = _tensor_subset(
        plan=plan,
        schedule_candidate=schedule_candidate,
        tensors=archive_tensors,
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        tensor_filename = None
        tensor_file_sha = None
        tensor_records = {}
        if subset:
            tensor_path = stage / CHECKPOINT_TENSOR_FILENAME
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
        cached = schedule_candidate["candidate_id"] == "cached-parent-reference"
        body = {
            "schema": CHECKPOINT_SCHEMA,
            "status": "NON-FINAL-PROTECTED-SHORT-CONV-DIFFERENTIAL-SAVED",
            "producer_receipt_sha256": publication["receipt_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "schedule_sha256": plan["schedule"]["schedule_sha256"],
            "candidate": _plain(schedule_candidate),
            "candidate_id": schedule_candidate["candidate_id"],
            "installation_receipt": _plain(installation),
            "tensor_filename": tensor_filename,
            "tensor_file_sha256": tensor_file_sha,
            "tensor_records": tensor_records,
            "differential_tensor_count": len(tensor_records),
            "model_state_sha256": _model_state_sha256(model),
            "unique_parameter_elements": _unique_parameter_elements(model),
            "unique_parameter_bytes": _unique_parameter_bytes(model),
            "stored_tensor_elements": _stored_tensor_elements(model),
            "parameter_alias_topology": _plain(_parameter_alias_topology(model)),
            "artifact_inventory": _artifact_inventory(stage),
            "differential_checkpoint_only": True,
            "fresh_parent_reconstruction_required": True,
            "explicit_counted_interface_adapters_present": (
                schedule_candidate.get("candidate_type")
                == "protected-standalone-adapter-control"
            ),
            "hidden_sidecars_present": False,
            "cached_parent_reference": cached,
            "cached_reference_candidate_runtime_executed": False,
            "cached_reference_response_origin": (
                "authenticated-bfloat16-parent-cache" if cached else None
            ),
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
        _write_json(stage / CHECKPOINT_MANIFEST_FILENAME, manifest)
        stage.replace(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return MappingProxyType(manifest)


def _load_checkpoint_manifest(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    value = _load_json(
        directory / CHECKPOINT_MANIFEST_FILENAME,
        label="protected differential checkpoint manifest",
    )
    body = dict(value)
    supplied = body.pop("manifest_sha256", None)
    cached = value.get("cached_parent_reference") is True
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or value.get("status") != "NON-FINAL-PROTECTED-SHORT-CONV-DIFFERENTIAL-SAVED"
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_manifest_sha256, label="expected checkpoint manifest")
        or value.get("differential_checkpoint_only") is not True
        or value.get("fresh_parent_reconstruction_required") is not True
        or value.get("hidden_sidecars_present") is not False
        or value.get("standalone_child_compression_claimed") is not False
        or cached != (value.get("tensor_filename") is None)
        or value.get("artifact_inventory")
        != _artifact_inventory(
            directory,
            exclude=frozenset({CHECKPOINT_MANIFEST_FILENAME}),
        )
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected differential checkpoint manifest is invalid."
        )
    return MappingProxyType(value)


def _fresh_load_differential_checkpoint(
    *,
    source: Path,
    expected_manifest_sha256: str,
    parent_checkpoint: Path,
    plan: Mapping[str, Any],
    publication: Mapping[str, Any],
    device: Any,
) -> tuple[Any, Mapping[str, Any]]:
    import torch

    directory = _ordinary_directory(source, label="differential checkpoint")
    manifest = _load_checkpoint_manifest(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if (
        manifest.get("producer_receipt_sha256") != publication["receipt_sha256"]
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or manifest.get("schedule_sha256") != plan["schedule"]["schedule_sha256"]
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Differential checkpoint ancestry differs from the plan."
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
        dtype=torch.float32,
    )
    if fingerprint_lfm2_direct_local_state(model) != plan["parent_model_state_sha256"]:
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Fresh parent shell differs before protected reconstruction."
        )
    installation = _install_candidate(
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
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Fresh protected reconstruction differs from its checkpoint."
        )
    model.requires_grad_(False)
    model.eval()
    return model, manifest


def _schedule_shard(
    schedule: Mapping[str, Any],
    *,
    shard_count: int,
    shard_index: int,
) -> Mapping[str, Any]:
    producer.authenticate_schedule(schedule)
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError("Shard count/index are invalid.")
    selected = [
        row
        for index, row in enumerate(schedule["candidates"])
        if index % shard_count == shard_index
    ]
    body = {
        "schema": SHARD_SCHEMA,
        "schedule_sha256": schedule["schedule_sha256"],
        "shard_count": shard_count,
        "shard_index": shard_index,
        "selection_policy": "schedule-index-modulo-shard-count",
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "selected_candidate_count": len(selected),
        "partial_sweep": shard_count != 1,
    }
    return {**body, "shard_sha256": canonical_sha256(body)}


def _authenticate_schedule_shard(
    schedule: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    body = dict(shard)
    supplied = body.pop("shard_sha256", None)
    expected = _schedule_shard(
        schedule,
        shard_count=int(shard.get("shard_count", 0)),
        shard_index=int(shard.get("shard_index", -1)),
    )
    if supplied != canonical_sha256(body) or dict(shard) != dict(expected):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected schedule shard seal differs."
        )
    selected = set(shard["selected_candidate_ids"])
    return tuple(
        row for row in schedule["candidates"] if row["candidate_id"] in selected
    )


def _authenticate_parent_cache_binding(
    parent_cache: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> Mapping[str, Any]:
    binding = plan.get("parent_cache_binding")
    audit = parent_cache.get("execution_audit")
    if (
        not isinstance(binding, Mapping)
        or not isinstance(audit, Mapping)
        or parent_cache.get("maximum_prompt_tokens") != maximum_prompt_tokens
        or parent_cache.get("maximum_new_tokens") != maximum_new_tokens
        or maximum_prompt_tokens != 1024
        or maximum_new_tokens != 256
        or parent_cache.get("prompt_count") != 16
        or parent_cache.get("parent_state_sha256") != binding.get("parent_state_sha256")
        or parent_cache.get("parent_identity_sha256")
        != binding.get("parent_identity_sha256")
        or parent_cache.get("parent_unique_parameters")
        != binding.get("parent_unique_parameters")
        or parent_cache.get("tokenizer_sha256") != binding.get("tokenizer_sha256")
        or parent_cache.get("inference_dtype")
        != binding.get("parent_runtime_materialization_dtype")
        or audit.get("parent_model_loads") != 1
        or audit.get("parent_generate_invocations") != 16
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
        or binding.get("direct_local_checkpoint_source_authenticated") is not True
        or binding.get("hub_loader_permitted") is not False
        or binding.get("network_access_permitted") is not False
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Cached parent is not bound to the exact direct-local parent, tokenizer, "
            "16-prompt 1024/256 horizon, and zero-training provenance."
        )
    body = {
        "schema": "neural-forge-lfm2-protected-short-conv-cache-authentication-v1",
        "status": "cached-parent-exactly-bound-to-protected-plan",
        "envelope_sha256": parent_cache["envelope_sha256"],
        "producer_parent_cache_binding_sha256": binding["binding_sha256"],
        "parent_state_sha256": parent_cache["parent_state_sha256"],
        "parent_identity_sha256": parent_cache["parent_identity_sha256"],
        "tokenizer_sha256": parent_cache["tokenizer_sha256"],
        "prompt_count": 16,
        "maximum_prompt_tokens": 1024,
        "maximum_new_tokens": 256,
        "parent_generation_performed_by_runner": False,
        "direct_local_source_verified": True,
        "zero_training_provenance_verified": True,
    }
    return MappingProxyType({**body, "authentication_sha256": canonical_sha256(body)})


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
    executed = candidate["candidate_id"] != "cached-parent-reference"
    body = {
        "schema": MANUAL_REVIEW_SCHEMA,
        "status": "awaiting-manual-intelligence-review",
        "candidate_id": candidate["candidate_id"],
        "mode": candidate["mode"],
        "candidate_type": candidate["candidate_type"],
        "checkpoint_manifest_sha256": manifest["manifest_sha256"],
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
            if executed
            else {"reference_integrity_notes": None}
        ),
        "fresh_executed_decode_is_primary_candidate_evidence": executed,
        "cached_bfloat16_text_is_reference_evidence_only": not executed,
        "exact_token_match_required": False,
        "numeric_metrics_have_decision_authority": False,
        "automatic_acceptance_or_rejection_performed": False,
        "candidate_runtime_executed": executed,
        "response_origin": (
            "fresh-fp32-protected-candidate-generation"
            if executed
            else "authenticated-bfloat16-parent-cache"
        ),
        "response_runtime_dtype": "float32" if executed else "bfloat16",
        "fresh_cache_created_per_generated_prompt": executed,
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
        "candidate_ids": list(candidate_ids),
        "completed_candidate_ids": [row["candidate_id"] for row in completed],
        "completed_count": len(completed),
        "total_count": len(candidate_ids),
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
    expected_a0019_source_run_result_sha256: str,
    expected_a0019_source_run_file_sha256: str,
    producer_receipt: Path,
    expected_producer_receipt_sha256: str,
    expected_producer_receipt_file_sha256: str,
    protected_plan: Path,
    expected_protected_plan_sha256: str,
    expected_protected_plan_file_sha256: str,
    protected_archive: Path,
    expected_protected_archive_file_sha256: str,
    prompts_json: Path,
    expected_prompts_file_sha256: str,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    expected_parent_envelope_file_sha256: str,
    device: str,
    maximum_prompt_tokens: int = 1024,
    maximum_new_tokens: int = 256,
    checkpoint_retention: str = "receipts-only",
    shard_count: int = 1,
    shard_index: int = 0,
) -> Mapping[str, Any]:
    """Run one authenticated full schedule or deterministic modulo shard."""

    import torch

    if maximum_prompt_tokens != 1024 or maximum_new_tokens != 256:
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected diagnostics require the sealed 1024-input/256-output horizon."
        )
    if checkpoint_retention not in {"receipts-only", "last"}:
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "checkpoint_retention must be receipts-only or last."
        )
    run_root = _new_directory(run_directory, label="run directory")
    parent_path = _ordinary_directory(parent_checkpoint, label="parent checkpoint")
    final_path = _ordinary_directory(final_checkpoint, label="final checkpoint")
    component_root = _ordinary_directory(
        component_publication_directory, label="component publication directory"
    )
    prompt_path = _ordinary_file(prompts_json, label="prompt bank")
    envelope_path = _ordinary_file(parent_envelope, label="cached parent envelope")
    publication_path = _ordinary_file(
        producer_receipt, label="protected producer receipt"
    )
    plan_path = _ordinary_file(protected_plan, label="protected plan")
    archive_path = _ordinary_file(protected_archive, label="protected archive")
    _forbid_run_inside_authenticated_inputs(
        run_root,
        input_directories=(
            parent_path,
            final_path,
            component_root,
            publication_path.parent,
            plan_path.parent,
            archive_path.parent,
            prompt_path.parent,
            envelope_path.parent,
        ),
    )
    if _file_sha256(prompt_path) != _digest(
        expected_prompts_file_sha256, label="expected prompt bank"
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError("Prompt bank bytes changed.")
    publication = _load_publication(
        publication_path,
        expected_receipt_sha256=expected_producer_receipt_sha256,
        expected_file_sha256=expected_producer_receipt_file_sha256,
    )
    plan = _load_plan(
        plan_path,
        expected_plan_sha256=expected_protected_plan_sha256,
        expected_file_sha256=expected_protected_plan_file_sha256,
    )
    _authenticate_publication(
        receipt=publication,
        receipt_path=publication_path,
        plan=plan,
        plan_path=plan_path,
        archive_path=archive_path,
        expected_plan_sha256=expected_protected_plan_sha256,
        expected_plan_file_sha256=expected_protected_plan_file_sha256,
        expected_archive_file_sha256=expected_protected_archive_file_sha256,
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
    source_auth = plan["source_authentication"]
    if (
        source_auth["a0019_source_run_result_sha256"]
        != _digest(
            expected_a0019_source_run_result_sha256,
            label="expected A0019 source run result",
        )
        or source_auth["a0019_source_run_receipt_file_sha256"]
        != _digest(
            expected_a0019_source_run_file_sha256,
            label="expected A0019 source run receipt file",
        )
        or source_auth["component_producer_receipt_sha256"]
        != component_receipt["receipt_sha256"]
        or source_auth["component_plan_sha256"] != component_plan["plan_sha256"]
        or source_auth["component_bridges_file_sha256"]
        != _file_sha256(component_bridge_path)
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Fresh A0019/B038 authentication differs from protected ancestry."
        )
    parent_inventory_sha = canonical_sha256(_checkpoint_inventory(parent_path))
    final_inventory_sha = canonical_sha256(_checkpoint_inventory(final_path))
    if (
        parent_inventory_sha
        != _digest(
            expected_parent_checkpoint_inventory_sha256,
            label="expected parent inventory",
        )
        or final_inventory_sha
        != _digest(
            expected_final_checkpoint_inventory_sha256,
            label="expected final inventory",
        )
        or parent_inventory_sha != plan["parent_checkpoint_inventory_sha256"]
        or final_inventory_sha != plan["final_checkpoint_inventory_sha256"]
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Parent/final inventory differs from protected ancestry."
        )
    parent_cache_source = load_lfm2_direct_local_model(
        parent_path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    binding = plan["parent_cache_binding"]
    if (
        fingerprint_lfm2_direct_local_state(parent_cache_source)
        != binding["parent_state_sha256"]
        or fingerprint_lfm2_parent(parent_cache_source).identity_sha256
        != binding["parent_identity_sha256"]
        or _unique_parameter_elements(parent_cache_source)
        != binding["parent_unique_parameters"]
        or fingerprint_local_tokenizer(parent_path) != binding["tokenizer_sha256"]
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Fresh BF16 parent/tokenizer differs from protected cache binding."
        )
    del parent_cache_source
    final_model = load_lfm2_direct_local_model(
        final_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if (
        fingerprint_lfm2_direct_local_state(final_model)
        != plan["final_model_state_sha256"]
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Fresh exact4x checkpoint differs from protected plan."
        )
    del final_model
    archive_tensors = _load_tensor_archive(
        archive_path,
        expected_file_sha256=expected_protected_archive_file_sha256,
        records=plan["archive"]["tensor_records"],
    )
    resolved_device = torch.device(device)
    parent_template = load_lfm2_direct_local_model(
        parent_path,
        device=resolved_device,
        dtype=torch.float32,
    )
    if (
        fingerprint_lfm2_direct_local_state(parent_template)
        != plan["parent_model_state_sha256"]
    ):
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Direct-local parent state differs from protected plan."
        )
    del parent_template

    prompt_rows = [dict(value) for value in load_review_prompts(prompt_path)]
    if len(prompt_rows) != 16:
        raise Lfm2ProtectedShortConvDiagnosticRunError(
            "Protected diagnostics require the complete sealed 16-prompt bank."
        )
    prompt_set_sha = canonical_sha256(prompt_rows)
    parent_cache = _load_parent_envelope(
        envelope_path,
        expected_envelope_sha256=_digest(
            expected_parent_envelope_sha256, label="expected parent envelope"
        ),
        expected_file_sha256=_digest(
            expected_parent_envelope_file_sha256,
            label="expected parent envelope file",
        ),
        expected_prompt_set_sha256=prompt_set_sha,
        expected_prompt_ids=[str(row["id"]) for row in prompt_rows],
    )
    cache_authentication = _authenticate_parent_cache_binding(
        parent_cache,
        plan=plan,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    parent_rows = {str(row["id"]): row for row in parent_cache["responses"]}
    tokenizer = load_lfm2_direct_local_tokenizer(parent_path)
    shard = _schedule_shard(
        plan["schedule"], shard_count=shard_count, shard_index=shard_index
    )
    schedule_rows = _authenticate_schedule_shard(plan["schedule"], shard)
    candidate_ids = [row["candidate_id"] for row in schedule_rows]
    run_root.mkdir()
    checkpoint_root = run_root / "differential-checkpoints"
    review_root = run_root / "manual-review"
    checkpoint_root.mkdir()
    review_root.mkdir()
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
            raise Lfm2ProtectedShortConvDiagnosticRunError(
                "Fresh parent shell changed between candidates."
            )
        installation = _install_candidate(
            model=model,
            plan=plan,
            schedule_candidate=schedule_candidate,
            tensors=archive_tensors,
            device=resolved_device,
        )
        checkpoint = checkpoint_root / schedule_candidate["candidate_id"]
        manifest = _write_differential_checkpoint(
            destination=checkpoint,
            model=model,
            plan=plan,
            publication=publication,
            schedule_candidate=schedule_candidate,
            installation=installation,
            archive_tensors=archive_tensors,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reloaded, authenticated_manifest = _fresh_load_differential_checkpoint(
            source=checkpoint,
            expected_manifest_sha256=manifest["manifest_sha256"],
            parent_checkpoint=parent_path,
            plan=plan,
            publication=publication,
            device=resolved_device,
        )
        executed = schedule_candidate["candidate_id"] != "cached-parent-reference"
        if executed:
            responses = _generate_rows(
                reloaded,
                tokenizer,
                prompt_rows,
                maximum_prompt_tokens=1024,
                maximum_new_tokens=256,
            )
            _attach_parent(responses, parent_rows=parent_rows)
            generated_candidates += 1
        else:
            responses = _cached_parent_reference_rows(
                prompt_rows, parent_rows=parent_rows
            )
            cached_parent_references += 1
        review = _manual_review_document(
            candidate=schedule_candidate,
            manifest=authenticated_manifest,
            responses=responses,
        )
        review_path = review_root / f"{schedule_candidate['candidate_id']}.json"
        _write_json(review_path, review)
        record = {
            "candidate_id": schedule_candidate["candidate_id"],
            "mode": schedule_candidate["mode"],
            "candidate_type": schedule_candidate["candidate_type"],
            "checkpoint_manifest_sha256": authenticated_manifest["manifest_sha256"],
            "installation_receipt": authenticated_manifest["installation_receipt"],
            "fresh_fp32_parent_reconstruction_verified": True,
            "candidate_runtime_executed": executed,
            "runtime_decode_performed": executed,
            "fresh_cache_per_candidate_and_prompt": executed,
            "cached_parent_reference_reused": not executed,
            "response_origin": (
                "fresh-fp32-protected-candidate-generation"
                if executed
                else "authenticated-bfloat16-parent-cache"
            ),
            "response_runtime_dtype": "float32" if executed else "bfloat16",
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
                shard=shard,
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
    _write_json(review_root / "index.json", review_index)
    body = {
        "schema": RUN_SCHEMA,
        "status": (
            "NON-FINAL-PROTECTED-SHORT-CONV-FULL-REVIEW-EVIDENCE-COMPLETE"
            if shard_count == 1
            else "NON-FINAL-PROTECTED-SHORT-CONV-PARTIAL-REVIEW-EVIDENCE-COMPLETE"
        ),
        "producer_receipt_sha256": publication["receipt_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": plan["schedule"]["schedule_sha256"],
        "schedule_shard": _plain(shard),
        "partial_sweep": shard["partial_sweep"],
        "parent_model_state_sha256": plan["parent_model_state_sha256"],
        "final_model_state_sha256": plan["final_model_state_sha256"],
        "prompt_set_sha256": prompt_set_sha,
        "parent_cache_authentication": _plain(cache_authentication),
        "prompt_count": 16,
        "maximum_prompt_tokens": 1024,
        "maximum_new_tokens": 256,
        "candidate_ids": candidate_ids,
        "reviewed_candidate_count": len(completed),
        "candidate_reviews": completed,
        "manual_review_index": review_index,
        "checkpoint_retention": checkpoint_retention,
        "retained_checkpoint_paths": [] if retained is None else [str(retained)],
        "evaluation_policy": {
            "complete_sealed_16_prompt_bank_required": True,
            "fresh_executed_candidate_decodes_are_primary_evidence": True,
            "cached_bfloat16_parent_is_reference_evidence_only": True,
            "exact_token_match_required": False,
            "numeric_metrics_have_decision_authority": False,
            "manual_intelligence_review_required": True,
            "automatic_acceptance_or_rejection_performed": False,
        },
        "interpretation": {
            "protected_parent_channel_algebra_isolated": True,
            "layer_zero_input_output_and_both_roundtrips_present_in_full_plan": True,
            "layer_zero_dense_compact_negative_control_present_in_full_plan": True,
            "layer_one_both_roundtrip_control_present_in_full_plan": True,
            "cached_native_parent_reference_present_in_full_plan": True,
            "differential_save_and_fresh_reload_verified": True,
            "hidden_sidecars_present": False,
            "explicit_counted_interface_adapters_present_only_for_standalone_control": True,
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
    parser.add_argument("--expected-a0019-source-run-result-sha256", required=True)
    parser.add_argument("--expected-a0019-source-run-file-sha256", required=True)
    parser.add_argument("--producer-receipt", type=Path, required=True)
    parser.add_argument("--expected-producer-receipt-sha256", required=True)
    parser.add_argument("--expected-producer-receipt-file-sha256", required=True)
    parser.add_argument("--protected-plan", type=Path, required=True)
    parser.add_argument("--expected-protected-plan-sha256", required=True)
    parser.add_argument("--expected-protected-plan-file-sha256", required=True)
    parser.add_argument("--protected-archive", type=Path, required=True)
    parser.add_argument("--expected-protected-archive-file-sha256", required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument(
        "--checkpoint-retention",
        choices=("receipts-only", "last"),
        default="receipts-only",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(**vars(arguments))
    print(json.dumps(_plain(result), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
