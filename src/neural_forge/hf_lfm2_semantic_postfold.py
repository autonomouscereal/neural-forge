"""Transactional no-training semantic causal-stack postfold.

This module resumes from a *saved standalone* CPCF-v5 LFM2 child.  It never
constructs or executes the parent model.  Instead it:

* authenticates the dense child artifact and the complete causal trace cache;
* authenticates legacy c61 or fast semantic-foundation ancestry and
  direct-local parent Q/K/V/O bytes;
* builds per-phase donor/target semantic-span attention evidence;
* requires causal attention/short-conv/replay-fresh-MLP installation in order;
* defers the shared final-layer W2 writer;
* jointly solves final-MLP, parent-terminal, and lexical-continuation evidence;
* atomically publishes and freshly reloads a new dense standalone checkpoint.

The module does not equate an attention-only correction with a coherent child.
Until a complete causal layer-stack adapter exists, the default path fails
closed before final-W2 mutation or checkpoint publication and reports both the
long-horizon short-conv and replay-fresh MLP gaps.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge import dynamic_lfm2_cpcf_resources as cpcf_resources
from neural_forge import hf_lfm2_cpcf_model_compiler as cpcf_model_compiler
from neural_forge import hf_lfm2_cpcf_model_fold as cpcf_model_fold
from neural_forge import hf_lfm2_cpcf_v5_model_compiler as cpcf_v5_compiler
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_cpcf_v5_quarantine import (
    LFM2_CPCF_V5_ARTIFACT_METADATA,
    LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
    Lfm2CpcfV5QuarantineSaver,
    Lfm2CpcfV5StrictLocalLoader,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    Lfm2CpcfModelFoldPolicy,
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_cpcf_macrocell_replay import (
    replay_lfm2_cpcf_macrocell,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import _rope_from_target
from neural_forge.hf_lfm2_cpcf_selective_evidence import (
    open_lfm2_cpcf_selective_evidence_provider_from_catalog,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionEvidence,
    Lfm2CpcfTerminalActionPolicy,
    Lfm2CpcfTerminalActionResidualResult,
    Lfm2CpcfTerminalActionSeed,
    _all_passed,
    _evaluate_split,
    _metric_gates,
    _prepare_split,
    compile_lfm2_cpcf_terminal_action_residual,
)
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    Lfm2DenseQkvoSourceInventory,
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_long_horizon_terminal_refold import (
    Lfm2LongHorizonTrajectorySplitPlan,
    Lfm2LocalParentFinalRmsNormLoad,
    _build_split_evidence,
    _collect_cached_semantic_trajectories,
    _target_runtime_seed,
    load_lfm2_parent_final_rmsnorm_from_local_safetensors,
)
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (
    Lfm2SemanticAttentionReplayEndpoint,
)
from neural_forge.hf_lfm2_semantic_foundation import (
    LFM2_SEMANTIC_FOUNDATION_LINEAGE_SCHEMA,
    LFM2_SEMANTIC_FOUNDATION_SCHEMA,
    Lfm2SemanticFoundationError,
    authenticate_lfm2_semantic_foundation_receipt,
    rehydrate_lfm2_semantic_foundation_interface,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
    Lfm2SemanticSpanCpcfAlignment,
    Lfm2SemanticSpanCpcfPhaseAlignment,
    audit_lfm2_semantic_span_cpcf_refinement,
    build_lfm2_semantic_span_cpcf_alignment,
)
from neural_forge.hf_lfm2_semantic_span_phase_bridge import (
    _decode_exact,
    _encode_exact,
    build_lfm2_semantic_span_selected_input_evidence,
    build_lfm2_semantic_span_phase_bridge,
)
from neural_forge.hf_semantic_attention_lattice import (
    Lfm2SemanticAttentionCacheCapability,
    SemanticAttentionExpansionTransport,
    SemanticAttentionPreRopeHistory,
    assemble_lfm2_explicit_pre_rope_history,
    inspect_lfm2_semantic_attention_cache_capability,
    transport_lfm2_phase_probability_preserving_attention,
)
from neural_forge.hf_tied_causal_interface_fold import TiedCausalInterfaceFold
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


LFM2_SEMANTIC_POSTFOLD_SCHEMA = "neural-forge-lfm2-semantic-postfold-v1"
LFM2_SEMANTIC_ATTENTION_STAGE_SCHEMA = "neural-forge-lfm2-semantic-attention-stage-v1"
LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-semantic-attention-installation-v1"
)
LFM2_SEMANTIC_POSTFOLD_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-semantic-postfold-checkpoint-v1"
)
LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA = "neural-forge-lfm2-semantic-postfold-failure-v1"
LFM2_SEMANTIC_COMPILER_LINEAGE_SCHEMA = "neural-forge-lfm2-semantic-compiler-lineage-v1"
LFM2_SEMANTIC_RESIDUAL_CHART_SCHEMA = "neural-forge-lfm2-semantic-residual-chart-v1"
LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA = (
    "neural-forge-lfm2-semantic-lexical-completion-v1"
)
SEMANTIC_ATTENTION_INSTALLER_CALLABLE = (
    "compile_install_lfm2_semantic_attention_postfold"
)
SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE = (
    "compile_install_lfm2_semantic_layer_stack_postfold"
)

_ZERO_OPERATIONS = MappingProxyType(
    {
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "child_model_forward_calls": 0,
        "child_model_generate_calls": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
)


class Lfm2SemanticPostfoldError(RuntimeError):
    """A sealed input, analytic fold, or checkpoint transaction failed."""


class Lfm2SemanticPostfoldBlockedError(Lfm2SemanticPostfoldError):
    """The exact no-training path is unavailable and published nothing."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = MappingProxyType(dict(report))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic postfold requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticPostfoldError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(block)
    except OSError as exc:
        raise Lfm2SemanticPostfoldError(
            f"Could not hash authenticated local file {path}."
        ) from exc
    return digest.hexdigest()


def _sealed(body: Mapping[str, Any], seal_name: str) -> dict[str, Any]:
    if seal_name in body:
        raise Lfm2SemanticPostfoldError(f"Receipt body already contains {seal_name!r}.")
    value = dict(body)
    value[seal_name] = canonical_sha256(value)
    return value


def _verify_sealed(
    value: Mapping[str, Any],
    *,
    seal_name: str,
    label: str,
) -> dict[str, Any]:
    result = dict(value)
    supplied = result.pop(seal_name, None)
    if supplied != canonical_sha256(result):
        raise Lfm2SemanticPostfoldError(f"{label} self-seal is invalid.")
    return {**result, seal_name: supplied}


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2SemanticPostfoldError(
            "Cannot authenticate a non-materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _interface_tensor_sha256(value: Any) -> str:
    """Match ``hf_tied_causal_interface_fold`` tensor authentication."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for value in model.parameters():
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += int(value.numel())
    return total


def _state_tensor_hashes(model: Any) -> dict[str, str]:
    return {
        name: _tensor_sha256(value)
        for name, value in sorted(model.state_dict().items())
    }


_DENSE_PRODUCER_WEIGHT_ROLES = MappingProxyType(
    {
        "full_attention": (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.out_proj.weight",
        ),
        "short_conv": (
            "conv.in_proj.weight",
            "conv.conv.weight",
            "conv.out_proj.weight",
        ),
        "mlp": (
            "feed_forward.w1.weight",
            "feed_forward.w3.weight",
            "feed_forward.w2.weight",
        ),
    }
)


def _dense_weight_structural_coverage(
    value: Any,
    *,
    parameter_name: str,
    require_parameter: bool = True,
) -> Mapping[str, Any]:
    """Authenticate that a dense producer spans every output and input axis.

    This is deliberately an exact structural gate, not a quality metric.  A
    dense folded weight is considered planted only when it is finite, contains
    a nonzero coefficient, and leaves no output row/channel or input/lag column
    structurally disconnected.  It does not require parent/child numeric weight
    equality and it does not impose an arbitrary magnitude threshold.
    """

    torch = _torch()
    if (
        (require_parameter and not isinstance(value, torch.nn.Parameter))
        or not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or value.ndim < 2
        or value.numel() <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            f"{parameter_name} is not a materialized dense floating "
            + ("Parameter." if require_parameter else "tensor.")
        )
    tensor = value.detach().to(device="cpu").contiguous()
    finite = bool(torch.isfinite(tensor.to(dtype=torch.float32)).all().item())
    nonzero = tensor.ne(0)
    nonzero_count = int(nonzero.sum().item())
    output_units = int(tensor.shape[0])
    input_units = int(tensor.shape[-1])
    covered_output_units = int(
        nonzero.reshape(output_units, -1).any(dim=1).sum().item()
    )
    covered_input_units = int(
        nonzero.reshape(-1, input_units).any(dim=0).sum().item()
    )
    body = {
        "schema": "neural-forge-dense-weight-structural-coverage-v1",
        "parameter_name": parameter_name,
        "ordinary_parameter_required": require_parameter,
        "ordinary_parameter": isinstance(value, torch.nn.Parameter),
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "tensor_sha256": _tensor_sha256(tensor),
        "element_count": int(tensor.numel()),
        "nonzero_element_count": nonzero_count,
        "nonzero_element_fraction": nonzero_count / int(tensor.numel()),
        "finite": finite,
        "output_structural_unit_count": output_units,
        "covered_output_structural_unit_count": covered_output_units,
        "input_structural_unit_count": input_units,
        "covered_input_structural_unit_count": covered_input_units,
        "all_output_structural_units_covered": (
            covered_output_units == output_units
        ),
        "all_input_structural_units_covered": covered_input_units == input_units,
        "structurally_nonzero": bool(
            finite
            and nonzero_count > 0
            and covered_output_units == output_units
            and covered_input_units == input_units
        ),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _lfm2_dense_producer_chain_coverage(
    target_model: Any,
    *,
    layer_index: int,
    producer_kind: str,
    defer_down_weight: bool = False,
) -> Mapping[str, Any]:
    """Seal structural coverage of one existing dense LFM2 producer chain.

    The final MLP may defer only W2.  Its W1/W3 producer must still be planted
    before terminal evidence can be trusted; the deferred W2 bytes are recorded
    and authenticated but are not counted as required until terminal publish.
    """

    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or producer_kind not in _DENSE_PRODUCER_WEIGHT_ROLES
        or (defer_down_weight and producer_kind != "mlp")
    ):
        raise Lfm2SemanticPostfoldError(
            "Dense producer-chain coverage request is invalid."
        )
    roles = tuple(_DENSE_PRODUCER_WEIGHT_ROLES[producer_kind])
    deferred_roles = (
        ("feed_forward.w2.weight",) if defer_down_weight else ()
    )
    required_roles = tuple(role for role in roles if role not in deferred_roles)
    named_rows = list(target_model.named_parameters(remove_duplicate=False))
    named: dict[str, Any] = {}
    duplicate_names: set[str] = set()
    for name, parameter in named_rows:
        if name in named:
            duplicate_names.add(name)
        named[name] = parameter
    receipts: dict[str, Mapping[str, Any]] = {}
    for role in roles:
        parameter_name = f"model.layers.{layer_index}.{role}"
        if parameter_name in duplicate_names or parameter_name not in named:
            raise Lfm2SemanticPostfoldError(
                f"Dense producer Parameter {parameter_name} is missing or ambiguous."
            )
        receipts[role] = _dense_weight_structural_coverage(
            named[parameter_name],
            parameter_name=parameter_name,
        )
    role_hashes = {
        role: receipts[role]["tensor_sha256"] for role in roles
    }
    body = {
        "schema": "neural-forge-lfm2-dense-producer-chain-coverage-v1",
        "layer_index": layer_index,
        "producer_kind": producer_kind,
        "required_roles": list(required_roles),
        "deferred_roles": list(deferred_roles),
        "role_receipts": {role: dict(receipts[role]) for role in roles},
        "required_role_state_sha256": canonical_sha256(
            {role: role_hashes[role] for role in required_roles}
        ),
        "all_role_state_sha256": canonical_sha256(role_hashes),
        "required_role_count": len(required_roles),
        "structurally_nonzero_required_role_count": sum(
            int(receipts[role]["structurally_nonzero"])
            for role in required_roles
        ),
        "blank_or_partially_blank_required_role_count": sum(
            int(not receipts[role]["structurally_nonzero"])
            for role in required_roles
        ),
        "all_required_roles_structurally_nonzero": all(
            receipts[role]["structurally_nonzero"] for role in required_roles
        ),
        "dense_existing_topology_only": True,
        "numeric_parent_child_weight_match_objective": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _changed_state_keys(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[str, ...]:
    if set(before) != set(after):
        raise Lfm2SemanticPostfoldError(
            "Analytic attention installation changed the child state topology."
        )
    return tuple(name for name in before if before[name] != after[name])


def _parameter_runtime_snapshot(model: Any) -> Mapping[str, Any]:
    """Capture topology, object/storage identity, and exact gradient state."""

    snapshot: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        snapshot[name] = {
            "shape": tuple(int(size) for size in parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "parameter_id": id(parameter),
            "storage_data_ptr": int(parameter.data_ptr()),
            "requires_grad": bool(parameter.requires_grad),
            "gradient": (
                None
                if gradient is None
                else {
                    "shape": tuple(int(size) for size in gradient.shape),
                    "dtype": str(gradient.dtype),
                    "device": str(gradient.device),
                    "tensor_id": id(gradient),
                    "storage_data_ptr": int(gradient.data_ptr()),
                    "sha256": _tensor_sha256(gradient),
                }
            ),
        }
    return MappingProxyType(snapshot)


def _parameter_runtime_proof(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Mapping[str, Any]:
    before_names = tuple(sorted(before))
    after_names = tuple(sorted(after))
    topology_fields = ("shape", "dtype", "device")
    topology_preserved = bool(
        before_names == after_names
        and all(
            tuple(before[name][field] for field in topology_fields)
            == tuple(after[name][field] for field in topology_fields)
            for name in before_names
        )
    )
    identities_preserved = bool(
        topology_preserved
        and all(
            before[name]["parameter_id"] == after[name]["parameter_id"]
            for name in before_names
        )
    )
    storage_preserved = bool(
        topology_preserved
        and all(
            before[name]["storage_data_ptr"] == after[name]["storage_data_ptr"]
            for name in before_names
        )
    )
    gradient_state_preserved = bool(
        topology_preserved
        and all(
            before[name]["requires_grad"] == after[name]["requires_grad"]
            and before[name]["gradient"] == after[name]["gradient"]
            for name in before_names
        )
    )
    topology_body = {
        name: {field: before[name][field] for field in topology_fields}
        for name in before_names
    }
    after_topology_body = {
        name: {field: after[name][field] for field in topology_fields}
        for name in after_names
    }
    return MappingProxyType(
        {
            "parameter_topology_sha256_before": canonical_sha256(topology_body),
            "parameter_topology_sha256_after": canonical_sha256(after_topology_body),
            "parameter_topology_preserved": topology_preserved,
            "parameter_identities_preserved": identities_preserved,
            "parameter_storage_preserved": storage_preserved,
            "gradient_state_preserved": gradient_state_preserved,
        }
    )


def _ordinary_directory(value: str | os.PathLike[str], name: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{name} must not be a symlink.")
    path = lexical.resolve()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{name} must be a real local directory.")
    return path


def _ordinary_file(value: str | os.PathLike[str], name: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{name} must not be a symlink.")
    path = lexical.resolve()
    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{name} must be a real local file.")
    return path


def _new_output_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise Lfm2SemanticPostfoldError(
            f"Refusing to replace existing output checkpoint {path}."
        )
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2SemanticPostfoldError(
            "Output checkpoint parent must be a real existing directory."
        )
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticPostfoldError(f"{label} is missing or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticPostfoldError(f"Could not parse {label}.") from exc
    if not isinstance(value, dict):
        raise Lfm2SemanticPostfoldError(f"{label} must contain a JSON object.")
    return value


def _read_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    """Read one canonical runner artifact and retain its exact file identity."""

    path = _ordinary_file(path, label)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticPostfoldError(f"Could not parse {label}.") from exc
    if not isinstance(value, dict):
        raise Lfm2SemanticPostfoldError(f"{label} must contain a JSON object.")
    expected = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if raw != expected:
        raise Lfm2SemanticPostfoldError(f"{label} is not canonical JSON.")
    return value, hashlib.sha256(raw).hexdigest()


def _authenticated_c61_diagnostic(
    diagnostic_json: str | os.PathLike[str],
    *,
    expected_diagnostic_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    path = _ordinary_file(diagnostic_json, "c61 compiler diagnostic")
    diagnostic, file_sha256 = _read_canonical_json(
        path,
        "c61 compiler diagnostic",
    )
    expected = _digest(
        expected_diagnostic_sha256,
        "expected c61 compiler diagnostic SHA-256",
    )
    verified = _verify_sealed(
        diagnostic,
        seal_name="receipt_sha256",
        label="c61 compiler diagnostic",
    )
    if (
        verified["receipt_sha256"] != expected
        or verified.get("schema")
        != "neural-forge-lfm2-cpcf-v5-selective-blank-fold-run-v1"
        or verified.get("status") != "compiled_in_memory_diagnostic_not_published"
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 compiler diagnostic differs from its externally pinned receipt."
        )
    return MappingProxyType(verified), file_sha256


def _claims_semantic_foundation_schema(
    ancestry_json: str | os.PathLike[str],
) -> bool:
    """Peek only for dispatch; the selected authenticator remains authoritative."""

    try:
        path = Path(ancestry_json).expanduser().resolve()
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, Mapping)
        and value.get("schema") == LFM2_SEMANTIC_FOUNDATION_SCHEMA
    )


def _input_artifact_metadata(
    checkpoint: Path,
    *,
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    expected = _digest(expected_artifact_sha256, "expected input artifact SHA-256")
    value = _read_json(
        checkpoint / LFM2_CPCF_V5_ARTIFACT_METADATA,
        "standalone CPCF-v5 artifact metadata",
    )
    body = dict(value)
    supplied = body.pop("artifact_sha256", None)
    if (
        supplied != expected
        or supplied != canonical_sha256(body)
        or body.get("schema") != LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA
        or body.get("dense_safetensors_only") is not True
        or body.get("target_tokenizer_saved_with_child") is not True
        or body.get("local_reload_only") is not True
        or body.get("network_access") is not False
        or body.get("hub_loader_used") is not False
    ):
        raise Lfm2SemanticPostfoldError(
            "Input child artifact differs from its externally rooted identity."
        )
    return {**body, "artifact_sha256": supplied}


def load_lfm2_cpcf_v5_standalone_child(
    checkpoint_directory: str | os.PathLike[str],
    *,
    expected_artifact_sha256: str,
    device: str,
    dtype: str = "bfloat16",
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Strictly load one saved dense child through the existing local contract."""

    checkpoint = _ordinary_directory(checkpoint_directory, "input checkpoint")
    metadata = _input_artifact_metadata(
        checkpoint,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    inventory = metadata.get("parameter_inventory")
    if not isinstance(inventory, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Input artifact has no authenticated parameter inventory."
        )
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=_digest(
            metadata.get("context_binding_sha256"),
            "input context binding",
        ),
        device=device,
        dtype=dtype,
        expected_ffn_width=int(metadata.get("materialized_ffn_width", 0)),
        expected_model_state_sha256=_digest(
            metadata.get("model_state_sha256"),
            "input model state",
        ),
        expected_parameter_inventory_sha256=_digest(
            metadata.get("parameter_inventory_sha256"),
            "input parameter inventory",
        ),
        expected_tokenizer_semantics_sha256=_digest(
            metadata.get("tokenizer_semantics_sha256"),
            "input tokenizer semantics",
        ),
    )
    model = loader(checkpoint)
    tokenizer = getattr(model, "_neural_forge_quarantine_tokenizer", None)
    if tokenizer is None:
        raise Lfm2SemanticPostfoldError(
            "Strict local child reload did not retain its target tokenizer."
        )
    return model, tokenizer, MappingProxyType(metadata)


def _local_safetensor_shards(directory: Path) -> tuple[Path, ...]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file() and not index_path.is_symlink():
        index = _read_json(index_path, "parent safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Lfm2SemanticPostfoldError(
                "Parent safetensors index has no weight map."
            )
        shards = tuple(directory / name for name in sorted(set(weight_map.values())))
    else:
        single = directory / "model.safetensors"
        shards = (
            (single,)
            if single.is_file()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not shards or any(not path.is_file() or path.is_symlink() for path in shards):
        raise Lfm2SemanticPostfoldError(
            "Parent direct-local safetensors are incomplete or unsafe."
        )
    return shards


def load_lfm2_mapped_parent_tied_table_from_local_safetensors(
    model_directory: str | os.PathLike[str],
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    authenticated_parent_final_norm: Lfm2LocalParentFinalRmsNormLoad,
    tensor_key: str = "model.embed_tokens.weight",
) -> tuple[Any, Mapping[str, Any]]:
    """Read only the mapped tied rows; never construct the parent module."""

    torch = _torch()
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Direct-local tied-table loading needs safetensors."
        ) from exc
    directory = _ordinary_directory(model_directory, "parent model directory")
    config = _read_json(directory / "config.json", "parent LFM2 config")
    if (
        config.get("model_type") != "lfm2"
        or config.get("tie_word_embeddings") is not True
        or not authenticated_parent_final_norm.final_norm.verify_authentication()
        or authenticated_parent_final_norm.final_norm.cache_key != cache.cache_key
    ):
        raise Lfm2SemanticPostfoldError(
            "Parent tied-table source is not bound to the authenticated cache."
        )
    selected = None
    source_shard = None
    for shard in _local_safetensor_shards(directory):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if tensor_key in handle.keys():
                if selected is not None:
                    raise Lfm2SemanticPostfoldError(
                        "Parent tied table appears in more than one shard."
                    )
                selected = handle.get_tensor(tensor_key)
                source_shard = shard
    if selected is None or selected.ndim != 2:
        raise Lfm2SemanticPostfoldError(
            f"Parent checkpoint lacks tied table {tensor_key!r}."
        )
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    if (
        not mapping
        or max(mapping) >= int(selected.shape[0])
        or int(selected.shape[0]) != bridge.transport_profile.donor_row_capacity
    ):
        raise Lfm2SemanticPostfoldError(
            "Tokenizer bridge does not address the authenticated parent table."
        )
    rows = selected.index_select(0, torch.tensor(mapping, dtype=torch.long))
    rows = rows.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
    body = {
        "schema": "neural-forge-lfm2-direct-local-mapped-tied-table-v1",
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "parent_final_norm_load_sha256": (
            authenticated_parent_final_norm.result_sha256
        ),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "tensor_key": tensor_key,
        "source_shard_name": source_shard.name,
        "target_to_donor_rows_sha256": canonical_sha256(list(mapping)),
        "mapped_table_sha256": _interface_tensor_sha256(rows.to(torch.float64)),
        "mapped_table_shape": [int(size) for size in rows.shape],
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "network_access_performed": False,
    }
    return rows, MappingProxyType(_sealed(body, "result_sha256"))


def reconstruct_lfm2_saved_tied_interface(
    target_model: Any,
    source_selected_tied_table: Any,
) -> TiedCausalInterfaceFold:
    """Rebuild the analytical producer map from the saved native tied table.

    CPCF's mean-preserving tied fold stores a constant softmax-invariant first
    column and scaled vocabulary basis columns.  The checkpoint needs only the
    target table for inference; this inverse reconstructs the producer map
    needed by the later terminal compiler without consulting old run memory.
    """

    torch = _torch()
    source = source_selected_tied_table
    target = getattr(target_model.get_input_embeddings(), "weight", None)
    output = getattr(target_model.get_output_embeddings(), "weight", None)
    if (
        not isinstance(source, torch.Tensor)
        or source.ndim != 2
        or source.dtype != torch.bfloat16
        or not isinstance(target, torch.nn.Parameter)
        or output is not target
        or target.ndim != 2
        or int(source.shape[0]) != int(target.shape[0])
        or int(target.shape[1]) < 2
    ):
        raise Lfm2SemanticPostfoldError(
            "Saved child/source tables cannot reconstruct a tied interface."
        )
    source64 = source.detach().to(device="cpu", dtype=torch.float64).contiguous()
    target64 = target.detach().to(device="cpu", dtype=torch.float64).contiguous()
    first = target64[:, 0]
    component_rms = float(first.mean().item())
    constant_error = float((first - component_rms).abs().max().item())
    if (
        not math.isfinite(component_rms)
        or component_rms <= 0.0
        or constant_error != 0.0
    ):
        raise Lfm2SemanticPostfoldError(
            "Saved target tied table lacks CPCF's exact mean coordinate."
        )
    vocabulary = int(target64.shape[0])
    column_scale = component_rms * math.sqrt(vocabulary)
    vocabulary_vectors = target64[:, 1:] / column_scale
    mean = source64.mean(dim=0)
    centered = source64 - mean
    encoder = torch.cat(
        (
            (mean / component_rms).unsqueeze(1),
            centered.T @ vocabulary_vectors / column_scale,
        ),
        dim=1,
    ).contiguous()
    if tuple(encoder.shape) != (int(source.shape[1]), int(target.shape[1])) or not bool(
        torch.isfinite(encoder).all().item()
    ):
        raise Lfm2SemanticPostfoldError(
            "Reconstructed saved-interface producer map is invalid."
        )
    body = {
        "schema": "neural-forge-tied-causal-interface-resume-v1",
        "method": "inverse_saved_mean_coordinate_and_vocabulary_basis",
        "source_width": int(source.shape[1]),
        "target_width": int(target.shape[1]),
        "vocabulary_size": vocabulary,
        "source_table_sha256": _interface_tensor_sha256(source64),
        "target_weight_sha256": _interface_tensor_sha256(target64),
        "producer_encoder_sha256": _interface_tensor_sha256(encoder),
        "mean_coordinate_value": component_rms,
        "mean_coordinate_maximum_absolute_error": constant_error,
        "checkpoint_target_bytes_are_authoritative": True,
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    report = _sealed(body, "report_sha256")
    return TiedCausalInterfaceFold(
        target_weight=target64,
        producer_encoder=encoder,
        report=report,
    )


def rehydrate_lfm2_c61_tied_interface(
    target_model: Any,
    source_selected_tied_table: Any,
    *,
    semantic_cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_tokenizer: Any,
    compiler_diagnostic_json: str | os.PathLike[str],
    expected_compiler_diagnostic_sha256: str,
) -> tuple[TiedCausalInterfaceFold, Mapping[str, Any]]:
    """Replay c61's exact tied-interface compiler from its sealed evidence.

    Anchor repair changes the saved target table while deliberately keeping the
    producer encoder fixed.  Consequently the encoder cannot be inverted from
    an anchor-active checkpoint.  This routine instead warm-opens c61's exact
    compiler cache/catalog, loads only final-normalization evidence, and reruns
    the deterministic interface compiler.  Every resulting tensor and report
    is matched to both the c61 lineage and the saved dense child.
    """

    torch = _torch()
    diagnostic, diagnostic_file_sha256 = _authenticated_c61_diagnostic(
        compiler_diagnostic_json,
        expected_diagnostic_sha256=expected_compiler_diagnostic_sha256,
    )
    inputs = diagnostic.get("inputs")
    request = diagnostic.get("request")
    preparation_value = diagnostic.get("selective_catalog_artifact")
    compiled_value = diagnostic.get("compiled_model_fold")
    bundle_value = diagnostic.get("boundary_bundle")
    install_value = diagnostic.get("interface_install")
    seam = diagnostic.get("compiled_candidate_consumer_seam")
    if any(
        not isinstance(value, Mapping)
        for value in (
            inputs,
            request,
            preparation_value,
            compiled_value,
            bundle_value,
            install_value,
            seam,
        )
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 diagnostic lacks exact interface-rehydration ancestry."
        )
    preparation = _verify_sealed(
        preparation_value,
        seal_name="preparation_receipt_sha256",
        label="c61 selective-catalog preparation",
    )
    compiled = _verify_sealed(
        compiled_value,
        seal_name="result_sha256",
        label="c61 compiled-model receipt",
    )
    bundle = _verify_sealed(
        bundle_value,
        seal_name="bundle_sha256",
        label="c61 boundary bundle",
    )
    install = _verify_sealed(
        install_value,
        seal_name="report_sha256",
        label="c61 tied-interface installation",
    )
    compiler_cache_key = _digest(
        inputs.get("cache_key"),
        "c61 compiler cache key",
    )
    common_parent = _digest(
        inputs.get("parent_fingerprint"),
        "c61 parent fingerprint",
    )
    if (
        semantic_cache.parent_fingerprint != common_parent
        or inputs.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or seam.get("cache_sha256") != compiler_cache_key
        or seam.get("parent_sha256") != common_parent
        or seam.get("bridge_sha256") != bridge.artifact_sha256
        or preparation.get("cache_key") != compiler_cache_key
        or preparation.get("trace_cache_directory")
        != inputs.get("trace_cache_directory")
        or preparation.get("catalog_store_directory")
        != request.get("selective_catalog_store_directory")
        or inputs.get("selective_catalog_preparation_receipt_sha256")
        != preparation["preparation_receipt_sha256"]
        or inputs.get("selective_catalog_binding_sha256")
        != preparation.get("binding_sha256")
        or inputs.get("selective_catalog_artifact_manifest_sha256")
        != preparation.get("artifact_manifest_sha256")
        or inputs.get("selective_catalog_artifact_sha256")
        != preparation.get("artifact_catalog_sha256")
        or compiled.get("selective_catalog_sha256")
        != preparation.get("selective_catalog_sha256")
        or compiled.get("selective_source_sha256")
        != preparation.get("selective_source_sha256")
        or compiled.get("selective_provider_sha256")
        != preparation.get("selective_provider_sha256")
    ):
        raise Lfm2SemanticPostfoldError(
            "Compiler-cache, catalog, bridge, or parent ancestry diverged."
        )

    preparation_path = _ordinary_file(
        preparation.get("catalog_receipt_json"),
        "c61 selective-catalog preparation receipt",
    )
    preparation_disk, preparation_file_sha256 = _read_canonical_json(
        preparation_path,
        "c61 selective-catalog preparation receipt",
    )
    if preparation_disk != preparation:
        raise Lfm2SemanticPostfoldError(
            "On-disk selective-catalog receipt differs from c61's sealed copy."
        )
    compiler_cache_directory = _ordinary_directory(
        inputs.get("trace_cache_directory"),
        "c61 compiler trace cache",
    )
    catalog_store_directory = _ordinary_directory(
        preparation.get("catalog_store_directory"),
        "c61 selective-catalog store",
    )
    artifact_path = _ordinary_directory(
        preparation.get("artifact_path"),
        "c61 selective-catalog artifact",
    )
    try:
        artifact_path.relative_to(catalog_store_directory)
    except ValueError as exc:
        raise Lfm2SemanticPostfoldError(
            "c61 selective-catalog artifact lies outside its sealed store."
        ) from exc
    trajectory_path = _ordinary_file(
        inputs.get("trajectory_corpus"),
        "c61 trajectory corpus",
    )
    trajectory_sha256 = _file_sha256(trajectory_path)
    if trajectory_sha256 != _digest(
        inputs.get("trajectory_corpus_sha256"),
        "c61 trajectory corpus SHA-256",
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 trajectory corpus changed before interface rehydration."
        )

    compiler_cache = open_lfm2_circuit_trace_cache(
        compiler_cache_directory,
        verify=True,
    )
    if (
        compiler_cache.cache_key != compiler_cache_key
        or compiler_cache.parent_fingerprint != common_parent
    ):
        raise Lfm2SemanticPostfoldError(
            "The reopened c61 compiler cache differs from its sealed identity."
        )
    rows = cpcf_resources._read_trajectory_rows(trajectory_path)
    trace_adapter = cpcf_resources._build_trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache=compiler_cache,
    )
    if trace_adapter.evidence_sha256 != _digest(
        inputs.get("trace_adapter_sha256"),
        "c61 trace-adapter SHA-256",
    ):
        raise Lfm2SemanticPostfoldError(
            "Rebuilt c61 target-token trace adapter changed."
        )

    provider = None
    final_payload = None
    final_norm_authentication_sha256 = None
    try:
        provider = open_lfm2_cpcf_selective_evidence_provider_from_catalog(
            compiler_cache,
            artifact_store=catalog_store_directory,
            expected_binding_sha256=_digest(
                preparation.get("binding_sha256"),
                "c61 selective-catalog binding SHA-256",
            ),
            expected_manifest_sha256=_digest(
                preparation.get("artifact_manifest_sha256"),
                "c61 selective-catalog manifest SHA-256",
            ),
            expected_catalog_sha256=_digest(
                preparation.get("artifact_catalog_sha256"),
                "c61 selective-catalog artifact SHA-256",
            ),
        )
        catalog = provider.catalog
        if (
            catalog.catalog_sha256 != preparation.get("selective_catalog_sha256")
            or catalog.source_sha256 != preparation.get("selective_source_sha256")
            or catalog.provider_sha256 != preparation.get("selective_provider_sha256")
        ):
            raise Lfm2SemanticPostfoldError(
                "Warm-opened selective catalog differs from c61."
            )
        final_payload = provider.load_final_norm_payload()
        final_norm_authentication_sha256 = provider.authenticate_final_norm_payload(
            final_payload
        )
        if final_norm_authentication_sha256 != _digest(
            compiled.get("final_norm_authentication_sha256"),
            "c61 final-normalization authentication SHA-256",
        ):
            raise Lfm2SemanticPostfoldError(
                "Final-normalization evidence changed since c61 compilation."
            )
        shells = cpcf_v5_compiler.build_lfm2_cpcf_v5_target_phase_shells(
            trace_adapter,
            catalog,
            None,
            final_norm_payload=final_payload,
        )
        final_phases = cpcf_v5_compiler.materialize_lfm2_cpcf_v5_final_norm_phases(
            shells,
            catalog,
            final_payload,
        )
        boundary_compiler = cpcf_v5_compiler._SelectiveBoundaryCompiler(
            shells,
            catalog,
            source_selected_tied_table,
            target_hidden_width=int(target_model.config.hidden_size),
            model_policy=Lfm2CpcfModelFoldPolicy(),
        )
        interface = boundary_compiler.fit_interface(
            final_phases,
            target_norm_epsilon=cpcf_model_fold._final_norm_epsilon(target_model),
        )
    finally:
        if provider is not None and final_payload is not None:
            provider.release_final_norm_payload(final_payload)
        if provider is not None:
            provider.close()

    lineage = bundle.get("interface_lineage")
    if not isinstance(lineage, Mapping):
        raise Lfm2SemanticPostfoldError(
            "c61 boundary bundle has no tied-interface lineage."
        )
    base_report = _verify_sealed(
        lineage.get("base_interface_report"),
        seal_name="report_sha256",
        label="c61 base tied-interface report",
    )
    anchor_report = _verify_sealed(
        lineage.get("causal_anchor_report"),
        seal_name="report_sha256",
        label="c61 causal-anchor interface report",
    )
    active_report = _verify_sealed(
        interface.report,
        seal_name="report_sha256",
        label="rehydrated c61 active interface",
    )
    active_anchor_value = anchor_report.get("active_anchor_residual_selected")
    base_control_value = anchor_report.get("selected_candidate_is_base_control")
    if (
        not isinstance(active_anchor_value, bool)
        or not isinstance(base_control_value, bool)
        or active_anchor_value is base_control_value
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 causal-anchor selection flags are not exact booleans."
        )
    active_anchor = active_anchor_value
    source64 = (
        source_selected_tied_table.detach()
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )
    target = getattr(target_model.get_input_embeddings(), "weight", None)
    output = getattr(target_model.get_output_embeddings(), "weight", None)
    if not isinstance(target, torch.nn.Parameter) or output is not target:
        raise Lfm2SemanticPostfoldError(
            "Saved c61 child no longer exposes one native tied table."
        )
    saved_target = target.detach().to(device="cpu").contiguous()
    expected_saved_target = interface.target_weight.to(
        dtype=saved_target.dtype,
    ).contiguous()
    saved_target_runtime_sha256 = cpcf_model_compiler._tensor_sha256(saved_target)
    if (
        dict(boundary_compiler.base_interface.report) != base_report
        or dict(boundary_compiler.anchor_fold.report) != anchor_report
        or active_report != anchor_report
        or lineage.get("base_interface_report_sha256") != base_report["report_sha256"]
        or lineage.get("causal_anchor_report_sha256") != anchor_report["report_sha256"]
        or lineage.get("active_interface_report_sha256")
        != active_report["report_sha256"]
        or lineage.get("active_interface_is_anchor_repaired") is not active_anchor
        or lineage.get("active_interface_is_base_control") != (not active_anchor)
        or lineage.get("producer_encoder_fixed_by_anchor_selection") is not True
        or bundle.get("interface_report_sha256") != active_report["report_sha256"]
        or bundle.get("base_interface_report_sha256") != base_report["report_sha256"]
        or bundle.get("causal_anchor_report_sha256") != anchor_report["report_sha256"]
        or compiled.get("boundary_bundle_sha256") != bundle["bundle_sha256"]
        or install.get("interface_report_sha256") != active_report["report_sha256"]
        or install.get("target_weight_sha256") != saved_target_runtime_sha256
        or not torch.equal(saved_target, expected_saved_target)
        or active_report.get("target_weight_sha256")
        != _interface_tensor_sha256(interface.target_weight)
        or active_report.get("producer_encoder_sha256")
        != _interface_tensor_sha256(interface.producer_encoder)
        or base_report.get("source_table_sha256") != _interface_tensor_sha256(source64)
        or anchor_report.get("source_tied_table_sha256")
        != _interface_tensor_sha256(source64)
    ):
        raise Lfm2SemanticPostfoldError(
            "Rehydrated producer coordinates differ from c61 or the saved child."
        )
    body = {
        "schema": "neural-forge-lfm2-c61-interface-rehydration-v1",
        "compiler_diagnostic_receipt_sha256": diagnostic["receipt_sha256"],
        "compiler_diagnostic_file_sha256": diagnostic_file_sha256,
        "compiler_cache_key": compiler_cache_key,
        "semantic_evidence_cache_key": semantic_cache.cache_key,
        "common_parent_fingerprint": common_parent,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "trajectory_corpus_sha256": trajectory_sha256,
        "trace_adapter_sha256": trace_adapter.evidence_sha256,
        "selective_catalog_preparation_receipt_sha256": (
            preparation["preparation_receipt_sha256"]
        ),
        "selective_catalog_preparation_file_sha256": preparation_file_sha256,
        "selective_catalog_binding_sha256": preparation["binding_sha256"],
        "selective_catalog_manifest_sha256": preparation["artifact_manifest_sha256"],
        "selective_catalog_artifact_sha256": preparation["artifact_catalog_sha256"],
        "final_norm_authentication_sha256": final_norm_authentication_sha256,
        "base_interface_report_sha256": base_report["report_sha256"],
        "active_interface_report_sha256": active_report["report_sha256"],
        "active_interface_is_anchor_repaired": active_anchor,
        "active_interface_is_base_control": not active_anchor,
        "source_table_sha256": _interface_tensor_sha256(source64),
        "active_target_weight_sha256": active_report["target_weight_sha256"],
        "saved_target_weight_runtime_sha256": saved_target_runtime_sha256,
        "producer_encoder_sha256": active_report["producer_encoder_sha256"],
        "producer_encoder_recovered_from_repaired_target_table": False,
        "producer_encoder_recomputed_from_authenticated_final_norm_evidence": True,
        "compiler_and_semantic_cache_identities_bound_separately": True,
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return interface, MappingProxyType(_sealed(body, "result_sha256"))


@dataclass(frozen=True, slots=True)
class Lfm2SemanticCompilerLayerLineage:
    """Normalized, authenticated c61 attention ancestry for one layer."""

    layer_index: int
    source_inventory: Lfm2DenseQkvoSourceInventory
    query_groups: tuple[tuple[int, ...], ...]
    query_group_weights: tuple[tuple[float, ...], ...]
    compiler_lineage_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
            or not isinstance(self.source_inventory, Lfm2DenseQkvoSourceInventory)
            or self.source_inventory.verify() != self.source_inventory.inventory_sha256
        ):
            raise Lfm2SemanticPostfoldError(
                "Compiler layer lineage has invalid coordinates or inventory."
            )
        groups = tuple(
            tuple(int(item) for item in group) for group in self.query_groups
        )
        weights = tuple(
            tuple(float(item) for item in group) for group in self.query_group_weights
        )
        if (
            not groups
            or len(groups) != len(weights)
            or any(not group for group in groups)
            or any(
                len(group) != len(weight)
                for group, weight in zip(groups, weights, strict=True)
            )
            or any(
                not math.isclose(math.fsum(weight), 1.0, rel_tol=1e-8, abs_tol=1e-10)
                or any(not math.isfinite(value) or value <= 0.0 for value in weight)
                for weight in weights
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Compiler layer lineage has invalid query-head groups."
            )
        lineage = _digest(
            self.compiler_lineage_sha256,
            "compiler layer lineage SHA-256",
        )
        report = _verify_sealed(
            self.report,
            seal_name="lineage_sha256",
            label="compiler layer lineage",
        )
        if (
            report["lineage_sha256"] != lineage
            or report.get("layer_index") != self.layer_index
            or report.get("source_inventory_sha256")
            != self.source_inventory.inventory_sha256
            or report.get("query_groups") != [list(value) for value in groups]
            or report.get("query_group_weights") != [list(value) for value in weights]
        ):
            raise Lfm2SemanticPostfoldError(
                "Normalized compiler layer lineage differs from its seal."
            )
        object.__setattr__(self, "query_groups", groups)
        object.__setattr__(self, "query_group_weights", weights)
        object.__setattr__(self, "compiler_lineage_sha256", lineage)
        object.__setattr__(self, "report", MappingProxyType(report))


def _typed_qkvo_inventory(value: Any) -> Lfm2DenseQkvoSourceInventory:
    if not isinstance(value, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Compiler diagnostic has no typed parent Q/K/V/O inventory."
        )
    try:
        inventory = Lfm2DenseQkvoSourceInventory(
            tensor_receipts=value.get("tensor_receipts", {}),
            source_state_sha256=value.get("source_state_sha256"),
            structural_source_state_sha256=value.get("structural_source_state_sha256"),
            logical_parameters=value.get("logical_parameters"),
            inventory_sha256=value.get("inventory_sha256", ""),
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticPostfoldError(
            "Compiler parent Q/K/V/O inventory is invalid."
        ) from exc
    if inventory.as_dict() != dict(value):
        raise Lfm2SemanticPostfoldError(
            "Compiler parent Q/K/V/O inventory is not canonical."
        )
    return inventory


def _load_lfm2_semantic_foundation_lineage(
    foundation_receipt_json: str | os.PathLike[str],
    *,
    expected_foundation_receipt_sha256: str,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    input_metadata: Mapping[str, Any],
    parent_model_directory: str | os.PathLike[str],
) -> Mapping[int, Lfm2SemanticCompilerLayerLineage]:
    """Normalize the fast foundation's direct-local attention ancestry."""

    try:
        verified, receipt_file_sha256 = (
            authenticate_lfm2_semantic_foundation_receipt(
                foundation_receipt_json,
                expected_receipt_sha256=expected_foundation_receipt_sha256,
                semantic_cache=cache,
                bridge=bridge,
                input_metadata=input_metadata,
                parent_model_directory=parent_model_directory,
            )
        )
    except Lfm2SemanticFoundationError as exc:
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation ancestry authentication failed."
        ) from exc
    raw_lineages = verified.get("attention_lineage")
    topology = verified.get("target_topology")
    context = verified.get("foundation_context")
    parent_inventory = verified.get("parent_source_inventory")
    if (
        not isinstance(raw_lineages, Mapping)
        or not isinstance(topology, Mapping)
        or not isinstance(context, Mapping)
        or not isinstance(parent_inventory, Mapping)
        or context.get("attention_lineage_sha256")
        != canonical_sha256(raw_lineages)
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation receipt lacks bound attention lineage."
        )
    expected_layers = topology.get("attention_layer_indices")
    if (
        not isinstance(expected_layers, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in expected_layers)
        or set(raw_lineages) != {str(value) for value in expected_layers}
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation attention lineage differs from target topology."
        )
    normalized: dict[int, Lfm2SemanticCompilerLayerLineage] = {}
    for layer_text, raw in sorted(raw_lineages.items(), key=lambda item: int(item[0])):
        layer_index = int(layer_text)
        lineage = _verify_sealed(
            raw,
            seal_name="lineage_sha256",
            label=f"semantic-foundation layer {layer_index} lineage",
        )
        group_report = _verify_sealed(
            lineage.get("head_group_report"),
            seal_name="result_sha256",
            label=f"semantic-foundation layer {layer_index} head groups",
        )
        inventory = _typed_qkvo_inventory(lineage.get("source_inventory"))
        groups = tuple(
            tuple(int(value) for value in group)
            for group in lineage.get("query_groups", ())
        )
        weights = tuple(
            tuple(float(value) for value in group)
            for group in lineage.get("query_group_weights", ())
        )
        target_head_dim = topology.get("head_dim")
        target_query_heads = topology.get("query_heads")
        q_shape = inventory.tensor_receipts["q_proj.weight"]["shape"]
        source_query_heads = (
            0
            if isinstance(target_head_dim, bool)
            or not isinstance(target_head_dim, int)
            or target_head_dim <= 0
            or int(q_shape[0]) % target_head_dim
            else int(q_shape[0]) // target_head_dim
        )
        if (
            lineage.get("schema") != LFM2_SEMANTIC_FOUNDATION_LINEAGE_SCHEMA
            or lineage.get("layer_index") != layer_index
            or lineage.get("parent_inventory_sha256")
            != parent_inventory.get("result_sha256")
            or lineage.get("source_inventory_sha256")
            != inventory.inventory_sha256
            or lineage.get("head_group_report_sha256")
            != group_report["result_sha256"]
            or group_report.get("query_groups")
            != [list(value) for value in groups]
            or group_report.get("query_group_weights")
            != [list(value) for value in weights]
            or group_report.get("full_source_query_coverage") is not True
            or group_report.get("all_weights_positive") is not True
            or group_report.get("source_query_heads") != source_query_heads
            or group_report.get("target_query_heads") != target_query_heads
            or source_query_heads <= 0
            or len(groups) != target_query_heads
            or sorted(value for group in groups for value in group)
            != list(range(source_query_heads))
            or lineage.get("target_hidden_size") != topology.get("hidden_size")
            or lineage.get("target_query_heads") != target_query_heads
            or lineage.get("target_key_value_heads")
            != topology.get("key_value_heads")
            or lineage.get("target_head_dim") != topology.get("head_dim")
            or lineage.get("legacy_layer_compiler_calls") != 0
        ):
            raise Lfm2SemanticPostfoldError(
                "Semantic-foundation layer lineage is internally inconsistent."
            )
        report = {
            **lineage,
            "foundation_lineage_sha256": lineage["lineage_sha256"],
            "foundation_receipt_sha256": verified["receipt_sha256"],
            "foundation_receipt_file_sha256": receipt_file_sha256,
            "compiler_cache_key": verified["compiler_cache_key"],
            "semantic_evidence_cache_key": cache.cache_key,
            "common_parent_fingerprint": cache.parent_fingerprint,
            "compiler_and_semantic_cache_identities_bound_separately": True,
        }
        report.pop("lineage_sha256")
        report = _sealed(report, "lineage_sha256")
        normalized[layer_index] = Lfm2SemanticCompilerLayerLineage(
            layer_index=layer_index,
            source_inventory=inventory,
            query_groups=groups,
            query_group_weights=weights,
            compiler_lineage_sha256=report["lineage_sha256"],
            report=report,
        )
    if set(normalized) != set(expected_layers):
        raise Lfm2SemanticPostfoldError(
            "Semantic-foundation attention lineage is incomplete."
        )
    return MappingProxyType(normalized)


def load_lfm2_semantic_compiler_lineage(
    diagnostic_json: str | os.PathLike[str],
    *,
    expected_diagnostic_sha256: str,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    input_metadata: Mapping[str, Any],
    parent_model_directory: str | os.PathLike[str] | None = None,
) -> Mapping[int, Lfm2SemanticCompilerLayerLineage]:
    """Load legacy c61 or fast-foundation attention ancestry."""

    if _claims_semantic_foundation_schema(diagnostic_json):
        if parent_model_directory is None:
            raise Lfm2SemanticPostfoldError(
                "Semantic-foundation ancestry requires the direct-local parent."
            )
        return _load_lfm2_semantic_foundation_lineage(
            diagnostic_json,
            expected_foundation_receipt_sha256=expected_diagnostic_sha256,
            cache=cache,
            bridge=bridge,
            input_metadata=input_metadata,
            parent_model_directory=parent_model_directory,
        )

    verified, diagnostic_file_sha256 = _authenticated_c61_diagnostic(
        diagnostic_json,
        expected_diagnostic_sha256=expected_diagnostic_sha256,
    )
    expected = verified["receipt_sha256"]
    inputs = verified.get("inputs")
    seam = verified.get("compiled_candidate_consumer_seam")
    compiled = verified.get("compiled_model_fold")
    compiler_cache_key = (
        None if not isinstance(inputs, Mapping) else inputs.get("cache_key")
    )
    if (
        not isinstance(inputs, Mapping)
        or _digest(compiler_cache_key, "c61 compiler cache key") != compiler_cache_key
        or inputs.get("parent_fingerprint") != cache.parent_fingerprint
        or inputs.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or not isinstance(seam, Mapping)
        or seam.get("cache_sha256") != compiler_cache_key
        or seam.get("parent_sha256") != cache.parent_fingerprint
        or seam.get("bridge_sha256") != bridge.artifact_sha256
        or seam.get("target_state_sha256") != input_metadata.get("model_state_sha256")
        or not isinstance(compiled, Mapping)
    ):
        raise Lfm2SemanticPostfoldError(
            "c61 diagnostic is not the compiler ancestry of this saved child."
        )
    compiled_report = _verify_sealed(
        compiled,
        seal_name="result_sha256",
        label="c61 compiled-model receipt",
    )
    layers = verified.get("layer_results")
    if not isinstance(layers, list) or not layers:
        raise Lfm2SemanticPostfoldError(
            "c61 diagnostic contains no ordered layer results."
        )
    normalized: dict[int, Lfm2SemanticCompilerLayerLineage] = {}
    for raw in layers:
        if not isinstance(raw, Mapping):
            raise Lfm2SemanticPostfoldError("c61 layer result is not a mapping.")
        layer = _verify_sealed(
            raw,
            seal_name="result_sha256",
            label="c61 layer result",
        )
        if layer.get("operator_kind") != "full_attention":
            continue
        layer_index = layer.get("layer_index")
        solver = layer.get("solver_evidence_policy")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or layer_index in normalized
            or not isinstance(solver, Mapping)
        ):
            raise Lfm2SemanticPostfoldError(
                "c61 attention layer lineage is missing or duplicated."
            )
        evaluations = solver.get("attention_candidate_evaluations")
        direct = (
            None
            if not isinstance(evaluations, Mapping)
            else evaluations.get("direct_monolithic")
        )
        direct_receipt = _verify_sealed(
            direct,
            seal_name="report_sha256",
            label=f"c61 layer {layer_index} direct attention candidate",
        )
        artifact = direct_receipt.get("operator_artifact")
        context = direct_receipt.get("selection_context_metrics")
        if not isinstance(artifact, Mapping) or not isinstance(context, Mapping):
            raise Lfm2SemanticPostfoldError(
                "c61 direct attention candidate lacks groups or context audit."
            )
        groups = tuple(
            tuple(int(item) for item in group)
            for group in artifact.get("query_groups", ())
        )
        weights = tuple(
            tuple(float(item) for item in group)
            for group in artifact.get("query_barycentric_weights", ())
        )
        group_hash = canonical_sha256(
            {
                "groups": [list(value) for value in groups],
                "weights": [list(value) for value in weights],
            }
        )
        endpoint = solver.get("endpoint_residual_attention_artifact")
        endpoint_receipt = _verify_sealed(
            endpoint,
            seal_name="artifact_sha256",
            label=f"c61 layer {layer_index} endpoint-residual artifact",
        )
        inventory = _typed_qkvo_inventory(endpoint_receipt.get("source_inventory"))
        if (
            direct_receipt.get("candidate_id") != "direct_monolithic"
            or direct_receipt.get("operator_artifact_sha256")
            != canonical_sha256(dict(artifact))
            or context.get("query_groups_sha256") != group_hash
            or endpoint_receipt.get("source_inventory_sha256")
            != inventory.inventory_sha256
            or solver.get("endpoint_residual_attention_artifact_sha256")
            != endpoint_receipt["artifact_sha256"]
        ):
            raise Lfm2SemanticPostfoldError(
                "c61 attention groups or parent inventory changed after compilation."
            )
        body = {
            "schema": LFM2_SEMANTIC_COMPILER_LINEAGE_SCHEMA,
            "diagnostic_receipt_sha256": expected,
            "diagnostic_file_sha256": diagnostic_file_sha256,
            "compiler_cache_key": compiler_cache_key,
            "semantic_evidence_cache_key": cache.cache_key,
            "common_parent_fingerprint": cache.parent_fingerprint,
            "compiler_and_semantic_cache_identities_bound_separately": True,
            "compiled_model_fold_sha256": compiled_report["result_sha256"],
            "layer_index": layer_index,
            "layer_result_sha256": layer["result_sha256"],
            "direct_candidate_receipt_sha256": direct_receipt["report_sha256"],
            "direct_operator_artifact_sha256": direct_receipt[
                "operator_artifact_sha256"
            ],
            "endpoint_artifact_sha256": endpoint_receipt["artifact_sha256"],
            "source_inventory_sha256": inventory.inventory_sha256,
            "query_groups": [list(value) for value in groups],
            "query_group_weights": [list(value) for value in weights],
            "query_groups_sha256": group_hash,
            "numeric_parent_child_weight_match_objective": False,
        }
        report = _sealed(body, "lineage_sha256")
        normalized[layer_index] = Lfm2SemanticCompilerLayerLineage(
            layer_index=layer_index,
            source_inventory=inventory,
            query_groups=groups,
            query_group_weights=weights,
            compiler_lineage_sha256=report["lineage_sha256"],
            report=report,
        )
    if not normalized:
        raise Lfm2SemanticPostfoldError(
            "c61 diagnostic contains no authenticated attention lineage."
        )
    return MappingProxyType(normalized)


def load_lfm2_parent_qkvo_inventory_from_local_safetensors(
    model_directory: str | os.PathLike[str],
    *,
    layer_index: int,
    expected_inventory: Lfm2DenseQkvoSourceInventory,
) -> Lfm2DenseQkvoSourceInventory:
    """Hash direct-local parent Q/K/V/O bytes and match c61's sealed inventory."""

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Direct-local Q/K/V/O loading needs safetensors.") from exc
    directory = _ordinary_directory(model_directory, "parent model directory")
    prefix = f"model.layers.{layer_index}.self_attn."
    keys = {
        "q": prefix + "q_proj.weight",
        "k": prefix + "k_proj.weight",
        "v": prefix + "v_proj.weight",
        "o": prefix + "out_proj.weight",
    }
    tensors: dict[str, Any] = {}
    for shard in _local_safetensor_shards(directory):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name, key in keys.items():
                if key in handle.keys():
                    if name in tensors:
                        raise Lfm2SemanticPostfoldError(
                            f"Parent tensor {key!r} appears in multiple shards."
                        )
                    tensors[name] = handle.get_tensor(key)
    if set(tensors) != set(keys):
        raise Lfm2SemanticPostfoldError(
            f"Direct-local parent layer {layer_index} lacks dense Q/K/V/O."
        )
    observed = seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=tensors["q"],
        source_key_weight=tensors["k"],
        source_value_weight=tensors["v"],
        source_output_weight=tensors["o"],
    )
    if (
        not isinstance(expected_inventory, Lfm2DenseQkvoSourceInventory)
        or observed.as_dict() != expected_inventory.as_dict()
    ):
        raise Lfm2SemanticPostfoldError(
            f"Direct-local parent Q/K/V/O bytes changed for layer {layer_index}."
        )
    return observed


@dataclass(frozen=True, slots=True)
class Lfm2SemanticAttentionTransportEvidence:
    """One bounded batch/layer teacher lattice handed to the installer."""

    batch_index: int
    layer_index: int
    alignment_sha256: str
    alignment: Lfm2SemanticSpanCpcfPhaseAlignment = field(
        repr=False,
        compare=False,
    )
    capability: Mapping[str, Any]
    history: Mapping[str, Any]
    actual_child_replay: Lfm2SemanticAttentionReplayEndpoint = field(
        repr=False,
        compare=False,
    )
    source_inventory: Lfm2DenseQkvoSourceInventory = field(
        repr=False,
        compare=False,
    )
    query_groups: tuple[tuple[int, ...], ...]
    query_group_weights: tuple[tuple[float, ...], ...]
    compiler_lineage_sha256: str
    transport: SemanticAttentionExpansionTransport = field(
        repr=False,
        compare=False,
    )
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, int)
            or self.batch_index < 0
            or isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
            or not isinstance(
                self.alignment,
                Lfm2SemanticSpanCpcfPhaseAlignment,
            )
            or not isinstance(self.transport, SemanticAttentionExpansionTransport)
            or not isinstance(
                self.actual_child_replay,
                Lfm2SemanticAttentionReplayEndpoint,
            )
            or not isinstance(
                self.source_inventory,
                Lfm2DenseQkvoSourceInventory,
            )
        ):
            raise Lfm2SemanticPostfoldError(
                "Semantic attention evidence coordinates are invalid."
            )
        _digest(self.alignment_sha256, "semantic alignment SHA-256")
        capability = _verify_sealed(
            self.capability,
            seal_name="capability_sha256",
            label="attention capability",
        )
        history = _verify_sealed(
            self.history,
            seal_name="history_sha256",
            label="attention history",
        )
        groups = tuple(
            tuple(int(item) for item in group) for group in self.query_groups
        )
        weights = tuple(
            tuple(float(item) for item in group) for group in self.query_group_weights
        )
        lineage = _digest(
            self.compiler_lineage_sha256,
            "compiler lineage SHA-256",
        )
        if (
            self.alignment.alignment_sha256 != self.alignment_sha256
            or self.transport.semantic_alignment_sha256 != self.alignment_sha256
            or self.actual_child_replay.alignment_sha256 != self.alignment_sha256
            or self.actual_child_replay.batch_index != self.batch_index
            or self.actual_child_replay.layer_index != self.layer_index
            or self.source_inventory.verify() != self.source_inventory.inventory_sha256
            or not groups
            or len(groups) != len(weights)
        ):
            raise Lfm2SemanticPostfoldError(
                "Semantic attention transport/replay/compiler lineage diverges."
            )
        object.__setattr__(self, "capability", MappingProxyType(capability))
        object.__setattr__(self, "history", MappingProxyType(history))
        object.__setattr__(self, "query_groups", groups)
        object.__setattr__(self, "query_group_weights", weights)
        object.__setattr__(self, "compiler_lineage_sha256", lineage)
        object.__setattr__(self, "evidence_sha256", canonical_sha256(self._body()))

    def endpoint_residual_evidence(self) -> dict[str, Any]:
        return self.transport.endpoint_residual_evidence()

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-lfm2-semantic-attention-transport-evidence-v1",
            "batch_index": self.batch_index,
            "layer_index": self.layer_index,
            "alignment_sha256": self.alignment_sha256,
            "phase": self.alignment.phase,
            "capability_sha256": self.capability["capability_sha256"],
            "history_sha256": self.history["history_sha256"],
            "transport_sha256": self.transport.transport_sha256,
            "actual_child_replay_sha256": self.actual_child_replay.replay_sha256,
            "source_inventory_sha256": self.source_inventory.inventory_sha256,
            "query_groups_sha256": canonical_sha256(
                {
                    "groups": [list(value) for value in self.query_groups],
                    "weights": [list(value) for value in self.query_group_weights],
                }
            ),
            "compiler_lineage_sha256": self.compiler_lineage_sha256,
            "direct_endpoint_case_suffix_compatible": (
                self.transport.direct_endpoint_case_suffix_compatible
            ),
            "probability_mass_error": (self.transport.maximum_probability_mass_error),
            "context_absolute_error": (self.transport.maximum_context_absolute_error),
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2SemanticLayerStackInstallation:
    """Installed causal stack plus deferred evidence for its shared final W2."""

    receipt: Mapping[str, Any]
    final_layer_mlp_evidence: Mapping[
        str,
        Lfm2CpcfTerminalActionEvidence,
    ] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        receipt = _verify_sealed(
            self.receipt,
            seal_name="result_sha256",
            label="semantic layer-stack installation",
        )
        evidence = dict(self.final_layer_mlp_evidence)
        expected_splits = {"fit", "fit_check", "selection", "validation"}
        if set(evidence) != expected_splits or any(
            not isinstance(value, Lfm2CpcfTerminalActionEvidence)
            or value.split != split
            or not value.verify_authentication()
            for split, value in evidence.items()
        ):
            raise Lfm2SemanticPostfoldError(
                "Layer-stack final-MLP evidence does not cover every split."
            )
        evidence_sha = {
            split: evidence[split].authentication_sha256 for split in sorted(evidence)
        }
        if (
            receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not True
            or receipt.get("final_layer_mlp_w2_install_count") != 0
            or receipt.get("final_layer_mlp_evidence_sha256") != evidence_sha
        ):
            raise Lfm2SemanticPostfoldError(
                "Layer-stack receipt does not authenticate its deferred final W2."
            )
        object.__setattr__(self, "receipt", MappingProxyType(receipt))
        object.__setattr__(
            self,
            "final_layer_mlp_evidence",
            MappingProxyType(evidence),
        )


def seal_lfm2_semantic_attention_installation_receipt(
    *,
    layer_index: int,
    transport_evidence_sha256: Sequence[str],
    model_state_before_sha256: str,
    model_state_after_sha256: str,
    changed_state_keys: Sequence[str],
    parameter_element_count: int,
) -> Mapping[str, Any]:
    """Canonical receipt helper for the future analytic attention compiler."""

    evidence = tuple(
        _digest(value, "transport evidence SHA-256")
        for value in transport_evidence_sha256
    )
    changed = tuple(changed_state_keys)
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or not evidence
        or not changed
        or any(not isinstance(value, str) or not value for value in changed)
        or isinstance(parameter_element_count, bool)
        or not isinstance(parameter_element_count, int)
        or parameter_element_count <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            "Attention installation receipt inputs are incomplete."
        )
    before = _digest(model_state_before_sha256, "model state before")
    after = _digest(model_state_after_sha256, "model state after")
    if before == after:
        raise Lfm2SemanticPostfoldError(
            "An installed attention correction must change model state."
        )
    body = {
        "schema": LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA,
        "layer_index": layer_index,
        "installed": True,
        "transport_evidence_sha256": list(evidence),
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
        "changed_state_keys": list(changed),
        "parameter_element_count_before": parameter_element_count,
        "parameter_element_count_after": parameter_element_count,
        "parameter_count_changed": False,
        "analytic_cached_evidence_only": True,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _validate_attention_installation_receipt(
    value: Mapping[str, Any],
    *,
    layer_index: int,
    evidence: Sequence[Lfm2SemanticAttentionTransportEvidence],
    before_sha256: str,
    after_sha256: str,
    changed_keys: Sequence[str],
    parameter_count: int,
) -> dict[str, Any]:
    receipt = _verify_sealed(
        value,
        seal_name="result_sha256",
        label="semantic attention installation receipt",
    )
    allowed_prefix = f"model.layers.{layer_index}.self_attn."
    expected_evidence = [item.evidence_sha256 for item in evidence]
    if (
        receipt.get("schema") != LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA
        or receipt.get("layer_index") != layer_index
        or receipt.get("installed") is not True
        or receipt.get("transport_evidence_sha256") != expected_evidence
        or receipt.get("model_state_before_sha256") != before_sha256
        or receipt.get("model_state_after_sha256") != after_sha256
        or receipt.get("changed_state_keys") != list(changed_keys)
        or not changed_keys
        or any(not name.startswith(allowed_prefix) for name in changed_keys)
        or receipt.get("parameter_element_count_before") != parameter_count
        or receipt.get("parameter_element_count_after") != parameter_count
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("analytic_cached_evidence_only") is not True
        or receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
    ):
        raise Lfm2SemanticPostfoldError(
            "Analytic attention installer returned an invalid or over-broad receipt."
        )
    return receipt


def _attention_layer_indices(
    cache: Lfm2CircuitTraceCache,
    target_model: Any,
) -> tuple[int, ...]:
    topology = cache.manifest.get("layer_topology", {}).get("layers")
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if not isinstance(topology, list) or not layers:
        raise Lfm2SemanticPostfoldError(
            "Cache or target model has no LFM2 layer topology."
        )
    indices = tuple(
        index
        for index, value in enumerate(topology)
        if isinstance(value, Mapping)
        and isinstance(value.get("contract"), Mapping)
        and value["contract"].get("kind") == "full_attention"
    )
    if (
        not indices
        or len(topology) != len(layers)
        or any(index >= len(layers) for index in indices)
    ):
        raise Lfm2SemanticPostfoldError(
            "Parent-cache and saved-child attention layer topology differs."
        )
    child_types = tuple(getattr(target_model.config, "layer_types", ()))
    if (
        child_types
        and tuple(
            index
            for index, value in enumerate(child_types)
            if value == "full_attention"
        )
        != indices
    ):
        raise Lfm2SemanticPostfoldError(
            "Saved child attention placement differs from the complete cache."
        )
    return indices


def _target_rope(target_model: Any) -> tuple[Any, float]:
    torch = _torch()
    rotary = getattr(getattr(target_model, "model", None), "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    scaling = getattr(rotary, "attention_scaling", 1.0)
    if (
        not isinstance(inv_freq, torch.Tensor)
        or inv_freq.ndim != 1
        or inv_freq.numel() == 0
        or not bool(torch.isfinite(inv_freq).all().item())
        or isinstance(scaling, bool)
        or not isinstance(scaling, (int, float))
        or not math.isfinite(float(scaling))
        or float(scaling) <= 0.0
    ):
        raise Lfm2SemanticPostfoldError(
            "Saved child exposes no finite target RoPE contract."
        )
    return (
        inv_freq.detach().to(device="cpu", dtype=torch.float64).contiguous().clone(),
        float(scaling),
    )


def _input_tensor_names(reference: Any) -> tuple[str, ...]:
    inventory = set(reference.ordered_tensor_names)
    record = reference.record
    capture = record.get("capture") if isinstance(record, Mapping) else None
    decode_steps = (
        capture.get("decode_steps") if isinstance(capture, Mapping) else None
    )
    if (
        isinstance(decode_steps, bool)
        or not isinstance(decode_steps, int)
        or decode_steps <= 0
    ):
        raise Lfm2SemanticPostfoldError(
            "Complete cache input evidence has no forced decode horizon."
        )
    names = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
        *(f"decode.{step:03d}.forced_token_ids" for step in range(decode_steps)),
    }
    if "input.position_ids" in inventory:
        names.add("input.position_ids")
    missing = names.difference(inventory)
    if missing:
        raise Lfm2SemanticPostfoldError(
            f"Complete cache input evidence is missing {sorted(missing)!r}."
        )
    return tuple(sorted(names))


def _semantic_alignments_from_session(
    *,
    cache: Lfm2CircuitTraceCache,
    session: Any,
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> tuple[
    tuple[Lfm2SemanticSpanCpcfAlignment, ...],
    Mapping[str, Any],
]:
    manifest = cache.manifest
    manifest_sha256 = _digest(
        manifest.get("manifest_sha256"),
        "cache manifest SHA-256",
    )
    session_receipt = session.as_dict()
    alignments = []
    with bridge.authenticated_live_session(
        donor_tokenizer,
        target_tokenizer,
        selective_session_sha256=_digest(
            session_receipt.get("session_sha256"),
            "selective cache session SHA-256",
        ),
    ) as tokenizer_session:
        donor_sha = tokenizer_session.donor_tokenizer_semantics_sha256
        target_sha = tokenizer_session.target_tokenizer_semantics_sha256
        for reference in session.batch_references:
            payload = session.load_selected_tensors(
                reference.index,
                _input_tensor_names(reference),
            )
            try:
                selected_input = build_lfm2_semantic_span_selected_input_evidence(
                    reference,
                    payload,
                    session_receipt,
                    expected_cache_key=cache.cache_key,
                )
                semantic = build_lfm2_semantic_span_phase_bridge(
                    selected_input,
                    manifest,
                    donor_tokenizer,
                    target_tokenizer,
                    expected_source_cache_key=cache.cache_key,
                    expected_source_manifest_sha256=manifest_sha256,
                    expected_source_input_sha256=reference.input_sha256,
                    expected_donor_tokenizer_fingerprint=donor_sha,
                    expected_target_tokenizer_fingerprint=target_sha,
                    tokenizer_session=tokenizer_session,
                )
                prompt = len(semantic.source_prompt_token_ids)
                suffix_counts = {
                    phase: prompt if phase == "prefill" else 1
                    for phase in semantic.phase_order
                }
                alignments.append(
                    build_lfm2_semantic_span_cpcf_alignment(
                        semantic,
                        donor_tokenizer,
                        target_tokenizer,
                        parent_sampled_suffix_tokens=suffix_counts,
                        parent_history_suffix_tokens=suffix_counts,
                        tokenizer_session=tokenizer_session,
                    )
                )
            finally:
                session.release_selected_tensors(payload)
    return tuple(alignments), MappingProxyType(tokenizer_session.as_dict())


def _attention_tensor_names(
    reference: Any,
    *,
    layer_index: int,
    phase_order: Sequence[str],
) -> tuple[str, ...]:
    inventory = set(reference.ordered_tensor_names)
    names = set()
    for phase in phase_order:
        prefix = f"{phase}.layer.{layer_index:03d}.attention."
        names.update(
            {
                prefix + "q_normalized_pre_rope",
                prefix + "k_normalized_pre_rope",
                prefix + "q",
                prefix + "k",
                prefix + "v",
            }
        )
    names.add(f"{phase_order[-1]}.layer.{layer_index:03d}.attention.value_history")
    missing = tuple(sorted(names.difference(inventory)))
    if missing:
        raise Lfm2SemanticPostfoldError(
            "Complete causal cache lacks semantic attention tensors: "
            f"{list(missing[:6])}."
        )
    return tuple(sorted(names))


def _parent_residual_tensor_names(
    reference: Any,
    *,
    layer_index: int,
    phase_order: Sequence[str],
) -> tuple[str, ...]:
    inventory = set(reference.ordered_tensor_names)
    names = {
        f"{phase}.layer.{layer_index:03d}.residual.{boundary}"
        for phase in phase_order
        for boundary in ("pre_operator", "post_operator")
    }
    missing = tuple(sorted(names.difference(inventory)))
    if missing:
        raise Lfm2SemanticPostfoldError(
            "Complete causal cache lacks parent residual endpoints: "
            f"{list(missing[:6])}."
        )
    return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class _SemanticLayerReplay:
    batch_index: int
    raw_pre_attention_residual: Any = field(repr=False, compare=False)
    operator_input: Any = field(repr=False, compare=False)
    replay_sha256: str


def _layer_runtime(layer: Any) -> tuple[Any, Any]:
    torch = _torch()
    parameters = tuple(layer.parameters())
    if (
        not parameters
        or any(
            value.device.type == "meta"
            or value.dtype != torch.bfloat16
            or not torch.is_floating_point(value)
            for value in parameters
        )
        or len({value.device for value in parameters}) != 1
    ):
        raise Lfm2SemanticPostfoldError(
            "Every child macrocell must use one materialized BF16 device."
        )
    return parameters[0].device, parameters[0].dtype


def _additive_causal_mask(attention_mask: Any, residual: Any) -> Any:
    torch = _torch()
    sequence = int(attention_mask.shape[1])
    allowed = torch.tril(
        torch.ones(
            (sequence, sequence),
            device=residual.device,
            dtype=torch.bool,
        )
    )[None, None]
    active = attention_mask.to(
        device=residual.device,
        dtype=torch.bool,
    )[:, None, None, :]
    allowed = allowed & active
    minimum = torch.finfo(residual.dtype).min
    return torch.where(
        allowed,
        torch.zeros((), device=residual.device, dtype=residual.dtype),
        torch.full((), minimum, device=residual.device, dtype=residual.dtype),
    )


def _replay_child_to_attention_layer(
    target_model: Any,
    *,
    alignment: Lfm2SemanticSpanCpcfAlignment,
    layer_index: int,
) -> _SemanticLayerReplay:
    """Replay one final target prefix analytically; never call a model/layer."""

    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index >= len(layers):
        raise Lfm2SemanticPostfoldError("Child replay layer lies outside topology.")
    final = alignment.phases[-1]
    ids = final.canonical_target_input_ids
    mask = final.canonical_target_attention_mask
    positions = final.canonical_target_position_ids
    embedding = getattr(target_model.get_input_embeddings(), "weight", None)
    if (
        not isinstance(embedding, torch.nn.Parameter)
        or embedding.dtype != torch.bfloat16
        or bool((ids < 0).any().item())
        or bool((ids >= int(embedding.shape[0])).any().item())
    ):
        raise Lfm2SemanticPostfoldError(
            "Target-native replay IDs exceed the frozen tied BF16 table."
        )
    with torch.no_grad():
        device_ids = ids.to(device=embedding.device, dtype=torch.long)
        residual = embedding.index_select(0, device_ids.reshape(-1)).reshape(
            1,
            int(device_ids.shape[1]),
            int(embedding.shape[1]),
        )
    replay_sha = []
    raw = None
    operator = None
    for index, layer in enumerate(layers[: layer_index + 1]):
        device, dtype = _layer_runtime(layer)
        residual = residual.to(device=device, dtype=dtype, non_blocking=False)
        if index == layer_index:
            raw = residual.detach().to(device="cpu").contiguous().clone()
        if bool(getattr(layer, "is_attention_layer", False)):
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                position_embeddings=_rope_from_target(
                    target_model,
                    residual,
                    positions,
                ),
                additive_causal_mask=_additive_causal_mask(mask, residual),
            )
        else:
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                padding_mask=mask.to(device=device, dtype=dtype),
            )
        replay_sha.append(replay.replay_sha256)
        if index == layer_index:
            operator = replay.operator_input
        residual = replay.post_layer_residual
    if raw is None or operator is None:
        raise Lfm2SemanticPostfoldError(
            "Target-native replay produced no attention endpoint."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-child-attention-replay-v1",
        "batch_index": alignment.source_batch_index,
        "layer_index": layer_index,
        "alignment_sha256": alignment.alignment_sha256,
        "final_phase_alignment_sha256": final.alignment_sha256,
        "raw_pre_attention_residual_sha256": _tensor_sha256(raw),
        "operator_input_sha256": _tensor_sha256(operator),
        "macrocell_replay_sha256": replay_sha,
        "model_forward_calls": 0,
        "layer_forward_calls": 0,
        "training_steps": 0,
    }
    return _SemanticLayerReplay(
        batch_index=alignment.source_batch_index,
        raw_pre_attention_residual=raw,
        operator_input=operator,
        replay_sha256=canonical_sha256(body),
    )


def _semantic_residual_chart(
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent_pre_residual: Mapping[int, Any],
    child_replays: Mapping[int, _SemanticLayerReplay],
    producer_encoder: Any,
    layer_index: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Fit one FIT-only orthogonal gauge around the saved interface chart."""

    torch = _torch()
    ordered = tuple(sorted(alignments, key=lambda value: value.source_batch_index))
    fit_count = max(1, len(ordered) // 2)
    if len(ordered) < 4:
        raise Lfm2SemanticPostfoldError(
            "Residual chart requires four trajectory-isolated cache batches."
        )
    encoder = producer_encoder.detach().to(device="cpu", dtype=torch.float64)
    fit_left = []
    fit_right = []
    check_left = []
    check_right = []
    fit_batches = {value.source_batch_index for value in ordered[:fit_count]}
    for alignment in ordered:
        batch = alignment.source_batch_index
        parent = parent_pre_residual[batch].to(torch.float64)
        replay = child_replays[batch].raw_pre_attention_residual.to(torch.float64)
        boundary = alignment.phases[-1].target_to_donor_boundary_positions
        targets = tuple(
            index for index, value in enumerate(boundary) if value is not None
        )
        donors = tuple(int(value) for value in boundary if value is not None)
        if donors != tuple(range(int(parent.shape[1]))):
            raise Lfm2SemanticPostfoldError(
                "Final semantic endpoints do not cover every donor residual row."
            )
        selected = replay.index_select(
            1,
            torch.tensor(targets, dtype=torch.long),
        )
        left = (parent[0] @ encoder).contiguous()
        right = selected[0].contiguous()
        destination_left = fit_left if batch in fit_batches else check_left
        destination_right = fit_right if batch in fit_batches else check_right
        # Equal total trajectory mass prevents long prompts dominating the chart.
        destination_left.append(left / math.sqrt(int(left.shape[0])))
        destination_right.append(right / math.sqrt(int(right.shape[0])))
    left_fit = torch.cat(fit_left, dim=0)
    right_fit = torch.cat(fit_right, dim=0)
    cross = left_fit.T @ right_fit
    left_vectors, singular, right_vectors_h = torch.linalg.svd(
        cross,
        full_matrices=False,
    )
    rotation = (left_vectors @ right_vectors_h).contiguous()
    if tuple(rotation.shape) != (
        int(encoder.shape[1]),
        int(encoder.shape[1]),
    ) or not bool(torch.isfinite(rotation).all().item()):
        raise Lfm2SemanticPostfoldError("FIT-only semantic residual gauge is invalid.")

    def metrics(left_rows: Sequence[Any], right_rows: Sequence[Any]) -> dict[str, Any]:
        left = torch.cat(tuple(left_rows), dim=0)
        right = torch.cat(tuple(right_rows), dim=0)
        prediction = left @ rotation
        difference = prediction - right
        denominator = right.square().mean().sqrt().clamp_min(1e-30)
        return {
            "rows": int(left.shape[0]),
            "relative_rmse": float(
                (difference.square().mean().sqrt() / denominator).item()
            ),
            "maximum_absolute_error": float(difference.abs().max().item()),
        }

    body = {
        "schema": LFM2_SEMANTIC_RESIDUAL_CHART_SCHEMA,
        "layer_index": layer_index,
        "fit_batch_indices": sorted(fit_batches),
        "heldout_batch_indices": sorted(
            value.source_batch_index for value in ordered[fit_count:]
        ),
        "base_producer_encoder_sha256": _tensor_sha256(encoder),
        "rotation_sha256": _tensor_sha256(rotation),
        "singular_values_sha256": _tensor_sha256(singular),
        "fit_metrics": metrics(fit_left, fit_right),
        "heldout_metrics": metrics(check_left, check_right),
        "method": "fit_only_orthogonal_procrustes_around_saved_interface_chart",
        "validation_rows_used_for_fit": 0,
        "numeric_parent_child_weight_match_objective": False,
        "model_forward_calls": 0,
        "training_steps": 0,
    }
    return (encoder @ rotation).contiguous(), MappingProxyType(
        _sealed(body, "chart_sha256")
    )


def _transport_evidence_for_layer(
    *,
    cache: Lfm2CircuitTraceCache,
    session: Any,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    layer_index: int,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float,
    target_model: Any,
    bridge: StrictTokenizerBridgeArtifact,
    interface: TiedCausalInterfaceFold,
    compiler_lineage: Lfm2SemanticCompilerLayerLineage,
) -> tuple[Lfm2SemanticAttentionTransportEvidence, ...]:
    torch = _torch()
    parent_pre: dict[int, Any] = {}
    parent_post: dict[int, Any] = {}
    child_replays = {
        alignment.source_batch_index: _replay_child_to_attention_layer(
            target_model,
            alignment=alignment,
            layer_index=layer_index,
        )
        for alignment in alignments
    }
    for alignment, reference in zip(
        alignments,
        session.batch_references,
        strict=True,
    ):
        payload = session.load_selected_tensors(
            reference.index,
            _parent_residual_tensor_names(
                reference,
                layer_index=layer_index,
                phase_order=alignment.phase_order,
            ),
        )
        try:
            parent_pre[reference.index] = torch.cat(
                tuple(
                    payload.tensors[
                        f"{phase}.layer.{layer_index:03d}.residual.pre_operator"
                    ]
                    for phase in alignment.phase_order
                ),
                dim=1,
            ).contiguous()
            parent_post[reference.index] = torch.cat(
                tuple(
                    payload.tensors[
                        f"{phase}.layer.{layer_index:03d}.residual.post_operator"
                    ]
                    for phase in alignment.phase_order
                ),
                dim=1,
            ).contiguous()
        finally:
            session.release_selected_tensors(payload)
    chart, chart_report = _semantic_residual_chart(
        alignments=alignments,
        parent_pre_residual=parent_pre,
        child_replays=child_replays,
        producer_encoder=interface.producer_encoder,
        layer_index=layer_index,
    )
    captured_state = fingerprint_lfm2_parent(target_model).identity_sha256
    result = []
    for alignment, reference in zip(
        alignments,
        session.batch_references,
        strict=True,
    ):
        phases = alignment.phase_order
        payload = session.load_selected_tensors(
            reference.index,
            _attention_tensor_names(
                reference,
                layer_index=layer_index,
                phase_order=phases,
            ),
        )
        try:
            tensors = payload.tensors
            prompt = alignment.phases[0].donor_full_prefix_tokens
            capability = inspect_lfm2_semantic_attention_cache_capability(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            capability.require_exact_history()
            history = assemble_lfm2_explicit_pre_rope_history(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            parent_query = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.q"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            parent_key = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.k"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            replay = child_replays[reference.index]
            for phase_index, phase_alignment in enumerate(alignment.phases):
                donor_tokens = phase_alignment.donor_full_prefix_tokens
                target_tokens = phase_alignment.target_full_prefix_tokens
                phase_order = alignment.phase_order[: phase_index + 1]
                phase_capability = Lfm2SemanticAttentionCacheCapability(
                    layer_index=layer_index,
                    phase_order=phase_order,
                    donor_prompt_token_count=prompt,
                    selected_route="explicit_pre_rope",
                    prefill_captured_token_count=prompt,
                    assembled_current_row_count=donor_tokens,
                    missing_enhanced_tensor_names=(),
                    missing_legacy_tensor_names=(),
                    missing_donor_prefix_positions=(),
                    blocking_reasons=(),
                    exact_history_supported=True,
                )
                phase_history = SemanticAttentionPreRopeHistory(
                    query_normalized_pre_rope=(
                        history.query_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    key_normalized_pre_rope=(
                        history.key_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    value=history.value[:, :, :donor_tokens],
                    source_capability_sha256=(phase_capability.capability_sha256),
                    source_tensor_names=tuple(
                        name
                        for name in history.source_tensor_names
                        if any(name.startswith(item + ".") for item in phase_order)
                    ),
                )
                transport = transport_lfm2_phase_probability_preserving_attention(
                    phase_alignment,
                    parent_query_post_rope=parent_query[:, :, :donor_tokens],
                    parent_key_post_rope=parent_key[:, :, :donor_tokens],
                    parent_value=history.value[:, :, :donor_tokens],
                    target_rope_inv_freq=target_rope_inv_freq,
                    target_rope_attention_scaling=target_rope_attention_scaling,
                    query_mode="semantic_endpoints",
                )
                query_positions = torch.tensor(
                    transport.query_target_positions,
                    dtype=torch.long,
                )
                donor_positions = torch.tensor(
                    transport.query_donor_positions,
                    dtype=torch.long,
                )
                child_query = replay.operator_input[:, :target_tokens].index_select(
                    1,
                    query_positions,
                )
                child_key_value = replay.operator_input[:, :target_tokens]
                actual_pre = replay.raw_pre_attention_residual[
                    :, :target_tokens
                ].index_select(1, query_positions)
                parent_target = (
                    parent_post[reference.index][:, :donor_tokens]
                    .index_select(1, donor_positions)
                    .to(torch.float64)
                    @ chart
                ).to(torch.float32)
                weights = torch.ones(
                    (1, len(transport.query_target_positions)),
                    dtype=torch.float64,
                )
                actual_child = Lfm2SemanticAttentionReplayEndpoint(
                    batch_index=reference.index,
                    layer_index=layer_index,
                    alignment_sha256=phase_alignment.alignment_sha256,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                    captured_child_model_state_sha256=captured_state,
                    target_query_inputs=child_query,
                    target_key_value_inputs=child_key_value,
                    actual_child_pre_attention_residual=actual_pre,
                    parent_post_attention_residual_target_space=parent_target,
                    query_row_sqrt_weights=weights,
                )
                result.append(
                    Lfm2SemanticAttentionTransportEvidence(
                        batch_index=reference.index,
                        layer_index=layer_index,
                        alignment_sha256=phase_alignment.alignment_sha256,
                        alignment=phase_alignment,
                        capability=phase_capability.to_evidence_dict(),
                        history=phase_history.to_evidence_dict(),
                        actual_child_replay=actual_child,
                        source_inventory=compiler_lineage.source_inventory,
                        query_groups=compiler_lineage.query_groups,
                        query_group_weights=(compiler_lineage.query_group_weights),
                        compiler_lineage_sha256=canonical_sha256(
                            {
                                "compiler_lineage_sha256": (
                                    compiler_lineage.compiler_lineage_sha256
                                ),
                                "semantic_residual_chart_sha256": (
                                    chart_report["chart_sha256"]
                                ),
                            }
                        ),
                        transport=transport,
                    )
                )
        finally:
            session.release_selected_tensors(payload)
    return tuple(result)


def _resolve_attention_installer(
    explicit: Callable[..., Mapping[str, Any]] | None,
) -> Callable[..., Mapping[str, Any]] | None:
    if explicit is not None:
        if not callable(explicit):
            raise TypeError("attention_installer must be callable.")
        return explicit
    from neural_forge import hf_lfm2_semantic_attention_refold_adapter as adapter

    value = getattr(adapter, SEMANTIC_ATTENTION_INSTALLER_CALLABLE, None)
    if callable(value):
        return value
    from neural_forge import hf_semantic_attention_lattice as lattice

    fallback = getattr(lattice, SEMANTIC_ATTENTION_INSTALLER_CALLABLE, None)
    return fallback if callable(fallback) else None


def _resolve_layer_stack_installer(
    explicit: Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    if explicit is not None:
        if not callable(explicit):
            raise TypeError("semantic_layer_stack_compiler must be callable.")
        return explicit
    try:
        from neural_forge import hf_lfm2_semantic_layer_stack_refold_adapter as adapter
    except ImportError:
        return None
    value = getattr(adapter, SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE, None)
    return value if callable(value) else None


def _final_w2_parameter(target_model: Any) -> Any:
    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    parameter = None
    if layers:
        parameter = getattr(
            getattr(getattr(layers[-1], "feed_forward", None), "w2", None),
            "weight",
            None,
        )
    if not isinstance(parameter, torch.nn.Parameter):
        raise Lfm2SemanticPostfoldError("Child final W2 is not a Parameter.")
    return parameter


def _validate_live_layer_stack_producer_coverage(
    target_model: Any,
    receipt: Mapping[str, Any],
) -> None:
    """Cross-check every nested coverage proof against the live folded child."""

    layer_values = receipt.get("layer_installations")
    if not isinstance(layer_values, list) or len(layer_values) != 14:
        raise Lfm2SemanticPostfoldError(
            "Layer-stack receipt omitted its fourteen producer coverage decisions."
        )
    for layer_index, raw_layer in enumerate(layer_values):
        if not isinstance(raw_layer, Mapping):
            raise Lfm2SemanticPostfoldError(
                "Layer-stack producer coverage decision is not a mapping."
            )
        layer = _verify_sealed(
            raw_layer,
            seal_name="result_sha256",
            label=f"semantic layer {layer_index} transaction",
        )
        expected_operator_kind = (
            "full_attention" if layer_index in (2, 4, 6, 8, 10, 12) else "short_conv"
        )
        if (
            layer.get("layer_index") != layer_index
            or layer.get("operator_kind") != expected_operator_kind
            or layer.get("operator_and_mlp_producer_chains_structurally_nonzero")
            is not True
        ):
            raise Lfm2SemanticPostfoldError(
                f"Layer {layer_index} structural producer receipt is inconsistent."
            )
        for lane, producer_kind, defer_down_weight in (
            ("operator", expected_operator_kind, False),
            ("mlp", "mlp", layer_index == 13),
        ):
            decision_value = layer.get(f"{lane}_producer_coverage_decision")
            if not isinstance(decision_value, Mapping):
                raise Lfm2SemanticPostfoldError(
                    f"Layer {layer_index} omitted {lane} producer coverage."
                )
            decision = _verify_sealed(
                decision_value,
                seal_name="result_sha256",
                label=f"layer {layer_index} {lane} producer coverage decision",
            )
            before_value = decision.get("coverage_before")
            after_value = decision.get("coverage_after")
            if not isinstance(before_value, Mapping) or not isinstance(
                after_value, Mapping
            ):
                raise Lfm2SemanticPostfoldError(
                    f"Layer {layer_index} {lane} coverage omitted before/after proof."
                )
            before_coverage = _verify_sealed(
                before_value,
                seal_name="result_sha256",
                label=f"layer {layer_index} {lane} pre-decision coverage",
            )
            after_coverage = _verify_sealed(
                after_value,
                seal_name="result_sha256",
                label=f"layer {layer_index} {lane} post-decision coverage",
            )
            installed = decision.get(
                "installed_complete_structurally_nonzero_chain"
            )
            retained = decision.get(
                "authenticated_nonzero_folded_seed_retained_by_noop"
            )
            expected_deferred = (
                ["feed_forward.w2.weight"] if defer_down_weight else []
            )
            if (
                decision.get("layer_index") != layer_index
                or decision.get("producer_kind") != producer_kind
                or decision.get("deferred_roles") != expected_deferred
                or decision.get("coverage_before_sha256")
                != before_coverage.get("result_sha256")
                or decision.get("coverage_after_sha256")
                != after_coverage.get("result_sha256")
                or decision.get(
                    "all_required_roles_structurally_nonzero_after_decision"
                )
                is not True
                or decision.get(
                    "blank_or_partially_blank_required_role_count_after_decision"
                )
                != 0
                or type(installed) is not bool
                or type(retained) is not bool
                or installed == retained
                or (decision.get("cold_start_or_partial_seed_at_entry") and not installed)
                or (
                    retained
                    and (
                        before_coverage.get(
                            "all_required_roles_structurally_nonzero"
                        )
                        is not True
                        or before_coverage.get("required_role_state_sha256")
                        != after_coverage.get("required_role_state_sha256")
                    )
                )
                or layer.get(f"{lane}_installed") is not installed
            ):
                raise Lfm2SemanticPostfoldError(
                    f"Layer {layer_index} {lane} coverage decision is internally false."
                )
            live = _lfm2_dense_producer_chain_coverage(
                target_model,
                layer_index=layer_index,
                producer_kind=producer_kind,
                defer_down_weight=defer_down_weight,
            )
            if live.get("result_sha256") != after_coverage.get("result_sha256"):
                raise Lfm2SemanticPostfoldError(
                    f"Layer {layer_index} live {lane} producer bytes differ from "
                    "the authenticated structural coverage decision."
                )


def compile_install_lfm2_semantic_layer_stack_stage(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
    parent_model_directory: str | os.PathLike[str],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    attention_installer: Callable[..., Mapping[str, Any]] | None = None,
    semantic_layer_stack_compiler: Callable[..., Any] | None = None,
) -> Lfm2SemanticLayerStackInstallation:
    """Require causal operator->MLP layer order before terminal publication."""

    compiler = _resolve_layer_stack_installer(semantic_layer_stack_compiler)
    if compiler is None:
        topology = cache.manifest.get("layer_topology", {}).get("layers", [])
        if not isinstance(topology, list):
            topology = []
        attention_count = sum(
            isinstance(value, Mapping)
            and isinstance(value.get("contract"), Mapping)
            and value["contract"].get("kind") == "full_attention"
            for value in topology
        )
        report = _sealed(
            {
                "schema": LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA,
                "status": "blocked-no-checkpoint-published",
                "stage": "semantic_layer_stack",
                "reason": "complete-causal-layer-stack-installer-unavailable",
                "blocking_reasons": [
                    "long-horizon-short-conv-refold-missing",
                    "replay-fresh-per-layer-mlp-refold-missing",
                    "attention-cannot-skip-intervening-conv-and-mlp-installs",
                ],
                "required_causal_layer_count": len(topology),
                "required_attention_operator_count": attention_count,
                "required_short_conv_operator_count": (len(topology) - attention_count),
                "required_replay_fresh_mlp_count": len(topology),
                "missing_callable": (
                    "neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter."
                    + SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE
                ),
                "available_partial_attention_callable": (
                    "neural_forge.hf_lfm2_semantic_attention_refold_adapter."
                    + SEMANTIC_ATTENTION_INSTALLER_CALLABLE
                ),
                "fused_terminal_refold_called": False,
                "checkpoint_save_called": False,
                "operation_audit": dict(_ZERO_OPERATIONS),
            },
            "failure_sha256",
        )
        raise Lfm2SemanticPostfoldBlockedError(
            "A complete causal operator/MLP layer-stack installer is required",
            report,
        )
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    parameter_count = _parameter_count(target_model)
    final_w2_before = _tensor_sha256(_final_w2_parameter(target_model))
    result = compiler(
        cache=cache,
        bridge=bridge,
        target_model=target_model,
        interface=interface,
        compiler_lineage=compiler_lineage,
        parent_model_directory=parent_model_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_installer=_resolve_attention_installer(attention_installer),
    )
    if not isinstance(result, Lfm2SemanticLayerStackInstallation):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer-stack compiler returned no typed deferred-W2 result."
        )
    verified = _verify_sealed(
        result.receipt,
        seal_name="result_sha256",
        label="semantic layer-stack installation",
    )
    after = fingerprint_lfm2_parent(target_model).identity_sha256
    final_w2_after = _tensor_sha256(_final_w2_parameter(target_model))
    if (
        verified.get("complete_causal_layer_stack_adjudicated") is not True
        or verified.get("all_attention_operator_decisions_completed") is not True
        or verified.get("attention_operator_decision_count") != 6
        or verified.get("attention_operator_install_count", -1)
        + verified.get("attention_operator_incumbent_noop_count", -1)
        != 6
        or verified.get("all_short_conv_operator_decisions_completed") is not True
        or verified.get("short_conv_operator_decision_count") != 8
        or verified.get("short_conv_operator_install_count", -1)
        + verified.get("short_conv_operator_incumbent_noop_count", -1)
        != 8
        or verified.get("all_replay_fresh_mlp_evidence_compiled") is not True
        or verified.get("replay_fresh_mlp_evidence_compile_count") != 14
        or verified.get("all_nonterminal_mlp_writer_decisions_completed") is not True
        or verified.get("nonterminal_mlp_writer_decision_count") != 13
        or verified.get("nonterminal_mlp_writer_install_count", -1)
        + verified.get("nonterminal_mlp_writer_incumbent_noop_count", -1)
        != 13
        or verified.get("final_mlp_producer_install_count", -1)
        + verified.get("final_mlp_producer_incumbent_noop_count", -1)
        != 1
        or verified.get("complete_nonzero_structural_producer_coverage")
        is not True
        or verified.get("all_producer_chains_installed_or_retained_nonzero_seed")
        is not True
        or verified.get("attention_producer_structural_coverage_count") != 6
        or verified.get("short_conv_producer_structural_coverage_count") != 8
        or verified.get("nonterminal_mlp_producer_structural_coverage_count") != 13
        or verified.get("final_mlp_producer_structural_coverage_count") != 1
        or verified.get("blank_or_partially_blank_producer_chain_count") != 0
        or verified.get("installed_complete_producer_chain_count", -1)
        + verified.get(
            "authenticated_nonzero_folded_seed_retained_noop_count", -1
        )
        != 28
        or verified.get(
            "authenticated_nonzero_folded_seed_retained_noop_count", -1
        )
        != (
            verified.get("attention_operator_incumbent_noop_count", -2)
            + verified.get("short_conv_operator_incumbent_noop_count", -2)
            + verified.get("nonterminal_mlp_writer_incumbent_noop_count", -2)
            + verified.get("final_mlp_producer_incumbent_noop_count", -2)
        )
        or verified.get("whole_stage_contains_meaningful_parameter_changes") is not True
        or verified.get("operator_parameter_mutation_count", -1)
        != verified.get("attention_operator_install_count", -2)
        + verified.get("short_conv_operator_install_count", -2)
        or verified.get("mlp_parameter_mutation_count", -1)
        != verified.get("nonterminal_mlp_writer_install_count", -2)
        + verified.get("final_mlp_producer_install_count", -2)
        or verified.get("operator_parameter_mutation_count", 0)
        + verified.get("mlp_parameter_mutation_count", 0)
        <= 0
        or verified.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not True
        or verified.get("final_layer_mlp_w2_install_count") != 0
        or verified.get("causal_install_order")
        != "operator_then_replay_fresh_mlp_then_next_layer"
        or verified.get("model_state_before_sha256") != before
        or verified.get("model_state_after_sha256") != after
        or before == after
        or not verified.get("changed_state_keys")
        or verified.get("parameter_element_count") != parameter_count
        or _parameter_count(target_model) != parameter_count
        or verified.get("parameter_count_changed") is not False
        or final_w2_after != final_w2_before
        or verified.get("deferred_final_w2_sha256") != final_w2_before
        or verified.get("operation_audit") != dict(_ZERO_OPERATIONS)
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic layer-stack compiler returned an incomplete receipt."
        )
    _validate_live_layer_stack_producer_coverage(target_model, verified)
    return Lfm2SemanticLayerStackInstallation(
        receipt=verified,
        final_layer_mlp_evidence=result.final_layer_mlp_evidence,
    )


def compile_install_lfm2_semantic_attention_stage(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
    parent_model_directory: str | os.PathLike[str],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    attention_installer: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Audit and install every exact long-horizon semantic attention layer."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    saved_tokenizer = getattr(
        target_model,
        "_neural_forge_quarantine_tokenizer",
        target_tokenizer,
    )
    if saved_tokenizer is not target_tokenizer and fingerprint_hf_tokenizer_semantics(
        saved_tokenizer
    ) != fingerprint_hf_tokenizer_semantics(target_tokenizer):
        raise Lfm2SemanticPostfoldError(
            "Saved child tokenizer differs from the semantic target tokenizer."
        )
    layers = _attention_layer_indices(cache, target_model)
    if set(compiler_lineage) != set(layers):
        raise Lfm2SemanticPostfoldError(
            "c61 compiler lineage does not cover every child attention layer."
        )
    for layer_index in layers:
        load_lfm2_parent_qkvo_inventory_from_local_safetensors(
            parent_model_directory,
            layer_index=layer_index,
            expected_inventory=compiler_lineage[layer_index].source_inventory,
        )
    inv_freq, attention_scaling = _target_rope(target_model)
    installer = _resolve_attention_installer(attention_installer)
    blocked_report = None
    layer_receipts = []
    state_before = fingerprint_lfm2_parent(target_model).identity_sha256
    parameter_count = _parameter_count(target_model)
    with cache.selective_authenticated_session() as session:
        alignments, tokenizer_session_receipt = _semantic_alignments_from_session(
            cache=cache,
            session=session,
            bridge=bridge,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        refinement = dict(
            audit_lfm2_semantic_span_cpcf_refinement(
                alignments,
                expected_batch_count=cache.batch_count,
            )
        )
        if not refinement["probability_preserving_attention_transport_go"]:
            blocked_report = {
                "reason": "semantic-span-attention-transport-not-exact",
                "blocking_reasons": list(refinement["blocking_reasons"]),
                "missing_callable": None,
            }
        elif installer is None:
            blocked_report = {
                "reason": "analytic-attention-weight-installer-unavailable",
                "blocking_reasons": [
                    "semantic-lattice-produces-teacher-tensors-but-no-qkvo-compiler"
                ],
                "missing_callable": (
                    "neural_forge.hf_lfm2_semantic_attention_refold_adapter."
                    + SEMANTIC_ATTENTION_INSTALLER_CALLABLE
                ),
            }
        else:
            for layer_index in layers:
                evidence = _transport_evidence_for_layer(
                    cache=cache,
                    session=session,
                    alignments=alignments,
                    layer_index=layer_index,
                    target_rope_inv_freq=inv_freq,
                    target_rope_attention_scaling=attention_scaling,
                    target_model=target_model,
                    bridge=bridge,
                    interface=interface,
                    compiler_lineage=compiler_lineage[layer_index],
                )
                before_hashes = _state_tensor_hashes(target_model)
                before = fingerprint_lfm2_parent(target_model).identity_sha256
                observed_count = _parameter_count(target_model)
                if observed_count != parameter_count:
                    raise Lfm2SemanticPostfoldError(
                        "Child parameter count changed before attention installation."
                    )
                candidate_receipt = installer(
                    target_model=target_model,
                    layer_index=layer_index,
                    evidence=evidence,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                )
                after_hashes = _state_tensor_hashes(target_model)
                after = fingerprint_lfm2_parent(target_model).identity_sha256
                changed = _changed_state_keys(before_hashes, after_hashes)
                if _parameter_count(target_model) != parameter_count:
                    raise Lfm2SemanticPostfoldError(
                        "Attention installer changed child parameter count."
                    )
                layer_receipts.append(
                    _validate_attention_installation_receipt(
                        candidate_receipt,
                        layer_index=layer_index,
                        evidence=evidence,
                        before_sha256=before,
                        after_sha256=after,
                        changed_keys=changed,
                        parameter_count=parameter_count,
                    )
                )
        session.verify()
        session_receipt = session.as_dict()
    alignment_receipts = [value.to_evidence_dict() for value in alignments]
    common = {
        "schema": LFM2_SEMANTIC_ATTENTION_STAGE_SCHEMA,
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "cache_selective_session_sha256": session_receipt["session_sha256"],
        "tokenizer_session_receipt_sha256": tokenizer_session_receipt["receipt_sha256"],
        "tokenizer_session": dict(tokenizer_session_receipt),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "attention_layer_indices": list(layers),
        "semantic_alignment_sha256": [value.alignment_sha256 for value in alignments],
        "semantic_alignments": alignment_receipts,
        "refinement_audit": refinement,
        "target_rope_inv_freq_sha256": _tensor_sha256(inv_freq),
        "target_rope_attention_scaling": attention_scaling,
        "model_state_before_sha256": state_before,
        "parameter_element_count": parameter_count,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    if blocked_report is not None:
        failure = _sealed(
            {
                "schema": LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA,
                "status": "blocked-no-checkpoint-published",
                "stage": "semantic_attention",
                **blocked_report,
                "semantic_attention_audit": common,
                "terminal_refold_called": False,
                "checkpoint_save_called": False,
                "operation_audit": dict(_ZERO_OPERATIONS),
            },
            "failure_sha256",
        )
        missing = blocked_report["missing_callable"]
        detail = (
            f"; missing exact callable {missing}"
            if missing is not None
            else f"; blockers={blocked_report['blocking_reasons']}"
        )
        raise Lfm2SemanticPostfoldBlockedError(
            "Semantic attention postfold cannot publish" + detail,
            failure,
        )
    state_after = fingerprint_lfm2_parent(target_model).identity_sha256
    if (
        len(layer_receipts) != len(layers)
        or state_after == state_before
        or _parameter_count(target_model) != parameter_count
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic attention stage did not install every required layer."
        )
    return MappingProxyType(
        _sealed(
            {
                **common,
                "status": "installed",
                "model_state_after_sha256": state_after,
                "layer_installation_receipts": layer_receipts,
                "all_attention_layers_installed": True,
                "parameter_count_changed": False,
            },
            "result_sha256",
        )
    )


def _replay_lexical_rows(
    target_model: Any,
    *,
    token_ids: Sequence[int],
    position_origin: int,
    row_positions: Sequence[int],
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Replay bridge-known intermediate prefixes and expose final-W2 inputs."""

    torch = _torch()
    ids = torch.tensor([tuple(token_ids)], dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.arange(
        position_origin,
        position_origin + int(ids.shape[1]),
        dtype=torch.long,
    ).unsqueeze(0)
    selected = tuple(int(value) for value in row_positions)
    if (
        not selected
        or any(value < 0 or value >= int(ids.shape[1]) for value in selected)
        or len(set(selected)) != len(selected)
    ):
        raise Lfm2SemanticPostfoldError(
            "Lexical continuation replay rows are empty, repeated, or out of range."
        )
    embedding = getattr(target_model.get_input_embeddings(), "weight", None)
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if (
        not isinstance(embedding, torch.nn.Parameter)
        or embedding.dtype != torch.bfloat16
        or not layers
        or bool((ids >= int(embedding.shape[0])).any().item())
    ):
        raise Lfm2SemanticPostfoldError(
            "Lexical continuation replay has no frozen dense BF16 child."
        )
    with torch.no_grad():
        device_ids = ids.to(device=embedding.device)
        residual = embedding.index_select(0, device_ids.reshape(-1)).reshape(
            1,
            int(ids.shape[1]),
            int(embedding.shape[1]),
        )
    replay_sha = []
    final_mlp = None
    final_pre = None
    for layer_index, layer in enumerate(layers):
        device, dtype = _layer_runtime(layer)
        residual = residual.to(device=device, dtype=dtype, non_blocking=False)
        if bool(getattr(layer, "is_attention_layer", False)):
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                position_embeddings=_rope_from_target(
                    target_model,
                    residual,
                    positions,
                ),
                additive_causal_mask=_additive_causal_mask(mask, residual),
            )
        else:
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                padding_mask=mask.to(device=device, dtype=dtype),
            )
        replay_sha.append(replay.replay_sha256)
        if layer_index == len(layers) - 1:
            index = torch.tensor(selected, dtype=torch.long)
            final_mlp = replay.mlp_input.index_select(1, index)[0]
            final_pre = replay.post_operator_residual.index_select(1, index)[0]
        residual = replay.post_layer_residual
    if final_mlp is None or final_pre is None:
        raise Lfm2SemanticPostfoldError(
            "Lexical continuation replay produced no final-W2 endpoints."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-lexical-replay-v1",
        "token_ids_sha256": canonical_sha256(list(token_ids)),
        "position_origin": position_origin,
        "row_positions": list(selected),
        "mlp_input_sha256": _tensor_sha256(final_mlp),
        "pre_mlp_residual_sha256": _tensor_sha256(final_pre),
        "macrocell_replay_sha256": replay_sha,
        "model_forward_calls": 0,
        "layer_forward_calls": 0,
        "training_steps": 0,
    }
    return final_mlp, final_pre, MappingProxyType(_sealed(body, "replay_sha256"))


def _append_only_lexical_supervision_coordinates(
    semantic: Any,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    """Return every bridge-inserted continuation query/target pair exactly once."""

    phases = tuple(semantic.phases)
    if len(phases) < 2:
        raise Lfm2SemanticPostfoldError(
            "Lexical span completion requires prefill plus decode phases."
        )
    batch_index = int(semantic.source_batch_index)
    previous = phases[0]
    row_positions = []
    desired = []
    row_ids = []
    multiplicities = []
    for phase in phases[1:]:
        replacement = tuple(int(value) for value in phase.target_replacement_token_ids)
        multiplicities.append(len(replacement))
        if (
            phase.target_replaced_suffix_token_count != 0
            or phase.target_rewrite_start != previous.target_full_prefix_length
            or not replacement
            or phase.target_full_prefix_length
            != previous.target_full_prefix_length + len(replacement)
        ):
            raise Lfm2SemanticPostfoldBlockedError(
                "Lexical completion currently requires append-only spans",
                _sealed(
                    {
                        "schema": LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA,
                        "status": "blocked-no-checkpoint-published",
                        "stage": "lexical_span_completion",
                        "reason": "non-append-only-semantic-transition",
                        "batch_index": batch_index,
                        "phase": phase.phase,
                        "operation_audit": dict(_ZERO_OPERATIONS),
                    },
                    "failure_sha256",
                ),
            )
        for offset in range(1, len(replacement)):
            row_positions.append(previous.target_full_prefix_length + offset - 1)
            desired.append(replacement[offset])
            row_ids.append(
                f"lexical:batch-{batch_index:06d}:{phase.phase}:subtoken-{offset:02d}"
            )
        previous = phase
    return (
        tuple(row_positions),
        tuple(desired),
        tuple(row_ids),
        tuple(multiplicities),
    )


def _lexical_teacher_rows(
    target_tied_weight: Any,
    desired_token_ids: Sequence[int],
) -> tuple[Any, Mapping[str, Any]]:
    """Select exact-top1 hidden teachers from a fixed closed-form frontier."""

    torch = _torch()
    table = target_tied_weight.detach().to(device="cpu", dtype=torch.float64)
    desired = tuple(int(value) for value in desired_token_ids)
    if not desired or min(desired) < 0 or max(desired) >= int(table.shape[0]):
        raise Lfm2SemanticPostfoldError(
            "Lexical desired token IDs exceed the installed tied table."
        )
    unique = tuple(sorted(set(desired)))
    indices = torch.tensor(unique, dtype=torch.long)
    rows = table.index_select(0, indices)
    gram = table.T @ table
    scale = float(gram.diagonal().mean().clamp_min(1e-30).item())
    candidates = [("tied_row", rows)]
    for ridge in (1e-8, 1e-6, 1e-4, 1e-2):
        inverse = torch.linalg.solve(
            gram + (ridge * scale) * torch.eye(int(gram.shape[0]), dtype=torch.float64),
            torch.eye(int(gram.shape[0]), dtype=torch.float64),
        )
        candidates.append((f"ridge_whitened_{ridge:.0e}", rows @ inverse))
    selected: dict[int, Any] = {}
    selected_method: dict[int, str] = {}
    for method, candidate64 in candidates:
        candidate = candidate64.to(torch.bfloat16)
        for start in range(0, len(unique), 32):
            stop = min(len(unique), start + 32)
            logits = candidate[start:stop].to(torch.float64) @ table.T
            winners = logits.argmax(dim=1)
            for offset, winner in enumerate(winners.tolist(), start=start):
                token = unique[offset]
                if token == winner and token not in selected:
                    selected[token] = candidate[offset].clone()
                    selected_method[token] = method
    missing = tuple(token for token in unique if token not in selected)
    if missing:
        raise Lfm2SemanticPostfoldBlockedError(
            "Closed-form lexical teacher frontier has no exact top-1 witness",
            _sealed(
                {
                    "schema": LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA,
                    "status": "blocked-no-checkpoint-published",
                    "stage": "lexical_span_completion",
                    "reason": "no-exact-top1-closed-form-teacher",
                    "missing_token_ids": list(missing),
                    "teacher_frontier": [name for name, _value in candidates],
                    "operation_audit": dict(_ZERO_OPERATIONS),
                },
                "failure_sha256",
            ),
        )
    teacher = torch.stack(tuple(selected[value] for value in desired), dim=0)
    winners = (teacher.to(torch.float64) @ table.T).argmax(dim=1)
    if tuple(winners.tolist()) != desired:
        raise AssertionError("Selected lexical teacher lost its exact winner.")
    methods = {
        name: sum(selected_method[value] == name for value in unique)
        for name, _candidate in candidates
    }
    body = {
        "schema": "neural-forge-lfm2-semantic-lexical-teacher-v1",
        "desired_token_ids_sha256": canonical_sha256(list(desired)),
        "desired_row_count": len(desired),
        "unique_desired_token_count": len(unique),
        "teacher_rows_sha256": _tensor_sha256(teacher),
        "frontier": [name for name, _candidate in candidates],
        "selected_unique_token_counts": methods,
        "all_teacher_rows_exact_target_top1": True,
        "gradient_search_used": False,
        "training_steps": 0,
    }
    return teacher, MappingProxyType(_sealed(body, "teacher_sha256"))


def _global_tokenizer_expansion_inventory(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> Mapping[str, Any]:
    """Audit global one-donor-token lexical expansion separately from traces."""

    donor_capacity = bridge.transport_profile.donor_row_capacity
    records = []
    multiplicities: dict[int, int] = {}
    exact_roundtrips = 0
    for donor_id in range(donor_capacity):
        text = _decode_exact(donor_tokenizer, (donor_id,), "donor")
        target_ids = _encode_exact(target_tokenizer, text, "target")
        multiplicity = len(target_ids)
        multiplicities[multiplicity] = multiplicities.get(multiplicity, 0) + 1
        roundtrip = _decode_exact(target_tokenizer, target_ids, "target") == text
        exact_roundtrips += int(roundtrip)
        records.append(
            {
                "donor_token_id": donor_id,
                "target_token_ids": list(target_ids),
                "exact_roundtrip": roundtrip,
            }
        )
    body = {
        "schema": "neural-forge-global-tokenizer-lexical-expansion-inventory-v1",
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "donor_tokenizer_semantics_sha256": (bridge.donor_tokenizer_semantics_sha256),
        "target_tokenizer_semantics_sha256": (bridge.target_tokenizer_semantics_sha256),
        "donor_token_count": donor_capacity,
        "exact_roundtrip_count": exact_roundtrips,
        "multiplicity_counts": {
            str(key): value for key, value in sorted(multiplicities.items())
        },
        "multi_token_expansion_count": sum(
            value for key, value in multiplicities.items() if key > 1
        ),
        "maximum_target_multiplicity": max(multiplicities),
        "mapping_sha256": canonical_sha256(records),
        "scope": "standalone_donor_token_text_not_contextual_trace_coverage",
        "training_steps": 0,
    }
    return MappingProxyType(_sealed(body, "inventory_sha256"))


def _convert_parent_terminal_evidence(
    evidence: Lfm2CpcfTerminalActionEvidence,
    producer_encoder: Any,
) -> Lfm2CpcfTerminalActionEvidence:
    torch = _torch()
    target = (
        evidence.parent_final_norm_output.to(torch.float64)
        @ producer_encoder.detach().to(device="cpu", dtype=torch.float64)
    ).to(torch.bfloat16)
    return Lfm2CpcfTerminalActionEvidence(
        split=evidence.split,
        row_ids=tuple("parent-endpoint:" + value for value in evidence.row_ids),
        mlp_input=evidence.mlp_input,
        pre_mlp_residual=evidence.pre_mlp_residual,
        parent_final_norm_output=target,
        sample_weights=evidence.sample_weights,
    )


def _merge_terminal_evidence(
    lanes: Mapping[str, Lfm2CpcfTerminalActionEvidence],
) -> Lfm2CpcfTerminalActionEvidence:
    torch = _torch()
    ordered = tuple((name, lanes[name]) for name in sorted(lanes))
    if len(ordered) < 2 or len({value.split for _name, value in ordered}) != 1:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal evidence must contain aligned independent lanes."
        )
    mass = 1.0 / len(ordered)
    return Lfm2CpcfTerminalActionEvidence(
        split=ordered[0][1].split,
        row_ids=tuple(
            f"{name}:{row}" for name, value in ordered for row in value.row_ids
        ),
        mlp_input=torch.cat(
            tuple(value.mlp_input for _name, value in ordered),
            dim=0,
        ),
        pre_mlp_residual=torch.cat(
            tuple(value.pre_mlp_residual for _name, value in ordered),
            dim=0,
        ),
        parent_final_norm_output=torch.cat(
            tuple(value.parent_final_norm_output for _name, value in ordered),
            dim=0,
        ),
        sample_weights=torch.cat(
            tuple(value.sample_weights * mass for _name, value in ordered),
            dim=0,
        ),
    )


def _terminal_publication_decision(
    terminal: Lfm2CpcfTerminalActionResidualResult,
) -> tuple[Any, Mapping[str, Any]]:
    """Authenticate either a changed winner or the exact published incumbent."""

    torch = _torch()
    if not isinstance(terminal, Lfm2CpcfTerminalActionResidualResult):
        raise TypeError("terminal must be a typed terminal-action result.")
    if not terminal.verify_authentication():
        raise Lfm2SemanticPostfoldError(
            "Fused terminal result changed after authentication."
        )
    report = terminal.as_dict()
    operation_audit = report.get("operation_audit")
    if (
        not isinstance(operation_audit, Mapping)
        or not operation_audit
        or any(type(value) is not int or value != 0 for value in operation_audit.values())
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal result does not prove a zero-operation compilation."
        )
    published = terminal.published_down_weight
    incumbent = terminal.incumbent_down_weight
    selected = terminal.selected_down_weight
    published_is_incumbent = bool(torch.equal(published, incumbent))
    published_is_selected = bool(torch.equal(published, selected))
    validation = report.get("validation")
    validation_passed = bool(
        isinstance(validation, Mapping) and validation.get("passed") is True
    )
    common_valid = bool(
        report.get("source_seed_writer_unchanged") is True
        and report.get("parameter_identity_or_topology_mutation") is False
        and report.get("parameter_count_before") == report.get("parameter_count_after")
        and report.get("rollback_exact") is True
    )
    changed_winner = bool(
        common_valid
        and terminal.accepted
        and terminal.changed
        and terminal.selected_candidate_id != "incumbent"
        and report.get("selected_nonincumbent") is True
        and report.get("published_is_incumbent") is False
        and report.get("rollback_performed") is False
        and validation_passed
        and published_is_selected
        and not published_is_incumbent
    )
    incumbent_retained = bool(
        common_valid
        and not terminal.accepted
        and not terminal.changed
        and report.get("published_is_incumbent") is True
        and published_is_incumbent
    )
    if incumbent_retained:
        if terminal.selected_candidate_id == "incumbent":
            incumbent_retained = bool(
                report.get("selected_nonincumbent") is False
                and validation_passed
                and report.get("rollback_performed") is False
            )
            reason = "heldout-selected-incumbent"
        else:
            incumbent_retained = bool(
                report.get("selected_nonincumbent") is True
                and not validation_passed
                and report.get("rollback_performed") is True
            )
            reason = "frozen-candidate-validation-rollback-to-incumbent"
    else:
        reason = ""
    if not changed_winner and not incumbent_retained:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal published neither an accepted changed winner nor an "
            "authenticated exact incumbent."
        )
    if changed_winner:
        reason = "accepted-nonincumbent-mutation"
    decision = {
        "publication_decision": reason,
        "terminal_publication_decision_accepted": True,
        "terminal_nonincumbent_candidate_accepted": changed_winner,
        "terminal_incumbent_retained": incumbent_retained,
        "terminal_writer_changed": changed_winner,
        "published_w2_sha256": _tensor_sha256(published),
        "selected_w2_sha256": _tensor_sha256(selected),
        "incumbent_w2_sha256": _tensor_sha256(incumbent),
        "published_is_selected": published_is_selected,
        "published_is_incumbent": published_is_incumbent,
    }
    return published, MappingProxyType(decision)


def _install_final_w2_once(
    target_model: Any,
    *,
    published_down_weight: Any,
) -> Mapping[str, Any]:
    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    parameter = getattr(
        getattr(getattr(layers[-1], "feed_forward", None), "w2", None),
        "weight",
        None,
    )
    if not isinstance(parameter, torch.nn.Parameter):
        raise Lfm2SemanticPostfoldError("Child final W2 is not a Parameter.")
    if (
        not isinstance(published_down_weight, torch.Tensor)
        or published_down_weight.layout != torch.strided
        or published_down_weight.device.type == "meta"
        or tuple(published_down_weight.shape) != tuple(parameter.shape)
        or published_down_weight.requires_grad
        or published_down_weight.grad_fn is not None
        or not bool(torch.isfinite(published_down_weight).all().item())
    ):
        raise Lfm2SemanticPostfoldError(
            "Published final W2 is not a finite detached dense topology match."
        )
    parameter_name = f"model.layers.{len(layers) - 1}.feed_forward.w2.weight"
    before_state = fingerprint_lfm2_parent(target_model).identity_sha256
    before_hashes = _state_tensor_hashes(target_model)
    before_count = _parameter_count(target_model)
    before_runtime = _parameter_runtime_snapshot(target_model)
    incumbent = parameter.detach().to(device="cpu").contiguous().clone()
    published = published_down_weight.detach().to(
        device=parameter.device,
        dtype=parameter.dtype,
    ).contiguous()
    published_coverage = _dense_weight_structural_coverage(
        published,
        parameter_name=parameter_name,
        require_parameter=False,
    )
    if published_coverage.get("structurally_nonzero") is not True:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal refused to publish a blank or structurally partial W2."
        )
    install = not torch.equal(parameter.detach(), published)
    try:
        if install:
            with torch.no_grad():
                parameter.copy_(published)
        after_state = fingerprint_lfm2_parent(target_model).identity_sha256
        after_hashes = _state_tensor_hashes(target_model)
        after_runtime = _parameter_runtime_snapshot(target_model)
        runtime_proof = _parameter_runtime_proof(before_runtime, after_runtime)
        changed_state_keys = _changed_state_keys(before_hashes, after_hashes)
        runtime_preserved = all(
            runtime_proof[name] is True
            for name in (
                "parameter_topology_preserved",
                "parameter_identities_preserved",
                "parameter_storage_preserved",
                "gradient_state_preserved",
            )
        )
        mutation_valid = bool(
            install
            and after_state != before_state
            and changed_state_keys == (parameter_name,)
        )
        noop_valid = bool(
            not install
            and after_state == before_state
            and not changed_state_keys
        )
        if (
            not runtime_preserved
            or _parameter_count(target_model) != before_count
            or not torch.equal(parameter.detach(), published)
            or not (mutation_valid or noop_valid)
        ):
            raise Lfm2SemanticPostfoldError(
                "Fused terminal publication changed topology, identity, storage, "
                "gradients, or unexpected state bytes."
            )
    except BaseException:
        with torch.no_grad():
            parameter.copy_(incumbent.to(device=parameter.device))
        rollback_runtime = _parameter_runtime_proof(
            before_runtime,
            _parameter_runtime_snapshot(target_model),
        )
        if (
            fingerprint_lfm2_parent(target_model).identity_sha256 != before_state
            or _state_tensor_hashes(target_model) != before_hashes
            or not all(
                rollback_runtime[name] is True
                for name in (
                    "parameter_topology_preserved",
                    "parameter_identities_preserved",
                    "parameter_storage_preserved",
                    "gradient_state_preserved",
                )
            )
        ):
            raise Lfm2SemanticPostfoldError("Fused terminal rollback was not exact.")
        raise
    body = {
        "schema": "neural-forge-lfm2-fused-semantic-terminal-install-v1",
        "status": (
            "installed-accepted-terminal-writer"
            if install
            else "authenticated-published-incumbent-noop"
        ),
        "parameter_name": parameter_name,
        "model_state_before_sha256": before_state,
        "model_state_after_sha256": after_state,
        "incumbent_w2_sha256": _tensor_sha256(incumbent),
        "published_w2_sha256": _tensor_sha256(parameter),
        "published_w2_structural_coverage": dict(published_coverage),
        "published_w2_structural_coverage_sha256": published_coverage[
            "result_sha256"
        ],
        "published_w2_structurally_nonzero": True,
        "installed_w2_sha256": _tensor_sha256(parameter) if install else None,
        "changed_state_keys": list(changed_state_keys),
        **dict(runtime_proof),
        "parameter_identity_preserved": runtime_proof[
            "parameter_identities_preserved"
        ],
        "parameter_count_changed": False,
        "decision_count": 1,
        "installed": install,
        "install_count": int(install),
        "mutation_count": int(install),
        "incumbent_retention_count": int(not install),
        "exact_zero_copy_noop": not install,
        "parameter_copy_calls": int(install),
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def compile_install_lfm2_fused_semantic_terminal_refold(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    layer_stack: Lfm2SemanticLayerStackInstallation,
    interface: TiedCausalInterfaceFold,
    source_selected_tied_table: Any,
    parent_final_norm: Lfm2LocalParentFinalRmsNormLoad,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    split_plan: Lfm2LongHorizonTrajectorySplitPlan | None = None,
    policy: Lfm2CpcfTerminalActionPolicy | None = None,
) -> Mapping[str, Any]:
    """Jointly compile final-MLP, parent, and lexical lanes into one final W2."""

    torch = _torch()
    if not isinstance(layer_stack, Lfm2SemanticLayerStackInstallation):
        raise TypeError("layer_stack must authenticate deferred final-MLP evidence.")
    if not isinstance(parent_final_norm, Lfm2LocalParentFinalRmsNormLoad):
        raise TypeError("parent_final_norm must be an authenticated direct-local load.")
    if (
        parent_final_norm.final_norm.cache_key != cache.cache_key
        or parent_final_norm.final_norm.parent_fingerprint != cache.parent_fingerprint
        or parent_final_norm.final_norm.layer_topology_fingerprint
        != cache.layer_topology_fingerprint
    ):
        raise Lfm2SemanticPostfoldError(
            "Direct-local parent final RMSNorm is not bound to the semantic "
            "evidence cache."
        )
    layer_receipt = layer_stack.receipt
    current_state = fingerprint_lfm2_parent(target_model).identity_sha256
    current_w2 = _tensor_sha256(_final_w2_parameter(target_model))
    final_mlp_producer_coverage_before_terminal = (
        _lfm2_dense_producer_chain_coverage(
            target_model,
            layer_index=13,
            producer_kind="mlp",
            defer_down_weight=True,
        )
    )
    layer_installations = layer_receipt.get("layer_installations")
    final_layer_installation = (
        layer_installations[-1]
        if isinstance(layer_installations, list) and layer_installations
        else None
    )
    recorded_final_mlp_coverage = (
        final_layer_installation.get("mlp_producer_coverage_decision")
        if isinstance(final_layer_installation, Mapping)
        else None
    )
    if (
        layer_receipt.get("model_state_after_sha256") != current_state
        or layer_receipt.get("deferred_final_w2_sha256") != current_w2
        or layer_receipt.get("complete_nonzero_structural_producer_coverage")
        is not True
        or final_mlp_producer_coverage_before_terminal.get(
            "all_required_roles_structurally_nonzero"
        )
        is not True
        or final_mlp_producer_coverage_before_terminal.get(
            "blank_or_partially_blank_required_role_count"
        )
        != 0
        or not isinstance(recorded_final_mlp_coverage, Mapping)
        or recorded_final_mlp_coverage.get("coverage_after_sha256")
        != final_mlp_producer_coverage_before_terminal.get("result_sha256")
        or layer_receipt.get("whole_stage_contains_meaningful_parameter_changes")
        is not True
        or layer_receipt.get("model_state_before_sha256") == current_state
        or not layer_receipt.get("changed_state_keys")
    ):
        raise Lfm2SemanticPostfoldError(
            "Child changed after deferred final-MLP evidence was captured or the "
            "layer stack proved no meaningful mutation."
        )
    selected_policy = policy or Lfm2CpcfTerminalActionPolicy()
    plan = split_plan or Lfm2LongHorizonTrajectorySplitPlan.deterministic_round_robin(
        cache
    )
    plan.verify_for(cache)
    seed_parent, _down, layers, _epsilon = _target_runtime_seed(
        target_model,
        interface,
        source_selected_tied_table,
    )
    before_replay = fingerprint_lfm2_parent(target_model).identity_sha256
    cached, semantic_bridges, cache_receipt = _collect_cached_semantic_trajectories(
        cache,
        bridge,
        donor_tokenizer,
        target_tokenizer,
        parent_final_norm.final_norm,
        source_selected_tied_table,
        plan,
    )
    if not all(
        value.parent_final_norm_endpoint.get("runtime_replay_to_cached_output_verified")
        is True
        and value.parent_final_norm_endpoint.get(
            "authenticated_cached_output_used_as_compiler_target"
        )
        is True
        for value in cached
    ):
        raise Lfm2SemanticPostfoldError(
            "Semantic postfold requires every cached pre-final residual to "
            "authenticate the direct-local parent RMSNorm output."
        )
    parent_raw, _horizons, parent_replays, parent_rope_calls = _build_split_evidence(
        cached,
        None,
        target_model,
        layers,
    )
    parent = {
        value.split: _convert_parent_terminal_evidence(
            value,
            interface.producer_encoder,
        )
        for value in parent_raw
    }
    lexical_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("fit", "fit_check", "selection", "validation")
    }
    observed = []
    all_desired = []
    pending = []
    for semantic in semantic_bridges:
        batch_index = semantic.source_batch_index
        split = plan.assignments[batch_index]
        (
            row_positions,
            desired,
            row_ids,
            transition_multiplicities,
        ) = _append_only_lexical_supervision_coordinates(semantic)
        if not row_positions:
            raise Lfm2SemanticPostfoldError(
                f"Cache batch {batch_index} has no inserted lexical rows."
            )
        mlp, pre, replay = _replay_lexical_rows(
            target_model,
            token_ids=semantic.phases[-1].target_full_prefix_token_ids,
            position_origin=semantic.target_position_origin,
            row_positions=row_positions,
        )
        pending.append((split, row_ids, mlp, pre, tuple(desired)))
        all_desired.extend(desired)
        observed.append(
            {
                "batch_index": batch_index,
                "split": split,
                "parent_decode_steps": len(semantic.phases) - 1,
                "target_decode_tokens": (
                    semantic.phases[-1].target_full_prefix_length
                    - semantic.phases[0].target_full_prefix_length
                ),
                "inserted_continuation_rows": len(row_positions),
                "maximum_donor_span_multiplicity": max(transition_multiplicities),
                "transition_multiplicity_counts": {
                    str(value): transition_multiplicities.count(value)
                    for value in sorted(set(transition_multiplicities))
                },
                "lexical_row_ids_sha256": canonical_sha256(row_ids),
                "desired_token_ids_sha256": canonical_sha256(desired),
                "replay_sha256": replay["replay_sha256"],
            }
        )
    teachers, teacher_report = _lexical_teacher_rows(
        seed_parent.target_tied_weight,
        all_desired,
    )
    cursor = 0
    for split, row_ids, mlp, pre, desired in pending:
        count = len(desired)
        lexical_rows[split].append(
            {
                "row_ids": tuple(row_ids),
                "mlp": mlp,
                "pre": pre,
                "teacher": teachers[cursor : cursor + count],
            }
        )
        cursor += count
    lexical = {}
    for split, values in lexical_rows.items():
        if not values:
            raise Lfm2SemanticPostfoldError(
                f"Lexical split {split!r} has no whole-trajectory rows."
            )
        rows = sum(len(value["row_ids"]) for value in values)
        lexical[split] = Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=tuple(row for value in values for row in value["row_ids"]),
            mlp_input=torch.cat(tuple(value["mlp"] for value in values), dim=0),
            pre_mlp_residual=torch.cat(
                tuple(value["pre"] for value in values),
                dim=0,
            ),
            parent_final_norm_output=torch.cat(
                tuple(value["teacher"] for value in values),
                dim=0,
            ),
            sample_weights=torch.full((rows,), 1.0 / rows, dtype=torch.float64),
        )
    hidden = seed_parent.target_hidden_size
    identity = torch.eye(hidden, dtype=torch.float64)
    seed = Lfm2CpcfTerminalActionSeed(
        gate_weight=seed_parent.gate_weight,
        up_weight=seed_parent.up_weight,
        down_weight=seed_parent.down_weight,
        final_norm_weight=seed_parent.final_norm_weight,
        producer_encoder=identity,
        target_tied_weight=seed_parent.target_tied_weight,
        source_tied_table=seed_parent.target_tied_weight,
        norm_epsilon=seed_parent.norm_epsilon,
    )
    final_mlp = dict(layer_stack.final_layer_mlp_evidence)
    fused = {
        split: _merge_terminal_evidence(
            {
                "final_layer_mlp": final_mlp[split],
                "lexical": lexical[split],
                "parent_endpoint": parent[split],
            }
        )
        for split in lexical
    }
    terminal = compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=fused["fit"],
        fit_check_evidence=fused["fit_check"],
        selection_evidence=fused["selection"],
        validation_evidence=fused["validation"],
        policy=selected_policy,
    )
    published_down_weight, publication_decision = _terminal_publication_decision(
        terminal
    )
    fused_heldout_gates = {}
    for split in ("selection", "validation"):
        prepared = _prepare_split(fused[split], seed)
        incumbent_metrics = _evaluate_split(
            prepared,
            seed,
            seed.down_weight,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
        published_metrics = _evaluate_split(
            prepared,
            seed,
            published_down_weight,
            metric_chunk_rows=selected_policy.metric_chunk_rows,
        )
        gates = _metric_gates(
            published_metrics,
            incumbent_metrics,
            selected_policy,
            require_selection_improvement=False,
        )
        fused_heldout_gates[split] = {
            "incumbent": incumbent_metrics.as_dict(),
            "published": published_metrics.as_dict(),
            "gates": gates,
            "passed": _all_passed(gates),
        }
    lane_gates = {}
    for lane_name, banks in (
        ("parent_endpoint", parent),
        ("lexical", lexical),
        ("final_layer_mlp", final_mlp),
    ):
        lane_gates[lane_name] = {}
        for split in ("selection", "validation"):
            prepared = _prepare_split(banks[split], seed)
            incumbent_metrics = _evaluate_split(
                prepared,
                seed,
                seed.down_weight,
                metric_chunk_rows=selected_policy.metric_chunk_rows,
            )
            published_metrics = _evaluate_split(
                prepared,
                seed,
                published_down_weight,
                metric_chunk_rows=selected_policy.metric_chunk_rows,
            )
            gates = _metric_gates(
                published_metrics,
                incumbent_metrics,
                selected_policy,
                require_selection_improvement=False,
            )
            lane_gates[lane_name][split] = {
                "incumbent": incumbent_metrics.as_dict(),
                "published": published_metrics.as_dict(),
                "gates": gates,
                "passed": _all_passed(gates),
            }
    if not (
        all(value["passed"] for value in fused_heldout_gates.values())
        and all(
            value["passed"] for lane in lane_gates.values() for value in lane.values()
        )
    ):
        raise Lfm2SemanticPostfoldBlockedError(
            "Fused final W2 regressed a held-out evidence lane",
            _sealed(
                {
                    "schema": LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA,
                    "status": "blocked-no-checkpoint-published",
                    "stage": "fused_semantic_terminal",
                    "reason": "separate-heldout-lane-regression",
                    "fused_heldout_gates": fused_heldout_gates,
                    "lane_gates": lane_gates,
                    "operation_audit": dict(_ZERO_OPERATIONS),
                },
                "failure_sha256",
            ),
        )
    if fingerprint_lfm2_parent(target_model).identity_sha256 != before_replay:
        raise Lfm2SemanticPostfoldError(
            "Fused terminal evidence replay mutated the child before installation."
        )
    global_inventory = _global_tokenizer_expansion_inventory(
        bridge,
        donor_tokenizer,
        target_tokenizer,
    )
    installation = _install_final_w2_once(
        target_model,
        published_down_weight=published_down_weight,
    )
    final_mlp_full_coverage_after_terminal = _lfm2_dense_producer_chain_coverage(
        target_model,
        layer_index=13,
        producer_kind="mlp",
        defer_down_weight=False,
    )
    if (
        final_mlp_full_coverage_after_terminal.get(
            "all_required_roles_structurally_nonzero"
        )
        is not True
        or final_mlp_full_coverage_after_terminal.get(
            "blank_or_partially_blank_required_role_count"
        )
        != 0
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused terminal did not leave a complete nonzero final MLP chain."
        )
    observed_counts = {value["inserted_continuation_rows"] for value in observed}
    decode_counts = {value["parent_decode_steps"] for value in observed}
    target_counts = {value["target_decode_tokens"] for value in observed}
    maximum_multiplicity = max(
        value["maximum_donor_span_multiplicity"] for value in observed
    )
    body = {
        "schema": LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA,
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "semantic_layer_stack_result_sha256": layer_receipt["result_sha256"],
        "split_plan": plan.as_dict(),
        "parent_cache_authentication": cache_receipt,
        "parent_final_norm_direct_local_source": parent_final_norm.as_dict(),
        "parent_final_norm_provenance": {
            "direct_local_full_state_identity_reconstructed": True,
            "reconstructed_parent_fingerprint": (
                parent_final_norm.final_norm.parent_fingerprint
            ),
            "matches_authenticated_cache_parent_fingerprint": (
                parent_final_norm.final_norm.parent_fingerprint
                == cache.parent_fingerprint
            ),
            "cached_final_residual_to_norm_input_exact": True,
            "runtime_equivalent_norm_replay_verified_at_every_phase": True,
            "authenticated_cached_norm_output_used_as_terminal_target": True,
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
        },
        "parent_final_norm_endpoint_sha256": [
            value.parent_final_norm_endpoint["authentication_sha256"]
            for value in cached
        ],
        "parent_action_projection_sha256": [
            value.parent_action_projection["authentication_sha256"] for value in cached
        ],
        "parent_endpoint_evidence_sha256": {
            split: value.authentication_sha256 for split, value in parent.items()
        },
        "lexical_evidence_sha256": {
            split: value.authentication_sha256 for split, value in lexical.items()
        },
        "deferred_final_layer_mlp_evidence_sha256": {
            split: value.authentication_sha256 for split, value in final_mlp.items()
        },
        "fused_evidence_sha256": {
            split: value.authentication_sha256 for split, value in fused.items()
        },
        "observed_trace_coverage": {
            "batch_count": len(observed),
            "per_batch": observed,
            "uniform_inserted_continuation_rows": (
                next(iter(observed_counts)) if len(observed_counts) == 1 else None
            ),
            "uniform_parent_decode_steps": (
                next(iter(decode_counts)) if len(decode_counts) == 1 else None
            ),
            "uniform_target_decode_tokens": (
                next(iter(target_counts)) if len(target_counts) == 1 else None
            ),
            "maximum_donor_span_multiplicity": maximum_multiplicity,
            "total_inserted_continuation_rows": sum(observed_counts)
            if len(observed) == 1
            else sum(value["inserted_continuation_rows"] for value in observed),
            "all_inserted_rows_supervised": True,
        },
        "global_tokenizer_expansion_inventory": dict(global_inventory),
        "observed_trace_is_not_global_lexical_proof": True,
        "lexical_teacher": dict(teacher_report),
        "terminal_result": terminal.as_dict(),
        "terminal_publication_decision": dict(publication_decision),
        "fused_heldout_publication_gates": fused_heldout_gates,
        "separate_heldout_lane_gates": lane_gates,
        "installation": dict(installation),
        "final_mlp_producer_coverage_before_terminal": dict(
            final_mlp_producer_coverage_before_terminal
        ),
        "final_mlp_full_coverage_after_terminal": dict(
            final_mlp_full_coverage_after_terminal
        ),
        "complete_nonzero_final_mlp_structural_coverage": True,
        "terminal_writer_decision_count": 1,
        "terminal_writer_install_count": installation["install_count"],
        "terminal_writer_mutation_count": installation["mutation_count"],
        "terminal_writer_incumbent_retention_count": installation[
            "incumbent_retention_count"
        ],
        "one_frozen_final_w2_publication_decision": True,
        "one_final_w2_install": installation["install_count"] == 1,
        "joint_terminal_evidence_compiled_and_heldout_adjudicated": True,
        "final_layer_mlp_parent_endpoint_and_lexical_rows_jointly_fitted": (
            publication_decision["terminal_nonincumbent_candidate_accepted"]
        ),
        "prior_final_layer_mlp_w2_install_count": 0,
        "parent_trajectory_replay_receipts": list(parent_replays),
        "parent_attention_rope_materializations": parent_rope_calls,
        "accepted": True,
        "terminal_publication_decision_accepted": True,
        "terminal_nonincumbent_candidate_accepted": publication_decision[
            "terminal_nonincumbent_candidate_accepted"
        ],
        "terminal_incumbent_retained": publication_decision[
            "terminal_incumbent_retained"
        ],
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


def _validate_fused_semantic_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    layer_receipt: Mapping[str, Any],
    current_model_state_sha256: str,
) -> Mapping[str, Any]:
    installation = receipt.get("installation")
    if not isinstance(installation, Mapping):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal returned no installation decision."
        )
    coverage_before_value = receipt.get(
        "final_mlp_producer_coverage_before_terminal"
    )
    coverage_after_value = receipt.get("final_mlp_full_coverage_after_terminal")
    published_coverage_value = installation.get(
        "published_w2_structural_coverage"
    )
    if not all(
        isinstance(value, Mapping)
        for value in (
            coverage_before_value,
            coverage_after_value,
            published_coverage_value,
        )
    ):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal omitted structural producer coverage."
        )
    coverage_before = _verify_sealed(
        coverage_before_value,
        seal_name="result_sha256",
        label="preterminal final-MLP producer coverage",
    )
    coverage_after = _verify_sealed(
        coverage_after_value,
        seal_name="result_sha256",
        label="published final-MLP producer coverage",
    )
    published_coverage = _verify_sealed(
        published_coverage_value,
        seal_name="result_sha256",
        label="published final-W2 structural coverage",
    )
    before_roles = coverage_before.get("role_receipts")
    after_roles = coverage_after.get("role_receipts")
    w1_w3_unchanged = bool(
        isinstance(before_roles, Mapping)
        and isinstance(after_roles, Mapping)
        and all(
            isinstance(before_roles.get(role), Mapping)
            and isinstance(after_roles.get(role), Mapping)
            and before_roles[role].get("tensor_sha256")
            == after_roles[role].get("tensor_sha256")
            for role in ("feed_forward.w1.weight", "feed_forward.w3.weight")
        )
    )
    counts = {
        "decision": receipt.get("terminal_writer_decision_count"),
        "install": receipt.get("terminal_writer_install_count"),
        "mutation": receipt.get("terminal_writer_mutation_count"),
        "retention": receipt.get("terminal_writer_incumbent_retention_count"),
    }
    if any(type(value) is not int for value in counts.values()):
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal decision counts are not exact integers."
        )
    operation_audit = receipt.get("operation_audit")
    invalid = bool(
        receipt.get("schema") != LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA
        or receipt.get("semantic_layer_stack_result_sha256")
        != layer_receipt.get("result_sha256")
        or receipt.get("accepted") is not True
        or receipt.get("terminal_publication_decision_accepted") is not True
        or counts["decision"] != 1
        or counts["install"] + counts["retention"] != 1
        or counts["mutation"] != counts["install"]
        or installation.get("decision_count") != 1
        or installation.get("install_count") != counts["install"]
        or installation.get("mutation_count") != counts["mutation"]
        or installation.get("incumbent_retention_count") != counts["retention"]
        or installation.get("parameter_topology_preserved") is not True
        or installation.get("parameter_identities_preserved") is not True
        or installation.get("parameter_storage_preserved") is not True
        or installation.get("gradient_state_preserved") is not True
        or installation.get("parameter_count_changed") is not False
        or installation.get("published_w2_structurally_nonzero") is not True
        or installation.get("published_w2_structural_coverage_sha256")
        != published_coverage.get("result_sha256")
        or published_coverage.get("structurally_nonzero") is not True
        or published_coverage.get("tensor_sha256")
        != installation.get("published_w2_sha256")
        or receipt.get("complete_nonzero_final_mlp_structural_coverage")
        is not True
        or coverage_before.get("producer_kind") != "mlp"
        or coverage_before.get("layer_index") != 13
        or coverage_before.get("deferred_roles")
        != ["feed_forward.w2.weight"]
        or coverage_before.get("all_required_roles_structurally_nonzero")
        is not True
        or coverage_after.get("producer_kind") != "mlp"
        or coverage_after.get("layer_index") != 13
        or coverage_after.get("deferred_roles") != []
        or coverage_after.get("all_required_roles_structurally_nonzero")
        is not True
        or coverage_after.get("blank_or_partially_blank_required_role_count")
        != 0
        or not w1_w3_unchanged
        or not isinstance(after_roles, Mapping)
        or not isinstance(after_roles.get("feed_forward.w2.weight"), Mapping)
        or after_roles["feed_forward.w2.weight"].get("tensor_sha256")
        != published_coverage.get("tensor_sha256")
        or installation.get("model_state_before_sha256")
        != layer_receipt.get("model_state_after_sha256")
        or installation.get("model_state_after_sha256")
        != current_model_state_sha256
        or not isinstance(operation_audit, Mapping)
        or operation_audit != dict(_ZERO_OPERATIONS)
        or layer_receipt.get("whole_stage_contains_meaningful_parameter_changes")
        is not True
        or layer_receipt.get("model_state_before_sha256")
        == current_model_state_sha256
    )
    if counts["install"] == 1:
        invalid = bool(
            invalid
            or installation.get("installed") is not True
            or installation.get("exact_zero_copy_noop") is not False
            or not installation.get("changed_state_keys")
            or installation.get("model_state_before_sha256")
            == installation.get("model_state_after_sha256")
            or receipt.get("terminal_nonincumbent_candidate_accepted") is not True
            or receipt.get("terminal_incumbent_retained") is not False
        )
    else:
        invalid = bool(
            invalid
            or installation.get("installed") is not False
            or installation.get("exact_zero_copy_noop") is not True
            or installation.get("parameter_copy_calls") != 0
            or installation.get("changed_state_keys") != []
            or installation.get("model_state_before_sha256")
            != installation.get("model_state_after_sha256")
            or receipt.get("terminal_nonincumbent_candidate_accepted") is not False
            or receipt.get("terminal_incumbent_retained") is not True
        )
    if invalid:
        raise Lfm2SemanticPostfoldError(
            "Fused semantic terminal receipt is incomplete or internally false."
        )
    return MappingProxyType(dict(receipt))


def _ffn_width(model: Any) -> int:
    widths = set()
    for layer in tuple(getattr(getattr(model, "model", None), "layers", ())):
        feed_forward = getattr(layer, "feed_forward", None)
        w1 = getattr(getattr(feed_forward, "w1", None), "weight", None)
        w2 = getattr(getattr(feed_forward, "w2", None), "weight", None)
        w3 = getattr(getattr(feed_forward, "w3", None), "weight", None)
        if w1 is None or w2 is None or w3 is None:
            continue
        if int(w1.shape[0]) != int(w3.shape[0]) or int(w2.shape[1]) != int(w1.shape[0]):
            raise Lfm2SemanticPostfoldError("Child FFN shapes are inconsistent.")
        widths.add(int(w1.shape[0]))
    if len(widths) != 1:
        raise Lfm2SemanticPostfoldError("Child does not expose one dense FFN width.")
    return next(iter(widths))


def _safe_remove_staging(path: Path, parent: Path) -> None:
    try:
        resolved = path.resolve()
        root = parent.resolve()
    except OSError:
        return
    if (
        resolved.parent == root
        and resolved.name.startswith(".semantic-postfold-")
        and resolved.is_dir()
        and not resolved.is_symlink()
    ):
        shutil.rmtree(resolved)


def _safe_remove_failed_publication(path: Path, expected: Path) -> None:
    """Remove only the exact new output created by this failed transaction."""

    try:
        resolved = path.resolve()
        intended = expected.resolve()
    except OSError:
        return
    if (
        resolved == intended
        and resolved.parent == intended.parent
        and resolved.is_dir()
        and not resolved.is_symlink()
    ):
        shutil.rmtree(resolved)


def publish_lfm2_semantic_postfold_checkpoint(
    *,
    target_model: Any,
    target_tokenizer: Any,
    output_directory: str | os.PathLike[str],
    context_binding_sha256: str,
    device: str,
    dtype: str = "bfloat16",
) -> tuple[Any, Mapping[str, Any]]:
    """Atomically save and twice reload one dense parent-free checkpoint."""

    output = _new_output_path(output_directory)
    context = _digest(context_binding_sha256, "postfold context binding")
    model_state = fingerprint_model_state(target_model)
    inventory = independent_parameter_inventory(target_model)
    inventory_sha = _digest(
        inventory.get("inventory_sha256"),
        "postfold parameter inventory",
    )
    tokenizer_sha = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    width = _ffn_width(target_model)
    saver = Lfm2CpcfV5QuarantineSaver(
        target_tokenizer=target_tokenizer,
        context_binding_sha256=context,
        expected_ffn_width=width,
        expected_model_state_sha256=model_state,
        expected_parameter_inventory_sha256=inventory_sha,
        expected_tokenizer_semantics_sha256=tokenizer_sha,
    )
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=context,
        device=device,
        dtype=dtype,
        expected_ffn_width=width,
        expected_model_state_sha256=model_state,
        expected_parameter_inventory_sha256=inventory_sha,
        expected_tokenizer_semantics_sha256=tokenizer_sha,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".semantic-postfold-",
            dir=str(output.parent),
        )
    ).resolve()
    publication_moved = False
    try:
        saver(target_model, staging)
        staged = loader(staging)
        if fingerprint_model_state(staged) != model_state:
            raise Lfm2SemanticPostfoldError(
                "Staged checkpoint reload differs from the postfold child."
            )
        os.replace(staging, output)
        publication_moved = True
        reloaded = loader(output)
        if fingerprint_model_state(reloaded) != model_state or _parameter_count(
            reloaded
        ) != _parameter_count(target_model):
            raise Lfm2SemanticPostfoldError(
                "Published checkpoint reload differs from the postfold child."
            )
    except BaseException:
        if staging.exists():
            _safe_remove_staging(staging, output.parent)
        if publication_moved and output.exists():
            _safe_remove_failed_publication(output, output)
        raise
    metadata = _input_artifact_metadata(
        output,
        expected_artifact_sha256=_read_json(
            output / LFM2_CPCF_V5_ARTIFACT_METADATA,
            "published artifact metadata",
        )["artifact_sha256"],
    )
    body = {
        "schema": LFM2_SEMANTIC_POSTFOLD_CHECKPOINT_SCHEMA,
        "output_directory": str(output),
        "context_binding_sha256": context,
        "artifact_sha256": metadata["artifact_sha256"],
        "model_state_sha256": model_state,
        "parameter_inventory_sha256": inventory_sha,
        "tokenizer_semantics_sha256": tokenizer_sha,
        "parameter_element_count": _parameter_count(reloaded),
        "materialized_ffn_width": width,
        "staged_reload_verified": True,
        "published_reload_verified": True,
        "standalone_dense_safetensors": True,
        "parent_model_required_for_reload": False,
        "cache_required_for_reload": False,
        "network_access_performed": False,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return reloaded, MappingProxyType(_sealed(body, "result_sha256"))


def run_lfm2_semantic_postfold(
    *,
    input_checkpoint_directory: str | os.PathLike[str],
    expected_input_artifact_sha256: str,
    output_checkpoint_directory: str | os.PathLike[str],
    cache_directory: str | os.PathLike[str],
    expected_cache_key: str,
    bridge_directory: str | os.PathLike[str],
    expected_bridge_artifact_sha256: str,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    parent_model_directory: str | os.PathLike[str],
    compiler_diagnostic_json: str | os.PathLike[str],
    expected_compiler_diagnostic_sha256: str,
    device: str,
    dtype: str = "bfloat16",
    parent_final_norm_tensor_key: str = "model.embedding_norm.weight",
    parent_norm_epsilon_config_key: str = "norm_eps",
    parent_tied_tensor_key: str = "model.embed_tokens.weight",
    attention_installer: Callable[..., Mapping[str, Any]] | None = None,
    semantic_layer_stack_compiler: Callable[..., Any] | None = None,
    fused_terminal_compiler: Callable[..., Mapping[str, Any]] = (
        compile_install_lfm2_fused_semantic_terminal_refold
    ),
    split_plan: Lfm2LongHorizonTrajectorySplitPlan | None = None,
) -> Mapping[str, Any]:
    """Execute the complete saved-child semantic postfold transaction."""

    output = _new_output_path(output_checkpoint_directory)
    cache = open_lfm2_circuit_trace_cache(cache_directory, verify=True)
    if cache.cache_key != _digest(expected_cache_key, "expected cache key"):
        raise Lfm2SemanticPostfoldError("Complete causal cache identity changed.")
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=expected_bridge_artifact_sha256,
    )
    target_model, saved_tokenizer, input_metadata = load_lfm2_cpcf_v5_standalone_child(
        input_checkpoint_directory,
        expected_artifact_sha256=expected_input_artifact_sha256,
        device=device,
        dtype=dtype,
    )
    if fingerprint_hf_tokenizer_semantics(
        saved_tokenizer
    ) != fingerprint_hf_tokenizer_semantics(target_tokenizer):
        raise Lfm2SemanticPostfoldError(
            "Saved child tokenizer differs from the target bridge package."
        )
    parent_norm = load_lfm2_parent_final_rmsnorm_from_local_safetensors(
        cache,
        parent_model_directory,
        tensor_key=parent_final_norm_tensor_key,
        epsilon_config_key=parent_norm_epsilon_config_key,
    )
    source_table, source_table_report = (
        load_lfm2_mapped_parent_tied_table_from_local_safetensors(
            parent_model_directory,
            cache=cache,
            bridge=bridge,
            authenticated_parent_final_norm=parent_norm,
            tensor_key=parent_tied_tensor_key,
        )
    )
    foundation_ancestry = _claims_semantic_foundation_schema(
        compiler_diagnostic_json
    )
    lineage = load_lfm2_semantic_compiler_lineage(
        compiler_diagnostic_json,
        expected_diagnostic_sha256=expected_compiler_diagnostic_sha256,
        cache=cache,
        bridge=bridge,
        input_metadata=input_metadata,
        **(
            {"parent_model_directory": parent_model_directory}
            if foundation_ancestry
            else {}
        ),
    )
    if foundation_ancestry:
        try:
            interface, interface_rehydration = (
                rehydrate_lfm2_semantic_foundation_interface(
                    target_model,
                    source_table,
                    semantic_cache=cache,
                    bridge=bridge,
                    target_tokenizer=target_tokenizer,
                    parent_model_directory=parent_model_directory,
                    input_metadata=input_metadata,
                    foundation_receipt_json=compiler_diagnostic_json,
                    expected_foundation_receipt_sha256=(
                        expected_compiler_diagnostic_sha256
                    ),
                )
            )
        except Lfm2SemanticFoundationError as exc:
            raise Lfm2SemanticPostfoldError(
                "Semantic-foundation interface rehydration failed."
            ) from exc
    else:
        interface, interface_rehydration = rehydrate_lfm2_c61_tied_interface(
            target_model,
            source_table,
            semantic_cache=cache,
            bridge=bridge,
            target_tokenizer=target_tokenizer,
            compiler_diagnostic_json=compiler_diagnostic_json,
            expected_compiler_diagnostic_sha256=(
                expected_compiler_diagnostic_sha256
            ),
        )
    layer_stack = compile_install_lfm2_semantic_layer_stack_stage(
        cache=cache,
        bridge=bridge,
        target_model=target_model,
        interface=interface,
        compiler_lineage=lineage,
        parent_model_directory=parent_model_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        attention_installer=attention_installer,
        semantic_layer_stack_compiler=semantic_layer_stack_compiler,
    )
    terminal = fused_terminal_compiler(
        cache=cache,
        bridge=bridge,
        target_model=target_model,
        layer_stack=layer_stack,
        interface=interface,
        source_selected_tied_table=source_table,
        parent_final_norm=parent_norm,
        split_plan=split_plan,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    terminal_receipt = _verify_sealed(
        terminal,
        seal_name="result_sha256",
        label="fused semantic terminal",
    )
    state_after_terminal = fingerprint_lfm2_parent(target_model).identity_sha256
    terminal_receipt = dict(
        _validate_fused_semantic_terminal_receipt(
            terminal_receipt,
            layer_receipt=layer_stack.receipt,
            current_model_state_sha256=state_after_terminal,
        )
    )
    context_body = {
        "schema": "neural-forge-lfm2-semantic-postfold-context-v1",
        "input_artifact_sha256": input_metadata["artifact_sha256"],
        "compiler_ancestry_schema": (
            LFM2_SEMANTIC_FOUNDATION_SCHEMA
            if foundation_ancestry
            else "neural-forge-lfm2-cpcf-v5-selective-blank-fold-run-v1"
        ),
        "compiler_cache_key": interface_rehydration["compiler_cache_key"],
        "semantic_evidence_cache_key": cache.cache_key,
        "common_parent_fingerprint": cache.parent_fingerprint,
        "compiler_and_semantic_cache_identities_bound_separately": True,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "mapped_parent_tied_table_sha256": source_table_report["result_sha256"],
        "rehydrated_interface_report_sha256": interface.report["report_sha256"],
        "interface_rehydration_sha256": interface_rehydration["result_sha256"],
        "compiler_diagnostic_sha256": _digest(
            expected_compiler_diagnostic_sha256,
            "expected compiler diagnostic SHA-256",
        ),
        "semantic_layer_stack_stage_sha256": layer_stack.receipt["result_sha256"],
        "fused_semantic_terminal_refold_sha256": terminal_receipt["result_sha256"],
        "complete_nonzero_structural_producer_coverage": layer_stack.receipt[
            "complete_nonzero_structural_producer_coverage"
        ],
        "complete_nonzero_final_mlp_structural_coverage": terminal_receipt[
            "complete_nonzero_final_mlp_structural_coverage"
        ],
    }
    context_sha = canonical_sha256(context_body)
    reloaded, checkpoint = publish_lfm2_semantic_postfold_checkpoint(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        output_directory=output,
        context_binding_sha256=context_sha,
        device=device,
        dtype=dtype,
    )
    state_after_reload = fingerprint_lfm2_parent(reloaded).identity_sha256
    body = {
        "schema": LFM2_SEMANTIC_POSTFOLD_SCHEMA,
        "status": "standalone-semantic-postfold-saved-and-reloaded",
        "input_child": {
            "checkpoint_directory": str(
                _ordinary_directory(input_checkpoint_directory, "input checkpoint")
            ),
            "artifact_sha256": input_metadata["artifact_sha256"],
            "model_state_sha256": input_metadata["model_state_sha256"],
        },
        "compiler_ancestry_schema": context_body["compiler_ancestry_schema"],
        "compiler_cache_key": interface_rehydration["compiler_cache_key"],
        "semantic_evidence_cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "mapped_parent_tied_table": dict(source_table_report),
        "rehydrated_interface": dict(interface.report),
        "interface_rehydration": dict(interface_rehydration),
        "compiler_lineage": {
            str(index): dict(value.report) for index, value in lineage.items()
        },
        "semantic_layer_stack": dict(layer_stack.receipt),
        "fused_semantic_terminal": terminal_receipt,
        "terminal_publication_decision_accepted": True,
        "terminal_nonincumbent_candidate_accepted": terminal_receipt[
            "terminal_nonincumbent_candidate_accepted"
        ],
        "terminal_incumbent_retained": terminal_receipt[
            "terminal_incumbent_retained"
        ],
        "checkpoint": dict(checkpoint),
        "fresh_reload_parent_free_state_sha256": state_after_reload,
        "context_binding": {**context_body, "context_sha256": context_sha},
        "training_replaced_by": (
            "sealed-cache semantic lattice transport plus closed-form linear solves"
        ),
        "operation_audit": dict(_ZERO_OPERATIONS),
        "claims": {
            "complete_causal_layer_stack_adjudicated": True,
            "complete_nonzero_structural_producer_coverage": (
                layer_stack.receipt[
                    "complete_nonzero_structural_producer_coverage"
                ]
            ),
            "blank_or_partially_blank_producer_chain_count": (
                layer_stack.receipt[
                    "blank_or_partially_blank_producer_chain_count"
                ]
            ),
            "complete_nonzero_final_mlp_structural_coverage": (
                terminal_receipt[
                    "complete_nonzero_final_mlp_structural_coverage"
                ]
            ),
            "complete_causal_layer_stack_installed": layer_stack.receipt[
                "complete_causal_layer_stack_installed"
            ],
            "attention_operator_decision_count": layer_stack.receipt[
                "attention_operator_decision_count"
            ],
            "attention_correction_install_count": layer_stack.receipt[
                "attention_operator_install_count"
            ],
            "attention_correction_installed": (
                layer_stack.receipt["attention_operator_install_count"] > 0
            ),
            "short_conv_operator_decision_count": layer_stack.receipt[
                "short_conv_operator_decision_count"
            ],
            "short_conv_correction_install_count": layer_stack.receipt[
                "short_conv_operator_install_count"
            ],
            "short_conv_correction_installed": (
                layer_stack.receipt["short_conv_operator_install_count"] > 0
            ),
            "replay_fresh_mlp_evidence_compile_count": layer_stack.receipt[
                "replay_fresh_mlp_evidence_compile_count"
            ],
            "nonterminal_mlp_writer_decision_count": layer_stack.receipt[
                "nonterminal_mlp_writer_decision_count"
            ],
            "replay_fresh_mlp_correction_install_count": layer_stack.receipt[
                "nonterminal_mlp_writer_install_count"
            ],
            "replay_fresh_mlp_corrections_installed": (
                layer_stack.receipt["nonterminal_mlp_writer_install_count"] > 0
            ),
            "final_mlp_producer_install_count": layer_stack.receipt[
                "final_mlp_producer_install_count"
            ],
            "terminal_writer_decision_count": terminal_receipt[
                "terminal_writer_decision_count"
            ],
            "terminal_writer_install_count": terminal_receipt[
                "terminal_writer_install_count"
            ],
            "fused_parent_endpoint_and_lexical_w2_installed_once": (
                terminal_receipt["terminal_writer_install_count"] == 1
            ),
            "terminal_w2_heldout_adjudicated": True,
            "overall_postfold_contains_meaningful_parameter_changes": True,
            "standalone_child": True,
            "parameter_count_compaction_changed_by_postfold": False,
            "training_performed": False,
            "pruning_performed": False,
            "quantization_performed": False,
        },
    }
    return MappingProxyType(_sealed(body, "result_sha256"))


__all__ = [
    "LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA",
    "LFM2_SEMANTIC_ATTENTION_STAGE_SCHEMA",
    "LFM2_SEMANTIC_COMPILER_LINEAGE_SCHEMA",
    "LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA",
    "LFM2_SEMANTIC_POSTFOLD_CHECKPOINT_SCHEMA",
    "LFM2_SEMANTIC_POSTFOLD_FAILURE_SCHEMA",
    "LFM2_SEMANTIC_POSTFOLD_SCHEMA",
    "LFM2_SEMANTIC_RESIDUAL_CHART_SCHEMA",
    "Lfm2SemanticAttentionTransportEvidence",
    "Lfm2SemanticCompilerLayerLineage",
    "Lfm2SemanticLayerStackInstallation",
    "Lfm2SemanticPostfoldBlockedError",
    "Lfm2SemanticPostfoldError",
    "SEMANTIC_ATTENTION_INSTALLER_CALLABLE",
    "SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE",
    "compile_install_lfm2_semantic_attention_stage",
    "compile_install_lfm2_fused_semantic_terminal_refold",
    "compile_install_lfm2_semantic_layer_stack_stage",
    "load_lfm2_cpcf_v5_standalone_child",
    "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
    "load_lfm2_parent_qkvo_inventory_from_local_safetensors",
    "load_lfm2_semantic_compiler_lineage",
    "publish_lfm2_semantic_postfold_checkpoint",
    "rehydrate_lfm2_c61_tied_interface",
    "reconstruct_lfm2_saved_tied_interface",
    "run_lfm2_semantic_postfold",
    "seal_lfm2_semantic_attention_installation_receipt",
]
