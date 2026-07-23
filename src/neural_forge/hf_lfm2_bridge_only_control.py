"""Authenticated same-body LFM2 tokenizer-bridge compaction control.

This is deliberately a diagnostic control, not a substitute for structural
folding.  The child keeps the source depth, hidden width, head geometry, and
body tensor shapes.  It starts as a separately synthesized dense zero blank,
receives every non-lexical source tensor through an exact identity transport,
and receives one smaller native tied embedding/readout table through the
sealed target-token-to-donor-row bridge.  The result therefore isolates the
behavioral cost of the tokenizer/interface contraction from body folding.

No forward pass, gradient, optimizer, pruning, quantization, sparsity, parent
sidecar, or runtime weight sharing is used.  Native input/output tying remains
part of the LFM2 architecture.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_BRIDGE_ONLY_CONTROL_SCHEMA = "neural-forge-lfm2-bridge-only-control-v1"


class Lfm2BridgeOnlyControlError(RuntimeError):
    """The same-body bridge control could not be authenticated."""


@dataclass(frozen=True, slots=True)
class Lfm2BridgeOnlyControlResult:
    """One installed child and its sealed structural receipt."""

    model: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.model is None:
            raise TypeError("model must be a materialized LFM2 child.")
        body = dict(self.receipt)
        supplied = body.pop("result_sha256", None)
        if supplied != canonical_sha256(body):
            raise Lfm2BridgeOnlyControlError(
                "Bridge-only control receipt seal is invalid."
            )
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("LFM2 bridge-only folding requires PyTorch.") from exc
    return torch


def _storage_identity(value: Any) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        tuple(int(size) for size in value.shape),
    )


def _named_values(model: Any, method: str) -> tuple[tuple[str, Any], ...]:
    callback = getattr(model, method, None)
    if not callable(callback):
        raise Lfm2BridgeOnlyControlError(f"Model has no {method}() method.")
    try:
        return tuple(callback(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        return tuple(callback())


def _all_named_tensors(model: Any) -> tuple[tuple[str, Any], ...]:
    return _named_values(model, "named_parameters") + _named_values(
        model, "named_buffers"
    )


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
    ):
        raise Lfm2BridgeOnlyControlError(
            "Bridge-only folding requires materialized dense tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _unique_parameter_count(model: Any) -> int:
    parameters = tuple(model.parameters())
    if not parameters:
        raise Lfm2BridgeOnlyControlError("Model has no learned parameters.")
    return sum(int(value.numel()) for value in parameters)


def _config_topology(config: Any) -> dict[str, Any]:
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)) or not layer_types:
        raise Lfm2BridgeOnlyControlError("LFM2 config has no layer topology.")
    hidden_size = int(getattr(config, "hidden_size"))
    query_heads = int(getattr(config, "num_attention_heads"))
    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is None:
        if hidden_size % query_heads:
            raise Lfm2BridgeOnlyControlError(
                "LFM2 hidden width is not divisible by its query-head count."
            )
        head_dim = hidden_size // query_heads
    else:
        head_dim = int(configured_head_dim)
    return {
        "model_type": getattr(config, "model_type", None),
        "vocabulary_size": int(getattr(config, "vocab_size")),
        "hidden_size": hidden_size,
        "intermediate_size": int(getattr(config, "intermediate_size")),
        "layer_count": int(getattr(config, "num_hidden_layers")),
        "layer_types": [str(value) for value in layer_types],
        "query_heads": query_heads,
        "key_value_heads": int(getattr(config, "num_key_value_heads")),
        "head_dim": head_dim,
        "conv_L_cache": int(getattr(config, "conv_L_cache")),
        "conv_bias": bool(getattr(config, "conv_bias")),
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings")),
    }


def _body_topology(topology: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(topology)
    result.pop("vocabulary_size", None)
    return result


def _interface_storage_ids(model: Any) -> set[tuple[int, int, tuple[int, ...]]]:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if input_weight is None or output_weight is None or input_weight is not output_weight:
        raise Lfm2BridgeOnlyControlError(
            "Bridge-only control requires one native tied embedding/readout table."
        )
    return {_storage_identity(input_weight)}


def _body_records(
    model: Any,
    *,
    interface_storage_ids: set[tuple[int, int, tuple[int, ...]]],
) -> tuple[dict[str, Any], ...]:
    records = []
    for name, value in _all_named_tensors(model):
        if _storage_identity(value) in interface_storage_ids:
            continue
        records.append(
            {
                "name": name,
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        )
    if not records:
        raise Lfm2BridgeOnlyControlError("LFM2 child has no non-interface body.")
    return tuple(records)


def fold_lfm2_bridge_only_control(
    parent_model: Any,
    blank_target_model: Any,
    *,
    bridge: StrictTokenizerBridgeArtifact,
    blank_target_receipt: Mapping[str, Any],
) -> Lfm2BridgeOnlyControlResult:
    """Install an exact source body and bridged tied table into one zero blank."""

    torch = _torch()
    if parent_model is blank_target_model:
        raise Lfm2BridgeOnlyControlError(
            "Parent and blank child must be distinct model objects."
        )
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    bridge.verify_files()
    blank = dict(blank_target_receipt)
    blank_sha = blank.pop("report_sha256", None)
    if (
        blank_sha != canonical_sha256(blank)
        or blank_target_receipt.get("compiler_created_blank") is not True
        or blank_target_receipt.get("all_parameters_zero") is not True
        or blank_target_receipt.get("dense_materialized_parameters") is not True
        or blank_target_receipt.get("training_used") is not False
        or blank_target_receipt.get("pruning_used") is not False
        or blank_target_receipt.get("quantization_used") is not False
        or blank_target_receipt.get("weights_installed") is not False
    ):
        raise Lfm2BridgeOnlyControlError(
            "Target is not authenticated as a separately synthesized dense zero blank."
        )

    parent_topology = _config_topology(parent_model.config)
    target_topology = _config_topology(blank_target_model.config)
    if (
        parent_topology["model_type"] != "lfm2"
        or target_topology["model_type"] != "lfm2"
        or _body_topology(parent_topology) != _body_topology(target_topology)
        or parent_topology["tie_word_embeddings"] is not True
        or target_topology["tie_word_embeddings"] is not True
    ):
        raise Lfm2BridgeOnlyControlError(
            "Bridge-only control requires exact parent body/depth/width topology."
        )

    parent_interface_ids = _interface_storage_ids(parent_model)
    target_interface_ids = _interface_storage_ids(blank_target_model)
    parent_input = parent_model.get_input_embeddings().weight
    target_input = blank_target_model.get_input_embeddings().weight
    profile = bridge.transport_profile
    mapping = tuple(int(value) for value in profile.target_to_donor_row_indices)
    if (
        int(parent_input.shape[0]) != profile.donor_row_capacity
        or int(target_input.shape[0]) != profile.target_row_capacity
        or parent_topology["vocabulary_size"] != profile.donor_row_capacity
        or target_topology["vocabulary_size"] != profile.target_row_capacity
        or int(parent_input.shape[1]) != int(target_input.shape[1])
        or len(mapping) != profile.target_row_capacity
        or len(set(mapping)) != len(mapping)
        or min(mapping, default=-1) < 0
        or max(mapping, default=-1) >= profile.donor_row_capacity
    ):
        raise Lfm2BridgeOnlyControlError(
            "Sealed token bridge does not exactly address the source/target tables."
        )

    parent_parameters = _unique_parameter_count(parent_model)
    target_parameters = _unique_parameter_count(blank_target_model)
    expected_target_parameters = (
        parent_parameters
        - profile.donor_row_capacity * int(parent_input.shape[1])
        + profile.target_row_capacity * int(target_input.shape[1])
    )
    if (
        target_parameters != expected_target_parameters
        or target_parameters >= parent_parameters
        or blank_target_receipt.get("parent_unique_parameters") != parent_parameters
        or blank_target_receipt.get("target_unique_parameters") != target_parameters
    ):
        raise Lfm2BridgeOnlyControlError(
            "Blank parameter inventory is not the exact vocabulary-only contraction."
        )

    if any(
        parameter.layout != torch.strided
        or int(torch.count_nonzero(parameter.detach()).item()) != 0
        for parameter in blank_target_model.parameters()
    ):
        raise Lfm2BridgeOnlyControlError(
            "Compiler-authenticated blank contains nonzero or non-dense parameters."
        )

    parent_state_before = fingerprint_model_state(parent_model)
    blank_state_sha256 = fingerprint_model_state(blank_target_model)
    parent_named = dict(_all_named_tensors(parent_model))
    target_named = dict(_all_named_tensors(blank_target_model))
    parent_body_names = {
        name
        for name, value in parent_named.items()
        if _storage_identity(value) not in parent_interface_ids
    }
    target_body_names = {
        name
        for name, value in target_named.items()
        if _storage_identity(value) not in target_interface_ids
    }
    if parent_body_names != target_body_names:
        raise Lfm2BridgeOnlyControlError(
            "Parent and blank child non-interface state inventories differ."
        )
    for name in sorted(parent_body_names):
        source = parent_named[name]
        target = target_named[name]
        if tuple(source.shape) != tuple(target.shape):
            raise Lfm2BridgeOnlyControlError(
                f"Body tensor shape differs for {name!r}."
            )

    mapped_table = parent_input.detach().index_select(
        0,
        torch.tensor(mapping, dtype=torch.long, device=parent_input.device),
    ).to(device=target_input.device, dtype=target_input.dtype)
    mapped_table_sha256 = _tensor_sha256(mapped_table)
    with torch.no_grad():
        for name in sorted(target_body_names):
            target_named[name].copy_(
                parent_named[name].detach().to(
                    device=target_named[name].device,
                    dtype=target_named[name].dtype,
                )
            )
        target_input.copy_(mapped_table)

    target_after_sha256 = fingerprint_model_state(blank_target_model)
    parent_state_after = fingerprint_model_state(parent_model)
    source_body = _body_records(
        parent_model,
        interface_storage_ids=parent_interface_ids,
    )
    installed_body = _body_records(
        blank_target_model,
        interface_storage_ids=target_interface_ids,
    )
    if (
        source_body != installed_body
        or target_after_sha256 == blank_state_sha256
        or parent_state_after != parent_state_before
        or _tensor_sha256(target_input) != mapped_table_sha256
        or blank_target_model.get_input_embeddings().weight
        is not blank_target_model.get_output_embeddings().weight
        or _unique_parameter_count(blank_target_model) != target_parameters
    ):
        raise Lfm2BridgeOnlyControlError(
            "Installed bridge-only child failed exact structural replay."
        )

    body = {
        "schema": LFM2_BRIDGE_ONLY_CONTROL_SCHEMA,
        "status": "same-body-bridge-only-child-installed",
        "diagnostic_role": (
            "isolates-tokenizer-and-tied-interface-contraction-from-body-folding"
        ),
        "parent_topology": parent_topology,
        "target_topology": target_topology,
        "exact_parent_body_depth_width_preserved": True,
        "body_transport": "exact-name-and-shape-identity-transport",
        "body_tensor_records": list(installed_body),
        "body_tensor_inventory_sha256": canonical_sha256(list(installed_body)),
        "body_tensor_count": len(installed_body),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "target_to_donor_row_indices_sha256": canonical_sha256(list(mapping)),
        "mapped_parent_tied_table_sha256": mapped_table_sha256,
        "tied_interface_transport": (
            "exact-target-to-donor-row-select-with-identity-hidden-chart"
        ),
        "native_input_output_tying_preserved": True,
        "parent_unique_parameters": parent_parameters,
        "target_unique_parameters": target_parameters,
        "realized_parameter_compaction_ratio": parent_parameters / target_parameters,
        "parameter_reduction_is_vocabulary_rows_only": True,
        "blank_target_report_sha256": blank_sha,
        "blank_model_state_sha256": blank_state_sha256,
        "installed_model_state_sha256": target_after_sha256,
        "parent_model_state_before_sha256": parent_state_before,
        "parent_model_state_after_sha256": parent_state_after,
        "parent_state_preserved": True,
        "standalone_save_reload_pending": True,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "child_generate_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "external_adapter_parameters": 0,
            "parent_sidecar_parameters": 0,
            "runtime_shared_parent_parameters": 0,
            "network_access": False,
        },
    }
    return Lfm2BridgeOnlyControlResult(
        model=blank_target_model,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


__all__ = [
    "LFM2_BRIDGE_ONLY_CONTROL_SCHEMA",
    "Lfm2BridgeOnlyControlError",
    "Lfm2BridgeOnlyControlResult",
    "fold_lfm2_bridge_only_control",
]
