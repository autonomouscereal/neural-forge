"""Typed serialization and restoration bridge for LFM2 CPCF resume artifacts.

This module is deliberately smaller than a continuation orchestrator.  It
only translates already-authenticated in-memory CPCF objects to and from the
immutable resume-artifact store, reconstructs one archived phase-state bank,
and transactionally installs one archived target prefix into a caller-created
blank of the exact same topology.

It never loads a model, executes a model, copies parent evidence into the
store, trains, prunes, quantizes, or claims that an intermediate artifact is a
successful compacted child.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import (
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_closed_form_causal_circuit_fold import (
    PredictiveBoundaryMap,
)
from neural_forge.hf_cpcf_boundary_chart import CpcfBoundaryChart
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
    Lfm2CpcfBoundaryBundle,
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    LFM2_CPCF_TARGET_PHASE_SCHEMA,
    LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FAILURE_SCHEMA,
    LFM2_CPCF_MODEL_PROGRESS_EVENT_SCHEMA,
    Lfm2CpcfModelFoldFailure,
    Lfm2CpcfModelFoldProgressEvent,
)
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    Lfm2CpcfArtifactReference,
    Lfm2CpcfBoundaryBindings,
    Lfm2CpcfFailureBindings,
    Lfm2CpcfResumeArtifactIntegrityError,
    Lfm2CpcfResumeArtifactStore,
    VerifiedLfm2CpcfBoundaryBundle,
    VerifiedLfm2CpcfFailureDiagnostic,
    VerifiedLfm2CpcfPrefixCheckpoint,
    build_lfm2_cpcf_checkpoint_bindings,
    build_lfm2_cpcf_completed_layers_document,
    build_lfm2_cpcf_installed_parameter_paths,
    build_lfm2_cpcf_target_topology_snapshot,
    flatten_lfm2_cpcf_incoming_state_bank,
)
from neural_forge.hf_tied_causal_interface_fold import (
    TiedCausalInterfaceFold,
)


LFM2_CPCF_RESUME_BOUNDARY_BRIDGE_SCHEMA = (
    "neural-forge-lfm2-cpcf-resume-boundary-bridge-v1"
)
LFM2_CPCF_PREFIX_INSTALL_REPORT_SCHEMA = "neural-forge-lfm2-cpcf-prefix-install-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BRIDGE_FIELDS = frozenset(
    {
        "schema",
        "bundle_schema",
        "bundle_sha256",
        "bundle_report",
        "interface",
        "layers",
        "split_phase_sha256",
        "split_row_sha256",
        "tensor_inventory",
        "bridge_sha256",
    }
)
_INTERFACE_FIELDS = frozenset(
    {
        "target_weight_tensor",
        "producer_encoder_tensor",
        "report",
    }
)
_LAYER_FIELDS = frozenset(
    {
        "layer_index",
        "operator_kind",
        "layer_charts_sha256",
        "post_operator",
        "post_layer",
    }
)
_CHART_FIELDS = frozenset(
    {
        "name",
        "report",
        "mapping_report",
        "encoder_tensor",
        "decoder_tensor",
        "feature_predictors",
    }
)
_PREDICTOR_FIELDS = frozenset({"name", "tensor"})
_TENSOR_FIELDS = frozenset(
    {
        "name",
        "dtype",
        "shape",
        "numel",
        "element_size",
        "byte_sha256",
        "runtime_tensor_sha256",
    }
)
_SPLITS = ("selection", "fit", "validation")


class Lfm2CpcfResumeBridgeError(Lfm2CpcfResumeArtifactIntegrityError):
    """A typed CPCF object differs from its immutable resume representation."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for the CPCF resume bridge.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfResumeBridgeError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _canonical(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(_deep_thaw(value)).decode("ascii"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfResumeBridgeError(
            f"{name} must contain finite canonical JSON."
        ) from exc


def _canonical_mapping(value: Any, name: str) -> dict[str, Any]:
    result = _canonical(value, name)
    if not isinstance(result, dict):
        raise Lfm2CpcfResumeBridgeError(f"{name} must be a JSON object.")
    return result


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    return value


def _tensor_bytes(value: Any, name: str) -> tuple[Any, bytes]:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
        or value.is_complex()
        or not torch.is_floating_point(value)
        or value.numel() == 0
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2CpcfResumeBridgeError(
            f"{name} must be detached, dense, finite, real floating evidence."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(tensor).all().item()):
        raise Lfm2CpcfResumeBridgeError(f"{name} contains NaN or infinity.")
    payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return tensor, payload


def _runtime_tensor_sha256(value: Any) -> str:
    tensor, payload = _tensor_bytes(value, "tensor")
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(canonical_json_bytes(descriptor))
    digest.update(payload)
    return digest.hexdigest()


def _phase_tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
        or value.requires_grad
    ):
        raise Lfm2CpcfResumeBridgeError(
            f"{name} must be detached, dense, materialized phase evidence."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(canonical_json_bytes(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _reauthenticate_target_phase(phase: Lfm2CpcfTargetPhase) -> None:
    parent_evidence_sha = _digest(
        getattr(getattr(phase, "parent_evidence", None), "evidence_sha256", None),
        "phase parent evidence SHA-256",
    )
    body = {
        "schema": LFM2_CPCF_TARGET_PHASE_SCHEMA,
        "batch_index": phase.batch_index,
        "record_id": phase.record_id,
        "split": phase.split,
        "phase": phase.phase,
        "donor_input_sha256": phase.donor_input_sha256,
        "target_alignment_sha256": phase.target_alignment_sha256,
        "input_ids_sha256": _phase_tensor_sha256(
            phase.input_ids,
            "phase input_ids",
        ),
        "attention_mask_sha256": _phase_tensor_sha256(
            phase.attention_mask,
            "phase attention_mask",
        ),
        "position_ids_sha256": _phase_tensor_sha256(
            phase.position_ids,
            "phase position_ids",
        ),
        "batch_size": phase.batch_size,
        "full_prefix_tokens": phase.full_prefix_tokens,
        "sampled_suffix_tokens": phase.sampled_suffix_tokens,
        "parent_evidence_sha256": parent_evidence_sha,
        "numeric_token_id_equality_required": False,
        "actual_child_full_prefix_required": True,
    }
    if canonical_sha256(body) != phase.phase_sha256:
        raise Lfm2CpcfResumeBridgeError(
            "Authenticated target phase changed after its typed seal."
        )


def _interface_tensor_sha256(value: Any) -> str:
    """Match the tied-interface and boundary-bundle tensor seal."""

    tensor, payload = _tensor_bytes(value, "interface tensor")
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
    digest.update(payload)
    return digest.hexdigest()


def _predictive_tensor_sha256(value: Any) -> str:
    """Match the predictive-boundary-map encoder/decoder seal."""

    tensor, payload = _tensor_bytes(value, "predictive tensor")
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        canonical_sha256([int(size) for size in tensor.shape]).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _tensor_record(name: str, value: Any) -> dict[str, Any]:
    tensor, payload = _tensor_bytes(value, name)
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    runtime = hashlib.sha256(canonical_json_bytes(descriptor))
    runtime.update(payload)
    return {
        "name": name,
        "dtype": descriptor["dtype"],
        "shape": descriptor["shape"],
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "byte_sha256": hashlib.sha256(payload).hexdigest(),
        "runtime_tensor_sha256": runtime.hexdigest(),
    }


def _checkpoint_tensor_content_sha256(value: Any, name: str) -> str:
    """Match the resume store's raw-byte digest for any dense model tensor."""

    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2CpcfResumeBridgeError(
            f"{name} must be detached, dense, materialized checkpoint evidence."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
        torch.isfinite(tensor).all().item()
    ):
        raise Lfm2CpcfResumeBridgeError(f"{name} contains NaN or infinity.")
    payload = (
        b""
        if tensor.numel() == 0
        else tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def _self_sealed_report(value: Any, name: str) -> dict[str, Any]:
    report = _canonical_mapping(value, name)
    body = dict(report)
    claimed = _digest(body.pop("report_sha256", None), f"{name}.report_sha256")
    if canonical_sha256(body) != claimed:
        raise Lfm2CpcfResumeBridgeError(f"{name} report seal is invalid.")
    return report


def _chart_payload(
    chart: CpcfBoundaryChart,
    *,
    layer_index: int,
    slot: str,
    tensors: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(chart, CpcfBoundaryChart):
        raise Lfm2CpcfResumeBridgeError(f"{slot} is not a CpcfBoundaryChart.")
    mapping = chart.mapping
    if not isinstance(mapping, PredictiveBoundaryMap):
        raise Lfm2CpcfResumeBridgeError(f"{slot} mapping is not predictive.")
    prefix = f"layer.{layer_index:08d}.{slot}"
    encoder_name = f"{prefix}.encoder"
    decoder_name = f"{prefix}.decoder"
    tensors[encoder_name] = (
        mapping.encoder.detach().to(device="cpu").contiguous().clone()
    )
    tensors[decoder_name] = (
        mapping.decoder.detach().to(device="cpu").contiguous().clone()
    )
    predictors = []
    for predictor_index, feature_name in enumerate(sorted(mapping.feature_predictors)):
        tensor_name = f"{prefix}.predictor.{predictor_index:08d}"
        tensors[tensor_name] = (
            mapping.feature_predictors[feature_name]
            .detach()
            .to(device="cpu")
            .contiguous()
            .clone()
        )
        predictors.append({"name": feature_name, "tensor": tensor_name})
    return {
        "name": chart.name,
        "report": _canonical_mapping(chart.report, f"{slot} chart report"),
        "mapping_report": _canonical_mapping(
            mapping.report,
            f"{slot} mapping report",
        ),
        "encoder_tensor": encoder_name,
        "decoder_tensor": decoder_name,
        "feature_predictors": predictors,
    }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfFlattenedBoundaryBundle:
    """A complete boundary bundle represented by canonical reports and tensors."""

    reports: Mapping[str, Any]
    tensors: Mapping[str, Any]

    def __post_init__(self) -> None:
        reports, tensors = _validate_boundary_payload(self.reports, self.tensors)
        object.__setattr__(self, "reports", _deep_freeze(reports))
        object.__setattr__(
            self,
            "tensors",
            MappingProxyType(
                {
                    name: tensor.detach().to(device="cpu").contiguous().clone()
                    for name, tensor in tensors.items()
                }
            ),
        )

    @property
    def bridge_sha256(self) -> str:
        return str(self.reports["bridge_sha256"])


def flatten_lfm2_cpcf_boundary_bundle(
    bundle: Lfm2CpcfBoundaryBundle,
) -> Lfm2CpcfFlattenedBoundaryBundle:
    """Losslessly flatten every bundle tensor, report, split hash, and layer seal."""

    if not isinstance(bundle, Lfm2CpcfBoundaryBundle):
        raise TypeError("bundle must be Lfm2CpcfBoundaryBundle.")
    tensors: dict[str, Any] = {
        "interface.target_weight": (
            bundle.interface.target_weight.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
        ),
        "interface.producer_encoder": (
            bundle.interface.producer_encoder.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
        ),
    }
    layers = []
    for expected_index, layer in enumerate(bundle.layers):
        if layer.layer_index != expected_index:
            raise Lfm2CpcfResumeBridgeError(
                "Boundary bundle layer inventory is not contiguous."
            )
        layers.append(
            {
                "layer_index": expected_index,
                "operator_kind": layer.operator_kind,
                "layer_charts_sha256": layer.layer_charts_sha256,
                "post_operator": _chart_payload(
                    layer.post_operator,
                    layer_index=expected_index,
                    slot="post_operator",
                    tensors=tensors,
                ),
                "post_layer": _chart_payload(
                    layer.post_layer,
                    layer_index=expected_index,
                    slot="post_layer",
                    tensors=tensors,
                ),
            }
        )
    body = {
        "schema": LFM2_CPCF_RESUME_BOUNDARY_BRIDGE_SCHEMA,
        "bundle_schema": LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_report": _canonical_mapping(bundle.report, "bundle report"),
        "interface": {
            "target_weight_tensor": "interface.target_weight",
            "producer_encoder_tensor": "interface.producer_encoder",
            "report": _canonical_mapping(
                bundle.interface.report,
                "interface report",
            ),
        },
        "layers": layers,
        "split_phase_sha256": {
            split: bundle.split_phase_sha256[split] for split in _SPLITS
        },
        "split_row_sha256": {
            split: bundle.split_row_sha256[split] for split in _SPLITS
        },
        "tensor_inventory": [
            _tensor_record(name, tensors[name]) for name in sorted(tensors)
        ],
    }
    reports = {**body, "bridge_sha256": canonical_sha256(body)}
    return Lfm2CpcfFlattenedBoundaryBundle(reports=reports, tensors=tensors)


def _validate_chart_payload(
    value: Any,
    *,
    layer_index: int,
    slot: str,
    tensors: Mapping[str, Any],
) -> dict[str, Any]:
    chart = _canonical_mapping(value, f"layer {layer_index} {slot}")
    if set(chart) != _CHART_FIELDS:
        raise Lfm2CpcfResumeBridgeError(
            f"Layer {layer_index} {slot} has an unexpected field set."
        )
    expected_name = f"layer.{layer_index:03d}.residual." + (
        "post_operator" if slot == "post_operator" else "post_layer"
    )
    if chart.get("name") != expected_name:
        raise Lfm2CpcfResumeBridgeError(
            f"Layer {layer_index} {slot} name is not canonical."
        )
    chart_report = _self_sealed_report(
        chart.get("report"),
        f"layer {layer_index} {slot} chart",
    )
    mapping_report = _self_sealed_report(
        chart.get("mapping_report"),
        f"layer {layer_index} {slot} mapping",
    )
    if chart_report.get("predictive_mapping") != mapping_report:
        raise Lfm2CpcfResumeBridgeError(
            f"Layer {layer_index} {slot} chart lost its predictive report."
        )
    encoder_name = chart.get("encoder_tensor")
    decoder_name = chart.get("decoder_tensor")
    if (
        not isinstance(encoder_name, str)
        or not isinstance(decoder_name, str)
        or encoder_name not in tensors
        or decoder_name not in tensors
    ):
        raise Lfm2CpcfResumeBridgeError(
            f"Layer {layer_index} {slot} mapping tensors are absent."
        )
    if mapping_report.get("encoder_sha256") != _predictive_tensor_sha256(
        tensors[encoder_name]
    ) or mapping_report.get("decoder_sha256") != _predictive_tensor_sha256(
        tensors[decoder_name]
    ):
        raise Lfm2CpcfResumeBridgeError(
            f"Layer {layer_index} {slot} mapping report differs from its tensors."
        )
    predictors = chart.get("feature_predictors")
    if not isinstance(predictors, list) or not predictors:
        raise Lfm2CpcfResumeBridgeError(
            f"Layer {layer_index} {slot} has no feature predictors."
        )
    names: list[str] = []
    tensor_names: list[str] = []
    for predictor in predictors:
        if not isinstance(predictor, dict) or set(predictor) != _PREDICTOR_FIELDS:
            raise Lfm2CpcfResumeBridgeError(
                f"Layer {layer_index} {slot} predictor inventory is invalid."
            )
        name = predictor.get("name")
        tensor_name = predictor.get("tensor")
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or not isinstance(tensor_name, str)
            or tensor_name not in tensors
        ):
            raise Lfm2CpcfResumeBridgeError(
                f"Layer {layer_index} {slot} predictor is invalid."
            )
        names.append(name)
        tensor_names.append(tensor_name)
    if names != sorted(set(names)) or len(set(tensor_names)) != len(tensor_names):
        raise Lfm2CpcfResumeBridgeError(
            f"Layer {layer_index} {slot} predictors are duplicated or unordered."
        )
    return chart


def _validate_boundary_payload(
    reports_value: Any,
    tensors_value: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reports = _canonical_mapping(reports_value, "boundary bridge reports")
    if set(reports) != _BRIDGE_FIELDS:
        raise Lfm2CpcfResumeBridgeError(
            "Boundary bridge report has an unexpected field set."
        )
    body = dict(reports)
    claimed_bridge = _digest(
        body.pop("bridge_sha256", None),
        "boundary bridge SHA-256",
    )
    if (
        reports.get("schema") != LFM2_CPCF_RESUME_BOUNDARY_BRIDGE_SCHEMA
        or reports.get("bundle_schema") != LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA
        or canonical_sha256(body) != claimed_bridge
    ):
        raise Lfm2CpcfResumeBridgeError("Boundary bridge report seal is invalid.")
    if not isinstance(tensors_value, Mapping) or not tensors_value:
        raise Lfm2CpcfResumeBridgeError("Boundary bridge tensors cannot be empty.")
    tensors: dict[str, Any] = {}
    for name, value in tensors_value.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise Lfm2CpcfResumeBridgeError("Boundary tensor name is invalid.")
        tensor, _payload = _tensor_bytes(value, f"boundary tensor {name!r}")
        tensors[name] = tensor
    inventory = reports.get("tensor_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise Lfm2CpcfResumeBridgeError("Boundary tensor inventory is empty.")
    expected_names = []
    for record in inventory:
        if not isinstance(record, dict) or set(record) != _TENSOR_FIELDS:
            raise Lfm2CpcfResumeBridgeError(
                "Boundary tensor inventory has an unexpected field set."
            )
        name = record.get("name")
        if not isinstance(name, str) or name not in tensors:
            raise Lfm2CpcfResumeBridgeError(
                "Boundary tensor inventory names an absent tensor."
            )
        if record != _tensor_record(name, tensors[name]):
            raise Lfm2CpcfResumeBridgeError(
                f"Boundary tensor {name!r} differs byte-for-byte from its report."
            )
        expected_names.append(name)
    if expected_names != sorted(set(expected_names)) or set(tensors) != set(
        expected_names
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Boundary tensors are missing, extra, duplicated, or unordered."
        )
    bundle_report = _canonical_mapping(
        reports.get("bundle_report"),
        "boundary bundle report",
    )
    bundle_body = dict(bundle_report)
    bundle_sha = _digest(
        bundle_body.pop("bundle_sha256", None),
        "boundary bundle SHA-256",
    )
    if (
        bundle_sha != reports.get("bundle_sha256")
        or canonical_sha256(bundle_body) != bundle_sha
        or bundle_report.get("schema") != LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Boundary bundle report is not exact or self-sealed."
        )
    interface = _canonical_mapping(reports.get("interface"), "interface payload")
    if set(interface) != _INTERFACE_FIELDS:
        raise Lfm2CpcfResumeBridgeError(
            "Boundary interface payload has an unexpected field set."
        )
    target_name = interface.get("target_weight_tensor")
    encoder_name = interface.get("producer_encoder_tensor")
    if (
        target_name not in tensors
        or encoder_name not in tensors
        or not isinstance(target_name, str)
        or not isinstance(encoder_name, str)
    ):
        raise Lfm2CpcfResumeBridgeError("Boundary interface tensors are absent.")
    interface_report = _self_sealed_report(
        interface.get("report"),
        "active interface",
    )
    if interface_report.get("target_weight_sha256") != _runtime_tensor_sha256(
        tensors[target_name]
    ) or interface_report.get("producer_encoder_sha256") != _runtime_tensor_sha256(
        tensors[encoder_name]
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Active interface report differs from its exact tensors."
        )
    layers = reports.get("layers")
    if not isinstance(layers, list) or not layers:
        raise Lfm2CpcfResumeBridgeError("Boundary layer inventory cannot be empty.")
    used_tensor_names = {target_name, encoder_name}
    for expected_index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != _LAYER_FIELDS:
            raise Lfm2CpcfResumeBridgeError(
                "Boundary layer payload has an unexpected field set."
            )
        if layer.get("layer_index") != expected_index or layer.get(
            "operator_kind"
        ) not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfResumeBridgeError(
                "Boundary layer inventory is incomplete or unordered."
            )
        _digest(
            layer.get("layer_charts_sha256"),
            f"layer {expected_index} chart seal",
        )
        for slot in ("post_operator", "post_layer"):
            chart = _validate_chart_payload(
                layer.get(slot),
                layer_index=expected_index,
                slot=slot,
                tensors=tensors,
            )
            used_tensor_names.add(chart["encoder_tensor"])
            used_tensor_names.add(chart["decoder_tensor"])
            used_tensor_names.update(
                predictor["tensor"] for predictor in chart["feature_predictors"]
            )
    if used_tensor_names != set(tensors):
        raise Lfm2CpcfResumeBridgeError(
            "Boundary tensor inventory contains an unreferenced tensor."
        )
    for field in ("split_phase_sha256", "split_row_sha256"):
        hashes = reports.get(field)
        if not isinstance(hashes, dict) or set(hashes) != set(_SPLITS):
            raise Lfm2CpcfResumeBridgeError(
                f"{field} must exactly cover selection, fit, and validation."
            )
        for split in _SPLITS:
            _digest(hashes[split], f"{field}[{split!r}]")
        if bundle_report.get(field) != hashes:
            raise Lfm2CpcfResumeBridgeError(
                f"{field} differs from the sealed bundle report."
            )
    if bundle_report.get("layer_count") != len(layers) or bundle_report.get(
        "layer_operator_kinds"
    ) != [layer["operator_kind"] for layer in layers]:
        raise Lfm2CpcfResumeBridgeError(
            "Boundary layer inventory differs from the sealed bundle report."
        )
    return reports, tensors


def reconstruct_lfm2_cpcf_boundary_bundle(
    *,
    reports: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> Lfm2CpcfBoundaryBundle:
    """Reconstruct a typed bundle only from a complete authenticated payload."""

    normalized_reports, normalized_tensors = _validate_boundary_payload(
        reports,
        tensors,
    )
    interface_payload = normalized_reports["interface"]
    interface = TiedCausalInterfaceFold(
        target_weight=normalized_tensors[interface_payload["target_weight_tensor"]],
        producer_encoder=normalized_tensors[
            interface_payload["producer_encoder_tensor"]
        ],
        report=interface_payload["report"],
    )
    layers = []
    for layer_payload in normalized_reports["layers"]:
        charts = {}
        for slot in ("post_operator", "post_layer"):
            chart_payload = layer_payload[slot]
            predictors = {
                item["name"]: normalized_tensors[item["tensor"]]
                for item in chart_payload["feature_predictors"]
            }
            mapping = PredictiveBoundaryMap(
                encoder=normalized_tensors[chart_payload["encoder_tensor"]],
                decoder=normalized_tensors[chart_payload["decoder_tensor"]],
                feature_predictors=predictors,
                report=chart_payload["mapping_report"],
            )
            charts[slot] = CpcfBoundaryChart(
                name=chart_payload["name"],
                mapping=mapping,
                report=chart_payload["report"],
            )
        layers.append(
            Lfm2CpcfLayerBoundaryCharts(
                layer_index=layer_payload["layer_index"],
                operator_kind=layer_payload["operator_kind"],
                post_operator=charts["post_operator"],
                post_layer=charts["post_layer"],
                layer_charts_sha256=layer_payload["layer_charts_sha256"],
            )
        )
    bundle = Lfm2CpcfBoundaryBundle(
        interface=interface,
        layers=tuple(layers),
        split_phase_sha256=normalized_reports["split_phase_sha256"],
        split_row_sha256=normalized_reports["split_row_sha256"],
        report=normalized_reports["bundle_report"],
        bundle_sha256=normalized_reports["bundle_sha256"],
    )
    round_trip = flatten_lfm2_cpcf_boundary_bundle(bundle)
    if (
        _deep_thaw(round_trip.reports) != normalized_reports
        or set(round_trip.tensors) != set(normalized_tensors)
        or any(
            _tensor_record(name, round_trip.tensors[name])
            != _tensor_record(name, normalized_tensors[name])
            for name in normalized_tensors
        )
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Reconstructed boundary bundle is not a lossless round trip."
        )
    return bundle


def publish_lfm2_cpcf_boundary_bundle(
    store: Lfm2CpcfResumeArtifactStore,
    bundle: Lfm2CpcfBoundaryBundle,
    *,
    bindings: Lfm2CpcfBoundaryBindings,
) -> VerifiedLfm2CpcfBoundaryBundle:
    """Publish one typed boundary bundle with its bridge digest bound externally."""

    if not isinstance(store, Lfm2CpcfResumeArtifactStore):
        raise TypeError("store must be Lfm2CpcfResumeArtifactStore.")
    if not isinstance(bindings, Lfm2CpcfBoundaryBindings):
        raise TypeError("bindings must be Lfm2CpcfBoundaryBindings.")
    flattened = flatten_lfm2_cpcf_boundary_bundle(bundle)
    if bindings.bridge_sha256 != flattened.bridge_sha256:
        raise Lfm2CpcfResumeBridgeError(
            "Boundary bindings differ from the exact flattened bridge payload."
        )
    published = store.publish_boundary(
        bindings=bindings,
        reports=_deep_thaw(flattened.reports),
        tensors=flattened.tensors,
    )
    verified = store.final_reauthenticate_boundary(published)
    reconstructed = reconstruct_lfm2_cpcf_boundary_bundle(
        reports=verified.reports,
        tensors=verified.tensors,
    )
    store.final_reauthenticate_boundary(verified)
    if reconstructed.bundle_sha256 != bundle.bundle_sha256:
        raise Lfm2CpcfResumeBridgeError(
            "Published boundary reconstructed to a different bundle."
        )
    return verified


def open_lfm2_cpcf_boundary_bundle(
    store: Lfm2CpcfResumeArtifactStore,
    reference: Lfm2CpcfArtifactReference,
    *,
    expected_bindings: Lfm2CpcfBoundaryBindings | None = None,
) -> Lfm2CpcfBoundaryBundle:
    """Open, reauthenticate, and reconstruct a pinned boundary reference."""

    if not isinstance(store, Lfm2CpcfResumeArtifactStore):
        raise TypeError("store must be Lfm2CpcfResumeArtifactStore.")
    verified = store.open_boundary(
        reference,
        expected_bindings=expected_bindings,
    )
    verified = store.final_reauthenticate_boundary(verified)
    reports = _deep_thaw(verified.reports)
    bridge_sha = _digest(
        reports.get("bridge_sha256"),
        "stored boundary bridge SHA-256",
    )
    manifest_bindings = _deep_thaw(verified.manifest["bindings"])
    if manifest_bindings.get("bridge_sha256") != bridge_sha:
        raise Lfm2CpcfResumeBridgeError(
            "Stored boundary bindings do not bind the bridge payload."
        )
    bundle = reconstruct_lfm2_cpcf_boundary_bundle(
        reports=reports,
        tensors=verified.tensors,
    )
    store.final_reauthenticate_boundary(verified)
    return bundle


@dataclass(frozen=True, slots=True)
class Lfm2CpcfFailureResumePublications:
    """The three non-promotable publications derived from one model failure."""

    boundary: VerifiedLfm2CpcfBoundaryBundle
    checkpoint: VerifiedLfm2CpcfPrefixCheckpoint
    failure: VerifiedLfm2CpcfFailureDiagnostic


@dataclass(frozen=True, slots=True)
class Lfm2CpcfProgressCheckpointPublications:
    """One pinned resumable prefix and its exact external lineage bindings."""

    boundary: VerifiedLfm2CpcfBoundaryBundle
    checkpoint: VerifiedLfm2CpcfPrefixCheckpoint
    event_report_sha256: str
    full_policy_sha256: str
    compiler_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, VerifiedLfm2CpcfBoundaryBundle):
            raise TypeError("boundary must be a verified boundary bundle.")
        if not isinstance(self.checkpoint, VerifiedLfm2CpcfPrefixCheckpoint):
            raise TypeError("checkpoint must be a verified prefix checkpoint.")
        event_sha = _digest(
            self.event_report_sha256,
            "event_report_sha256",
        )
        policy_sha = _digest(
            self.full_policy_sha256,
            "full_policy_sha256",
        )
        compiler_sha = _digest(
            self.compiler_fingerprint_sha256,
            "compiler_fingerprint_sha256",
        )
        bindings = _canonical_mapping(
            self.checkpoint.manifest.get("bindings"),
            "progress checkpoint bindings",
        )
        if (
            bindings.get("boundary") != self.boundary.reference.as_dict()
            or bindings.get("full_policy_sha256") != policy_sha
            or bindings.get("compiler_fingerprint_sha256") != compiler_sha
        ):
            raise Lfm2CpcfResumeBridgeError(
                "Progress checkpoint publications have inconsistent lineage."
            )
        object.__setattr__(self, "event_report_sha256", event_sha)
        object.__setattr__(self, "full_policy_sha256", policy_sha)
        object.__setattr__(self, "compiler_fingerprint_sha256", compiler_sha)


def _progress_boundary_reference(
    value: (Lfm2CpcfArtifactReference | VerifiedLfm2CpcfBoundaryBundle | None),
) -> Lfm2CpcfArtifactReference | None:
    if value is None:
        return None
    if isinstance(value, VerifiedLfm2CpcfBoundaryBundle):
        return value.reference
    if (
        isinstance(value, Lfm2CpcfArtifactReference)
        and value.artifact_kind == "boundary"
    ):
        return value
    raise TypeError(
        "pinned_boundary must be a verified boundary, boundary reference, or None."
    )


def _progress_checkpoint_reference(
    value: (Lfm2CpcfArtifactReference | VerifiedLfm2CpcfPrefixCheckpoint | None),
) -> Lfm2CpcfArtifactReference | None:
    if value is None:
        return None
    if isinstance(value, VerifiedLfm2CpcfPrefixCheckpoint):
        return value.reference
    if (
        isinstance(value, Lfm2CpcfArtifactReference)
        and value.artifact_kind == "checkpoint"
    ):
        return value
    raise TypeError(
        "previous_checkpoint must be a verified checkpoint, checkpoint "
        "reference, or None."
    )


def _reference_from_exact_progress_mapping(
    value: Any,
    *,
    artifact_kind: str,
    name: str,
) -> Lfm2CpcfArtifactReference:
    mapping = _canonical_mapping(value, name)
    if set(mapping) != {"artifact_kind", "artifact_key", "manifest_sha256"}:
        raise Lfm2CpcfResumeBridgeError(
            f"{name} does not have the exact artifact-reference field set."
        )
    try:
        reference = Lfm2CpcfArtifactReference(**mapping)
    except (TypeError, Lfm2CpcfResumeArtifactIntegrityError) as exc:
        raise Lfm2CpcfResumeBridgeError(f"{name} is invalid.") from exc
    if reference.artifact_kind != artifact_kind:
        raise Lfm2CpcfResumeBridgeError(
            f"{name} is not a pinned {artifact_kind} reference."
        )
    return reference


def _verified_or_published_progress_boundary(
    store: Lfm2CpcfResumeArtifactStore,
    bundle: Lfm2CpcfBoundaryBundle,
    *,
    bindings: Lfm2CpcfBoundaryBindings,
    pinned_reference: Lfm2CpcfArtifactReference | None,
) -> VerifiedLfm2CpcfBoundaryBundle:
    flattened = flatten_lfm2_cpcf_boundary_bundle(bundle)
    if bindings.bridge_sha256 != flattened.bridge_sha256:
        raise Lfm2CpcfResumeBridgeError(
            "Progress boundary bindings differ from the sealed event bundle."
        )
    if pinned_reference is None:
        return publish_lfm2_cpcf_boundary_bundle(
            store,
            bundle,
            bindings=bindings,
        )
    verified = store.open_boundary(
        pinned_reference,
        expected_bindings=bindings,
    )
    verified = store.final_reauthenticate_boundary(verified)
    stored = Lfm2CpcfFlattenedBoundaryBundle(
        reports=verified.reports,
        tensors=verified.tensors,
    )
    if (
        _deep_thaw(stored.reports) != _deep_thaw(flattened.reports)
        or set(stored.tensors) != set(flattened.tensors)
        or any(
            _tensor_record(name, stored.tensors[name])
            != _tensor_record(name, flattened.tensors[name])
            for name in flattened.tensors
        )
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Pinned progress boundary differs from the sealed event bundle."
        )
    reconstructed = reconstruct_lfm2_cpcf_boundary_bundle(
        reports=stored.reports,
        tensors=stored.tensors,
    )
    if reconstructed.bundle_sha256 != bundle.bundle_sha256:
        raise Lfm2CpcfResumeBridgeError(
            "Pinned progress boundary reconstructs to a different bundle."
        )
    store.final_reauthenticate_boundary(verified)
    return verified


def _verified_previous_progress_checkpoint(
    store: Lfm2CpcfResumeArtifactStore,
    reference: Lfm2CpcfArtifactReference | None,
    *,
    boundary_reference: Lfm2CpcfArtifactReference | None,
    full_policy_sha256: str,
    compiler_fingerprint_sha256: str,
    current_next_layer_index: int,
    current_receipts: Sequence[Mapping[str, Any]],
) -> tuple[
    VerifiedLfm2CpcfPrefixCheckpoint | None,
    Lfm2CpcfArtifactReference | None,
]:
    if reference is None:
        return None, boundary_reference
    previous = store.open_checkpoint(reference)
    previous = store.final_reauthenticate_checkpoint(previous)
    bindings = _canonical_mapping(
        previous.manifest.get("bindings"),
        "previous progress checkpoint bindings",
    )
    previous_boundary = _reference_from_exact_progress_mapping(
        bindings.get("boundary"),
        artifact_kind="boundary",
        name="previous progress checkpoint boundary",
    )
    if boundary_reference is not None and previous_boundary != boundary_reference:
        raise Lfm2CpcfResumeBridgeError(
            "Pinned boundary and previous progress checkpoint disagree."
        )
    previous_completed = _canonical_mapping(
        previous.completed_layers,
        "previous completed layers",
    )
    previous_next = previous_completed.get("next_layer_index")
    previous_receipts = previous_completed.get("completed_layer_receipts")
    if (
        bindings.get("full_policy_sha256") != full_policy_sha256
        or bindings.get("compiler_fingerprint_sha256") != compiler_fingerprint_sha256
        or isinstance(previous_next, bool)
        or not isinstance(previous_next, int)
        or previous_next < 0
        or previous_next > current_next_layer_index
        or not isinstance(previous_receipts, list)
        or previous_receipts
        != [
            _canonical_mapping(receipt, "current completed layer receipt")
            for receipt in current_receipts[:previous_next]
        ]
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Previous progress checkpoint policy, compiler, or receipt lineage "
            "differs from the current event."
        )
    store.final_reauthenticate_checkpoint(previous)
    return previous, previous_boundary


def publish_lfm2_cpcf_model_fold_progress_checkpoint(
    store: Lfm2CpcfResumeArtifactStore,
    event: Lfm2CpcfModelFoldProgressEvent,
    *,
    boundary_bindings: Lfm2CpcfBoundaryBindings,
    full_policy_sha256: str,
    compiler_fingerprint_sha256: str,
    global_installed_parameter_paths: Sequence[str] = (),
    pinned_boundary: (
        Lfm2CpcfArtifactReference | VerifiedLfm2CpcfBoundaryBundle | None
    ) = None,
    previous_checkpoint: (
        Lfm2CpcfArtifactReference | VerifiedLfm2CpcfPrefixCheckpoint | None
    ) = None,
) -> Lfm2CpcfProgressCheckpointPublications:
    """Persist one sealed, resumable model-fold progress event.

    The first call publishes the event's boundary bundle.  Later calls pass the
    returned boundary and checkpoint, causing exact pinned reauthentication
    instead of copying or republishing boundary evidence.  Only target-prefix
    tensors and child incoming states are archived; cached parent evidence is
    represented solely by the scalar phase references already sealed by the
    progress event.
    """

    if not isinstance(store, Lfm2CpcfResumeArtifactStore):
        raise TypeError("store must be Lfm2CpcfResumeArtifactStore.")
    if not isinstance(event, Lfm2CpcfModelFoldProgressEvent):
        raise TypeError("event must be Lfm2CpcfModelFoldProgressEvent.")
    if not isinstance(boundary_bindings, Lfm2CpcfBoundaryBindings):
        raise TypeError("boundary_bindings must be Lfm2CpcfBoundaryBindings.")
    policy_sha = _digest(full_policy_sha256, "full_policy_sha256")
    compiler_sha = _digest(
        compiler_fingerprint_sha256,
        "compiler_fingerprint_sha256",
    )
    try:
        event.verify()
    except Exception as exc:
        raise Lfm2CpcfResumeBridgeError(
            "Model-fold progress event failed typed verification."
        ) from exc
    event_report = _canonical_mapping(event.report, "progress event report")
    event_report_sha = _digest(
        event.report_sha256,
        "progress event report SHA-256",
    )
    if (
        event_report.get("schema") != LFM2_CPCF_MODEL_PROGRESS_EVENT_SCHEMA
        or event_report.get("stage") != "progress_checkpoint_publish"
        or event_report.get("report_sha256") != event_report_sha
        or event.full_policy_sha256 != policy_sha
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Progress event schema, stage, report, or policy binding differs."
        )
    bundle = event.boundary_bundle
    next_layer_index = event.next_layer_index
    if next_layer_index >= len(bundle.layers):
        raise Lfm2CpcfResumeBridgeError(
            "Final all-layers progress is not a resumable prefix checkpoint."
        )
    receipts = tuple(
        _canonical_mapping(
            receipt,
            f"completed layer receipt {index}",
        )
        for index, receipt in enumerate(event.completed_layer_receipts)
    )
    if (
        next_layer_index < 0
        or len(receipts) != next_layer_index
        or event_report.get("completed_layer_count") != next_layer_index
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Progress event receipts do not end at the incoming layer."
        )
    pinned_reference = _progress_boundary_reference(pinned_boundary)
    previous_reference = _progress_checkpoint_reference(previous_checkpoint)
    _previous, pinned_reference = _verified_previous_progress_checkpoint(
        store,
        previous_reference,
        boundary_reference=pinned_reference,
        full_policy_sha256=policy_sha,
        compiler_fingerprint_sha256=compiler_sha,
        current_next_layer_index=next_layer_index,
        current_receipts=receipts,
    )
    boundary = _verified_or_published_progress_boundary(
        store,
        bundle,
        bindings=boundary_bindings,
        pinned_reference=pinned_reference,
    )
    incoming_tensors, incoming_metadata = flatten_lfm2_cpcf_incoming_state_bank(
        event.incoming_state_bank
    )
    if incoming_metadata["layer_index"] != next_layer_index:
        raise Lfm2CpcfResumeBridgeError(
            "Progress incoming child-state bank ends at a different layer."
        )
    prefix_model = event.target_prefix_cpu_tensor_snapshot
    prefix_metadata = event.target_prefix_snapshot_metadata
    topology = build_lfm2_cpcf_target_topology_snapshot(
        prefix_model=prefix_model,
        layer_count=len(bundle.layers),
        snapshot_metadata=prefix_metadata,
    )
    if _previous is not None:
        previous_bindings = _canonical_mapping(
            _previous.manifest.get("bindings"),
            "previous checkpoint bindings",
        )
        if (
            previous_bindings.get("target_topology_sha256")
            != topology["topology_sha256"]
        ):
            raise Lfm2CpcfResumeBridgeError(
                "Progress target topology differs from the previous checkpoint."
            )
    installed_paths = build_lfm2_cpcf_installed_parameter_paths(
        target_topology_snapshot=topology,
        next_layer_index=next_layer_index,
        global_parameter_paths=global_installed_parameter_paths,
    )
    if _previous is not None:
        previous_completed = _canonical_mapping(
            _previous.completed_layers,
            "previous completed layers",
        )
        previous_next = int(previous_completed["next_layer_index"])
        expected_previous_paths = build_lfm2_cpcf_installed_parameter_paths(
            target_topology_snapshot=topology,
            next_layer_index=previous_next,
            global_parameter_paths=global_installed_parameter_paths,
        )
        if previous_completed.get("installed_parameter_paths") != (
            expected_previous_paths
        ):
            raise Lfm2CpcfResumeBridgeError(
                "Progress global install inventory differs from the previous "
                "checkpoint."
            )
        if previous_next == next_layer_index:
            previous_bindings = _canonical_mapping(
                _previous.manifest.get("bindings"),
                "previous checkpoint bindings",
            )
            prefix_entry_sha = {
                name: _checkpoint_tensor_content_sha256(tensor, name)
                for name, tensor in prefix_model.items()
            }
            incoming_entry_sha = {
                name: _checkpoint_tensor_content_sha256(tensor, name)
                for name, tensor in incoming_tensors.items()
            }
            if (
                previous_bindings.get("prefix_model_entry_sha256") != prefix_entry_sha
                or previous_bindings.get("incoming_state_entry_sha256")
                != incoming_entry_sha
            ):
                raise Lfm2CpcfResumeBridgeError(
                    "Duplicate progress index differs from the pinned prefix "
                    "or incoming child-state bytes."
                )
    checkpoint_bindings = build_lfm2_cpcf_checkpoint_bindings(
        boundary=boundary.reference,
        completed_layer_receipts=receipts,
        installed_parameter_paths=installed_paths,
        target_topology_snapshot=topology,
        incoming_state_bank_metadata=incoming_metadata,
        full_policy_sha256=policy_sha,
        compiler_fingerprint_sha256=compiler_sha,
    )
    completed_document = build_lfm2_cpcf_completed_layers_document(
        bindings=checkpoint_bindings,
        completed_layer_receipts=receipts,
        installed_parameter_paths=installed_paths,
        target_topology_snapshot=topology,
        incoming_state_bank=incoming_tensors,
        incoming_state_bank_metadata=incoming_metadata,
    )
    checkpoint = store.publish_checkpoint(
        bindings=checkpoint_bindings,
        prefix_model=prefix_model,
        incoming_state_bank=incoming_tensors,
        completed_layers=completed_document,
    )
    checkpoint = store.final_reauthenticate_checkpoint(checkpoint)
    event.verify()
    if event.report_sha256 != event_report_sha:
        raise Lfm2CpcfResumeBridgeError(
            "Progress event changed during checkpoint publication."
        )
    store.final_reauthenticate_boundary(boundary)
    store.final_reauthenticate_checkpoint(checkpoint)
    return Lfm2CpcfProgressCheckpointPublications(
        boundary=boundary,
        checkpoint=checkpoint,
        event_report_sha256=event_report_sha,
        full_policy_sha256=policy_sha,
        compiler_fingerprint_sha256=compiler_sha,
    )


def publish_lfm2_cpcf_model_fold_failure_context(
    store: Lfm2CpcfResumeArtifactStore,
    failure: Lfm2CpcfModelFoldFailure,
    *,
    boundary_bindings: Lfm2CpcfBoundaryBindings,
    full_policy_sha256: str,
    compiler_fingerprint_sha256: str,
    invocation_sha256: str,
    guard_proof_sha256: str,
    parent_proof_sha256: str,
    rollback_proof_sha256: str,
    global_installed_parameter_paths: Sequence[str] = (),
) -> Lfm2CpcfFailureResumePublications:
    """Publish a resumable failed prefix without copying parent evidence tensors."""

    if not isinstance(store, Lfm2CpcfResumeArtifactStore):
        raise TypeError("store must be Lfm2CpcfResumeArtifactStore.")
    if not isinstance(failure, Lfm2CpcfModelFoldFailure):
        raise TypeError("failure must be Lfm2CpcfModelFoldFailure.")
    if not isinstance(boundary_bindings, Lfm2CpcfBoundaryBindings):
        raise TypeError("boundary_bindings must be Lfm2CpcfBoundaryBindings.")
    full_policy_sha = _digest(full_policy_sha256, "full_policy_sha256")
    compiler_sha = _digest(
        compiler_fingerprint_sha256,
        "compiler_fingerprint_sha256",
    )
    invocation_sha = _digest(invocation_sha256, "invocation_sha256")
    expected_guard = _digest(guard_proof_sha256, "guard_proof_sha256")
    expected_parent = _digest(parent_proof_sha256, "parent_proof_sha256")
    expected_rollback = _digest(
        rollback_proof_sha256,
        "rollback_proof_sha256",
    )
    bundle = failure.boundary_bundle
    incoming_snapshot = failure.current_incoming_bank
    if bundle is None or incoming_snapshot is None:
        raise Lfm2CpcfResumeBridgeError(
            "Failure does not contain a resumable boundary and incoming bank."
        )
    report = _canonical_mapping(failure.report, "model fold failure report")
    if report.get("schema") != LFM2_CPCF_MODEL_FAILURE_SCHEMA:
        raise Lfm2CpcfResumeBridgeError(
            "Failure is not an authenticated LFM2 CPCF model failure."
        )
    rollback = _canonical_mapping(
        failure.rollback_proof,
        "model fold rollback proof",
    )
    snapshot_receipt = _canonical_mapping(
        report.get("failed_prefix_target_cpu_tensor_snapshot"),
        "failed prefix snapshot receipt",
    )
    observed_guard = _digest(
        snapshot_receipt.get("snapshot_sha256"),
        "failed prefix snapshot SHA-256",
    )
    observed_parent = _digest(
        rollback.get("source_state_fingerprint_before"),
        "model parent proof",
    )
    observed_rollback = _digest(
        rollback.get("proof_sha256"),
        "model rollback proof",
    )
    if (
        expected_guard != observed_guard
        or expected_parent != observed_parent
        or expected_rollback != observed_rollback
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Caller-supplied failure proof bindings differ from typed failure context."
        )
    boundary = publish_lfm2_cpcf_boundary_bundle(
        store,
        bundle,
        bindings=boundary_bindings,
    )
    incoming_tensors, incoming_metadata = flatten_lfm2_cpcf_incoming_state_bank(
        incoming_snapshot
    )
    prefix_model = failure.prefix_target_cpu_tensor_snapshot
    topology = build_lfm2_cpcf_target_topology_snapshot(
        prefix_model=prefix_model,
        layer_count=len(bundle.layers),
        snapshot_metadata=failure.prefix_target_snapshot_metadata,
    )
    receipts = failure.completed_layer_receipts
    installed_paths = build_lfm2_cpcf_installed_parameter_paths(
        target_topology_snapshot=topology,
        next_layer_index=len(receipts),
        global_parameter_paths=global_installed_parameter_paths,
    )
    checkpoint_bindings = build_lfm2_cpcf_checkpoint_bindings(
        boundary=boundary.reference,
        completed_layer_receipts=receipts,
        installed_parameter_paths=installed_paths,
        target_topology_snapshot=topology,
        incoming_state_bank_metadata=incoming_metadata,
        full_policy_sha256=full_policy_sha,
        compiler_fingerprint_sha256=compiler_sha,
    )
    completed_document = build_lfm2_cpcf_completed_layers_document(
        bindings=checkpoint_bindings,
        completed_layer_receipts=receipts,
        installed_parameter_paths=installed_paths,
        target_topology_snapshot=topology,
        incoming_state_bank=incoming_tensors,
        incoming_state_bank_metadata=incoming_metadata,
    )
    checkpoint = store.publish_checkpoint(
        bindings=checkpoint_bindings,
        prefix_model=prefix_model,
        incoming_state_bank=incoming_tensors,
        completed_layers=completed_document,
    )
    checkpoint = store.final_reauthenticate_checkpoint(checkpoint)
    active_index = report.get("active_layer_index")
    active_kind = report.get("active_operator_kind")
    if (active_index is None) != (active_kind is None):
        raise Lfm2CpcfResumeBridgeError("Failure active layer identity is incomplete.")
    failure_bindings = Lfm2CpcfFailureBindings(
        invocation_sha256=invocation_sha,
        layer_index=active_index,
        operator_id=active_kind,
        stage_id=str(report.get("stage")),
        guard_proof_sha256=observed_guard,
        parent_proof_sha256=observed_parent,
        rollback_proof_sha256=observed_rollback,
        boundary=boundary.reference,
        checkpoint=checkpoint.reference,
    )
    diagnostic = store.publish_failure(
        bindings=failure_bindings,
        report=report,
    )
    diagnostic = store.open_failure(diagnostic.artifact_key)
    return Lfm2CpcfFailureResumePublications(
        boundary=boundary,
        checkpoint=checkpoint,
        failure=diagnostic,
    )


def reconstruct_lfm2_cpcf_phase_state_bank(
    store: Lfm2CpcfResumeArtifactStore,
    checkpoint: Lfm2CpcfArtifactReference | VerifiedLfm2CpcfPrefixCheckpoint,
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> Lfm2CpcfPhaseStateBank:
    """Reconstruct the typed incoming state bank from a pinned checkpoint."""

    if not isinstance(store, Lfm2CpcfResumeArtifactStore):
        raise TypeError("store must be Lfm2CpcfResumeArtifactStore.")
    if isinstance(checkpoint, VerifiedLfm2CpcfPrefixCheckpoint):
        reference = checkpoint.reference
    elif isinstance(checkpoint, Lfm2CpcfArtifactReference):
        reference = checkpoint
    else:
        raise TypeError("checkpoint must be a verified checkpoint or reference.")
    verified = store.open_checkpoint(reference)
    verified = store.final_reauthenticate_checkpoint(verified)
    ordered_phases = tuple(phases)
    if not ordered_phases or not all(
        isinstance(phase, Lfm2CpcfTargetPhase) for phase in ordered_phases
    ):
        raise Lfm2CpcfResumeBridgeError(
            "phases must contain authenticated Lfm2CpcfTargetPhase objects."
        )
    for phase in ordered_phases:
        _reauthenticate_target_phase(phase)
    completed = _deep_thaw(verified.completed_layers)
    metadata = _canonical_mapping(
        completed.get("incoming_state_bank_metadata"),
        "incoming state-bank metadata",
    )
    if (
        metadata.get("schema") != LFM2_CPCF_TARGET_RUNTIME_SCHEMA
        or metadata.get("scope") != "phase_state_bank"
        or metadata.get("actual_folded_upstream_states") is not True
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Checkpoint incoming bank does not have the typed runtime schema."
        )
    states_archive = verified.incoming_state_bank
    ordered_states = metadata.get("ordered_states")
    if not isinstance(ordered_states, list) or len(ordered_states) != len(
        ordered_phases
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Checkpoint phase inventory differs from authenticated phases."
        )
    states = {}
    expected_tensor_names = set()
    for phase, entry in zip(ordered_phases, ordered_states, strict=True):
        if not isinstance(entry, dict):
            raise Lfm2CpcfResumeBridgeError(
                "Checkpoint incoming state descriptor is invalid."
            )
        identity = entry.get("identity")
        tensor_name = entry.get("tensor_name")
        if (
            identity != [phase.batch_index, phase.phase]
            or entry.get("phase_sha256") != phase.phase_sha256
            or not isinstance(tensor_name, str)
            or tensor_name not in states_archive
        ):
            raise Lfm2CpcfResumeBridgeError(
                "Authenticated phase differs from checkpoint phase inventory."
            )
        tensor = states_archive[tensor_name]
        if _runtime_tensor_sha256(tensor) != entry.get("state_sha256"):
            raise Lfm2CpcfResumeBridgeError(
                "Checkpoint state bytes differ from their runtime seal."
            )
        states[phase.identity] = tensor
        expected_tensor_names.add(tensor_name)
    if expected_tensor_names != set(states_archive):
        raise Lfm2CpcfResumeBridgeError(
            "Checkpoint incoming state archive has missing or extra tensors."
        )
    bank = Lfm2CpcfPhaseStateBank(
        boundary=metadata["boundary"],
        layer_index=metadata["layer_index"],
        phases=ordered_phases,
        states=states,
        state_bank_sha256=metadata["state_bank_sha256"],
    )
    if bank.state_bank_sha256 != completed.get("incoming_state_bank_metadata", {}).get(
        "state_bank_sha256"
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Reconstructed phase-state bank differs from checkpoint seal."
        )
    store.final_reauthenticate_checkpoint(verified)
    return bank


def _named_values_all(model: Any, method_name: str) -> tuple[tuple[str, Any], ...]:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise Lfm2CpcfResumeBridgeError(f"Target must expose {method_name}().")
    try:
        values = tuple(method(remove_duplicate=False))
    except TypeError as exc:
        raise Lfm2CpcfResumeBridgeError(
            f"Target {method_name}() must expose aliases with remove_duplicate=False."
        ) from exc
    names = [str(name) for name, _value in values]
    if len(names) != len(set(names)):
        raise Lfm2CpcfResumeBridgeError(
            f"Target {method_name}() contains duplicate names."
        )
    return tuple((str(name), value) for name, value in values)


def _alias_groups(values: Mapping[str, Any], *, storage: bool) -> list[list[str]]:
    groups: dict[Any, list[str]] = {}
    for name, value in values.items():
        if storage and value.numel() != 0:
            key = (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
        elif storage:
            key = ("empty", id(value))
        else:
            key = id(value)
        groups.setdefault(key, []).append(name)
    return sorted([sorted(names) for names in groups.values()])


def _target_inventory(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    parameters = _named_values_all(model, "named_parameters")
    buffers = _named_values_all(model, "named_buffers")
    modules = dict(_named_values_all(model, "named_modules"))
    values = dict(parameters + buffers)
    if len(values) != len(parameters) + len(buffers) or not values:
        raise Lfm2CpcfResumeBridgeError(
            "Target parameter and buffer inventory collides or is empty."
        )
    parameter_names = {name for name, _value in parameters}
    metadata = {}
    for name, value in values.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type == "meta"
            or value.layout != torch.strided
            or value.is_quantized
        ):
            raise Lfm2CpcfResumeBridgeError(
                "Target tensors must be materialized, dense, and unquantized."
            )
        if name in parameter_names:
            role = "parameter"
            persistent = True
        else:
            owner_name, _separator, local_name = name.rpartition(".")
            try:
                owner = modules[owner_name]
            except KeyError as exc:
                raise Lfm2CpcfResumeBridgeError(
                    f"Buffer owner for {name!r} is absent."
                ) from exc
            role = "buffer"
            persistent = local_name not in owner._non_persistent_buffers_set
        metadata[name] = {
            "role": role,
            "persistent": persistent,
            "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype).removeprefix("torch."),
            "stride": [int(size) for size in value.stride()],
            "storage_offset": int(value.storage_offset()),
            "original_device_type": value.device.type,
            "original_device_index": value.device.index,
        }
    return values, {
        "tensor_metadata": metadata,
        "object_alias_groups": _alias_groups(values, storage=False),
        "storage_alias_groups": _alias_groups(values, storage=True),
    }


def _verified_checkpoint(
    store: Lfm2CpcfResumeArtifactStore,
    checkpoint: Lfm2CpcfArtifactReference | VerifiedLfm2CpcfPrefixCheckpoint,
) -> VerifiedLfm2CpcfPrefixCheckpoint:
    if isinstance(checkpoint, VerifiedLfm2CpcfPrefixCheckpoint):
        reference = checkpoint.reference
    elif isinstance(checkpoint, Lfm2CpcfArtifactReference):
        reference = checkpoint
    else:
        raise TypeError("checkpoint must be a verified checkpoint or reference.")
    return store.final_reauthenticate_checkpoint(store.open_checkpoint(reference))


def install_lfm2_cpcf_prefix_checkpoint(
    store: Lfm2CpcfResumeArtifactStore,
    checkpoint: Lfm2CpcfArtifactReference | VerifiedLfm2CpcfPrefixCheckpoint,
    target_model: Any,
    *,
    expected_blank_state_sha256: str,
) -> Mapping[str, Any]:
    """Transactionally install a verified prefix into an exact fresh blank.

    The caller creates the blank.  This function accepts no path or loader and
    checks the complete parameter/buffer inventory, roles, persistence,
    aliases, strides, offsets, shapes, dtypes, devices, and layer count before
    writing one byte.
    """

    if not isinstance(store, Lfm2CpcfResumeArtifactStore):
        raise TypeError("store must be Lfm2CpcfResumeArtifactStore.")
    expected_blank = _digest(
        expected_blank_state_sha256,
        "expected_blank_state_sha256",
    )
    verified = _verified_checkpoint(store, checkpoint)
    completed = _deep_thaw(verified.completed_layers)
    topology = _canonical_mapping(
        completed.get("target_topology_snapshot"),
        "checkpoint target topology",
    )
    if getattr(getattr(target_model, "config", None), "model_type", None) != "lfm2":
        raise Lfm2CpcfResumeBridgeError("Target blank must have model_type='lfm2'.")
    layers = getattr(getattr(target_model, "model", None), "layers", None)
    if layers is None or len(layers) != topology.get("layer_count"):
        raise Lfm2CpcfResumeBridgeError(
            "Target blank layer count differs from checkpoint topology."
        )
    values, inventory = _target_inventory(target_model)
    expected_metadata = topology.get("tensor_metadata")
    if (
        set(values) != set(verified.prefix_model)
        or inventory["tensor_metadata"] != expected_metadata
        or inventory["object_alias_groups"] != topology.get("object_alias_groups")
        or inventory["storage_alias_groups"] != topology.get("storage_alias_groups")
        or topology.get("all_named_parameters_and_buffers") is not True
        or topology.get("nonpersistent_buffers_included") is not True
    ):
        raise Lfm2CpcfResumeBridgeError(
            "Target blank inventory, aliases, shape, dtype, stride, or device differs "
            "from checkpoint topology."
        )
    parameters = dict(_named_values_all(target_model, "named_parameters"))
    if any(parameter.grad is not None for parameter in parameters.values()):
        raise Lfm2CpcfResumeBridgeError(
            "Fresh target blank must not contain parameter gradients."
        )
    blank_before = fingerprint_model_state(target_model)
    if blank_before != expected_blank:
        raise Lfm2CpcfResumeBridgeError(
            "Target model does not match the caller-bound fresh blank state."
        )
    prefix = verified.prefix_model
    for name, value in values.items():
        source = prefix[name]
        if source.dtype != value.dtype or tuple(source.shape) != tuple(value.shape):
            raise Lfm2CpcfResumeBridgeError(
                f"Prefix tensor {name!r} differs from target shape or dtype."
            )
    identities_before = {
        name: (
            id(value),
            int(value.untyped_storage().data_ptr()),
            int(value.storage_offset()),
            tuple(int(size) for size in value.stride()),
        )
        for name, value in values.items()
    }
    blank_snapshot = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in values.items()
    }
    torch = _torch()
    try:
        with torch.no_grad():
            copied_objects = set()
            for name, value in values.items():
                if id(value) in copied_objects:
                    continue
                value.copy_(prefix[name].to(device=value.device, dtype=value.dtype))
                copied_objects.add(id(value))
        after_values, after_inventory = _target_inventory(target_model)
        if after_inventory != inventory or set(after_values) != set(prefix):
            raise Lfm2CpcfResumeBridgeError(
                "Prefix installation changed target topology or inventory."
            )
        identities_after = {
            name: (
                id(value),
                int(value.untyped_storage().data_ptr()),
                int(value.storage_offset()),
                tuple(int(size) for size in value.stride()),
            )
            for name, value in after_values.items()
        }
        if identities_after != identities_before:
            raise Lfm2CpcfResumeBridgeError(
                "Prefix installation changed tensor identity or storage aliases."
            )
        for name, value in after_values.items():
            actual = value.detach().to(device="cpu").contiguous()
            if not torch.equal(actual, prefix[name]):
                raise Lfm2CpcfResumeBridgeError(
                    f"Installed prefix tensor {name!r} differs byte-for-byte."
                )
    except BaseException:
        with torch.no_grad():
            restored_objects = set()
            for name, value in values.items():
                if id(value) in restored_objects:
                    continue
                value.copy_(
                    blank_snapshot[name].to(device=value.device, dtype=value.dtype)
                )
                restored_objects.add(id(value))
        if fingerprint_model_state(target_model) != blank_before:
            raise Lfm2CpcfResumeBridgeError(
                "Prefix installation failed and exact blank rollback was not proven."
            )
        raise
    installed_sha = fingerprint_model_state(target_model)
    store.final_reauthenticate_checkpoint(verified)
    body = {
        "schema": LFM2_CPCF_PREFIX_INSTALL_REPORT_SCHEMA,
        "checkpoint_reference": verified.reference.as_dict(),
        "target_topology_sha256": topology["topology_sha256"],
        "blank_state_sha256_before": blank_before,
        "installed_state_sha256_after": installed_sha,
        "tensor_count": len(values),
        "next_layer_index": completed["next_layer_index"],
        "exact_inventory_verified": True,
        "exact_object_aliases_verified": True,
        "exact_storage_aliases_verified": True,
        "exact_shape_dtype_stride_offset_device_verified": True,
        "tensor_identities_preserved": True,
        "exact_prefix_tensor_bytes_installed": True,
        "loaders_used": 0,
        "parent_evidence_tensors_copied": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "promotion_claimed": False,
    }
    return MappingProxyType({**body, "report_sha256": canonical_sha256(body)})


__all__ = [
    "LFM2_CPCF_PREFIX_INSTALL_REPORT_SCHEMA",
    "LFM2_CPCF_RESUME_BOUNDARY_BRIDGE_SCHEMA",
    "Lfm2CpcfFailureResumePublications",
    "Lfm2CpcfFlattenedBoundaryBundle",
    "Lfm2CpcfProgressCheckpointPublications",
    "Lfm2CpcfResumeBridgeError",
    "flatten_lfm2_cpcf_boundary_bundle",
    "install_lfm2_cpcf_prefix_checkpoint",
    "open_lfm2_cpcf_boundary_bundle",
    "publish_lfm2_cpcf_boundary_bundle",
    "publish_lfm2_cpcf_model_fold_failure_context",
    "publish_lfm2_cpcf_model_fold_progress_checkpoint",
    "reconstruct_lfm2_cpcf_boundary_bundle",
    "reconstruct_lfm2_cpcf_phase_state_bank",
]
