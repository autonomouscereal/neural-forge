"""Fit the tied interface and all predictive LFM2 CPCF boundary charts.

The bundle compiler consumes only authenticated, already-cached parent
evidence.  It first derives the blank child's tied vocabulary interface from
the final normalized producer states.  The resulting producer coordinates
then serve as the common future/action view for *independent* charts at every
post-operator and post-layer residual boundary.

Rows tagged ``fit`` are the only rows allowed to influence fitted values.
``selection`` rows choose only the bounded causal-anchor ridge/margin after
each candidate has been fitted.  Validation values never influence fitted
parameters or candidate selection; base-interface diagnostics and boundary
capacity gates may evaluate them, while causal-anchor validation metrics are
computed only after the winning residual is frozen.  Each phase receives
equal total boundary-chart weight so a long prefill cannot silently dominate
short causal-decode phases.  Anchor evidence uses exactly one endpoint per
batch row and labeled causal step.

No model is accepted or executed by this module.  It performs no mutation,
training, pruning, quantization, or optimizer work.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_cpcf_boundary_chart import (
    CpcfBoundaryChart,
    fit_cpcf_boundary_chart,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import Lfm2CpcfTargetPhase
from neural_forge.hf_tied_causal_interface_anchor_fold import (
    TiedCausalAnchorSplit,
    fit_tied_causal_interface_anchor_residual,
)
from neural_forge.hf_tied_causal_interface_fold import (
    TiedCausalInterfaceFold,
    fit_tied_causal_interface,
)


LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA = "neural-forge-lfm2-cpcf-boundary-bundle-v2"
LFM2_CPCF_LAYER_CHARTS_SCHEMA = "neural-forge-lfm2-cpcf-layer-boundary-charts-v1"
_SPLITS = ("selection", "fit", "validation")
_BOUNDARIES = ("residual.post_operator", "residual.post_layer")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECODE_PHASE = re.compile(r"decode\.(\d{3,})\Z")


class Lfm2CpcfBoundaryBundleError(ValueError):
    """The cached phases cannot form a sealed predictive boundary bundle."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF boundary bundles."
        ) from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfBoundaryBundleError(
            "Boundary-bundle metadata must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfBoundaryBundleError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfBoundaryBundleError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfBoundaryBundleError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfBoundaryBundleError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfBoundaryBundleError(f"{name} must be finite and nonnegative.")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result > 1.0:
        raise Lfm2CpcfBoundaryBundleError(f"{name} must be within [0, 1].")
    return result


def _matrix(value: Any, name: str, *, width: int | None = None) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2CpcfBoundaryBundleError(
            f"{name} must be a nonempty dense real floating matrix."
        )
    if width is not None and int(value.shape[1]) != width:
        raise Lfm2CpcfBoundaryBundleError(f"{name} width must equal {width}.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfBoundaryBundleError(f"{name} contains NaN or infinity.")
    return result


def _phase_tensor(
    value: Any,
    name: str,
    *,
    phase: Lfm2CpcfTargetPhase,
    source_width: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 3
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2CpcfBoundaryBundleError(
            f"{name} must be detached dense real rank-three evidence."
        )
    expected = (
        phase.batch_size,
        phase.sampled_suffix_tokens,
        source_width,
    )
    if tuple(int(size) for size in value.shape) != expected:
        raise Lfm2CpcfBoundaryBundleError(
            f"{name} shape differs from its aligned target phase."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfBoundaryBundleError(f"{name} contains NaN or infinity.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _report_digest(value: Mapping[str, Any], name: str) -> str:
    return _digest(value.get("report_sha256"), f"{name}.report_sha256")


def _verified_report(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfBoundaryBundleError(f"{name} must be a report mapping.")
    report = dict(value)
    body = dict(report)
    seal = body.pop("report_sha256", None)
    if _digest(seal, f"{name}.report_sha256") != _sha256(body):
        raise Lfm2CpcfBoundaryBundleError(f"{name} report seal is invalid.")
    return report


@dataclass(frozen=True, slots=True)
class Lfm2CpcfLayerBoundaryCharts:
    """The two independently fitted residual charts for one LFM2 layer."""

    layer_index: int
    operator_kind: str
    post_operator: CpcfBoundaryChart
    post_layer: CpcfBoundaryChart
    layer_charts_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "layer_index must be a nonnegative integer."
            )
        if self.operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfBoundaryBundleError("operator_kind is not canonical LFM2.")
        if not isinstance(self.post_operator, CpcfBoundaryChart) or not isinstance(
            self.post_layer, CpcfBoundaryChart
        ):
            raise TypeError("post_operator and post_layer must be CpcfBoundaryChart.")
        prefix = f"layer.{self.layer_index:03d}."
        if (
            self.post_operator.name != prefix + "residual.post_operator"
            or self.post_layer.name != prefix + "residual.post_layer"
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "Layer chart names do not match their ordered boundary."
            )
        if (
            self.post_operator.source_width != self.post_layer.source_width
            or self.post_operator.target_width != self.post_layer.target_width
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "Layer boundary chart widths are inconsistent."
            )
        body = {
            "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
            "layer_index": self.layer_index,
            "operator_kind": self.operator_kind,
            "source_width": self.post_operator.source_width,
            "target_width": self.post_operator.target_width,
            "post_operator_report_sha256": _report_digest(
                self.post_operator.report,
                "post_operator",
            ),
            "post_layer_report_sha256": _report_digest(
                self.post_layer.report,
                "post_layer",
            ),
            "independent_boundary_coordinates": True,
        }
        if self.layer_charts_sha256 != _sha256(body):
            raise Lfm2CpcfBoundaryBundleError("Layer boundary-chart seal is invalid.")

    @property
    def source_width(self) -> int:
        return self.post_operator.source_width

    @property
    def target_width(self) -> int:
        return self.post_operator.target_width

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
            "layer_index": self.layer_index,
            "operator_kind": self.operator_kind,
            "source_width": self.post_operator.source_width,
            "target_width": self.post_operator.target_width,
            "post_operator_report_sha256": _report_digest(
                self.post_operator.report,
                "post_operator",
            ),
            "post_layer_report_sha256": _report_digest(
                self.post_layer.report,
                "post_layer",
            ),
            "independent_boundary_coordinates": True,
            "layer_charts_sha256": self.layer_charts_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfBoundaryBundle:
    """Sealed tied-interface plus ordered layer-boundary coordinates."""

    interface: TiedCausalInterfaceFold
    layers: tuple[Lfm2CpcfLayerBoundaryCharts, ...]
    split_phase_sha256: Mapping[str, str]
    split_row_sha256: Mapping[str, str]
    report: Mapping[str, Any]
    bundle_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.interface, TiedCausalInterfaceFold):
            raise TypeError("interface must be TiedCausalInterfaceFold.")
        layers = tuple(self.layers)
        if not layers or not all(
            isinstance(value, Lfm2CpcfLayerBoundaryCharts) for value in layers
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "layers must contain ordered layer boundary charts."
            )
        if tuple(layer.layer_index for layer in layers) != tuple(range(len(layers))):
            raise Lfm2CpcfBoundaryBundleError(
                "Layer boundary charts are incomplete or unordered."
            )
        if any(
            layer.target_width != self.interface.target_width
            or layer.source_width != self.interface.source_width
            for layer in layers
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "Interface and boundary-chart widths differ."
            )
        phase_hashes = {
            split: _digest(
                self.split_phase_sha256.get(split),
                f"split_phase_sha256[{split!r}]",
            )
            for split in _SPLITS
        }
        row_hashes = {
            split: _digest(
                self.split_row_sha256.get(split),
                f"split_row_sha256[{split!r}]",
            )
            for split in _SPLITS
        }
        if set(self.split_phase_sha256) != set(_SPLITS) or set(
            self.split_row_sha256
        ) != set(_SPLITS):
            raise Lfm2CpcfBoundaryBundleError(
                "Split hash maps must cover selection, fit, and validation."
            )
        bundle_sha = _digest(self.bundle_sha256, "bundle_sha256")
        report = dict(self.report)
        if report.pop("bundle_sha256", None) != bundle_sha:
            raise Lfm2CpcfBoundaryBundleError("Boundary-bundle report seal is missing.")
        if _sha256(report) != bundle_sha:
            raise Lfm2CpcfBoundaryBundleError("Boundary-bundle report seal is invalid.")
        if (
            report.get("split_phase_sha256") != phase_hashes
            or report.get("split_row_sha256") != row_hashes
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "Boundary-bundle split hashes differ from its report."
            )
        active_report = _verified_report(
            self.interface.report,
            "active_interface",
        )
        lineage = report.get("interface_lineage")
        if not isinstance(lineage, Mapping):
            raise Lfm2CpcfBoundaryBundleError(
                "Boundary-bundle interface lineage is missing."
            )
        base_report = _verified_report(
            lineage.get("base_interface_report"),
            "base_interface",
        )
        anchor_report = _verified_report(
            lineage.get("causal_anchor_report"),
            "causal_anchor",
        )
        anchor_base_lineage = anchor_report.get("base_interface_lineage")
        if not isinstance(anchor_base_lineage, Mapping):
            raise Lfm2CpcfBoundaryBundleError(
                "Causal-anchor base-interface ancestry is missing."
            )
        active_sha = _report_digest(active_report, "active_interface")
        base_sha = _report_digest(base_report, "base_interface")
        anchor_sha = _report_digest(anchor_report, "causal_anchor")
        if (
            report.get("interface_report_sha256") != active_sha
            or report.get("base_interface_report_sha256") != base_sha
            or report.get("causal_anchor_report_sha256") != anchor_sha
            or lineage.get("active_interface_report_sha256") != active_sha
            or lineage.get("base_interface_report_sha256") != base_sha
            or lineage.get("causal_anchor_report_sha256") != anchor_sha
            or active_report != anchor_report
            or anchor_report.get("base_interface_report_sha256") != base_sha
            or anchor_base_lineage.get("report") != base_report
            or anchor_base_lineage.get("report_sha256") != base_sha
            or anchor_base_lineage.get("target_weight_sha256")
            != base_report.get("target_weight_sha256")
            or anchor_base_lineage.get("producer_encoder_sha256")
            != base_report.get("producer_encoder_sha256")
            or anchor_base_lineage.get("source_table_sha256")
            != base_report.get("source_table_sha256")
            or anchor_base_lineage.get("tensor_identity_verified") is not True
            or _tensor_sha256(self.interface.producer_encoder)
            != base_report.get("producer_encoder_sha256")
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "Boundary-bundle base, anchor, or active interface lineage "
                "seal is inconsistent."
            )
        object.__setattr__(self, "layers", layers)
        object.__setattr__(
            self,
            "split_phase_sha256",
            MappingProxyType(phase_hashes),
        )
        object.__setattr__(
            self,
            "split_row_sha256",
            MappingProxyType(row_hashes),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "bundle_sha256": bundle_sha}),
        )

    @property
    def source_width(self) -> int:
        return self.interface.source_width

    @property
    def target_width(self) -> int:
        return self.interface.target_width

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


@dataclass(frozen=True, slots=True)
class _PhaseInventory:
    phases: tuple[Lfm2CpcfTargetPhase, ...]
    layer_kinds: tuple[str, ...]
    source_width: int
    tensors: Mapping[tuple[tuple[int, str], str], Any]
    split_phases: Mapping[str, tuple[Lfm2CpcfTargetPhase, ...]]
    split_phase_sha256: Mapping[str, str]
    split_row_sha256: Mapping[str, str]
    split_rows: Mapping[str, int]
    source_observation_sha256: str


def _row_keys(phase: Lfm2CpcfTargetPhase) -> list[dict[str, Any]]:
    first_suffix_position = phase.full_prefix_tokens - phase.sampled_suffix_tokens
    return [
        {
            "batch_index": phase.batch_index,
            "record_id": phase.record_id,
            "split": phase.split,
            "phase": phase.phase,
            "phase_sha256": phase.phase_sha256,
            "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
            "batch_row_index": batch_row,
            "sampled_suffix_index": sample_index,
            "full_prefix_position": first_suffix_position + sample_index,
        }
        for batch_row in range(phase.batch_size)
        for sample_index in range(phase.sampled_suffix_tokens)
    ]


def _inventory(
    phases: Sequence[Lfm2CpcfTargetPhase],
    *,
    source_width: int,
) -> _PhaseInventory:
    ordered = tuple(phases)
    if not ordered or not all(
        isinstance(value, Lfm2CpcfTargetPhase) for value in ordered
    ):
        raise Lfm2CpcfBoundaryBundleError(
            "phases must contain ordered Lfm2CpcfTargetPhase objects."
        )
    identities = tuple(phase.identity for phase in ordered)
    if len(set(identities)) != len(identities):
        raise Lfm2CpcfBoundaryBundleError("Target phase identities are duplicated.")
    unknown = sorted({phase.split for phase in ordered} - set(_SPLITS))
    if unknown:
        raise Lfm2CpcfBoundaryBundleError(
            f"Unsupported target phase splits: {unknown!r}."
        )
    grouped = {
        split: tuple(phase for phase in ordered if phase.split == split)
        for split in _SPLITS
    }
    if not grouped["fit"] or not grouped["validation"]:
        raise Lfm2CpcfBoundaryBundleError(
            "At least one fit and one validation phase are required."
        )

    layer_count = len(ordered[0].parent_evidence.layers)
    layer_kinds = tuple(
        layer.operator_kind for layer in ordered[0].parent_evidence.layers
    )
    if layer_count == 0:
        raise Lfm2CpcfBoundaryBundleError("Installable evidence contains no layers.")

    tensors: dict[tuple[tuple[int, str], str], Any] = {}
    observation_inventory = []
    for phase in ordered:
        evidence = phase.parent_evidence
        if (
            len(evidence.layers) != layer_count
            or tuple(layer.operator_kind for layer in evidence.layers) != layer_kinds
            or any(
                layer.layer_index != index or layer.source_width != source_width
                for index, layer in enumerate(evidence.layers)
            )
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "Parent layer topology or source width changes across phases."
            )
        final_name = "output.final_norm_output"
        try:
            final_value = evidence.final_norm_tensors[final_name]
        except KeyError as exc:
            raise Lfm2CpcfBoundaryBundleError(
                f"Phase {phase.identity!r} is missing {final_name!r}."
            ) from exc
        final = _phase_tensor(
            final_value,
            f"{phase.identity}.{final_name}",
            phase=phase,
            source_width=source_width,
        )
        tensors[(phase.identity, final_name)] = final
        observation_inventory.append(
            {
                "phase_sha256": phase.phase_sha256,
                "field": final_name,
                "tensor_sha256": _tensor_sha256(final),
            }
        )
        for layer_index, layer in enumerate(evidence.layers):
            for boundary in _BOUNDARIES:
                try:
                    value = layer.tensors[boundary]
                except KeyError as exc:
                    raise Lfm2CpcfBoundaryBundleError(
                        f"Phase {phase.identity!r} layer {layer_index} "
                        f"is missing {boundary!r}."
                    ) from exc
                name = f"layer.{layer_index:03d}.{boundary}"
                tensor = _phase_tensor(
                    value,
                    f"{phase.identity}.{name}",
                    phase=phase,
                    source_width=source_width,
                )
                tensors[(phase.identity, name)] = tensor
                observation_inventory.append(
                    {
                        "phase_sha256": phase.phase_sha256,
                        "field": name,
                        "tensor_sha256": _tensor_sha256(tensor),
                    }
                )

    split_phase_hashes: dict[str, str] = {}
    split_row_hashes: dict[str, str] = {}
    split_rows: dict[str, int] = {}
    for split, values in grouped.items():
        split_phase_hashes[split] = _sha256(
            [
                {
                    "ordinal": ordinal,
                    "batch_index": phase.batch_index,
                    "record_id": phase.record_id,
                    "phase": phase.phase,
                    "phase_sha256": phase.phase_sha256,
                    "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
                }
                for ordinal, phase in enumerate(values)
            ]
        )
        keys = [row for phase in values for row in _row_keys(phase)]
        split_row_hashes[split] = _sha256(keys)
        split_rows[split] = len(keys)
    if (
        split_phase_hashes["fit"] == split_phase_hashes["validation"]
        or split_row_hashes["fit"] == split_row_hashes["validation"]
    ):
        raise Lfm2CpcfBoundaryBundleError(
            "Fit and validation phase/row identities are not disjoint."
        )

    expected_observations = len(ordered) * (1 + 2 * layer_count)
    if len(observation_inventory) != expected_observations:
        raise Lfm2CpcfBoundaryBundleError("Not every source boundary was inventoried.")
    return _PhaseInventory(
        phases=ordered,
        layer_kinds=layer_kinds,
        source_width=source_width,
        tensors=MappingProxyType(tensors),
        split_phases=MappingProxyType(grouped),
        split_phase_sha256=MappingProxyType(split_phase_hashes),
        split_row_sha256=MappingProxyType(split_row_hashes),
        split_rows=MappingProxyType(split_rows),
        source_observation_sha256=_sha256(observation_inventory),
    )


def _rows(
    inventory: _PhaseInventory,
    split: str,
    field: str,
) -> Any:
    torch = _torch()
    values = [
        inventory.tensors[(phase.identity, field)].reshape(
            -1,
            inventory.source_width,
        )
        for phase in inventory.split_phases[split]
    ]
    return torch.cat(values, dim=0).contiguous()


def _causal_anchor_splits(
    inventory: _PhaseInventory,
) -> Mapping[str, TiedCausalAnchorSplit]:
    """Extract one next-token anchor per record and causal endpoint."""

    torch = _torch()
    grouped: dict[
        tuple[int, str],
        list[Lfm2CpcfTargetPhase],
    ] = {}
    record_order: list[tuple[int, str]] = []
    closed: set[tuple[int, str]] = set()
    previous: tuple[int, str] | None = None
    for phase in inventory.phases:
        key = (phase.batch_index, phase.record_id)
        if key != previous:
            if previous is not None:
                closed.add(previous)
            if key in closed:
                raise Lfm2CpcfBoundaryBundleError(
                    "Target phases for one record must be contiguous."
                )
            record_order.append(key)
            previous = key
        grouped.setdefault(key, []).append(phase)
    batch_indices = [key[0] for key in record_order]
    if batch_indices != sorted(batch_indices) or len(set(batch_indices)) != len(
        batch_indices
    ):
        raise Lfm2CpcfBoundaryBundleError(
            "Target records must be unique and ordered by cache batch."
        )

    split_rows: dict[str, list[tuple[str, int, Any, int]]] = {
        split: [] for split in _SPLITS
    }
    all_row_ids: set[str] = set()
    final_field = "output.final_norm_output"
    for key in record_order:
        phases = grouped[key]
        first = phases[0]
        if first.phase not in {"prefill", "cached_prefill"} or any(
            phase.batch_index != first.batch_index
            or phase.record_id != first.record_id
            or phase.split != first.split
            or phase.batch_size != first.batch_size
            or phase.donor_input_sha256 != first.donor_input_sha256
            or phase.target_alignment_sha256 != first.target_alignment_sha256
            for phase in phases
        ):
            raise Lfm2CpcfBoundaryBundleError(
                "Causal-anchor records require one canonical prefill and "
                "stable record, batch, and alignment provenance."
            )
        expected_names = (first.phase,) + tuple(
            f"decode.{index:03d}" for index in range(len(phases) - 1)
        )
        observed_names = tuple(phase.phase for phase in phases)
        if observed_names != expected_names:
            raise Lfm2CpcfBoundaryBundleError(
                "Target phases are not in canonical prefill/decode order."
            )

        for causal_step, (phase, next_phase) in enumerate(
            zip(phases[:-1], phases[1:], strict=True)
        ):
            if (
                next_phase.full_prefix_tokens != phase.full_prefix_tokens + 1
                or not torch.equal(
                    next_phase.input_ids[:, :-1],
                    phase.input_ids,
                )
                or not torch.equal(
                    next_phase.attention_mask[:, :-1],
                    phase.attention_mask,
                )
                or not torch.equal(
                    next_phase.position_ids[:, :-1],
                    phase.position_ids,
                )
                or not bool((next_phase.attention_mask[:, -1] == 1).all().item())
                or not torch.equal(
                    next_phase.position_ids[:, -1],
                    phase.position_ids[:, -1] + 1,
                )
            ):
                raise Lfm2CpcfBoundaryBundleError(
                    "A target decode phase does not append exactly one "
                    "teacher-forced token to its causal predecessor."
                )
            current_match = _DECODE_PHASE.fullmatch(phase.phase)
            if (
                causal_step == 0
                and phase.phase not in {"prefill", "cached_prefill"}
                or causal_step > 0
                and (
                    current_match is None
                    or int(current_match.group(1)) != causal_step - 1
                )
            ):
                raise Lfm2CpcfBoundaryBundleError(
                    "Causal phase-to-target labels are not canonical."
                )
            endpoints = inventory.tensors[(phase.identity, final_field)][:, -1, :]
            for batch_row in range(phase.batch_size):
                prefix_position = int(phase.position_ids[batch_row, -1].item())
                row_id = (
                    f"{phase.split}/batch.{phase.batch_index:06d}/"
                    f"record.{phase.record_id}/batch_row.{batch_row:04d}/"
                    f"causal.{causal_step:04d}/"
                    f"prefix.{prefix_position:08d}"
                )
                if row_id in all_row_ids:
                    raise Lfm2CpcfBoundaryBundleError(
                        "A record/batch-row/causal-position anchor row ID is "
                        "duplicated."
                    )
                all_row_ids.add(row_id)
                split_rows[phase.split].append(
                    (
                        row_id,
                        prefix_position,
                        endpoints[batch_row].contiguous(),
                        int(next_phase.input_ids[batch_row, -1].item()),
                    )
                )

    result: dict[str, TiedCausalAnchorSplit] = {}
    for split in _SPLITS:
        rows = split_rows[split]
        if not rows:
            raise Lfm2CpcfBoundaryBundleError(
                f"{split} has no labeled causal endpoint; every anchor split "
                "requires at least one non-final phase."
            )
        result[split] = TiedCausalAnchorSplit(
            name=split,
            row_ids=tuple(row[0] for row in rows),
            prefix_positions=torch.tensor(
                [row[1] for row in rows],
                dtype=torch.long,
            ),
            source_producer_rows=torch.stack(
                [row[2] for row in rows],
                dim=0,
            ).contiguous(),
            target_next_token_ids=torch.tensor(
                [row[3] for row in rows],
                dtype=torch.long,
            ),
        )
    if sum(split.row_count for split in result.values()) != len(all_row_ids):
        raise Lfm2CpcfBoundaryBundleError(
            "Causal-anchor extraction did not preserve one unique row ID per "
            "record, batch row, and causal step."
        )
    return MappingProxyType(result)


def build_lfm2_cpcf_causal_anchor_splits(
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> Mapping[str, TiedCausalAnchorSplit]:
    """Build sealed leakage-free causal-anchor splits from cached phases."""

    ordered = tuple(phases)
    if not ordered or not isinstance(
        ordered[0],
        Lfm2CpcfTargetPhase,
    ):
        raise Lfm2CpcfBoundaryBundleError(
            "phases must contain ordered Lfm2CpcfTargetPhase objects."
        )
    layers = ordered[0].parent_evidence.layers
    if not layers:
        raise Lfm2CpcfBoundaryBundleError("Installable evidence contains no layers.")
    inventory = _inventory(
        ordered,
        source_width=layers[0].source_width,
    )
    return _causal_anchor_splits(inventory)


def _balanced_phase_weights(
    inventory: _PhaseInventory,
    split: str,
) -> tuple[Any, list[dict[str, Any]]]:
    torch = _torch()
    phases = inventory.split_phases[split]
    phase_mass = 1.0 / len(phases)
    pieces = []
    summary = []
    for phase in phases:
        rows = phase.batch_size * phase.sampled_suffix_tokens
        row_weight = phase_mass / rows
        piece = torch.full(
            (rows,),
            row_weight,
            dtype=torch.float64,
        )
        pieces.append(piece)
        summary.append(
            {
                "batch_index": phase.batch_index,
                "record_id": phase.record_id,
                "phase": phase.phase,
                "rows": rows,
                "row_weight": row_weight,
                "total_phase_weight": float(piece.sum().item()),
            }
        )
    result = torch.cat(pieces).contiguous()
    if (
        int(result.numel()) != inventory.split_rows[split]
        or abs(float(result.sum().item()) - 1.0) > 1e-12
    ):
        raise Lfm2CpcfBoundaryBundleError(
            "Balanced phase weights failed their unit-mass invariant."
        )
    return result, summary


def _layer_charts(
    *,
    layer_index: int,
    operator_kind: str,
    post_operator: CpcfBoundaryChart,
    post_layer: CpcfBoundaryChart,
) -> Lfm2CpcfLayerBoundaryCharts:
    body = {
        "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
        "layer_index": layer_index,
        "operator_kind": operator_kind,
        "source_width": post_operator.source_width,
        "target_width": post_operator.target_width,
        "post_operator_report_sha256": _report_digest(
            post_operator.report,
            "post_operator",
        ),
        "post_layer_report_sha256": _report_digest(
            post_layer.report,
            "post_layer",
        ),
        "independent_boundary_coordinates": True,
    }
    return Lfm2CpcfLayerBoundaryCharts(
        layer_index=layer_index,
        operator_kind=operator_kind,
        post_operator=post_operator,
        post_layer=post_layer,
        layer_charts_sha256=_sha256(body),
    )


def fit_lfm2_cpcf_boundary_bundle(
    phases: Sequence[Lfm2CpcfTargetPhase],
    source_tied_table: Any,
    *,
    target_hidden_width: int,
    preserve_vocabulary_mean: bool = True,
    target_norm_epsilon: float = 1e-5,
    state_reconstruction_weight: float = 0.05,
    ridge: float = 1e-6,
    minimum_predictive_energy_fraction: float = 0.90,
    minimum_validation_block_retention: float = 0.0,
    enforce_validation_block_floor: bool = True,
    metric_chunk_rows: int = 512,
) -> Lfm2CpcfBoundaryBundle:
    """Compile a sealed interface-and-chart bundle from cached evidence.

    Parent top-1 agreement is recorded as an interface diagnostic.  It is not
    a capacity gate or the principal objective.  Every chart is instead gated
    by its disjoint held-out future-coordinate block retention.
    """

    table = _matrix(source_tied_table, "source_tied_table")
    source_width = int(table.shape[1])
    target_width = _positive_int(
        target_hidden_width,
        "target_hidden_width",
    )
    if target_width >= source_width:
        raise Lfm2CpcfBoundaryBundleError(
            "target_hidden_width must contract the parent hidden width."
        )
    if not isinstance(preserve_vocabulary_mean, bool):
        raise Lfm2CpcfBoundaryBundleError("preserve_vocabulary_mean must be boolean.")
    if not math.isfinite(target_norm_epsilon) or target_norm_epsilon <= 0.0:
        raise Lfm2CpcfBoundaryBundleError(
            "target_norm_epsilon must be positive and finite."
        )
    state_weight = _nonnegative_float(
        state_reconstruction_weight,
        "state_reconstruction_weight",
    )
    ridge_value = _nonnegative_float(ridge, "ridge")
    predictive_floor = _fraction(
        minimum_predictive_energy_fraction,
        "minimum_predictive_energy_fraction",
    )
    validation_floor = _fraction(
        minimum_validation_block_retention,
        "minimum_validation_block_retention",
    )
    if not isinstance(enforce_validation_block_floor, bool):
        raise Lfm2CpcfBoundaryBundleError(
            "enforce_validation_block_floor must be boolean."
        )
    chunk_rows = _positive_int(metric_chunk_rows, "metric_chunk_rows")

    inventory = _inventory(phases, source_width=source_width)
    fit_weights, fit_weight_summary = _balanced_phase_weights(
        inventory,
        "fit",
    )
    validation_weights, validation_weight_summary = _balanced_phase_weights(
        inventory, "validation"
    )
    final_field = "output.final_norm_output"
    fit_final = _rows(inventory, "fit", final_field)
    validation_final = _rows(inventory, "validation", final_field)

    base_interface = fit_tied_causal_interface(
        table,
        fit_final,
        target_width=target_width,
        validation_producer_rows=validation_final,
        fit_weights=fit_weights,
        validation_weights=validation_weights,
        preserve_vocabulary_mean=preserve_vocabulary_mean,
        target_norm_epsilon=target_norm_epsilon,
        minimum_validation_top1_agreement=0.0,
        metric_chunk_rows=chunk_rows,
    )
    anchor_splits = _causal_anchor_splits(inventory)
    anchor_fold = fit_tied_causal_interface_anchor_residual(
        base_interface,
        table,
        selection_split=anchor_splits["selection"],
        fit_split=anchor_splits["fit"],
        validation_split=anchor_splits["validation"],
        target_norm_epsilon=target_norm_epsilon,
        metric_chunk_rows=chunk_rows,
    )
    interface = anchor_fold.interface
    anchor_residual_selected = bool(
        anchor_fold.report["active_anchor_residual_selected"]
    )
    fit_future = interface.encode_producer_rows(fit_final)
    validation_future = interface.encode_producer_rows(validation_final)

    layer_results = []
    for layer_index, operator_kind in enumerate(inventory.layer_kinds):
        charts: dict[str, CpcfBoundaryChart] = {}
        for boundary in _BOUNDARIES:
            field = f"layer.{layer_index:03d}.{boundary}"
            charts[boundary] = fit_cpcf_boundary_chart(
                field,
                _rows(inventory, "fit", field),
                fit_future,
                target_width=target_width,
                validation_parent_states=_rows(
                    inventory,
                    "validation",
                    field,
                ),
                validation_future_coordinates=validation_future,
                fit_weights=fit_weights,
                validation_weights=validation_weights,
                state_reconstruction_weight=state_weight,
                ridge=ridge_value,
                minimum_predictive_energy_fraction=predictive_floor,
                minimum_validation_block_retention=validation_floor,
                enforce_validation_block_floor=(enforce_validation_block_floor),
            )
        layer_results.append(
            _layer_charts(
                layer_index=layer_index,
                operator_kind=operator_kind,
                post_operator=charts["residual.post_operator"],
                post_layer=charts["residual.post_layer"],
            )
        )
    layers = tuple(layer_results)

    phase_counts = {split: len(inventory.split_phases[split]) for split in _SPLITS}
    row_counts = {split: inventory.split_rows[split] for split in _SPLITS}
    split_phase_hashes = dict(inventory.split_phase_sha256)
    split_row_hashes = dict(inventory.split_row_sha256)
    expected_observations = len(inventory.phases) * (1 + 2 * len(inventory.layer_kinds))
    body = {
        "schema": LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
        "method": (
            "tied_causal_interface_then_behavior_preserving_anchor_selection_then_"
            "independent_predictive_boundary_charts"
        ),
        "source_width": source_width,
        "target_width": target_width,
        "vocabulary_size": int(table.shape[0]),
        "layer_count": len(layers),
        "layer_operator_kinds": list(inventory.layer_kinds),
        "interface_report_sha256": _report_digest(
            interface.report,
            "interface",
        ),
        "base_interface_report_sha256": _report_digest(
            base_interface.report,
            "base_interface",
        ),
        "causal_anchor_report_sha256": _report_digest(
            anchor_fold.report,
            "causal_anchor",
        ),
        "interface_lineage": {
            "base_interface_report": dict(base_interface.report),
            "causal_anchor_report": dict(anchor_fold.report),
            "base_interface_report_sha256": _report_digest(
                base_interface.report,
                "base_interface",
            ),
            "causal_anchor_report_sha256": _report_digest(
                anchor_fold.report,
                "causal_anchor",
            ),
            "active_interface_report_sha256": _report_digest(
                interface.report,
                "interface",
            ),
            "active_interface_is_anchor_repaired": anchor_residual_selected,
            "active_interface_is_base_control": not anchor_residual_selected,
            "producer_encoder_fixed_by_anchor_selection": True,
            "producer_encoder_fixed_by_anchor_repair": (anchor_residual_selected),
            "additional_runtime_parameter_count": 0,
        },
        "causal_anchor_extraction": {
            "method": ("one_last_sampled_final_norm_endpoint_per_record_causal_step"),
            "label_source": "next_phase_newly_appended_target_token",
            "phase_label_contract": {
                "prefill": "forced_decode[0]",
                "cached_prefill": "forced_decode[0]",
                "decode.k": "forced_decode[k+1]",
            },
            "final_unlabeled_phase_excluded": True,
            "canonical_prefill_or_cached_prefill_decode_order_required": True,
            "one_unique_row_id_per_record_batch_row_causal_step": True,
            "row_id_binds_batch_index_batch_row_and_prefix_position": True,
            "no_sampled_prefix_row_duplication": True,
            "split_inventory": {
                split: anchor_splits[split].as_dict() for split in _SPLITS
            },
        },
        "layer_charts": [layer.as_dict() for layer in layers],
        "split_phase_counts": phase_counts,
        "split_row_counts": row_counts,
        "split_phase_sha256": split_phase_hashes,
        "split_row_sha256": split_row_hashes,
        "fit_validation_disjoint_phase_identities": True,
        "fit_validation_disjoint_row_identities": True,
        "selection": {
            "phase_count": phase_counts["selection"],
            "row_count": row_counts["selection"],
            "phase_sha256": split_phase_hashes["selection"],
            "row_sha256": split_row_hashes["selection"],
            "used_to_fit_base_interface": False,
            "used_to_fit_causal_anchor_values": False,
            "used_to_select_causal_anchor_ridge_margin": True,
            "used_to_fit_boundary_charts": False,
            "used_for_validation_gates": False,
            "authenticated_and_receipted": True,
        },
        "phase_weighting": {
            "method": "equal_phase_mass_then_uniform_rows_within_phase",
            "balanced_phase_weights": True,
            "fit_total_weight": float(fit_weights.sum().item()),
            "validation_total_weight": float(validation_weights.sum().item()),
            "fit_weights_sha256": _tensor_sha256(fit_weights),
            "validation_weights_sha256": _tensor_sha256(validation_weights),
            "fit_phases": fit_weight_summary,
            "validation_phases": validation_weight_summary,
        },
        "interface_top1": {
            "principal_objective": False,
            "capacity_gate_used": False,
            "fit_rmsnorm_parent_top1_agreement": base_interface.report["fit"][
                "rmsnorm_parent_top1_agreement"
            ],
            "validation_rmsnorm_parent_top1_agreement": base_interface.report[
                "validation"
            ]["rmsnorm_parent_top1_agreement"],
            "diagnostic_only": True,
        },
        "predictive_heldout_gates": {
            "principal_boundary_objective": True,
            "minimum_predictive_energy_fraction": predictive_floor,
            "minimum_validation_block_retention": validation_floor,
            "validation_block_floor_enforced": (enforce_validation_block_floor),
            "fit_rows_used_for_parameters": True,
            "validation_rows_used_for_parameters": False,
        },
        "source_coverage": {
            "phase_count": len(inventory.phases),
            "layer_count": len(inventory.layer_kinds),
            "boundary_fields_per_layer": list(_BOUNDARIES),
            "final_output_field": final_field,
            "expected_tensor_observations": expected_observations,
            "observed_tensor_observations": expected_observations,
            "source_observation_sha256": (inventory.source_observation_sha256),
            "all_phase_final_outputs_covered": True,
            "all_layer_post_operator_boundaries_covered": True,
            "all_layer_post_layer_boundaries_covered": True,
            "full_source_coverage": True,
        },
        "coordinate_contract": {
            "future_coordinates": (
                "behavior_selected_tied_interface_encoded_parent_final_norm_output"
            ),
            "active_interface_is_anchor_repaired": anchor_residual_selected,
            "active_interface_is_base_control": not anchor_residual_selected,
            "one_global_residual_basis_used": False,
            "independent_post_operator_and_post_layer_charts": True,
            "residual_write_formula": (
                "encoded_parent_endpoint_minus_actual_child_pre_state"
            ),
        },
        "operation_audit": {
            "parent_forward_count": 0,
            "target_forward_count": 0,
            "parent_generation_count": 0,
            "parent_cache_regeneration_count": 0,
            "model_mutation_count": 0,
            "training_step_count": 0,
            "backward_call_count": 0,
            "optimizer_step_count": 0,
            "quantization_used": False,
            "pruning_used": False,
            "blank_target_values_read": False,
        },
    }
    bundle_sha = _sha256(body)
    report = {**body, "bundle_sha256": bundle_sha}
    return Lfm2CpcfBoundaryBundle(
        interface=interface,
        layers=layers,
        split_phase_sha256=split_phase_hashes,
        split_row_sha256=split_row_hashes,
        report=report,
        bundle_sha256=bundle_sha,
    )


__all__ = [
    "LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA",
    "Lfm2CpcfBoundaryBundle",
    "Lfm2CpcfBoundaryBundleError",
    "Lfm2CpcfLayerBoundaryCharts",
    "build_lfm2_cpcf_causal_anchor_splits",
    "fit_lfm2_cpcf_boundary_bundle",
]
