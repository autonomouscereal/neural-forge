"""Inference-only real Q/K anchor edits for residual-attention probes.

The attention codebook normally stores one coupled Q/K/V/O prototype.  When a
redirect is causally bad, it is useful to ask a narrower question before
committing another compaction: would a *different real survivor's* addressing
pair preserve the cached attention transition better while V/O stay fixed?

This module implements that diagnostic edit.  Q and K are always copied
together from one active physical prototype; Q from one survivor is never
mixed with K from another and neither tensor is averaged before softmax.  The
edit is bounded, gradient-free, content-addressed, and reversible.  It is not
a checkpoint transformation and deliberately has no save/commit operation.

Qwen3 applies q_norm/k_norm after the projections.  Those norms are logical
layer modules rather than group-prototype storage, so a cross-layer codebook
assignment necessarily uses the destination logical layer's unchanged norms.
``audit_qk_anchor_assignments`` records and verifies that exact deployed
semantics for every executable assignment sharing the edited survivor,
including role scales and canonical-to-physical group permutations.
"""

from __future__ import annotations

import hashlib
import math
import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


QK_ANCHOR_SCHEMA = "neural-forge-real-qk-anchor-edit-v1"
QK_ASSIGNMENT_AUDIT_SCHEMA = "neural-forge-qk-anchor-assignment-audit-v2"


class ResidualAttentionQKAnchorError(RuntimeError):
    """A Q/K anchor edit or its exact audit failed closed."""


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionQKAnchorError(f"{name} must be a non-negative integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionQKAnchorError(
            f"{name} must be a non-negative integer."
        ) from error
    if result < 0:
        raise ResidualAttentionQKAnchorError(f"{name} must be a non-negative integer.")
    return result


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise ResidualAttentionQKAnchorError("Expected a tensor for hashing.")
    raw = (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def _optional_tensor_sha256(value: Tensor | None) -> str | None:
    return None if value is None else _tensor_sha256(value)


def _tensor_bytes(value: Tensor | None) -> int:
    return 0 if value is None else int(value.numel() * value.element_size())


def _registry_sha256(registry: Any) -> str:
    digest = hashlib.sha256()
    states = registry.state_dict()
    if not states:
        raise ResidualAttentionQKAnchorError(
            "Attention registry has no content-addressable state."
        )
    for name, value in sorted(states.items()):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_tensor_sha256(value)))
    return digest.hexdigest()


def _state_sha256s(registry: Any) -> dict[str, str]:
    return {
        name: _tensor_sha256(value)
        for name, value in sorted(registry.state_dict().items())
    }


def _flat_prototype(registry: Any, name: str) -> Tensor | None:
    value = getattr(registry, name, None)
    if value is None:
        return None
    if not isinstance(value, Tensor) or value.ndim not in {2, 3, 4}:
        raise ResidualAttentionQKAnchorError(
            f"{name} has an unsupported prototype layout."
        )
    # Ragged weights are [slot, rows, input], rectangular weights are
    # [canonical_group, slot, rows, input].  Biases have one fewer axis.
    if name.endswith("_weight"):
        if value.ndim == 3:
            return value
        if value.ndim == 4:
            return value.reshape(-1, *value.shape[2:])
    else:
        if value.ndim == 2:
            return value
        if value.ndim == 3:
            return value.reshape(-1, value.shape[-1])
    raise ResidualAttentionQKAnchorError(
        f"{name} rank does not match its weight/bias role."
    )


def _global_assignments(registry: Any) -> Tensor:
    assignments = registry.assignments.detach().to(device="cpu", dtype=torch.long)
    q_weight = registry.prototype_q_weight
    if q_weight.ndim == 3:
        return assignments
    if q_weight.ndim != 4:
        raise ResidualAttentionQKAnchorError(
            "Q prototype storage must be ragged or rectangular."
        )
    groups, slots = map(int, q_weight.shape[:2])
    if int(assignments.shape[1]) != groups:
        raise ResidualAttentionQKAnchorError(
            "Rectangular assignments disagree with Q prototype groups."
        )
    offsets = torch.arange(groups, dtype=torch.long) * slots
    return assignments + offsets.unsqueeze(0)


def _flat_sources(registry: Any) -> Tensor:
    sources = registry.prototype_source_indices.detach().to(
        device="cpu", dtype=torch.long
    )
    return sources.reshape(-1)


def _slot_for_source(registry: Any, source_flat_index: int) -> int:
    source = _nonnegative_int(source_flat_index, "source_flat_index")
    matches = torch.nonzero(_flat_sources(registry) == source, as_tuple=False).reshape(
        -1
    )
    if int(matches.numel()) != 1:
        raise ResidualAttentionQKAnchorError(
            f"Source prototype {source} is not one unique active real survivor."
        )
    return int(matches.item())


def _slots_sha256(value: Tensor | None, excluded_slot: int) -> str | None:
    if value is None:
        return None
    digest = hashlib.sha256()
    for slot in range(int(value.shape[0])):
        if slot == excluded_slot:
            continue
        digest.update(bytes.fromhex(_tensor_sha256(value[slot])))
    return digest.hexdigest()


def _qk_pair_sha256(
    q_weight: Tensor,
    k_weight: Tensor,
    q_bias: Tensor | None,
    k_bias: Tensor | None,
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("q_bias", q_bias),
        ("k_bias", k_bias),
    ):
        digest.update(name.encode("ascii"))
        if value is None:
            digest.update(b"none")
        else:
            digest.update(bytes.fromhex(_tensor_sha256(value)))
    return digest.hexdigest()


@dataclass(frozen=True, order=True, slots=True)
class RealQKAnchor:
    """One active prototype provenance coordinate used as a Q+K pair."""

    layer_index: int
    physical_group: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_index",
            _nonnegative_int(self.layer_index, "layer_index"),
        )
        object.__setattr__(
            self,
            "physical_group",
            _nonnegative_int(self.physical_group, "physical_group"),
        )

    def source_flat_index(self, key_value_heads: int) -> int:
        groups = _nonnegative_int(key_value_heads, "key_value_heads")
        if groups <= 0 or self.physical_group >= groups:
            raise ResidualAttentionQKAnchorError(
                "Anchor physical_group is outside the registry's KV groups."
            )
        return self.layer_index * groups + self.physical_group

    @property
    def anchor_id(self) -> str:
        return f"L{self.layer_index}G{self.physical_group}"

    def as_dict(self, *, key_value_heads: int | None = None) -> dict[str, Any]:
        result = {
            "layer_index": self.layer_index,
            "physical_group": self.physical_group,
            "anchor_id": self.anchor_id,
        }
        if key_value_heads is not None:
            result["source_flat_index"] = self.source_flat_index(key_value_heads)
        return result


@dataclass(frozen=True, slots=True)
class QKAnchorEditReport:
    anchor: RealQKAnchor
    destination_source_flat_index: int
    anchor_slot: int
    destination_slot: int
    registry_sha256_before: str
    registry_sha256_installed: str
    destination_pair_sha256_before: str
    anchor_pair_sha256: str
    installed_pair_sha256: str
    q_weight_sha256: str
    k_weight_sha256: str
    q_bias_sha256: str | None
    k_bias_sha256: str | None
    edited_tensor_bytes: int
    mutation_required: bool
    unaffected_qk_slots_byte_exact: bool
    non_qk_registry_tensors_byte_exact: bool

    def as_dict(self, *, key_value_heads: int) -> dict[str, Any]:
        return {
            "schema": QK_ANCHOR_SCHEMA,
            "anchor": self.anchor.as_dict(key_value_heads=key_value_heads),
            "destination_source_flat_index": self.destination_source_flat_index,
            "anchor_slot": self.anchor_slot,
            "destination_slot": self.destination_slot,
            "registry_sha256_before": self.registry_sha256_before,
            "registry_sha256_installed": self.registry_sha256_installed,
            "destination_pair_sha256_before": (self.destination_pair_sha256_before),
            "anchor_pair_sha256": self.anchor_pair_sha256,
            "installed_pair_sha256": self.installed_pair_sha256,
            "q_weight_sha256": self.q_weight_sha256,
            "k_weight_sha256": self.k_weight_sha256,
            "q_bias_sha256": self.q_bias_sha256,
            "k_bias_sha256": self.k_bias_sha256,
            "edited_tensor_bytes": self.edited_tensor_bytes,
            "mutation_required": self.mutation_required,
            "unaffected_qk_slots_byte_exact": (self.unaffected_qk_slots_byte_exact),
            "non_qk_registry_tensors_byte_exact": (
                self.non_qk_registry_tensors_byte_exact
            ),
            "q_and_k_copied_from_same_real_active_slot": True,
            "q_or_k_averaged_before_softmax": False,
            "v_edited": False,
            "o_edited": False,
            "role_scales_edited": False,
            "group_permutations_edited": False,
            "gradient_free": True,
            "training_used": False,
            "checkpoint_save_supported": False,
        }


class InstalledRealQKAnchor:
    """One live reversible Q+K edit.  There is intentionally no commit method."""

    def __init__(
        self,
        registry: Any,
        report: QKAnchorEditReport,
        backups: dict[str, Tensor | None],
    ) -> None:
        self._registry = registry
        self.report = report
        self._backups = backups
        self._applied = True

    @property
    def is_applied(self) -> bool:
        return self._applied

    def assert_live(self) -> None:
        if not self._applied:
            raise ResidualAttentionQKAnchorError(
                "The real Q/K anchor edit is not live."
            )
        if _registry_sha256(self._registry) != self.report.registry_sha256_installed:
            raise ResidualAttentionQKAnchorError(
                "The live registry no longer matches the installed Q/K anchor."
            )

    def rollback(self) -> None:
        if not self._applied:
            return
        destination = self.report.destination_slot
        with torch.inference_mode():
            for name, backup in self._backups.items():
                flat = _flat_prototype(self._registry, name)
                if flat is None:
                    if backup is not None:
                        raise ResidualAttentionQKAnchorError(
                            f"Cannot restore missing {name}."
                        )
                    continue
                if backup is None:
                    raise ResidualAttentionQKAnchorError(
                        f"Cannot restore {name} from an absent backup."
                    )
                flat[destination].copy_(backup)
        restored = _registry_sha256(self._registry)
        if restored != self.report.registry_sha256_before:
            raise ResidualAttentionQKAnchorError(
                "Q/K anchor rollback did not restore the registry byte-exactly."
            )
        self._applied = False


def install_real_qk_anchor(
    registry: Any,
    anchor: RealQKAnchor,
    *,
    destination_source_flat_index: int,
) -> InstalledRealQKAnchor:
    """Copy one active survivor's complete Q+K affine pair into a survivor slot."""

    if not isinstance(anchor, RealQKAnchor):
        raise ResidualAttentionQKAnchorError("anchor must be RealQKAnchor.")
    groups = int(registry.key_value_heads)
    anchor_source = anchor.source_flat_index(groups)
    destination_source = _nonnegative_int(
        destination_source_flat_index, "destination_source_flat_index"
    )
    anchor_slot = _slot_for_source(registry, anchor_source)
    destination_slot = _slot_for_source(registry, destination_source)
    names = (
        "prototype_q_weight",
        "prototype_k_weight",
        "prototype_q_bias",
        "prototype_k_bias",
    )
    flat = {name: _flat_prototype(registry, name) for name in names}
    q_weight = flat["prototype_q_weight"]
    k_weight = flat["prototype_k_weight"]
    if q_weight is None or k_weight is None:
        raise ResidualAttentionQKAnchorError(
            "The registry must retain both Q and K prototype weights."
        )
    before_state = _state_sha256s(registry)
    before_registry = _registry_sha256(registry)
    backups = {
        name: None if value is None else value[destination_slot].detach().clone()
        for name, value in flat.items()
    }
    destination_pair_before = _qk_pair_sha256(
        q_weight[destination_slot],
        k_weight[destination_slot],
        None
        if flat["prototype_q_bias"] is None
        else flat["prototype_q_bias"][destination_slot],
        None
        if flat["prototype_k_bias"] is None
        else flat["prototype_k_bias"][destination_slot],
    )
    anchor_pair = _qk_pair_sha256(
        q_weight[anchor_slot],
        k_weight[anchor_slot],
        None
        if flat["prototype_q_bias"] is None
        else flat["prototype_q_bias"][anchor_slot],
        None
        if flat["prototype_k_bias"] is None
        else flat["prototype_k_bias"][anchor_slot],
    )
    unaffected_before = {
        name: _slots_sha256(value, destination_slot) for name, value in flat.items()
    }
    mutation_required = anchor_slot != destination_slot
    try:
        with torch.inference_mode():
            for value in flat.values():
                if value is not None:
                    value[destination_slot].copy_(value[anchor_slot])
        installed_pair = _qk_pair_sha256(
            q_weight[destination_slot],
            k_weight[destination_slot],
            None
            if flat["prototype_q_bias"] is None
            else flat["prototype_q_bias"][destination_slot],
            None
            if flat["prototype_k_bias"] is None
            else flat["prototype_k_bias"][destination_slot],
        )
        if installed_pair != anchor_pair:
            raise ResidualAttentionQKAnchorError(
                "Installed Q/K pair is not byte-identical to the real anchor pair."
            )
        unaffected_after = {
            name: _slots_sha256(value, destination_slot)
            for name, value in flat.items()
        }
        unaffected_exact = unaffected_before == unaffected_after
        if not unaffected_exact:
            raise ResidualAttentionQKAnchorError(
                "A Q/K anchor edit changed an unaffected Q/K prototype slot."
            )
        after_state = _state_sha256s(registry)
        allowed = {name for name in names if flat[name] is not None}
        non_qk_exact = all(
            before_state[name] == after_state[name]
            for name in before_state
            if name not in allowed
        )
        if not non_qk_exact:
            raise ResidualAttentionQKAnchorError(
                "A Q/K anchor edit changed V/O, assignments, scales, or other state."
            )
    except Exception as error:
        try:
            with torch.inference_mode():
                for name, backup in backups.items():
                    value = flat[name]
                    if value is not None and backup is not None:
                        value[destination_slot].copy_(backup)
            restored = _registry_sha256(registry)
        except Exception as rollback_error:  # pragma: no cover - defense in depth
            raise ResidualAttentionQKAnchorError(
                "Q/K anchor installation failed and its immediate rollback also "
                "failed."
            ) from rollback_error
        if restored != before_registry:
            raise ResidualAttentionQKAnchorError(
                "Q/K anchor installation failed and did not restore the registry "
                "byte-exactly."
            ) from error
        raise
    report = QKAnchorEditReport(
        anchor=anchor,
        destination_source_flat_index=destination_source,
        anchor_slot=anchor_slot,
        destination_slot=destination_slot,
        registry_sha256_before=before_registry,
        registry_sha256_installed=_registry_sha256(registry),
        destination_pair_sha256_before=destination_pair_before,
        anchor_pair_sha256=anchor_pair,
        installed_pair_sha256=installed_pair,
        q_weight_sha256=_tensor_sha256(q_weight[anchor_slot]),
        k_weight_sha256=_tensor_sha256(k_weight[anchor_slot]),
        q_bias_sha256=_optional_tensor_sha256(
            None
            if flat["prototype_q_bias"] is None
            else flat["prototype_q_bias"][anchor_slot]
        ),
        k_bias_sha256=_optional_tensor_sha256(
            None
            if flat["prototype_k_bias"] is None
            else flat["prototype_k_bias"][anchor_slot]
        ),
        edited_tensor_bytes=sum(
            _tensor_bytes(None if value is None else value[destination_slot])
            for value in flat.values()
        ),
        mutation_required=mutation_required,
        unaffected_qk_slots_byte_exact=unaffected_exact,
        non_qk_registry_tensors_byte_exact=non_qk_exact,
    )
    return InstalledRealQKAnchor(registry, report, backups)


def _norm_epsilon(module: nn.Module, runtime: nn.Module) -> float | None:
    for name in ("variance_epsilon", "eps", "epsilon"):
        value = getattr(module, name, None)
        if value is not None:
            result = float(value)
            if not math.isfinite(result) or result <= 0.0:
                raise ResidualAttentionQKAnchorError(
                    f"{type(module).__name__}.{name} is not a positive epsilon."
                )
            return result
    config_value = getattr(getattr(runtime, "config", None), "rms_norm_eps", None)
    if config_value is None:
        return None
    result = float(config_value)
    if not math.isfinite(result) or result <= 0.0:
        raise ResidualAttentionQKAnchorError(
            "runtime.config.rms_norm_eps is not positive."
        )
    return result


def _norm_payload(module: Any, runtime: nn.Module) -> dict[str, Any] | None:
    if module is None:
        return None
    if not isinstance(module, nn.Module):
        raise ResidualAttentionQKAnchorError(
            "Q/K norm must be a torch module when present."
        )
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor):
        raise ResidualAttentionQKAnchorError(
            "Q/K norm module must expose its real weight tensor."
        )
    return {
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "weight_shape": list(weight.shape),
        "weight_sha256": _tensor_sha256(weight),
        "epsilon": _norm_epsilon(module, runtime),
    }


def audit_qk_anchor_assignments(
    runtime: nn.Module,
    registry: Any,
    *,
    anchor: RealQKAnchor,
    destination_source_flat_index: int,
) -> dict[str, Any]:
    """Audit every executable assignment which consumes the edited Q/K slot."""

    if not isinstance(runtime, nn.Module):
        raise ResidualAttentionQKAnchorError("runtime must be a torch module.")
    groups = int(registry.key_value_heads)
    destination_source = _nonnegative_int(
        destination_source_flat_index, "destination_source_flat_index"
    )
    destination_slot = _slot_for_source(registry, destination_source)
    anchor_slot = _slot_for_source(registry, anchor.source_flat_index(groups))
    assignments = _global_assignments(registry)
    permutations = registry.group_permutations.detach().to(
        device="cpu", dtype=torch.long
    )
    layers, assignment_groups = map(int, assignments.shape)
    if assignment_groups != groups or tuple(permutations.shape) != (layers, groups):
        raise ResidualAttentionQKAnchorError(
            "Assignments and group_permutations disagree with KV groups."
        )
    expected_permutation = list(range(groups))
    if any(sorted(map(int, row)) != expected_permutation for row in permutations):
        raise ResidualAttentionQKAnchorError(
            "Every group_permutations row must remain bijective."
        )
    model_layers = getattr(getattr(runtime, "model", None), "layers", None)
    if model_layers is None or len(model_layers) != layers:
        raise ResidualAttentionQKAnchorError(
            "Runtime layers do not align with the attention registry."
        )
    q_prototypes = _flat_prototype(registry, "prototype_q_weight")
    k_prototypes = _flat_prototype(registry, "prototype_k_weight")
    q_biases = _flat_prototype(registry, "prototype_q_bias")
    k_biases = _flat_prototype(registry, "prototype_k_bias")
    if q_prototypes is None or k_prototypes is None:
        raise ResidualAttentionQKAnchorError("Q/K prototypes are required.")
    anchor_pair_sha256 = _qk_pair_sha256(
        q_prototypes[anchor_slot],
        k_prototypes[anchor_slot],
        None if q_biases is None else q_biases[anchor_slot],
        None if k_biases is None else k_biases[anchor_slot],
    )
    destination_pair_sha256 = _qk_pair_sha256(
        q_prototypes[destination_slot],
        k_prototypes[destination_slot],
        None if q_biases is None else q_biases[destination_slot],
        None if k_biases is None else k_biases[destination_slot],
    )
    if destination_pair_sha256 != anchor_pair_sha256:
        raise ResidualAttentionQKAnchorError(
            "The destination Q/K affine pair is not byte-identical to the "
            "named real anchor slot."
        )
    q_rows = int(q_prototypes.shape[1])
    k_rows = int(k_prototypes.shape[1])
    role_scales = getattr(registry, "role_scales", None)
    sites: list[dict[str, Any]] = []
    for layer_index in range(layers):
        materialized_to_assignment = torch.argsort(permutations[layer_index])
        q_weight, q_bias = registry.materialize_projection(layer_index, "q")
        k_weight, k_bias = registry.materialize_projection(layer_index, "k")
        attention = model_layers[layer_index].self_attn
        for physical_group in range(groups):
            assignment_axis = int(materialized_to_assignment[physical_group].item())
            if (
                int(assignments[layer_index, assignment_axis].item())
                != destination_slot
            ):
                continue
            q_scale = 1.0
            k_scale = 1.0
            if role_scales is not None:
                q_scale = float(
                    role_scales[layer_index, assignment_axis, 0]
                    .detach()
                    .to(torch.float64)
                    .item()
                )
                k_scale = float(
                    role_scales[layer_index, assignment_axis, 1]
                    .detach()
                    .to(torch.float64)
                    .item()
                )
                if not math.isfinite(q_scale) or not math.isfinite(k_scale):
                    raise ResidualAttentionQKAnchorError(
                        "A deployed Q/K role scale is non-finite."
                    )
            q_start = physical_group * q_rows
            k_start = physical_group * k_rows
            expected_q = q_prototypes[destination_slot] * q_scale
            expected_k = k_prototypes[destination_slot] * k_scale
            if not torch.equal(q_weight[q_start : q_start + q_rows], expected_q):
                raise ResidualAttentionQKAnchorError(
                    "Materialized Q does not honor assignments/scales/permutation."
                )
            if not torch.equal(k_weight[k_start : k_start + k_rows], expected_k):
                raise ResidualAttentionQKAnchorError(
                    "Materialized K does not honor assignments/scales/permutation."
                )
            expected_q_bias = (
                None if q_biases is None else q_biases[destination_slot] * q_scale
            )
            expected_k_bias = (
                None if k_biases is None else k_biases[destination_slot] * k_scale
            )
            actual_q_bias = (
                None if q_bias is None else q_bias[q_start : q_start + q_rows]
            )
            actual_k_bias = (
                None if k_bias is None else k_bias[k_start : k_start + k_rows]
            )
            if (expected_q_bias is None) != (actual_q_bias is None) or (
                expected_q_bias is not None
                and not torch.equal(expected_q_bias, actual_q_bias)
            ):
                raise ResidualAttentionQKAnchorError(
                    "Materialized Q bias violates the anchor assignment semantics."
                )
            if (expected_k_bias is None) != (actual_k_bias is None) or (
                expected_k_bias is not None
                and not torch.equal(expected_k_bias, actual_k_bias)
            ):
                raise ResidualAttentionQKAnchorError(
                    "Materialized K bias violates the anchor assignment semantics."
                )
            sites.append(
                {
                    "layer_index": layer_index,
                    "materialized_physical_group": physical_group,
                    "canonical_assignment_axis": assignment_axis,
                    "active_slot": destination_slot,
                    "source_flat_index": destination_source,
                    "q_role_scale": q_scale,
                    "k_role_scale": k_scale,
                    "permutation_row": [
                        int(value) for value in permutations[layer_index].tolist()
                    ],
                    "permutation_row_sha256": _tensor_sha256(permutations[layer_index]),
                    "q_norm": _norm_payload(
                        getattr(attention, "q_norm", None), runtime
                    ),
                    "k_norm": _norm_payload(
                        getattr(attention, "k_norm", None), runtime
                    ),
                    "materialized_q_sha256": _tensor_sha256(
                        q_weight[q_start : q_start + q_rows]
                    ),
                    "materialized_k_sha256": _tensor_sha256(
                        k_weight[k_start : k_start + k_rows]
                    ),
                    "materialized_q_bias_sha256": _optional_tensor_sha256(
                        actual_q_bias
                    ),
                    "materialized_k_bias_sha256": _optional_tensor_sha256(
                        actual_k_bias
                    ),
                }
            )
    if not sites:
        raise ResidualAttentionQKAnchorError(
            "The destination survivor has no executable shared assignments."
        )
    anchor_attention = model_layers[anchor.layer_index].self_attn
    return {
        "schema": QK_ASSIGNMENT_AUDIT_SCHEMA,
        "anchor": anchor.as_dict(key_value_heads=groups),
        "anchor_active_slot": anchor_slot,
        "anchor_provenance_norm_semantics": {
            "q_norm": _norm_payload(getattr(anchor_attention, "q_norm", None), runtime),
            "k_norm": _norm_payload(getattr(anchor_attention, "k_norm", None), runtime),
        },
        "destination_source_flat_index": destination_source,
        "destination_active_slot": destination_slot,
        "anchor_pair_sha256": anchor_pair_sha256,
        "destination_pair_sha256": destination_pair_sha256,
        "destination_pair_byte_identical_to_anchor": True,
        "assignment_count": len(sites),
        "assignments": sites,
        "assignment_coordinates": [
            [site["layer_index"], site["materialized_physical_group"]] for site in sites
        ],
        "assignments_sha256": hashlib.sha256(
            "|".join(
                f"{site['layer_index']}:{site['materialized_physical_group']}:"
                f"{site['canonical_assignment_axis']}:{site['active_slot']}"
                for site in sites
            ).encode("ascii")
        ).hexdigest(),
        "group_permutations_sha256": _tensor_sha256(permutations),
        "role_scales_sha256": _optional_tensor_sha256(role_scales),
        "q_and_k_copied_from_same_real_active_slot": True,
        "q_or_k_averaged_before_softmax": False,
        "logical_layer_q_norm_k_norm_preserved": True,
        "norm_semantics": (
            "Qwen3 q_norm/k_norm are logical-layer modules; each shared Q/K "
            "prototype is evaluated through that executable layer's unchanged "
            "real norm weights and epsilon"
        ),
        "all_materialized_qk_slices_verified_exact": True,
        "all_group_permutations_bijective": True,
        "every_shared_destination_assignment_audited": True,
    }
