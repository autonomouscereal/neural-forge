"""Chimera-omni Stable LatentMoE channel-mixer folding.

``StableLatentMoEFeedForward``
(``chimera-omni/src/chimera_omni/modeling/layers.py:362-445``) routes every
token to ``experts_per_token`` of ``num_experts`` latent experts, accumulates
their weighted outputs in a compact latent width, normalizes the accumulator,
projects it back into the trunk, and adds one or more full-width shared
experts.  The whole thing makes exactly one residual write, so it folds as one
composite transaction.  This implements section 2 of
``docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md``.

Every solve is a ridge, an SVD or a deterministic functional-column selection.
There is no optimizer, gradient, backward pass, parent forward, runtime sidecar
or database here.

Implementation deviations from the design document, with reasons
----------------------------------------------------------------

1. **An activation-parameterized dictionary fit instead of
   ``fit_functional_swiglu``.**  That helper hardcodes ``silu(gate) * up``
   (``hf_closed_form_causal_circuit_fold.py:1228``), and chimera's
   ``activation_kind = "situ"`` is a different nonlinearity
   (``layers.py:350-355``) that the k3-aligned reference actually uses.
   Reusing the helper would have closed this family for half its
   configurations.  The selection rule is the same one: ridge every source
   neuron into child coordinates, then pick a deterministic maximal-residual
   subset of the realizable feature columns, then solve the writer against the
   complete target.

2. **The experts are fitted on the parent's row assignment.**  The captured
   expert evidence *belongs* to the rows the parent routed there; fitting
   against the child's own recomputed assignment would pair rows the parent
   never sent to that expert with outputs it never produced.  The disagreement
   between the two assignments is measured (``router_top_k_agreement``) and
   gated, not assumed away.

3. **``router_bias`` is required to be equal, not transported.**  It is a
   non-gradient buffer (``layers.py:378``), so it is not a learned tensor, not
   in the IR, and not inside the snapshot/rollback envelope this lane relies
   on.  It enters *selection* only, so an unequal buffer means the child's
   selection boundary is somewhere else; that fails closed.

4. **The latent basis is the identity when the width does not change.**  A
   width-preserving fold is not perturbed by an arbitrary rotation it does not
   need, and the truncating case reports its discarded singular energy exactly.

5. **Alternation is guarded, not assumed.**  Each sub-step is an exact ridge
   given the others, so the *regularized* objective cannot rise; the
   unregularized one can.  Every sub-step is applied and reverted if the fit
   objective rose -- the same guard the CSA kernel uses.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_closed_form_causal_circuit_fold import GatedActivation, fit_rmsnorm_diagonal


SCHEMA = "neural-forge-chimera-stable-latent-moe-fold-v1"
EVIDENCE_SCHEMA = "neural-forge-chimera-stable-latent-moe-evidence-v1"

DESIGN_DOCUMENT = "docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md"

#: The activations chimera's ``FeedForward`` implements (``layers.py:350-355``).
GATED_ACTIVATIONS = ("silu", "situ")


class HFCausalStableLatentMoEFoldError(ValueError):
    """A Stable LatentMoE fold is unsupported or its evidence is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalStableLatentMoEFoldError(
            "PyTorch is required for Stable LatentMoE folding."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalStableLatentMoEFoldError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalStableLatentMoEFoldError(f"{name} must be positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise HFCausalStableLatentMoEFoldError(f"{name} must be positive and finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalStableLatentMoEFoldError(f"{name} must be non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise HFCausalStableLatentMoEFoldError(
            f"{name} must be finite and non-negative."
        )
    return result


def _tensor(value: Any, name: str, *, ndim: int | None = None) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise HFCausalStableLatentMoEFoldError(f"{name} must be a tensor.")
    if ndim is not None and value.ndim != ndim:
        raise HFCausalStableLatentMoEFoldError(
            f"{name} must be rank {ndim}, got rank {value.ndim}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise HFCausalStableLatentMoEFoldError(f"{name} contains non-finite values.")
    return result


def _mask(value: Any, name: str, shape: tuple[int, int]) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise HFCausalStableLatentMoEFoldError(f"{name} must have shape {shape}.")
    return value.detach().to(device="cpu", dtype=torch.bool).contiguous()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    decoded = json.loads(json.dumps(dict(body), allow_nan=False, sort_keys=True))
    decoded["report_sha256"] = canonical_sha256(decoded)
    return MappingProxyType(decoded)


def _relative_rmse(prediction: Any, target: Any) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _retained_energy(prediction: Any, target: Any) -> float:
    error = (prediction - target).square().sum()
    total = target.square().sum().clamp_min(1e-30)
    return float((1.0 - error / total).clamp(min=0.0, max=1.0).item())


def _energy_fraction(part: Any, total: Any) -> float:
    return float(
        (part.square().sum() / total.square().sum().clamp_min(1e-30)).item()
    )


def _ridge(source: Any, target: Any, ridge: float) -> Any:
    torch = _torch()
    if source.ndim != 2 or target.ndim != 2 or int(source.shape[0]) != int(
        target.shape[0]
    ):
        raise HFCausalStableLatentMoEFoldError(
            "Ridge evidence matrices are incompatible."
        )
    gram = source.T @ source
    return torch.linalg.solve(
        gram + ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64),
        source.T @ target,
    )


def _pivot_columns(values: Any, count: int) -> tuple[int, ...]:
    """Deterministic maximal-residual functional-column selection.

    The same rule as ``_pivot_function_columns``
    (``hf_closed_form_causal_circuit_fold.py:1099``): take the column with the
    most residual energy, deflate every other column against it, repeat.
    """

    torch = _torch()
    count = _positive_int(count, "count")
    available_count = int(values.shape[1])
    if count > available_count:
        raise HFCausalStableLatentMoEFoldError(
            "Cannot select more functional columns than the source exposes."
        )
    residual = values.clone()
    available = torch.ones(available_count, dtype=torch.bool)
    selected: list[int] = []
    for _index in range(count):
        energies = residual.square().sum(dim=0)
        energies = torch.where(
            available, energies, torch.full_like(energies, -1.0)
        )
        index = int(energies.argmax().item())
        selected.append(index)
        available[index] = False
        vector = residual[:, index]
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) <= 1e-30:
            continue
        direction = vector / norm
        residual = residual - direction.unsqueeze(1) * (
            direction.unsqueeze(0) @ residual
        )
    return tuple(selected)


# ---------------------------------------------------------------------------
# Geometry, policy, parameters, evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChimeraStableLatentMoEGeometry:
    """One ``StableLatentMoEFeedForward`` geometry (``layers.py:371-394``)."""

    hidden_size: int
    expert_count: int
    experts_per_token: int
    shared_expert_count: int
    latent_size: int
    routed_intermediate_size: int
    intermediate_size: int
    activation: str = "silu"
    situ_gate_beta: float = 4.0
    situ_up_beta: float = 25.0
    rms_norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "expert_count",
            "experts_per_token",
            "shared_expert_count",
            "latent_size",
            "routed_intermediate_size",
            "intermediate_size",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.experts_per_token > self.expert_count:
            raise HFCausalStableLatentMoEFoldError(
                "experts_per_token cannot exceed expert_count."
            )
        activation = str(self.activation).strip().lower()
        if activation not in GATED_ACTIVATIONS:
            raise HFCausalStableLatentMoEFoldError(
                "activation must be one of " + ", ".join(GATED_ACTIVATIONS) + "."
            )
        object.__setattr__(self, "activation", activation)
        for name in ("situ_gate_beta", "situ_up_beta", "rms_norm_epsilon"):
            object.__setattr__(
                self, name, _positive_float(getattr(self, name), name)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }

    def activate(self, gate: Any, up: Any) -> Any:
        """The parent's own gated activation (``layers.py:350-355``).

        One definition, shared with the dense fused gate/up lane: both fold the
        same ``FeedForward._activate``, so both read it from
        :class:`GatedActivation` instead of each carrying a copy that can drift
        from the executable.
        """

        return GatedActivation(
            kind=self.activation,
            gate_beta=self.situ_gate_beta,
            up_beta=self.situ_up_beta,
        )(gate, up)


@dataclass(frozen=True, slots=True)
class ChimeraStableLatentMoEFoldPolicy:
    """Frozen policy constants for one Stable LatentMoE fold (design section 5)."""

    ridge: float = 1e-6
    alternation_rounds: int = 2
    routing_identity_tolerance: float = 1e-5
    split_reconstruction_tolerance: float = 1e-6
    minimum_router_top_k_agreement: float = 0.5
    #: The sparse-routing row budget.  An expert fitted on a handful of rows is
    #: an interpolation of noise the endpoint gate cannot see, because it is
    #: only ever exercised on those same rare rows.
    minimum_expert_fit_rows: int = 8
    minimum_expert_row_fraction: float = 0.1
    bf16_replay: bool = True

    def __post_init__(self) -> None:
        for name in (
            "ridge",
            "routing_identity_tolerance",
            "split_reconstruction_tolerance",
            "minimum_router_top_k_agreement",
            "minimum_expert_row_fraction",
        ):
            object.__setattr__(
                self, name, _nonnegative_float(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "alternation_rounds",
            _positive_int(self.alternation_rounds, "alternation_rounds"),
        )
        object.__setattr__(
            self,
            "minimum_expert_fit_rows",
            _positive_int(self.minimum_expert_fit_rows, "minimum_expert_fit_rows"),
        )
        if not isinstance(self.bf16_replay, bool):
            raise HFCausalStableLatentMoEFoldError("bf16_replay must be boolean.")

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }


@dataclass(frozen=True, slots=True)
class ChimeraStableLatentMoEParameters:
    """The parent values a Stable LatentMoE fold carries rather than fits.

    ``router_bias`` is a non-gradient buffer, so it is carried here rather than
    compiled; the fold requires the child's to equal it.
    """

    router_bias: Any
    geometry: ChimeraStableLatentMoEGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, ChimeraStableLatentMoEGeometry):
            raise HFCausalStableLatentMoEFoldError(
                "geometry must be a ChimeraStableLatentMoEGeometry."
            )
        bias = _tensor(self.router_bias, "router_bias", ndim=1).clone()
        if int(bias.numel()) != self.geometry.expert_count:
            raise HFCausalStableLatentMoEFoldError(
                "router_bias must carry one entry per expert."
            )
        object.__setattr__(self, "router_bias", bias)


@dataclass(frozen=True, slots=True)
class ChimeraStableLatentMoELayerEvidence:
    """Complete cached circuit states for one chimera Stable LatentMoE branch.

    Block-level rows are ``[batch, sequence, width]``; everything the module
    computes on its flattened view is ``[batch * sequence, width]``, which is
    the layout the module itself uses (``layers.py:430``).  The per-expert row
    sets are ragged by construction: expert ``j`` only ever saw the rows the
    parent routed to it.
    """

    index: int
    block_input: Any
    feed_forward_norm_output: Any
    router_logit_rows: Any
    latent_rows: Any
    expert_input_rows: tuple[Any, ...]
    expert_gate_rows: tuple[Any, ...]
    expert_up_rows: tuple[Any, ...]
    expert_output_rows: tuple[Any, ...]
    routed_rows: Any
    normalized_routed_rows: Any
    routed_delta: Any
    shared_gate_rows: tuple[Any, ...]
    shared_up_rows: tuple[Any, ...]
    shared_delta_rows: tuple[Any, ...]
    feed_forward_delta: Any
    geometry: ChimeraStableLatentMoEGeometry
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or (
            self.index < 0
        ):
            raise HFCausalStableLatentMoEFoldError(
                "layer evidence index must be a non-negative integer."
            )
        if not isinstance(self.geometry, ChimeraStableLatentMoEGeometry):
            raise HFCausalStableLatentMoEFoldError(
                "geometry must be a ChimeraStableLatentMoEGeometry."
            )
        geometry = self.geometry
        block_rows = {
            "block_input": geometry.hidden_size,
            "feed_forward_norm_output": geometry.hidden_size,
            "feed_forward_delta": geometry.hidden_size,
        }
        for name, width in block_rows.items():
            value = _tensor(getattr(self, name), name, ndim=3).clone()
            if int(value.shape[2]) != width:
                raise HFCausalStableLatentMoEFoldError(
                    f"{name} must have width {width}."
                )
            object.__setattr__(self, name, value)
        batch = int(self.block_input.shape[0])
        sequence = int(self.block_input.shape[1])
        for name in block_rows:
            if tuple(int(size) for size in getattr(self, name).shape[:2]) != (
                batch,
                sequence,
            ):
                raise HFCausalStableLatentMoEFoldError(
                    f"{name} has the wrong sequence shape."
                )
        rows = batch * sequence
        flat_rows = {
            "router_logit_rows": geometry.expert_count,
            "latent_rows": geometry.latent_size,
            "routed_rows": geometry.latent_size,
            "normalized_routed_rows": geometry.latent_size,
            "routed_delta": geometry.hidden_size,
        }
        for name, width in flat_rows.items():
            value = _tensor(getattr(self, name), name, ndim=2).clone()
            if tuple(int(size) for size in value.shape) != (rows, width):
                raise HFCausalStableLatentMoEFoldError(
                    f"{name} must be [{rows}, {width}]."
                )
            object.__setattr__(self, name, value)
        expert_widths = (
            ("expert_input_rows", geometry.latent_size),
            ("expert_gate_rows", geometry.routed_intermediate_size),
            ("expert_up_rows", geometry.routed_intermediate_size),
            ("expert_output_rows", geometry.latent_size),
        )
        for name, width in expert_widths:
            values = tuple(getattr(self, name))
            if len(values) != geometry.expert_count:
                raise HFCausalStableLatentMoEFoldError(
                    f"{name} must carry one entry per expert."
                )
            cleaned = []
            for expert_index, value in enumerate(values):
                matrix = _tensor(value, f"{name}[{expert_index}]", ndim=2).clone()
                if int(matrix.shape[1]) != width:
                    raise HFCausalStableLatentMoEFoldError(
                        f"{name}[{expert_index}] must have width {width}."
                    )
                cleaned.append(matrix)
            object.__setattr__(self, name, tuple(cleaned))
        for expert_index in range(geometry.expert_count):
            counts = {
                int(getattr(self, name)[expert_index].shape[0])
                for name, _width in expert_widths
            }
            if len(counts) != 1:
                raise HFCausalStableLatentMoEFoldError(
                    f"Expert {expert_index} evidence row counts disagree: "
                    f"{sorted(counts)}."
                )
        shared_widths = (
            ("shared_gate_rows", geometry.intermediate_size),
            ("shared_up_rows", geometry.intermediate_size),
            ("shared_delta_rows", geometry.hidden_size),
        )
        for name, width in shared_widths:
            values = tuple(getattr(self, name))
            if len(values) != geometry.shared_expert_count:
                raise HFCausalStableLatentMoEFoldError(
                    f"{name} must carry one entry per shared expert."
                )
            cleaned = []
            for shared_index, value in enumerate(values):
                matrix = _tensor(value, f"{name}[{shared_index}]", ndim=2).clone()
                if tuple(int(size) for size in matrix.shape) != (rows, width):
                    raise HFCausalStableLatentMoEFoldError(
                        f"{name}[{shared_index}] must be [{rows}, {width}]."
                    )
                cleaned.append(matrix)
            object.__setattr__(self, name, tuple(cleaned))
        object.__setattr__(self, "metadata", _report(dict(self.metadata)))

    @property
    def batch(self) -> int:
        return int(self.block_input.shape[0])

    @property
    def sequence(self) -> int:
        return int(self.block_input.shape[1])

    @property
    def rows(self) -> int:
        return self.batch * self.sequence

    def flat(self, value: Any) -> Any:
        return value.reshape(self.rows, int(value.shape[-1])).contiguous()

    def tensor_items(self) -> tuple[tuple[str, Any], ...]:
        skipped = {"index", "geometry", "metadata"}
        items: list[tuple[str, Any]] = []
        for descriptor in fields(self):
            if descriptor.name in skipped:
                continue
            value = getattr(self, descriptor.name)
            if isinstance(value, tuple):
                items.extend(
                    (f"{descriptor.name}[{index}]", entry)
                    for index, entry in enumerate(value)
                )
            else:
                items.append((descriptor.name, value))
        return tuple(items)


# ---------------------------------------------------------------------------
# Parent capture
# ---------------------------------------------------------------------------


def chimera_stable_latent_moe_geometry(
    module: Any,
    *,
    hidden_size: int | None = None,
    activation: str | None = None,
    rms_norm_epsilon: float | None = None,
) -> ChimeraStableLatentMoEGeometry:
    """Read one chimera Stable LatentMoE geometry off a live module."""

    try:
        shared = tuple(module.shared_experts)
        routed = tuple(module.routed_experts)
        first_shared = shared[0]
        kind = activation
        if kind is None:
            kind = str(getattr(first_shared, "activation_kind", "silu"))
        epsilon = rms_norm_epsilon
        if epsilon is None:
            epsilon = float(getattr(module.routed_norm, "eps", 1e-6))
        return ChimeraStableLatentMoEGeometry(
            hidden_size=int(hidden_size or module.hidden_size),
            expert_count=len(routed),
            experts_per_token=int(module.experts_per_token),
            shared_expert_count=len(shared),
            latent_size=int(module.latent_size),
            routed_intermediate_size=int(routed[0].up.weight.shape[0]) // 2,
            intermediate_size=int(first_shared.up.weight.shape[0]) // 2,
            activation=kind,
            situ_gate_beta=float(getattr(first_shared, "situ_gate_beta", 4.0)),
            situ_up_beta=float(getattr(first_shared, "situ_up_beta", 25.0)),
            rms_norm_epsilon=epsilon,
        )
    except (AttributeError, IndexError) as exc:
        raise HFCausalStableLatentMoEFoldError(
            "module does not expose the chimera StableLatentMoEFeedForward "
            "layout."
        ) from exc


def chimera_stable_latent_moe_parameters(
    module: Any,
    *,
    geometry: ChimeraStableLatentMoEGeometry | None = None,
) -> ChimeraStableLatentMoEParameters:
    """Carry the parent's non-gradient router-balancing buffer."""

    resolved = geometry or chimera_stable_latent_moe_geometry(module)
    bias = getattr(module, "router_bias", None)
    if bias is None:
        raise HFCausalStableLatentMoEFoldError(
            "module does not expose the router_bias balancing buffer."
        )
    return ChimeraStableLatentMoEParameters(router_bias=bias, geometry=resolved)


def capture_chimera_stable_latent_moe_layer_evidence(
    module: Any,
    normalized_rows: Any,
    *,
    layer_index: int,
    geometry: ChimeraStableLatentMoEGeometry | None = None,
    block_input: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ChimeraStableLatentMoELayerEvidence:
    """Run the parent channel mixer exactly once and seal every boundary."""

    torch = _torch()
    resolved = geometry or chimera_stable_latent_moe_geometry(module)
    rows = normalized_rows.detach()
    if rows.ndim != 3 or int(rows.shape[2]) != resolved.hidden_size:
        raise HFCausalStableLatentMoEFoldError(
            "normalized_rows must be [batch, sequence, hidden_size]."
        )
    flat = rows.reshape(-1, resolved.hidden_size)
    with torch.no_grad():
        logits = module.router(flat)
        routing = stable_latent_moe_routing(
            logits,
            module.router_bias,
            experts_per_token=resolved.experts_per_token,
        )
        latent = module.latent_down(flat)
        expert_inputs = []
        expert_gates = []
        expert_ups = []
        expert_outputs = []
        for expert_index, expert in enumerate(module.routed_experts):
            expert_rows = routing.expert_rows[expert_index]
            block = latent.index_select(0, expert_rows)
            fused = expert.up(block)
            gate, up = fused.chunk(2, dim=-1)
            expert_inputs.append(block)
            expert_gates.append(gate)
            expert_ups.append(up)
            expert_outputs.append(expert(block))
        routed = _accumulate_routed(
            routing,
            [_tensor(value, "expert output", ndim=2) for value in expert_outputs],
            rows=int(flat.shape[0]),
            width=resolved.latent_size,
        )
        normalized_routed = module.routed_norm(routed.to(dtype=flat.dtype))
        routed_delta = module.latent_up(normalized_routed)
        shared_gates = []
        shared_ups = []
        shared_deltas = []
        for shared in module.shared_experts:
            fused = shared.up(flat)
            gate, up = fused.chunk(2, dim=-1)
            shared_gates.append(gate)
            shared_ups.append(up)
            shared_deltas.append(shared(flat))
        delta = module(rows)
    resolved_input = rows if block_input is None else block_input
    sealed = {
        "schema": EVIDENCE_SCHEMA,
        "component": "chimera_stable_latent_moe_branch_capture",
        "parent_forward_calls": 1,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        **dict(metadata or {}),
    }
    return ChimeraStableLatentMoELayerEvidence(
        index=int(layer_index),
        block_input=resolved_input,
        feed_forward_norm_output=rows,
        router_logit_rows=logits,
        latent_rows=latent,
        expert_input_rows=tuple(expert_inputs),
        expert_gate_rows=tuple(expert_gates),
        expert_up_rows=tuple(expert_ups),
        expert_output_rows=tuple(expert_outputs),
        routed_rows=routed,
        normalized_routed_rows=normalized_routed,
        routed_delta=routed_delta,
        shared_gate_rows=tuple(shared_gates),
        shared_up_rows=tuple(shared_ups),
        shared_delta_rows=tuple(shared_deltas),
        feed_forward_delta=delta,
        geometry=resolved,
        metadata=sealed,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StableLatentMoERouting:
    """One recomputed top-k routing: assignments, mixture weights, row sets."""

    raw_scores: Any
    expert_indices: Any
    weights: Any
    expert_rows: tuple[Any, ...]
    expert_weights: tuple[Any, ...]

    @property
    def expert_row_counts(self) -> tuple[int, ...]:
        return tuple(int(value.numel()) for value in self.expert_rows)


def stable_latent_moe_routing(
    router_logit_rows: Any,
    router_bias: Any,
    *,
    experts_per_token: int,
) -> StableLatentMoERouting:
    """Recompute the parent's own routing from one captured router boundary.

    ``router_bias`` enters *selection* only; the mixture weights are the
    unbiased sigmoid scores renormalized over the selected set
    (``layers.py:399-406``).
    """

    torch = _torch()
    logits = _tensor(router_logit_rows, "router_logit_rows", ndim=2)
    bias = _tensor(router_bias, "router_bias", ndim=1)
    experts = int(logits.shape[1])
    if int(bias.numel()) != experts:
        raise HFCausalStableLatentMoEFoldError(
            "router_bias must carry one entry per expert."
        )
    experts_per_token = _positive_int(experts_per_token, "experts_per_token")
    if experts_per_token > experts:
        raise HFCausalStableLatentMoEFoldError(
            "experts_per_token cannot exceed the expert count."
        )
    raw = torch.sigmoid(logits)
    biased = raw + bias.reshape(1, -1)
    _values, indices = torch.topk(biased, k=experts_per_token, dim=-1)
    selected = raw.gather(1, indices)
    weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    expert_rows = []
    expert_weights = []
    for expert_index in range(experts):
        assigned = indices == expert_index
        rows, slots = assigned.nonzero(as_tuple=True)
        expert_rows.append(rows.contiguous())
        expert_weights.append(weights[rows, slots].contiguous())
    return StableLatentMoERouting(
        raw_scores=raw.contiguous(),
        expert_indices=indices.contiguous(),
        weights=weights.contiguous(),
        expert_rows=tuple(expert_rows),
        expert_weights=tuple(expert_weights),
    )


def _accumulate_routed(
    routing: StableLatentMoERouting,
    outputs: Sequence[Any],
    *,
    rows: int,
    width: int,
) -> Any:
    torch = _torch()
    routed = torch.zeros(rows, width, dtype=torch.float64)
    for expert_index, expert_rows in enumerate(routing.expert_rows):
        if int(expert_rows.numel()) == 0:
            continue
        contribution = routing.expert_weights[expert_index].reshape(-1, 1) * (
            outputs[expert_index]
        )
        routed = routed.index_add(0, expert_rows, contribution)
    return routed.contiguous()


# ---------------------------------------------------------------------------
# The evidence-consistency gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StableLatentMoEConsistency:
    """The routing identity and the channel split of one sealed MoE branch."""

    routing: StableLatentMoERouting
    routing_relative_rmse: float
    expert_input_relative_rmse: float
    split_relative_rmse: float
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _report(dict(self.diagnostics)))


def stable_latent_moe_consistency(
    evidence: ChimeraStableLatentMoELayerEvidence,
    parameters: ChimeraStableLatentMoEParameters,
    *,
    row_mask: Any | None = None,
    routing_tolerance: float = 1e-5,
    split_tolerance: float = 1e-6,
    fail_closed: bool = True,
) -> StableLatentMoEConsistency:
    """Recover the parent's routing and check both exact identities.

    1. **Routing identity.**  The recovered row set of every expert must have
       exactly the row count the captured expert boundary has -- an integer
       equality, so a mis-recovered assignment fails before any tolerance is
       consulted -- the recovered rows must reproduce the captured expert input
       rows, and the weighted accumulation of the captured expert outputs must
       reproduce the captured accumulator.
    2. **Channel split.**  ``routed_delta + sum_i shared_delta_i`` must
       reconstruct the captured branch write, on three independently captured
       values.
    """

    torch = _torch()
    if not isinstance(evidence, ChimeraStableLatentMoELayerEvidence):
        raise HFCausalStableLatentMoEFoldError(
            "evidence must be a ChimeraStableLatentMoELayerEvidence."
        )
    if not isinstance(parameters, ChimeraStableLatentMoEParameters):
        raise HFCausalStableLatentMoEFoldError(
            "parameters must be a ChimeraStableLatentMoEParameters."
        )
    if parameters.geometry.as_dict() != evidence.geometry.as_dict():
        raise HFCausalStableLatentMoEFoldError(
            "The carried parameters and the evidence describe different "
            "geometries."
        )
    routing_tolerance = _nonnegative_float(routing_tolerance, "routing_tolerance")
    split_tolerance = _nonnegative_float(split_tolerance, "split_tolerance")
    geometry = evidence.geometry
    mask = (
        torch.ones(evidence.batch, evidence.sequence, dtype=torch.bool)
        if row_mask is None
        else _mask(row_mask, "row_mask", (evidence.batch, evidence.sequence))
    )
    flat_mask = mask.reshape(-1)
    routing = stable_latent_moe_routing(
        evidence.router_logit_rows,
        parameters.router_bias,
        experts_per_token=geometry.experts_per_token,
    )
    captured_counts = tuple(
        int(value.shape[0]) for value in evidence.expert_output_rows
    )
    if routing.expert_row_counts != captured_counts:
        raise HFCausalStableLatentMoEFoldError(
            "The routing recovered from the captured router output assigns "
            f"{list(routing.expert_row_counts)} rows per expert, but the "
            f"captured expert boundaries carry {list(captured_counts)}. The "
            "recomputed routing is not the parent's."
        )
    input_errors = []
    for expert_index, expert_rows in enumerate(routing.expert_rows):
        if int(expert_rows.numel()) == 0:
            continue
        input_errors.append(
            _relative_rmse(
                evidence.latent_rows.index_select(0, expert_rows),
                evidence.expert_input_rows[expert_index],
            )
        )
    input_error = max(input_errors) if input_errors else 0.0
    routed = _accumulate_routed(
        routing,
        evidence.expert_output_rows,
        rows=evidence.rows,
        width=geometry.latent_size,
    )
    routing_error = _relative_rmse(
        routed[flat_mask], evidence.routed_rows[flat_mask]
    )
    total = evidence.routed_delta.clone()
    for shared in evidence.shared_delta_rows:
        total = total + shared
    delta = evidence.flat(evidence.feed_forward_delta)
    split_error = _relative_rmse(total[flat_mask], delta[flat_mask])
    expected_rows = (
        float(evidence.rows * geometry.experts_per_token)
        / float(geometry.expert_count)
    )
    diagnostics = {
        "schema": SCHEMA,
        "component": "stable_latent_moe_evidence_consistency",
        "layer_index": int(evidence.index),
        "evidence_rows": int(flat_mask.sum().item()),
        "parent_forward_calls": 0,
        "expert_row_counts": list(routing.expert_row_counts),
        "expert_row_counts_match_captured_boundaries": True,
        "expected_expert_rows": expected_rows,
        "routing_identity_relative_rmse": routing_error,
        "routing_identity_tolerance": routing_tolerance,
        "expert_input_relative_rmse": input_error,
        "split_reconstruction_relative_rmse": split_error,
        "split_reconstruction_tolerance": split_tolerance,
        "routed_energy_fraction": _energy_fraction(
            evidence.routed_delta[flat_mask], delta[flat_mask]
        ),
        "shared_energy_fraction": _energy_fraction(
            sum(
                evidence.shared_delta_rows[1:],
                evidence.shared_delta_rows[0],
            )[flat_mask]
            if evidence.shared_delta_rows
            else torch.zeros_like(delta)[flat_mask],
            delta[flat_mask],
        ),
        "router_bias_sha256": _tensor_sha256(parameters.router_bias),
        "routed_rows_sha256": _tensor_sha256(evidence.routed_rows),
        "feed_forward_delta_sha256": _tensor_sha256(evidence.feed_forward_delta),
    }
    if fail_closed and routing_error > routing_tolerance:
        raise HFCausalStableLatentMoEFoldError(
            "The recovered routing does not reproduce the captured routed "
            f"accumulator ({routing_error:.3e} > {routing_tolerance:.3e})."
        )
    if fail_closed and input_error > routing_tolerance:
        raise HFCausalStableLatentMoEFoldError(
            "The recovered expert row sets do not reproduce the captured "
            f"expert inputs ({input_error:.3e} > {routing_tolerance:.3e})."
        )
    if fail_closed and split_error > split_tolerance:
        raise HFCausalStableLatentMoEFoldError(
            "The routed and shared sub-branch deltas do not reconstruct the "
            f"sealed branch write ({split_error:.3e} > {split_tolerance:.3e})."
        )
    return StableLatentMoEConsistency(
        routing=routing,
        routing_relative_rmse=routing_error,
        expert_input_relative_rmse=input_error,
        split_relative_rmse=split_error,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Latent basis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MoELatentBasis:
    """One orthonormal latent basis plus the energy its truncation discards."""

    basis: Any
    retained_energy_fraction: float
    identity: bool

    def project(self, rows: Any) -> Any:
        return (rows @ self.basis).contiguous()


def moe_latent_basis(rows: Any, target_size: int) -> MoELatentBasis:
    """Leading right singular vectors of the observed latent rows."""

    torch = _torch()
    matrix = rows.reshape(-1, int(rows.shape[-1]))
    source = int(matrix.shape[1])
    target_size = _positive_int(target_size, "target latent size")
    if target_size > source:
        raise HFCausalStableLatentMoEFoldError(
            "Latent expansion is outside the contraction-only fold."
        )
    if target_size == source:
        return MoELatentBasis(
            basis=torch.eye(source, dtype=torch.float64),
            retained_energy_fraction=1.0,
            identity=True,
        )
    _u, singular, vectors = torch.linalg.svd(matrix, full_matrices=False)
    energy = singular.square()
    retained = float(
        (energy[:target_size].sum() / energy.sum().clamp_min(1e-30)).item()
    )
    return MoELatentBasis(
        basis=vectors[:target_size].T.contiguous(),
        retained_energy_fraction=retained,
        identity=False,
    )


# ---------------------------------------------------------------------------
# The compiled child
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HFCausalStableLatentMoEFold:
    """One compiled chimera Stable LatentMoE channel mixer plus its receipt."""

    router_weight: Any
    router_bias: Any
    latent_down_weight: Any
    expert_gate_weights: tuple[Any, ...]
    expert_up_weights: tuple[Any, ...]
    expert_down_weights: tuple[Any, ...]
    routed_norm_weight: Any
    latent_up_weight: Any
    shared_gate_weights: tuple[Any, ...]
    shared_up_weights: tuple[Any, ...]
    shared_down_weights: tuple[Any, ...]
    geometry: ChimeraStableLatentMoEGeometry
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, ChimeraStableLatentMoEGeometry):
            raise HFCausalStableLatentMoEFoldError(
                "geometry must be a ChimeraStableLatentMoEGeometry."
            )
        geometry = self.geometry
        shapes = {
            "router_weight": (geometry.expert_count, geometry.hidden_size),
            "latent_down_weight": (geometry.latent_size, geometry.hidden_size),
            "latent_up_weight": (geometry.hidden_size, geometry.latent_size),
        }
        for name, shape in shapes.items():
            value = _tensor(getattr(self, name), name, ndim=2).clone()
            if tuple(int(size) for size in value.shape) != shape:
                raise HFCausalStableLatentMoEFoldError(
                    f"{name} must have shape {shape}."
                )
            object.__setattr__(self, name, value)
        bias = _tensor(self.router_bias, "router_bias", ndim=1).clone()
        if int(bias.numel()) != geometry.expert_count:
            raise HFCausalStableLatentMoEFoldError(
                "router_bias must carry one entry per expert."
            )
        object.__setattr__(self, "router_bias", bias)
        gain = _tensor(self.routed_norm_weight, "routed_norm_weight", ndim=1).clone()
        if int(gain.numel()) != geometry.latent_size:
            raise HFCausalStableLatentMoEFoldError(
                "routed_norm_weight has the wrong latent width."
            )
        object.__setattr__(self, "routed_norm_weight", gain)
        groups = (
            (
                "expert_gate_weights",
                geometry.expert_count,
                (geometry.routed_intermediate_size, geometry.latent_size),
            ),
            (
                "expert_up_weights",
                geometry.expert_count,
                (geometry.routed_intermediate_size, geometry.latent_size),
            ),
            (
                "expert_down_weights",
                geometry.expert_count,
                (geometry.latent_size, geometry.routed_intermediate_size),
            ),
            (
                "shared_gate_weights",
                geometry.shared_expert_count,
                (geometry.intermediate_size, geometry.hidden_size),
            ),
            (
                "shared_up_weights",
                geometry.shared_expert_count,
                (geometry.intermediate_size, geometry.hidden_size),
            ),
            (
                "shared_down_weights",
                geometry.shared_expert_count,
                (geometry.hidden_size, geometry.intermediate_size),
            ),
        )
        for name, count, shape in groups:
            values = tuple(getattr(self, name))
            if len(values) != count:
                raise HFCausalStableLatentMoEFoldError(
                    f"{name} must carry {count} entries."
                )
            cleaned = []
            for index, value in enumerate(values):
                matrix = _tensor(value, f"{name}[{index}]", ndim=2).clone()
                if tuple(int(size) for size in matrix.shape) != shape:
                    raise HFCausalStableLatentMoEFoldError(
                        f"{name}[{index}] must have shape {shape}."
                    )
                cleaned.append(matrix)
            object.__setattr__(self, name, tuple(cleaned))
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    # -- replay ------------------------------------------------------------

    def route(self, child_rows: Any) -> StableLatentMoERouting:
        return stable_latent_moe_routing(
            child_rows @ self.router_weight.T,
            self.router_bias,
            experts_per_token=self.geometry.experts_per_token,
        )

    def _expert(self, index: int, rows: Any) -> Any:
        gate = rows @ self.expert_gate_weights[index].T
        up = rows @ self.expert_up_weights[index].T
        return self.geometry.activate(gate, up) @ self.expert_down_weights[index].T

    def _shared(self, index: int, rows: Any) -> Any:
        gate = rows @ self.shared_gate_weights[index].T
        up = rows @ self.shared_up_weights[index].T
        return self.geometry.activate(gate, up) @ self.shared_down_weights[index].T

    def _normalize(self, rows: Any) -> Any:
        epsilon = self.geometry.rms_norm_epsilon
        return rows * (
            rows.square().mean(dim=-1, keepdim=True) + epsilon
        ).rsqrt() * self.routed_norm_weight.reshape(1, -1)

    def replay(self, child_rows: Any) -> tuple[Any, Any, tuple[Any, ...]]:
        """Replay the whole channel mixer in FP64.

        Returns the branch delta, the routed sub-branch delta and one delta per
        shared expert, all ``[batch * sequence, hidden]``.
        """

        rows = _tensor(child_rows, "child_rows", ndim=3)
        flat = rows.reshape(-1, int(rows.shape[-1]))
        routing = self.route(flat)
        latent = flat @ self.latent_down_weight.T
        outputs = []
        for expert_index, expert_rows in enumerate(routing.expert_rows):
            if int(expert_rows.numel()) == 0:
                outputs.append(
                    latent.new_zeros((0, self.geometry.latent_size))
                )
                continue
            outputs.append(
                self._expert(expert_index, latent.index_select(0, expert_rows))
            )
        routed = _accumulate_routed(
            routing,
            outputs,
            rows=int(flat.shape[0]),
            width=self.geometry.latent_size,
        )
        routed_delta = self._normalize(routed) @ self.latent_up_weight.T
        shared_deltas = tuple(
            self._shared(index, flat)
            for index in range(self.geometry.shared_expert_count)
        )
        total = routed_delta
        for value in shared_deltas:
            total = total + value
        return total.contiguous(), routed_delta.contiguous(), shared_deltas

    def child_state_dict(self) -> dict[str, Any]:
        """Emit the compiled child in chimera Stable LatentMoE names."""

        torch = _torch()
        state: dict[str, Any] = {
            "router.weight": self.router_weight,
            "latent_down.weight": self.latent_down_weight,
            "routed_norm.weight": self.routed_norm_weight,
            "latent_up.weight": self.latent_up_weight,
        }
        for index in range(self.geometry.expert_count):
            state[f"routed_experts.{index}.up.weight"] = torch.cat(
                (self.expert_gate_weights[index], self.expert_up_weights[index]),
                dim=0,
            ).contiguous()
            state[f"routed_experts.{index}.down.weight"] = self.expert_down_weights[
                index
            ]
        for index in range(self.geometry.shared_expert_count):
            state[f"shared_experts.{index}.up.weight"] = torch.cat(
                (self.shared_gate_weights[index], self.shared_up_weights[index]),
                dim=0,
            ).contiguous()
            state[f"shared_experts.{index}.down.weight"] = self.shared_down_weights[
                index
            ]
        return state


def _replay_stable_latent_moe_bf16(
    fold: HFCausalStableLatentMoEFold,
    child_rows: Any,
) -> Any:
    """Replay router, top-k, experts, norm and both writers entirely in BF16."""

    torch = _torch()
    geometry = fold.geometry
    rows = child_rows.to(dtype=torch.bfloat16)
    flat = rows.reshape(-1, int(rows.shape[-1]))

    def cast(value: Any) -> Any:
        return value.to(dtype=torch.bfloat16)

    raw = torch.sigmoid(flat @ cast(fold.router_weight).transpose(0, 1))
    biased = raw + cast(fold.router_bias).reshape(1, -1)
    _values, indices = torch.topk(biased, k=geometry.experts_per_token, dim=-1)
    selected = raw.gather(1, indices)
    weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
        torch.tensor(1.0e-6, dtype=torch.bfloat16)
    )
    latent = flat @ cast(fold.latent_down_weight).transpose(0, 1)
    routed = torch.zeros(
        int(flat.shape[0]), geometry.latent_size, dtype=torch.bfloat16
    )
    for expert_index in range(geometry.expert_count):
        assigned = indices == expert_index
        expert_rows, slots = assigned.nonzero(as_tuple=True)
        if int(expert_rows.numel()) == 0:
            continue
        block = latent.index_select(0, expert_rows)
        gate = block @ cast(fold.expert_gate_weights[expert_index]).transpose(0, 1)
        up = block @ cast(fold.expert_up_weights[expert_index]).transpose(0, 1)
        value = geometry.activate(gate, up) @ cast(
            fold.expert_down_weights[expert_index]
        ).transpose(0, 1)
        routed = routed.index_add(
            0, expert_rows, weights[expert_rows, slots].reshape(-1, 1) * value
        )
    epsilon = torch.tensor(geometry.rms_norm_epsilon, dtype=torch.bfloat16)
    normalized = routed * (
        routed.square().mean(dim=-1, keepdim=True) + epsilon
    ).rsqrt() * cast(fold.routed_norm_weight).reshape(1, -1)
    total = normalized @ cast(fold.latent_up_weight).transpose(0, 1)
    for index in range(geometry.shared_expert_count):
        gate = flat @ cast(fold.shared_gate_weights[index]).transpose(0, 1)
        up = flat @ cast(fold.shared_up_weights[index]).transpose(0, 1)
        total = total + geometry.activate(gate, up) @ cast(
            fold.shared_down_weights[index]
        ).transpose(0, 1)
    return total.to(dtype=torch.float64)


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


def _fit_gated_dictionary(
    inputs: Any,
    gate_targets: Any,
    up_targets: Any,
    outputs: Any,
    *,
    neurons: int,
    ridge: float,
    geometry: ChimeraStableLatentMoEGeometry,
) -> tuple[Any, Any, Any, tuple[int, ...]]:
    """Ridge every source neuron into child coordinates, then pick and solve."""

    all_gate = _ridge(inputs, gate_targets, ridge).T.contiguous()
    all_up = _ridge(inputs, up_targets, ridge).T.contiguous()
    realizable = geometry.activate(inputs @ all_gate.T, inputs @ all_up.T)
    selected = _pivot_columns(realizable, neurons)
    torch = _torch()
    index = torch.tensor(selected, dtype=torch.long)
    gate_weight = all_gate.index_select(0, index).contiguous()
    up_weight = all_up.index_select(0, index).contiguous()
    features = geometry.activate(inputs @ gate_weight.T, inputs @ up_weight.T)
    down_weight = _ridge(features, outputs, ridge).T.contiguous()
    return gate_weight, up_weight, down_weight, selected


def compile_hf_causal_stable_latent_moe_fold(
    evidence: ChimeraStableLatentMoELayerEvidence,
    parameters: ChimeraStableLatentMoEParameters,
    geometry: ChimeraStableLatentMoEGeometry,
    *,
    fit_token_mask: Any,
    validation_token_mask: Any,
    valid_token_mask: Any | None = None,
    child_normalized_rows: Any | None = None,
    endpoint_target: Any | None = None,
    routed_endpoint_target: Any | None = None,
    shared_endpoint_targets: Sequence[Any] | None = None,
    policy: ChimeraStableLatentMoEFoldPolicy | None = None,
    consistency: StableLatentMoEConsistency | None = None,
) -> HFCausalStableLatentMoEFold:
    """Compile one chimera Stable LatentMoE channel mixer, closed form.

    Ordered recipe of design section 2.4: the two evidence gates and the expert
    row budget, the router ridge and its agreement floor, the latent basis and
    ``latent_down``, one gated dictionary per routed expert on *its own* rows,
    the child accumulator replay, ``routed_norm`` and the endpoint-coupled
    ``latent_up``, the shared experts, bounded alternation, BF16 replay and the
    no-branch floor.
    """

    torch = _torch()
    if not isinstance(evidence, ChimeraStableLatentMoELayerEvidence):
        raise HFCausalStableLatentMoEFoldError(
            "evidence must be a ChimeraStableLatentMoELayerEvidence."
        )
    if not isinstance(geometry, ChimeraStableLatentMoEGeometry):
        raise HFCausalStableLatentMoEFoldError(
            "geometry must be a ChimeraStableLatentMoEGeometry."
        )
    source = evidence.geometry
    policy = policy or ChimeraStableLatentMoEFoldPolicy()
    constants = (
        "expert_count",
        "experts_per_token",
        "shared_expert_count",
        "activation",
        "situ_gate_beta",
        "situ_up_beta",
    )
    differing = [
        name
        for name in constants
        if getattr(geometry, name) != getattr(source, name)
    ]
    if differing:
        raise HFCausalStableLatentMoEFoldError(
            "The expert count, the experts per token, the shared expert count "
            "and the activation are contract constants, not fold coordinates; "
            f"changing {', '.join(sorted(differing))} is a kind translation "
            f"(see {DESIGN_DOCUMENT} section 2.1)."
        )
    batch, sequence = evidence.batch, evidence.sequence
    rows = evidence.rows
    child = (
        evidence.feed_forward_norm_output
        if child_normalized_rows is None
        else _tensor(child_normalized_rows, "child_normalized_rows", ndim=3)
    )
    if tuple(int(size) for size in child.shape[:2]) != (batch, sequence):
        raise HFCausalStableLatentMoEFoldError(
            "child rows have the wrong sequence shape."
        )
    if int(child.shape[2]) != geometry.hidden_size:
        raise HFCausalStableLatentMoEFoldError(
            "child rows do not match the child hidden size."
        )
    fit = _mask(fit_token_mask, "fit_token_mask", (batch, sequence))
    validation = _mask(
        validation_token_mask, "validation_token_mask", (batch, sequence)
    )
    valid = (
        (fit | validation)
        if valid_token_mask is None
        else _mask(valid_token_mask, "valid_token_mask", (batch, sequence))
    )
    if (
        bool((fit & validation).any().item())
        or not bool(fit.any().item())
        or not bool(validation.any().item())
        or bool(((fit | validation) & ~valid).any().item())
    ):
        raise HFCausalStableLatentMoEFoldError(
            "MoE fit/validation masks must be disjoint nonempty valid subsets."
        )
    flat_fit = fit.reshape(-1)
    flat_validation = validation.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_child = child.reshape(rows, geometry.hidden_size)

    def resolve(name: str, value: Any, fallback: Any) -> Any:
        if value is None:
            return fallback
        resolved = _tensor(value, name, ndim=2)
        if tuple(int(size) for size in resolved.shape) != (
            rows,
            geometry.hidden_size,
        ):
            raise HFCausalStableLatentMoEFoldError(
                f"{name} must be [{rows}, {geometry.hidden_size}]."
            )
        return resolved

    delta_target = resolve(
        "endpoint_target",
        endpoint_target,
        evidence.flat(evidence.feed_forward_delta),
    )
    routed_target = resolve(
        "routed_endpoint_target", routed_endpoint_target, evidence.routed_delta
    )
    if shared_endpoint_targets is None:
        shared_targets = tuple(evidence.shared_delta_rows)
    else:
        shared_targets = tuple(
            resolve(f"shared_endpoint_targets[{index}]", value, None)
            for index, value in enumerate(shared_endpoint_targets)
        )
    if len(shared_targets) != geometry.shared_expert_count:
        raise HFCausalStableLatentMoEFoldError(
            "shared_endpoint_targets must carry one target per shared expert."
        )

    # Step 1 -- the ridge-independent evidence gates.
    gates = consistency or stable_latent_moe_consistency(
        evidence,
        parameters,
        row_mask=valid,
        routing_tolerance=policy.routing_identity_tolerance,
        split_tolerance=policy.split_reconstruction_tolerance,
    )
    routing = gates.routing
    expected_rows = (
        float(rows * geometry.experts_per_token) / float(geometry.expert_count)
    )
    fit_counts = []
    for expert_index, expert_rows in enumerate(routing.expert_rows):
        selected = flat_fit.index_select(0, expert_rows) if int(
            expert_rows.numel()
        ) else torch.zeros(0, dtype=torch.bool)
        count = int(selected.sum().item())
        fit_counts.append(count)
        floor = max(
            policy.minimum_expert_fit_rows,
            int(math.ceil(policy.minimum_expert_row_fraction * expected_rows)),
        )
        if count < floor:
            raise HFCausalStableLatentMoEFoldError(
                f"Routed expert {expert_index} owns only {count} fit rows, "
                f"below the sealed floor of {floor} (the router sent it "
                f"{int(expert_rows.numel())} of {rows} rows against an expected "
                f"{expected_rows:.1f}); an expert fitted on that few rows is an "
                "interpolation the endpoint gate cannot see."
            )

    # Step 2 -- the router, then the child's own routing.
    child_fit = flat_child[flat_fit]
    router_weight = _ridge(
        child_fit, evidence.router_logit_rows[flat_fit], policy.ridge
    ).T.contiguous()
    child_routing = stable_latent_moe_routing(
        flat_child @ router_weight.T,
        parameters.router_bias,
        experts_per_token=geometry.experts_per_token,
    )
    agreement = float(
        (
            torch.sort(child_routing.expert_indices, dim=-1).values
            == torch.sort(routing.expert_indices, dim=-1).values
        )
        .all(dim=-1)[flat_valid]
        .to(dtype=torch.float64)
        .mean()
        .item()
    )
    if agreement < policy.minimum_router_top_k_agreement:
        raise HFCausalStableLatentMoEFoldError(
            f"The installed router keeps only {agreement:.4f} of the parent's "
            "top-k expert selection, below the sealed policy fraction "
            f"({policy.minimum_router_top_k_agreement:.4f})."
        )

    # Step 3 -- the latent basis and the latent consumer.
    basis = moe_latent_basis(evidence.latent_rows[flat_fit], geometry.latent_size)
    latent_down_weight = _ridge(
        child_fit, basis.project(evidence.latent_rows)[flat_fit], policy.ridge
    ).T.contiguous()
    child_latent = flat_child @ latent_down_weight.T

    # Step 4 -- one gated dictionary per routed expert, on its own rows.
    expert_gate_weights = []
    expert_up_weights = []
    expert_down_weights = []
    expert_selections = []
    for expert_index, expert_rows in enumerate(routing.expert_rows):
        selected = flat_fit.index_select(0, expert_rows)
        fit_rows = expert_rows[selected]
        gate_weight, up_weight, down_weight, chosen = _fit_gated_dictionary(
            child_latent.index_select(0, fit_rows),
            evidence.expert_gate_rows[expert_index][selected],
            evidence.expert_up_rows[expert_index][selected],
            basis.project(evidence.expert_output_rows[expert_index])[selected],
            neurons=geometry.routed_intermediate_size,
            ridge=policy.ridge,
            geometry=geometry,
        )
        expert_gate_weights.append(gate_weight)
        expert_up_weights.append(up_weight)
        expert_down_weights.append(down_weight)
        expert_selections.append(chosen)

    # Step 5 -- the child accumulator, the norm, and the routed writer.
    def child_expert_outputs(active: StableLatentMoERouting) -> list[Any]:
        outputs = []
        for expert_index, expert_rows in enumerate(active.expert_rows):
            if int(expert_rows.numel()) == 0:
                outputs.append(child_latent.new_zeros((0, geometry.latent_size)))
                continue
            block = child_latent.index_select(0, expert_rows)
            gate = block @ expert_gate_weights[expert_index].T
            up = block @ expert_up_weights[expert_index].T
            outputs.append(
                geometry.activate(gate, up) @ expert_down_weights[expert_index].T
            )
        return outputs

    child_routed = _accumulate_routed(
        child_routing,
        child_expert_outputs(child_routing),
        rows=rows,
        width=geometry.latent_size,
    )
    routed_norm_fold = fit_rmsnorm_diagonal(
        child_routed[flat_fit],
        basis.project(evidence.normalized_routed_rows)[flat_fit],
        epsilon=geometry.rms_norm_epsilon,
        ridge=policy.ridge,
    )
    child_normalized_routed = child_routed * (
        child_routed.square().mean(dim=-1, keepdim=True) + geometry.rms_norm_epsilon
    ).rsqrt() * routed_norm_fold.weight.reshape(1, -1)
    latent_up_weight = _ridge(
        child_normalized_routed[flat_fit], routed_target[flat_fit], policy.ridge
    ).T.contiguous()
    routed_delta = child_normalized_routed @ latent_up_weight.T

    # Step 6 -- the shared experts, each against its own captured delta.
    shared_gate_weights = []
    shared_up_weights = []
    shared_down_weights = []
    shared_selections = []
    shared_deltas = []
    for shared_index in range(geometry.shared_expert_count):
        gate_weight, up_weight, down_weight, chosen = _fit_gated_dictionary(
            child_fit,
            evidence.shared_gate_rows[shared_index][flat_fit],
            evidence.shared_up_rows[shared_index][flat_fit],
            shared_targets[shared_index][flat_fit],
            neurons=geometry.intermediate_size,
            ridge=policy.ridge,
            geometry=geometry,
        )
        shared_gate_weights.append(gate_weight)
        shared_up_weights.append(up_weight)
        shared_down_weights.append(down_weight)
        shared_selections.append(chosen)
        features = geometry.activate(
            flat_child @ gate_weight.T, flat_child @ up_weight.T
        )
        shared_deltas.append(features @ down_weight.T)

    # Step 7 -- bounded alternation between the routed and shared writers.
    def objective() -> float:
        prediction = routed_delta
        for value in shared_deltas:
            prediction = prediction + value
        residual = (delta_target - prediction)[flat_fit]
        return float(residual.square().sum().item())

    shared_features = [
        geometry.activate(
            flat_child @ shared_gate_weights[index].T,
            flat_child @ shared_up_weights[index].T,
        )
        for index in range(geometry.shared_expert_count)
    ]
    trace = [
        {
            "round": 0,
            "step": "independent_sub_branch_targets",
            "fit_objective": objective(),
            "accepted": True,
        }
    ]
    for round_index in range(1, policy.alternation_rounds):
        current = objective()
        remainder = delta_target - sum(
            shared_deltas[1:], shared_deltas[0]
        ) if shared_deltas else delta_target
        candidate = _ridge(
            child_normalized_routed[flat_fit], remainder[flat_fit], policy.ridge
        ).T.contiguous()
        candidate_delta = child_normalized_routed @ candidate.T
        previous_weight, previous_delta = latent_up_weight, routed_delta
        latent_up_weight, routed_delta = candidate, candidate_delta
        candidate_objective = objective()
        accepted = candidate_objective <= current
        if not accepted:
            latent_up_weight, routed_delta = previous_weight, previous_delta
        trace.append(
            {
                "round": round_index,
                "step": "routed_writer_against_delta_minus_shared",
                "fit_objective": candidate_objective if accepted else current,
                "candidate_fit_objective": candidate_objective,
                "accepted": bool(accepted),
            }
        )
        for shared_index in range(geometry.shared_expert_count):
            current = objective()
            others = routed_delta
            for other_index, value in enumerate(shared_deltas):
                if other_index != shared_index:
                    others = others + value
            candidate = _ridge(
                shared_features[shared_index][flat_fit],
                (delta_target - others)[flat_fit],
                policy.ridge,
            ).T.contiguous()
            candidate_delta = shared_features[shared_index] @ candidate.T
            previous_weight = shared_down_weights[shared_index]
            previous_delta = shared_deltas[shared_index]
            shared_down_weights[shared_index] = candidate
            shared_deltas[shared_index] = candidate_delta
            candidate_objective = objective()
            accepted = candidate_objective <= current
            if not accepted:
                shared_down_weights[shared_index] = previous_weight
                shared_deltas[shared_index] = previous_delta
            trace.append(
                {
                    "round": round_index,
                    "step": (
                        f"shared_writer_{shared_index:03d}_against_delta_minus_rest"
                    ),
                    "fit_objective": candidate_objective if accepted else current,
                    "candidate_fit_objective": candidate_objective,
                    "accepted": bool(accepted),
                }
            )

    fold = HFCausalStableLatentMoEFold(
        router_weight=router_weight,
        router_bias=parameters.router_bias,
        latent_down_weight=latent_down_weight,
        expert_gate_weights=tuple(expert_gate_weights),
        expert_up_weights=tuple(expert_up_weights),
        expert_down_weights=tuple(expert_down_weights),
        routed_norm_weight=routed_norm_fold.weight,
        latent_up_weight=latent_up_weight,
        shared_gate_weights=tuple(shared_gate_weights),
        shared_up_weights=tuple(shared_up_weights),
        shared_down_weights=tuple(shared_down_weights),
        geometry=geometry,
        report={"schema": SCHEMA, "component": "staged_before_receipt"},
    )
    predicted, replayed_routed, replayed_shared = fold.replay(child)

    # Step 8 -- BF16 replay, then the no-branch floor.
    predicted_bf16 = (
        _replay_stable_latent_moe_bf16(fold, child)
        if policy.bf16_replay
        else predicted
    )
    fit_target = delta_target[flat_fit]
    validation_target = delta_target[flat_validation]
    zero_baseline = _relative_rmse(
        torch.zeros_like(validation_target), validation_target
    )
    validation_error = _relative_rmse(predicted[flat_validation], validation_target)
    validation_error_bf16 = _relative_rmse(
        predicted_bf16[flat_validation], validation_target
    )
    if validation_error_bf16 >= zero_baseline:
        raise HFCausalStableLatentMoEFoldError(
            "The BF16 validation endpoint is at or above the no-branch baseline "
            f"({validation_error_bf16:.6f} >= {zero_baseline:.6f})."
        )
    shared_total = (
        sum(replayed_shared[1:], replayed_shared[0])
        if replayed_shared
        else torch.zeros_like(predicted)
    )
    shared_target_total = (
        sum(shared_targets[1:], shared_targets[0])
        if shared_targets
        else torch.zeros_like(delta_target)
    )
    body = {
        "schema": SCHEMA,
        "component": "chimera_stable_latent_moe_channel_mixer_fold",
        "method": (
            "routing_identity_gated_router_ridge_latent_pod_per_expert_"
            "dictionary_on_its_own_rows_endpoint_coupled_writers_with_bounded_"
            "alternation"
        ),
        "layer_index": int(evidence.index),
        "source_geometry": source.as_dict(),
        "target_geometry": geometry.as_dict(),
        "policy": policy.as_dict(),
        "consistency_report_sha256": gates.diagnostics["report_sha256"],
        "routing_identity_relative_rmse": gates.routing_relative_rmse,
        "expert_input_relative_rmse": gates.expert_input_relative_rmse,
        "split_reconstruction_relative_rmse": gates.split_relative_rmse,
        "routed_energy_fraction": gates.diagnostics["routed_energy_fraction"],
        "shared_energy_fraction": gates.diagnostics["shared_energy_fraction"],
        "expert_row_counts": list(routing.expert_row_counts),
        "expert_fit_row_counts": list(fit_counts),
        "expected_expert_rows": expected_rows,
        "expert_rows_are_the_parents_own_assignment": True,
        "router_top_k_agreement": agreement,
        "router_bias_carried_not_fitted": True,
        "router_bias_is_a_non_gradient_buffer": True,
        "latent_basis": "identity" if basis.identity else "observed_row_pod",
        "latent_retained_energy_fraction": basis.retained_energy_fraction,
        "selected_routed_neurons": [list(value) for value in expert_selections],
        "selected_shared_neurons": [list(value) for value in shared_selections],
        "routed_norm_report_sha256": routed_norm_fold.report["report_sha256"],
        "routed_norm_gain_relative_rmse": routed_norm_fold.report["relative_rmse"],
        "alternation_rounds": policy.alternation_rounds,
        "alternation_trace": trace,
        "alternation_objective_monotone": bool(
            all(
                trace[index]["fit_objective"]
                <= trace[index - 1]["fit_objective"] + 1e-12
                for index in range(1, len(trace))
            )
        ),
        "fit_rows": int(flat_fit.sum().item()),
        "validation_rows": int(flat_validation.sum().item()),
        "fit_residual_relative_rmse": _relative_rmse(
            predicted[flat_fit], fit_target
        ),
        "fit_residual_retained_energy_fraction": _retained_energy(
            predicted[flat_fit], fit_target
        ),
        "validation_residual_relative_rmse": validation_error,
        "validation_residual_retained_energy_fraction": _retained_energy(
            predicted[flat_validation], validation_target
        ),
        "validation_no_branch_relative_rmse": zero_baseline,
        "validation_beats_no_branch_baseline": bool(
            validation_error < zero_baseline
        ),
        "validation_residual_relative_rmse_bf16": validation_error_bf16,
        "bf16_versus_fp64_endpoint_relative_delta": _relative_rmse(
            predicted_bf16[flat_validation], predicted[flat_validation]
        ),
        "routed_sub_branch_validation_relative_rmse": _relative_rmse(
            replayed_routed[flat_validation], routed_target[flat_validation]
        ),
        "shared_sub_branch_validation_relative_rmse": _relative_rmse(
            shared_total[flat_validation], shared_target_total[flat_validation]
        ),
        "ridge": policy.ridge,
        "numeric_weight_matching_objective": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "router_weight_sha256": _tensor_sha256(router_weight),
        "latent_down_weight_sha256": _tensor_sha256(latent_down_weight),
        "latent_up_weight_sha256": _tensor_sha256(latent_up_weight),
    }
    return HFCausalStableLatentMoEFold(
        router_weight=router_weight,
        router_bias=parameters.router_bias,
        latent_down_weight=latent_down_weight,
        expert_gate_weights=tuple(expert_gate_weights),
        expert_up_weights=tuple(expert_up_weights),
        expert_down_weights=tuple(expert_down_weights),
        routed_norm_weight=routed_norm_fold.weight,
        latent_up_weight=latent_up_weight,
        shared_gate_weights=tuple(shared_gate_weights),
        shared_up_weights=tuple(shared_up_weights),
        shared_down_weights=tuple(shared_down_weights),
        geometry=geometry,
        report=_report(body),
    )


__all__ = [
    "DESIGN_DOCUMENT",
    "EVIDENCE_SCHEMA",
    "GATED_ACTIVATIONS",
    "SCHEMA",
    "ChimeraStableLatentMoEFoldPolicy",
    "ChimeraStableLatentMoEGeometry",
    "ChimeraStableLatentMoELayerEvidence",
    "ChimeraStableLatentMoEParameters",
    "HFCausalStableLatentMoEFold",
    "HFCausalStableLatentMoEFoldError",
    "MoELatentBasis",
    "StableLatentMoEConsistency",
    "StableLatentMoERouting",
    "capture_chimera_stable_latent_moe_layer_evidence",
    "chimera_stable_latent_moe_geometry",
    "chimera_stable_latent_moe_parameters",
    "compile_hf_causal_stable_latent_moe_fold",
    "moe_latent_basis",
    "stable_latent_moe_consistency",
    "stable_latent_moe_routing",
]
