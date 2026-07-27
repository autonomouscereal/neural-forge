"""Standalone semantic integration for LFM2 global residual closure.

The giant semantic layer-stack adapter is intentionally not imported here.
It can later translate its private cache/replay objects into the typed evidence
package below.  This module consumes only frozen tensors and the public
whole-stack linearization snapshot.  It performs no model forward, parent
recapture, backward pass, optimizer step, pruning, quantization, or runtime
sidecar installation.

A nonempty cache trajectory bank is joined through explicit semantic token maps:
donor rows are gathered from mapped parent histories, target rows are gathered
from frozen child primals, and both are scattered into one canonical boundary
archive.  Every trajectory receives equal total row mass regardless of token
count.  The final MLP writer is centered on the authenticated deferred Q-DEIM
prior and is returned as a corrected, still-uninstalled prior; all other dense
writers are installed transactionally.

This is the reusable two-output-writers-per-layer closure baseline, not the final
residual-attention solution.  In particular, it does not claim that O-only
attention repair is sufficient.  Site specifications explicitly carry their
parameterization and publication policy so a subsequent coupled version can
add Q/K/V input transports, short-conv B/C/X/FIR transports, and MLP gate/up
transports without changing evidence row assembly or transaction semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_fold_executor import run_fold_only_callback
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionDownWeightPrior,
    lfm2_cpcf_terminal_action_tensor_sha256,
)
from neural_forge.hf_lfm2_global_linearization import (
    Lfm2StackPrimalBatch,
    Lfm2WholeStackLinearizationKernel,
    Lfm2WholeStackPrimalSnapshot,
)
from neural_forge.hf_lfm2_global_residual_closure import (
    Lfm2GlobalResidualBasis,
    Lfm2GlobalResidualBoundary,
    Lfm2GlobalResidualBoundaryBundle,
    Lfm2GlobalResidualClosureOperator,
    Lfm2GlobalResidualClosurePolicy,
    Lfm2GlobalResidualClosureResult,
    Lfm2GlobalResidualCorrectedWriterState,
    Lfm2GlobalResidualWriterSite,
    Lfm2GlobalResidualWriterTopology,
    build_lfm2_global_residual_basis,
    fingerprint_lfm2_global_residual_tensor,
    materialize_lfm2_global_residual_corrected_writers,
    seal_lfm2_global_residual_boundary,
    seal_lfm2_global_residual_linearization_receipt,
    solve_lfm2_global_residual_closure,
)


LFM2_SEMANTIC_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA = (
    "neural-forge-lfm2-semantic-global-residual-integration-v1"
)
LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA = (
    "neural-forge-lfm2-semantic-source-local-output-global-residual-"
    "integration-v1"
)

_ABSOLUTE_OUTPUT_BOUNDARY_MODE = "absolute_endpoint_output_v1"
_SOURCE_LOCAL_OUTPUT_BOUNDARY_MODE = "source_local_action_output_v1"
_OUTPUT_BOUNDARY_MODES = frozenset(
    {_ABSOLUTE_OUTPUT_BOUNDARY_MODE, _SOURCE_LOCAL_OUTPUT_BOUNDARY_MODE}
)


class Lfm2SemanticGlobalResidualIntegrationError(RuntimeError):
    """Semantic closure evidence or transactional publication is invalid."""


class Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
    Lfm2SemanticGlobalResidualIntegrationError
):
    """Semantic closure evidence failed exact byte authentication."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for semantic global residual integration."
        ) from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2SemanticGlobalResidualIntegrationError(
            "Integration metadata must be finite canonical JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _seal(body: Mapping[str, Any]) -> Mapping[str, Any]:
    detached = dict(body)
    return MappingProxyType({**detached, "report_sha256": _canonical_sha256(detached)})


def verify_lfm2_semantic_global_residual_report(
    report: Mapping[str, Any],
) -> bool:
    if not isinstance(report, Mapping):
        return False
    try:
        body = dict(report)
        supplied = body.pop("report_sha256", None)
        return isinstance(supplied, str) and supplied == _canonical_sha256(body)
    except (TypeError, ValueError):
        return False


def _output_boundary_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in _OUTPUT_BOUNDARY_MODES:
        raise Lfm2SemanticGlobalResidualIntegrationError(
            "boundary_mode must be absolute_endpoint_output_v1 or "
            "source_local_action_output_v1."
        )
    return mode


def _owned_floating_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
    dtype: Any | None = None,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
        or value.requires_grad
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise Lfm2SemanticGlobalResidualIntegrationError(
            f"{name} must be a finite nonempty detached rank-{dimensions} "
            "floating tensor."
        )
    target_dtype = value.dtype if dtype is None else dtype
    return value.detach().to(device="cpu", dtype=target_dtype).contiguous().clone()


def _owned_index(value: Any, name: str) -> Any:
    torch = _torch()
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype not in integer_dtypes
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise Lfm2SemanticGlobalResidualIntegrationError(
            f"{name} must be a nonempty dense rank-one integer tensor."
        )
    return value.detach().to(device="cpu", dtype=torch.int64).contiguous().clone()


def _layer_operator_kind(batch: Lfm2StackPrimalBatch, layer_index: int) -> str:
    return batch.macrocells[layer_index].operator_kind


def _operator_site_id(layer_index: int, operator_kind: str) -> str:
    component = "self_attn" if operator_kind == "full_attention" else "conv"
    return f"model.layers.{layer_index}.{component}.out_proj.weight"


def _mlp_site_id(layer_index: int) -> str:
    return f"model.layers.{layer_index}.feed_forward.w2.weight"


@dataclass(frozen=True, slots=True)
class Lfm2SemanticGlobalResidualSiteSpec:
    """One extensible coefficient/materialization site declaration."""

    site_id: str
    layer_index: int
    causal_order: int
    writer_kind: str
    input_width: int
    role: str
    correction_parameterization: str
    publication_policy: str


def _site_specs(
    snapshot: Lfm2WholeStackPrimalSnapshot,
) -> tuple[Lfm2SemanticGlobalResidualSiteSpec, ...]:
    first = snapshot.batches[0]
    final_layer_index = snapshot.layer_count - 1
    specs = []
    for layer_index, macrocell in enumerate(first.macrocells):
        operator_kind = macrocell.operator_kind
        if operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2SemanticGlobalResidualIntegrationError(
                f"Layer {layer_index} has unsupported operator kind {operator_kind!r}."
            )
        operator_order = 2 * layer_index
        specs.extend(
            (
                Lfm2SemanticGlobalResidualSiteSpec(
                    site_id=_operator_site_id(layer_index, operator_kind),
                    layer_index=layer_index,
                    causal_order=operator_order,
                    writer_kind=(
                        "attention_output"
                        if operator_kind == "full_attention"
                        else "short_conv_output"
                    ),
                    input_width=macrocell.input_widths[0],
                    role="operator_output_writer",
                    correction_parameterization="delta_weight=U@C",
                    publication_policy="install_dense",
                ),
                Lfm2SemanticGlobalResidualSiteSpec(
                    site_id=_mlp_site_id(layer_index),
                    layer_index=layer_index,
                    causal_order=operator_order + 1,
                    writer_kind="mlp_down",
                    input_width=macrocell.input_widths[1],
                    role="mlp_down_writer",
                    correction_parameterization="delta_weight=U@C",
                    publication_policy=(
                        "return_corrected_prior"
                        if layer_index == final_layer_index
                        else "install_dense"
                    ),
                ),
            )
        )
    return tuple(specs)


def _site_spec_body(spec: Lfm2SemanticGlobalResidualSiteSpec) -> dict[str, Any]:
    return {
        "site_id": spec.site_id,
        "layer_index": spec.layer_index,
        "causal_order": spec.causal_order,
        "writer_kind": spec.writer_kind,
        "input_width": spec.input_width,
        "role": spec.role,
        "correction_parameterization": spec.correction_parameterization,
        "publication_policy": spec.publication_policy,
    }


@dataclass(frozen=True, slots=True)
class Lfm2SemanticGlobalResidualTrajectoryEvidence:
    """One cached donor trajectory joined to one frozen target primal."""

    batch_index: int
    primal_batch: Lfm2StackPrimalBatch = field(repr=False, compare=False)
    selected_donor_positions: Any = field(repr=False, compare=False)
    mapped_parent_post_operator_residuals: tuple[Any, ...] = field(
        repr=False,
        compare=False,
    )
    mapped_parent_post_layer_residuals: tuple[Any, ...] = field(
        repr=False,
        compare=False,
    )
    mapped_parent_terminal_final_norm: Any = field(repr=False, compare=False)
    authentication_sha256: str = ""
    mapped_parent_pre_residuals: tuple[Any, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, int)
            or self.batch_index < 0
        ):
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "batch_index must be a nonnegative integer."
            )
        primal = self.primal_batch
        if not isinstance(primal, Lfm2StackPrimalBatch) or not (
            primal.verify_authentication()
        ):
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                "primal_batch must be an authenticated public stack primal."
            )
        if primal.terminal is None:
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Every integration trajectory requires the terminal RMSNorm primal."
            )
        donor = _owned_index(
            self.selected_donor_positions,
            "selected_donor_positions",
        )
        if int(donor.numel()) != primal.row_count:
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Donor and target semantic row maps must have equal length."
            )
        operator = tuple(
            _owned_floating_tensor(
                value,
                f"mapped_parent_post_operator_residuals[{index}]",
                dimensions=3,
            )
            for index, value in enumerate(self.mapped_parent_post_operator_residuals)
        )
        layer = tuple(
            _owned_floating_tensor(
                value,
                f"mapped_parent_post_layer_residuals[{index}]",
                dimensions=3,
            )
            for index, value in enumerate(self.mapped_parent_post_layer_residuals)
        )
        pre = (
            None
            if self.mapped_parent_pre_residuals is None
            else tuple(
                _owned_floating_tensor(
                    value,
                    f"mapped_parent_pre_residuals[{index}]",
                    dimensions=3,
                )
                for index, value in enumerate(self.mapped_parent_pre_residuals)
            )
        )
        terminal = _owned_floating_tensor(
            self.mapped_parent_terminal_final_norm,
            "mapped_parent_terminal_final_norm",
            dimensions=3,
        )
        layer_count = len(primal.macrocells)
        if (
            len(operator) != layer_count
            or len(layer) != layer_count
            or (pre is not None and len(pre) != layer_count)
        ):
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Each trajectory must contain one operator, one layer, and, when "
                "supplied, one pre-residual history per primal macrocell."
            )
        hidden = primal.hidden_size
        parent_tensors = (
            (*operator, *layer, terminal)
            if pre is None
            else (*pre, *operator, *layer, terminal)
        )
        if any(int(value.shape[-1]) != hidden for value in parent_tensors):
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Mapped parent histories differ from target hidden width."
            )
        donor_rows = math.prod(terminal.shape[:-1])
        if any(math.prod(value.shape[:-1]) != donor_rows for value in parent_tensors):
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Mapped parent boundary histories disagree on donor row geometry."
            )
        if (
            bool((donor < 0).any().item())
            or bool((donor >= donor_rows).any().item())
            or int(_torch().unique(donor).numel()) != int(donor.numel())
        ):
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Selected donor positions must be unique and in range."
            )
        object.__setattr__(self, "selected_donor_positions", donor)
        object.__setattr__(
            self,
            "mapped_parent_post_operator_residuals",
            operator,
        )
        object.__setattr__(self, "mapped_parent_post_layer_residuals", layer)
        object.__setattr__(self, "mapped_parent_pre_residuals", pre)
        object.__setattr__(
            self,
            "mapped_parent_terminal_final_norm",
            terminal,
        )
        expected = self.recompute_authentication_sha256()
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                "Trajectory authentication_sha256 is incorrect."
            )
        object.__setattr__(self, "authentication_sha256", expected)

    def authentication_body(self) -> dict[str, Any]:
        body = {
            "schema": LFM2_SEMANTIC_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA,
            "kind": "semantic_global_residual_trajectory",
            "batch_index": self.batch_index,
            "primal_batch_sha256": self.primal_batch.authentication_sha256,
            "selected_donor_positions_sha256": (
                fingerprint_lfm2_global_residual_tensor(self.selected_donor_positions)
            ),
            "selected_target_positions_sha256": (
                fingerprint_lfm2_global_residual_tensor(
                    self.primal_batch.selected_target_positions
                )
            ),
            "canonical_boundary_row_indices_sha256": (
                fingerprint_lfm2_global_residual_tensor(
                    self.primal_batch.canonical_boundary_row_indices
                )
            ),
            "mapped_parent_post_operator_sha256": [
                fingerprint_lfm2_global_residual_tensor(value)
                for value in self.mapped_parent_post_operator_residuals
            ],
            "mapped_parent_post_layer_sha256": [
                fingerprint_lfm2_global_residual_tensor(value)
                for value in self.mapped_parent_post_layer_residuals
            ],
            "mapped_parent_terminal_final_norm_sha256": (
                fingerprint_lfm2_global_residual_tensor(
                    self.mapped_parent_terminal_final_norm
                )
            ),
            "semantic_row_count": self.primal_batch.row_count,
            "donor_gather": True,
            "target_gather": True,
            "canonical_scatter": True,
        }
        if self.mapped_parent_pre_residuals is not None:
            body.update(
                {
                    "mapped_parent_pre_residual_sha256": [
                        fingerprint_lfm2_global_residual_tensor(value)
                        for value in self.mapped_parent_pre_residuals
                    ],
                    "source_local_boundary_evidence_available": True,
                    "source_local_boundary_evidence_layer_count": len(
                        self.mapped_parent_pre_residuals
                    ),
                }
            )
        return body

    def recompute_authentication_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        return self.primal_batch.verify_authentication() and (
            self.authentication_sha256 == self.recompute_authentication_sha256()
        )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticGlobalResidualEvidencePackage:
    """Adapter-independent frozen evidence for one complete closure solve."""

    trajectories: tuple[Lfm2SemanticGlobalResidualTrajectoryEvidence, ...]
    live_writer_seeds: Mapping[str, Any] = field(repr=False, compare=False)
    deferred_final_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior = field(
        repr=False,
        compare=False,
    )
    mapped_parent_tied_table: Any = field(repr=False, compare=False)
    head_dimension: int
    evidence_sha256: str = ""
    snapshot: Lfm2WholeStackPrimalSnapshot = field(init=False, repr=False)

    def __post_init__(self) -> None:
        trajectories = tuple(self.trajectories)
        if (
            not trajectories
            or not all(
                isinstance(value, Lfm2SemanticGlobalResidualTrajectoryEvidence)
                and value.verify_authentication()
                for value in trajectories
            )
            or tuple(value.batch_index for value in trajectories)
            != tuple(range(len(trajectories)))
        ):
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                "Evidence must contain a nonempty authenticated trajectory bank "
                "with unique contiguous batch indices in canonical order."
            )
        snapshot = Lfm2WholeStackPrimalSnapshot(
            tuple(value.primal_batch for value in trajectories)
        )
        if not snapshot.verify_authentication() or not snapshot.terminal_present:
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                "Whole-stack snapshot authentication failed."
            )
        if (
            isinstance(self.head_dimension, bool)
            or not isinstance(self.head_dimension, int)
            or self.head_dimension <= 0
        ):
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "head_dimension must be a positive integer."
            )
        specs = _site_specs(snapshot)
        final_site_id = specs[-1].site_id
        expected_live_ids = {value.site_id for value in specs[:-1]}
        if (
            not isinstance(self.live_writer_seeds, Mapping)
            or set(self.live_writer_seeds) != expected_live_ids
        ):
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "live_writer_seeds must exactly cover every nondeferred writer."
            )
        prior = self.deferred_final_w2_prior
        if not isinstance(prior, Lfm2CpcfTerminalActionDownWeightPrior) or not (
            prior.verify_authentication()
        ):
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                "deferred_final_w2_prior is not authenticated."
            )
        seeds = {}
        first_batch = snapshot.batches[0]
        for site_index, spec in enumerate(specs[:-1]):
            site_id = spec.site_id
            width = spec.input_width
            seed = _owned_floating_tensor(
                self.live_writer_seeds[site_id],
                f"live_writer_seeds.{site_id}",
                dimensions=2,
            )
            if tuple(seed.shape) != (snapshot.hidden_size, width):
                raise Lfm2SemanticGlobalResidualIntegrationError(
                    f"Writer seed {site_id} has incorrect geometry."
                )
            layer_index, branch = divmod(site_index, 2)
            macrocell = first_batch.macrocells[layer_index]
            primal_seed = (
                macrocell.operator.parameters.output_weight
                if branch == 0
                else macrocell.mlp.parameters.down_weight
            )
            if not bool(_torch().equal(seed.to(_torch().float64), primal_seed)):
                raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                    f"Frozen primal is not centered on live writer seed {site_id}."
                )
            seeds[site_id] = seed
        final_width = specs[-1].input_width
        if tuple(prior.down_weight.shape) != (
            snapshot.hidden_size,
            final_width,
        ):
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Deferred final W2 prior has incorrect geometry."
            )
        final_primal_seed = first_batch.macrocells[-1].mlp.parameters.down_weight
        if not bool(
            _torch().equal(prior.down_weight.to(_torch().float64), final_primal_seed)
        ):
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                "Final macrocell primal is not centered on the deferred Q-DEIM "
                "W2 prior."
            )
        tied = _owned_floating_tensor(
            self.mapped_parent_tied_table,
            "mapped_parent_tied_table",
            dimensions=2,
        )
        if int(tied.shape[1]) != snapshot.hidden_size:
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Mapped tied table differs from target hidden width."
            )
        object.__setattr__(self, "trajectories", trajectories)
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "live_writer_seeds", MappingProxyType(seeds))
        object.__setattr__(self, "mapped_parent_tied_table", tied)
        expected = self.recompute_evidence_sha256()
        supplied = str(self.evidence_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                "Evidence package evidence_sha256 is incorrect."
            )
        object.__setattr__(self, "evidence_sha256", expected)
        if final_site_id in seeds:  # pragma: no cover - exact-set invariant
            raise AssertionError("deferred final writer leaked into live seeds")

    @property
    def final_site_id(self) -> str:
        return _site_specs(self.snapshot)[-1].site_id

    def writer_seed_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                **dict(self.live_writer_seeds),
                self.final_site_id: self.deferred_final_w2_prior.down_weight,
            }
        )

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_SEMANTIC_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA,
            "kind": "semantic_global_residual_evidence_package",
            "trajectory_count": len(self.trajectories),
            "trajectory_sha256": [
                value.authentication_sha256 for value in self.trajectories
            ],
            "whole_stack_snapshot_sha256": self.snapshot.authentication_sha256,
            "head_dimension": self.head_dimension,
            "site_specifications": [
                _site_spec_body(value) for value in _site_specs(self.snapshot)
            ],
            "live_writer_seed_sha256": {
                name: fingerprint_lfm2_global_residual_tensor(value)
                for name, value in sorted(self.live_writer_seeds.items())
            },
            "deferred_final_w2_prior_authentication_sha256": (
                self.deferred_final_w2_prior.authentication_sha256
            ),
            "mapped_parent_tied_table_sha256": (
                fingerprint_lfm2_global_residual_tensor(self.mapped_parent_tied_table)
            ),
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "target_model_forward_calls": 0,
            "all_evidence_frozen": True,
        }

    def recompute_evidence_sha256(self) -> str:
        return _canonical_sha256(self.authentication_body())

    def verify_authentication(self) -> bool:
        return (
            self.snapshot.verify_authentication()
            and self.deferred_final_w2_prior.verify_authentication()
            and all(value.verify_authentication() for value in self.trajectories)
            and self.evidence_sha256 == self.recompute_evidence_sha256()
        )


def build_lfm2_semantic_global_residual_topology(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
) -> Lfm2GlobalResidualWriterTopology:
    if not evidence.verify_authentication():
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Cannot build topology from mutated evidence."
        )
    sites = tuple(
        Lfm2GlobalResidualWriterSite(
            site_id=spec.site_id,
            layer_index=spec.layer_index,
            causal_order=spec.causal_order,
            writer_kind=spec.writer_kind,
            input_width=spec.input_width,
        )
        for spec in _site_specs(evidence.snapshot)
    )
    return Lfm2GlobalResidualWriterTopology(
        hidden_size=evidence.snapshot.hidden_size,
        head_dimension=evidence.head_dimension,
        sites=sites,
    )


def _selected_rows(value: Any, positions: Any, hidden: int) -> Any:
    return value.to(_torch().float64).reshape(-1, hidden).index_select(0, positions)


def build_lfm2_semantic_global_residual_boundaries(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
) -> Lfm2GlobalResidualBoundaryBundle:
    """Gather/scatter semantic rows into two-per-layer plus terminal banks."""

    torch = _torch()
    if not evidence.verify_authentication():
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Cannot build boundaries from mutated evidence."
        )
    hidden = evidence.snapshot.hidden_size
    layer_count = evidence.snapshot.layer_count
    writer_count = 2 * layer_count
    boundary_count = writer_count + 1
    total_rows = evidence.snapshot.row_count
    defects = [
        torch.empty((total_rows, hidden), dtype=torch.float64)
        for _ in range(boundary_count)
    ]
    masses = [torch.empty((total_rows,), dtype=torch.float64) for _ in defects]
    for trajectory in evidence.trajectories:
        primal = trajectory.primal_batch
        donor_index = trajectory.selected_donor_positions
        target_index = primal.selected_target_positions
        canonical_index = primal.canonical_boundary_row_indices
        per_row_mass = 1.0 / primal.row_count
        for layer_index, macrocell in enumerate(primal.macrocells):
            pairs = (
                (
                    trajectory.mapped_parent_post_operator_residuals[layer_index],
                    macrocell.operator.output,
                    2 * layer_index,
                ),
                (
                    trajectory.mapped_parent_post_layer_residuals[layer_index],
                    macrocell.mlp.output,
                    2 * layer_index + 1,
                ),
            )
            for parent, child, boundary_index in pairs:
                defect = _selected_rows(parent, donor_index, hidden) - _selected_rows(
                    child,
                    target_index,
                    hidden,
                )
                defects[boundary_index].index_copy_(0, canonical_index, defect)
                masses[boundary_index].index_fill_(
                    0,
                    canonical_index,
                    per_row_mass,
                )
        terminal_defect = _selected_rows(
            trajectory.mapped_parent_terminal_final_norm,
            donor_index,
            hidden,
        ) - _selected_rows(
            primal.terminal.output,
            target_index,
            hidden,
        )
        defects[-1].index_copy_(0, canonical_index, terminal_defect)
        masses[-1].index_fill_(0, canonical_index, per_row_mass)
    boundaries = []
    for boundary_index in range(boundary_count):
        if boundary_index == writer_count:
            boundary_id = "model.final_rmsnorm.output"
            layer_index = layer_count - 1
            kind = "terminal_observable"
        else:
            layer_index, branch = divmod(boundary_index, 2)
            suffix = "post_operator_residual" if branch == 0 else "post_layer_residual"
            boundary_id = f"model.layers.{layer_index}.{suffix}"
            kind = "post_operator_residual" if branch == 0 else "post_layer_residual"
        defect = defects[boundary_index]
        mass = masses[boundary_index]
        boundaries.append(
            seal_lfm2_global_residual_boundary(
                boundary_id=boundary_id,
                layer_index=layer_index,
                causal_order=boundary_index,
                boundary_kind=kind,
                defect=defect,
                defect_sha256=fingerprint_lfm2_global_residual_tensor(defect),
                row_mass=mass,
                row_mass_sha256=fingerprint_lfm2_global_residual_tensor(mass),
            )
        )
    return Lfm2GlobalResidualBoundaryBundle(tuple(boundaries))


def _seal_source_local_semantic_boundary(
    *,
    boundary_id: str,
    layer_index: int,
    causal_order: int,
    boundary_kind: str,
    defect: Any,
    row_mass: Any,
    parent_action_formula: str,
    child_action_formula: str,
    terminal_diagnostic_only: bool,
) -> Lfm2GlobalResidualBoundary:
    """Seal one local-action defect with its non-catch-up contract."""

    boundary = seal_lfm2_global_residual_boundary(
        boundary_id=boundary_id,
        layer_index=layer_index,
        causal_order=causal_order,
        boundary_kind=boundary_kind,
        defect=defect,
        defect_sha256=fingerprint_lfm2_global_residual_tensor(defect),
        row_mass=row_mass,
        row_mass_sha256=fingerprint_lfm2_global_residual_tensor(row_mass),
    )
    body = dict(boundary.report)
    body.pop("report_sha256")
    body.update(
        {
            "semantic_defect_mode": "mapped_source_local_action_difference",
            "parent_action_formula": parent_action_formula,
            "child_action_formula": child_action_formula,
            "absolute_parent_boundary_target_used": False,
            "accumulated_residual_origin_catchup_used": False,
            "common_parent_boundary_offset_invariant_by_construction": True,
            "independent_child_residual_origin_invariant_by_construction": True,
            "terminal_diagnostic_only": terminal_diagnostic_only,
            "solver_objective_active": not terminal_diagnostic_only,
            "terminal_absolute_observable_defect_used": False,
            "numeric_diagnostics_used_for_acceptance_or_veto": False,
            "quality_gate": False,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
    )
    return Lfm2GlobalResidualBoundary(
        boundary_id=boundary.boundary_id,
        layer_index=boundary.layer_index,
        causal_order=boundary.causal_order,
        boundary_kind=boundary.boundary_kind,
        defect=boundary.defect,
        row_mass=boundary.row_mass,
        report=_seal(body),
    )


def build_lfm2_semantic_source_local_global_residual_boundaries(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
) -> Lfm2GlobalResidualBoundaryBundle:
    """Build mapped parent-action minus live child-action boundary defects.

    This primitive deliberately does not target absolute mapped parent states.
    The terminal boundary is the analogous final-normalization local-action
    difference and is an explicit solver objective; no absolute terminal
    observable is used.
    """

    torch = _torch()
    if not evidence.verify_authentication():
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Cannot build source-local boundaries from mutated evidence."
        )
    if any(
        trajectory.mapped_parent_pre_residuals is None
        for trajectory in evidence.trajectories
    ):
        raise Lfm2SemanticGlobalResidualIntegrationError(
            "Source-local boundaries require authenticated mapped parent "
            "pre-residuals for every trajectory and layer."
        )
    hidden = evidence.snapshot.hidden_size
    layer_count = evidence.snapshot.layer_count
    writer_count = 2 * layer_count
    boundary_count = writer_count + 1
    total_rows = evidence.snapshot.row_count
    defects = [
        torch.empty((total_rows, hidden), dtype=torch.float64)
        for _ in range(boundary_count)
    ]
    masses = [torch.empty((total_rows,), dtype=torch.float64) for _ in defects]
    for trajectory in evidence.trajectories:
        primal = trajectory.primal_batch
        parent_pre = trajectory.mapped_parent_pre_residuals
        assert parent_pre is not None
        donor_index = trajectory.selected_donor_positions
        target_index = primal.selected_target_positions
        canonical_index = primal.canonical_boundary_row_indices
        per_row_mass = 1.0 / primal.row_count
        for layer_index, macrocell in enumerate(primal.macrocells):
            mapped_pre = _selected_rows(
                parent_pre[layer_index],
                donor_index,
                hidden,
            )
            mapped_post_operator = _selected_rows(
                trajectory.mapped_parent_post_operator_residuals[layer_index],
                donor_index,
                hidden,
            )
            mapped_post_layer = _selected_rows(
                trajectory.mapped_parent_post_layer_residuals[layer_index],
                donor_index,
                hidden,
            )
            child_pre = _selected_rows(
                macrocell.operator.residual,
                target_index,
                hidden,
            )
            child_post_operator = _selected_rows(
                macrocell.operator.output,
                target_index,
                hidden,
            )
            child_post_layer = _selected_rows(
                macrocell.mlp.output,
                target_index,
                hidden,
            )
            operator_defect = (mapped_post_operator - mapped_pre) - (
                child_post_operator - child_pre
            )
            layer_defect = (mapped_post_layer - mapped_post_operator) - (
                child_post_layer - child_post_operator
            )
            defects[2 * layer_index].index_copy_(
                0,
                canonical_index,
                operator_defect,
            )
            defects[2 * layer_index + 1].index_copy_(
                0,
                canonical_index,
                layer_defect,
            )
            masses[2 * layer_index].index_fill_(
                0,
                canonical_index,
                per_row_mass,
            )
            masses[2 * layer_index + 1].index_fill_(
                0,
                canonical_index,
                per_row_mass,
            )
        mapped_terminal_action = _selected_rows(
            trajectory.mapped_parent_terminal_final_norm,
            donor_index,
            hidden,
        ) - _selected_rows(
            trajectory.mapped_parent_post_layer_residuals[-1],
            donor_index,
            hidden,
        )
        child_terminal_action = _selected_rows(
            primal.terminal.output,
            target_index,
            hidden,
        ) - _selected_rows(
            primal.terminal.residual,
            target_index,
            hidden,
        )
        terminal_defect = mapped_terminal_action - child_terminal_action
        defects[-1].index_copy_(0, canonical_index, terminal_defect)
        masses[-1].index_fill_(0, canonical_index, per_row_mass)
    boundaries = []
    for boundary_index in range(boundary_count):
        terminal = boundary_index == writer_count
        if terminal:
            boundary_id = "model.final_rmsnorm.output"
            layer_index = layer_count - 1
            kind = "terminal_observable"
            parent_formula = (
                "mapped_parent_terminal_final_norm-"
                "mapped_parent_post_layer_residuals[-1]"
            )
            child_formula = "child_terminal_final_norm-child_terminal_input_residual"
        else:
            layer_index, branch = divmod(boundary_index, 2)
            suffix = "post_operator_residual" if branch == 0 else "post_layer_residual"
            boundary_id = f"model.layers.{layer_index}.{suffix}"
            kind = suffix
            if branch == 0:
                parent_formula = "mapped_parent_post_operator-mapped_parent_pre"
                child_formula = "child_post_operator-child_pre"
            else:
                parent_formula = "mapped_parent_post_layer-mapped_parent_post_operator"
                child_formula = "child_post_layer-child_post_operator"
        boundaries.append(
            _seal_source_local_semantic_boundary(
                boundary_id=boundary_id,
                layer_index=layer_index,
                causal_order=boundary_index,
                boundary_kind=kind,
                defect=defects[boundary_index],
                row_mass=masses[boundary_index],
                parent_action_formula=parent_formula,
                child_action_formula=child_formula,
                terminal_diagnostic_only=False,
            )
        )
    return Lfm2GlobalResidualBoundaryBundle(tuple(boundaries))


def build_lfm2_semantic_global_residual_basis(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    topology: Lfm2GlobalResidualWriterTopology,
    boundaries: Lfm2GlobalResidualBoundaryBundle,
) -> Lfm2GlobalResidualBasis:
    if not evidence.verify_authentication():
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Cannot build basis from mutated evidence."
        )
    defects = {
        boundary.boundary_id: boundary.defect for boundary in boundaries.boundaries
    }
    writers = dict(evidence.writer_seed_mapping())
    tied = evidence.mapped_parent_tied_table.to(_torch().float64)
    centered = (tied - tied.mean(dim=0, keepdim=True)).contiguous()
    terminal = {"mapped_parent_centered_tied_table": centered}
    return build_lfm2_global_residual_basis(
        topology,
        mapped_fit_residual_defects=defects,
        mapped_fit_residual_defects_sha256={
            name: fingerprint_lfm2_global_residual_tensor(value)
            for name, value in defects.items()
        },
        source_writer_weights=writers,
        source_writer_weights_sha256={
            name: fingerprint_lfm2_global_residual_tensor(value)
            for name, value in writers.items()
        },
        terminal_tied_table_directions=terminal,
        terminal_tied_table_directions_sha256={
            name: fingerprint_lfm2_global_residual_tensor(value)
            for name, value in terminal.items()
        },
    )


def _diagnostic_whole_stack_scales(
    snapshot: Lfm2WholeStackPrimalSnapshot,
    basis: Lfm2GlobalResidualBasis,
    boundaries: Lfm2GlobalResidualBoundaryBundle,
) -> tuple[
    float,
    tuple[float, ...],
    float,
    tuple[float, ...],
    Mapping[str, Any],
]:
    """Return useful deterministic scales, never an admission threshold.

    The evolving whole-stack linearizer may expose a propagated scale API.  A
    deterministic equal-batch local fallback remains available so integration
    tests and evidence assembly do not depend on private adapter types.  It is
    explicitly not represented as a certified global Lipschitz upper bound.
    """

    conditioning = snapshot.conditioning_scales(basis.output_basis)
    certified = snapshot.squared_norm_upper_bounds(basis.output_basis)
    # The linearizer sums unweighted per-batch boundary energy.  The closure W
    # normalizes each boundary, then averages boundary and hidden coordinates.
    # max(W) preserves those exact units as a conservative envelope when row
    # masses are nonuniform; it is not an exact weighted local energy.
    weight_normalization = max(
        float(boundary.row_mass.max().item()) for boundary in boundaries.boundaries
    ) / (len(boundaries.boundaries) * snapshot.hidden_size)
    raw_block = tuple(float(value) for value in conditioning.block_squared_scales)
    block = tuple(value * weight_normalization for value in raw_block)
    operator = math.fsum(block)
    certified_block = tuple(
        float(value) * weight_normalization
        for value in certified.block_squared_norm_upper_bounds
    )
    certified_operator = math.fsum(certified_block)
    if (
        len(block) != 2 * snapshot.layer_count
        or not math.isfinite(operator)
        or operator <= 0.0
        or any(not math.isfinite(value) or value <= 0.0 for value in block)
    ):
        raise Lfm2SemanticGlobalResidualIntegrationError(
            "Whole-stack diagnostic conditioning scales are invalid."
        )
    report = _seal(
        {
            "schema": LFM2_SEMANTIC_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA,
            "kind": "whole_stack_conditioning_scale",
            "source": conditioning.method,
            "raw_operator_squared_scale": conditioning.operator_squared_scale,
            "raw_block_squared_scales": list(raw_block),
            "closure_maximum_coordinate_weight": weight_normalization,
            "nonuniform_row_weight_treatment": "conservative_max_W_envelope",
            "operator_squared_norm_scale": operator,
            "block_squared_norm_scales": list(block),
            "weighted_certified_operator_squared_norm_upper_bound": (
                certified_operator
            ),
            "weighted_certified_block_squared_norm_upper_bounds": list(certified_block),
            "conditioning_scale_separate_from_certified_bound": True,
            "certified_global_lipschitz_upper_bound": False,
            "numeric_scale_used_for_semantic_decision": False,
            "numeric_scale_used_for_auto_veto": False,
        }
    )
    return operator, block, certified_operator, certified_block, report


@dataclass(frozen=True, slots=True)
class Lfm2SemanticGlobalResidualInstallation:
    """Authenticated result of one dense nonfinal-writer transaction."""

    topology: Lfm2GlobalResidualWriterTopology
    boundaries: Lfm2GlobalResidualBoundaryBundle
    basis: Lfm2GlobalResidualBasis
    closure_result: Lfm2GlobalResidualClosureResult
    corrected_final_w2_prior: Lfm2CpcfTerminalActionDownWeightPrior
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.corrected_final_w2_prior,
                Lfm2CpcfTerminalActionDownWeightPrior,
            )
            or not self.corrected_final_w2_prior.verify_authentication()
        ):
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                "Installation must carry an authenticated downstream terminal "
                "action W2 prior."
            )
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))

    def verify_authentication(self) -> bool:
        return (
            self.topology.verify_authentication()
            and self.boundaries.verify_authentication()
            and self.basis.verify_authentication()
            and self.closure_result.verify_authentication()
            and self.corrected_final_w2_prior.verify_authentication()
            and verify_lfm2_semantic_global_residual_report(self.report)
            and self.report.get("topology_sha256") == self.topology.topology_sha256
            and self.report.get("boundary_bundle_sha256")
            == self.boundaries.bundle_sha256
            and self.report.get("basis_sha256") == self.basis.basis_sha256
            and self.report.get("closure_result_sha256")
            == self.closure_result.report["report_sha256"]
            and self.report.get("corrected_final_w2_prior_sha256")
            == self.corrected_final_w2_prior.authentication_sha256
            and self.report.get(
                "final_layer_mlp_down_weight_prior_authentication_sha256"
            )
            == self.corrected_final_w2_prior.authentication_sha256
            and self.report.get("final_layer_mlp_down_weight_prior_tensor_sha256")
            == lfm2_cpcf_terminal_action_tensor_sha256(
                self.corrected_final_w2_prior.down_weight
            )
            and self.report.get("final_layer_mlp_down_weight_prior_install_count") == 0
        )


def _validate_live_parameters(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    live_writer_parameters: Mapping[str, Any],
    deferred_final_writer_parameter: Any,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    torch = _torch()
    expected = set(evidence.live_writer_seeds)
    if (
        not isinstance(live_writer_parameters, Mapping)
        or set(live_writer_parameters) != expected
    ):
        raise Lfm2SemanticGlobalResidualIntegrationError(
            "live_writer_parameters must exactly cover every nondeferred writer."
        )
    parameters = dict(live_writer_parameters)
    if any(
        not isinstance(value, torch.nn.Parameter)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        for value in parameters.values()
    ) or not isinstance(deferred_final_writer_parameter, torch.nn.Parameter):
        raise Lfm2SemanticGlobalResidualIntegrationError(
            "Every publication target must be an ordinary dense Parameter."
        )
    identities = [id(value) for value in parameters.values()]
    if len(set(identities)) != len(identities) or id(
        deferred_final_writer_parameter
    ) in set(identities):
        raise Lfm2SemanticGlobalResidualIntegrationError(
            "Writer Parameters must be independently addressable."
        )
    before = {}
    snapshots = {}
    for name, parameter in parameters.items():
        seed = evidence.live_writer_seeds[name]
        if tuple(parameter.shape) != tuple(seed.shape) or parameter.dtype != seed.dtype:
            raise Lfm2SemanticGlobalResidualIntegrationError(
                f"Live writer {name} differs from seed geometry or dtype."
            )
        observed = fingerprint_lfm2_global_residual_tensor(parameter)
        expected_sha = fingerprint_lfm2_global_residual_tensor(seed)
        if observed != expected_sha:
            raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                f"Live writer {name} is stale relative to frozen evidence."
            )
        before[name] = observed
        snapshots[name] = parameter.detach().to(device="cpu").contiguous().clone()
    final_sha = lfm2_cpcf_terminal_action_tensor_sha256(deferred_final_writer_parameter)
    if final_sha != evidence.deferred_final_w2_prior.live_deferred_w2_sha256:
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Live final W2 differs from the deferred prior lineage."
        )
    return parameters, snapshots, final_sha


def _transactionally_install_live_writers(
    parameters: Mapping[str, Any],
    snapshots: Mapping[str, Any],
    corrected: Lfm2GlobalResidualCorrectedWriterState,
    *,
    install_site_ids: Sequence[str],
) -> tuple[Mapping[str, str], Any]:
    torch = _torch()
    corrected_by_id = dict(
        zip(corrected.site_ids, corrected.writer_weights, strict=True)
    )
    installed: list[str] = []

    def transaction() -> Mapping[str, str]:
        try:
            with torch.no_grad():
                for site_id in install_site_ids:
                    parameters[site_id].copy_(
                        corrected_by_id[site_id].to(
                            device=parameters[site_id].device,
                            dtype=parameters[site_id].dtype,
                        )
                    )
                    installed.append(site_id)
            observed = {
                site_id: fingerprint_lfm2_global_residual_tensor(parameters[site_id])
                for site_id in install_site_ids
            }
            expected = {
                site_id: fingerprint_lfm2_global_residual_tensor(
                    corrected_by_id[site_id]
                )
                for site_id in install_site_ids
            }
            if observed != expected:
                raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
                    "Dense writer publication did not reproduce corrected bytes."
                )
            return MappingProxyType(observed)
        except Exception as exc:
            rollback_errors = []
            with torch.no_grad():
                for site_id in reversed(installed):
                    try:
                        parameters[site_id].copy_(
                            snapshots[site_id].to(
                                device=parameters[site_id].device,
                                dtype=parameters[site_id].dtype,
                            )
                        )
                    except Exception as rollback_exc:  # pragma: no cover - hardware
                        rollback_errors.append((site_id, repr(rollback_exc)))
            if rollback_errors:
                raise Lfm2SemanticGlobalResidualIntegrationError(
                    f"Writer transaction and rollback failed: {rollback_errors!r}."
                ) from exc
            raise Lfm2SemanticGlobalResidualIntegrationError(
                "Writer transaction failed and all changed writers were restored."
            ) from exc

    result, guard = run_fold_only_callback(
        transaction,
        phase="semantic-global-residual-dense-writer-transaction",
        protected_models=(),
        freeze_parameters=False,
    )
    return result, guard


def compile_install_lfm2_semantic_global_residual_closure(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    *,
    live_writer_parameters: Mapping[str, Any],
    deferred_final_writer_parameter: Any,
    policy: Lfm2GlobalResidualClosurePolicy | None = None,
    boundary_mode: str = _ABSOLUTE_OUTPUT_BOUNDARY_MODE,
) -> Lfm2SemanticGlobalResidualInstallation:
    """Solve one global fold, install nonfinal writers, and defer final W2."""

    mode = _output_boundary_mode(boundary_mode)
    source_local_mode = mode == _SOURCE_LOCAL_OUTPUT_BOUNDARY_MODE
    integration_schema = (
        LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA
        if source_local_mode
        else LFM2_SEMANTIC_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA
    )
    if not isinstance(evidence, Lfm2SemanticGlobalResidualEvidencePackage) or not (
        evidence.verify_authentication()
    ):
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "evidence must be an authenticated semantic closure package."
        )
    parameters, snapshots, final_before = _validate_live_parameters(
        evidence,
        live_writer_parameters,
        deferred_final_writer_parameter,
    )
    topology = build_lfm2_semantic_global_residual_topology(evidence)
    boundaries = (
        build_lfm2_semantic_source_local_global_residual_boundaries(evidence)
        if source_local_mode
        else build_lfm2_semantic_global_residual_boundaries(evidence)
    )
    if source_local_mode and any(
        boundary.report.get("semantic_defect_mode")
        != "mapped_source_local_action_difference"
        or boundary.report.get("absolute_parent_boundary_target_used") is not False
        or boundary.report.get("accumulated_residual_origin_catchup_used") is not False
        or boundary.report.get("solver_objective_active") is not True
        for boundary in boundaries.boundaries
    ):
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Source-local output closure received an absolute or inactive boundary."
        )
    basis = build_lfm2_semantic_global_residual_basis(
        evidence,
        topology,
        boundaries,
    )
    (
        operator_scale,
        block_scales,
        certified_operator_bound,
        certified_block_bounds,
        scale_report,
    ) = _diagnostic_whole_stack_scales(evidence.snapshot, basis, boundaries)
    linearization_receipt = seal_lfm2_global_residual_linearization_receipt(
        topology,
        boundaries,
        kernel_id=(
            "lfm2.semantic.whole-stack.source-local-output-residual-closure.v1"
            if source_local_mode
            else "lfm2.semantic.whole-stack.global-residual-closure.v1"
        ),
        source_sha256={
            "evidence_package": evidence.evidence_sha256,
            "whole_stack_snapshot": evidence.snapshot.authentication_sha256,
            "conditioning_scale": scale_report["report_sha256"],
        },
        analytic_math_components=(
            "semantic_token_bridge_donor_gather_target_gather_canonical_scatter",
            "continuous_fp64_whole_stack_attention_shortconv_swiglu_jvp",
            "explicit_exact_whole_stack_vjp",
            "terminal_final_rmsnorm_jvp_vjp",
        ),
        operator_squared_norm_upper_bound=certified_operator_bound,
        block_squared_norm_upper_bounds=certified_block_bounds,
        operator_squared_conditioning_scale=operator_scale,
        block_squared_conditioning_scales=block_scales,
        conditioning_scale_method=(
            "max-W-weighted-local-direct-two-boundary-energy-without-downstream-product"
        ),
    )
    kernel = Lfm2WholeStackLinearizationKernel(
        topology,
        boundaries,
        evidence.snapshot,
        linearization_receipt=linearization_receipt,
    )
    operator = Lfm2GlobalResidualClosureOperator(
        topology,
        basis,
        boundaries,
        kernel=kernel,
        policy=policy,
    )
    closure_result = solve_lfm2_global_residual_closure(operator)
    seeds = dict(evidence.writer_seed_mapping())
    corrected = materialize_lfm2_global_residual_corrected_writers(
        topology,
        closure_result,
        seed_writer_weights=seeds,
        seed_writer_weights_sha256={
            name: fingerprint_lfm2_global_residual_tensor(value)
            for name, value in seeds.items()
        },
    )
    corrected_by_id = dict(
        zip(corrected.site_ids, corrected.writer_weights, strict=True)
    )
    final_weight = corrected_by_id[evidence.final_site_id]
    base_prior = evidence.deferred_final_w2_prior
    corrected_final_prior = Lfm2CpcfTerminalActionDownWeightPrior(
        down_weight=final_weight,
        body_recipe_id=base_prior.body_recipe_id,
        body_recipe_sha256=base_prior.body_recipe_sha256,
        mlp_lane=base_prior.mlp_lane,
        selected_role=base_prior.selected_role,
        selected_candidate_sha256=base_prior.selected_candidate_sha256,
        preparation_sha256=base_prior.preparation_sha256,
        mlp_installation_core_sha256=base_prior.mlp_installation_core_sha256,
        live_deferred_w2_sha256=base_prior.live_deferred_w2_sha256,
    )
    final_prior_lineage = _seal(
        {
            "schema": integration_schema,
            "kind": "closure_corrected_terminal_action_prior_lineage",
            "boundary_mode": mode,
            "base_qdeim_prior_authentication_sha256": (
                base_prior.authentication_sha256
            ),
            "corrected_terminal_action_prior_authentication_sha256": (
                corrected_final_prior.authentication_sha256
            ),
            "closure_result_sha256": closure_result.report["report_sha256"],
            "topology_sha256": topology.topology_sha256,
            "site_id": evidence.final_site_id,
            "down_weight_sha256": lfm2_cpcf_terminal_action_tensor_sha256(
                corrected_final_prior.down_weight
            ),
            "dtype": str(final_weight.dtype).removeprefix("torch."),
            "shape": [int(value) for value in final_weight.shape],
            "installed": False,
            "installation_count": 0,
            "live_final_w2_sha256": final_before,
            "decoded_behavior_review_is_evidence_for_understanding": True,
            "verdict_required_for_continuation": False,
            "accept_veto_publish_delete_stop_authority": {
                "decoded_behavior_review": False,
                "numeric_metrics": False,
            },
            "quality_gate": False,
        }
    )
    if not evidence.verify_authentication():
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Evidence mutated before dense publication."
        )
    install_ids = tuple(site.site_id for site in topology.sites[:-1])
    post_hashes, install_guard = _transactionally_install_live_writers(
        parameters,
        snapshots,
        corrected,
        install_site_ids=install_ids,
    )
    final_after = lfm2_cpcf_terminal_action_tensor_sha256(
        deferred_final_writer_parameter
    )
    if final_after != final_before:
        # The transaction never targets final W2.  Restore live writers before
        # failing if external mutation nevertheless occurred concurrently.
        with _torch().no_grad():
            for site_id in install_ids:
                parameters[site_id].copy_(
                    snapshots[site_id].to(
                        device=parameters[site_id].device,
                        dtype=parameters[site_id].dtype,
                    )
                )
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Deferred final W2 changed during the nonfinal-writer transaction."
        )
    guard_body = install_guard.as_dict()
    body = {
        "schema": integration_schema,
        "kind": "semantic_global_residual_dense_installation",
        "boundary_mode": mode,
        "source_local_boundary_mode": source_local_mode,
        "absolute_endpoint_boundary_mode": not source_local_mode,
        "all_global_residual_defects_are_source_local_actions": source_local_mode,
        "absolute_parent_endpoint_defects_used": not source_local_mode,
        "accumulated_residual_origin_catchup_used": False,
        "source_local_terminal_action_solver_objective_active": source_local_mode,
        "evidence_package_sha256": evidence.evidence_sha256,
        "topology_sha256": topology.topology_sha256,
        "boundary_bundle_sha256": boundaries.bundle_sha256,
        "basis_sha256": basis.basis_sha256,
        "linearization_receipt_sha256": linearization_receipt["report_sha256"],
        "kernel_authentication_sha256": kernel.authentication_sha256,
        "closure_result_sha256": closure_result.report["report_sha256"],
        "corrected_writer_state_sha256": corrected.report["report_sha256"],
        "corrected_final_w2_prior_sha256": (
            corrected_final_prior.authentication_sha256
        ),
        "corrected_final_w2_closure_lineage_sha256": final_prior_lineage[
            "report_sha256"
        ],
        "trajectory_count": len(evidence.trajectories),
        "writer_site_count": len(topology.sites),
        "site_specifications": [
            _site_spec_body(value) for value in _site_specs(evidence.snapshot)
        ],
        "closure_scope": (
            f"{2 * evidence.snapshot.layer_count}_output_writers_only_"
            f"{'source_local_v1b' if source_local_mode else 'baseline_v1'}"
        ),
        "o_only_attention_repair_claimed_sufficient": False,
        "final_residual_attention_solution_claimed": False,
        "v2_input_transport_extension_roadmap": {
            "implemented_in_this_result": False,
            "attention": "coupled_q_k_v_input_transports_plus_o_output",
            "short_conv": "coupled_b_c_x_input_transports_optional_fir_plus_output",
            "mlp": "coupled_gate_up_input_transports_plus_down_output",
            "input_transport_materialization": "delta_weight=W_role@U@C",
            "output_writer_materialization": "delta_weight=U@C",
            "reuses_semantic_boundary_evidence": True,
            "reuses_dense_transaction_framework": True,
        },
        "residual_boundary_count": len(boundaries.boundaries) - 1,
        "terminal_boundary_count": 1,
        "installed_writer_site_ids": list(install_ids),
        "installed_writer_count": len(install_ids),
        "dense_live_writer_sha256_before": dict(
            sorted(
                (name, fingerprint_lfm2_global_residual_tensor(snapshots[name]))
                for name in install_ids
            )
        ),
        "dense_live_writer_sha256_after": dict(sorted(post_hashes.items())),
        "final_w2_site_id": evidence.final_site_id,
        "deferred_final_w2_sha256": final_before,
        "final_w2_installation_count": 0,
        "final_w2_live_sha256_before": final_before,
        "final_w2_live_sha256_after": final_after,
        "final_w2_unchanged": final_after == final_before,
        "final_w2_corrected_prior_returned": True,
        "final_layer_mlp_w2_deferred_to_fused_terminal": True,
        "final_layer_mlp_w2_install_count": 0,
        "final_layer_mlp_down_weight_prior_authentication_sha256": (
            corrected_final_prior.authentication_sha256
        ),
        "final_layer_mlp_down_weight_prior_tensor_sha256": (
            lfm2_cpcf_terminal_action_tensor_sha256(corrected_final_prior.down_weight)
        ),
        "final_layer_mlp_down_weight_prior_live_w2_sha256": (
            corrected_final_prior.live_deferred_w2_sha256
        ),
        "final_layer_mlp_down_weight_prior_installed_before_terminal": False,
        "final_layer_mlp_down_weight_prior_install_count": 0,
        "final_layer_mlp_down_weight_prior_closure_lineage_sha256": (
            final_prior_lineage["report_sha256"]
        ),
        "conditioning_scale_report_sha256": scale_report["report_sha256"],
        "conditioning_scale_certified_global_upper_bound": False,
        "semantic_token_bridge_gather_scatter": True,
        "equal_total_row_mass_per_batch": True,
        "all_writer_outputs_dense": True,
        "runtime_sidecar_required": False,
        "quality_gate": False,
        "numeric_diagnostics_used_for_semantic_decision": False,
        "numeric_diagnostics_used_for_auto_veto": False,
        "decoded_behavior_review_is_evidence_for_understanding": True,
        "verdict_required_for_continuation": False,
        "accept_veto_publish_delete_stop_authority": {
            "decoded_behavior_review": False,
            "numeric_metrics": False,
        },
        "transaction_fold_only_guard": guard_body,
        "execution": {
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "target_model_forward_calls": 0,
            "generation_calls": 0,
            "backward_calls": guard_body["backward_calls"],
            "autograd_grad_calls": guard_body["autograd_grad_calls"],
            "optimizer_constructions": guard_body["optimizer_constructions"],
            "optimizer_steps": guard_body["optimizer_steps"],
            "training_steps": 0,
            "pruning_operations": guard_body["pruning_operations"],
            "quantization_operations": guard_body["quantization_operations"],
        },
    }
    result = Lfm2SemanticGlobalResidualInstallation(
        topology=topology,
        boundaries=boundaries,
        basis=basis,
        closure_result=closure_result,
        corrected_final_w2_prior=corrected_final_prior,
        report=_seal(body),
    )
    if not result.verify_authentication():  # pragma: no cover - construction seam
        raise Lfm2SemanticGlobalResidualIntegrationAuthenticationError(
            "Published semantic closure result failed self-authentication."
        )
    return result


def compile_install_lfm2_semantic_source_local_output_residual_closure(
    evidence: Lfm2SemanticGlobalResidualEvidencePackage,
    *,
    live_writer_parameters: Mapping[str, Any],
    deferred_final_writer_parameter: Any,
    policy: Lfm2GlobalResidualClosurePolicy | None = None,
) -> Lfm2SemanticGlobalResidualInstallation:
    """Reclose only O/conv-output/MLP-down writers on local-action defects.

    This is the post-Q/K closure lane: attention Q/K/V and every producer-side
    matrix are absent from the publication transaction and therefore remain
    byte-identical.  The same authenticated whole-stack linearization is solved
    against mapped parent-action minus current child-action boundaries.
    """

    return compile_install_lfm2_semantic_global_residual_closure(
        evidence,
        live_writer_parameters=live_writer_parameters,
        deferred_final_writer_parameter=deferred_final_writer_parameter,
        policy=policy,
        boundary_mode=_SOURCE_LOCAL_OUTPUT_BOUNDARY_MODE,
    )


__all__ = [
    "LFM2_SEMANTIC_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA",
    "LFM2_SEMANTIC_SOURCE_LOCAL_OUTPUT_GLOBAL_RESIDUAL_INTEGRATION_SCHEMA",
    "Lfm2SemanticGlobalResidualEvidencePackage",
    "Lfm2SemanticGlobalResidualInstallation",
    "Lfm2SemanticGlobalResidualIntegrationAuthenticationError",
    "Lfm2SemanticGlobalResidualIntegrationError",
    "Lfm2SemanticGlobalResidualSiteSpec",
    "Lfm2SemanticGlobalResidualTrajectoryEvidence",
    "build_lfm2_semantic_global_residual_basis",
    "build_lfm2_semantic_global_residual_boundaries",
    "build_lfm2_semantic_source_local_global_residual_boundaries",
    "build_lfm2_semantic_global_residual_topology",
    "compile_install_lfm2_semantic_global_residual_closure",
    "compile_install_lfm2_semantic_source_local_output_residual_closure",
    "verify_lfm2_semantic_global_residual_report",
]
