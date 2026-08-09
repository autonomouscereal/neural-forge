"""The differentiable stage-2 student: one carrier-ABI child, one route path.

The frozen review decodes a carrier child through
``Lfm2SemanticStepRoutingRuntime``: donor ids are reconstructed into child
embeddings by ``C_in``, the body advances once per donor action, and the child
readout is projected back to donor logits by ``C_out``
(``docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`` section 2.5).  That
runtime is a sealed *inference* surface: it detaches every tensor, hashes the
live table before each use, and runs under ``no_grad``.  A trainer cannot use
it, and reimplementing "something similar" would mean stage 2 trains one
function and reviews another.

This module therefore rebuilds exactly that arithmetic in a differentiable
form -- same coalesced edge order, same accumulation dtype rule, same chunking
-- and ships the equality as a checkable claim:
:func:`assert_route_surface_matches_frozen_runtime` compares both channels
against the frozen runtime on the same weights and refuses on any difference.
The trainer calls it once per run before the first optimizer step, so "the
training surface equals the decode surface" is a receipt line rather than a
belief.

Initialization is one path with one flag (design section 1: the controls are
"byte-identical config ... including both k32 route tensors as trainable
parameters").  Both arms load the same authenticated published child through
the same loader the review runtime uses; the control then overwrites every
dense parameter and both route coefficient vectors from a recorded seed, so
the two arms cannot differ in topology, route structure, or plumbing -- only in
where their numbers came from.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

from .dynamic_circuit_ir import canonical_sha256
from .lfm2_carrier_bridge_publication import (
    PACKAGED_TARGET_ROUTE_KEY,
    child_table_tensor_sha256,
)
from .lfm2_carrier_child_package import (
    Lfm2CarrierChildCheckpointIO,
    load_authenticated_lfm2_carrier_child_package,
)
from .stage2_kd_objective import Stage2DonorVocabulary


STAGE2_STUDENT_SCHEMA = "neural-forge-stage2-routed-student-v1"
STAGE2_FOLD_INITIALIZATION = "fold-init"
STAGE2_RANDOM_INITIALIZATION = "random-init"
STAGE2_INITIALIZATION_MODES = (
    STAGE2_FOLD_INITIALIZATION,
    STAGE2_RANDOM_INITIALIZATION,
)
# Design section 1: "random init (scaled-normal per LFM2 init scheme, seed
# recorded)".  The scheme name is recorded verbatim in every receipt so a
# replicate can reproduce the control without reading this file.
STAGE2_RANDOM_INITIALIZATION_SCHEME = (
    "scaled-normal-lfm2-initializer-range;"
    "matrix-weights~N(0,range);norm-weights=1;biases=0"
)
STAGE2_DEFAULT_INITIALIZER_RANGE = 0.02
# The runtime reconstructs donor embeddings 1,024 rows at a time
# (``hf_lfm2_semantic_step_model_runtime``); matching the chunk keeps the
# floating-point accumulation order identical between training and decode.
STAGE2_EMBEDDING_ROWS_PER_CHUNK = 1_024


class Stage2CarrierStudentError(RuntimeError):
    """A stage-2 student surface, route layout, or initialization is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage2CarrierStudentError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage2CarrierStudentError(f"{name} must be a nonnegative integer.")
    return value


def _routing_accumulation_dtype(table_dtype: torch.dtype) -> torch.dtype:
    """Mirror the frozen runtime's accumulation rule exactly."""

    if table_dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    if table_dtype in {torch.float32, torch.float64}:
        return table_dtype
    raise Stage2CarrierStudentError(
        "Live child table has no supported routing accumulation dtype."
    )


@dataclass(frozen=True, slots=True)
class Stage2CoalescedRouteLayout:
    """One route channel's integer structure in the runtime's coalesced order.

    ``torch.sparse_coo_tensor(...).coalesce()`` sorts edges row-major and sums
    duplicate ``(donor, target)`` pairs.  The trainer needs that same layout
    every step without paying a sort every step *and* without losing the
    gradient path back to the coefficient parameter, so the permutation and the
    duplicate-merge segments are computed once here and replayed by
    :meth:`coalesced_values` as ordinary differentiable gathers.
    """

    donor_row_count: int
    target_row_count: int
    edge_order: Tensor
    merge_slot: Tensor
    coalesced_donor_ids: Tensor
    coalesced_target_ids: Tensor
    row_offsets: Tensor
    duplicate_free: bool
    raw_edge_count: int

    @property
    def coalesced_edge_count(self) -> int:
        return int(self.coalesced_target_ids.numel())

    def coalesced_values(self, coefficients: Tensor, *, dtype: torch.dtype) -> Tensor:
        """Coefficients reordered (and duplicate-summed) like ``coalesce()``.

        The cast happens *before* the merge so the arithmetic matches the
        runtime, which builds its sparse operator directly in the accumulation
        dtype and lets ``coalesce()`` sum in that dtype.
        """

        if (
            not isinstance(coefficients, Tensor)
            or coefficients.ndim != 1
            or int(coefficients.numel()) != self.raw_edge_count
        ):
            raise Stage2CarrierStudentError(
                "Route coefficients do not match this coalesced layout."
            )
        ordered = coefficients.to(dtype=dtype).index_select(
            0,
            self.edge_order.to(device=coefficients.device),
        )
        if self.duplicate_free:
            return ordered
        return torch.index_add(
            torch.zeros(
                self.coalesced_edge_count,
                device=coefficients.device,
                dtype=dtype,
            ),
            0,
            self.merge_slot.to(device=coefficients.device),
            ordered,
        )

    def sparse_operator(
        self,
        coefficients: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        values = self.coalesced_values(coefficients, dtype=dtype)
        indices = torch.stack(
            (
                self.coalesced_donor_ids.to(device=device),
                self.coalesced_target_ids.to(device=device),
            ),
            dim=0,
        )
        return torch.sparse_coo_tensor(
            indices,
            values.to(device=device),
            size=(self.donor_row_count, self.target_row_count),
            device=device,
            dtype=dtype,
            check_invariants=True,
        ).coalesce()

    def as_dict(self) -> dict[str, Any]:
        return {
            "donor_row_count": self.donor_row_count,
            "target_row_count": self.target_row_count,
            "raw_edge_count": self.raw_edge_count,
            "coalesced_edge_count": self.coalesced_edge_count,
            "duplicate_free": self.duplicate_free,
        }


def build_coalesced_route_layout(
    offsets: Tensor,
    target_ids: Tensor,
    *,
    donor_row_count: int,
    target_row_count: int,
) -> Stage2CoalescedRouteLayout:
    """Derive the runtime's coalesced edge order from one CSR route channel."""

    rows = _positive_int(donor_row_count, "donor_row_count")
    columns = _positive_int(target_row_count, "target_row_count")
    if (
        not isinstance(offsets, Tensor)
        or offsets.dtype is not torch.int64
        or offsets.ndim != 1
        or int(offsets.numel()) != rows + 1
        or not isinstance(target_ids, Tensor)
        or target_ids.dtype is not torch.int64
        or target_ids.ndim != 1
        or int(offsets[0].item()) != 0
        or int(offsets[-1].item()) != int(target_ids.numel())
        or bool((offsets[1:] < offsets[:-1]).any())
        or bool((target_ids < 0).any())
        or bool((target_ids >= columns).any())
    ):
        raise Stage2CarrierStudentError(
            "Route CSR offsets and target ids are not a valid channel."
        )
    lengths = offsets[1:] - offsets[:-1]
    donor_ids = torch.repeat_interleave(
        torch.arange(rows, dtype=torch.int64),
        lengths,
    )
    keys = donor_ids * columns + target_ids
    edge_order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, edge_order)
    unique_keys, merge_slot = torch.unique_consecutive(
        sorted_keys,
        return_inverse=True,
    )
    coalesced_donor_ids = torch.div(unique_keys, columns, rounding_mode="floor")
    coalesced_target_ids = unique_keys - coalesced_donor_ids * columns
    row_counts = torch.bincount(coalesced_donor_ids, minlength=rows)
    row_offsets = torch.cat((row_counts.new_zeros(1), row_counts.cumsum(dim=0)))
    return Stage2CoalescedRouteLayout(
        donor_row_count=rows,
        target_row_count=columns,
        edge_order=edge_order,
        merge_slot=merge_slot,
        coalesced_donor_ids=coalesced_donor_ids,
        coalesced_target_ids=coalesced_target_ids,
        row_offsets=row_offsets,
        duplicate_free=int(unique_keys.numel()) == int(keys.numel()),
        raw_edge_count=int(keys.numel()),
    )


def protected_anchor_edge_positions(
    offsets: Tensor,
    target_ids: Tensor,
    *,
    direct_target_row_to_semantic_id: Tensor,
    donor_row_count: int,
    channel: str,
) -> Tensor:
    """Locate the coefficients the routed runtime requires to stay 1.0.

    ``_validated_sparse_semantic_code`` refuses to bind a carrier runtime
    unless every target row's protected donor row is an exact one-hot at
    coefficient ``1.0`` -- these are the strict target bridge's 4,096 injective
    exact anchors (``docs/CURRENT_FOLDING_GOAL.md``).  They are structure, not
    knowledge, so stage 2 holds them fixed and trains the rest.  Training them
    would produce a checkpoint that cannot decode at all, which no admission
    list could repair.
    """

    anchors = direct_target_row_to_semantic_id.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    row_offsets = offsets.detach().to(device="cpu", dtype=torch.int64)
    rows = target_ids.detach().to(device="cpu", dtype=torch.int64)
    if anchors.ndim != 1 or int(torch.unique(anchors).numel()) != int(
        anchors.numel()
    ):
        raise Stage2CarrierStudentError(
            "target_to_donor must injectively protect every target anchor."
        )
    positions: list[int] = []
    for target_row, donor_id in enumerate(int(value) for value in anchors):
        if donor_id < 0 or donor_id >= _positive_int(
            donor_row_count,
            "donor_row_count",
        ):
            raise Stage2CarrierStudentError(
                "A protected anchor names a donor row outside the routing."
            )
        start = int(row_offsets[donor_id].item())
        stop = int(row_offsets[donor_id + 1].item())
        if stop - start != 1 or int(rows[start].item()) != target_row:
            raise Stage2CarrierStudentError(
                f"{channel} route anchor for target row {target_row} is not an "
                "exact one-hot row; this child cannot bind the routed decode."
            )
        positions.append(start)
    return torch.tensor(sorted(positions), dtype=torch.int64)


@dataclass(frozen=True, slots=True)
class Stage2StudentInitialization:
    """Where one arm's numbers came from -- the arm identity in the receipt."""

    mode: str
    child_directory: str
    child_package_sha256: str
    config_sha256: str
    source_bridge_artifact_sha256: str
    train_routes: bool
    seed: int | None = None
    initializer_range: float | None = None
    scheme: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in STAGE2_INITIALIZATION_MODES:
            raise Stage2CarrierStudentError(
                f"Initialization mode must be one of {STAGE2_INITIALIZATION_MODES}."
            )
        if self.mode == STAGE2_RANDOM_INITIALIZATION and (
            self.seed is None or self.scheme is None
        ):
            raise Stage2CarrierStudentError(
                "A random-init arm must record its seed and scheme."
            )

    @property
    def init_source_sha256(self) -> str:
        """One digest naming this arm's initialization, fold or control."""

        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE2_STUDENT_SCHEMA,
            "mode": self.mode,
            "child_directory": self.child_directory,
            "child_package_sha256": self.child_package_sha256,
            "config_sha256": self.config_sha256,
            "source_bridge_artifact_sha256": (
                self.source_bridge_artifact_sha256
            ),
            "route_coefficients_are_trainable": bool(self.train_routes),
            "seed": self.seed,
            "initializer_range": self.initializer_range,
            "scheme": self.scheme,
            "fold_weights_used": self.mode == STAGE2_FOLD_INITIALIZATION,
        }


@dataclass(slots=True)
class Stage2CachedDecodeState:
    """The child body's own cache plus the context length it covers.

    Mutable on purpose: HF cache objects are advanced in place by some layer
    types and returned fresh by others, so the state holds whatever the body
    handed back last and the length it is now valid for.  The length is what
    the attention mask is built from, exactly as the deployed decode does.
    """

    cache: Any
    length: int


class Stage2RoutedStudent(nn.Module):
    """A carrier-ABI child plus its two route channels, end to end trainable."""

    def __init__(
        self,
        child: Any,
        *,
        input_offsets: Tensor,
        input_target_ids: Tensor,
        input_coefficients: Tensor,
        output_offsets: Tensor,
        output_target_ids: Tensor,
        output_coefficients: Tensor,
        direct_target_row_to_semantic_id: Tensor,
        donor_row_count: int,
        target_row_count: int,
        hidden_size: int,
        train_routes: bool = True,
        initialization: Stage2StudentInitialization | None = None,
    ) -> None:
        super().__init__()
        self.child = child
        self.donor_row_count = _positive_int(donor_row_count, "donor_row_count")
        self.target_row_count = _positive_int(target_row_count, "target_row_count")
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.initialization = initialization
        self.train_routes = bool(train_routes)
        for name, coefficients in (
            ("input_coefficients", input_coefficients),
            ("output_coefficients", output_coefficients),
        ):
            if (
                not isinstance(coefficients, Tensor)
                or coefficients.dtype is not torch.float32
                or coefficients.ndim != 1
                or not bool(torch.isfinite(coefficients).all())
            ):
                raise Stage2CarrierStudentError(
                    f"{name} must be one finite FP32 vector."
                )
        self._input_layout = build_coalesced_route_layout(
            input_offsets,
            input_target_ids,
            donor_row_count=self.donor_row_count,
            target_row_count=self.target_row_count,
        )
        self._output_layout = build_coalesced_route_layout(
            output_offsets,
            output_target_ids,
            donor_row_count=self.donor_row_count,
            target_row_count=self.target_row_count,
        )
        self.input_coefficients = nn.Parameter(
            input_coefficients.detach().clone(),
            requires_grad=self.train_routes,
        )
        self.output_coefficients = nn.Parameter(
            output_coefficients.detach().clone(),
            requires_grad=self.train_routes,
        )
        self.register_buffer(
            "input_protected_edges",
            protected_anchor_edge_positions(
                input_offsets,
                input_target_ids,
                direct_target_row_to_semantic_id=direct_target_row_to_semantic_id,
                donor_row_count=self.donor_row_count,
                channel="input",
            ),
            persistent=False,
        )
        self.register_buffer(
            "output_protected_edges",
            protected_anchor_edge_positions(
                output_offsets,
                output_target_ids,
                direct_target_row_to_semantic_id=direct_target_row_to_semantic_id,
                donor_row_count=self.donor_row_count,
                channel="output",
            ),
            persistent=False,
        )
        self.restore_protected_anchor_coefficients()
        if self.train_routes:
            # Two independent guards, because one is not enough: the hook stops
            # anchor gradients from entering Adam's moments, and the post-step
            # restore stops AdamW's *decoupled* weight decay -- which ignores
            # the gradient entirely -- from walking an anchor off 1.0.
            self.input_coefficients.register_hook(
                lambda grad: grad.index_fill(0, self.input_protected_edges, 0.0)
            )
            self.output_coefficients.register_hook(
                lambda grad: grad.index_fill(0, self.output_protected_edges, 0.0)
            )
        tables = self.child_tables()
        if tuple(tables[0].shape) != (self.target_row_count, self.hidden_size):
            raise Stage2CarrierStudentError(
                "Child tables do not match the declared carrier topology."
            )

    # -- child access ----------------------------------------------------

    def child_tables(self) -> tuple[Tensor, Tensor]:
        """The live untied ``C_in``/``C_out`` tables, by reference."""

        get_input = getattr(self.child, "get_input_embeddings", None)
        get_output = getattr(self.child, "get_output_embeddings", None)
        if not callable(get_input) or not callable(get_output):
            raise Stage2CarrierStudentError(
                "Carrier child must expose input and readout table accessors."
            )
        input_table = getattr(get_input(), "weight", None)
        output_table = getattr(get_output(), "weight", None)
        if (
            not isinstance(input_table, Tensor)
            or not isinstance(output_table, Tensor)
            or input_table.ndim != 2
            or output_table.ndim != 2
            or tuple(input_table.shape) != tuple(output_table.shape)
            or input_table is output_table
            or input_table.untyped_storage().data_ptr()
            == output_table.untyped_storage().data_ptr()
        ):
            raise Stage2CarrierStudentError(
                "Carrier child tables must be untied with independent storage."
            )
        return input_table, output_table

    def child_table_sha256(self) -> dict[str, str]:
        input_table, output_table = self.child_tables()
        return {
            "model.embed_tokens.weight": child_table_tensor_sha256(input_table),
            "lm_head.weight": child_table_tensor_sha256(output_table),
        }

    def restore_protected_anchor_coefficients(self) -> None:
        """Pin every protected anchor coefficient back to exactly 1.0."""

        with torch.no_grad():
            self.input_coefficients.index_fill_(
                0,
                self.input_protected_edges.to(self.input_coefficients.device),
                1.0,
            )
            self.output_coefficients.index_fill_(
                0,
                self.output_protected_edges.to(self.output_coefficients.device),
                1.0,
            )

    def protected_anchors_are_intact(self) -> bool:
        with torch.no_grad():
            return bool(
                torch.all(
                    self.input_coefficients.index_select(
                        0,
                        self.input_protected_edges.to(
                            self.input_coefficients.device
                        ),
                    )
                    == 1.0
                )
                and torch.all(
                    self.output_coefficients.index_select(
                        0,
                        self.output_protected_edges.to(
                            self.output_coefficients.device
                        ),
                    )
                    == 1.0
                )
            )

    def route_training_mask(self) -> dict[str, Any]:
        protected = int(
            self.input_protected_edges.numel()
            + self.output_protected_edges.numel()
        )
        total = int(
            self.input_coefficients.numel() + self.output_coefficients.numel()
        )
        return {
            "route_coefficients_are_trainable": self.train_routes,
            "route_coefficient_elements": total,
            "protected_anchor_coefficient_elements": protected,
            "trainable_route_coefficient_elements": total - protected,
            "protected_anchors_held_at_exactly_one": True,
            "protected_anchors_intact": self.protected_anchors_are_intact(),
        }

    def counted_parameter_elements(self) -> dict[str, int]:
        """Dense body + counted route coefficients, the honest counted split."""

        route_elements = int(
            self.input_coefficients.numel() + self.output_coefficients.numel()
        )
        dense_elements = sum(
            int(parameter.numel()) for parameter in self.child.parameters()
        )
        return {
            "dense_model_parameter_elements": dense_elements,
            "continuous_route_parameter_elements": route_elements,
            "counted_total_parameter_elements": dense_elements + route_elements,
        }

    # -- route channels ---------------------------------------------------

    def reconstruct_donor_embeddings(
        self,
        donor_ids: Tensor,
        *,
        maximum_rows_per_chunk: int = STAGE2_EMBEDDING_ROWS_PER_CHUNK,
    ) -> Tensor:
        """``C_in`` reconstruction, differentiable, runtime-order identical."""

        if (
            not isinstance(donor_ids, Tensor)
            or donor_ids.dtype not in {torch.int32, torch.int64}
            or donor_ids.numel() == 0
        ):
            raise Stage2CarrierStudentError(
                "donor_ids must be a nonempty integer tensor."
            )
        flat = donor_ids.reshape(-1).to(torch.int64)
        if bool((flat < 0).any()) or bool((flat >= self.donor_row_count).any()):
            raise Stage2CarrierStudentError(
                "Selected donor ID leaves semantic-step routing."
            )
        chunk_size = _positive_int(maximum_rows_per_chunk, "maximum_rows_per_chunk")
        input_table, _output_table = self.child_tables()
        accumulation_dtype = _routing_accumulation_dtype(input_table.dtype)
        device = input_table.device
        # Distinct donor rows are reconstructed once.  Each row's accumulation
        # only ever touches its own edges, so deduplication cannot change a
        # single reconstructed value -- it only removes repeated work.
        unique_ids, inverse = torch.unique(flat.to(device), return_inverse=True)
        values = self._input_layout.coalesced_values(
            self.input_coefficients,
            dtype=accumulation_dtype,
        ).to(device)
        row_offsets = self._input_layout.row_offsets.to(device)
        edge_target_ids = self._input_layout.coalesced_target_ids.to(device)
        outputs: list[Tensor] = []
        for start in range(0, int(unique_ids.numel()), chunk_size):
            selected = unique_ids[start : start + chunk_size]
            edge_starts = row_offsets.index_select(0, selected)
            edge_stops = row_offsets.index_select(0, selected + 1)
            edge_lengths = edge_stops - edge_starts
            packed_starts = torch.cat(
                (edge_lengths.new_zeros(1), edge_lengths.cumsum(dim=0)[:-1])
            )
            local_rows = torch.repeat_interleave(
                torch.arange(
                    int(selected.numel()),
                    device=device,
                    dtype=torch.int64,
                ),
                edge_lengths,
            )
            edge_positions = torch.arange(
                int(edge_lengths.sum().item()),
                device=device,
                dtype=torch.int64,
            ) + torch.repeat_interleave(edge_starts - packed_starts, edge_lengths)
            target_ids = edge_target_ids.index_select(0, edge_positions)
            weights = values.index_select(0, edge_positions)
            contribution = (
                input_table.index_select(0, target_ids).to(dtype=accumulation_dtype)
                * weights.unsqueeze(1)
            )
            chunk = torch.index_add(
                torch.zeros(
                    (int(selected.numel()), self.hidden_size),
                    device=device,
                    dtype=accumulation_dtype,
                ),
                0,
                local_rows,
                contribution,
            )
            outputs.append(chunk.to(dtype=input_table.dtype))
        reconstructed = torch.cat(outputs, dim=0).index_select(0, inverse)
        return reconstructed.reshape(*donor_ids.shape, self.hidden_size)

    def score_target_tokens(self, hidden_states: Tensor) -> Tensor:
        _input_table, output_table = self.child_tables()
        if (
            not isinstance(hidden_states, Tensor)
            or hidden_states.ndim < 1
            or int(hidden_states.shape[-1]) != self.hidden_size
        ):
            raise Stage2CarrierStudentError(
                "hidden_states must end in the semantic hidden size."
            )
        return hidden_states.to(
            device=output_table.device,
            dtype=output_table.dtype,
        ) @ output_table.transpose(0, 1)

    def donor_logits_from_target_scores(self, target_scores: Tensor) -> Tensor:
        """``C_out`` readout, differentiable, runtime-order identical."""

        _input_table, output_table = self.child_tables()
        accumulation_dtype = _routing_accumulation_dtype(output_table.dtype)
        if (
            not isinstance(target_scores, Tensor)
            or target_scores.ndim < 1
            or int(target_scores.shape[-1]) != self.target_row_count
        ):
            raise Stage2CarrierStudentError(
                "target_scores must end in target_row_count floats."
            )
        operator = self._output_layout.sparse_operator(
            self.output_coefficients,
            dtype=accumulation_dtype,
            device=target_scores.device,
        )
        leading = tuple(int(size) for size in target_scores.shape[:-1])
        flat = target_scores.reshape(-1, self.target_row_count).to(
            dtype=accumulation_dtype
        )
        donor_logits = torch.sparse.mm(operator, flat.transpose(0, 1)).transpose(0, 1)
        return donor_logits.reshape((*leading, self.donor_row_count))

    # -- full forward ------------------------------------------------------

    def _body(self) -> Any:
        body = getattr(self.child, "model", None)
        if not callable(body):
            raise Stage2CarrierStudentError(
                "Semantic-step training requires a callable child body at model."
            )
        return body

    def _body_hidden(self, output: Any, *, expected_shape: tuple[int, int]) -> Tensor:
        hidden = output if isinstance(output, Tensor) else None
        if hidden is None:
            hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if (
            not isinstance(hidden, Tensor)
            or hidden.ndim != 3
            or tuple(int(size) for size in hidden.shape[:2]) != expected_shape
            or int(hidden.shape[2]) != self.hidden_size
        ):
            raise Stage2CarrierStudentError(
                "Child body returned invalid semantic hidden states."
            )
        return hidden

    def body_hidden_states(self, inputs_embeds: Tensor) -> Tensor:
        """Advance the child body once per donor action, as decode does."""

        attention_mask = torch.ones(
            tuple(inputs_embeds.shape[:2]),
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        output = self._body()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self._body_hidden(
            output,
            expected_shape=(
                int(inputs_embeds.shape[0]),
                int(inputs_embeds.shape[1]),
            ),
        )

    def body_hidden_states_with_cache(
        self,
        inputs_embeds: Tensor,
        *,
        past_key_values: Any = None,
        past_length: int = 0,
    ) -> tuple[Tensor, Any]:
        """One cached body advance, called exactly as the routed decode calls it.

        ``hf_lfm2_semantic_step_model_runtime`` prefills the prompt with
        ``use_cache=True`` and then advances one semantic token at a time with
        ``past_key_values=cache`` and an attention mask covering the *whole*
        context so far.  This reproduces that call shape verbatim rather than
        inventing a second decode contract, so the sampler and the deployed
        decode exercise the same code path in the child's hybrid conv/attention
        stack.
        """

        total_length = int(past_length) + int(inputs_embeds.shape[1])
        attention_mask = torch.ones(
            (int(inputs_embeds.shape[0]), total_length),
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        output = self._body()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden = self._body_hidden(
            output,
            expected_shape=(
                int(inputs_embeds.shape[0]),
                int(inputs_embeds.shape[1]),
            ),
        )
        cache = getattr(output, "past_key_values", None)
        if cache is None and isinstance(output, (tuple, list)) and len(output) > 1:
            cache = output[1]
        if cache is None:
            raise Stage2CarrierStudentError(
                "Child body returned no cache while use_cache=True; the "
                "uncached sampling path is the flag for such a body."
            )
        return hidden, cache

    def donor_logits_from_hidden(self, hidden_states: Tensor) -> Tensor:
        """The routed readout on already-computed hidden states."""

        return self.donor_logits_from_target_scores(
            self.score_target_tokens(hidden_states)
        )

    def forward(self, donor_ids: Tensor) -> Tensor:
        """``[batch, length]`` donor ids -> ``[batch, length, donor_rows]``."""

        if (
            not isinstance(donor_ids, Tensor)
            or donor_ids.ndim != 2
            or donor_ids.dtype not in {torch.int32, torch.int64}
        ):
            raise Stage2CarrierStudentError(
                "Student forward takes one [batch, length] donor id tensor."
            )
        embeddings = self.reconstruct_donor_embeddings(donor_ids)
        hidden = self.body_hidden_states(embeddings)
        return self.donor_logits_from_target_scores(
            self.score_target_tokens(hidden)
        )

    # -- cached incremental decode ------------------------------------------

    def start_cached_decode(
        self,
        donor_ids: Tensor,
    ) -> tuple[Tensor, "Stage2CachedDecodeState"]:
        """Prefill a context and return its last position's donor logits.

        The returned logits are ``[batch, donor_rows]`` for the *next* token,
        which is all a sampler needs; the intermediate positions are never
        scored, and skipping their readout is a second saving on top of the
        cache itself (the routed ``C_out`` readout is a sparse mm over the full
        donor vocabulary at every position).
        """

        if (
            not isinstance(donor_ids, Tensor)
            or donor_ids.ndim != 2
            or donor_ids.dtype not in {torch.int32, torch.int64}
            or int(donor_ids.shape[1]) == 0
        ):
            raise Stage2CarrierStudentError(
                "Cached decode starts from one non-empty [batch, length] "
                "donor id tensor."
            )
        embeddings = self.reconstruct_donor_embeddings(donor_ids)
        hidden, cache = self.body_hidden_states_with_cache(embeddings)
        state = Stage2CachedDecodeState(
            cache=cache,
            length=int(donor_ids.shape[1]),
        )
        return self.donor_logits_from_hidden(hidden[:, -1, :]), state

    def advance_cached_decode(
        self,
        state: "Stage2CachedDecodeState",
        donor_ids: Tensor,
    ) -> Tensor:
        """Feed one already-chosen donor token; return the next donor logits."""

        if not isinstance(state, Stage2CachedDecodeState):
            raise Stage2CarrierStudentError(
                "advance_cached_decode needs the state start_cached_decode "
                "returned."
            )
        if (
            not isinstance(donor_ids, Tensor)
            or donor_ids.ndim != 2
            or int(donor_ids.shape[1]) != 1
            or donor_ids.dtype not in {torch.int32, torch.int64}
        ):
            raise Stage2CarrierStudentError(
                "Cached decode advances exactly one donor token per call."
            )
        embeddings = self.reconstruct_donor_embeddings(donor_ids)
        hidden, cache = self.body_hidden_states_with_cache(
            embeddings,
            past_key_values=state.cache,
            past_length=state.length,
        )
        state.cache = cache
        state.length += 1
        return self.donor_logits_from_hidden(hidden[:, -1, :])

    def route_layouts(self) -> dict[str, Any]:
        return {
            "input": self._input_layout.as_dict(),
            "output": self._output_layout.as_dict(),
        }


def assert_route_surface_matches_frozen_runtime(
    student: Stage2RoutedStudent,
    runtime: Any,
    *,
    donor_ids: Sequence[int],
    target_scores: Tensor,
) -> dict[str, Any]:
    """Refuse unless the trainable surface equals the frozen decode surface.

    The frozen runtime authenticates the live child table by digest, so this
    check is only meaningful while the student still carries the sealed
    weights -- which is exactly when the trainer runs it: once, before the
    first optimizer step.
    """

    if not isinstance(student, Stage2RoutedStudent):
        raise Stage2CarrierStudentError("student must be a Stage2RoutedStudent.")
    input_table, output_table = student.child_tables()
    ids = [int(value) for value in donor_ids]
    if not ids:
        raise Stage2CarrierStudentError(
            "Route surface equivalence needs at least one donor id."
        )
    with torch.no_grad():
        # The dual-table session pair is exactly how the decode runtime drives
        # these two calls (``hf_lfm2_semantic_step_model_runtime``); the
        # single-tensor convenience path authenticates under the shared role
        # and cannot serve an untied readout at all.
        with runtime.authenticated_live_table_sessions(
            input_table,
            output_table,
        ) as (input_session, output_session):
            expected_embeddings = runtime.reconstruct_selected_donor_embeddings(
                input_session,
                ids,
                maximum_rows_per_chunk=STAGE2_EMBEDDING_ROWS_PER_CHUNK,
            )
            expected_logits = (
                runtime.compute_all_donor_logits_from_target_scores(
                    output_session,
                    target_scores,
                )
            )
        observed_embeddings = student.reconstruct_donor_embeddings(
            torch.tensor(ids, dtype=torch.int64, device=input_table.device)
        )
        observed_logits = student.donor_logits_from_target_scores(target_scores)
    embedding_equal = bool(torch.equal(expected_embeddings, observed_embeddings))
    logit_equal = bool(torch.equal(expected_logits, observed_logits))
    if not embedding_equal or not logit_equal:
        raise Stage2CarrierStudentError(
            "Trainable route surface differs from the frozen decode surface "
            f"(embeddings equal: {embedding_equal}; logits equal: {logit_equal})."
        )
    return {
        "schema": STAGE2_STUDENT_SCHEMA,
        "input_route_bitwise_identical_to_frozen_runtime": True,
        "output_route_bitwise_identical_to_frozen_runtime": True,
        "checked_donor_row_count": len(ids),
        "checked_target_score_positions": int(
            target_scores.reshape(-1, student.target_row_count).shape[0]
        ),
    }


def _config_sha256(child_directory: Path) -> str:
    config = child_directory / "config.json"
    if not config.is_file() or config.is_symlink():
        raise Stage2CarrierStudentError(
            "Published child directory has no ordinary config.json."
        )
    return hashlib.sha256(config.read_bytes()).hexdigest()


def seeded_scaled_normal_reinitialization(
    child: Any,
    *,
    seed: int,
    initializer_range: float = STAGE2_DEFAULT_INITIALIZER_RANGE,
) -> dict[str, Any]:
    """Overwrite every dense child parameter from one recorded seed.

    Values are drawn on CPU from an explicit generator and copied in, so the
    control is reproducible from ``(seed, scheme, initializer_range)`` alone and
    does not depend on device RNG behaviour.
    """

    _nonnegative_int(seed, "seed")
    if (
        isinstance(initializer_range, bool)
        or not isinstance(initializer_range, (int, float))
        or float(initializer_range) <= 0.0
    ):
        raise Stage2CarrierStudentError(
            "initializer_range must be a positive number."
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    scale = float(initializer_range)
    reinitialized: list[str] = []
    with torch.no_grad():
        for name, parameter in sorted(
            child.named_parameters(),
            key=lambda item: item[0],
        ):
            if parameter.ndim >= 2:
                values = torch.empty(
                    tuple(parameter.shape),
                    dtype=torch.float32,
                ).normal_(mean=0.0, std=scale, generator=generator)
            elif name.endswith("bias"):
                values = torch.zeros(tuple(parameter.shape), dtype=torch.float32)
            else:
                values = torch.ones(tuple(parameter.shape), dtype=torch.float32)
            parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))
            reinitialized.append(name)
    return {
        "scheme": STAGE2_RANDOM_INITIALIZATION_SCHEME,
        "seed": int(seed),
        "initializer_range": scale,
        "reinitialized_parameter_names": tuple(reinitialized),
    }


def _seeded_route_coefficients(
    reference: Tensor,
    *,
    seed: int,
    initializer_range: float,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.empty(
        tuple(reference.shape),
        dtype=torch.float32,
    ).normal_(mean=0.0, std=float(initializer_range), generator=generator)


def load_stage2_routed_student(
    child_directory: Path,
    *,
    base: Any,
    mode: str = STAGE2_FOLD_INITIALIZATION,
    donor_vocabulary: Stage2DonorVocabulary | None = None,
    train_routes: bool = True,
    seed: int | None = None,
    initializer_range: float = STAGE2_DEFAULT_INITIALIZER_RANGE,
    target_tokenizer_loader: Callable[[Path], Any] | None = None,
) -> tuple[Stage2RoutedStudent, Any, Any]:
    """Load one arm through the loader the frozen review itself uses.

    Returns ``(student, decode_model, packaged_routes)``.  ``decode_model`` is
    the authenticated carrier child with its route runtime installed -- the same
    object the review would decode -- and is handed back so a caller can run the
    surface-equivalence check before training touches anything.
    """

    if mode not in STAGE2_INITIALIZATION_MODES:
        raise Stage2CarrierStudentError(
            f"Initialization mode must be one of {STAGE2_INITIALIZATION_MODES}."
        )
    if mode == STAGE2_RANDOM_INITIALIZATION and seed is None:
        raise Stage2CarrierStudentError(
            "A random-init control must be given an explicit recorded seed."
        )
    root = Path(child_directory).expanduser().absolute()
    loader_arguments: dict[str, Any] = {
        "base": base,
        "child_directory": root,
    }
    if target_tokenizer_loader is not None:
        loader_arguments["target_tokenizer_loader"] = target_tokenizer_loader
    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(**loader_arguments)
    decode_model = loader.load(root)
    package = load_authenticated_lfm2_carrier_child_package(
        root,
        child_model=decode_model,
        expected_package_sha256=loader.expected_child_package_sha256,
    )
    tensors = package.routes.tensors
    target_rows = int(package.receipt["target_rows"])
    hidden_size = int(package.receipt["hidden_size"])
    donor_rows = int(tensors["input_offsets"].numel()) - 1
    if donor_vocabulary is not None and (
        donor_vocabulary.defined_row_count != donor_rows
    ):
        raise Stage2CarrierStudentError(
            "Child routes score a different donor row count than the declared "
            f"vocabulary ({donor_rows} vs "
            f"{donor_vocabulary.defined_row_count})."
        )
    if int(tensors[PACKAGED_TARGET_ROUTE_KEY].numel()) != target_rows:
        raise Stage2CarrierStudentError(
            "Packaged target route does not cover the sealed target rows."
        )
    input_coefficients = tensors["input_coefficients"].detach().clone()
    output_coefficients = tensors["output_coefficients"].detach().clone()
    initialization_scheme: str | None = None
    used_range: float | None = None
    if mode == STAGE2_RANDOM_INITIALIZATION:
        record = seeded_scaled_normal_reinitialization(
            decode_model,
            seed=int(seed),
            initializer_range=initializer_range,
        )
        initialization_scheme = str(record["scheme"])
        used_range = float(record["initializer_range"])
        # Design section 5.4 item 1: the control carries both k32 route tensors
        # as ordinary randomly initialized trainable parameters.  Only the
        # coefficients are redrawn; the integer routing (item 2) is shared
        # verbatim, which is precisely the fold's plumbing without its
        # knowledge.
        input_coefficients = _seeded_route_coefficients(
            input_coefficients,
            seed=int(seed) + 1,
            initializer_range=initializer_range,
        )
        output_coefficients = _seeded_route_coefficients(
            output_coefficients,
            seed=int(seed) + 2,
            initializer_range=initializer_range,
        )
    initialization = Stage2StudentInitialization(
        mode=mode,
        child_directory=str(root),
        child_package_sha256=package.package_sha256,
        config_sha256=_config_sha256(root),
        source_bridge_artifact_sha256=str(
            package.receipt["source_bridge_artifact_sha256"]
        ),
        train_routes=bool(train_routes),
        seed=None if seed is None else int(seed),
        initializer_range=used_range,
        scheme=initialization_scheme,
    )
    student = Stage2RoutedStudent(
        decode_model,
        input_offsets=tensors["input_offsets"],
        input_target_ids=tensors["input_target_ids"],
        input_coefficients=input_coefficients,
        output_offsets=tensors["output_offsets"],
        output_target_ids=tensors["output_target_ids"],
        output_coefficients=output_coefficients,
        direct_target_row_to_semantic_id=tensors[PACKAGED_TARGET_ROUTE_KEY],
        donor_row_count=donor_rows,
        target_row_count=target_rows,
        hidden_size=hidden_size,
        train_routes=train_routes,
        initialization=initialization,
    )
    return student, decode_model, package


__all__ = [
    "STAGE2_DEFAULT_INITIALIZER_RANGE",
    "STAGE2_EMBEDDING_ROWS_PER_CHUNK",
    "STAGE2_FOLD_INITIALIZATION",
    "STAGE2_INITIALIZATION_MODES",
    "STAGE2_RANDOM_INITIALIZATION",
    "STAGE2_RANDOM_INITIALIZATION_SCHEME",
    "STAGE2_STUDENT_SCHEMA",
    "Stage2CarrierStudentError",
    "Stage2CoalescedRouteLayout",
    "Stage2RoutedStudent",
    "Stage2StudentInitialization",
    "assert_route_surface_matches_frozen_runtime",
    "build_coalesced_route_layout",
    "load_stage2_routed_student",
    "protected_anchor_edge_positions",
    "seeded_scaled_normal_reinitialization",
]
