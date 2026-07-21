"""Target-native CPCF evidence alignment across tokenizer boundaries.

The semantic-span phase bridge proves which target token sequence canonically
encodes each cached parent prefix.  It does *not* make a parent activation row
exist at every target token boundary.  This module supplies that missing,
deliberately strict boundary:

* target rows are matched to parent rows only when the decoded causal prefixes
  are exactly equal;
* residual/MLP rows and complete histories are gathered only through those
  proven endpoint correspondences;
* suffix rewrites are replayed as complete target-native prefixes and are
  explicitly forbidden from masquerading as append-only KV-cache updates;
* cached post-RoPE attention is accepted unchanged only for an isomorphic
  causal lattice; and
* a pure token-span expansion can preserve attention exactly by duplicating
  K/V and partitioning each donor key's probability mass over its causally
  visible target subtokens; and
* contraction fails closed as an exact attention transport because a fixed
  merged K/V row cannot generally preserve query-dependent softmax mixtures.

There are no model calls, gradients, optimizers, parameter mutations, pruning,
or quantization here.  The attention projection is tensor algebra over sealed
cached evidence and caller-supplied target RoPE terms.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_semantic_span_phase_bridge import (
    Lfm2SemanticSpanPhase,
    Lfm2SemanticSpanPhaseBridge,
    _decode_exact,
)


SEMANTIC_SPAN_CPCF_ALIGNMENT_SCHEMA = (
    "neural-forge-lfm2-semantic-span-cpcf-alignment-v1"
)
SEMANTIC_SPAN_CPCF_PHASE_ALIGNMENT_SCHEMA = (
    "neural-forge-lfm2-semantic-span-cpcf-phase-alignment-v1"
)
SEMANTIC_SPAN_CPCF_REFINEMENT_AUDIT_SCHEMA = (
    "neural-forge-lfm2-semantic-span-cpcf-refinement-audit-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "model_forward_calls": 0,
        "model_parameters_read": 0,
        "model_parameters_written": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
    }
)


class Lfm2SemanticSpanCpcfAlignmentError(ValueError):
    """Semantic target evidence is absent, ambiguous, or noncausal."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic CPCF alignment requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2SemanticSpanCpcfAlignmentError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SemanticSpanCpcfAlignmentError(f"{name} must be positive.")
    return value


def _axis(value: int, dimensions: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Lfm2SemanticSpanCpcfAlignmentError(f"{name} must be an integer.")
    result = value + dimensions if value < 0 else value
    if result < 0 or result >= dimensions:
        raise Lfm2SemanticSpanCpcfAlignmentError(
            f"{name} is outside a rank-{dimensions} tensor."
        )
    return result


def _finite_tensor(value: Any, name: str, *, dimensions: int | None = None) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.numel() == 0
        or (dimensions is not None and value.ndim != dimensions)
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        rank = "" if dimensions is None else f" rank-{dimensions}"
        raise Lfm2SemanticSpanCpcfAlignmentError(
            f"{name} must be a detached finite dense{rank} floating tensor."
        )
    return value.detach().to(device="cpu").contiguous()


def _donor_prefix_ids(
    bridge: Lfm2SemanticSpanPhaseBridge,
    phase: Lfm2SemanticSpanPhase,
) -> tuple[int, ...]:
    if phase.phase == "prefill":
        return bridge.source_prompt_token_ids
    if phase.parent_decode_step is None:  # pragma: no cover - phase validates this
        raise Lfm2SemanticSpanCpcfAlignmentError("Decode phase lost its parent step.")
    return (
        *bridge.source_prompt_token_ids,
        *bridge.source_forced_decode_token_ids[: phase.parent_decode_step + 1],
    )


def _prefix_semantics(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    full_text: str,
    name: str,
) -> tuple[str | None, ...]:
    """Decode every causal boundary, retaining only literal text prefixes."""

    result = []
    for stop in range(1, len(token_ids) + 1):
        value = _decode_exact(tokenizer, token_ids[:stop], name)
        result.append(value if full_text.startswith(value) else None)
    if not result or result[-1] != full_text:
        raise Lfm2SemanticSpanCpcfAlignmentError(
            f"{name} tokenizer does not expose the complete phase text at its "
            "terminal causal boundary."
        )
    return tuple(result)


def _target_to_donor_boundaries(
    *,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    donor_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
) -> tuple[tuple[int | None, ...], tuple[int | None, ...]]:
    full_text = _decode_exact(donor_tokenizer, donor_ids, "donor")
    if _decode_exact(target_tokenizer, target_ids, "target") != full_text:
        raise Lfm2SemanticSpanCpcfAlignmentError(
            "Semantic bridge target IDs no longer decode to the donor phase text."
        )
    donor_prefixes = _prefix_semantics(
        donor_tokenizer,
        donor_ids,
        full_text=full_text,
        name="donor",
    )
    target_prefixes = _prefix_semantics(
        target_tokenizer,
        target_ids,
        full_text=full_text,
        name="target",
    )
    donor_by_text: dict[str, list[int]] = {}
    for index, value in enumerate(donor_prefixes):
        if value is not None:
            donor_by_text.setdefault(value, []).append(index)

    mapped: list[int | None] = []
    owners: list[int | None] = []
    prior = -1
    prior_owner = -1
    for value in target_prefixes:
        candidates = () if value is None else tuple(donor_by_text.get(value, ()))
        # Empty/zero-width special tokens can make a semantic boundary
        # ambiguous.  Guessing one would silently break causal order.
        if len(candidates) != 1 or candidates[0] <= prior:
            mapped.append(None)
        else:
            prior = candidates[0]
            mapped.append(prior)

        owner_candidates = (
            ()
            if value in {None, ""}
            else tuple(
                index
                for index, donor_prefix in enumerate(donor_prefixes)
                if index >= prior_owner
                and donor_prefix is not None
                and donor_prefix.startswith(value)
            )
        )
        if mapped[-1] is not None:
            owner = mapped[-1]
        elif owner_candidates:
            owner = owner_candidates[0]
        else:
            owner = None
        if owner is not None and owner < prior_owner:
            owner = None
        owners.append(owner)
        if owner is not None:
            prior_owner = owner
    if mapped[-1] != len(donor_ids) - 1:
        raise Lfm2SemanticSpanCpcfAlignmentError(
            "The target terminal boundary is not the authenticated donor endpoint."
        )
    if owners[-1] != len(donor_ids) - 1:
        raise Lfm2SemanticSpanCpcfAlignmentError(
            "The target terminal span has no authenticated donor owner."
        )
    return tuple(mapped), tuple(owners)


@dataclass(frozen=True, slots=True)
class Lfm2SemanticSpanCpcfPhaseAlignment:
    """One target-native phase plus its exact cached-parent row map."""

    phase: str
    semantic_phase_sha256: str
    donor_full_prefix_token_ids: tuple[int, ...]
    target_full_prefix_token_ids: tuple[int, ...]
    target_position_origin: int
    target_to_donor_boundary_positions: tuple[int | None, ...]
    target_to_donor_span_owner_positions: tuple[int | None, ...]
    parent_sampled_suffix_tokens: int
    parent_history_suffix_tokens: int
    target_rewrite_start: int
    target_replaced_suffix_tokens: int
    target_replacement_token_ids: tuple[int, ...]
    alignment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.semantic_phase_sha256, "semantic phase SHA-256")
        donor = tuple(self.donor_full_prefix_token_ids)
        target = tuple(self.target_full_prefix_token_ids)
        mapping = tuple(self.target_to_donor_boundary_positions)
        owners = tuple(self.target_to_donor_span_owner_positions)
        sampled = _positive_int(
            self.parent_sampled_suffix_tokens,
            "parent_sampled_suffix_tokens",
        )
        history = _positive_int(
            self.parent_history_suffix_tokens,
            "parent_history_suffix_tokens",
        )
        if (
            not donor
            or not target
            or len(mapping) != len(target)
            or len(owners) != len(target)
            or sampled > len(donor)
            or history > len(donor)
            or any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    or value >= len(donor)
                )
                for value in mapping
            )
            or any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    or value >= len(donor)
                )
                for value in owners
            )
        ):
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Phase token geometry or endpoint map is invalid."
            )
        observed = tuple(value for value in mapping if value is not None)
        observed_owners = tuple(value for value in owners if value is not None)
        if any(right <= left for left, right in zip(observed, observed[1:])):
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Mapped target endpoints do not preserve strict causal order."
            )
        if any(
            right < left for left, right in zip(observed_owners, observed_owners[1:])
        ):
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Target span owners do not preserve causal order."
            )
        if mapping[-1] != len(donor) - 1:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Phase endpoint map does not preserve the terminal boundary."
            )
        if owners[-1] != len(donor) - 1:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Phase span-owner map does not preserve the terminal donor node."
            )
        if (
            self.target_rewrite_start < 0
            or self.target_rewrite_start > len(target)
            or self.target_replaced_suffix_tokens < 0
            or tuple(self.target_replacement_token_ids)
            != target[self.target_rewrite_start :]
        ):
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Semantic suffix-rewrite geometry is invalid."
            )
        object.__setattr__(self, "donor_full_prefix_token_ids", donor)
        object.__setattr__(self, "target_full_prefix_token_ids", target)
        object.__setattr__(self, "target_to_donor_boundary_positions", mapping)
        object.__setattr__(self, "target_to_donor_span_owner_positions", owners)
        object.__setattr__(
            self,
            "target_replacement_token_ids",
            tuple(self.target_replacement_token_ids),
        )
        object.__setattr__(self, "alignment_sha256", canonical_sha256(self._body()))

    @property
    def donor_full_prefix_tokens(self) -> int:
        return len(self.donor_full_prefix_token_ids)

    @property
    def target_full_prefix_tokens(self) -> int:
        return len(self.target_full_prefix_token_ids)

    @property
    def missing_target_boundary_positions(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, value in enumerate(self.target_to_donor_boundary_positions)
            if value is None
        )

    @property
    def missing_target_span_owner_positions(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, value in enumerate(self.target_to_donor_span_owner_positions)
            if value is None
        )

    @property
    def donor_span_multiplicities(self) -> tuple[int, ...]:
        return tuple(
            sum(
                value == donor_index
                for value in self.target_to_donor_span_owner_positions
            )
            for donor_index in range(self.donor_full_prefix_tokens)
        )

    @property
    def omitted_donor_boundary_positions(self) -> tuple[int, ...]:
        retained = {
            value
            for value in self.target_to_donor_boundary_positions
            if value is not None
        }
        return tuple(
            index
            for index in range(self.donor_full_prefix_tokens)
            if index not in retained
        )

    @property
    def exact_endpoint_coverage(self) -> bool:
        return not self.missing_target_boundary_positions

    @property
    def exact_cached_attention_lattice(self) -> bool:
        """Whether cached post-RoPE scores/probabilities remain exact."""

        return self.target_to_donor_boundary_positions == tuple(
            range(self.donor_full_prefix_tokens)
        )

    @property
    def endpoint_row_gather_supported(self) -> bool:
        """Whether every target endpoint has a literal cached parent row."""

        return self.exact_endpoint_coverage

    @property
    def probability_preserving_attention_transport_supported(self) -> bool:
        """Whether donor attention nodes can be losslessly refined.

        Every target node must belong to one donor semantic span, and every
        donor node must own at least one target node.  Repeated owners are a
        pure refinement: duplicate K/V and partition each donor key's logit
        mass across the causally available duplicates.  A zero multiplicity is
        contraction and cannot generally preserve softmax context with a fixed
        target K/V row.
        """

        return not self.missing_target_span_owner_positions and all(
            value >= 1 for value in self.donor_span_multiplicities
        )

    @property
    def attention_transport_kind(self) -> str:
        if self.exact_cached_attention_lattice:
            return "isomorphic"
        if self.probability_preserving_attention_transport_supported:
            return "probability-preserving-span-expansion"
        if any(value == 0 for value in self.donor_span_multiplicities):
            return "contraction-not-exact"
        return "missing-span-owner-evidence"

    @property
    def append_only_transition(self) -> bool:
        if self.phase == "prefill":
            return True
        previous_length = self.target_rewrite_start + self.target_replaced_suffix_tokens
        return (
            self.target_replaced_suffix_tokens == 0
            and self.target_rewrite_start == previous_length
        )

    @property
    def requires_full_prefix_replay(self) -> bool:
        return self.target_replaced_suffix_tokens > 0

    @property
    def target_sampled_suffix_start(self) -> int:
        parent_start = self.donor_full_prefix_tokens - self.parent_sampled_suffix_tokens
        start = self.target_full_prefix_tokens
        for index in range(self.target_full_prefix_tokens - 1, -1, -1):
            mapped = self.target_to_donor_boundary_positions[index]
            if mapped is None or mapped < parent_start:
                break
            start = index
        return start

    @property
    def target_sampled_suffix_tokens(self) -> int:
        return self.target_full_prefix_tokens - self.target_sampled_suffix_start

    @property
    def parent_sampled_gather_indices(self) -> tuple[int, ...]:
        start = self.donor_full_prefix_tokens - self.parent_sampled_suffix_tokens
        result = self.target_to_donor_boundary_positions[
            self.target_sampled_suffix_start :
        ]
        if not result or any(value is None for value in result):
            raise Lfm2SemanticSpanCpcfAlignmentError(
                f"Phase {self.phase!r} has no exact terminal parent evidence rows."
            )
        return tuple(int(value) - start for value in result)

    @property
    def target_span_sampled_suffix_start(self) -> int:
        """First target node owned by the parent's sampled donor suffix."""

        parent_start = self.donor_full_prefix_tokens - self.parent_sampled_suffix_tokens
        start = self.target_full_prefix_tokens
        for index in range(self.target_full_prefix_tokens - 1, -1, -1):
            owner = self.target_to_donor_span_owner_positions[index]
            if owner is None or owner < parent_start:
                break
            start = index
        return start

    @property
    def target_span_sampled_suffix_tokens(self) -> int:
        return self.target_full_prefix_tokens - self.target_span_sampled_suffix_start

    @property
    def parent_sampled_span_gather_indices(self) -> tuple[int, ...]:
        start = self.donor_full_prefix_tokens - self.parent_sampled_suffix_tokens
        result = self.target_to_donor_span_owner_positions[
            self.target_span_sampled_suffix_start :
        ]
        if not result or any(value is None for value in result):
            raise Lfm2SemanticSpanCpcfAlignmentError(
                f"Phase {self.phase!r} has no owned terminal parent span rows."
            )
        return tuple(int(value) - start for value in result)

    @property
    def canonical_target_input_ids(self) -> Any:
        return _torch().tensor([self.target_full_prefix_token_ids], dtype=_torch().long)

    @property
    def canonical_target_attention_mask(self) -> Any:
        return _torch().ones((1, self.target_full_prefix_tokens), dtype=_torch().long)

    @property
    def canonical_target_position_ids(self) -> Any:
        return (
            _torch()
            .arange(
                self.target_position_origin,
                self.target_position_origin + self.target_full_prefix_tokens,
                dtype=_torch().long,
            )
            .unsqueeze(0)
        )

    def canonical_target_replay_batch(self) -> dict[str, Any]:
        """Return the exact full-prefix IDs used by target-native inference."""

        return {
            "input_ids": self.canonical_target_input_ids,
            "attention_mask": self.canonical_target_attention_mask,
            "position_ids": self.canonical_target_position_ids,
        }

    def require_exact_cached_attention_lattice(self) -> None:
        if not self.exact_cached_attention_lattice:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                f"Phase {self.phase!r} changed the causal token lattice; cached "
                "post-RoPE scores, probabilities, and context are not exact on "
                "the target prefix. Rephase normalized Q/K and recompute attention."
            )

    def require_endpoint_row_gather_support(self) -> None:
        if not self.endpoint_row_gather_supported:
            missing = ", ".join(
                str(value) for value in self.missing_target_boundary_positions
            )
            raise Lfm2SemanticSpanCpcfAlignmentError(
                f"Phase {self.phase!r} has target boundaries ({missing}) with no "
                "cached parent endpoint activation. Literal row gathering is absent."
            )

    def require_probability_preserving_attention_transport(self) -> None:
        if not self.probability_preserving_attention_transport_supported:
            omitted = ", ".join(
                str(index)
                for index, count in enumerate(self.donor_span_multiplicities)
                if count == 0
            )
            raise Lfm2SemanticSpanCpcfAlignmentError(
                f"Phase {self.phase!r} is not a pure donor-node refinement; "
                f"omitted donor nodes [{omitted}] make exact probability/context "
                "transport unavailable."
            )

    def require_incremental_target_transition(self) -> None:
        if self.requires_full_prefix_replay:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                f"Phase {self.phase!r} rewrites {self.target_replaced_suffix_tokens} "
                "prior target tokens and must be replayed from its complete "
                "canonical prefix; an append-only KV-cache transition is invalid."
            )

    def require_every_target_action_logit(self) -> None:
        if self.phase != "prefill" and (
            self.requires_full_prefix_replay
            or len(self.target_replacement_token_ids) != 1
        ):
            raise Lfm2SemanticSpanCpcfAlignmentError(
                f"Phase {self.phase!r} needs intermediate target actions for which "
                "the one cached parent logit row contains no evidence."
            )

    def gather_parent_history_rows(self, value: Any, *, sequence_axis: int) -> Any:
        """Gather one complete parent history at exact target endpoints."""

        self.require_endpoint_row_gather_support()
        tensor = _finite_tensor(value, "parent history")
        axis = _axis(sequence_axis, tensor.ndim, "sequence_axis")
        if int(tensor.shape[axis]) != self.donor_full_prefix_tokens:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Parent history length differs from the donor phase prefix."
            )
        indices = _torch().tensor(
            self.target_to_donor_boundary_positions,
            dtype=_torch().long,
        )
        return tensor.index_select(axis, indices).contiguous()

    def expand_parent_history_rows(self, value: Any, *, sequence_axis: int) -> Any:
        """Duplicate complete parent rows over a lossless target refinement."""

        self.require_probability_preserving_attention_transport()
        tensor = _finite_tensor(value, "parent refinement history")
        axis = _axis(sequence_axis, tensor.ndim, "sequence_axis")
        if int(tensor.shape[axis]) != self.donor_full_prefix_tokens:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Parent refinement history differs from the donor phase prefix."
            )
        indices = _torch().tensor(
            self.target_to_donor_span_owner_positions,
            dtype=_torch().long,
        )
        return tensor.index_select(axis, indices).contiguous()

    def gather_parent_sampled_rows(self, value: Any, *, sequence_axis: int) -> Any:
        """Gather the maximal exact target suffix from sampled parent rows."""

        tensor = _finite_tensor(value, "sampled parent rows")
        axis = _axis(sequence_axis, tensor.ndim, "sequence_axis")
        if int(tensor.shape[axis]) != self.parent_sampled_suffix_tokens:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Sampled parent tensor length differs from its sealed suffix."
            )
        indices = _torch().tensor(
            self.parent_sampled_gather_indices,
            dtype=_torch().long,
        )
        return tensor.index_select(axis, indices).contiguous()

    def expand_parent_sampled_rows(self, value: Any, *, sequence_axis: int) -> Any:
        """Duplicate sampled donor rows over target nodes in a pure refinement."""

        self.require_probability_preserving_attention_transport()
        tensor = _finite_tensor(value, "sampled parent refinement rows")
        axis = _axis(sequence_axis, tensor.ndim, "sequence_axis")
        if int(tensor.shape[axis]) != self.parent_sampled_suffix_tokens:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Sampled parent refinement tensor differs from its sealed suffix."
            )
        indices = _torch().tensor(
            self.parent_sampled_span_gather_indices,
            dtype=_torch().long,
        )
        return tensor.index_select(axis, indices).contiguous()

    def gather_parent_attention_matrix(
        self,
        value: Any,
        *,
        query_axis: int = -2,
        key_axis: int = -1,
    ) -> Any:
        """Gather raw QK/mask cells; probabilities must be recomputed."""

        tensor = _finite_tensor(value, "parent attention matrix")
        q_axis = _axis(query_axis, tensor.ndim, "query_axis")
        k_axis = _axis(key_axis, tensor.ndim, "key_axis")
        if q_axis == k_axis:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Attention query and key axes must differ."
            )
        if int(tensor.shape[q_axis]) != self.parent_sampled_suffix_tokens:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Attention query rows differ from the sampled parent suffix."
            )
        if int(tensor.shape[k_axis]) != self.donor_full_prefix_tokens:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Attention key columns differ from the complete donor prefix."
            )
        query_indices = _torch().tensor(
            self.parent_sampled_gather_indices,
            dtype=_torch().long,
        )
        key_indices = _torch().tensor(
            self.target_to_donor_boundary_positions,
            dtype=_torch().long,
        )
        return (
            tensor.index_select(q_axis, query_indices)
            .index_select(k_axis, key_indices)
            .contiguous()
        )

    def align_terminal_logits(
        self,
        value: Any,
        *,
        require_every_target_action: bool = True,
    ) -> Any:
        """Bind a cached terminal logit row without inventing span logits."""

        if require_every_target_action:
            self.require_every_target_action_logit()
        tensor = _finite_tensor(value, "cached terminal logits")
        if tensor.ndim < 2 or int(tensor.shape[-2]) != 1:
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "Cached phase logits must contain exactly one terminal row."
            )
        return tensor.clone()

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_SPAN_CPCF_PHASE_ALIGNMENT_SCHEMA,
            "phase": self.phase,
            "semantic_phase_sha256": self.semantic_phase_sha256,
            "donor_full_prefix_token_ids_sha256": canonical_sha256(
                list(self.donor_full_prefix_token_ids)
            ),
            "target_full_prefix_token_ids": list(self.target_full_prefix_token_ids),
            "target_position_origin": self.target_position_origin,
            "target_to_donor_boundary_positions": list(
                self.target_to_donor_boundary_positions
            ),
            "target_to_donor_span_owner_positions": list(
                self.target_to_donor_span_owner_positions
            ),
            "missing_target_boundary_positions": list(
                self.missing_target_boundary_positions
            ),
            "omitted_donor_boundary_positions": list(
                self.omitted_donor_boundary_positions
            ),
            "missing_target_span_owner_positions": list(
                self.missing_target_span_owner_positions
            ),
            "donor_span_multiplicities": list(self.donor_span_multiplicities),
            "parent_sampled_suffix_tokens": self.parent_sampled_suffix_tokens,
            "parent_history_suffix_tokens": self.parent_history_suffix_tokens,
            "target_sampled_suffix_start": self.target_sampled_suffix_start,
            "target_sampled_suffix_tokens": self.target_sampled_suffix_tokens,
            "parent_sampled_gather_indices": list(self.parent_sampled_gather_indices),
            "target_span_sampled_suffix_start": (self.target_span_sampled_suffix_start),
            "target_span_sampled_suffix_tokens": (
                self.target_span_sampled_suffix_tokens
            ),
            "parent_sampled_span_gather_indices": list(
                self.parent_sampled_span_gather_indices
            ),
            "target_rewrite_start": self.target_rewrite_start,
            "target_replaced_suffix_tokens": self.target_replaced_suffix_tokens,
            "target_replacement_token_ids": list(self.target_replacement_token_ids),
            "exact_endpoint_coverage": self.exact_endpoint_coverage,
            "exact_cached_attention_lattice": self.exact_cached_attention_lattice,
            "endpoint_row_gather_supported": self.endpoint_row_gather_supported,
            "probability_preserving_attention_transport_supported": (
                self.probability_preserving_attention_transport_supported
            ),
            "attention_transport_kind": self.attention_transport_kind,
            "append_only_transition": self.append_only_transition,
            "requires_full_prefix_replay": self.requires_full_prefix_replay,
            "numeric_token_id_equality_required": False,
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "alignment_sha256": self.alignment_sha256}


@dataclass(frozen=True, slots=True)
class Lfm2SemanticSpanCpcfAlignment:
    """All phase alignments for one sealed semantic bridge."""

    semantic_bridge_sha256: str
    source_batch_index: int
    phases: tuple[Lfm2SemanticSpanCpcfPhaseAlignment, ...]
    alignment_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _digest(self.semantic_bridge_sha256, "semantic bridge SHA-256")
        phases = tuple(self.phases)
        if (
            not phases
            or phases[0].phase != "prefill"
            or tuple(value.phase for value in phases[1:])
            != tuple(f"decode.{index:03d}" for index in range(len(phases) - 1))
        ):
            raise Lfm2SemanticSpanCpcfAlignmentError(
                "CPCF alignment phases are incomplete or out of order."
            )
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "alignment_sha256", canonical_sha256(self._body()))

    @property
    def phase_order(self) -> tuple[str, ...]:
        return tuple(value.phase for value in self.phases)

    @property
    def model_operation_audit(self) -> dict[str, int]:
        return dict(_MODEL_OPERATION_AUDIT)

    def get_phase(self, phase: str) -> Lfm2SemanticSpanCpcfPhaseAlignment:
        for value in self.phases:
            if value.phase == phase:
                return value
        raise KeyError(phase)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_SPAN_CPCF_ALIGNMENT_SCHEMA,
            "semantic_bridge_sha256": self.semantic_bridge_sha256,
            "source_batch_index": self.source_batch_index,
            "phase_order": list(self.phase_order),
            "phases": [value.to_evidence_dict() for value in self.phases],
            "coverage": {
                "all_target_endpoints_available": all(
                    value.exact_endpoint_coverage for value in self.phases
                ),
                "all_cached_attention_lattices_exact": all(
                    value.exact_cached_attention_lattice for value in self.phases
                ),
                "all_probability_preserving_attention_transports_supported": all(
                    value.probability_preserving_attention_transport_supported
                    for value in self.phases
                ),
                "suffix_rewrite_phase_count": sum(
                    value.requires_full_prefix_replay for value in self.phases
                ),
            },
            "operation_audit": self.model_operation_audit,
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "alignment_sha256": self.alignment_sha256}


def audit_lfm2_semantic_span_cpcf_refinement(
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    *,
    expected_batch_count: int = 16,
) -> Mapping[str, Any]:
    """Seal the all-cache go/no-go gate for long-horizon attention refolding.

    The gate covers both tokenizer geometry and sealed tensor coverage.  A
    sampled prefill that starts after donor position zero leaves an unrecoverable
    prefix hole even if every later decode row is present.  Rolling decode
    suffixes may extend a complete prefill, but cannot repair such a hole.
    Endpoint rows at inserted target subtoken boundaries remain a separate
    coverage statistic and are not invented.
    """

    expected = _positive_int(expected_batch_count, "expected_batch_count")
    ordered = tuple(alignments)
    if not ordered or any(
        not isinstance(value, Lfm2SemanticSpanCpcfAlignment) for value in ordered
    ):
        raise Lfm2SemanticSpanCpcfAlignmentError(
            "alignments must contain semantic CPCF batch alignments."
        )
    batch_indices = tuple(value.source_batch_index for value in ordered)
    if len(set(batch_indices)) != len(batch_indices):
        raise Lfm2SemanticSpanCpcfAlignmentError(
            "Refinement audit batch indices must be unique."
        )

    phases = tuple(phase for alignment in ordered for phase in alignment.phases)
    donor_nodes = sum(phase.donor_full_prefix_tokens for phase in phases)
    target_nodes = sum(phase.target_full_prefix_tokens for phase in phases)
    exact_target_endpoints = sum(
        phase.target_full_prefix_tokens - len(phase.missing_target_boundary_positions)
        for phase in phases
    )
    multiplicities = tuple(
        count for phase in phases for count in phase.donor_span_multiplicities
    )
    contraction_count = sum(count == 0 for count in multiplicities)
    expanded_donor_count = sum(count > 1 for count in multiplicities)
    missing_owner_count = sum(
        len(phase.missing_target_span_owner_positions) for phase in phases
    )

    transition_count = 0
    monotonic = True
    missing_parent_history_node_occurrences = 0
    maximum_missing_parent_history_nodes = 0
    final_missing_parent_history_nodes = []
    for alignment in ordered:
        previous: Lfm2SemanticSpanCpcfPhaseAlignment | None = None
        covered_parent_positions: set[int] = set()
        for phase in alignment.phases:
            if previous is None:
                covered_parent_positions.clear()
            elif phase.donor_full_prefix_token_ids[
                : previous.donor_full_prefix_tokens
            ] != (previous.donor_full_prefix_token_ids):
                # This is also a nonmonotonic phase boundary; retaining numeric
                # row indices across it would manufacture history coverage.
                covered_parent_positions.clear()
            if previous is not None:
                transition_count += 1
                prior_tokens = previous.target_full_prefix_tokens
                if (
                    phase.requires_full_prefix_replay
                    or phase.target_full_prefix_tokens < prior_tokens
                    or phase.target_full_prefix_token_ids[:prior_tokens]
                    != previous.target_full_prefix_token_ids
                    or phase.target_to_donor_span_owner_positions[:prior_tokens]
                    != previous.target_to_donor_span_owner_positions
                ):
                    monotonic = False
            direct_start = (
                phase.donor_full_prefix_tokens - phase.parent_history_suffix_tokens
            )
            covered_parent_positions.update(
                range(direct_start, phase.donor_full_prefix_tokens)
            )
            missing_history = phase.donor_full_prefix_tokens - len(
                covered_parent_positions
            )
            missing_parent_history_node_occurrences += missing_history
            maximum_missing_parent_history_nodes = max(
                maximum_missing_parent_history_nodes,
                missing_history,
            )
            previous = phase
        final_missing_parent_history_nodes.append(
            {
                "batch_index": alignment.source_batch_index,
                "missing_donor_nodes": (
                    0
                    if previous is None
                    else previous.donor_full_prefix_tokens
                    - len(covered_parent_positions)
                ),
            }
        )

    batch_count_exact = len(ordered) == expected
    all_donor_nodes_owned = contraction_count == 0
    all_target_nodes_owned = missing_owner_count == 0
    complete_parent_attention_histories = missing_parent_history_node_occurrences == 0
    go = (
        batch_count_exact
        and all_donor_nodes_owned
        and all_target_nodes_owned
        and monotonic
        and complete_parent_attention_histories
    )
    blockers = []
    if not batch_count_exact:
        blockers.append("expected-batch-count-not-covered")
    if not all_donor_nodes_owned:
        blockers.append("donor-node-contraction")
    if not all_target_nodes_owned:
        blockers.append("target-node-without-donor-span")
    if not monotonic:
        blockers.append("nonmonotonic-target-phase-boundary")
    if not complete_parent_attention_histories:
        blockers.append("parent-attention-history-incomplete")
    body = {
        "schema": SEMANTIC_SPAN_CPCF_REFINEMENT_AUDIT_SCHEMA,
        "expected_batch_count": expected,
        "observed_batch_count": len(ordered),
        "ordered_batch_indices": list(batch_indices),
        "phase_count": len(phases),
        "phase_transition_count": transition_count,
        "total_donor_node_occurrences": donor_nodes,
        "total_target_node_occurrences": target_nodes,
        "maximum_target_nodes_per_donor_node": max(multiplicities, default=0),
        "expanded_donor_node_occurrences": expanded_donor_count,
        "contracted_donor_node_occurrences": contraction_count,
        "target_nodes_without_span_owner": missing_owner_count,
        "exact_endpoint_coverage": {
            "covered_target_node_occurrences": exact_target_endpoints,
            "total_target_node_occurrences": target_nodes,
            "rate": exact_target_endpoints / target_nodes,
        },
        "every_donor_node_owns_target_node": all_donor_nodes_owned,
        "every_target_node_has_donor_span_owner": all_target_nodes_owned,
        "no_contractions": all_donor_nodes_owned,
        "phase_boundaries_monotonic": monotonic,
        "parent_attention_history_coverage": {
            "all_phase_histories_reconstructable": (
                complete_parent_attention_histories
            ),
            "missing_donor_node_occurrences": (missing_parent_history_node_occurrences),
            "maximum_missing_donor_nodes_in_one_phase": (
                maximum_missing_parent_history_nodes
            ),
            "final_phase_missing_donor_nodes_by_batch": (
                final_missing_parent_history_nodes
            ),
            "rolling_suffixes_may_extend_but_not_repair_prefill": True,
        },
        "probability_preserving_attention_transport_go": go,
        "blocking_reasons": blockers,
        "model_operations": dict(_MODEL_OPERATION_AUDIT),
    }
    return MappingProxyType(
        {
            **body,
            "report_sha256": canonical_sha256(body),
        }
    )


def build_lfm2_semantic_span_cpcf_alignment(
    semantic_bridge: Lfm2SemanticSpanPhaseBridge,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    parent_sampled_suffix_tokens: Mapping[str, int],
    parent_history_suffix_tokens: Mapping[str, int] | None = None,
) -> Lfm2SemanticSpanCpcfAlignment:
    """Build a target-native row map from one authenticated semantic bridge."""

    if not isinstance(semantic_bridge, Lfm2SemanticSpanPhaseBridge):
        raise TypeError("semantic_bridge must be Lfm2SemanticSpanPhaseBridge.")
    semantic_bridge.verify(donor_tokenizer, target_tokenizer)
    if not isinstance(parent_sampled_suffix_tokens, Mapping) or set(
        parent_sampled_suffix_tokens
    ) != set(semantic_bridge.phase_order):
        raise Lfm2SemanticSpanCpcfAlignmentError(
            "Parent sampled-suffix counts must cover every semantic phase exactly."
        )
    history_counts = (
        parent_sampled_suffix_tokens
        if parent_history_suffix_tokens is None
        else parent_history_suffix_tokens
    )
    if not isinstance(history_counts, Mapping) or set(history_counts) != set(
        semantic_bridge.phase_order
    ):
        raise Lfm2SemanticSpanCpcfAlignmentError(
            "Parent history-suffix counts must cover every semantic phase exactly."
        )
    phases = []
    for semantic_phase in semantic_bridge.phases:
        donor_ids = _donor_prefix_ids(semantic_bridge, semantic_phase)
        sampled = _positive_int(
            parent_sampled_suffix_tokens[semantic_phase.phase],
            f"parent sampled suffix for {semantic_phase.phase}",
        )
        history = _positive_int(
            history_counts[semantic_phase.phase],
            f"parent history suffix for {semantic_phase.phase}",
        )
        mapping, owners = _target_to_donor_boundaries(
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
            donor_ids=donor_ids,
            target_ids=semantic_phase.target_full_prefix_token_ids,
        )
        phases.append(
            Lfm2SemanticSpanCpcfPhaseAlignment(
                phase=semantic_phase.phase,
                semantic_phase_sha256=semantic_phase.phase_sha256,
                donor_full_prefix_token_ids=donor_ids,
                target_full_prefix_token_ids=(
                    semantic_phase.target_full_prefix_token_ids
                ),
                target_position_origin=semantic_phase.target_position_origin,
                target_to_donor_boundary_positions=mapping,
                target_to_donor_span_owner_positions=owners,
                parent_sampled_suffix_tokens=sampled,
                parent_history_suffix_tokens=history,
                target_rewrite_start=semantic_phase.target_rewrite_start,
                target_replaced_suffix_tokens=(
                    semantic_phase.target_replaced_suffix_token_count
                ),
                target_replacement_token_ids=(
                    semantic_phase.target_replacement_token_ids
                ),
            )
        )
    result = Lfm2SemanticSpanCpcfAlignment(
        semantic_bridge_sha256=semantic_bridge.bridge_sha256,
        source_batch_index=semantic_bridge.source_batch_index,
        phases=tuple(phases),
    )
    semantic_bridge.verify(donor_tokenizer, target_tokenizer)
    return result


__all__ = [
    "Lfm2SemanticSpanCpcfAlignment",
    "Lfm2SemanticSpanCpcfAlignmentError",
    "Lfm2SemanticSpanCpcfPhaseAlignment",
    "SEMANTIC_SPAN_CPCF_ALIGNMENT_SCHEMA",
    "SEMANTIC_SPAN_CPCF_PHASE_ALIGNMENT_SCHEMA",
    "SEMANTIC_SPAN_CPCF_REFINEMENT_AUDIT_SCHEMA",
    "audit_lfm2_semantic_span_cpcf_refinement",
    "build_lfm2_semantic_span_cpcf_alignment",
]
