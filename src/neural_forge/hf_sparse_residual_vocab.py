"""Tied low-rank vocabulary modules with exact sparse token-row residuals.

The runtime owns one physical core shared by both the token embedding and LM
head.  The core stores:

* one vocabulary factor ``A[vocab, rank]``;
* one hidden factor ``B[rank, hidden]``;
* one sorted integer token-ID buffer ``S[k]``; and
* one residual Parameter ``R[k, hidden]``.

Embedding lookup computes the low-rank base and adds ``R`` only when the input
ID belongs to ``S``.  The LM head computes all base logits and index-adds
``hidden @ R.T`` only into the same selected vocabulary columns.  No dense
vocabulary matrix or dense token-to-residual lookup is retained.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neural_forge.hf_factorized_vocab import (
    TiedVocabularySVDAnalysis,
    exact_parameter_report,
)


class SparseResidualVocabularyError(ValueError):
    """A factor, selected-token set, residual, or budget is invalid."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SparseResidualVocabularyError(
            f"{name} must be a positive integer."
        )
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SparseResidualVocabularyError(f"{name} must be in (0, 1).")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SparseResidualVocabularyError(f"{name} must be in (0, 1).") from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise SparseResidualVocabularyError(f"{name} must be in (0, 1).")
    return result


def _frozen_parameter(value: Tensor, name: str) -> nn.Parameter:
    if not isinstance(value, Tensor):
        raise SparseResidualVocabularyError(f"{name} must be a Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise SparseResidualVocabularyError(
            f"{name} must be a dense non-meta Tensor."
        )
    if not torch.is_floating_point(value) or not bool(
        torch.isfinite(value.detach()).all()
    ):
        raise SparseResidualVocabularyError(
            f"{name} must be a finite floating-point Tensor."
        )
    if isinstance(value, nn.Parameter) and not value.requires_grad:
        if value.grad is not None:
            raise SparseResidualVocabularyError(f"{name} must not carry a gradient.")
        return value
    return nn.Parameter(value.detach().clone().contiguous(), requires_grad=False)


def _selected_ids(
    value: Any,
    *,
    vocab_size: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise SparseResidualVocabularyError(
            "selected_token_ids must be a Tensor."
        )
    if (
        value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 1
        or value.numel() <= 0
        or value.dtype != torch.long
    ):
        raise SparseResidualVocabularyError(
            "selected_token_ids must be a nonempty one-dimensional int64 Tensor."
        )
    result = value.detach().clone().contiguous()
    if (
        int(result[0].item()) < 0
        or int(result[-1].item()) >= vocab_size
        or not bool((result[1:] > result[:-1]).all())
    ):
        raise SparseResidualVocabularyError(
            "selected_token_ids must be sorted, unique, and inside the vocabulary."
        )
    return result


@dataclass(frozen=True, slots=True)
class SparseResidualVocabularyPlan:
    """Physical learned-scalar and integer-buffer accounting for one core."""

    vocab_size: int
    hidden_size: int
    rank: int
    selected_row_count: int
    factor_parameter_count: int
    residual_parameter_count: int
    head_bias_parameter_count: int
    unique_parameter_count: int
    logical_parameter_count: int
    dense_unique_parameter_count: int
    integer_buffer_count: int
    integer_buffer_bytes: int
    selected_token_ids_sha256: str
    selected_reconstruction_max_abs: float
    selected_reconstruction_relative_rmse: float

    @property
    def compression_ratio(self) -> float:
        return self.unique_parameter_count / self.dense_unique_parameter_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "selected_row_count": self.selected_row_count,
            "factor_parameter_count": self.factor_parameter_count,
            "residual_parameter_count": self.residual_parameter_count,
            "head_bias_parameter_count": self.head_bias_parameter_count,
            "unique_parameter_count": self.unique_parameter_count,
            "logical_parameter_count": self.logical_parameter_count,
            "dense_unique_parameter_count": self.dense_unique_parameter_count,
            "compression_ratio": self.compression_ratio,
            "integer_buffer_count": self.integer_buffer_count,
            "integer_buffer_bytes": self.integer_buffer_bytes,
            "selected_token_ids_sha256": self.selected_token_ids_sha256,
            "selected_reconstruction_max_abs": (
                self.selected_reconstruction_max_abs
            ),
            "selected_reconstruction_relative_rmse": (
                self.selected_reconstruction_relative_rmse
            ),
        }


class SparseResidualVocabularyCore(nn.Module):
    """One physical low-rank base and exact sparse residual-row dictionary."""

    vocab_factor: nn.Parameter
    hidden_factor: nn.Parameter
    residual_rows: nn.Parameter
    selected_token_ids: Tensor

    def __init__(
        self,
        vocab_factor: Tensor,
        hidden_factor: Tensor,
        selected_token_ids: Tensor,
        residual_rows: Tensor,
        *,
        plan: SparseResidualVocabularyPlan | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(vocab_factor, Tensor)
            or not isinstance(hidden_factor, Tensor)
            or vocab_factor.ndim != 2
            or hidden_factor.ndim != 2
            or vocab_factor.shape[1] != hidden_factor.shape[0]
            or vocab_factor.dtype != hidden_factor.dtype
            or vocab_factor.device != hidden_factor.device
        ):
            raise SparseResidualVocabularyError(
                "Vocabulary and hidden factors are incompatible."
            )
        vocab_size = int(vocab_factor.shape[0])
        hidden_size = int(hidden_factor.shape[1])
        rank = int(vocab_factor.shape[1])
        if rank <= 0 or rank > min(vocab_size, hidden_size):
            raise SparseResidualVocabularyError("Factor rank is invalid.")
        selected = _selected_ids(selected_token_ids, vocab_size=vocab_size)
        if (
            not isinstance(residual_rows, Tensor)
            or residual_rows.ndim != 2
            or tuple(residual_rows.shape)
            != (int(selected.numel()), hidden_size)
            or residual_rows.dtype != vocab_factor.dtype
            or residual_rows.device != vocab_factor.device
        ):
            raise SparseResidualVocabularyError(
                "residual_rows must have shape [selected_count, hidden_size] "
                "and match the factor dtype/device."
            )
        self.vocab_factor = _frozen_parameter(vocab_factor, "vocab_factor")
        self.hidden_factor = _frozen_parameter(hidden_factor, "hidden_factor")
        self.residual_rows = _frozen_parameter(residual_rows, "residual_rows")
        self.register_buffer(
            "selected_token_ids",
            selected.to(device=vocab_factor.device),
            persistent=True,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.selected_row_count = int(selected.numel())
        self.plan = plan

    def base_embedding(self, input_ids: Tensor) -> Tensor:
        rows = F.embedding(input_ids, self.vocab_factor)
        return torch.matmul(rows, self.hidden_factor)

    def _selection_positions(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        flat = input_ids.reshape(-1)
        positions = torch.searchsorted(self.selected_token_ids, flat)
        safe = positions.clamp_max(self.selected_row_count - 1)
        matched = self.selected_token_ids.index_select(0, safe).eq(flat)
        return safe.reshape(input_ids.shape), matched.reshape(input_ids.shape)

    def embed(self, input_ids: Tensor) -> Tensor:
        base = self.base_embedding(input_ids)
        positions, matched = self._selection_positions(input_ids)
        residual = F.embedding(positions, self.residual_rows)
        return base + residual * matched.unsqueeze(-1).to(dtype=base.dtype)

    def base_logits(self, hidden_states: Tensor) -> Tensor:
        rank_activations = F.linear(hidden_states, self.hidden_factor)
        return F.linear(rank_activations, self.vocab_factor)

    def logits(self, hidden_states: Tensor, bias: Tensor | None = None) -> Tensor:
        base = self.base_logits(hidden_states)
        residual_logits = F.linear(hidden_states, self.residual_rows)
        base.index_add_(
            -1,
            self.selected_token_ids,
            residual_logits,
        )
        if bias is not None:
            base = base + bias
        return base

    def materialize_base_weight(self) -> Tensor:
        with torch.no_grad():
            return (self.vocab_factor @ self.hidden_factor).detach().contiguous()

    def materialize_weight(self) -> Tensor:
        with torch.no_grad():
            weight = self.materialize_base_weight()
            weight.index_add_(
                0,
                self.selected_token_ids,
                self.residual_rows,
            )
            return weight.detach().contiguous()


class SparseResidualEmbedding(nn.Module):
    """Embedding wrapper around one shared sparse-residual vocabulary core."""

    def __init__(
        self,
        core: SparseResidualVocabularyCore,
        *,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(core, SparseResidualVocabularyCore):
            raise SparseResidualVocabularyError(
                "core must be a SparseResidualVocabularyCore."
            )
        self.core = core
        self.num_embeddings = core.vocab_size
        self.embedding_dim = core.hidden_size
        self.padding_idx = padding_idx

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.core.embed(input_ids)

    def materialize_weight(self) -> Tensor:
        return self.core.materialize_weight()


class SparseResidualLMHead(nn.Module):
    """LM-head wrapper around the same sparse-residual vocabulary core."""

    def __init__(
        self,
        core: SparseResidualVocabularyCore,
        *,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(core, SparseResidualVocabularyCore):
            raise SparseResidualVocabularyError(
                "core must be a SparseResidualVocabularyCore."
            )
        self.core = core
        self.in_features = core.hidden_size
        self.out_features = core.vocab_size
        if bias is None:
            self.register_parameter("bias", None)
        else:
            if (
                not isinstance(bias, Tensor)
                or bias.ndim != 1
                or int(bias.shape[0]) != core.vocab_size
                or bias.dtype != core.vocab_factor.dtype
                or bias.device != core.vocab_factor.device
            ):
                raise SparseResidualVocabularyError(
                    "bias must match the vocabulary factor."
                )
            self.bias = _frozen_parameter(bias, "bias")

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.core.logits(hidden_states, self.bias)

    def materialize_weight(self) -> Tensor:
        return self.core.materialize_weight()


class SparseResidualVocabularyPair(nn.Module):
    """Embedding/head pair sharing one core, factors, residual, and ID buffer."""

    def __init__(
        self,
        core: SparseResidualVocabularyCore,
        embedding: SparseResidualEmbedding,
        lm_head: SparseResidualLMHead,
        *,
        plan: SparseResidualVocabularyPlan,
    ) -> None:
        super().__init__()
        if (
            not isinstance(core, SparseResidualVocabularyCore)
            or embedding.core is not core
            or lm_head.core is not core
            or not isinstance(plan, SparseResidualVocabularyPlan)
        ):
            raise SparseResidualVocabularyError(
                "Sparse residual pair must share one exact core."
            )
        self.core = core
        self.embedding = embedding
        self.lm_head = lm_head
        self.plan = plan

    @property
    def unique_parameter_count(self) -> int:
        return int(
            exact_parameter_report(
                self.embedding,
                self.lm_head,
            )["unique_parameter_count"]
        )

    def parameter_report(self) -> dict[str, Any]:
        physical = exact_parameter_report(self.embedding, self.lm_head)
        buffer_ids = {
            id(buffer): buffer
            for module in (self.embedding, self.lm_head)
            for buffer in module.buffers()
        }
        return {
            **physical,
            "planned_unique_parameter_count": self.plan.unique_parameter_count,
            "unique_integer_buffer_objects": len(buffer_ids),
            "integer_buffer_count": sum(
                int(buffer.numel()) for buffer in buffer_ids.values()
            ),
            "integer_buffer_bytes": sum(
                int(buffer.numel() * buffer.element_size())
                for buffer in buffer_ids.values()
            ),
            "one_shared_core_object": self.embedding.core is self.lm_head.core,
            "one_shared_residual_parameter_object": (
                self.embedding.core.residual_rows
                is self.lm_head.core.residual_rows
            ),
            "one_shared_selected_token_buffer_object": (
                self.embedding.core.selected_token_ids
                is self.lm_head.core.selected_token_ids
            ),
            "dense_vocabulary_matrix_retained": False,
        }


def build_sparse_residual_vocabulary_pair(
    analysis: TiedVocabularySVDAnalysis,
    source_embedding: nn.Embedding,
    source_lm_head: nn.Linear,
    *,
    rank: int,
    selected_token_ids: Tensor,
) -> SparseResidualVocabularyPair:
    """Build one shared core whose selected rows exactly residualize the base."""

    if not isinstance(analysis, TiedVocabularySVDAnalysis):
        raise SparseResidualVocabularyError(
            "analysis must be a TiedVocabularySVDAnalysis."
        )
    if not isinstance(source_embedding, nn.Embedding) or not isinstance(
        source_lm_head, nn.Linear
    ):
        raise SparseResidualVocabularyError(
            "Source vocabulary modules must be dense Embedding/Linear."
        )
    if source_embedding.weight is not source_lm_head.weight:
        raise SparseResidualVocabularyError(
            "Source embedding and LM head must share one exact Parameter."
        )
    if analysis.source_weight is not source_embedding.weight:
        with torch.no_grad():
            same_analyzed_matrix = bool(
                analysis.source_weight.shape == source_embedding.weight.shape
                and analysis.source_weight.dtype == source_embedding.weight.dtype
                and analysis.source_weight.device == source_embedding.weight.device
                and torch.equal(
                    analysis.source_weight.detach(),
                    source_embedding.weight.detach(),
                )
            )
        if not same_analyzed_matrix:
            raise SparseResidualVocabularyError(
                "The reusable SVD analysis is not bound to an exact copy of the "
                "source Parameter."
            )
    if (
        source_embedding.num_embeddings != analysis.vocab_size
        or source_embedding.embedding_dim != analysis.hidden_size
    ):
        raise SparseResidualVocabularyError(
            "Source modules differ from the analyzed tied matrix."
        )
    if (
        source_embedding.max_norm is not None
        or source_embedding.scale_grad_by_freq
        or source_embedding.sparse
    ):
        raise SparseResidualVocabularyError(
            "Sparse-residual inference requires an ordinary dense lookup source."
        )
    base_pair = analysis.factorize(rank)
    selected = _selected_ids(
        selected_token_ids,
        vocab_size=analysis.vocab_size,
    ).to(device=source_embedding.weight.device)
    with torch.no_grad():
        source_rows = source_embedding.weight.detach().index_select(0, selected)
        base_rows = base_pair.embedding.vocab_factor.index_select(
            0,
            selected,
        ) @ base_pair.embedding.hidden_factor
        residual = (source_rows - base_rows).contiguous()
        reconstructed = base_rows + residual
        error = (reconstructed - source_rows).to(dtype=torch.float64)
        max_abs = float(error.abs().max().item())
        rmse = torch.sqrt(torch.mean(error.square()))
        source_rms = torch.sqrt(
            torch.mean(source_rows.to(dtype=torch.float64).square())
        )
        relative = float((rmse / source_rms.clamp_min(1e-30)).item())
    factor_parameters = int(
        base_pair.embedding.vocab_factor.numel()
        + base_pair.embedding.hidden_factor.numel()
    )
    residual_parameters = int(residual.numel())
    bias_count = (
        0 if source_lm_head.bias is None else int(source_lm_head.bias.numel())
    )
    selected_list = [int(value) for value in selected.to(device="cpu").tolist()]
    plan = SparseResidualVocabularyPlan(
        vocab_size=analysis.vocab_size,
        hidden_size=analysis.hidden_size,
        rank=int(rank),
        selected_row_count=len(selected_list),
        factor_parameter_count=factor_parameters,
        residual_parameter_count=residual_parameters,
        head_bias_parameter_count=bias_count,
        unique_parameter_count=factor_parameters + residual_parameters + bias_count,
        logical_parameter_count=(
            2 * (factor_parameters + residual_parameters) + bias_count
        ),
        dense_unique_parameter_count=(
            analysis.vocab_size * analysis.hidden_size + bias_count
        ),
        integer_buffer_count=len(selected_list),
        integer_buffer_bytes=len(selected_list) * torch.tensor(
            [],
            dtype=torch.long,
        ).element_size(),
        selected_token_ids_sha256=_canonical_sha256(selected_list),
        selected_reconstruction_max_abs=max_abs,
        selected_reconstruction_relative_rmse=relative,
    )
    core = SparseResidualVocabularyCore(
        base_pair.embedding.vocab_factor,
        base_pair.embedding.hidden_factor,
        selected,
        residual,
        plan=plan,
    )
    embedding = SparseResidualEmbedding(
        core,
        padding_idx=source_embedding.padding_idx,
    )
    embedding.train(source_embedding.training)
    lm_head = SparseResidualLMHead(core, bias=source_lm_head.bias)
    lm_head.train(source_lm_head.training)
    pair = SparseResidualVocabularyPair(
        core,
        embedding,
        lm_head,
        plan=plan,
    )
    if pair.unique_parameter_count != plan.unique_parameter_count:
        raise SparseResidualVocabularyError(
            "Planned and physical sparse-residual parameter counts differ."
        )
    return pair


@dataclass(frozen=True, slots=True)
class SparseResidualTarget:
    """Exact total-budget allocation between a chosen base rank and residual rows."""

    original_parent_unique_parameters: int
    source_unique_parameters: int
    requested_total_compression_fraction: float
    requested_target_unique_parameters: int
    rank: int
    selected_row_budget: int
    physical_target_unique_parameters: int
    physical_total_compression_fraction: float
    learned_parameter_headroom: int
    integer_buffer_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_parent_unique_parameters": (
                self.original_parent_unique_parameters
            ),
            "source_unique_parameters": self.source_unique_parameters,
            "requested_total_compression_fraction": (
                self.requested_total_compression_fraction
            ),
            "requested_target_unique_parameters": (
                self.requested_target_unique_parameters
            ),
            "rank": self.rank,
            "selected_row_budget": self.selected_row_budget,
            "physical_target_unique_parameters": (
                self.physical_target_unique_parameters
            ),
            "physical_total_compression_fraction": (
                self.physical_total_compression_fraction
            ),
            "learned_parameter_headroom": self.learned_parameter_headroom,
            "integer_buffer_bytes": self.integer_buffer_bytes,
        }


def solve_sparse_residual_target(
    *,
    original_parent_unique_parameters: int,
    source_unique_parameters: int,
    vocab_size: int,
    hidden_size: int,
    total_compression_fraction: float,
    rank: int,
    minimum_selected_rows: int = 1,
) -> SparseResidualTarget:
    """Spend the remaining exact learned-parameter budget on residual rows."""

    parent = _positive_int(
        original_parent_unique_parameters,
        "original_parent_unique_parameters",
    )
    source = _positive_int(source_unique_parameters, "source_unique_parameters")
    vocab = _positive_int(vocab_size, "vocab_size")
    hidden = _positive_int(hidden_size, "hidden_size")
    selected_minimum = _positive_int(
        minimum_selected_rows,
        "minimum_selected_rows",
    )
    selected_rank = _positive_int(rank, "rank")
    if selected_rank > min(vocab, hidden):
        raise SparseResidualVocabularyError(
            "rank exceeds the vocabulary matrix maximum rank."
        )
    fraction = _fraction(
        total_compression_fraction,
        "total_compression_fraction",
    )
    target = math.floor(parent * (1.0 - fraction))
    dense = vocab * hidden
    factor = selected_rank * (vocab + hidden)
    fixed = source - dense + factor
    row_budget = min(vocab, (target - fixed) // hidden)
    if row_budget < selected_minimum:
        raise SparseResidualVocabularyError(
            "The rank leaves too little budget for the required residual rows."
        )
    physical = fixed + row_budget * hidden
    if physical > target or physical >= source:
        raise SparseResidualVocabularyError(
            "Sparse-residual allocation missed the compression target."
        )
    return SparseResidualTarget(
        original_parent_unique_parameters=parent,
        source_unique_parameters=source,
        requested_total_compression_fraction=fraction,
        requested_target_unique_parameters=target,
        rank=selected_rank,
        selected_row_budget=row_budget,
        physical_target_unique_parameters=physical,
        physical_total_compression_fraction=(parent - physical) / parent,
        learned_parameter_headroom=target - physical,
        integer_buffer_bytes=row_budget * torch.tensor(
            [],
            dtype=torch.long,
        ).element_size(),
    )


__all__ = [
    "SparseResidualEmbedding",
    "SparseResidualLMHead",
    "SparseResidualTarget",
    "SparseResidualVocabularyCore",
    "SparseResidualVocabularyError",
    "SparseResidualVocabularyPair",
    "SparseResidualVocabularyPlan",
    "build_sparse_residual_vocabulary_pair",
    "solve_sparse_residual_target",
]
