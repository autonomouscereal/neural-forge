"""Deterministic low-rank vocabulary modules for inference-only HF models.

The canonical factorization is ``weight ~= vocab_factor @ hidden_factor``.
It lets an embedding look up only rank-width rows before projecting to the
hidden width, and lets an LM head project hidden states to rank before scoring
the vocabulary.  Neither forward path constructs the dense vocabulary matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VocabularyFactorizationError(ValueError):
    """Base class for fail-closed vocabulary factorization errors."""


class FactorizationShapeError(VocabularyFactorizationError):
    """A module, tensor, or factor has an invalid shape or type."""


class FactorizationFiniteError(VocabularyFactorizationError):
    """A source tensor, factor, or decomposition is non-finite."""


class FactorizationRankError(VocabularyFactorizationError):
    """A requested or computed factor rank is invalid."""


class FactorizationBudgetError(VocabularyFactorizationError):
    """A parameter budget cannot represent a valid factorization."""


@dataclass(frozen=True, slots=True)
class FactorizationPlan:
    """Auditable rank selection and exact parameter counts for one matrix."""

    vocab_size: int
    hidden_size: int
    rank: int
    max_rank: int
    selection: str
    requested_energy: float | None
    retained_energy_fraction: float
    singular_values: tuple[float, ...]
    factor_copies: int
    factor_parameter_count: int
    fixed_parameter_count: int
    total_parameter_count: int
    dense_parameter_count: int
    parameter_budget: int | None
    budget_headroom: int | None

    @property
    def compression_ratio(self) -> float:
        """Factorized-to-dense parameter ratio, including fixed parameters."""

        return self.total_parameter_count / self.dense_parameter_count

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "max_rank": self.max_rank,
            "selection": self.selection,
            "requested_energy": self.requested_energy,
            "retained_energy_fraction": self.retained_energy_fraction,
            "singular_values": list(self.singular_values),
            "factor_copies": self.factor_copies,
            "factor_parameter_count": self.factor_parameter_count,
            "fixed_parameter_count": self.fixed_parameter_count,
            "total_parameter_count": self.total_parameter_count,
            "dense_parameter_count": self.dense_parameter_count,
            "compression_ratio": self.compression_ratio,
            "parameter_budget": self.parameter_budget,
            "budget_headroom": self.budget_headroom,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class VocabularyFactorizationPlan:
    """Pair-level plan with storage-aware embedding/head parameter counts."""

    tied: bool
    embedding: FactorizationPlan
    lm_head: FactorizationPlan
    head_bias_parameters: int
    logical_parameter_count: int
    unique_parameter_count: int
    dense_unique_parameter_count: int
    parameter_budget: int | None
    budget_headroom: int | None
    retained_energy_fraction: float

    @property
    def compression_ratio(self) -> float:
        """Unique factorized-to-dense parameter ratio."""

        return self.unique_parameter_count / self.dense_unique_parameter_count

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible pair report."""

        return {
            "tied": self.tied,
            "embedding": self.embedding.as_dict(),
            "lm_head": self.lm_head.as_dict(),
            "head_bias_parameters": self.head_bias_parameters,
            "logical_parameter_count": self.logical_parameter_count,
            "unique_parameter_count": self.unique_parameter_count,
            "dense_unique_parameter_count": self.dense_unique_parameter_count,
            "compression_ratio": self.compression_ratio,
            "parameter_budget": self.parameter_budget,
            "budget_headroom": self.budget_headroom,
            "retained_energy_fraction": self.retained_energy_fraction,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class _SVDAnalysis:
    source_weight: Tensor
    left_vectors: Tensor
    singular_values: Tensor
    right_vectors_t: Tensor

    @property
    def vocab_size(self) -> int:
        return int(self.source_weight.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.source_weight.shape[1])

    @property
    def max_rank(self) -> int:
        return int(self.singular_values.numel())


@dataclass(frozen=True, slots=True)
class TiedVocabularySVDAnalysis:
    """One reusable deterministic SVD for a truly tied vocabulary interface.

    The expensive decomposition is computed once by
    :func:`analyze_tied_vocabulary_modules`.  Multiple rank prefixes can then
    be materialized without recomputing the SVD or changing the canonical
    ``U S @ Vh`` factorization used by the ordinary pair API.
    """

    _analysis: _SVDAnalysis
    source_weight: Tensor
    padding_idx: int | None
    max_norm: float | None
    norm_type: float
    scale_grad_by_freq: bool
    sparse: bool
    head_bias: Tensor | None
    embedding_training: bool
    head_training: bool

    @property
    def vocab_size(self) -> int:
        return self._analysis.vocab_size

    @property
    def hidden_size(self) -> int:
        return self._analysis.hidden_size

    @property
    def max_rank(self) -> int:
        return self._analysis.max_rank

    @property
    def singular_values(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._analysis.singular_values.tolist())

    def factorize(self, rank: int) -> FactorizedVocabularyPair:
        """Build one tied factor pair from a prefix of this exact SVD."""

        return _factorize_tied_from_precomputed_analysis(self, rank=rank)


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FactorizationRankError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise FactorizationRankError(f"{name} cannot exceed {maximum}.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FactorizationShapeError(f"{name} must be a non-negative integer.")
    return value


def _energy_target(value: Any) -> float:
    if isinstance(value, bool):
        raise FactorizationRankError("energy must be in the interval (0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FactorizationRankError("energy must be in the interval (0, 1].") from exc
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise FactorizationRankError("energy must be in the interval (0, 1].")
    return result


def _parameter_budget(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FactorizationBudgetError("parameter_budget must be a positive integer.")
    return value


def _validate_selector(
    rank: int | None,
    energy: float | None,
    parameter_budget: int | None,
) -> None:
    supplied = sum(value is not None for value in (rank, energy, parameter_budget))
    if supplied > 1:
        raise VocabularyFactorizationError(
            "Specify at most one of rank, energy, or parameter_budget."
        )


def _validate_matrix_metadata(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise FactorizationShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise FactorizationShapeError(f"{name} must be a dense strided tensor.")
    if value.device.type == "meta":
        raise FactorizationShapeError(f"{name} cannot be a meta tensor.")
    if value.ndim != 2:
        raise FactorizationShapeError(f"{name} must be two-dimensional.")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise FactorizationShapeError(f"{name} cannot have an empty dimension.")
    if not torch.is_floating_point(value):
        raise FactorizationShapeError(f"{name} must have a real floating-point dtype.")
    return value


def _float64_matrix(value: Any, name: str) -> Tensor:
    value = _validate_matrix_metadata(value, name)
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise FactorizationFiniteError(f"{name} contains NaN or infinity.")
    return result


def _validate_vector(value: Any, name: str, *, length: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise FactorizationShapeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise FactorizationShapeError(f"{name} must be a dense non-meta tensor.")
    if value.ndim != 1 or value.shape[0] != length:
        raise FactorizationShapeError(f"{name} must have shape ({length},).")
    if not torch.is_floating_point(value):
        raise FactorizationShapeError(f"{name} must have a real floating-point dtype.")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise FactorizationFiniteError(f"{name} contains NaN or infinity.")
    return detached


def _analyze_weight(weight: Any, name: str = "weight") -> _SVDAnalysis:
    source_weight = _float64_matrix(weight, name)
    with torch.no_grad():
        try:
            left, singular_values, right_t = torch.linalg.svd(
                source_weight,
                full_matrices=False,
            )
        except RuntimeError as exc:
            raise FactorizationRankError("Deterministic float64 SVD failed.") from exc
        if not all(
            bool(torch.isfinite(value).all())
            for value in (left, singular_values, right_t)
        ):
            raise FactorizationFiniteError("Float64 SVD produced NaN or infinity.")

        columns = torch.arange(left.shape[1], dtype=torch.long)
        pivot_rows = torch.argmax(torch.abs(left), dim=0)
        pivots = left[pivot_rows, columns]
        signs = torch.where(
            pivots < 0.0, -torch.ones_like(pivots), torch.ones_like(pivots)
        )
        left = (left * signs.unsqueeze(0)).contiguous()
        right_t = (right_t * signs.unsqueeze(1)).contiguous()

    return _SVDAnalysis(
        source_weight=source_weight,
        left_vectors=left,
        singular_values=singular_values.contiguous(),
        right_vectors_t=right_t,
    )


def _retained_energy_fraction(singular_values: Tensor, rank: int) -> float:
    largest = float(singular_values[0].item())
    if largest == 0.0:
        return 1.0
    scaled_energy = (singular_values / largest).square()
    fraction = float((scaled_energy[:rank].sum() / scaled_energy.sum()).item())
    if not math.isfinite(fraction):
        raise FactorizationFiniteError("Retained spectral energy is non-finite.")
    return min(1.0, max(0.0, fraction))


def _rank_for_energy(singular_values: Tensor, energy: float) -> int:
    if float(singular_values[0].item()) == 0.0:
        return 1
    if energy == 1.0:
        return int(singular_values.numel())
    scaled_energy = (singular_values / singular_values[0]).square()
    cumulative = torch.cumsum(scaled_energy, dim=0) / scaled_energy.sum()
    selected = int(torch.searchsorted(cumulative, energy, right=False).item()) + 1
    return min(selected, int(singular_values.numel()))


def _make_plan(
    analysis: _SVDAnalysis,
    *,
    rank: int,
    selection: str,
    requested_energy: float | None = None,
    parameter_budget: int | None = None,
    fixed_parameter_count: int = 0,
    factor_copies: int = 1,
) -> FactorizationPlan:
    factor_count = factor_copies * rank * (analysis.vocab_size + analysis.hidden_size)
    total_count = factor_count + fixed_parameter_count
    dense_count = (
        factor_copies * analysis.vocab_size * analysis.hidden_size
        + fixed_parameter_count
    )
    return FactorizationPlan(
        vocab_size=analysis.vocab_size,
        hidden_size=analysis.hidden_size,
        rank=rank,
        max_rank=analysis.max_rank,
        selection=selection,
        requested_energy=requested_energy,
        retained_energy_fraction=_retained_energy_fraction(
            analysis.singular_values, rank
        ),
        singular_values=tuple(
            float(value) for value in analysis.singular_values.tolist()
        ),
        factor_copies=factor_copies,
        factor_parameter_count=factor_count,
        fixed_parameter_count=fixed_parameter_count,
        total_parameter_count=total_count,
        dense_parameter_count=dense_count,
        parameter_budget=parameter_budget,
        budget_headroom=(
            None if parameter_budget is None else parameter_budget - total_count
        ),
    )


def _select_plan(
    analysis: _SVDAnalysis,
    *,
    rank: int | None,
    energy: float | None,
    parameter_budget: int | None,
    fixed_parameter_count: int,
    factor_copies: int,
) -> FactorizationPlan:
    _validate_selector(rank, energy, parameter_budget)
    fixed = _nonnegative_int(fixed_parameter_count, "fixed_parameter_count")
    copies = _positive_int(factor_copies, "factor_copies")

    if rank is not None:
        selected_rank = _positive_int(rank, "rank", maximum=analysis.max_rank)
        selection = "rank"
        requested_energy = None
        budget = None
    elif energy is not None:
        requested_energy = _energy_target(energy)
        selected_rank = _rank_for_energy(analysis.singular_values, requested_energy)
        selection = "energy"
        budget = None
    elif parameter_budget is not None:
        budget = _parameter_budget(parameter_budget)
        per_rank = copies * (analysis.vocab_size + analysis.hidden_size)
        selected_rank = min(analysis.max_rank, (budget - fixed) // per_rank)
        if selected_rank < 1:
            minimum = fixed + per_rank
            raise FactorizationBudgetError(
                f"parameter_budget must be at least {minimum} for rank 1."
            )
        selection = "parameter_budget"
        requested_energy = None
    else:
        selected_rank = analysis.max_rank
        selection = "full_rank"
        requested_energy = None
        budget = None

    return _make_plan(
        analysis,
        rank=selected_rank,
        selection=selection,
        requested_energy=requested_energy,
        parameter_budget=budget,
        fixed_parameter_count=fixed,
        factor_copies=copies,
    )


def plan_factorization(
    weight: Tensor,
    *,
    rank: int | None = None,
    energy: float | None = None,
    parameter_budget: int | None = None,
    fixed_parameter_count: int = 0,
    factor_copies: int = 1,
) -> FactorizationPlan:
    """Plan one deterministic SVD factorization without creating a module."""

    with torch.no_grad():
        analysis = _analyze_weight(weight)
        return _select_plan(
            analysis,
            rank=rank,
            energy=energy,
            parameter_budget=parameter_budget,
            fixed_parameter_count=fixed_parameter_count,
            factor_copies=factor_copies,
        )


def _factors_from_analysis(
    analysis: _SVDAnalysis,
    rank: int,
    source: Tensor,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        vocab_factor = analysis.left_vectors[:, :rank] * analysis.singular_values[
            :rank
        ].unsqueeze(0)
        hidden_factor = analysis.right_vectors_t[:rank]
        vocab_factor = vocab_factor.to(
            device=source.device,
            dtype=source.dtype,
        ).contiguous()
        hidden_factor = hidden_factor.to(
            device=source.device,
            dtype=source.dtype,
        ).contiguous()
        if not bool(torch.isfinite(vocab_factor).all()) or not bool(
            torch.isfinite(hidden_factor).all()
        ):
            raise FactorizationFiniteError(
                "Casting SVD factors to the source dtype produced NaN or infinity."
            )
        return vocab_factor, hidden_factor


def _frozen_parameter(value: Tensor, name: str) -> nn.Parameter:
    if isinstance(value, nn.Parameter) and not value.requires_grad:
        if value.grad is not None:
            raise FactorizationShapeError(f"{name} must not carry a gradient.")
        return value
    return nn.Parameter(value.detach().clone().contiguous(), requires_grad=False)


def _validate_factor_pair(
    vocab_factor: Any,
    hidden_factor: Any,
) -> tuple[Tensor, Tensor]:
    if not isinstance(vocab_factor, Tensor) or not isinstance(hidden_factor, Tensor):
        raise FactorizationShapeError("Both factors must be torch.Tensor values.")
    if vocab_factor.layout != torch.strided or hidden_factor.layout != torch.strided:
        raise FactorizationShapeError("Both factors must be dense strided tensors.")
    if vocab_factor.device.type == "meta" or hidden_factor.device.type == "meta":
        raise FactorizationShapeError("Factors cannot be meta tensors.")
    if vocab_factor.ndim != 2 or hidden_factor.ndim != 2:
        raise FactorizationShapeError("Both factors must be two-dimensional.")
    if min(vocab_factor.shape) <= 0 or min(hidden_factor.shape) <= 0:
        raise FactorizationShapeError("Factors cannot have an empty dimension.")
    if vocab_factor.shape[1] != hidden_factor.shape[0]:
        raise FactorizationShapeError(
            "vocab_factor columns must equal hidden_factor rows."
        )
    rank = int(vocab_factor.shape[1])
    if rank > min(vocab_factor.shape[0], hidden_factor.shape[1]):
        raise FactorizationRankError(
            "Factor rank cannot exceed min(vocab_size, hidden_size)."
        )
    if vocab_factor.dtype != hidden_factor.dtype:
        raise FactorizationShapeError("Factors must have the same dtype.")
    if vocab_factor.device != hidden_factor.device:
        raise FactorizationShapeError("Factors must be on the same device.")
    if not torch.is_floating_point(vocab_factor) or not torch.is_floating_point(
        hidden_factor
    ):
        raise FactorizationShapeError("Factors must have a real floating-point dtype.")
    if not bool(torch.isfinite(vocab_factor.detach()).all()) or not bool(
        torch.isfinite(hidden_factor.detach()).all()
    ):
        raise FactorizationFiniteError("Factors contain NaN or infinity.")
    return vocab_factor, hidden_factor


def _validate_embedding_module(module: Any) -> nn.Embedding:
    if not isinstance(module, nn.Embedding):
        raise FactorizationShapeError("embedding must be torch.nn.Embedding.")
    weight = _validate_matrix_metadata(module.weight, "embedding.weight")
    expected = (module.num_embeddings, module.embedding_dim)
    if tuple(weight.shape) != expected:
        raise FactorizationShapeError(f"embedding.weight must have shape {expected}.")
    return module


def _validate_head_module(module: Any) -> nn.Linear:
    if not isinstance(module, nn.Linear):
        raise FactorizationShapeError("lm_head must be torch.nn.Linear.")
    weight = _validate_matrix_metadata(module.weight, "lm_head.weight")
    expected = (module.out_features, module.in_features)
    if tuple(weight.shape) != expected:
        raise FactorizationShapeError(f"lm_head.weight must have shape {expected}.")
    if module.bias is not None:
        _validate_vector(module.bias, "lm_head.bias", length=module.out_features)
    return module


def _validated_optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FactorizationShapeError(f"{name} must be a finite positive float.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FactorizationShapeError(
            f"{name} must be a finite positive float."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise FactorizationShapeError(f"{name} must be a finite positive float.")
    return result


class FactorizedEmbedding(nn.Module):
    """An inference-only embedding represented as vocabulary and hidden factors."""

    vocab_factor: nn.Parameter
    hidden_factor: nn.Parameter

    def __init__(
        self,
        vocab_factor: Tensor,
        hidden_factor: Tensor,
        *,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        plan: FactorizationPlan | None = None,
    ) -> None:
        super().__init__()
        vocab_factor, hidden_factor = _validate_factor_pair(vocab_factor, hidden_factor)
        self.vocab_factor = _frozen_parameter(vocab_factor, "vocab_factor")
        self.hidden_factor = _frozen_parameter(hidden_factor, "hidden_factor")
        self.num_embeddings = int(vocab_factor.shape[0])
        self.embedding_dim = int(hidden_factor.shape[1])
        self.rank = int(vocab_factor.shape[1])
        if padding_idx is not None:
            if isinstance(padding_idx, bool) or not isinstance(padding_idx, int):
                raise FactorizationShapeError("padding_idx must be an integer or None.")
            if padding_idx < 0:
                padding_idx += self.num_embeddings
            if not 0 <= padding_idx < self.num_embeddings:
                raise FactorizationShapeError("padding_idx is outside the vocabulary.")
        if not isinstance(scale_grad_by_freq, bool) or not isinstance(sparse, bool):
            raise FactorizationShapeError(
                "scale_grad_by_freq and sparse must be booleans."
            )
        self.padding_idx = padding_idx
        self.max_norm = _validated_optional_positive_float(max_norm, "max_norm")
        validated_norm = _validated_optional_positive_float(norm_type, "norm_type")
        if validated_norm is None:  # pragma: no cover - norm_type is not optional
            raise FactorizationShapeError("norm_type must be a finite positive float.")
        self.norm_type = validated_norm
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.plan = plan

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        *,
        rank: int | None = None,
        energy: float | None = None,
        parameter_budget: int | None = None,
    ) -> FactorizedEmbedding:
        """Factor an ordinary embedding by detached CPU float64 SVD."""

        embedding = _validate_embedding_module(embedding)
        with torch.no_grad():
            analysis = _analyze_weight(embedding.weight, "embedding.weight")
            plan = _select_plan(
                analysis,
                rank=rank,
                energy=energy,
                parameter_budget=parameter_budget,
                fixed_parameter_count=0,
                factor_copies=1,
            )
            factors = _factors_from_analysis(analysis, plan.rank, embedding.weight)
            result = cls(
                *factors,
                padding_idx=embedding.padding_idx,
                max_norm=embedding.max_norm,
                norm_type=embedding.norm_type,
                scale_grad_by_freq=embedding.scale_grad_by_freq,
                sparse=embedding.sparse,
                plan=plan,
            )
            result.train(embedding.training)
            return result

    def forward(self, input_ids: Tensor) -> Tensor:
        rank_rows = F.embedding(
            input_ids,
            self.vocab_factor,
            padding_idx=self.padding_idx,
            max_norm=None,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )
        output = torch.matmul(rank_rows, self.hidden_factor)
        if self.max_norm is not None:
            norms = torch.linalg.vector_norm(
                output,
                ord=self.norm_type,
                dim=-1,
                keepdim=True,
            )
            scale = torch.where(
                norms > self.max_norm,
                self.max_norm / (norms + 1e-7),
                torch.ones_like(norms),
            )
            output = output * scale
        return output

    def materialize_weight(self) -> Tensor:
        """Expand the factors into a detached dense embedding matrix."""

        with torch.no_grad():
            weight = (self.vocab_factor @ self.hidden_factor).detach().contiguous()
            if not bool(torch.isfinite(weight).all()):
                raise FactorizationFiniteError(
                    "Materialized embedding weight contains NaN or infinity."
                )
            return weight

    def materialize(self) -> nn.Embedding:
        """Return an ordinary frozen ``nn.Embedding`` with expanded weight."""

        weight = self.materialize_weight()
        result = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            _weight=weight,
            _freeze=True,
            device=weight.device,
            dtype=weight.dtype,
        )
        result.train(self.training)
        return result

    def to_embedding(self) -> nn.Embedding:
        return self.materialize()

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, rank={self.rank}"
        )


class FactorizedLMHead(nn.Module):
    """An inference-only LM head that computes hidden -> rank -> vocabulary."""

    vocab_factor: nn.Parameter
    hidden_factor: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        vocab_factor: Tensor,
        hidden_factor: Tensor,
        bias: Tensor | None = None,
        *,
        plan: FactorizationPlan | None = None,
    ) -> None:
        super().__init__()
        vocab_factor, hidden_factor = _validate_factor_pair(vocab_factor, hidden_factor)
        self.vocab_factor = _frozen_parameter(vocab_factor, "vocab_factor")
        self.hidden_factor = _frozen_parameter(hidden_factor, "hidden_factor")
        self.out_features = int(vocab_factor.shape[0])
        self.in_features = int(hidden_factor.shape[1])
        self.rank = int(vocab_factor.shape[1])
        if bias is None:
            self.register_parameter("bias", None)
        else:
            bias = _validate_vector(bias, "bias", length=self.out_features)
            if bias.dtype != vocab_factor.dtype or bias.device != vocab_factor.device:
                raise FactorizationShapeError(
                    "bias must match the factor dtype and device."
                )
            self.bias = _frozen_parameter(bias, "bias")
        self.plan = plan

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int | None = None,
        energy: float | None = None,
        parameter_budget: int | None = None,
    ) -> FactorizedLMHead:
        """Factor an ordinary vocabulary projection by float64 SVD."""

        linear = _validate_head_module(linear)
        bias_count = 0 if linear.bias is None else linear.out_features
        with torch.no_grad():
            analysis = _analyze_weight(linear.weight, "lm_head.weight")
            plan = _select_plan(
                analysis,
                rank=rank,
                energy=energy,
                parameter_budget=parameter_budget,
                fixed_parameter_count=bias_count,
                factor_copies=1,
            )
            factors = _factors_from_analysis(analysis, plan.rank, linear.weight)
            result = cls(*factors, bias=linear.bias, plan=plan)
            result.train(linear.training)
            return result

    def forward(self, hidden_states: Tensor) -> Tensor:
        rank_activations = F.linear(hidden_states, self.hidden_factor)
        return F.linear(rank_activations, self.vocab_factor, self.bias)

    def materialize_weight(self) -> Tensor:
        """Expand the factors into a detached dense LM-head matrix."""

        with torch.no_grad():
            weight = (self.vocab_factor @ self.hidden_factor).detach().contiguous()
            if not bool(torch.isfinite(weight).all()):
                raise FactorizationFiniteError(
                    "Materialized LM-head weight contains NaN or infinity."
                )
            return weight

    def materialize(self) -> nn.Linear:
        """Return an ordinary frozen ``nn.Linear`` with expanded weight."""

        weight = self.materialize_weight()
        result = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            result.weight.copy_(weight)
            if self.bias is not None and result.bias is not None:
                result.bias.copy_(self.bias)
        result.requires_grad_(False)
        result.train(self.training)
        return result

    def to_linear(self) -> nn.Linear:
        return self.materialize()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )


def _same_tensor_storage(left: Tensor, right: Tensor) -> bool:
    if left is right:
        return True
    if (
        left.device != right.device
        or left.dtype != right.dtype
        or left.shape != right.shape
        or left.stride() != right.stride()
        or left.storage_offset() != right.storage_offset()
    ):
        return False
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _pair_budget_ranks(
    embedding_analysis: _SVDAnalysis,
    head_analysis: _SVDAnalysis,
    *,
    tied: bool,
    head_bias_parameters: int,
    parameter_budget: int,
) -> tuple[int, int]:
    budget = _parameter_budget(parameter_budget)
    per_rank = embedding_analysis.vocab_size + embedding_analysis.hidden_size
    available_slots = (budget - head_bias_parameters) // per_rank
    minimum_slots = 1 if tied else 2
    if available_slots < minimum_slots:
        minimum = head_bias_parameters + minimum_slots * per_rank
        raise FactorizationBudgetError(
            f"parameter_budget must be at least {minimum} for rank 1."
        )
    if tied:
        rank = min(embedding_analysis.max_rank, available_slots)
        return rank, rank

    embedding_rank = 1
    head_rank = 1
    remaining = min(
        available_slots - 2,
        embedding_analysis.max_rank + head_analysis.max_rank - 2,
    )
    largest = max(
        float(embedding_analysis.singular_values[0].item()),
        float(head_analysis.singular_values[0].item()),
    )
    candidates: list[tuple[float, int, int]] = []
    if largest > 0.0:
        for source_index, analysis in enumerate((embedding_analysis, head_analysis)):
            scaled = analysis.singular_values / largest
            for component in range(1, analysis.max_rank):
                candidates.append(
                    (-float(scaled[component].square().item()), source_index, component)
                )
        candidates.sort()
        for _, source_index, _ in candidates[:remaining]:
            if source_index == 0:
                embedding_rank += 1
            else:
                head_rank += 1
    else:
        add_to_embedding = min(remaining, embedding_analysis.max_rank - 1)
        embedding_rank += add_to_embedding
        head_rank += min(
            remaining - add_to_embedding,
            head_analysis.max_rank - 1,
        )
    return embedding_rank, head_rank


def _combined_retained_energy(
    embedding_analysis: _SVDAnalysis,
    embedding_rank: int,
    head_analysis: _SVDAnalysis,
    head_rank: int,
    *,
    tied: bool,
) -> float:
    if tied:
        return _retained_energy_fraction(
            embedding_analysis.singular_values, embedding_rank
        )
    largest = max(
        float(embedding_analysis.singular_values[0].item()),
        float(head_analysis.singular_values[0].item()),
    )
    if largest == 0.0:
        return 1.0
    embedding_energy = (embedding_analysis.singular_values / largest).square()
    head_energy = (head_analysis.singular_values / largest).square()
    total = embedding_energy.sum() + head_energy.sum()
    retained = embedding_energy[:embedding_rank].sum() + head_energy[:head_rank].sum()
    value = float((retained / total).item())
    if not math.isfinite(value):
        raise FactorizationFiniteError("Combined retained energy is non-finite.")
    return min(1.0, max(0.0, value))


class FactorizedVocabularyPair(nn.Module):
    """Factorized embedding/head modules plus pair-level storage evidence."""

    def __init__(
        self,
        embedding: FactorizedEmbedding,
        lm_head: FactorizedLMHead,
        *,
        tied: bool,
        plan: VocabularyFactorizationPlan,
    ) -> None:
        super().__init__()
        if not isinstance(embedding, FactorizedEmbedding) or not isinstance(
            lm_head, FactorizedLMHead
        ):
            raise FactorizationShapeError(
                "FactorizedVocabularyPair requires factorized modules."
            )
        if embedding.num_embeddings != lm_head.out_features:
            raise FactorizationShapeError(
                "Embedding vocabulary and LM-head output size must match."
            )
        if embedding.embedding_dim != lm_head.in_features:
            raise FactorizationShapeError(
                "Embedding and LM-head hidden widths must match."
            )
        if tied and not (
            embedding.vocab_factor is lm_head.vocab_factor
            and embedding.hidden_factor is lm_head.hidden_factor
        ):
            raise FactorizationShapeError(
                "A tied pair must share both factor Parameter objects."
            )
        self.embedding = embedding
        self.lm_head = lm_head
        self.tied = tied
        self.plan = plan

    def __iter__(self) -> Iterator[nn.Module]:
        yield self.embedding
        yield self.lm_head

    @property
    def unique_parameter_count(self) -> int:
        return exact_unique_parameter_count(self.embedding, self.lm_head)

    def parameter_report(self) -> dict[str, Any]:
        report = exact_parameter_report(self.embedding, self.lm_head)
        report["planned_unique_parameter_count"] = self.plan.unique_parameter_count
        report["tied"] = self.tied
        return report

    def materialize(self) -> tuple[nn.Embedding, nn.Linear]:
        """Expand both modules and preserve dense weight tying when present."""

        embedding = self.embedding.materialize()
        lm_head = self.lm_head.materialize()
        if self.tied:
            lm_head.weight = embedding.weight
        return embedding, lm_head

    def to_dense(self) -> tuple[nn.Embedding, nn.Linear]:
        return self.materialize()


FactorizedVocabPair = FactorizedVocabularyPair


def analyze_tied_vocabulary_modules(
    embedding: nn.Embedding,
    lm_head: nn.Linear,
) -> TiedVocabularySVDAnalysis:
    """Compute one reusable float64 SVD for a tied embedding/LM-head matrix."""

    embedding = _validate_embedding_module(embedding)
    lm_head = _validate_head_module(lm_head)
    if (
        embedding.num_embeddings != lm_head.out_features
        or embedding.embedding_dim != lm_head.in_features
    ):
        raise FactorizationShapeError(
            "Embedding and LM head must have matching vocabulary and hidden sizes."
        )
    if not _same_tensor_storage(embedding.weight, lm_head.weight):
        raise FactorizationShapeError(
            "Reusable tied-vocabulary analysis requires exact shared weight storage."
        )
    with torch.no_grad():
        analysis = _analyze_weight(embedding.weight, "embedding.weight")
    return TiedVocabularySVDAnalysis(
        _analysis=analysis,
        source_weight=embedding.weight,
        padding_idx=embedding.padding_idx,
        max_norm=embedding.max_norm,
        norm_type=float(embedding.norm_type),
        scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse,
        head_bias=lm_head.bias,
        embedding_training=embedding.training,
        head_training=lm_head.training,
    )


def _factorize_tied_from_precomputed_analysis(
    analysis: TiedVocabularySVDAnalysis,
    *,
    rank: int,
) -> FactorizedVocabularyPair:
    if not isinstance(analysis, TiedVocabularySVDAnalysis):
        raise FactorizationShapeError(
            "analysis must be a TiedVocabularySVDAnalysis."
        )
    selected_rank = _positive_int(
        rank,
        "rank",
        maximum=analysis.max_rank,
    )
    bias_count = 0 if analysis.head_bias is None else analysis.vocab_size
    with torch.no_grad():
        plan = _make_plan(
            analysis._analysis,
            rank=selected_rank,
            selection="rank",
        )
        vocab_factor, hidden_factor = _factors_from_analysis(
            analysis._analysis,
            selected_rank,
            analysis.source_weight,
        )
        embedding = FactorizedEmbedding(
            vocab_factor,
            hidden_factor,
            padding_idx=analysis.padding_idx,
            max_norm=analysis.max_norm,
            norm_type=analysis.norm_type,
            scale_grad_by_freq=analysis.scale_grad_by_freq,
            sparse=analysis.sparse,
            plan=plan,
        )
        embedding.train(analysis.embedding_training)
        lm_head = FactorizedLMHead(
            embedding.vocab_factor,
            embedding.hidden_factor,
            bias=analysis.head_bias,
            plan=plan,
        )
        lm_head.train(analysis.head_training)
        factor_count = plan.factor_parameter_count
        pair_plan = VocabularyFactorizationPlan(
            tied=True,
            embedding=plan,
            lm_head=plan,
            head_bias_parameters=bias_count,
            logical_parameter_count=2 * factor_count + bias_count,
            unique_parameter_count=factor_count + bias_count,
            dense_unique_parameter_count=(
                analysis.vocab_size * analysis.hidden_size + bias_count
            ),
            parameter_budget=None,
            budget_headroom=None,
            retained_energy_fraction=plan.retained_energy_fraction,
        )
        result = FactorizedVocabularyPair(
            embedding,
            lm_head,
            tied=True,
            plan=pair_plan,
        )
        if result.unique_parameter_count != pair_plan.unique_parameter_count:
            raise RuntimeError(
                "Precomputed tied-factor plan and physical parameters disagree."
            )
        return result


def factorize_embedding_and_lm_head(
    embedding: nn.Embedding,
    lm_head: nn.Linear,
    *,
    rank: int | None = None,
    energy: float | None = None,
    parameter_budget: int | None = None,
) -> FactorizedVocabularyPair:
    """Factor a compatible embedding/head pair, preserving true weight ties."""

    embedding = _validate_embedding_module(embedding)
    lm_head = _validate_head_module(lm_head)
    if (
        embedding.num_embeddings != lm_head.out_features
        or embedding.embedding_dim != lm_head.in_features
    ):
        raise FactorizationShapeError(
            "Embedding and LM head must have matching vocabulary and hidden sizes."
        )
    _validate_selector(rank, energy, parameter_budget)
    tied = _same_tensor_storage(embedding.weight, lm_head.weight)
    bias_count = 0 if lm_head.bias is None else lm_head.out_features

    with torch.no_grad():
        embedding_analysis = _analyze_weight(embedding.weight, "embedding.weight")
        head_analysis = (
            embedding_analysis
            if tied
            else _analyze_weight(lm_head.weight, "lm_head.weight")
        )

        if parameter_budget is not None:
            embedding_rank, head_rank = _pair_budget_ranks(
                embedding_analysis,
                head_analysis,
                tied=tied,
                head_bias_parameters=bias_count,
                parameter_budget=parameter_budget,
            )
            selection = "parameter_budget"
            requested_energy = None
            budget = _parameter_budget(parameter_budget)
        elif energy is not None:
            requested_energy = _energy_target(energy)
            embedding_rank = _rank_for_energy(
                embedding_analysis.singular_values, requested_energy
            )
            head_rank = (
                embedding_rank
                if tied
                else _rank_for_energy(head_analysis.singular_values, requested_energy)
            )
            selection = "energy"
            budget = None
        elif rank is not None:
            embedding_rank = _positive_int(
                rank, "rank", maximum=embedding_analysis.max_rank
            )
            head_rank = _positive_int(rank, "rank", maximum=head_analysis.max_rank)
            selection = "rank"
            requested_energy = None
            budget = None
        else:
            embedding_rank = embedding_analysis.max_rank
            head_rank = head_analysis.max_rank
            selection = "full_rank"
            requested_energy = None
            budget = None

        embedding_plan = _make_plan(
            embedding_analysis,
            rank=embedding_rank,
            selection=selection,
            requested_energy=requested_energy,
        )
        head_plan = (
            embedding_plan
            if tied
            else _make_plan(
                head_analysis,
                rank=head_rank,
                selection=selection,
                requested_energy=requested_energy,
            )
        )

        embedding_factors = _factors_from_analysis(
            embedding_analysis, embedding_rank, embedding.weight
        )
        factorized_embedding = FactorizedEmbedding(
            *embedding_factors,
            padding_idx=embedding.padding_idx,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
            plan=embedding_plan,
        )
        factorized_embedding.train(embedding.training)

        if tied:
            factorized_head = FactorizedLMHead(
                factorized_embedding.vocab_factor,
                factorized_embedding.hidden_factor,
                bias=lm_head.bias,
                plan=head_plan,
            )
        else:
            head_factors = _factors_from_analysis(
                head_analysis, head_rank, lm_head.weight
            )
            factorized_head = FactorizedLMHead(
                *head_factors,
                bias=lm_head.bias,
                plan=head_plan,
            )
        factorized_head.train(lm_head.training)

        factor_cost = embedding_plan.factor_parameter_count
        logical_count = factor_cost + head_plan.factor_parameter_count + bias_count
        unique_count = factor_cost + bias_count if tied else logical_count
        dense_unique = (
            embedding.num_embeddings * embedding.embedding_dim * (1 if tied else 2)
            + bias_count
        )
        pair_plan = VocabularyFactorizationPlan(
            tied=tied,
            embedding=embedding_plan,
            lm_head=head_plan,
            head_bias_parameters=bias_count,
            logical_parameter_count=logical_count,
            unique_parameter_count=unique_count,
            dense_unique_parameter_count=dense_unique,
            parameter_budget=budget,
            budget_headroom=None if budget is None else budget - unique_count,
            retained_energy_fraction=_combined_retained_energy(
                embedding_analysis,
                embedding_rank,
                head_analysis,
                head_rank,
                tied=tied,
            ),
        )
        result = FactorizedVocabularyPair(
            factorized_embedding,
            factorized_head,
            tied=tied,
            plan=pair_plan,
        )
        actual_count = result.unique_parameter_count
        if actual_count != unique_count:
            raise RuntimeError(
                "Planned and actual unique factor parameter counts disagree."
            )
        return result


def plan_vocabulary_factorization(
    embedding: nn.Embedding,
    lm_head: nn.Linear,
    *,
    rank: int | None = None,
    energy: float | None = None,
    parameter_budget: int | None = None,
) -> VocabularyFactorizationPlan:
    """Return the exact pair plan produced by the deterministic factorizer."""

    return factorize_embedding_and_lm_head(
        embedding,
        lm_head,
        rank=rank,
        energy=energy,
        parameter_budget=parameter_budget,
    ).plan


def _exact_parameter_key(parameter: nn.Parameter) -> tuple[Any, ...]:
    return (
        parameter.device.type,
        parameter.device.index,
        str(parameter.dtype),
        parameter.untyped_storage().data_ptr(),
        parameter.storage_offset(),
        tuple(parameter.shape),
        tuple(parameter.stride()),
    )


def exact_parameter_report(*modules: nn.Module) -> dict[str, Any]:
    """Count logical references and exact unique Parameter storage aliases."""

    if not modules:
        raise FactorizationShapeError("At least one module is required.")
    logical_count = 0
    logical_tensors = 0
    unique: dict[tuple[Any, ...], nn.Parameter] = {}
    module_counts: list[dict[str, Any]] = []
    for index, module in enumerate(modules):
        if not isinstance(module, nn.Module):
            raise FactorizationShapeError("All values must be torch.nn.Module objects.")
        module_count = 0
        module_tensors = 0
        for _, parameter in module.named_parameters(
            recurse=True, remove_duplicate=False
        ):
            count = int(parameter.numel())
            logical_count += count
            module_count += count
            logical_tensors += 1
            module_tensors += 1
            key = _exact_parameter_key(parameter)
            existing = unique.get(key)
            if existing is not None and existing.numel() != parameter.numel():
                raise RuntimeError("Aliased parameter storage has inconsistent shapes.")
            unique[key] = parameter
        module_counts.append(
            {
                "module_index": index,
                "module_type": type(module).__name__,
                "logical_parameter_count": module_count,
                "logical_parameter_tensors": module_tensors,
            }
        )
    unique_count = sum(int(parameter.numel()) for parameter in unique.values())
    trainable_count = sum(
        int(parameter.numel())
        for parameter in unique.values()
        if parameter.requires_grad
    )
    return {
        "logical_parameter_count": logical_count,
        "unique_parameter_count": unique_count,
        "shared_parameter_references": logical_count - unique_count,
        "logical_parameter_tensors": logical_tensors,
        "unique_parameter_tensors": len(unique),
        "trainable_unique_parameter_count": trainable_count,
        "all_parameters_frozen": trainable_count == 0,
        "modules": module_counts,
    }


def exact_unique_parameter_count(*modules: nn.Module) -> int:
    """Return the exact number of unique parameter elements across modules."""

    return int(exact_parameter_report(*modules)["unique_parameter_count"])


def materialize_embedding_and_lm_head(
    embedding: FactorizedEmbedding | FactorizedVocabularyPair,
    lm_head: FactorizedLMHead | None = None,
) -> tuple[nn.Embedding, nn.Linear]:
    """Expand a pair (or two factorized modules) back to dense modules."""

    if isinstance(embedding, FactorizedVocabularyPair):
        if lm_head is not None:
            raise FactorizationShapeError(
                "lm_head must be omitted when materializing a pair."
            )
        return embedding.materialize()
    if not isinstance(embedding, FactorizedEmbedding) or not isinstance(
        lm_head, FactorizedLMHead
    ):
        raise FactorizationShapeError(
            "Expected a FactorizedVocabularyPair or compatible factorized modules."
        )
    dense_embedding = embedding.materialize()
    dense_head = lm_head.materialize()
    tied = (
        embedding.vocab_factor is lm_head.vocab_factor
        and embedding.hidden_factor is lm_head.hidden_factor
    )
    if tied:
        dense_head.weight = dense_embedding.weight
    return dense_embedding, dense_head


factorize_vocabulary_modules = factorize_embedding_and_lm_head
factorize_vocab_pair = factorize_embedding_and_lm_head
materialize_vocabulary_modules = materialize_embedding_and_lm_head
unique_parameter_report = exact_parameter_report
count_unique_parameters = exact_unique_parameter_count


__all__ = [
    "FactorizationBudgetError",
    "FactorizationFiniteError",
    "FactorizationPlan",
    "FactorizationRankError",
    "FactorizationShapeError",
    "FactorizedEmbedding",
    "FactorizedLMHead",
    "FactorizedVocabPair",
    "FactorizedVocabularyPair",
    "TiedVocabularySVDAnalysis",
    "VocabularyFactorizationError",
    "VocabularyFactorizationPlan",
    "analyze_tied_vocabulary_modules",
    "count_unique_parameters",
    "exact_parameter_report",
    "exact_unique_parameter_count",
    "factorize_embedding_and_lm_head",
    "factorize_vocab_pair",
    "factorize_vocabulary_modules",
    "materialize_embedding_and_lm_head",
    "materialize_vocabulary_modules",
    "plan_factorization",
    "plan_vocabulary_factorization",
    "unique_parameter_report",
]
