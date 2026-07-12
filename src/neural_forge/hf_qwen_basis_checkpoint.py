"""Reloadable standalone Qwen3 depth-basis checkpoints.

The artifact is intentionally structural rather than a serialized Python object:
JSON describes the Qwen3 and factorized topology, while safetensors (or a small
raw fallback) stores each physical tensor exactly once.  Reload constructs the
factorized topology on the meta device before reading tensor payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.hf_arch_layout import build_decoder_layout, resolve_module
from neural_forge.hf_basis_linear import (
    DEFAULT_FEATURE_CHUNK_SIZE,
    DepthBasisLinear,
)
from neural_forge.hf_depth_basis_ir import (
    PROJECTION_ROLES,
    DepthBasisRegistry,
    FixedLayerDepthBasisLinear,
    compile_hf_depth_basis_ir,
)
from neural_forge.hf_factorized_vocab import (
    FactorizedEmbedding,
    FactorizedLMHead,
    factorize_embedding_and_lm_head,
)


CHECKPOINT_SCHEMA = "neural-forge-qwen3-depth-basis-checkpoint-v1"
MANIFEST_FILENAME = "neural_forge_basis_manifest.json"
CONFIG_FILENAME = "config.json"
GENERATION_CONFIG_FILENAME = "generation_config.json"
SAFETENSORS_FILENAME = "model.safetensors"
RAW_TENSORS_FILENAME = "model.nfraw"
DEFAULT_FULL_VOCAB_SVD_MAX_ELEMENTS = 8_000_000
DEFAULT_VOCAB_ROW_CHUNK_SIZE = 2_048
DEFAULT_VOCAB_OVERSAMPLING = 8
DEFAULT_VOCAB_POWER_ITERATIONS = 1
DEFAULT_VOCAB_RANDOM_SEED = 0
_REGISTRY_NAME = "_depth_basis_ir"
_MANIFEST_HASH_KEY = "manifest_sha256"
_ROLE_PATH_PARTS = {
    "q": ("self_attn", "q_proj"),
    "k": ("self_attn", "k_proj"),
    "v": ("self_attn", "v_proj"),
    "o": ("self_attn", "o_proj"),
    "gate": ("mlp", "gate_proj"),
    "up": ("mlp", "up_proj"),
    "down": ("mlp", "down_proj"),
}
_SUPPORTED_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


class Qwen3BasisCheckpointError(ValueError):
    """Base class for malformed plans, models, or artifacts."""


class Qwen3BasisBudgetError(Qwen3BasisCheckpointError):
    """A requested cap cannot contain the requested explicit rank plan."""


class Qwen3BasisFormatError(Qwen3BasisCheckpointError):
    """A checkpoint artifact is incomplete, inconsistent, or corrupted."""


@dataclass(frozen=True, slots=True)
class Qwen3BasisRankPlan:
    """Explicit ranks for every depth role and the tied vocabulary matrix."""

    q: int
    k: int
    v: int
    o: int
    gate: int
    up: int
    down: int
    vocab: int

    @property
    def role_ranks(self) -> dict[str, int]:
        return {role: int(getattr(self, role)) for role in PROJECTION_ROLES}

    @property
    def vocab_rank(self) -> int:
        return self.vocab

    def as_dict(self) -> dict[str, int]:
        return {**self.role_ranks, "vocab": self.vocab}


@dataclass(frozen=True, slots=True)
class Qwen3BasisCheckpointPlan:
    """Closed-form Qwen3 basis plan with physical and logical accounting."""

    ranks: Qwen3BasisRankPlan
    parameter_budget: int | None
    budget_headroom: int | None
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    query_width: int
    key_value_width: int
    head_dim: int
    vocab_size: int
    attention_bias: bool
    role_parameter_counts: tuple[tuple[str, int], ...]
    fixed_parameter_count: int
    vocabulary_unique_parameter_count: int
    vocabulary_logical_parameter_count: int
    checkpoint_unique_parameter_count: int
    checkpoint_logical_parameter_count: int
    dense_unique_parameter_count: int
    dense_logical_parameter_count: int

    @property
    def unique_parameter_count(self) -> int:
        return self.checkpoint_unique_parameter_count

    @property
    def logical_parameter_count(self) -> int:
        return self.checkpoint_logical_parameter_count

    @property
    def role_counts(self) -> dict[str, int]:
        return dict(self.role_parameter_counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank_plan": self.ranks.as_dict(),
            "parameter_budget": self.parameter_budget,
            "budget_headroom": self.budget_headroom,
            "dimensions": {
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "query_width": self.query_width,
                "key_value_width": self.key_value_width,
                "head_dim": self.head_dim,
                "vocab_size": self.vocab_size,
                "attention_bias": self.attention_bias,
            },
            "role_unique_parameter_counts": self.role_counts,
            "fixed_parameter_count": self.fixed_parameter_count,
            "vocabulary_unique_parameter_count": (
                self.vocabulary_unique_parameter_count
            ),
            "vocabulary_logical_parameter_count": (
                self.vocabulary_logical_parameter_count
            ),
            "unique_parameter_count": self.checkpoint_unique_parameter_count,
            "logical_parameter_count": self.checkpoint_logical_parameter_count,
            "dense_equivalent_unique_parameter_count": (
                self.dense_unique_parameter_count
            ),
            "dense_equivalent_logical_parameter_count": (
                self.dense_logical_parameter_count
            ),
            "tied_vocabulary": True,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class _Qwen3Dimensions:
    layers: int
    hidden: int
    intermediate: int
    query_width: int
    key_value_width: int
    head_dim: int
    vocab: int
    attention_bias: bool

    @property
    def role_features(self) -> dict[str, int]:
        query_bias = self.query_width if self.attention_bias else 0
        key_value_bias = self.key_value_width if self.attention_bias else 0
        output_bias = self.hidden if self.attention_bias else 0
        return {
            "q": self.query_width * self.hidden + query_bias,
            "k": self.key_value_width * self.hidden + key_value_bias,
            "v": self.key_value_width * self.hidden + key_value_bias,
            "o": self.hidden * self.query_width + output_bias,
            "gate": self.intermediate * self.hidden,
            "up": self.intermediate * self.hidden,
            "down": self.hidden * self.intermediate,
        }

    @property
    def role_shapes(self) -> dict[str, tuple[int, int, bool]]:
        return {
            "q": (self.hidden, self.query_width, self.attention_bias),
            "k": (self.hidden, self.key_value_width, self.attention_bias),
            "v": (self.hidden, self.key_value_width, self.attention_bias),
            "o": (self.query_width, self.hidden, self.attention_bias),
            "gate": (self.hidden, self.intermediate, False),
            "up": (self.hidden, self.intermediate, False),
            "down": (self.intermediate, self.hidden, False),
        }

    @property
    def fixed_parameters(self) -> int:
        per_layer_norms = 2 * self.hidden + 2 * self.head_dim
        return self.layers * per_layer_norms + self.hidden


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Qwen3BasisCheckpointError(f"{name} must be a positive integer.")
    return value


def _optional_budget(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Qwen3BasisBudgetError(
            "parameter_budget must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Qwen3BasisCheckpointError(
            f"{name} must be a non-negative integer."
        )
    return value


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _qwen3_dimensions(config: Any) -> _Qwen3Dimensions:
    model_type = _config_value(config, "model_type")
    if model_type != "qwen3":
        raise Qwen3BasisCheckpointError(
            f"Expected a Qwen3 config, got model_type={model_type!r}."
        )
    if _config_value(config, "tie_word_embeddings", False) is not True:
        raise Qwen3BasisCheckpointError(
            "Qwen3 basis checkpoints require tie_word_embeddings=True."
        )
    attention_bias = bool(_config_value(config, "attention_bias", False))
    if attention_bias:
        raise Qwen3BasisCheckpointError(
            "Qwen3 basis checkpoints require attention_bias=False because the "
            "shared Qwen3 depth IR supports bias-free attention only."
        )

    hidden = _positive_int(_config_value(config, "hidden_size"), "hidden_size")
    heads = _positive_int(
        _config_value(config, "num_attention_heads"), "num_attention_heads"
    )
    key_value_heads = _positive_int(
        _config_value(config, "num_key_value_heads"), "num_key_value_heads"
    )
    raw_head_dim = _config_value(config, "head_dim")
    head_dim = (
        hidden // heads
        if raw_head_dim is None
        else _positive_int(raw_head_dim, "head_dim")
    )
    if hidden % heads != 0 and raw_head_dim is None:
        raise Qwen3BasisCheckpointError(
            "hidden_size must be divisible by num_attention_heads when head_dim "
            "is omitted."
        )
    return _Qwen3Dimensions(
        layers=_positive_int(
            _config_value(config, "num_hidden_layers"), "num_hidden_layers"
        ),
        hidden=hidden,
        intermediate=_positive_int(
            _config_value(config, "intermediate_size"), "intermediate_size"
        ),
        query_width=heads * head_dim,
        key_value_width=key_value_heads * head_dim,
        head_dim=head_dim,
        vocab=_positive_int(_config_value(config, "vocab_size"), "vocab_size"),
        attention_bias=attention_bias,
    )


def _mapping_rank_plan(value: Mapping[str, Any]) -> tuple[dict[str, int], int]:
    allowed = set(PROJECTION_ROLES) | {"vocab", "vocab_rank"}
    unknown = set(value) - allowed
    if unknown:
        raise Qwen3BasisCheckpointError(
            f"rank_plan has unsupported keys {sorted(unknown)!r}."
        )
    missing = set(PROJECTION_ROLES) - set(value)
    if missing:
        raise Qwen3BasisCheckpointError(
            f"rank_plan is missing roles {sorted(missing)!r}."
        )
    if "vocab" in value and "vocab_rank" in value:
        raise Qwen3BasisCheckpointError(
            "rank_plan cannot contain both vocab and vocab_rank."
        )
    raw_vocab = value.get("vocab", value.get("vocab_rank"))
    if raw_vocab is None:
        raise Qwen3BasisCheckpointError("rank_plan must include vocab.")
    roles = {
        role: _positive_int(value[role], f"rank_plan[{role!r}]")
        for role in PROJECTION_ROLES
    }
    return roles, _positive_int(raw_vocab, "rank_plan['vocab']")


def _resolve_rank_plan(
    dimensions: _Qwen3Dimensions,
    *,
    rank_plan: Qwen3BasisRankPlan | Mapping[str, Any] | None,
    role_ranks: Mapping[str, Any] | None,
    vocab_rank: int | None,
    parameter_budget: int | None,
) -> Qwen3BasisRankPlan:
    if rank_plan is not None:
        if role_ranks is not None or vocab_rank is not None:
            raise Qwen3BasisCheckpointError(
                "rank_plan cannot be combined with role_ranks or vocab_rank."
            )
        if isinstance(rank_plan, Qwen3BasisRankPlan):
            roles = rank_plan.role_ranks
            selected_vocab_rank = rank_plan.vocab
        elif isinstance(rank_plan, Mapping):
            roles, selected_vocab_rank = _mapping_rank_plan(rank_plan)
        else:
            raise Qwen3BasisCheckpointError(
                "rank_plan must be Qwen3BasisRankPlan or a mapping."
            )
    else:
        if role_ranks is None:
            if parameter_budget is None:
                raise Qwen3BasisCheckpointError(
                    "Provide an explicit rank_plan or role_ranks."
                )
            roles = dict.fromkeys(PROJECTION_ROLES, 1)
        else:
            if not isinstance(role_ranks, Mapping):
                raise Qwen3BasisCheckpointError("role_ranks must be a mapping.")
            unknown = set(role_ranks) - set(PROJECTION_ROLES)
            missing = set(PROJECTION_ROLES) - set(role_ranks)
            if unknown or missing:
                raise Qwen3BasisCheckpointError(
                    "role_ranks must contain exactly q/k/v/o/gate/up/down; "
                    f"missing={sorted(missing)!r}, unknown={sorted(unknown)!r}."
                )
            roles = {
                role: _positive_int(role_ranks[role], f"role_ranks[{role!r}]")
                for role in PROJECTION_ROLES
            }
        if vocab_rank is None:
            if parameter_budget is None:
                raise Qwen3BasisCheckpointError(
                    "vocab_rank is required when no parameter_budget is supplied."
                )
            base = dimensions.fixed_parameters + sum(
                roles[role]
                * (dimensions.role_features[role] + dimensions.layers)
                for role in PROJECTION_ROLES
            )
            per_vocab_rank = dimensions.vocab + dimensions.hidden
            selected_vocab_rank = min(
                dimensions.vocab,
                dimensions.hidden,
                (parameter_budget - base) // per_vocab_rank,
            )
            if selected_vocab_rank < 1:
                minimum = base + per_vocab_rank
                raise Qwen3BasisBudgetError(
                    "parameter_budget cannot contain the role plan and tied "
                    f"vocabulary rank 1; minimum is {minimum}."
                )
        else:
            selected_vocab_rank = _positive_int(vocab_rank, "vocab_rank")

    for role, selected_rank in roles.items():
        if selected_rank > dimensions.layers:
            raise Qwen3BasisCheckpointError(
                f"Rank for role {role!r} cannot exceed depth {dimensions.layers}."
            )
    maximum_vocab_rank = min(dimensions.vocab, dimensions.hidden)
    if selected_vocab_rank > maximum_vocab_rank:
        raise Qwen3BasisCheckpointError(
            f"vocab rank cannot exceed {maximum_vocab_rank}."
        )
    return Qwen3BasisRankPlan(
        **roles,
        vocab=selected_vocab_rank,
    )


def plan_qwen3_basis_checkpoint(
    config: Any,
    *,
    rank_plan: Qwen3BasisRankPlan | Mapping[str, Any] | None = None,
    role_ranks: Mapping[str, int] | None = None,
    vocab_rank: int | None = None,
    parameter_budget: int | None = None,
) -> Qwen3BasisCheckpointPlan:
    """Return an exact deterministic plan without materializing a Qwen model.

    When ``vocab_rank`` is omitted under a cap, the largest tied vocabulary rank
    that fits after the explicit role ranks is selected.  The result is always
    at or below ``parameter_budget`` or the function raises.
    """

    dimensions = _qwen3_dimensions(config)
    budget = _optional_budget(parameter_budget)
    ranks = _resolve_rank_plan(
        dimensions,
        rank_plan=rank_plan,
        role_ranks=role_ranks,
        vocab_rank=vocab_rank,
        parameter_budget=budget,
    )
    role_counts = tuple(
        (
            role,
            ranks.role_ranks[role]
            * (dimensions.role_features[role] + dimensions.layers),
        )
        for role in PROJECTION_ROLES
    )
    vocab_unique = ranks.vocab * (dimensions.vocab + dimensions.hidden)
    vocab_logical = 2 * vocab_unique
    checkpoint_unique = (
        dimensions.fixed_parameters
        + sum(count for _, count in role_counts)
        + vocab_unique
    )
    checkpoint_logical = checkpoint_unique + vocab_unique
    dense_roles = dimensions.layers * sum(dimensions.role_features.values())
    dense_vocab = dimensions.vocab * dimensions.hidden
    dense_unique = dimensions.fixed_parameters + dense_roles + dense_vocab
    dense_logical = dense_unique + dense_vocab
    if budget is not None and checkpoint_unique > budget:
        raise Qwen3BasisBudgetError(
            "The explicit rank plan exceeds parameter_budget by "
            f"{checkpoint_unique - budget} parameters."
        )
    return Qwen3BasisCheckpointPlan(
        ranks=ranks,
        parameter_budget=budget,
        budget_headroom=None if budget is None else budget - checkpoint_unique,
        num_hidden_layers=dimensions.layers,
        hidden_size=dimensions.hidden,
        intermediate_size=dimensions.intermediate,
        query_width=dimensions.query_width,
        key_value_width=dimensions.key_value_width,
        head_dim=dimensions.head_dim,
        vocab_size=dimensions.vocab,
        attention_bias=dimensions.attention_bias,
        role_parameter_counts=role_counts,
        fixed_parameter_count=dimensions.fixed_parameters,
        vocabulary_unique_parameter_count=vocab_unique,
        vocabulary_logical_parameter_count=vocab_logical,
        checkpoint_unique_parameter_count=checkpoint_unique,
        checkpoint_logical_parameter_count=checkpoint_logical,
        dense_unique_parameter_count=dense_unique,
        dense_logical_parameter_count=dense_logical,
    )


def count_qwen3_basis_parameters(config: Any, **kwargs: Any) -> int:
    """Return the physical parameter count for a Qwen3 basis plan."""

    return plan_qwen3_basis_checkpoint(config, **kwargs).unique_parameter_count


def _exact_storage_key(tensor: Tensor) -> tuple[Any, ...]:
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise Qwen3BasisCheckpointError(
            "Checkpoint tensors must be dense, strided, and materialized."
        )
    return (
        tensor.device.type,
        tensor.device.index,
        str(tensor.dtype),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _model_parameter_counts(model: nn.Module) -> tuple[int, int]:
    logical = 0
    unique: dict[tuple[Any, ...], nn.Parameter] = {}
    for _, parameter in model.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        logical += int(parameter.numel())
        unique[_exact_storage_key(parameter)] = parameter
    return logical, sum(int(parameter.numel()) for parameter in unique.values())


def _same_storage(left: Tensor, right: Tensor) -> bool:
    return _exact_storage_key(left) == _exact_storage_key(right)


def _prepared_rank_plan(model: nn.Module) -> Qwen3BasisRankPlan | None:
    registry = getattr(model, _REGISTRY_NAME, None)
    embedding = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    markers = (
        isinstance(registry, DepthBasisRegistry),
        isinstance(embedding, FactorizedEmbedding),
        isinstance(lm_head, FactorizedLMHead),
    )
    if not any(markers):
        return None
    if not all(markers):
        raise Qwen3BasisCheckpointError(
            "Model has a partial basis topology and cannot be exported."
        )
    if tuple(registry.representations) != PROJECTION_ROLES:
        raise Qwen3BasisCheckpointError("Depth registry has an invalid role set.")
    if not (
        embedding.vocab_factor is lm_head.vocab_factor
        and embedding.hidden_factor is lm_head.hidden_factor
    ):
        raise Qwen3BasisCheckpointError(
            "Prepared vocabulary factors are not tied by Parameter identity."
        )
    roles = {
        role: int(registry.representations[role].rank)
        for role in PROJECTION_ROLES
    }
    return Qwen3BasisRankPlan(**roles, vocab=int(embedding.rank))


def _assert_plan_counts(
    model: nn.Module,
    plan: Qwen3BasisCheckpointPlan,
) -> None:
    logical, unique = _model_parameter_counts(model)
    if logical != plan.checkpoint_logical_parameter_count:
        raise Qwen3BasisCheckpointError(
            "Prepared logical parameter count disagrees with the plan: "
            f"expected {plan.checkpoint_logical_parameter_count}, got {logical}."
        )
    if unique != plan.checkpoint_unique_parameter_count:
        raise Qwen3BasisCheckpointError(
            "Prepared unique parameter count disagrees with the plan: "
            f"expected {plan.checkpoint_unique_parameter_count}, got {unique}."
        )


def _cpu_float64_weight_chunk(
    weight: Tensor,
    start: int,
    stop: int,
) -> Tensor:
    chunk = weight[start:stop].detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(chunk).all()):
        raise Qwen3BasisCheckpointError(
            "Vocabulary weight contains NaN or infinity."
        )
    return chunk.contiguous()


def _chunked_randomized_vocabulary_factors(
    weight: Tensor,
    *,
    rank: int,
    row_chunk_size: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Fit deterministic truncated factors without a full float64 weight copy."""

    if not isinstance(weight, Tensor) or weight.ndim != 2:
        raise Qwen3BasisCheckpointError(
            "Vocabulary weight must be a two-dimensional tensor."
        )
    if weight.layout != torch.strided or weight.device.type == "meta":
        raise Qwen3BasisCheckpointError(
            "Vocabulary weight must be dense, strided, and materialized."
        )
    if not torch.is_floating_point(weight) or torch.is_complex(weight):
        raise Qwen3BasisCheckpointError(
            "Vocabulary weight must have a real floating dtype."
        )
    vocab_size, hidden_size = map(int, weight.shape)
    maximum_rank = min(vocab_size, hidden_size)
    selected_rank = _positive_int(rank, "vocab_rank")
    if selected_rank > maximum_rank:
        raise Qwen3BasisCheckpointError(
            f"vocab_rank cannot exceed {maximum_rank}."
        )
    chunk_rows = _positive_int(row_chunk_size, "vocab_row_chunk_size")
    extra = _nonnegative_int(oversampling, "vocab_oversampling")
    iterations = _nonnegative_int(
        power_iterations,
        "vocab_power_iterations",
    )
    random_seed = _nonnegative_int(seed, "vocab_seed")
    sample_rank = min(maximum_rank, selected_rank + extra)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    omega = torch.randn(
        (hidden_size, sample_rank),
        dtype=torch.float64,
        generator=generator,
    )
    omega, omega_qr_factor = torch.linalg.qr(omega, mode="reduced")
    del omega_qr_factor
    projected = torch.empty(
        (vocab_size, sample_rank),
        dtype=torch.float64,
    )
    source_squared_norm = 0.0
    with torch.no_grad():
        for start in range(0, vocab_size, chunk_rows):
            stop = min(vocab_size, start + chunk_rows)
            chunk = _cpu_float64_weight_chunk(weight, start, stop)
            projected[start:stop].copy_(chunk @ omega)
            source_squared_norm += float(chunk.square().sum().item())
        del chunk
        q_basis, qr_factor = torch.linalg.qr(projected, mode="reduced")
        del projected, omega, qr_factor

        for _ in range(iterations):
            transpose_projection = torch.zeros(
                (hidden_size, sample_rank),
                dtype=torch.float64,
            )
            for start in range(0, vocab_size, chunk_rows):
                stop = min(vocab_size, start + chunk_rows)
                chunk = _cpu_float64_weight_chunk(weight, start, stop)
                transpose_projection.add_(chunk.T @ q_basis[start:stop])
            del chunk
            transpose_basis, qr_factor = torch.linalg.qr(
                transpose_projection,
                mode="reduced",
            )
            del transpose_projection, q_basis, qr_factor

            projected = torch.empty(
                (vocab_size, sample_rank),
                dtype=torch.float64,
            )
            for start in range(0, vocab_size, chunk_rows):
                stop = min(vocab_size, start + chunk_rows)
                chunk = _cpu_float64_weight_chunk(weight, start, stop)
                projected[start:stop].copy_(chunk @ transpose_basis)
            del chunk
            q_basis, qr_factor = torch.linalg.qr(projected, mode="reduced")
            del projected, transpose_basis, qr_factor

        compressed = torch.zeros(
            (sample_rank, hidden_size),
            dtype=torch.float64,
        )
        for start in range(0, vocab_size, chunk_rows):
            stop = min(vocab_size, start + chunk_rows)
            chunk = _cpu_float64_weight_chunk(weight, start, stop)
            compressed.add_(q_basis[start:stop].T @ chunk)
        del chunk
        small_left, singular_values, right_vectors_t = torch.linalg.svd(
            compressed,
            full_matrices=False,
        )
        if not all(
            bool(torch.isfinite(value).all())
            for value in (small_left, singular_values, right_vectors_t)
        ):
            raise Qwen3BasisCheckpointError(
                "Chunked randomized vocabulary SVD produced NaN or infinity."
            )

        transform = small_left[:, :selected_rank] * singular_values[
            :selected_rank
        ].unsqueeze(0)
        vocab_factor_cpu = torch.empty(
            (vocab_size, selected_rank),
            dtype=weight.dtype,
        )
        for start in range(0, vocab_size, chunk_rows):
            stop = min(vocab_size, start + chunk_rows)
            factor_chunk = q_basis[start:stop] @ transform
            vocab_factor_cpu[start:stop].copy_(factor_chunk.to(weight.dtype))
        hidden_factor_cpu = right_vectors_t[:selected_rank].to(weight.dtype)

        pivot_rows = torch.argmax(torch.abs(vocab_factor_cpu), dim=0)
        columns = torch.arange(selected_rank, dtype=torch.long)
        pivots = vocab_factor_cpu[pivot_rows, columns]
        signs = torch.where(
            pivots < 0,
            -torch.ones_like(pivots),
            torch.ones_like(pivots),
        )
        vocab_factor_cpu.mul_(signs.unsqueeze(0))
        hidden_factor_cpu.mul_(signs.unsqueeze(1))
        if not bool(torch.isfinite(vocab_factor_cpu).all()) or not bool(
            torch.isfinite(hidden_factor_cpu).all()
        ):
            raise Qwen3BasisCheckpointError(
                "Casting randomized vocabulary factors produced NaN or infinity."
            )

    retained_energy = float(
        singular_values[:selected_rank].square().sum().item()
    )
    retained_fraction = (
        1.0
        if source_squared_norm == 0.0
        else min(1.0, max(0.0, retained_energy / source_squared_norm))
    )
    tall_elements = vocab_size * sample_rank
    chunk_elements = min(chunk_rows, vocab_size) * hidden_size
    small_elements = hidden_size * sample_rank
    explicit_peak_elements = max(
        2 * tall_elements,
        tall_elements + chunk_elements + small_elements,
    )
    report = {
        "algorithm": "deterministic-chunked-randomized-svd-v1",
        "deterministic": True,
        "rank": selected_rank,
        "sample_rank": sample_rank,
        "oversampling": extra,
        "power_iterations": iterations,
        "seed": random_seed,
        "row_chunk_size": chunk_rows,
        "source_shape": [vocab_size, hidden_size],
        "source_matrix_elements": int(weight.numel()),
        "source_dtype": str(weight.dtype),
        "compute_device": "cpu",
        "compute_dtype": str(torch.float64),
        "source_passes": 2 + 2 * iterations,
        "tall_workspace_shape": [vocab_size, sample_rank],
        "chunk_workspace_shape": [min(chunk_rows, vocab_size), hidden_size],
        "estimated_explicit_peak_float64_elements": explicit_peak_elements,
        "estimated_explicit_peak_float64_bytes": 8 * explicit_peak_elements,
        "peak_estimate_excludes_lapack_workspace": True,
        "materialized_full_float64_source_matrix": False,
        "materialized_full_left_singular_matrix": False,
        "retained_projected_energy_fraction": retained_fraction,
        "projected_singular_values": [
            float(value) for value in singular_values.tolist()
        ],
    }
    return (
        vocab_factor_cpu.to(device=weight.device),
        hidden_factor_cpu.to(device=weight.device),
        report,
    )


def _factorize_tied_vocabulary(
    embedding: nn.Embedding,
    lm_head: nn.Linear,
    *,
    rank: int,
    algorithm: str,
    full_svd_max_elements: int,
    row_chunk_size: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
) -> tuple[FactorizedEmbedding, FactorizedLMHead, dict[str, Any]]:
    if algorithm not in {"auto", "full", "randomized"}:
        raise Qwen3BasisCheckpointError(
            "vocab_algorithm must be 'auto', 'full', or 'randomized'."
        )
    full_limit = _positive_int(
        full_svd_max_elements,
        "full_vocab_svd_max_elements",
    )
    source_elements = int(embedding.weight.numel())
    selected_algorithm = algorithm
    if selected_algorithm == "auto":
        selected_algorithm = (
            "full" if source_elements <= full_limit else "randomized"
        )
    if selected_algorithm == "full":
        if source_elements > full_limit:
            raise Qwen3BasisCheckpointError(
                "Refusing full float64 vocabulary SVD for "
                f"{source_elements} elements; the configured fail-closed limit is "
                f"{full_limit}. Use vocab_algorithm='randomized'."
            )
        pair = factorize_embedding_and_lm_head(
            embedding,
            lm_head,
            rank=rank,
        )
        report = {
            "algorithm": "full-float64-svd",
            "deterministic": True,
            "rank": rank,
            "source_shape": list(embedding.weight.shape),
            "source_matrix_elements": source_elements,
            "source_dtype": str(embedding.weight.dtype),
            "full_svd_max_elements": full_limit,
            "materialized_full_float64_source_matrix": True,
            "materialized_full_left_singular_matrix": True,
            "retained_energy_fraction": pair.plan.retained_energy_fraction,
        }
        return pair.embedding, pair.lm_head, report

    vocab_factor, hidden_factor, report = (
        _chunked_randomized_vocabulary_factors(
            embedding.weight,
            rank=rank,
            row_chunk_size=row_chunk_size,
            oversampling=oversampling,
            power_iterations=power_iterations,
            seed=seed,
        )
    )
    factorized_embedding = FactorizedEmbedding(
        vocab_factor,
        hidden_factor,
        padding_idx=embedding.padding_idx,
        max_norm=embedding.max_norm,
        norm_type=embedding.norm_type,
        scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse,
    )
    factorized_embedding.train(embedding.training)
    factorized_head = FactorizedLMHead(
        factorized_embedding.vocab_factor,
        factorized_embedding.hidden_factor,
        bias=lm_head.bias,
    )
    factorized_head.train(lm_head.training)
    report["full_svd_max_elements"] = full_limit
    return factorized_embedding, factorized_head, report


def prepare_qwen3_basis_model(
    model: nn.Module,
    *,
    rank_plan: Qwen3BasisRankPlan | Mapping[str, Any] | None = None,
    role_ranks: Mapping[str, int] | None = None,
    vocab_rank: int | None = None,
    parameter_budget: int | None = None,
    feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
    vocab_algorithm: str = "auto",
    full_vocab_svd_max_elements: int = DEFAULT_FULL_VOCAB_SVD_MAX_ELEMENTS,
    vocab_row_chunk_size: int = DEFAULT_VOCAB_ROW_CHUNK_SIZE,
    vocab_oversampling: int = DEFAULT_VOCAB_OVERSAMPLING,
    vocab_power_iterations: int = DEFAULT_VOCAB_POWER_ITERATIONS,
    vocab_seed: int = DEFAULT_VOCAB_RANDOM_SEED,
) -> nn.Module:
    """Convert a tied dense Qwen3 model to the standalone factorized topology.

    Conversion is in place because cloning a multi-billion parameter source
    would defeat the checkpoint's bounded-memory purpose.  The returned model
    has no parent, teacher, or adapter dependency.
    """

    if not isinstance(model, nn.Module) or not hasattr(model, "config"):
        raise Qwen3BasisCheckpointError("model must be a Qwen3 nn.Module.")
    existing = _prepared_rank_plan(model)
    if existing is not None:
        use_existing = (
            rank_plan is None and role_ranks is None and vocab_rank is None
        )
        requested = plan_qwen3_basis_checkpoint(
            model.config,
            rank_plan=existing if use_existing else rank_plan,
            role_ranks=role_ranks,
            vocab_rank=vocab_rank,
            parameter_budget=parameter_budget,
        )
        if requested.ranks != existing:
            raise Qwen3BasisCheckpointError(
                "Model is already prepared with a different rank plan."
            )
        _assert_plan_counts(model, requested)
        setattr(model, "neural_forge_basis_plan", requested)
        if not hasattr(model, "neural_forge_vocabulary_factorization"):
            setattr(
                model,
                "neural_forge_vocabulary_factorization",
                {
                    "algorithm": "pre-existing-factorized-vocabulary",
                    "deterministic": True,
                    "rank": existing.vocab,
                    "source_shape": [
                        requested.vocab_size,
                        requested.hidden_size,
                    ],
                },
            )
        return model

    plan = plan_qwen3_basis_checkpoint(
        model.config,
        rank_plan=rank_plan,
        role_ranks=role_ranks,
        vocab_rank=vocab_rank,
        parameter_budget=parameter_budget,
    )
    source_logical, source_unique = _model_parameter_counts(model)
    if source_logical != plan.dense_logical_parameter_count:
        raise Qwen3BasisCheckpointError(
            "Dense Qwen3 logical parameter count differs from the closed form: "
            f"expected {plan.dense_logical_parameter_count}, got {source_logical}."
        )
    if source_unique != plan.dense_unique_parameter_count:
        raise Qwen3BasisCheckpointError(
            "Dense Qwen3 unique parameter count differs from the closed form: "
            f"expected {plan.dense_unique_parameter_count}, got {source_unique}."
        )

    embedding = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    if type(embedding) is not nn.Embedding or type(lm_head) is not nn.Linear:
        raise Qwen3BasisCheckpointError(
            "Dense source must expose exact nn.Embedding and nn.Linear modules."
        )
    if not _same_storage(embedding.weight, lm_head.weight):
        raise Qwen3BasisCheckpointError(
            "Dense source embedding and LM head must share one tied weight."
        )

    factorized_embedding, factorized_head, vocab_report = _factorize_tied_vocabulary(
        embedding,
        lm_head,
        rank=plan.ranks.vocab,
        algorithm=vocab_algorithm,
        full_svd_max_elements=full_vocab_svd_max_elements,
        row_chunk_size=vocab_row_chunk_size,
        oversampling=vocab_oversampling,
        power_iterations=vocab_power_iterations,
        seed=vocab_seed,
    )
    if not (
        factorized_embedding.vocab_factor is factorized_head.vocab_factor
        and factorized_embedding.hidden_factor is factorized_head.hidden_factor
    ):
        raise Qwen3BasisCheckpointError(
            "Vocabulary factorization unexpectedly lost the dense weight tie."
        )
    was_training = model.training
    compiled = compile_hf_depth_basis_ir(
        model,
        role_ranks=plan.ranks.role_ranks,
        feature_chunk_size=feature_chunk_size,
    )
    try:
        model.set_input_embeddings(factorized_embedding)
        model.set_output_embeddings(factorized_head)
    except Exception:
        compiled.materialize()
        model.set_input_embeddings(embedding)
        model.set_output_embeddings(lm_head)
        raise
    model.train(was_training)
    actual_ranks = _prepared_rank_plan(model)
    if actual_ranks != plan.ranks:
        raise Qwen3BasisCheckpointError(
            "Prepared module ranks disagree with the requested plan."
        )
    _assert_plan_counts(model, plan)
    setattr(model, "neural_forge_basis_plan", plan)
    setattr(model, "neural_forge_vocabulary_factorization", vocab_report)
    return model


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3BasisCheckpointError(
            "Checkpoint metadata must be finite JSON data."
        ) from exc
    return encoded + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _tensor_bytes(tensor: Tensor) -> bytes:
    cpu = tensor.detach().to(device="cpu").contiguous()
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def _canonical_state(
    model: nn.Module,
) -> tuple[dict[str, Tensor], dict[str, str], dict[str, dict[str, Any]]]:
    state = model.state_dict()
    parameter_grad = {
        name: bool(parameter.requires_grad)
        for name, parameter in model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    }
    storage_to_name: dict[tuple[Any, ...], str] = {}
    tensors: dict[str, Tensor] = {}
    key_to_tensor: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for state_key, value in state.items():
        storage_key = _exact_storage_key(value)
        canonical_name = storage_to_name.get(storage_key)
        if canonical_name is None:
            canonical_name = state_key
            storage_to_name[storage_key] = canonical_name
            canonical = value.detach().to(device="cpu").contiguous()
            tensors[canonical_name] = canonical
            raw = _tensor_bytes(canonical)
            metadata[canonical_name] = {
                "shape": list(canonical.shape),
                "dtype": str(canonical.dtype),
                "numel": int(canonical.numel()),
                "nbytes": len(raw),
                "sha256": _sha256_bytes(raw),
                "is_parameter": state_key in parameter_grad,
            }
        key_to_tensor[state_key] = canonical_name
    return tensors, key_to_tensor, metadata


def _topology_manifest(model: nn.Module) -> dict[str, Any]:
    registry = getattr(model, _REGISTRY_NAME)
    roles: dict[str, Any] = {}
    for role in PROJECTION_ROLES:
        representation = registry.representations[role]
        roles[role] = {
            "rank": int(representation.rank),
            "logical_layers": int(representation.logical_layers),
            "coefficient_rows": int(representation.coefficient_rows),
            "in_features": int(representation.in_features),
            "out_features": int(representation.out_features),
            "has_bias": bool(representation.has_bias),
        }
    embedding = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    return {
        "model_class": "transformers.Qwen3ForCausalLM",
        "registry_path": _REGISTRY_NAME,
        "roles": roles,
        "vocabulary": {
            "tied": True,
            "rank": int(embedding.rank),
            "vocab_size": int(embedding.num_embeddings),
            "hidden_size": int(embedding.embedding_dim),
            "head_bias": lm_head.bias is not None,
            "padding_idx": embedding.padding_idx,
            "max_norm": embedding.max_norm,
            "norm_type": embedding.norm_type,
            "scale_grad_by_freq": embedding.scale_grad_by_freq,
            "sparse": embedding.sparse,
        },
    }


def _depth_factorization_manifest(model: nn.Module) -> dict[str, Any]:
    existing = getattr(model, "neural_forge_depth_factorization", None)
    if isinstance(existing, Mapping):
        return {role: dict(existing[role]) for role in PROJECTION_ROLES}
    registry = getattr(model, _REGISTRY_NAME)
    reports: dict[str, Any] = {}
    for role in PROJECTION_ROLES:
        report = registry.representations[role].report
        if report is None:
            reports[role] = {
                "rank": int(registry.representations[role].rank),
                "quality_metrics_available": False,
            }
        else:
            reports[role] = {
                **report.as_dict(),
                "quality_metrics_available": True,
            }
    return reports


def _select_tensor_format(requested: str | None) -> str:
    if requested not in {None, "safetensors", "raw"}:
        raise Qwen3BasisCheckpointError(
            "tensor_format must be 'safetensors', 'raw', or None."
        )
    if requested == "raw":
        return "raw"
    try:
        import safetensors.torch  # noqa: F401
    except ImportError:
        if requested == "safetensors":
            raise Qwen3BasisCheckpointError(
                "safetensors was requested but is not installed."
            ) from None
        return "raw"
    return "safetensors"


def _save_tensor_payload(
    directory: Path,
    tensors: Mapping[str, Tensor],
    metadata: dict[str, dict[str, Any]],
    tensor_format: str,
) -> tuple[str, Path]:
    ordered = {name: tensors[name] for name in sorted(tensors)}
    if tensor_format == "safetensors":
        from safetensors.torch import save_file

        path = directory / SAFETENSORS_FILENAME
        save_file(
            ordered,
            str(path),
            metadata={"format": CHECKPOINT_SCHEMA},
        )
        return "safetensors", path

    path = directory / RAW_TENSORS_FILENAME
    offset = 0
    with path.open("xb") as handle:
        for name, tensor in ordered.items():
            raw = _tensor_bytes(tensor)
            metadata[name]["byte_offset"] = offset
            handle.write(raw)
            offset += len(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return "neural-forge-raw-v1", path


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def export_qwen3_basis_checkpoint(
    model: nn.Module,
    output_directory: str | os.PathLike[str],
    *,
    rank_plan: Qwen3BasisRankPlan | Mapping[str, Any] | None = None,
    role_ranks: Mapping[str, int] | None = None,
    vocab_rank: int | None = None,
    parameter_budget: int | None = None,
    feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
    vocab_algorithm: str = "auto",
    full_vocab_svd_max_elements: int = DEFAULT_FULL_VOCAB_SVD_MAX_ELEMENTS,
    vocab_row_chunk_size: int = DEFAULT_VOCAB_ROW_CHUNK_SIZE,
    vocab_oversampling: int = DEFAULT_VOCAB_OVERSAMPLING,
    vocab_power_iterations: int = DEFAULT_VOCAB_POWER_ITERATIONS,
    vocab_seed: int = DEFAULT_VOCAB_RANDOM_SEED,
    tensor_format: str | None = None,
) -> dict[str, Any]:
    """Prepare and atomically export a standalone Qwen3 basis checkpoint."""

    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise Qwen3BasisCheckpointError(
            f"Refusing to replace existing checkpoint path {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_format = _select_tensor_format(tensor_format)
    prepared = prepare_qwen3_basis_model(
        model,
        rank_plan=rank_plan,
        role_ranks=role_ranks,
        vocab_rank=vocab_rank,
        parameter_budget=parameter_budget,
        feature_chunk_size=feature_chunk_size,
        vocab_algorithm=vocab_algorithm,
        full_vocab_svd_max_elements=full_vocab_svd_max_elements,
        vocab_row_chunk_size=vocab_row_chunk_size,
        vocab_oversampling=vocab_oversampling,
        vocab_power_iterations=vocab_power_iterations,
        vocab_seed=vocab_seed,
    )
    plan = getattr(prepared, "neural_forge_basis_plan")
    tensors, key_to_tensor, tensor_metadata = _canonical_state(prepared)
    logical, unique = _model_parameter_counts(prepared)
    parameter_requires_grad = {
        name: bool(parameter.requires_grad)
        for name, parameter in prepared.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    }
    canonical_parameters = {
        key_to_tensor[name] for name in parameter_requires_grad
    }
    stored_parameter_count = sum(
        int(tensors[name].numel()) for name in canonical_parameters
    )
    if stored_parameter_count != unique:
        raise Qwen3BasisCheckpointError(
            "Canonical state storage does not match unique parameter storage: "
            f"expected {unique}, got {stored_parameter_count}."
        )
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        config_path = temporary / CONFIG_FILENAME
        config_bytes = _canonical_json_bytes(prepared.config.to_dict())
        _write_bytes(config_path, config_bytes)

        generation_path = temporary / GENERATION_CONFIG_FILENAME
        generation_config = getattr(prepared, "generation_config", None)
        generation_data = (
            {} if generation_config is None else generation_config.to_dict()
        )
        _write_bytes(generation_path, _canonical_json_bytes(generation_data))

        stored_format, tensor_path = _save_tensor_payload(
            temporary,
            tensors,
            tensor_metadata,
            selected_format,
        )
        manifest_payload: dict[str, Any] = {
            "schema": CHECKPOINT_SCHEMA,
            "serialization": {
                "python_pickle": False,
                "serialized_modules": False,
                "serialized_weakrefs": False,
                "tensor_format": stored_format,
                "byte_order": sys.byteorder,
            },
            "artifacts": {
                "config": _artifact_entry(config_path),
                "generation_config": _artifact_entry(generation_path),
                "tensors": _artifact_entry(tensor_path),
            },
            "rank_plan": plan.ranks.as_dict(),
            "parameter_counts": {
                **plan.as_dict(),
                "actual_logical_parameter_count": logical,
                "actual_unique_parameter_count": unique,
            },
            "topology": _topology_manifest(prepared),
            "depth_factorization": _depth_factorization_manifest(prepared),
            "vocabulary_factorization": getattr(
                prepared,
                "neural_forge_vocabulary_factorization",
            ),
            "state_dict": {
                "key_to_tensor": key_to_tensor,
                "logical_tensor_count": len(key_to_tensor),
                "stored_tensor_count": len(tensors),
                "alias_count": len(key_to_tensor) - len(tensors),
                "parameter_requires_grad": parameter_requires_grad,
            },
            "tensors": tensor_metadata,
            "model_training": bool(prepared.training),
            "behavioral_validation": {
                "status": "not_evaluated",
                "structural_checks_only": True,
                "warning": (
                    "Exact checkpoint reload and valid logical topology do not "
                    "establish behavioral viability; evaluate the compiled model."
                ),
            },
            "standalone": True,
            "external_weight_dependencies": [],
        }
        manifest_hash = _sha256_bytes(_canonical_json_bytes(manifest_payload))
        manifest = {**manifest_payload, _MANIFEST_HASH_KEY: manifest_hash}
        _write_bytes(
            temporary / MANIFEST_FILENAME,
            _canonical_json_bytes(manifest),
        )
        os.replace(temporary, output)
    except Exception:
        resolved_temp = temporary.resolve()
        if resolved_temp.parent == output.parent and resolved_temp.exists():
            shutil.rmtree(resolved_temp)
        raise

    setattr(prepared, "neural_forge_checkpoint_manifest", manifest)
    return manifest


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Qwen3BasisFormatError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen3BasisFormatError(f"Cannot read JSON artifact {path.name}.") from exc
    if not isinstance(value, dict):
        raise Qwen3BasisFormatError(f"{path.name} must contain a JSON object.")
    return value


def _checked_artifact_path(root: Path, entry: Any, label: str) -> Path:
    if not isinstance(entry, Mapping):
        raise Qwen3BasisFormatError(f"Artifact entry {label!r} must be an object.")
    filename = entry.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise Qwen3BasisFormatError(f"Artifact {label!r} has an unsafe filename.")
    path = root / filename
    if not path.is_file():
        raise Qwen3BasisFormatError(f"Artifact {filename!r} is missing.")
    expected_size = entry.get("size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise Qwen3BasisFormatError(f"Artifact {label!r} has no valid size.")
    if path.stat().st_size != expected_size:
        raise Qwen3BasisFormatError(f"Artifact {filename!r} has the wrong size.")
    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise Qwen3BasisFormatError(f"Artifact {label!r} has no valid hash.")
    if _sha256_file(path) != expected_hash:
        raise Qwen3BasisFormatError(f"Artifact {filename!r} failed SHA-256.")
    return path


def _dtype_from_name(name: Any) -> torch.dtype:
    if not isinstance(name, str) or name not in _SUPPORTED_DTYPES:
        raise Qwen3BasisFormatError(f"Unsupported tensor dtype {name!r}.")
    return _SUPPORTED_DTYPES[name]


def _parameter_required(
    manifest: Mapping[str, Any],
    state_key: str,
    default: bool,
) -> bool:
    state = manifest.get("state_dict")
    values = state.get("parameter_requires_grad") if isinstance(state, Mapping) else {}
    if not isinstance(values, Mapping):
        raise Qwen3BasisFormatError("Invalid parameter_requires_grad table.")
    result = values.get(state_key, default)
    if not isinstance(result, bool):
        raise Qwen3BasisFormatError(
            f"Invalid requires_grad value for {state_key!r}."
        )
    return result


def _meta_parameter(
    shape: Sequence[int],
    dtype: torch.dtype,
    *,
    requires_grad: bool,
) -> nn.Parameter:
    return nn.Parameter(
        torch.empty(tuple(shape), device="meta", dtype=dtype),
        requires_grad=requires_grad,
    )


def _tensor_dtype_for_state_key(
    manifest: Mapping[str, Any],
    state_key: str,
) -> torch.dtype:
    state = manifest.get("state_dict")
    tensors = manifest.get("tensors")
    if not isinstance(state, Mapping) or not isinstance(tensors, Mapping):
        raise Qwen3BasisFormatError("Invalid tensor manifest tables.")
    aliases = state.get("key_to_tensor")
    if not isinstance(aliases, Mapping):
        raise Qwen3BasisFormatError("Invalid state tensor alias table.")
    canonical = aliases.get(state_key)
    tensor_entry = tensors.get(canonical) if isinstance(canonical, str) else None
    if not isinstance(tensor_entry, Mapping):
        raise Qwen3BasisFormatError(
            f"No tensor metadata for state key {state_key!r}."
        )
    return _dtype_from_name(tensor_entry.get("dtype"))


def _new_meta_depth_basis(
    role: str,
    dimensions: _Qwen3Dimensions,
    rank: int,
    manifest: Mapping[str, Any],
) -> DepthBasisLinear:
    in_features, out_features, has_bias = dimensions.role_shapes[role]
    prefix = f"{_REGISTRY_NAME}.representations.{role}"
    dtype = _tensor_dtype_for_state_key(manifest, f"{prefix}.basis_weight")
    module = DepthBasisLinear.__new__(DepthBasisLinear)
    nn.Module.__init__(module)
    module.in_features = in_features
    module.out_features = out_features
    module.rank = rank
    module.logical_layers = dimensions.layers
    module.coefficient_rows = dimensions.layers
    module.basis_weight = _meta_parameter(
        (rank, out_features, in_features),
        dtype,
        requires_grad=_parameter_required(
            manifest,
            f"{prefix}.basis_weight",
            True,
        ),
    )
    module.coefficients = _meta_parameter(
        (dimensions.layers, rank),
        dtype,
        requires_grad=_parameter_required(
            manifest,
            f"{prefix}.coefficients",
            True,
        ),
    )
    if has_bias:
        module.basis_bias = _meta_parameter(
            (rank, out_features),
            dtype,
            requires_grad=_parameter_required(
                manifest,
                f"{prefix}.basis_bias",
                True,
            ),
        )
    else:
        module.register_parameter("basis_bias", None)
    module.register_buffer(
        "layer_to_coefficient",
        torch.empty((dimensions.layers,), device="meta", dtype=torch.int64),
    )
    module.report = None
    return module


def _new_meta_vocabulary(
    dimensions: _Qwen3Dimensions,
    rank: int,
    topology: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[FactorizedEmbedding, FactorizedLMHead]:
    prefix = "model.embed_tokens"
    dtype = _tensor_dtype_for_state_key(manifest, f"{prefix}.vocab_factor")
    vocab_factor = _meta_parameter(
        (dimensions.vocab, rank),
        dtype,
        requires_grad=_parameter_required(
            manifest,
            f"{prefix}.vocab_factor",
            False,
        ),
    )
    hidden_factor = _meta_parameter(
        (rank, dimensions.hidden),
        dtype,
        requires_grad=_parameter_required(
            manifest,
            f"{prefix}.hidden_factor",
            False,
        ),
    )
    embedding = FactorizedEmbedding.__new__(FactorizedEmbedding)
    nn.Module.__init__(embedding)
    embedding.vocab_factor = vocab_factor
    embedding.hidden_factor = hidden_factor
    embedding.num_embeddings = dimensions.vocab
    embedding.embedding_dim = dimensions.hidden
    embedding.rank = rank
    embedding.padding_idx = topology.get("padding_idx")
    embedding.max_norm = topology.get("max_norm")
    embedding.norm_type = float(topology.get("norm_type", 2.0))
    embedding.scale_grad_by_freq = bool(topology.get("scale_grad_by_freq", False))
    embedding.sparse = bool(topology.get("sparse", False))
    embedding.plan = None

    lm_head = FactorizedLMHead.__new__(FactorizedLMHead)
    nn.Module.__init__(lm_head)
    lm_head.vocab_factor = vocab_factor
    lm_head.hidden_factor = hidden_factor
    lm_head.out_features = dimensions.vocab
    lm_head.in_features = dimensions.hidden
    lm_head.rank = rank
    if topology.get("head_bias") is True:
        lm_head.bias = _meta_parameter(
            (dimensions.vocab,),
            dtype,
            requires_grad=_parameter_required(manifest, "lm_head.bias", False),
        )
    else:
        lm_head.register_parameter("bias", None)
    lm_head.plan = None
    return embedding, lm_head


def _set_module(model: nn.Module, path: str, module: nn.Module) -> None:
    owner_path, separator, name = path.rpartition(".")
    owner = resolve_module(model, owner_path) if separator else model
    setattr(owner, name, module)


def _role_paths(model: nn.Module) -> dict[str, tuple[str, ...]]:
    layout = build_decoder_layout(model)
    if layout.family != "qwen3":
        raise Qwen3BasisFormatError("Reload topology is not Qwen3.")
    paths: dict[str, list[str]] = {role: [] for role in PROJECTION_ROLES}
    for index in range(layout.layer_count):
        for role, (section, projection) in _ROLE_PATH_PARTS.items():
            paths[role].append(f"model.layers.{index}.{section}.{projection}")
    return {role: tuple(paths[role]) for role in PROJECTION_ROLES}


def _validate_topology_manifest(
    topology: Any,
    plan: Qwen3BasisCheckpointPlan,
) -> Mapping[str, Any]:
    if not isinstance(topology, Mapping):
        raise Qwen3BasisFormatError("Manifest topology must be an object.")
    roles = topology.get("roles")
    vocabulary = topology.get("vocabulary")
    if not isinstance(roles, Mapping) or not isinstance(vocabulary, Mapping):
        raise Qwen3BasisFormatError("Manifest topology is incomplete.")
    dimensions = _qwen3_dimensions(
        {
            "model_type": "qwen3",
            "tie_word_embeddings": True,
            "hidden_size": plan.hidden_size,
            "num_attention_heads": plan.query_width // plan.head_dim,
            "num_key_value_heads": plan.key_value_width // plan.head_dim,
            "head_dim": plan.head_dim,
            "num_hidden_layers": plan.num_hidden_layers,
            "intermediate_size": plan.intermediate_size,
            "vocab_size": plan.vocab_size,
            "attention_bias": plan.attention_bias,
        }
    )
    for role in PROJECTION_ROLES:
        value = roles.get(role)
        if not isinstance(value, Mapping):
            raise Qwen3BasisFormatError(f"Missing topology for role {role!r}.")
        in_features, out_features, has_bias = dimensions.role_shapes[role]
        expected = {
            "rank": plan.ranks.role_ranks[role],
            "logical_layers": dimensions.layers,
            "coefficient_rows": dimensions.layers,
            "in_features": in_features,
            "out_features": out_features,
            "has_bias": has_bias,
        }
        if any(value.get(name) != expected_value for name, expected_value in expected.items()):
            raise Qwen3BasisFormatError(
                f"Topology metadata for role {role!r} disagrees with the plan."
            )
    expected_vocab = {
        "tied": True,
        "rank": plan.ranks.vocab,
        "vocab_size": plan.vocab_size,
        "hidden_size": plan.hidden_size,
        "head_bias": False,
    }
    if any(
        vocabulary.get(name) != expected
        for name, expected in expected_vocab.items()
    ):
        raise Qwen3BasisFormatError(
            "Vocabulary topology metadata disagrees with the plan."
        )
    return vocabulary


def _build_reload_topology(
    config_data: Mapping[str, Any],
    generation_data: Mapping[str, Any],
    manifest: Mapping[str, Any],
    plan: Qwen3BasisCheckpointPlan,
    device: torch.device,
) -> nn.Module:
    try:
        from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
    except ImportError as exc:
        raise Qwen3BasisCheckpointError(
            "Loading requires Transformers with Qwen3 support."
        ) from exc

    config = Qwen3Config.from_dict(dict(config_data))
    dimensions = _qwen3_dimensions(config)
    vocabulary_topology = _validate_topology_manifest(
        manifest.get("topology"),
        plan,
    )
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    rotary_type = type(model.model.rotary_emb)
    model.model.rotary_emb = rotary_type(config, device=device)

    representations = {
        role: _new_meta_depth_basis(
            role,
            dimensions,
            plan.ranks.role_ranks[role],
            manifest,
        )
        for role in PROJECTION_ROLES
    }
    registry = DepthBasisRegistry(representations)
    setattr(model, _REGISTRY_NAME, registry)
    paths = _role_paths(model)
    for role in PROJECTION_ROLES:
        representation = representations[role]
        for layer_index, path in enumerate(paths[role]):
            _set_module(
                model,
                path,
                FixedLayerDepthBasisLinear(representation, layer_index),
            )

    embedding, lm_head = _new_meta_vocabulary(
        dimensions,
        plan.ranks.vocab,
        vocabulary_topology,
        manifest,
    )
    model.set_input_embeddings(embedding)
    model.set_output_embeddings(lm_head)
    model.generation_config = GenerationConfig.from_dict(dict(generation_data))
    model.train(bool(manifest.get("model_training", False)))

    state = manifest.get("state_dict")
    key_to_tensor = state.get("key_to_tensor") if isinstance(state, Mapping) else None
    if not isinstance(key_to_tensor, Mapping):
        raise Qwen3BasisFormatError("Manifest has no state tensor map.")
    expected_keys = set(model.state_dict())
    manifest_keys = set(key_to_tensor)
    if expected_keys != manifest_keys:
        missing = sorted(expected_keys - manifest_keys)
        unexpected = sorted(manifest_keys - expected_keys)
        raise Qwen3BasisFormatError(
            "Constructed topology and state manifest differ: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    return model


def _validate_tensor(
    name: str,
    tensor: Tensor,
    entry: Any,
) -> None:
    if not isinstance(entry, Mapping):
        raise Qwen3BasisFormatError(f"Tensor metadata for {name!r} is invalid.")
    shape = entry.get("shape")
    if not isinstance(shape, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in shape
    ):
        raise Qwen3BasisFormatError(f"Tensor {name!r} has an invalid shape.")
    if tuple(shape) != tuple(tensor.shape):
        raise Qwen3BasisFormatError(f"Tensor {name!r} has the wrong shape.")
    if _dtype_from_name(entry.get("dtype")) != tensor.dtype:
        raise Qwen3BasisFormatError(f"Tensor {name!r} has the wrong dtype.")
    if entry.get("numel") != int(tensor.numel()):
        raise Qwen3BasisFormatError(f"Tensor {name!r} has the wrong element count.")
    raw = _tensor_bytes(tensor)
    if entry.get("nbytes") != len(raw):
        raise Qwen3BasisFormatError(f"Tensor {name!r} has the wrong byte count.")
    if entry.get("sha256") != _sha256_bytes(raw):
        raise Qwen3BasisFormatError(f"Tensor {name!r} failed SHA-256.")


def _load_raw_tensors(
    path: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Tensor]:
    payload = path.read_bytes()
    result: dict[str, Tensor] = {}
    occupied: list[tuple[int, int]] = []
    for name in sorted(metadata):
        entry = metadata[name]
        if not isinstance(entry, Mapping):
            raise Qwen3BasisFormatError(f"Invalid tensor entry {name!r}.")
        offset = entry.get("byte_offset")
        nbytes = entry.get("nbytes")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(nbytes, bool)
            or not isinstance(nbytes, int)
            or nbytes < 0
            or offset + nbytes > len(payload)
        ):
            raise Qwen3BasisFormatError(f"Invalid raw range for tensor {name!r}.")
        occupied.append((offset, offset + nbytes))
        buffer = bytearray(payload[offset : offset + nbytes])
        dtype = _dtype_from_name(entry.get("dtype"))
        tensor = torch.frombuffer(buffer, dtype=dtype).clone()
        tensor = tensor.reshape(tuple(entry.get("shape", []))).to(device=device)
        result[name] = tensor
    if any(
        left[1] > right[0]
        for left, right in zip(sorted(occupied), sorted(occupied)[1:])
    ):
        raise Qwen3BasisFormatError("Raw tensor ranges overlap.")
    if sum(end - start for start, end in occupied) != len(payload):
        raise Qwen3BasisFormatError("Raw tensor payload has unclaimed bytes.")
    return result


def _load_tensor_payload(
    path: Path,
    manifest: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Tensor]:
    metadata = manifest.get("tensors")
    serialization = manifest.get("serialization")
    if not isinstance(metadata, Mapping) or not isinstance(serialization, Mapping):
        raise Qwen3BasisFormatError("Tensor manifest metadata is invalid.")
    tensor_format = serialization.get("tensor_format")
    if tensor_format == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise Qwen3BasisCheckpointError(
                "This artifact requires the safetensors package."
            ) from exc
        tensors = load_file(str(path), device=str(device))
    elif tensor_format == "neural-forge-raw-v1":
        if serialization.get("byte_order") != sys.byteorder:
            raise Qwen3BasisFormatError(
                "Raw artifact byte order differs from this runtime."
            )
        tensors = _load_raw_tensors(path, metadata, device)
    else:
        raise Qwen3BasisFormatError(
            f"Unsupported tensor format {tensor_format!r}."
        )
    if set(tensors) != set(metadata):
        missing = sorted(set(metadata) - set(tensors))
        unexpected = sorted(set(tensors) - set(metadata))
        raise Qwen3BasisFormatError(
            f"Tensor payload keys differ: missing={missing!r}, "
            f"unexpected={unexpected!r}."
        )
    for name, tensor in tensors.items():
        _validate_tensor(name, tensor, metadata[name])
    return tensors


def _validate_manifest_plan(
    config_data: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Qwen3BasisCheckpointPlan:
    rank_plan = manifest.get("rank_plan")
    counts = manifest.get("parameter_counts")
    if not isinstance(rank_plan, Mapping) or not isinstance(counts, Mapping):
        raise Qwen3BasisFormatError("Manifest rank or count metadata is missing.")
    budget = counts.get("parameter_budget")
    plan = plan_qwen3_basis_checkpoint(
        config_data,
        rank_plan=rank_plan,
        parameter_budget=budget,
    )
    expected = plan.as_dict()
    for key, value in expected.items():
        if counts.get(key) != value:
            raise Qwen3BasisFormatError(
                f"Manifest parameter count field {key!r} is inconsistent."
            )
    if counts.get("actual_logical_parameter_count") != plan.logical_parameter_count:
        raise Qwen3BasisFormatError("Manifest actual logical count is inconsistent.")
    if counts.get("actual_unique_parameter_count") != plan.unique_parameter_count:
        raise Qwen3BasisFormatError("Manifest actual unique count is inconsistent.")
    vocab_factorization = manifest.get("vocabulary_factorization")
    if not isinstance(vocab_factorization, Mapping):
        raise Qwen3BasisFormatError(
            "Manifest vocabulary factorization report is missing."
        )
    if vocab_factorization.get("rank") != plan.ranks.vocab:
        raise Qwen3BasisFormatError(
            "Vocabulary factorization rank disagrees with the plan."
        )
    if vocab_factorization.get("deterministic") is not True:
        raise Qwen3BasisFormatError(
            "Vocabulary factorization is not marked deterministic."
        )
    depth_factorization = manifest.get("depth_factorization")
    if not isinstance(depth_factorization, Mapping) or set(
        depth_factorization
    ) != set(PROJECTION_ROLES):
        raise Qwen3BasisFormatError(
            "Manifest depth factorization reports are incomplete."
        )
    for role in PROJECTION_ROLES:
        role_report = depth_factorization[role]
        if (
            not isinstance(role_report, Mapping)
            or role_report.get("rank") != plan.ranks.role_ranks[role]
        ):
            raise Qwen3BasisFormatError(
                f"Depth factorization report for {role!r} is inconsistent."
            )
    return plan


def load_qwen3_basis_checkpoint(
    checkpoint_directory: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load a standalone Qwen3 basis model with strict state validation."""

    root = Path(checkpoint_directory).expanduser().resolve()
    if not root.is_dir():
        raise Qwen3BasisFormatError(f"Checkpoint directory {root} does not exist.")
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise Qwen3BasisFormatError(f"Missing {MANIFEST_FILENAME}.")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise Qwen3BasisFormatError("Unsupported checkpoint schema.")
    claimed_hash = manifest.get(_MANIFEST_HASH_KEY)
    hash_payload = dict(manifest)
    hash_payload.pop(_MANIFEST_HASH_KEY, None)
    if claimed_hash != _sha256_bytes(_canonical_json_bytes(hash_payload)):
        raise Qwen3BasisFormatError("Manifest SHA-256 mismatch.")
    serialization = manifest.get("serialization")
    if not isinstance(serialization, Mapping) or any(
        serialization.get(name) is not False
        for name in ("python_pickle", "serialized_modules", "serialized_weakrefs")
    ):
        raise Qwen3BasisFormatError("Manifest permits unsafe object serialization.")
    if manifest.get("standalone") is not True or manifest.get(
        "external_weight_dependencies"
    ) != []:
        raise Qwen3BasisFormatError(
            "Manifest does not describe a standalone checkpoint."
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Qwen3BasisFormatError("Manifest has no artifact table.")
    config_entry = artifacts.get("config")
    generation_entry = artifacts.get("generation_config")
    tensor_entry = artifacts.get("tensors")
    config_path = _checked_artifact_path(root, config_entry, "config")
    generation_path = _checked_artifact_path(
        root,
        generation_entry,
        "generation_config",
    )
    config_data = _read_json(config_path)
    generation_data = _read_json(generation_path)
    plan = _validate_manifest_plan(config_data, manifest)

    target_device = torch.device(device)
    model = _build_reload_topology(
        config_data,
        generation_data,
        manifest,
        plan,
        target_device,
    )

    # Payload hashing and loading deliberately happen after topology construction.
    tensor_path = _checked_artifact_path(root, tensor_entry, "tensors")
    tensors = _load_tensor_payload(tensor_path, manifest, target_device)
    state = manifest.get("state_dict")
    aliases = state.get("key_to_tensor") if isinstance(state, Mapping) else None
    if not isinstance(aliases, Mapping):
        raise Qwen3BasisFormatError("Invalid state alias table.")
    expanded: dict[str, Tensor] = {}
    for state_key, canonical_name in aliases.items():
        if not isinstance(state_key, str) or not isinstance(canonical_name, str):
            raise Qwen3BasisFormatError("State alias entries must be strings.")
        if canonical_name not in tensors:
            raise Qwen3BasisFormatError(
                f"State key {state_key!r} references a missing tensor."
            )
        expanded[state_key] = tensors[canonical_name]
    incompatible = model.load_state_dict(expanded, strict=True, assign=True)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise Qwen3BasisFormatError(
            f"Strict state load failed: missing={missing!r}, "
            f"unexpected={unexpected!r}."
        )

    embedding = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    lm_head.vocab_factor = embedding.vocab_factor
    lm_head.hidden_factor = embedding.hidden_factor
    parameter_requires_grad = state.get("parameter_requires_grad")
    if not isinstance(parameter_requires_grad, Mapping):
        raise Qwen3BasisFormatError("Invalid parameter requires-grad table.")
    for name, parameter in model.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        required = parameter_requires_grad.get(name)
        if not isinstance(required, bool):
            raise Qwen3BasisFormatError(
                f"No valid requires_grad metadata for {name!r}."
            )
        parameter.requires_grad_(required)

    _assert_plan_counts(model, plan)
    loaded_state = model.state_dict()
    for state_key, canonical_name in aliases.items():
        if not torch.equal(loaded_state[state_key], tensors[canonical_name]):
            raise Qwen3BasisFormatError(
                f"Loaded state tensor {state_key!r} differs from its payload."
            )
    setattr(model, "neural_forge_basis_plan", plan)
    setattr(
        model,
        "neural_forge_vocabulary_factorization",
        dict(manifest["vocabulary_factorization"]),
    )
    setattr(
        model,
        "neural_forge_depth_factorization",
        {
            role: dict(manifest["depth_factorization"][role])
            for role in PROJECTION_ROLES
        },
    )
    setattr(model, "neural_forge_checkpoint_manifest", manifest)
    setattr(
        model,
        "neural_forge_load_report",
        {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "stored_tensor_count": len(tensors),
            "logical_state_tensor_count": len(aliases),
            "manifest_sha256": manifest[_MANIFEST_HASH_KEY],
        },
    )
    return model


def evaluate_qwen3_basis_model(
    model: nn.Module,
    tokenizer: Any,
    output_file: str | os.PathLike[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate an already-prepared model without reloading its checkpoint."""

    if _prepared_rank_plan(model) is None:
        raise Qwen3BasisCheckpointError(
            "model must be prepared before immediate in-memory evaluation."
        )
    from neural_forge.hf_tools import evaluate_loaded_text_model

    return evaluate_loaded_text_model(
        model,
        tokenizer,
        output_file,
        **kwargs,
    )


save_qwen3_basis_checkpoint = export_qwen3_basis_checkpoint
compile_qwen3_basis_checkpoint_model = prepare_qwen3_basis_model
plan_qwen_basis_checkpoint = plan_qwen3_basis_checkpoint
load_qwen_basis_checkpoint = load_qwen3_basis_checkpoint
export_qwen_basis_checkpoint = export_qwen3_basis_checkpoint


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_FILENAME",
    "DEFAULT_FULL_VOCAB_SVD_MAX_ELEMENTS",
    "DEFAULT_VOCAB_OVERSAMPLING",
    "DEFAULT_VOCAB_POWER_ITERATIONS",
    "DEFAULT_VOCAB_RANDOM_SEED",
    "DEFAULT_VOCAB_ROW_CHUNK_SIZE",
    "GENERATION_CONFIG_FILENAME",
    "MANIFEST_FILENAME",
    "Qwen3BasisBudgetError",
    "Qwen3BasisCheckpointError",
    "Qwen3BasisCheckpointPlan",
    "Qwen3BasisFormatError",
    "Qwen3BasisRankPlan",
    "compile_qwen3_basis_checkpoint_model",
    "count_qwen3_basis_parameters",
    "evaluate_qwen3_basis_model",
    "export_qwen3_basis_checkpoint",
    "export_qwen_basis_checkpoint",
    "load_qwen3_basis_checkpoint",
    "load_qwen_basis_checkpoint",
    "plan_qwen3_basis_checkpoint",
    "plan_qwen_basis_checkpoint",
    "prepare_qwen3_basis_model",
    "save_qwen3_basis_checkpoint",
]
