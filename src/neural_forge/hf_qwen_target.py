"""Deterministic Qwen3 blank-target shape planning and materialization.

Parameter planning in this module is framework independent.  Torch and
Transformers are imported only when a selected plan is materialized.  The
materializer constructs ``Qwen3ForCausalLM`` directly from ``Qwen3Config``;
it never loads child weights from a checkpoint.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


FORMULA_VERSION = "transformers-qwen3-dense-v1"
PLAN_SCHEMA = "neural-forge-qwen3-target-plan-v1"
ARTIFACT_SCHEMA = "neural-forge-qwen3-random-target-v1"
TARGET_REPORT_FILENAME = "neural_forge_qwen_target_report.json"

_DEFAULT_HIDDEN_SIZES = (
    64,
    96,
    128,
    160,
    192,
    256,
    320,
    384,
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
)
_DEFAULT_LAYER_COUNTS = (1, 2, 4, 6, 8, 12, 16, 18, 24, 28, 32, 36, 40, 48, 64)
_DEFAULT_QUERY_HEAD_COUNTS = (1, 2, 4, 8, 12, 16, 24, 32)
_DEFAULT_KEY_VALUE_HEAD_COUNTS = (1, 2, 4, 8)
_DEFAULT_HEAD_DIMS = (32, 64, 128)
_DEFAULT_INTERMEDIATE_RATIOS = (
    (2, 1),
    (5, 2),
    (3, 1),
    (7, 2),
    (4, 1),
    (21, 4),
    (43, 8),
)

_COUNTED_CONFIG_FIELDS = frozenset(
    {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "tie_word_embeddings",
        "attention_bias",
    }
)
_MATERIALIZABLE_CONFIG_FIELDS = frozenset(
    {
        "architectures",
        "attention_dropout",
        "bos_token_id",
        "chunk_size_feed_forward",
        "eos_token_id",
        "hidden_act",
        "id2label",
        "initializer_range",
        "is_encoder_decoder",
        "label2id",
        "layer_types",
        "max_position_embeddings",
        "max_window_layers",
        "output_hidden_states",
        "pad_token_id",
        "problem_type",
        "return_dict",
        "rms_norm_eps",
        "rope_parameters",
        "rope_scaling",
        "sliding_window",
        "use_cache",
        "use_sliding_window",
    }
)


class Qwen3TargetError(ValueError):
    """Base class for malformed Qwen3 target requests."""


class Qwen3PlanInfeasibleError(Qwen3TargetError):
    """Raised when well-formed constraints contain no acceptable shape."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


Qwen3TargetInfeasibleError = Qwen3PlanInfeasibleError


def _exact_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise Qwen3TargetError(f"{name} must be an integer, not bool.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3TargetError(f"{name} must be an integer.") from exc
    if result != value:
        raise Qwen3TargetError(f"{name} must be an integer.")
    if positive and result <= 0:
        raise Qwen3TargetError(f"{name} must be positive.")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise Qwen3TargetError(f"{name} must be bool.")
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Qwen3TargetError("Provenance values must be finite JSON data.") from exc
    return text.encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        if name in config:
            return config[name]
        nested = config.get("config")
        if isinstance(nested, Mapping) and name in nested:
            return nested[name]
        selected = config.get("selected_config")
        if isinstance(selected, Mapping) and name in selected:
            return selected[name]
        return default
    return getattr(config, name, default)


@dataclass(frozen=True, slots=True)
class Qwen3Shape:
    """The fields that determine unique Qwen3ForCausalLM parameter storage."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    tie_word_embeddings: bool = False
    attention_bias: bool = False

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            object.__setattr__(
                self,
                name,
                _exact_int(getattr(self, name), name, positive=True),
            )
        object.__setattr__(
            self,
            "tie_word_embeddings",
            _strict_bool(self.tie_word_embeddings, "tie_word_embeddings"),
        )
        object.__setattr__(
            self,
            "attention_bias",
            _strict_bool(self.attention_bias, "attention_bias"),
        )
        if self.num_attention_heads % self.num_key_value_heads:
            raise Qwen3TargetError(
                "num_attention_heads must be divisible by num_key_value_heads "
                "for grouped-query attention."
            )

    @classmethod
    def from_config(cls, config: Any) -> "Qwen3Shape":
        """Extract counted dimensions from a mapping or Qwen3 config object."""

        query_heads = _config_value(config, "num_attention_heads")
        key_value_heads = _config_value(config, "num_key_value_heads")
        if key_value_heads is None:
            key_value_heads = query_heads
        return cls(
            vocab_size=_config_value(config, "vocab_size"),
            hidden_size=_config_value(config, "hidden_size"),
            intermediate_size=_config_value(config, "intermediate_size"),
            num_hidden_layers=_config_value(config, "num_hidden_layers"),
            num_attention_heads=query_heads,
            num_key_value_heads=key_value_heads,
            head_dim=_config_value(config, "head_dim"),
            tie_word_embeddings=_config_value(config, "tie_word_embeddings", False),
            attention_bias=_config_value(config, "attention_bias", False),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
        }

    def search_tuple(self) -> tuple[int, ...]:
        return (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )


def _shape_with_overrides(
    config: Any = None,
    *,
    vocab_size: Any = None,
    hidden_size: Any = None,
    intermediate_size: Any = None,
    num_hidden_layers: Any = None,
    num_attention_heads: Any = None,
    num_key_value_heads: Any = None,
    head_dim: Any = None,
    tie_word_embeddings: Any = None,
    attention_bias: Any = None,
) -> Qwen3Shape:
    values = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "tie_word_embeddings": tie_word_embeddings,
        "attention_bias": attention_bias,
    }
    for name in values:
        if values[name] is None:
            default = (
                False if name in {"tie_word_embeddings", "attention_bias"} else None
            )
            values[name] = _config_value(config, name, default)
    if values["num_key_value_heads"] is None:
        values["num_key_value_heads"] = values["num_attention_heads"]
    return Qwen3Shape(**values)


def _count_shape_parameters(shape: Qwen3Shape) -> int:
    query_width = shape.num_attention_heads * shape.head_dim
    key_value_width = shape.num_key_value_heads * shape.head_dim
    attention_weights = 2 * shape.hidden_size * (query_width + key_value_width)
    attention_biases = (
        query_width + 2 * key_value_width + shape.hidden_size
        if shape.attention_bias
        else 0
    )
    per_layer = (
        attention_weights
        + attention_biases
        + 2 * shape.head_dim
        + 3 * shape.hidden_size * shape.intermediate_size
        + 2 * shape.hidden_size
    )
    embedding_and_head = (
        shape.vocab_size * shape.hidden_size * (1 if shape.tie_word_embeddings else 2)
    )
    return embedding_and_head + shape.num_hidden_layers * per_layer + shape.hidden_size


def qwen3_parameter_breakdown(
    config: Any = None,
    *,
    vocab_size: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    num_hidden_layers: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    tie_word_embeddings: bool | None = None,
    attention_bias: bool | None = None,
) -> dict[str, Any]:
    """Return the exact unique-parameter count for dense Transformers Qwen3.

    Qwen3's query/output widths are ``num_attention_heads * head_dim`` and
    its key/value widths are ``num_key_value_heads * head_dim``.  These widths
    are intentionally not assumed to equal ``hidden_size``.
    """

    shape = _shape_with_overrides(
        config,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=attention_bias,
    )
    vocab = shape.vocab_size
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size
    query_width = shape.num_attention_heads * shape.head_dim
    key_value_width = shape.num_key_value_heads * shape.head_dim

    attention_weights = {
        "q_proj": hidden * query_width,
        "k_proj": hidden * key_value_width,
        "v_proj": hidden * key_value_width,
        "o_proj": query_width * hidden,
    }
    attention_biases = {
        "q_proj": query_width if shape.attention_bias else 0,
        "k_proj": key_value_width if shape.attention_bias else 0,
        "v_proj": key_value_width if shape.attention_bias else 0,
        "o_proj": hidden if shape.attention_bias else 0,
    }
    attention_norms = {
        "q_norm": shape.head_dim,
        "k_norm": shape.head_dim,
    }
    swiglu_weights = {
        "gate_proj": hidden * intermediate,
        "up_proj": hidden * intermediate,
        "down_proj": intermediate * hidden,
    }
    decoder_norms = {
        "input_layernorm": hidden,
        "post_attention_layernorm": hidden,
    }
    per_layer = (
        sum(attention_weights.values())
        + sum(attention_biases.values())
        + sum(attention_norms.values())
        + sum(swiglu_weights.values())
        + sum(decoder_norms.values())
    )
    token_embedding = vocab * hidden
    logical_output_head = vocab * hidden
    unique_output_head = 0 if shape.tie_word_embeddings else logical_output_head
    all_decoder_layers = shape.num_hidden_layers * per_layer
    final_norm = hidden
    total = token_embedding + unique_output_head + all_decoder_layers + final_norm
    if total != _count_shape_parameters(shape):
        raise RuntimeError("Internal Qwen3 parameter formulas disagree.")

    return {
        "formula_version": FORMULA_VERSION,
        "shape": shape.as_dict(),
        "derived_dimensions": {
            "query_projection_width": query_width,
            "key_projection_width": key_value_width,
            "value_projection_width": key_value_width,
            "query_heads_per_key_value_head": (
                shape.num_attention_heads // shape.num_key_value_heads
            ),
        },
        "embedding_and_head": {
            "token_embedding": token_embedding,
            "output_head_logical": logical_output_head,
            "output_head_unique": unique_output_head,
            "tied": shape.tie_word_embeddings,
            "shared_parameter": (
                "model.embed_tokens.weight" if shape.tie_word_embeddings else None
            ),
        },
        "per_decoder_layer": {
            "attention_weights": attention_weights,
            "attention_biases": attention_biases,
            "attention_rmsnorms": attention_norms,
            "swiglu_weights": swiglu_weights,
            "decoder_rmsnorms": decoder_norms,
            "total": per_layer,
        },
        "components": {
            "token_embedding": token_embedding,
            "output_head_unique": unique_output_head,
            "decoder_layers": all_decoder_layers,
            "final_rmsnorm": final_norm,
        },
        "architecture_details": {
            "attention_bias_from_config": shape.attention_bias,
            "mlp_bias": False,
            "output_head_bias": False,
            "rmsnorm_bias": False,
            "qk_rmsnorm_width": "head_dim_each_not_per_head",
            "rotary_embedding_parameters": 0,
            "activation_parameters": 0,
            "embedding_padding_row_still_counted": True,
        },
        "num_hidden_layers": shape.num_hidden_layers,
        "all_decoder_layers": all_decoder_layers,
        "total": total,
    }


def count_qwen3_parameters(config: Any = None, **dimensions: Any) -> int:
    """Return only the exact unique-parameter total."""

    return _count_shape_parameters(_shape_with_overrides(config, **dimensions))


qwen3_parameter_count = count_qwen3_parameters


@dataclass(frozen=True, slots=True)
class Qwen3ShapeConstraints:
    """Finite deterministic choices and relational Qwen3 search constraints."""

    hidden_sizes: Any = _DEFAULT_HIDDEN_SIZES
    intermediate_sizes: Any = None
    layer_counts: Any = _DEFAULT_LAYER_COUNTS
    query_head_counts: Any = _DEFAULT_QUERY_HEAD_COUNTS
    key_value_head_counts: Any = _DEFAULT_KEY_VALUE_HEAD_COUNTS
    head_dims: Any = _DEFAULT_HEAD_DIMS
    intermediate_ratios: Any = _DEFAULT_INTERMEDIATE_RATIOS
    intermediate_multiple: int = 8
    require_attention_width_equals_hidden_size: bool = False


_CONSTRAINT_ALIASES = {
    "hidden_size": "hidden_sizes",
    "intermediate_size": "intermediate_sizes",
    "num_hidden_layers": "layer_counts",
    "num_attention_heads": "query_head_counts",
    "query_heads": "query_head_counts",
    "num_key_value_heads": "key_value_head_counts",
    "kv_heads": "key_value_head_counts",
    "head_dim": "head_dims",
}
_CONSTRAINT_FIELDS = frozenset(
    {
        "hidden_sizes",
        "intermediate_sizes",
        "layer_counts",
        "query_head_counts",
        "key_value_head_counts",
        "head_dims",
        "intermediate_ratios",
        "intermediate_multiple",
        "require_attention_width_equals_hidden_size",
    }
)


def _choices(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bool)):
        raw = [value]
    else:
        try:
            raw = list(value)
        except TypeError:
            raw = [value]
    if not raw:
        raise Qwen3TargetError(f"{name} must contain at least one choice.")
    return tuple(sorted({_exact_int(item, name, positive=True) for item in raw}))


def _ratios(value: Any) -> tuple[Fraction, ...]:
    if isinstance(value, (str, bytes)):
        raw = [value]
    else:
        try:
            raw = list(value)
        except TypeError:
            raw = [value]
    if not raw:
        raise Qwen3TargetError("intermediate_ratios must not be empty.")
    normalized = set()
    for item in raw:
        try:
            if isinstance(item, (tuple, list)):
                if len(item) != 2:
                    raise ValueError
                ratio = Fraction(
                    _exact_int(item[0], "intermediate ratio numerator", positive=True),
                    _exact_int(
                        item[1], "intermediate ratio denominator", positive=True
                    ),
                )
            else:
                ratio = Fraction(str(item)).limit_denominator(4096)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise Qwen3TargetError(
                "Each intermediate ratio must be positive numeric data or [numerator, denominator]."
            ) from exc
        if ratio <= 0:
            raise Qwen3TargetError("intermediate ratios must be positive.")
        normalized.add(ratio)
    return tuple(sorted(normalized))


def _constraint_source_values(
    constraints: Qwen3ShapeConstraints | Mapping[str, Any] | None,
) -> dict[str, Any]:
    defaults = Qwen3ShapeConstraints()
    values = {name: getattr(defaults, name) for name in _CONSTRAINT_FIELDS}
    if constraints is None:
        return values
    if isinstance(constraints, Qwen3ShapeConstraints):
        return {name: getattr(constraints, name) for name in _CONSTRAINT_FIELDS}
    if not isinstance(constraints, Mapping):
        raise Qwen3TargetError(
            "constraints must be Qwen3ShapeConstraints, a mapping, or None."
        )
    for raw_name, value in constraints.items():
        name = _CONSTRAINT_ALIASES.get(str(raw_name), str(raw_name))
        if name not in _CONSTRAINT_FIELDS:
            raise Qwen3TargetError(f"Unknown Qwen3 constraint: {raw_name}.")
        values[name] = value
    return values


def _normalized_constraints(
    constraints: Qwen3ShapeConstraints | Mapping[str, Any] | None,
    direct: Mapping[str, Any],
    aliases: Mapping[str, Any],
) -> dict[str, Any]:
    values = _constraint_source_values(constraints)
    assigned = set()
    for name, value in direct.items():
        if value is not None:
            values[name] = value
            assigned.add(name)
    for raw_name, value in aliases.items():
        name = _CONSTRAINT_ALIASES.get(raw_name, raw_name)
        if name not in _CONSTRAINT_FIELDS:
            raise Qwen3TargetError(f"Unknown Qwen3 constraint: {raw_name}.")
        if name in assigned:
            raise Qwen3TargetError(f"Constraint {name} was provided more than once.")
        values[name] = value
        assigned.add(name)

    explicit_intermediate = values["intermediate_sizes"] is not None
    normalized = {
        "hidden_sizes": _choices(values["hidden_sizes"], "hidden_sizes"),
        "intermediate_sizes": (
            _choices(values["intermediate_sizes"], "intermediate_sizes")
            if explicit_intermediate
            else None
        ),
        "layer_counts": _choices(values["layer_counts"], "layer_counts"),
        "query_head_counts": _choices(values["query_head_counts"], "query_head_counts"),
        "key_value_head_counts": _choices(
            values["key_value_head_counts"], "key_value_head_counts"
        ),
        "head_dims": _choices(values["head_dims"], "head_dims"),
        "intermediate_ratios": _ratios(values["intermediate_ratios"]),
        "intermediate_multiple": _exact_int(
            values["intermediate_multiple"],
            "intermediate_multiple",
            positive=True,
        ),
        "require_attention_width_equals_hidden_size": _strict_bool(
            values["require_attention_width_equals_hidden_size"],
            "require_attention_width_equals_hidden_size",
        ),
        "intermediate_source": (
            "explicit-choices" if explicit_intermediate else "hidden-ratio-derived"
        ),
    }
    return normalized


def _derived_intermediate_sizes(
    hidden_size: int, ratios: tuple[Fraction, ...], multiple: int
) -> tuple[int, ...]:
    values = set()
    for ratio in ratios:
        nearest_integer = (
            hidden_size * ratio.numerator + ratio.denominator // 2
        ) // ratio.denominator
        aligned = ((nearest_integer + multiple // 2) // multiple) * multiple
        values.add(max(multiple, aligned))
    return tuple(sorted(values))


def _constraints_report(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hidden_sizes": list(values["hidden_sizes"]),
        "intermediate_sizes": (
            None
            if values["intermediate_sizes"] is None
            else list(values["intermediate_sizes"])
        ),
        "layer_counts": list(values["layer_counts"]),
        "query_head_counts": list(values["query_head_counts"]),
        "key_value_head_counts": list(values["key_value_head_counts"]),
        "head_dims": list(values["head_dims"]),
        "intermediate_ratios": [
            [ratio.numerator, ratio.denominator]
            for ratio in values["intermediate_ratios"]
        ],
        "intermediate_multiple": values["intermediate_multiple"],
        "intermediate_source": values["intermediate_source"],
        "require_attention_width_equals_hidden_size": values[
            "require_attention_width_equals_hidden_size"
        ],
        "require_query_heads_divisible_by_key_value_heads": True,
    }


def _normalize_config_overrides(
    config_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if config_overrides is None:
        return {}
    if not isinstance(config_overrides, Mapping):
        raise Qwen3TargetError("config_overrides must be a mapping or None.")
    values = dict(config_overrides)
    collision = sorted(_COUNTED_CONFIG_FIELDS.intersection(values))
    if collision:
        raise Qwen3TargetError(
            "Counted Qwen3 fields must be planner arguments, not config_overrides: "
            + ", ".join(collision)
        )
    unsupported = sorted(set(values) - _MATERIALIZABLE_CONFIG_FIELDS)
    if unsupported:
        raise Qwen3TargetError(
            "Unsupported Qwen3 config override fields: " + ", ".join(unsupported)
        )
    _canonical_json(values)
    return values


def _tokenizer_size(tokenizer: Any) -> int:
    try:
        size = len(tokenizer)
    except (TypeError, AttributeError) as exc:
        raise Qwen3TargetError(
            "The supplied tokenizer must define len(tokenizer)."
        ) from exc
    return _exact_int(size, "tokenizer length", positive=True)


def _candidate_summary(
    shape: Qwen3Shape, parameter_count: int, budget: int
) -> dict[str, Any]:
    delta = parameter_count - budget
    return {
        "shape": shape.as_dict(),
        "parameter_count": parameter_count,
        "under_budget": delta <= 0,
        "budget_delta": delta,
        "absolute_budget_distance": abs(delta),
        "budget_utilization": parameter_count / budget,
    }


def _compression_report(
    selected_count: int,
    budget: int,
    source_count: int | None,
) -> dict[str, Any]:
    report = {
        "selected_parameter_count": selected_count,
        "parameter_budget": budget,
        "selected_to_budget_fraction": selected_count / budget,
        "budget_headroom_parameters": budget - selected_count,
        "at_or_under_budget": selected_count <= budget,
        "source_parameter_count": source_count,
        "source_to_selected_ratio": None,
        "selected_to_source_fraction": None,
        "source_parameter_reduction": None,
        "source_parameter_reduction_fraction": None,
    }
    if source_count is not None:
        report.update(
            {
                "source_to_selected_ratio": source_count / selected_count,
                "selected_to_source_fraction": selected_count / source_count,
                "source_parameter_reduction": source_count - selected_count,
                "source_parameter_reduction_fraction": (
                    (source_count - selected_count) / source_count
                ),
            }
        )
    return report


def plan_qwen3_target(
    parameter_budget: int,
    *,
    vocab_size: int | None = None,
    tokenizer: Any = None,
    constraints: Qwen3ShapeConstraints | Mapping[str, Any] | None = None,
    hidden_sizes: Iterable[int] | int | None = None,
    intermediate_sizes: Iterable[int] | int | None = None,
    layer_counts: Iterable[int] | int | None = None,
    query_head_counts: Iterable[int] | int | None = None,
    key_value_head_counts: Iterable[int] | int | None = None,
    head_dims: Iterable[int] | int | None = None,
    tie_word_embeddings: bool = False,
    attention_bias: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
    source_config: Any = None,
    source_parameter_count: int | None = None,
    source_name: str | None = None,
    require_under_budget: bool = False,
    ranked_candidate_limit: int = 20,
    **constraint_aliases: Any,
) -> dict[str, Any]:
    """Search finite valid Qwen3 shapes for a user parameter budget.

    Selection first chooses the closest candidate at or below the budget.  If
    no such candidate exists, the closest candidate by absolute distance is
    used unless ``require_under_budget`` is true.  Exact ties use ascending
    ``(hidden, intermediate, layers, query_heads, kv_heads, head_dim)``.
    """

    budget = _exact_int(parameter_budget, "parameter_budget", positive=True)
    tie = _strict_bool(tie_word_embeddings, "tie_word_embeddings")
    bias = _strict_bool(attention_bias, "attention_bias")
    require_under = _strict_bool(require_under_budget, "require_under_budget")
    rank_limit = _exact_int(
        ranked_candidate_limit, "ranked_candidate_limit", positive=True
    )

    tokenizer_length = _tokenizer_size(tokenizer) if tokenizer is not None else None
    if vocab_size is None:
        if tokenizer_length is not None:
            resolved_vocab = tokenizer_length
        elif source_config is not None:
            resolved_vocab = _exact_int(
                _config_value(source_config, "vocab_size"),
                "source_config.vocab_size",
                positive=True,
            )
        else:
            raise Qwen3TargetError(
                "vocab_size, tokenizer, or source_config with vocab_size is required."
            )
    else:
        resolved_vocab = _exact_int(vocab_size, "vocab_size", positive=True)
    if tokenizer_length is not None and tokenizer_length > resolved_vocab:
        raise Qwen3TargetError(
            "vocab_size cannot be smaller than the supplied tokenizer length."
        )

    normalized = _normalized_constraints(
        constraints,
        {
            "hidden_sizes": hidden_sizes,
            "intermediate_sizes": intermediate_sizes,
            "layer_counts": layer_counts,
            "query_head_counts": query_head_counts,
            "key_value_head_counts": key_value_head_counts,
            "head_dims": head_dims,
        },
        constraint_aliases,
    )
    overrides = _normalize_config_overrides(config_overrides)

    source_breakdown = None
    if source_config is not None:
        source_breakdown = qwen3_parameter_breakdown(source_config)
    if source_parameter_count is not None:
        resolved_source_count = _exact_int(
            source_parameter_count, "source_parameter_count", positive=True
        )
        source_count_provenance = "explicit"
    elif source_breakdown is not None:
        resolved_source_count = int(source_breakdown["total"])
        source_count_provenance = "qwen3-formula"
    else:
        resolved_source_count = None
        source_count_provenance = None

    constraints_provenance = _constraints_report(normalized)
    base_report = {
        "schema": PLAN_SCHEMA,
        "operation": "plan_qwen3_random_blank_target",
        "status": "searching",
        "parameter_budget": budget,
        "vocab_provenance": {
            "vocab_size": resolved_vocab,
            "tokenizer_supplied": tokenizer is not None,
            "tokenizer_length": tokenizer_length,
            "reserved_rows_beyond_tokenizer": (
                None if tokenizer_length is None else resolved_vocab - tokenizer_length
            ),
        },
        "fixed_counted_config": {
            "tie_word_embeddings": tie,
            "attention_bias": bias,
        },
        "non_parameter_config_overrides": overrides,
        "constraints": constraints_provenance,
        "ranking": {
            "selection": "closest-at-or-under-budget-else-closest-absolute",
            "under_budget_precedes_over_budget": True,
            "distance": "absolute-parameter-count-difference",
            "exact_tie_break": (
                "ascending-(hidden,intermediate,layers,query_heads,kv_heads,head_dim)"
            ),
            "ranked_candidate_limit": rank_limit,
            "require_under_budget": require_under,
        },
        "source": {
            "name": source_name,
            "parameter_count": resolved_source_count,
            "parameter_count_provenance": source_count_provenance,
            "formula_breakdown": source_breakdown,
        },
        "parameter_formula": {
            "version": FORMULA_VERSION,
            "framework_dependency": False,
            "counts_unique_parameter_storage": True,
        },
    }

    ranked: list[tuple[tuple[Any, ...], Qwen3Shape, int]] = []
    best_under: tuple[tuple[Any, ...], Qwen3Shape, int] | None = None
    best_absolute: tuple[tuple[Any, ...], Qwen3Shape, int] | None = None
    stats = {
        "cartesian_candidate_count": 0,
        "valid_candidate_count": 0,
        "under_budget_candidate_count": 0,
        "over_budget_candidate_count": 0,
        "rejected_gqa_divisibility_count": 0,
        "rejected_attention_width_count": 0,
    }

    for hidden in normalized["hidden_sizes"]:
        intermediates = normalized["intermediate_sizes"]
        if intermediates is None:
            intermediates = _derived_intermediate_sizes(
                hidden,
                normalized["intermediate_ratios"],
                normalized["intermediate_multiple"],
            )
        for intermediate in intermediates:
            for layers in normalized["layer_counts"]:
                for query_heads in normalized["query_head_counts"]:
                    for key_value_heads in normalized["key_value_head_counts"]:
                        for candidate_head_dim in normalized["head_dims"]:
                            stats["cartesian_candidate_count"] += 1
                            if query_heads % key_value_heads:
                                stats["rejected_gqa_divisibility_count"] += 1
                                continue
                            if (
                                normalized["require_attention_width_equals_hidden_size"]
                                and query_heads * candidate_head_dim != hidden
                            ):
                                stats["rejected_attention_width_count"] += 1
                                continue
                            shape = Qwen3Shape(
                                vocab_size=resolved_vocab,
                                hidden_size=hidden,
                                intermediate_size=intermediate,
                                num_hidden_layers=layers,
                                num_attention_heads=query_heads,
                                num_key_value_heads=key_value_heads,
                                head_dim=candidate_head_dim,
                                tie_word_embeddings=tie,
                                attention_bias=bias,
                            )
                            total = count_qwen3_parameters(shape)
                            delta = total - budget
                            stats["valid_candidate_count"] += 1
                            if delta <= 0:
                                stats["under_budget_candidate_count"] += 1
                                selection_key = (abs(delta), shape.search_tuple())
                                if best_under is None or selection_key < best_under[0]:
                                    best_under = (selection_key, shape, total)
                            else:
                                stats["over_budget_candidate_count"] += 1
                            absolute_key = (
                                abs(delta),
                                0 if delta <= 0 else 1,
                                shape.search_tuple(),
                            )
                            if best_absolute is None or absolute_key < best_absolute[0]:
                                best_absolute = (absolute_key, shape, total)
                            rank_key = (
                                0 if delta <= 0 else 1,
                                abs(delta),
                                shape.search_tuple(),
                            )
                            bisect.insort(ranked, (rank_key, shape, total))
                            if len(ranked) > rank_limit:
                                ranked.pop()

    base_report["search"] = stats
    input_payload = {
        "parameter_budget": budget,
        "vocab_size": resolved_vocab,
        "fixed_counted_config": base_report["fixed_counted_config"],
        "constraints": constraints_provenance,
        "non_parameter_config_overrides": overrides,
        "require_under_budget": require_under,
        "source_parameter_count": resolved_source_count,
    }
    input_hash = _json_sha256(input_payload)

    if stats["valid_candidate_count"] == 0:
        failure = {
            **base_report,
            "status": "infeasible",
            "infeasibility": {
                "reason": "no_valid_shape",
                "message": (
                    "No Cartesian candidate satisfies Qwen3 GQA and attention-width constraints."
                ),
            },
            "hashes": {"input_sha256": input_hash},
        }
        raise Qwen3PlanInfeasibleError(
            "No valid Qwen3 shape satisfies the supplied constraints.", failure
        )
    if require_under and best_under is None:
        assert best_absolute is not None
        nearest = _candidate_summary(best_absolute[1], best_absolute[2], budget)
        failure = {
            **base_report,
            "status": "infeasible",
            "infeasibility": {
                "reason": "no_candidate_at_or_under_budget",
                "nearest_candidate": nearest,
            },
            "hashes": {"input_sha256": input_hash},
        }
        raise Qwen3PlanInfeasibleError(
            "Valid Qwen3 shapes exist, but none is at or under the budget.",
            failure,
        )

    selected_entry = best_under if best_under is not None else best_absolute
    assert selected_entry is not None
    selected_shape = selected_entry[1]
    selected_count = selected_entry[2]
    selected_breakdown = qwen3_parameter_breakdown(selected_shape)
    selected_config = {**selected_shape.as_dict(), **overrides}
    layer_types = selected_config.get("layer_types")
    if layer_types is not None and len(layer_types) != selected_shape.num_hidden_layers:
        raise Qwen3TargetError(
            "config_overrides.layer_types must have one entry per selected hidden layer."
        )

    ranked_summaries = [
        {
            "rank": index + 1,
            **_candidate_summary(shape, total, budget),
        }
        for index, (_key, shape, total) in enumerate(ranked)
    ]
    best_absolute_summary = None
    if best_absolute is not None:
        best_absolute_summary = _candidate_summary(
            best_absolute[1], best_absolute[2], budget
        )
    compression = _compression_report(selected_count, budget, resolved_source_count)
    plan_hash_payload = {
        "input_sha256": input_hash,
        "selected_config": selected_config,
        "selected_parameter_count": selected_count,
        "formula_version": FORMULA_VERSION,
    }
    plan_hash = _json_sha256(plan_hash_payload)
    report = {
        **base_report,
        "status": "feasible",
        "selected": _candidate_summary(selected_shape, selected_count, budget),
        "selected_parameter_breakdown": selected_breakdown,
        "closest_absolute_candidate": best_absolute_summary,
        "ranked_candidates": ranked_summaries,
        "compression": compression,
        "shape_provenance": {
            "selected_config": selected_config,
            "search_tuple": list(selected_shape.search_tuple()),
            "valid_qwen3_gqa": True,
            "attention_width_may_differ_from_hidden_size": not normalized[
                "require_attention_width_equals_hidden_size"
            ],
            "materialization_class": "transformers.Qwen3ForCausalLM",
            "pretrained_child_weights_required": False,
        },
        "hashes": {
            "algorithm": "sha256",
            "input_sha256": input_hash,
            "plan_sha256": plan_hash,
        },
    }
    return {
        "config": selected_config,
        "shape": selected_shape.as_dict(),
        "parameter_count": selected_count,
        "parameter_breakdown": selected_breakdown,
        "under_budget": selected_count <= budget,
        "budget_delta": selected_count - budget,
        "compression": compression,
        "ranked_candidates": ranked_summaries,
        "plan_sha256": plan_hash,
        "report": report,
    }


plan_qwen3_blank_target = plan_qwen3_target


def _plan_config(plan_or_config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(plan_or_config, Qwen3Shape):
        return plan_or_config.as_dict(), {
            "provenance": "direct-qwen3-shape",
            "plan_sha256": None,
            "parameter_budget": None,
        }
    if isinstance(plan_or_config, Mapping):
        raw_config = plan_or_config.get("config")
        if not isinstance(raw_config, Mapping):
            shape_provenance = plan_or_config.get("shape_provenance")
            if isinstance(shape_provenance, Mapping):
                raw_config = shape_provenance.get("selected_config")
        if not isinstance(raw_config, Mapping):
            selected = plan_or_config.get("selected")
            if isinstance(selected, Mapping):
                raw_config = selected.get("shape")
        if not isinstance(raw_config, Mapping):
            raw_config = plan_or_config
        config = {
            key: value
            for key, value in raw_config.items()
            if key in _COUNTED_CONFIG_FIELDS or key in _MATERIALIZABLE_CONFIG_FIELDS
        }
        report = plan_or_config.get("report")
        if not isinstance(report, Mapping):
            report = (
                plan_or_config if plan_or_config.get("schema") == PLAN_SCHEMA else {}
            )
        hashes = report.get("hashes")
        report_plan_hash = (
            hashes.get("plan_sha256") if isinstance(hashes, Mapping) else None
        )
        selected = report.get("selected")
        report_parameter_count = (
            selected.get("parameter_count") if isinstance(selected, Mapping) else None
        )
        metadata = {
            "provenance": (
                "deterministic-shape-plan"
                if "config" in plan_or_config
                else "direct-config-mapping"
            ),
            "plan_sha256": plan_or_config.get("plan_sha256") or report_plan_hash,
            "parameter_budget": report.get("parameter_budget"),
            "report": dict(report),
            "expected_parameter_count": plan_or_config.get("parameter_count")
            or report_parameter_count,
        }
        return config, metadata
    shape = Qwen3Shape.from_config(plan_or_config)
    return shape.as_dict(), {
        "provenance": "direct-config-object",
        "plan_sha256": None,
        "parameter_budget": None,
    }


def _torch_dtype(torch: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        name = value.strip().lower().replace("torch.", "")
        aliases = {"fp16": "float16", "fp32": "float32", "bf16": "bfloat16"}
        name = aliases.get(name, name)
        if name not in {"float16", "float32", "float64", "bfloat16"}:
            raise Qwen3TargetError(
                "torch_dtype must be float16, bfloat16, float32, float64, or None."
            )
        return getattr(torch, name)
    if value in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        return value
    raise Qwen3TargetError(
        "torch_dtype must be a supported torch floating dtype or its string name."
    )


def _state_dict_sha256(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(target: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(item for item in target.rglob("*") if item.is_file()):
        if path.name == TARGET_REPORT_FILENAME:
            continue
        files.append(
            {
                "path": path.relative_to(target).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return files


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_qwen3_target(
    plan_or_config: Any,
    output_dir: str | Path,
    tokenizer: Any,
    *,
    seed: int = 0,
    torch_dtype: Any = "float32",
    safe_serialization: bool = True,
) -> dict[str, Any]:
    """Create and save a seeded random Qwen3 target plus supplied tokenizer.

    No model ``from_pretrained`` path is used.  The only model construction is
    ``Qwen3ForCausalLM(Qwen3Config(**selected_config))`` on CPU.
    """

    seed_value = _exact_int(seed, "seed")
    safe = _strict_bool(safe_serialization, "safe_serialization")
    if not callable(getattr(tokenizer, "save_pretrained", None)):
        raise Qwen3TargetError("The supplied tokenizer must define save_pretrained().")
    tokenizer_length = _tokenizer_size(tokenizer)
    config_data, plan_metadata = _plan_config(plan_or_config)
    shape = Qwen3Shape.from_config(config_data)
    expected_count = count_qwen3_parameters(shape)
    planned_count = plan_metadata.get("expected_parameter_count")
    if (
        planned_count is not None
        and _exact_int(planned_count, "planned parameter_count", positive=True)
        != expected_count
    ):
        raise Qwen3TargetError(
            "The plan's parameter_count does not match its selected Qwen3 config."
        )
    if tokenizer_length > shape.vocab_size:
        raise Qwen3TargetError(
            "The supplied tokenizer is larger than the planned Qwen3 vocabulary."
        )

    try:
        import torch
        import transformers
        from transformers import Qwen3Config, Qwen3ForCausalLM
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "materialize_qwen3_target requires torch and Transformers with Qwen3 support."
        ) from exc

    dtype = _torch_dtype(torch, torch_dtype)
    constructor_kwargs = {
        key: value
        for key, value in config_data.items()
        if key in _COUNTED_CONFIG_FIELDS or key in _MATERIALIZABLE_CONFIG_FIELDS
    }
    tokenizer_special_ids = {}
    for field in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if field not in constructor_kwargs:
            value = getattr(tokenizer, field, None)
            if value is not None:
                constructor_kwargs[field] = value
                tokenizer_special_ids[field] = value
    layer_types = constructor_kwargs.get("layer_types")
    if layer_types is not None and len(layer_types) != shape.num_hidden_layers:
        raise Qwen3TargetError(
            "layer_types must have one entry per selected hidden layer."
        )

    config = Qwen3Config(**constructor_kwargs)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed_value)
        with torch.no_grad():
            model = Qwen3ForCausalLM(config)
            if dtype is not None:
                model.to(dtype=dtype)
    model.eval()
    actual_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if actual_count != expected_count:
        raise RuntimeError(
            "Qwen3 formula/model parameter mismatch: "
            f"formula={expected_count}, actual={actual_count}."
        )
    parameters_with_grad = sum(
        int(parameter.grad is not None) for parameter in model.parameters()
    )
    if parameters_with_grad:
        raise RuntimeError(
            "Random Qwen3 construction unexpectedly populated gradients."
        )
    state_hash = _state_dict_sha256(torch, model)

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=safe)
    tokenizer_result = tokenizer.save_pretrained(target)
    tokenizer_files = (
        [str(item) for item in tokenizer_result]
        if isinstance(tokenizer_result, (tuple, list))
        else []
    )
    artifact_files = _artifact_files(target)
    config_dict = model.config.to_dict()
    report = {
        "schema": ARTIFACT_SCHEMA,
        "operation": "materialize_qwen3_random_blank_target",
        "output_dir": str(target),
        "seed": seed_value,
        "torch_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "safe_serialization": safe,
        "random_initialization": {
            "constructor": "Qwen3ForCausalLM(Qwen3Config(...))",
            "seed": seed_value,
            "rng_scope": "forked-cpu-torch-rng",
            "pretrained_child_weights_loaded": False,
            "pretrained_model_loader_called": False,
            "state_dict_sha256": state_hash,
        },
        "training": {
            "performed": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parameters_with_grad": parameters_with_grad,
        },
        "shape": shape.as_dict(),
        "saved_config": config_dict,
        "parameter_count": {
            "formula": expected_count,
            "actual_unique_model_parameters": actual_count,
            "matches": actual_count == expected_count,
            "breakdown": qwen3_parameter_breakdown(shape),
        },
        "shape_plan": plan_metadata,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "length": tokenizer_length,
            "model_vocab_size": shape.vocab_size,
            "reserved_model_rows": shape.vocab_size - tokenizer_length,
            "supplied_object_saved_directly": True,
            "special_token_ids_applied_to_config": tokenizer_special_ids,
            "save_pretrained_result": tokenizer_files,
        },
        "software": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
        "artifacts": {
            "metadata_file": TARGET_REPORT_FILENAME,
            "files": artifact_files,
        },
    }
    report = json.loads(_canonical_json(report))
    _write_json(target / TARGET_REPORT_FILENAME, report)
    return report


materialize_qwen3_blank_target = materialize_qwen3_target


__all__ = [
    "ARTIFACT_SCHEMA",
    "FORMULA_VERSION",
    "PLAN_SCHEMA",
    "Qwen3PlanInfeasibleError",
    "Qwen3Shape",
    "Qwen3ShapeConstraints",
    "Qwen3TargetError",
    "Qwen3TargetInfeasibleError",
    "TARGET_REPORT_FILENAME",
    "count_qwen3_parameters",
    "materialize_qwen3_blank_target",
    "materialize_qwen3_target",
    "plan_qwen3_blank_target",
    "plan_qwen3_target",
    "qwen3_parameter_breakdown",
    "qwen3_parameter_count",
]
