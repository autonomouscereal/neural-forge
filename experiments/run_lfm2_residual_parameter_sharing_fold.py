#!/usr/bin/env python3
"""Run a zero-training residual-parameter-sharing experiment on local LFM2.

The runner consumes an already sealed parent circuit cache.  Cached post-SwiGLU
rows select disjoint FFN pairs and align their neurons; no parent forward is
allowed.  One member of every pair remains byte-for-byte intact and the other
is represented by the preserved base plus deterministic truncated-SVD
residuals.

Only the live candidate executes.  It must pass both the cached-parent
trajectory gate and the sealed 28-case whole-output gate.  Exact token or text
matching is diagnostic, never a promotion requirement.  This custom residual
topology does not yet have an authenticated standalone serializer, so even a
gate-passing candidate is rolled back exactly and published with
``standalone_checkpoint_not_yet_implemented`` rather than writing a dense or
otherwise fraudulent checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

try:
    from experiments import run_lfm2_residual_attention_fold as fold_runtime
except ImportError:  # Direct execution from the experiments directory.
    import run_lfm2_residual_attention_fold as fold_runtime

from neural_forge.dynamic_lfm2_behavior import (
    Lfm2WholeOutputBehaviorEvaluator,
    load_lfm2_cached_whole_output_baseline,
)
from neural_forge.hf_lfm2_canonical import inspect_lfm2_layer
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_parameter_sharing import (
    seal_lfm2_activation_evidence,
)
from neural_forge.hf_lfm2_residual_parameter_sharing import (
    InstalledLfm2ResidualParameterSharing,
    Lfm2ResidualParameterSharingPlan,
    Lfm2SharedBaseResidualLinear,
    install_lfm2_residual_parameter_sharing_plan,
    lfm2_residual_parameter_sharing_savings,
    plan_lfm2_residual_parameter_sharing,
)


RUN_SCHEMA = "neural-forge-lfm2-residual-parameter-sharing-fold-run-v1"
PUBLICATION_BLOCK_STATUS = "standalone_checkpoint_not_yet_implemented"
EXPECTED_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_LAYER_COUNT = 14
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_INTERMEDIATE_SIZE = 2560
EXPECTED_WHOLE_OUTPUT_CASE_COUNT = 28
DEFAULT_PAIR_COUNT = 4
DEFAULT_RANK = 64
DEFAULT_PRESERVE: Literal["later"] = "later"
DEFAULT_MAX_EVIDENCE_ROWS = 256
DEFAULT_MINIMUM_RETENTION = 0.90
DEFAULT_MAXIMUM_CASE_REGRESSIONS = 2
DEFAULT_MINIMUM_RESIDUAL_COSINE = 0.80
_ROLES = ("w1", "w3", "w2")
_ORIENTATIONS = ("earlier", "later")


class Lfm2ResidualParameterSharingRunnerError(RuntimeError):
    """A runner input or authenticated invariant failed closed."""


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
        raise Lfm2ResidualParameterSharingRunnerError(
            "Run evidence is not finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} must be a non-negative integer."
        )
    return value


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} must be a finite float."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} must be a finite float."
        ) from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} is outside its valid range."
        )
    return result


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} must be an explicit local directory, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} must be an explicit local file, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2ResidualParameterSharingRunnerError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def parse_weight_pair(value: str) -> tuple[float, float]:
    """Parse two finite non-negative weights with a positive total."""

    try:
        pieces = tuple(float(piece.strip()) for piece in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "weights must be two comma-separated finite numbers"
        ) from exc
    if (
        len(pieces) != 2
        or any(not math.isfinite(piece) or piece < 0.0 for piece in pieces)
        or sum(pieces) <= 0.0
    ):
        raise argparse.ArgumentTypeError(
            "weights must be two finite non-negative numbers with a positive total"
        )
    return pieces


def parse_layer_pairs(value: str) -> tuple[tuple[int, int], ...]:
    """Parse disjoint ``earlier:later`` layer pairs."""

    if not value.strip():
        raise argparse.ArgumentTypeError("layer pairs cannot be empty")
    result: list[tuple[int, int]] = []
    used: set[int] = set()
    for raw_pair in value.split(","):
        pieces = raw_pair.strip().split(":")
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(
                "layer pairs must use left:right comma-separated syntax"
            )
        try:
            left, right = (int(piece) for piece in pieces)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "layer-pair indices must be integers"
            ) from exc
        if left < 0 or right < 0 or left == right:
            raise argparse.ArgumentTypeError(
                "layer-pair indices must be non-negative and distinct"
            )
        pair = (min(left, right), max(left, right))
        if pair[0] in used or pair[1] in used:
            raise argparse.ArgumentTypeError("layer pairs must be disjoint")
        used.update(pair)
        result.append(pair)
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layer pairs must be unique")
    return tuple(sorted(result))


def parse_preserve(value: str) -> str | tuple[str, ...]:
    """Parse one orientation or one orientation per selected pair."""

    pieces = tuple(piece.strip().lower() for piece in value.split(","))
    if not pieces or any(piece not in _ORIENTATIONS for piece in pieces):
        raise argparse.ArgumentTypeError(
            "preserve must be earlier, later, or a comma-separated orientation list"
        )
    return pieces[0] if len(pieces) == 1 else pieces


@dataclass(frozen=True, slots=True)
class ResidualParameterSharingRunConfig:
    """Authenticated inputs and exact physical target for one experiment."""

    parent_model: Path
    parent_cache: Path
    cached_parent_outputs: Path
    cached_parent_outputs_sha256: str
    work_root: Path
    checkpoint_destination: Path
    report_path: Path
    tokenizer_directory: Path | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    pair_count: int = DEFAULT_PAIR_COUNT
    rank: int = DEFAULT_RANK
    preserve: str | tuple[str, ...] = DEFAULT_PRESERVE
    layer_pairs: tuple[tuple[int, int], ...] | None = None
    max_evidence_rows: int = DEFAULT_MAX_EVIDENCE_ROWS
    pair_score_weights: tuple[float, float] = (0.5, 0.5)
    exact_assignment_limit: int = 256
    correlation_chunk_size: int = 256
    variance_floor: float = 0.0
    expected_parent_unique_parameters: int = (
        EXPECTED_PARENT_UNIQUE_PARAMETERS
    )
    expected_layer_count: int = EXPECTED_LAYER_COUNT
    expected_hidden_size: int = EXPECTED_HIDDEN_SIZE
    expected_intermediate_size: int = EXPECTED_INTERMEDIATE_SIZE
    expected_whole_output_case_count: int = (
        EXPECTED_WHOLE_OUTPUT_CASE_COUNT
    )
    minimum_residual_cosine: float = DEFAULT_MINIMUM_RESIDUAL_COSINE
    whole_output_max_new_tokens: int = 64
    whole_output_maximum_prompt_tokens: int = 512
    whole_output_fragment_minimum_characters: int = 12
    whole_output_maximum_new_empty_responses: int = 2
    whole_output_maximum_new_fragment_responses: int = 2
    whole_output_minimum_parent_nonempty_retention: float = (
        DEFAULT_MINIMUM_RETENTION
    )
    whole_output_minimum_task_score_retention: float = (
        DEFAULT_MINIMUM_RETENTION
    )
    whole_output_maximum_parent_task_score_regressions: int = (
        DEFAULT_MAXIMUM_CASE_REGRESSIONS
    )
    whole_output_maximum_new_generation_limit_saturations: int = 2
    whole_output_maximum_repeated_ngram_share_regressions: int = 2
    whole_output_maximum_repeated_bigram_share_delta: float = 0.10

    def __post_init__(self) -> None:
        parent_model = _local_directory(self.parent_model, "parent_model")
        parent_cache = _local_directory(self.parent_cache, "parent_cache")
        outputs = _local_file(
            self.cached_parent_outputs,
            "cached_parent_outputs",
        )
        tokenizer = (
            parent_model
            if self.tokenizer_directory is None
            else _local_directory(
                self.tokenizer_directory,
                "tokenizer_directory",
            )
        )
        object.__setattr__(self, "parent_model", parent_model)
        object.__setattr__(self, "parent_cache", parent_cache)
        object.__setattr__(self, "cached_parent_outputs", outputs)
        object.__setattr__(self, "tokenizer_directory", tokenizer)
        for name in ("work_root", "checkpoint_destination", "report_path"):
            path = Path(getattr(self, name)).expanduser().absolute()
            object.__setattr__(self, name, path)
        for protected in (parent_model, parent_cache, tokenizer):
            for output in (
                self.work_root,
                self.checkpoint_destination,
                self.report_path,
            ):
                if _is_within(output, protected):
                    raise Lfm2ResidualParameterSharingRunnerError(
                        "Work, checkpoint, and report paths must not be inside "
                        "source model/cache directories."
                    )
        if self.checkpoint_destination == self.report_path:
            raise Lfm2ResidualParameterSharingRunnerError(
                "checkpoint_destination and report_path must differ."
            )
        if self.checkpoint_destination.exists():
            raise Lfm2ResidualParameterSharingRunnerError(
                "checkpoint_destination already exists; refusing to replace it."
            )
        if self.report_path.exists():
            raise Lfm2ResidualParameterSharingRunnerError(
                "report_path already exists; refusing to replace it."
            )
        for name in (
            "pair_count",
            "rank",
            "max_evidence_rows",
            "correlation_chunk_size",
            "expected_parent_unique_parameters",
            "expected_layer_count",
            "expected_hidden_size",
            "expected_intermediate_size",
            "expected_whole_output_case_count",
            "whole_output_max_new_tokens",
            "whole_output_maximum_prompt_tokens",
            "whole_output_fragment_minimum_characters",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        if self.pair_count > self.expected_layer_count // 2:
            raise Lfm2ResidualParameterSharingRunnerError(
                "pair_count cannot exceed half the expected layer count."
            )
        if self.rank > min(
            self.expected_hidden_size,
            self.expected_intermediate_size,
        ):
            raise Lfm2ResidualParameterSharingRunnerError(
                "rank cannot exceed the full projection rank."
            )
        if self.layer_pairs is not None:
            if len(self.layer_pairs) != self.pair_count:
                raise Lfm2ResidualParameterSharingRunnerError(
                    "Explicit layer-pair count must equal pair_count."
                )
            normalized_pairs: list[tuple[int, int]] = []
            used: set[int] = set()
            for pair in self.layer_pairs:
                if (
                    not isinstance(pair, Sequence)
                    or isinstance(pair, (str, bytes))
                    or len(pair) != 2
                    or any(
                        isinstance(index, bool) or not isinstance(index, int)
                        for index in pair
                    )
                ):
                    raise Lfm2ResidualParameterSharingRunnerError(
                        "Each explicit layer pair must contain two integers."
                    )
                left, right = pair
                if (
                    left == right
                    or left < 0
                    or right < 0
                    or left >= self.expected_layer_count
                    or right >= self.expected_layer_count
                    or left in used
                    or right in used
                ):
                    raise Lfm2ResidualParameterSharingRunnerError(
                        "Explicit layer pairs must be disjoint, distinct, and "
                        "inside the expected layer range."
                    )
                used.update((left, right))
                normalized_pairs.append((min(left, right), max(left, right)))
            object.__setattr__(
                self,
                "layer_pairs",
                tuple(sorted(normalized_pairs)),
            )
        orientations = (
            (self.preserve,) * self.pair_count
            if isinstance(self.preserve, str)
            else tuple(self.preserve)
        )
        if (
            len(orientations) != self.pair_count
            or any(value not in _ORIENTATIONS for value in orientations)
        ):
            raise Lfm2ResidualParameterSharingRunnerError(
                "preserve must be earlier/later or one orientation per pair."
            )
        object.__setattr__(self, "preserve", orientations)
        for name in (
            "whole_output_maximum_new_empty_responses",
            "whole_output_maximum_new_fragment_responses",
            "whole_output_maximum_parent_task_score_regressions",
            "whole_output_maximum_new_generation_limit_saturations",
            "whole_output_maximum_repeated_ngram_share_regressions",
            "exact_assignment_limit",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        if self.exact_assignment_limit > 512:
            raise Lfm2ResidualParameterSharingRunnerError(
                "exact_assignment_limit cannot exceed 512."
            )
        for name in (
            "whole_output_minimum_parent_nonempty_retention",
            "whole_output_minimum_task_score_retention",
            "whole_output_maximum_repeated_bigram_share_delta",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "minimum_residual_cosine",
            _finite_float(
                self.minimum_residual_cosine,
                "minimum_residual_cosine",
                minimum=-1.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "variance_floor",
            _finite_float(
                self.variance_floor,
                "variance_floor",
                minimum=0.0,
            ),
        )
        weights = tuple(float(value) for value in self.pair_score_weights)
        if (
            len(weights) != 2
            or any(not math.isfinite(value) or value < 0.0 for value in weights)
            or sum(weights) <= 0.0
        ):
            raise Lfm2ResidualParameterSharingRunnerError(
                "pair_score_weights must contain two finite non-negative weights."
            )
        object.__setattr__(self, "pair_score_weights", weights)
        if self.expected_pair_savings <= 0:
            raise Lfm2ResidualParameterSharingRunnerError(
                "rank does not produce positive physical compression."
            )
        if self.expected_child_unique_parameters <= 0:
            raise Lfm2ResidualParameterSharingRunnerError(
                "Expected residual savings consume the complete parent."
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise Lfm2ResidualParameterSharingRunnerError(
                "device must be nonempty."
            )
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise Lfm2ResidualParameterSharingRunnerError(
                "dtype must be nonempty."
            )
        fold_runtime._torch_dtype(self.dtype)
        object.__setattr__(
            self,
            "cached_parent_outputs_sha256",
            _digest(
                self.cached_parent_outputs_sha256,
                "cached_parent_outputs_sha256",
            ),
        )

    @property
    def orientations(self) -> tuple[str, ...]:
        return tuple(self.preserve)

    @property
    def expected_pair_savings(self) -> int:
        return lfm2_residual_parameter_sharing_savings(
            self.expected_hidden_size,
            self.expected_intermediate_size,
            self.pair_count,
            self.rank,
        )

    @property
    def expected_child_unique_parameters(self) -> int:
        return (
            self.expected_parent_unique_parameters
            - self.expected_pair_savings
        )

    @property
    def expected_compression_fraction(self) -> float:
        return (
            self.expected_pair_savings
            / self.expected_parent_unique_parameters
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_model": str(self.parent_model),
            "parent_cache": str(self.parent_cache),
            "cached_parent_outputs": str(self.cached_parent_outputs),
            "cached_parent_outputs_sha256": (
                self.cached_parent_outputs_sha256
            ),
            "tokenizer_directory": str(self.tokenizer_directory),
            "work_root": str(self.work_root),
            "checkpoint_destination": str(self.checkpoint_destination),
            "report_path": str(self.report_path),
            "device": self.device,
            "dtype": self.dtype,
            "pair_count": self.pair_count,
            "rank": self.rank,
            "preserve": list(self.orientations),
            "layer_pairs": (
                None
                if self.layer_pairs is None
                else [list(pair) for pair in self.layer_pairs]
            ),
            "max_evidence_rows": self.max_evidence_rows,
            "pair_score_weights": list(self.pair_score_weights),
            "exact_assignment_limit": self.exact_assignment_limit,
            "correlation_chunk_size": self.correlation_chunk_size,
            "variance_floor": self.variance_floor,
            "expected_parent_unique_parameters": (
                self.expected_parent_unique_parameters
            ),
            "expected_layer_count": self.expected_layer_count,
            "expected_hidden_size": self.expected_hidden_size,
            "expected_intermediate_size": (
                self.expected_intermediate_size
            ),
            "expected_whole_output_case_count": (
                self.expected_whole_output_case_count
            ),
            "expected_pair_savings": self.expected_pair_savings,
            "expected_child_unique_parameters": (
                self.expected_child_unique_parameters
            ),
            "expected_compression_fraction": (
                self.expected_compression_fraction
            ),
            "minimum_residual_cosine": self.minimum_residual_cosine,
            "whole_output_gate": {
                "max_new_tokens": self.whole_output_max_new_tokens,
                "maximum_prompt_tokens": (
                    self.whole_output_maximum_prompt_tokens
                ),
                "fragment_minimum_characters": (
                    self.whole_output_fragment_minimum_characters
                ),
                "maximum_new_empty_responses": (
                    self.whole_output_maximum_new_empty_responses
                ),
                "maximum_new_fragment_responses": (
                    self.whole_output_maximum_new_fragment_responses
                ),
                "minimum_parent_nonempty_retention": (
                    self.whole_output_minimum_parent_nonempty_retention
                ),
                "minimum_task_score_retention": (
                    self.whole_output_minimum_task_score_retention
                ),
                "maximum_parent_task_score_regressions": (
                    self.whole_output_maximum_parent_task_score_regressions
                ),
                "maximum_new_generation_limit_saturations": (
                    self.whole_output_maximum_new_generation_limit_saturations
                ),
                "maximum_repeated_ngram_share_regressions": (
                    self.whole_output_maximum_repeated_ngram_share_regressions
                ),
                "maximum_repeated_bigram_share_delta": (
                    self.whole_output_maximum_repeated_bigram_share_delta
                ),
                "exact_parent_text_required": False,
                "exact_token_match_required": False,
            },
            "offline": True,
            "new_parent_forwards_allowed": False,
            "training_allowed": False,
            "standalone_checkpoint_available": False,
        }


# Compatibility with the ordinary sharing runner's public config spelling.
ParameterSharingRunConfig = ResidualParameterSharingRunConfig


class ParentForwardProhibition:
    """Make any source-model execution before installation fail immediately."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.attempted_forward_calls = 0
        self._had_instance_forward = False
        self._instance_forward: Any = None

    def __enter__(self) -> ParentForwardProhibition:
        self._had_instance_forward = "forward" in self.model.__dict__
        self._instance_forward = self.model.__dict__.get("forward")

        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            self.attempted_forward_calls += 1
            raise Lfm2ResidualParameterSharingRunnerError(
                "A source-model forward was attempted after the sealed cache "
                "was selected."
            )

        self.model.forward = blocked
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._had_instance_forward:
            self.model.forward = self._instance_forward
        else:
            delattr(self.model, "forward")


def _model_layers(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise Lfm2ResidualParameterSharingRunnerError(
            "LFM2 model does not expose model.layers."
        )
    return tuple(layers)


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Return exact logical and unique learned-scalar counts."""

    named = tuple(model.named_parameters(remove_duplicate=False))
    logical = sum(int(parameter.numel()) for _, parameter in named)
    unique = {id(parameter): parameter for _, parameter in named}
    return {
        "logical": logical,
        "unique": sum(
            int(parameter.numel()) for parameter in unique.values()
        ),
        "logical_parameter_objects": len(named),
        "unique_parameter_objects": len(unique),
    }


def gradient_audit(model: torch.nn.Module) -> dict[str, Any]:
    with_grad = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    return {
        "unique_parameter_objects": len(tuple(model.parameters())),
        "parameters_with_grad": len(with_grad),
        "parameter_names_with_grad": list(with_grad),
        "zero_gradient": not with_grad,
    }


def topology_signature(model: torch.nn.Module) -> dict[str, Any]:
    """Hash execution depth, operator kinds, and recurrent buffer surfaces."""

    rows = []
    for index, layer in enumerate(_model_layers(model)):
        contract = inspect_lfm2_layer(layer)
        operator = (
            layer.self_attn if layer.is_attention_layer else layer.conv
        )
        buffers = [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in operator.named_buffers(
                remove_duplicate=False
            )
        ]
        rows.append(
            {
                "index": index,
                "contract": contract.as_dict(),
                "layer_class": (
                    f"{type(layer).__module__}.{type(layer).__qualname__}"
                ),
                "operator_class": (
                    f"{type(operator).__module__}."
                    f"{type(operator).__qualname__}"
                ),
                "ffn_class": (
                    f"{type(layer.feed_forward).__module__}."
                    f"{type(layer.feed_forward).__qualname__}"
                ),
                "operator_buffers": buffers,
            }
        )
    body = {
        "layer_count": len(rows),
        "layers": rows,
        "all_layers_retained": True,
        "attention_and_short_conv_retained": True,
    }
    return {**body, "topology_sha256": _json_sha256(body)}


def audit_residual_installation(
    model: torch.nn.Module,
    plan: Lfm2ResidualParameterSharingPlan,
) -> dict[str, Any]:
    """Authenticate each shared base, factor shape, rank, and selected module."""

    if not isinstance(plan, Lfm2ResidualParameterSharingPlan):
        raise Lfm2ResidualParameterSharingRunnerError(
            "A residual parameter-sharing plan is required."
        )
    layers = _model_layers(model)
    expected_modules: set[tuple[int, str]] = set()
    rows: list[dict[str, Any]] = []
    residual_parameters = 0
    for pair in plan.pair_plans:
        for role in _ROLES:
            preserved = getattr(
                layers[pair.preserved_layer].feed_forward,
                role,
            )
            approximated = getattr(
                layers[pair.approximated_layer].feed_forward,
                role,
            )
            role_plan = pair.role_plan(role)
            if not isinstance(approximated, Lfm2SharedBaseResidualLinear):
                raise Lfm2ResidualParameterSharingRunnerError(
                    "An approximated FFN projection is not a residual module."
                )
            expected_modules.add((pair.approximated_layer, role))
            left_shape = tuple(int(value) for value in approximated.residual_left.shape)
            right_shape = tuple(
                int(value) for value in approximated.residual_right.shape
            )
            expected_left = (
                int(preserved.weight.shape[0]),
                plan.rank,
            )
            expected_right = (
                plan.rank,
                int(preserved.weight.shape[1]),
            )
            exact = bool(
                approximated.weight is preserved.weight
                and approximated.residual_rank == plan.rank
                and left_shape == expected_left
                and right_shape == expected_right
                and approximated.bias is None
            )
            if not exact:
                raise Lfm2ResidualParameterSharingRunnerError(
                    "Residual module base identity, rank, or factor shape differs "
                    "from the sealed plan."
                )
            role_parameters = int(
                approximated.residual_left.numel()
                + approximated.residual_right.numel()
            )
            if role_parameters != role_plan.residual_parameters:
                raise Lfm2ResidualParameterSharingRunnerError(
                    "Residual factor Parameter accounting differs from the plan."
                )
            residual_parameters += role_parameters
            rows.append(
                {
                    "pair": list(pair.layers),
                    "role": role,
                    "preserve": pair.preserve,
                    "preserved_layer": pair.preserved_layer,
                    "approximated_layer": pair.approximated_layer,
                    "shared_base_exact_parameter_identity": True,
                    "rank": plan.rank,
                    "residual_left_shape": list(left_shape),
                    "residual_right_shape": list(right_shape),
                    "residual_parameters": role_parameters,
                }
            )
    actual_modules = {
        (layer_index, role)
        for layer_index, layer in enumerate(layers)
        for role in _ROLES
        if isinstance(
            getattr(layer.feed_forward, role),
            Lfm2SharedBaseResidualLinear,
        )
    }
    if actual_modules != expected_modules:
        raise Lfm2ResidualParameterSharingRunnerError(
            "Residual module surface differs from the selected disjoint pairs."
        )
    return {
        "selected_pairs": [list(pair) for pair in plan.selected_pairs],
        "orientations": list(plan.orientations),
        "roles": rows,
        "expected_residual_modules": 3 * len(plan.selected_pairs),
        "actual_residual_modules": len(actual_modules),
        "residual_parameters": residual_parameters,
        "shared_base_aliases": len(actual_modules),
        "exact": True,
    }


def capture_config_from_parent_cache(
    cache: Lfm2CircuitTraceCache,
) -> Lfm2CircuitTraceCaptureConfig:
    """Reconstruct and authenticate the exact sealed capture protocol."""

    capture = cache.manifest["capture"]
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=int(capture["sample_last_tokens"]),
        max_decode_steps=int(capture["max_decode_steps"]),
        max_capture_bytes=int(capture["max_capture_bytes"]),
        logit_rtol=float(capture["logit_rtol"]),
        logit_atol=float(capture["logit_atol"]),
        require_top1_agreement=bool(capture["require_top1_agreement"]),
    )
    if config.as_dict() != capture:
        raise Lfm2ResidualParameterSharingRunnerError(
            "Parent cache capture protocol cannot be reconstructed exactly."
        )
    return config


def _bounded_indices(row_count: int, maximum: int) -> tuple[int, ...]:
    if row_count <= maximum:
        return tuple(range(row_count))
    return tuple(
        (index * row_count) // maximum for index in range(maximum)
    )


def select_post_swiglu_rows(
    trace_rows: Any,
    *,
    maximum_rows: int,
) -> tuple[dict[int, torch.Tensor], tuple[int, ...]]:
    """Select one deterministic, path-aligned row set for every parent layer."""

    maximum = _positive_int(maximum_rows, "maximum_rows")
    layer_count = len(trace_rows.layer_kinds)
    if layer_count <= 0:
        raise Lfm2ResidualParameterSharingRunnerError(
            "Parent trace has no layers."
        )
    row_count = int(
        trace_rows.field(0, "mlp.post_swiglu").shape[0]
    )
    if row_count < 2:
        raise Lfm2ResidualParameterSharingRunnerError(
            "Residual sharing requires at least two cached activation rows."
        )
    indices = _bounded_indices(row_count, min(row_count, maximum))
    index = torch.tensor(indices, dtype=torch.long)
    selected: dict[int, torch.Tensor] = {}
    width = None
    for layer_index in range(layer_count):
        rows = trace_rows.field(layer_index, "mlp.post_swiglu")
        if rows.ndim != 2 or int(rows.shape[0]) != row_count:
            raise Lfm2ResidualParameterSharingRunnerError(
                "Cached post-SwiGLU rows are not path aligned."
            )
        width = int(rows.shape[1]) if width is None else width
        if int(rows.shape[1]) != width:
            raise Lfm2ResidualParameterSharingRunnerError(
                "Cached post-SwiGLU layer widths differ."
            )
        selected[layer_index] = (
            rows.index_select(0, index)
            .detach()
            .to(device="cpu")
            .contiguous()
        )
    return selected, indices


def _directory_keys(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        child.name
        for child in path.iterdir()
        if child.is_dir() and not child.is_symlink()
    }


def _expected_capture_forwards(cache: Lfm2CircuitTraceCache) -> int:
    total = 0
    for batch in cache.iter_batches():
        total += 2 + 2 * int(batch.forced_decode_ids.shape[1])
    return total


def evaluate_cached_trajectory(
    model: torch.nn.Module,
    *,
    parent_cache: Lfm2CircuitTraceCache,
    parent_rows: Any,
    capture_config: Lfm2CircuitTraceCaptureConfig,
    cache_root: Path,
    minimum_residual_cosine: float,
    phase: str,
) -> dict[str, Any]:
    """Evaluate the child against existing parent traces without parent execution."""

    batches = fold_runtime.calibration_batches_from_cache(parent_cache)
    existing = _directory_keys(cache_root)
    child_cache, child_rows = fold_runtime.build_child_trace_cache(
        cache_root,
        model,
        batches,
        capture_config=capture_config,
    )
    gate_report = fold_runtime.evaluate_trajectory_gate(
        parent_cache,
        child_cache,
        model,
        parent_rows,
        child_rows,
        config=fold_runtime.TrajectoryGateConfig(
            min_mean_residual_cosine=minimum_residual_cosine,
        ),
    )
    cache_reused = child_cache.cache_key in existing
    return {
        "phase": phase,
        "decision": {
            "accepted": bool(gate_report["passed"]),
            "reasons": list(gate_report["reasons"]),
        },
        "cache_key": child_cache.cache_key,
        "trace_rows_sha256": child_rows.trace_rows_sha256,
        "cache_reused": cache_reused,
        "child_forward_calls": (
            0 if cache_reused else _expected_capture_forwards(parent_cache)
        ),
        "parent_forward_calls": 0,
        "report": gate_report,
        "report_sha256": _json_sha256(gate_report),
    }


def _tokenizer_fingerprint(path: Path) -> str:
    try:
        from experiments.run_lfm2_dynamic_catalog import (
            fingerprint_local_tokenizer,
        )
    except ImportError:
        from run_lfm2_dynamic_catalog import fingerprint_local_tokenizer

    return fingerprint_local_tokenizer(path)


def _behavior_evaluator(
    config: ResidualParameterSharingRunConfig,
    *,
    cache: Lfm2CircuitTraceCache,
    tokenizer: Any,
    tokenizer_fingerprint: str,
) -> Lfm2WholeOutputBehaviorEvaluator:
    baseline = load_lfm2_cached_whole_output_baseline(
        config.cached_parent_outputs,
        expected_sha256=config.cached_parent_outputs_sha256,
        expected_parent_cache_key=cache.cache_key,
        expected_parent_state_fingerprint=cache.parent_fingerprint,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        expected_generation_limit=config.whole_output_max_new_tokens,
        fragment_minimum_characters=(
            config.whole_output_fragment_minimum_characters
        ),
    )
    if len(baseline.cases) != config.expected_whole_output_case_count:
        raise Lfm2ResidualParameterSharingRunnerError(
            "Sealed parent whole-output envelope has an unexpected case count."
        )
    return Lfm2WholeOutputBehaviorEvaluator(
        baseline=baseline,
        tokenizer=tokenizer,
        maximum_prompt_tokens=config.whole_output_maximum_prompt_tokens,
        fragment_minimum_characters=(
            config.whole_output_fragment_minimum_characters
        ),
        maximum_new_empty_responses=(
            config.whole_output_maximum_new_empty_responses
        ),
        maximum_new_fragment_responses=(
            config.whole_output_maximum_new_fragment_responses
        ),
        minimum_parent_nonempty_retention=(
            config.whole_output_minimum_parent_nonempty_retention
        ),
        minimum_task_score_retention=(
            config.whole_output_minimum_task_score_retention
        ),
        maximum_parent_task_score_regressions=(
            config.whole_output_maximum_parent_task_score_regressions
        ),
        maximum_new_generation_limit_saturations=(
            config.whole_output_maximum_new_generation_limit_saturations
        ),
        maximum_repeated_ngram_share_regressions=(
            config.whole_output_maximum_repeated_ngram_share_regressions
        ),
        maximum_repeated_bigram_share_delta=(
            config.whole_output_maximum_repeated_bigram_share_delta
        ),
    )


def _behavior_event(
    evaluator: Lfm2WholeOutputBehaviorEvaluator,
    model: torch.nn.Module,
    *,
    phase: str,
) -> dict[str, Any]:
    decision = evaluator.evaluate_model(model, phase=phase)
    event = dict(evaluator.events[-1])
    return {
        "phase": phase,
        "decision": decision.as_dict(),
        "event": event,
        "event_sha256": _json_sha256(event),
        "parent_forward_calls": 0,
        "child_generate_invocations": len(evaluator.baseline.cases),
        "exact_parent_text_required": False,
        "exact_token_match_required": False,
    }


def candidate_accepted(
    trajectory: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> bool:
    """Require both quality gates, never exact token or generated-text identity."""

    return bool(
        trajectory["decision"]["accepted"]
        and behavior["decision"]["accepted"]
    )


def _atomic_write_new_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Lfm2ResidualParameterSharingRunnerError(
            f"Refusing to replace existing report {destination}."
        )
    payload = _canonical_json(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sealed_report(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("run_sha256", None)
    return {**body, "run_sha256": _json_sha256(body)}


def _authenticated_rollback(
    installation: InstalledLfm2ResidualParameterSharing,
    model: torch.nn.Module,
    *,
    parent_state_sha256: str,
    parent_counts: Mapping[str, int],
    parent_topology_sha256: str,
) -> dict[str, Any]:
    rolled_back = installation.rollback()
    observed_state = fold_runtime.model_state_sha256(model)
    observed_counts = parameter_counts(model)
    observed_topology = topology_signature(model)
    gradients = gradient_audit(model)
    exact = bool(
        rolled_back
        and observed_state == parent_state_sha256
        and observed_counts == dict(parent_counts)
        and observed_topology["topology_sha256"]
        == parent_topology_sha256
        and gradients["zero_gradient"]
    )
    if not exact:
        raise Lfm2ResidualParameterSharingRunnerError(
            "Candidate did not roll back to the exact source model."
        )
    return {
        "performed": True,
        "exact": True,
        "source_state_sha256_restored": observed_state,
        "source_parameter_counts_restored": observed_counts,
        "source_topology_sha256_restored": (
            observed_topology["topology_sha256"]
        ),
        "gradient_audit": gradients,
    }


def _publish_report(
    config: ResidualParameterSharingRunConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = _sealed_report(report)
    _atomic_write_new_json(config.report_path, sealed)
    return sealed


def _checkpoint_absence_audit(
    config: ResidualParameterSharingRunConfig,
) -> dict[str, Any]:
    exists = config.checkpoint_destination.exists()
    if exists:
        raise Lfm2ResidualParameterSharingRunnerError(
            "A checkpoint appeared even though residual serialization is unavailable."
        )
    return {
        "requested_destination": str(config.checkpoint_destination),
        "exists": False,
        "files_written": 0,
        "python_object_serialization_used": False,
        "dense_fallback_used": False,
        "ordinary_full_sharing_checkpoint_used": False,
    }


def run_experiment(
    config: ResidualParameterSharingRunConfig,
) -> dict[str, Any]:
    """Run one complete in-memory experiment and fail closed at publication."""

    if not isinstance(config, ResidualParameterSharingRunConfig):
        raise TypeError(
            "config must be ResidualParameterSharingRunConfig."
        )
    config.work_root.mkdir(parents=True, exist_ok=True)
    config.checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    installation: InstalledLfm2ResidualParameterSharing | None = None
    model: torch.nn.Module | None = None
    parent_state_sha256: str | None = None
    parent_counts: dict[str, int] | None = None
    parent_topology: dict[str, Any] | None = None
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "initializing",
        "publication_status": "not_evaluated",
        "config": config.as_dict(),
        "training": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "optimizer_constructed": False,
            "gradients_used": False,
        },
        "source_parent_forward_calls": 0,
        "checkpoint_published": False,
        "checkpoint_absence": _checkpoint_absence_audit(config),
        "promotion_policy": {
            "cached_parent_trajectory_required": True,
            "sealed_whole_output_gate_required": True,
            "whole_output_case_count": (
                config.expected_whole_output_case_count
            ),
            "exact_parent_text_required": False,
            "exact_token_match_required": False,
            "standalone_reload_required_for_publication": True,
            "standalone_reload_implemented": False,
        },
    }
    try:
        cache = Lfm2CircuitTraceCache.open(
            config.parent_cache,
            verify=True,
        )
        capture_config = capture_config_from_parent_cache(cache)
        parent_rows = fold_runtime.extract_trace_rows(cache)
        selected_rows, selected_indices = select_post_swiglu_rows(
            parent_rows,
            maximum_rows=config.max_evidence_rows,
        )
        dtype = fold_runtime._torch_dtype(config.dtype)
        model = fold_runtime.load_local_lfm2_model(
            config.parent_model,
            device=config.device,
            dtype=dtype,
        )
        model.eval()
        parent_counts = parameter_counts(model)
        parent_topology = topology_signature(model)
        parent_state_sha256 = fold_runtime.model_state_sha256(model)
        gradients_before = gradient_audit(model)
        if not gradients_before["zero_gradient"]:
            raise Lfm2ResidualParameterSharingRunnerError(
                "The local source model already contains gradients."
            )
        if (
            cache.parent_fingerprint
            != fingerprint_lfm2_parent(model).identity_sha256
        ):
            raise Lfm2ResidualParameterSharingRunnerError(
                "The sealed cache does not belong to the exact local source model."
            )
        if (
            parent_counts["unique"]
            != config.expected_parent_unique_parameters
            or parent_topology["layer_count"]
            != config.expected_layer_count
            or any(
                row["contract"]["hidden_size"]
                != config.expected_hidden_size
                or row["contract"]["ffn_intermediate_size"]
                != config.expected_intermediate_size
                for row in parent_topology["layers"]
            )
        ):
            raise Lfm2ResidualParameterSharingRunnerError(
                "The local source does not match the authenticated LFM2 target "
                "contract."
            )
        tokenizer_fingerprint = _tokenizer_fingerprint(
            config.tokenizer_directory
        )
        tokenizer = fold_runtime.load_local_tokenizer(
            config.tokenizer_directory
        )
        behavior_evaluator = _behavior_evaluator(
            config,
            cache=cache,
            tokenizer=tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        with ParentForwardProhibition(model) as source_guard:
            evidence = seal_lfm2_activation_evidence(
                model,
                selected_rows,
                cache_key=cache.cache_key,
                heldout_corpus_sha256=cache.corpus_fingerprint,
            )
            plan = plan_lfm2_residual_parameter_sharing(
                model,
                evidence,
                pair_count=config.pair_count,
                rank=config.rank,
                preserve=config.orientations,
                layer_pairs=config.layer_pairs,
                pair_score_weights=config.pair_score_weights,
                exact_assignment_limit=config.exact_assignment_limit,
                chunk_size=config.correlation_chunk_size,
                variance_floor=config.variance_floor,
            )
            if (
                plan.projected_parameters_saved
                != config.expected_pair_savings
                or plan.projected_unique_parameters_after
                != config.expected_child_unique_parameters
                or len(plan.selected_pairs) != config.pair_count
                or plan.rank != config.rank
                or plan.orientations != config.orientations
            ):
                raise Lfm2ResidualParameterSharingRunnerError(
                    "Residual plan does not meet exact physical target accounting."
                )
            installation = (
                install_lfm2_residual_parameter_sharing_plan(
                    model,
                    evidence,
                    plan,
                )
            )
        if source_guard.attempted_forward_calls != 0:
            raise Lfm2ResidualParameterSharingRunnerError(
                "Source-model forward prohibition recorded an attempted call."
            )
        candidate_state_sha256 = fold_runtime.model_state_sha256(model)
        candidate_counts = parameter_counts(model)
        candidate_topology = topology_signature(model)
        candidate_residuals = audit_residual_installation(model, plan)
        gradients_candidate = gradient_audit(model)
        observed_savings = (
            parent_counts["unique"] - candidate_counts["unique"]
        )
        if (
            candidate_counts["unique"]
            != config.expected_child_unique_parameters
            or observed_savings != config.expected_pair_savings
            or candidate_counts["logical"]
            != plan.projected_logical_parameters_after
            or candidate_topology["topology_sha256"]
            != parent_topology["topology_sha256"]
            or not candidate_residuals["exact"]
            or not gradients_candidate["zero_gradient"]
        ):
            raise Lfm2ResidualParameterSharingRunnerError(
                "Installed child failed count, topology, residual, or gradient "
                "authentication."
            )
        in_memory_trajectory = evaluate_cached_trajectory(
            model,
            parent_cache=cache,
            parent_rows=parent_rows,
            capture_config=capture_config,
            cache_root=config.work_root / "candidate-trajectory",
            minimum_residual_cosine=config.minimum_residual_cosine,
            phase="residual_sharing_in_memory_child",
        )
        in_memory_behavior = _behavior_event(
            behavior_evaluator,
            model,
            phase="residual_sharing_in_memory_child",
        )
        gates_accepted = candidate_accepted(
            in_memory_trajectory,
            in_memory_behavior,
        )
        report.update(
            {
                "status": "candidate_evaluated",
                "local_io": fold_runtime.direct_local_io_audit(
                    config.parent_model
                ),
                "cache": {
                    "path": str(cache.path),
                    "cache_key": cache.cache_key,
                    "parent_fingerprint": cache.parent_fingerprint,
                    "corpus_fingerprint": cache.corpus_fingerprint,
                    "implementation_fingerprint": (
                        cache.implementation_fingerprint
                    ),
                    "layer_topology_fingerprint": (
                        cache.layer_topology_fingerprint
                    ),
                    "verified": True,
                    "new_parent_forward_calls": 0,
                },
                "activation_evidence": {
                    **evidence.as_dict(),
                    "selected_path_indices": list(selected_indices),
                    "field": "mlp.post_swiglu",
                    "source": "existing_sealed_parent_circuit_cache",
                    "new_parent_forward_calls": 0,
                },
                "source": {
                    "state_sha256": parent_state_sha256,
                    "parameter_counts": parent_counts,
                    "topology": parent_topology,
                    "gradient_audit": gradients_before,
                },
                "plan": plan.as_dict(),
                "installation": installation.report.as_dict(),
                "candidate": {
                    "state_sha256": candidate_state_sha256,
                    "parameter_counts": candidate_counts,
                    "physical_parameters_saved": observed_savings,
                    "compression_fraction": (
                        observed_savings / parent_counts["unique"]
                    ),
                    "topology": candidate_topology,
                    "residual_topology": candidate_residuals,
                    "gradient_audit": gradients_candidate,
                    "all_decoder_layers_retained": True,
                    "attention_and_recurrent_operators_retained": True,
                },
                "in_memory_evaluation": {
                    "trajectory": in_memory_trajectory,
                    "whole_output": in_memory_behavior,
                    "accepted": gates_accepted,
                    "exact_generated_text_is_diagnostic_only": True,
                    "exact_token_match_is_diagnostic_only": True,
                },
                "tokenizer_fingerprint": tokenizer_fingerprint,
            }
        )
        report["rollback"] = _authenticated_rollback(
            installation,
            model,
            parent_state_sha256=parent_state_sha256,
            parent_counts=parent_counts,
            parent_topology_sha256=parent_topology["topology_sha256"],
        )
        report["checkpoint_absence"] = _checkpoint_absence_audit(config)
        if not gates_accepted:
            report.update(
                {
                    "status": "rejected",
                    "publication_status": "rejected_by_in_memory_gates",
                    "rejection_stage": "in_memory_gates",
                    "checkpoint_published": False,
                    "diagnostics_only": True,
                    "candidate_gate_accepted": False,
                }
            )
            return _publish_report(config, report)
        report.update(
            {
                "status": PUBLICATION_BLOCK_STATUS,
                "publication_status": PUBLICATION_BLOCK_STATUS,
                "checkpoint_published": False,
                "diagnostics_only": True,
                "candidate_gate_accepted": True,
                "publication_blocker": {
                    "code": PUBLICATION_BLOCK_STATUS,
                    "reason": (
                        "The residual-base-plus-low-rank topology has no "
                        "authenticated standalone checkpoint format yet."
                    ),
                    "required_next_component": (
                        "alias-aware residual-module standalone serializer "
                        "and fresh-reload authenticator"
                    ),
                    "dense_fallback_permitted": False,
                    "ordinary_full_sharing_checkpoint_permitted": False,
                    "python_object_serialization_permitted": False,
                },
            }
        )
        return _publish_report(config, report)
    except Exception as exc:
        cleanup_errors: list[str] = []
        rollback = None
        if (
            installation is not None
            and installation.active
            and model is not None
            and parent_state_sha256 is not None
            and parent_counts is not None
            and parent_topology is not None
        ):
            try:
                rollback = _authenticated_rollback(
                    installation,
                    model,
                    parent_state_sha256=parent_state_sha256,
                    parent_counts=parent_counts,
                    parent_topology_sha256=(
                        parent_topology["topology_sha256"]
                    ),
                )
            except Exception as rollback_exc:  # pragma: no cover - catastrophic
                cleanup_errors.append(
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
        try:
            checkpoint_absence = _checkpoint_absence_audit(config)
        except Exception as checkpoint_exc:  # pragma: no cover - catastrophic
            checkpoint_absence = {
                "audit_error": (
                    f"{type(checkpoint_exc).__name__}: {checkpoint_exc}"
                )
            }
            cleanup_errors.append(checkpoint_absence["audit_error"])
        report.update(
            {
                "status": "error",
                "publication_status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "rollback": rollback,
                "cleanup_errors": cleanup_errors,
                "checkpoint_published": False,
                "checkpoint_absence": checkpoint_absence,
                "diagnostics_only": True,
            }
        )
        return _publish_report(config, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Activation-align LFM2 FFN pairs and install shared-base plus "
            "low-rank residuals without training or new parent forwards."
        )
    )
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--cached-parent-outputs", type=Path, required=True)
    parser.add_argument("--cached-parent-outputs-sha256", required=True)
    parser.add_argument("--tokenizer-directory", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-destination",
        type=Path,
        required=True,
        help=(
            "Reserved future destination. It must not exist and will not be "
            "created until authenticated residual serialization exists."
        ),
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--preserve",
        type=parse_preserve,
        default=DEFAULT_PRESERVE,
        help=(
            "earlier/later for every pair, or one comma-separated orientation "
            "per pair"
        ),
    )
    parser.add_argument("--layer-pairs", type=parse_layer_pairs)
    parser.add_argument(
        "--max-evidence-rows",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_ROWS,
    )
    parser.add_argument(
        "--pair-score-weights",
        type=parse_weight_pair,
        default=(0.5, 0.5),
    )
    parser.add_argument("--exact-assignment-limit", type=int, default=256)
    parser.add_argument("--correlation-chunk-size", type=int, default=256)
    parser.add_argument("--variance-floor", type=float, default=0.0)
    parser.add_argument(
        "--expected-parent-unique-parameters",
        type=int,
        default=EXPECTED_PARENT_UNIQUE_PARAMETERS,
    )
    parser.add_argument(
        "--expected-layer-count",
        type=int,
        default=EXPECTED_LAYER_COUNT,
    )
    parser.add_argument(
        "--expected-hidden-size",
        type=int,
        default=EXPECTED_HIDDEN_SIZE,
    )
    parser.add_argument(
        "--expected-intermediate-size",
        type=int,
        default=EXPECTED_INTERMEDIATE_SIZE,
    )
    parser.add_argument(
        "--expected-whole-output-case-count",
        type=int,
        default=EXPECTED_WHOLE_OUTPUT_CASE_COUNT,
    )
    parser.add_argument(
        "--minimum-residual-cosine",
        type=float,
        default=DEFAULT_MINIMUM_RESIDUAL_COSINE,
    )
    parser.add_argument("--whole-output-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--whole-output-maximum-prompt-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--whole-output-fragment-minimum-characters",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--whole-output-maximum-new-empty-responses",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--whole-output-maximum-new-fragment-responses",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--whole-output-minimum-parent-nonempty-retention",
        type=float,
        default=DEFAULT_MINIMUM_RETENTION,
    )
    parser.add_argument(
        "--whole-output-minimum-task-score-retention",
        type=float,
        default=DEFAULT_MINIMUM_RETENTION,
    )
    parser.add_argument(
        "--whole-output-maximum-parent-task-score-regressions",
        type=int,
        default=DEFAULT_MAXIMUM_CASE_REGRESSIONS,
    )
    parser.add_argument(
        "--whole-output-maximum-new-generation-limit-saturations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--whole-output-maximum-repeated-ngram-share-regressions",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--whole-output-maximum-repeated-bigram-share-delta",
        type=float,
        default=0.10,
    )
    return parser


def _config_from_namespace(
    arguments: argparse.Namespace,
) -> ResidualParameterSharingRunConfig:
    return ResidualParameterSharingRunConfig(**vars(arguments))


def main(argv: Sequence[str] | None = None) -> int:
    config = _config_from_namespace(build_parser().parse_args(argv))
    report = run_experiment(config)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    if report["status"] == "rejected":
        return 2
    if report["status"] == PUBLICATION_BLOCK_STATUS:
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
