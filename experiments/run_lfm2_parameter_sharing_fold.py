#!/usr/bin/env python3
"""Zero-training, activation-aligned physical FFN compaction for local LFM2.

The source model is loaded only from explicit local JSON and safetensors.  Its
already-sealed circuit cache supplies every post-SwiGLU activation used for
selection and alignment; the source model is prohibited from executing before
the sharing transaction is installed.  Candidate and fresh-reload behavior are
then judged against cached parent trajectories and the sealed 28-case parent
whole-output envelope.

Accepted children retain every decoder layer and every attention/short-conv
operator while selected disjoint FFN pairs share three full-width Parameter
objects (w1, w3, and w2).  Rejected candidates are rolled back to the exact
source Parameter objects and publish diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

try:
    from experiments import run_lfm2_residual_attention_fold as residual
except ImportError:  # Direct execution from the experiments directory.
    import run_lfm2_residual_attention_fold as residual

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
    InstalledLfm2ParameterSharing,
    install_lfm2_parameter_sharing_plan,
    lfm2_ffn_parameter_sharing_savings,
    plan_lfm2_activation_aligned_parameter_sharing,
    seal_lfm2_activation_evidence,
)
from neural_forge.hf_lfm2_shared_checkpoint import (
    export_lfm2_shared_checkpoint,
    load_lfm2_shared_checkpoint,
    plan_lfm2_shared_checkpoint,
)


RUN_SCHEMA = "neural-forge-lfm2-activation-aligned-ffn-sharing-run-v1"
EXPECTED_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_LAYER_COUNT = 14
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_INTERMEDIATE_SIZE = 2560
DEFAULT_PAIR_COUNT = 3
DEFAULT_MAX_EVIDENCE_ROWS = 256
DEFAULT_MINIMUM_RETENTION = 0.90
DEFAULT_MAXIMUM_CASE_REGRESSIONS = 2
DEFAULT_MINIMUM_RESIDUAL_COSINE = 0.80
_ROLES = ("w1", "w3", "w2")


class Lfm2ParameterSharingRunnerError(RuntimeError):
    """The sharing run could not authenticate a required invariant."""


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
        raise Lfm2ParameterSharingRunnerError(
            "Run evidence is not finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ParameterSharingRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2ParameterSharingRunnerError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2ParameterSharingRunnerError(
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
        raise Lfm2ParameterSharingRunnerError(f"{name} must be a finite float.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2ParameterSharingRunnerError(
            f"{name} must be a finite float."
        ) from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise Lfm2ParameterSharingRunnerError(f"{name} is outside its valid range.")
    return result


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2ParameterSharingRunnerError(
            f"{name} must be an explicit local directory, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2ParameterSharingRunnerError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2ParameterSharingRunnerError(
            f"{name} must be an explicit local file, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2ParameterSharingRunnerError(
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
    """Parse two finite, non-negative weights with a positive total."""

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
    """Parse ``anchor:source`` pairs while rejecting overlap and ambiguity."""

    if not value.strip():
        raise argparse.ArgumentTypeError("layer pairs cannot be empty")
    result: list[tuple[int, int]] = []
    used: set[int] = set()
    for raw_pair in value.split(","):
        pieces = raw_pair.strip().split(":")
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(
                "layer pairs must use anchor:source comma-separated syntax"
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


@dataclass(frozen=True, slots=True)
class ParameterSharingRunConfig:
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
    layer_pairs: tuple[tuple[int, int], ...] | None = None
    max_evidence_rows: int = DEFAULT_MAX_EVIDENCE_ROWS
    pair_score_weights: tuple[float, float] = (0.5, 0.5)
    merge_weights: tuple[float, float] = (0.5, 0.5)
    exact_assignment_limit: int = 256
    correlation_chunk_size: int = 256
    variance_floor: float = 0.0
    expected_parent_unique_parameters: int = EXPECTED_PARENT_UNIQUE_PARAMETERS
    expected_layer_count: int = EXPECTED_LAYER_COUNT
    expected_hidden_size: int = EXPECTED_HIDDEN_SIZE
    expected_intermediate_size: int = EXPECTED_INTERMEDIATE_SIZE
    expected_whole_output_case_count: int = 28
    minimum_residual_cosine: float = DEFAULT_MINIMUM_RESIDUAL_COSINE
    whole_output_max_new_tokens: int = 64
    whole_output_maximum_prompt_tokens: int = 512
    whole_output_fragment_minimum_characters: int = 12
    whole_output_maximum_new_empty_responses: int = 2
    whole_output_maximum_new_fragment_responses: int = 2
    whole_output_minimum_parent_nonempty_retention: float = (
        DEFAULT_MINIMUM_RETENTION
    )
    whole_output_minimum_task_score_retention: float = DEFAULT_MINIMUM_RETENTION
    whole_output_maximum_parent_task_score_regressions: int = (
        DEFAULT_MAXIMUM_CASE_REGRESSIONS
    )
    whole_output_maximum_new_generation_limit_saturations: int = 2
    whole_output_maximum_repeated_ngram_share_regressions: int = 2
    whole_output_maximum_repeated_bigram_share_delta: float = 0.10
    tensor_format: str = "safetensors"

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
                    raise Lfm2ParameterSharingRunnerError(
                        "Work, checkpoint, and report paths must not be inside "
                        "source model/cache directories."
                    )
        if self.checkpoint_destination == self.report_path:
            raise Lfm2ParameterSharingRunnerError(
                "checkpoint_destination and report_path must differ."
            )
        if self.checkpoint_destination.exists():
            raise Lfm2ParameterSharingRunnerError(
                "checkpoint_destination already exists; refusing to replace it."
            )
        if self.report_path.exists():
            raise Lfm2ParameterSharingRunnerError(
                "report_path already exists; refusing to replace it."
            )
        for name in (
            "pair_count",
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
            raise Lfm2ParameterSharingRunnerError(
                "pair_count cannot exceed half the expected layer count."
            )
        if self.layer_pairs is not None:
            if len(self.layer_pairs) != self.pair_count:
                raise Lfm2ParameterSharingRunnerError(
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
                    raise Lfm2ParameterSharingRunnerError(
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
                    raise Lfm2ParameterSharingRunnerError(
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
        for name in ("pair_score_weights", "merge_weights"):
            values = tuple(float(value) for value in getattr(self, name))
            if (
                len(values) != 2
                or any(not math.isfinite(value) or value < 0.0 for value in values)
                or sum(values) <= 0.0
            ):
                raise Lfm2ParameterSharingRunnerError(
                    f"{name} must contain two finite non-negative weights."
                )
            object.__setattr__(self, name, values)
        if self.tensor_format not in {"safetensors", "raw"}:
            raise Lfm2ParameterSharingRunnerError(
                "tensor_format must be safetensors or raw."
            )
        if self.exact_assignment_limit > 512:
            raise Lfm2ParameterSharingRunnerError(
                "exact_assignment_limit cannot exceed 512."
            )
        if self.expected_child_unique_parameters <= 0:
            raise Lfm2ParameterSharingRunnerError(
                "Expected FFN savings consume the complete parent."
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise Lfm2ParameterSharingRunnerError("device must be nonempty.")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise Lfm2ParameterSharingRunnerError("dtype must be nonempty.")
        residual._torch_dtype(self.dtype)
        object.__setattr__(
            self,
            "cached_parent_outputs_sha256",
            _digest(
                self.cached_parent_outputs_sha256,
                "cached_parent_outputs_sha256",
            ),
        )

    @property
    def expected_pair_savings(self) -> int:
        return lfm2_ffn_parameter_sharing_savings(
            self.expected_hidden_size,
            self.expected_intermediate_size,
            self.pair_count,
        )

    @property
    def expected_child_unique_parameters(self) -> int:
        return self.expected_parent_unique_parameters - self.expected_pair_savings

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_model": str(self.parent_model),
            "parent_cache": str(self.parent_cache),
            "cached_parent_outputs": str(self.cached_parent_outputs),
            "cached_parent_outputs_sha256": self.cached_parent_outputs_sha256,
            "tokenizer_directory": str(self.tokenizer_directory),
            "work_root": str(self.work_root),
            "checkpoint_destination": str(self.checkpoint_destination),
            "report_path": str(self.report_path),
            "device": self.device,
            "dtype": self.dtype,
            "pair_count": self.pair_count,
            "layer_pairs": (
                None
                if self.layer_pairs is None
                else [list(pair) for pair in self.layer_pairs]
            ),
            "max_evidence_rows": self.max_evidence_rows,
            "pair_score_weights": list(self.pair_score_weights),
            "merge_weights": list(self.merge_weights),
            "exact_assignment_limit": self.exact_assignment_limit,
            "correlation_chunk_size": self.correlation_chunk_size,
            "variance_floor": self.variance_floor,
            "expected_parent_unique_parameters": (
                self.expected_parent_unique_parameters
            ),
            "expected_layer_count": self.expected_layer_count,
            "expected_hidden_size": self.expected_hidden_size,
            "expected_intermediate_size": self.expected_intermediate_size,
            "expected_whole_output_case_count": (
                self.expected_whole_output_case_count
            ),
            "expected_pair_savings": self.expected_pair_savings,
            "expected_child_unique_parameters": (
                self.expected_child_unique_parameters
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
                "exact_text_required": False,
            },
            "tensor_format": self.tensor_format,
            "offline": True,
            "training_allowed": False,
        }


class ParentForwardProhibition:
    """Make any source-model execution before installation fail immediately."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.attempted_forward_calls = 0
        self._had_instance_forward = False
        self._instance_forward: Any = None

    def __enter__(self) -> "ParentForwardProhibition":
        self._had_instance_forward = "forward" in self.model.__dict__
        self._instance_forward = self.model.__dict__.get("forward")

        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            self.attempted_forward_calls += 1
            raise Lfm2ParameterSharingRunnerError(
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
        raise Lfm2ParameterSharingRunnerError(
            "LFM2 model does not expose model.layers."
        )
    return tuple(layers)


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Return exact logical and unique learned-scalar counts."""

    named = tuple(model.named_parameters(remove_duplicate=False))
    logical = sum(int(parameter.numel()) for _, parameter in named)
    unique = {
        id(parameter): parameter
        for _, parameter in named
    }
    return {
        "logical": logical,
        "unique": sum(int(parameter.numel()) for parameter in unique.values()),
        "logical_parameter_objects": len(named),
        "unique_parameter_objects": len(unique),
    }


def gradient_audit(model: torch.nn.Module) -> dict[str, Any]:
    parameters = tuple(model.parameters())
    with_grad = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    return {
        "unique_parameter_objects": len(parameters),
        "parameters_with_grad": len(with_grad),
        "parameter_names_with_grad": list(with_grad),
        "zero_gradient": not with_grad,
    }


def topology_signature(model: torch.nn.Module) -> dict[str, Any]:
    """Hash execution depth, operator kinds, and recurrent buffer surfaces."""

    rows = []
    for index, layer in enumerate(_model_layers(model)):
        contract = inspect_lfm2_layer(layer)
        operator = layer.self_attn if layer.is_attention_layer else layer.conv
        buffers = [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in operator.named_buffers(remove_duplicate=False)
        ]
        rows.append(
            {
                "index": index,
                "contract": contract.as_dict(),
                "layer_class": (
                    f"{type(layer).__module__}.{type(layer).__qualname__}"
                ),
                "operator_class": (
                    f"{type(operator).__module__}.{type(operator).__qualname__}"
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


def audit_shared_aliases(
    model: torch.nn.Module,
    pairs: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Authenticate exact pair-local w1/w3/w2 Parameter identity."""

    layers = _model_layers(model)
    expected = {
        frozenset({(int(pair[0]), role), (int(pair[1]), role)})
        for pair in pairs
        for role in _ROLES
    }
    occurrences: dict[int, set[tuple[int, str]]] = {}
    for layer_index, layer in enumerate(layers):
        for role in _ROLES:
            parameter = getattr(layer.feed_forward, role).weight
            occurrences.setdefault(id(parameter), set()).add((layer_index, role))
    actual = {
        frozenset(bindings)
        for bindings in occurrences.values()
        if len(bindings) > 1
    }
    if actual != expected:
        raise Lfm2ParameterSharingRunnerError(
            "Live FFN Parameter aliases differ from the selected disjoint pairs."
        )
    identity_checks = []
    for left, right in pairs:
        for role in _ROLES:
            anchor = getattr(layers[int(left)].feed_forward, role).weight
            source = getattr(layers[int(right)].feed_forward, role).weight
            identity_checks.append(
                {
                    "pair": [int(left), int(right)],
                    "role": role,
                    "exact_parameter_identity": anchor is source,
                    "numel": int(anchor.numel()),
                }
            )
    if not all(row["exact_parameter_identity"] for row in identity_checks):
        raise Lfm2ParameterSharingRunnerError(
            "At least one selected FFN role is not the exact same Parameter."
        )
    return {
        "shared_groups": [list(map(int, pair)) for pair in pairs],
        "identity_checks": identity_checks,
        "actual_shared_parameter_groups": len(actual),
        "expected_shared_parameter_groups": len(expected),
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
        raise Lfm2ParameterSharingRunnerError(
            "Parent cache capture protocol cannot be reconstructed exactly."
        )
    return config


def _bounded_indices(row_count: int, maximum: int) -> tuple[int, ...]:
    if row_count <= maximum:
        return tuple(range(row_count))
    return tuple((index * row_count) // maximum for index in range(maximum))


def select_post_swiglu_rows(
    trace_rows: Any,
    *,
    maximum_rows: int,
) -> tuple[dict[int, torch.Tensor], tuple[int, ...]]:
    """Select one deterministic, path-aligned row set for every parent layer."""

    maximum = _positive_int(maximum_rows, "maximum_rows")
    layer_count = len(trace_rows.layer_kinds)
    if layer_count <= 0:
        raise Lfm2ParameterSharingRunnerError("Parent trace has no layers.")
    row_count = int(trace_rows.field(0, "mlp.post_swiglu").shape[0])
    if row_count < 2:
        raise Lfm2ParameterSharingRunnerError(
            "Parameter sharing requires at least two cached activation rows."
        )
    indices = _bounded_indices(row_count, min(row_count, maximum))
    index = torch.tensor(indices, dtype=torch.long)
    selected: dict[int, torch.Tensor] = {}
    width = None
    for layer_index in range(layer_count):
        rows = trace_rows.field(layer_index, "mlp.post_swiglu")
        if rows.ndim != 2 or int(rows.shape[0]) != row_count:
            raise Lfm2ParameterSharingRunnerError(
                "Cached post-SwiGLU rows are not path aligned."
            )
        width = int(rows.shape[1]) if width is None else width
        if int(rows.shape[1]) != width:
            raise Lfm2ParameterSharingRunnerError(
                "Cached post-SwiGLU layer widths differ."
            )
        selected[layer_index] = (
            rows.index_select(0, index).detach().to(device="cpu").contiguous()
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
    """Evaluate with existing cached-parent helpers and no parent execution."""

    batches = residual.calibration_batches_from_cache(parent_cache)
    existing = _directory_keys(cache_root)
    child_cache, child_rows = residual.build_child_trace_cache(
        cache_root,
        model,
        batches,
        capture_config=capture_config,
    )
    report = residual.evaluate_trajectory_gate(
        parent_cache,
        child_cache,
        model,
        parent_rows,
        child_rows,
        config=residual.TrajectoryGateConfig(
            min_mean_residual_cosine=minimum_residual_cosine,
        ),
    )
    cache_reused = child_cache.cache_key in existing
    return {
        "phase": phase,
        "decision": {
            "accepted": bool(report["passed"]),
            "reasons": list(report["reasons"]),
        },
        "cache_key": child_cache.cache_key,
        "trace_rows_sha256": child_rows.trace_rows_sha256,
        "cache_reused": cache_reused,
        "child_forward_calls": (
            0 if cache_reused else _expected_capture_forwards(parent_cache)
        ),
        "parent_forward_calls": 0,
        "report": report,
        "report_sha256": _json_sha256(report),
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
    config: ParameterSharingRunConfig,
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
        raise Lfm2ParameterSharingRunnerError(
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
    }


def candidate_accepted(
    trajectory: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> bool:
    return bool(
        trajectory["decision"]["accepted"]
        and behavior["decision"]["accepted"]
    )


def behavior_equivalence(
    in_memory: Mapping[str, Any],
    reloaded: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare reload behavior without treating GPU text identity as a gate."""

    left_event = in_memory["event"]
    right_event = reloaded["event"]
    left = {
        "decision": left_event["decision"],
        "per_case_task_regression_ids": (
            left_event["per_case_task_regression_ids"]
        ),
        "outputs": left_event["outputs"],
    }
    right = {
        "decision": right_event["decision"],
        "per_case_task_regression_ids": (
            right_event["per_case_task_regression_ids"]
        ),
        "outputs": right_event["outputs"],
    }
    gate_binding_equal = all(
        left_event.get(name) == right_event.get(name)
        for name in (
            "baseline_fingerprint",
            "behavior_gate_manifest_fingerprint",
        )
    )
    gate_decision_equivalent = bool(
        left_event["decision"]["accepted"]
        and right_event["decision"]["accepted"]
        and gate_binding_equal
    )
    return {
        "in_memory_sha256": _json_sha256(left),
        "reloaded_sha256": _json_sha256(right),
        "exact_output_and_metrics_diagnostic": left == right,
        "gate_binding_equal": gate_binding_equal,
        "gate_decision_equivalent": gate_decision_equivalent,
        "exact_output_required": False,
    }


def trajectory_equivalence(
    in_memory: Mapping[str, Any],
    reloaded: Mapping[str, Any],
) -> dict[str, Any]:
    left = {
        "decision": in_memory["decision"],
        "report": in_memory["report"],
    }
    right = {
        "decision": reloaded["decision"],
        "report": reloaded["report"],
    }
    gate_decision_equivalent = bool(
        in_memory["decision"]["accepted"]
        and reloaded["decision"]["accepted"]
    )
    return {
        "in_memory_sha256": _json_sha256(left),
        "reloaded_sha256": _json_sha256(right),
        "exact_metrics_diagnostic": left == right,
        "cache_key_equal": in_memory["cache_key"] == reloaded["cache_key"],
        "gate_decision_equivalent": gate_decision_equivalent,
        "exact_metrics_required": False,
    }


def _checkpoint_files(path: Path) -> dict[str, Any]:
    rows = []
    for child in sorted(path.iterdir(), key=lambda value: value.name):
        if not child.is_file() or child.is_symlink():
            raise Lfm2ParameterSharingRunnerError(
                "Checkpoint staging contains a non-file or symlink."
            )
        rows.append(
            {
                "name": child.name,
                "bytes": child.stat().st_size,
                "sha256": _file_sha256(child),
            }
        )
    body = {"files": rows, "file_count": len(rows)}
    return {**body, "files_sha256": _json_sha256(body)}


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Lfm2ParameterSharingRunnerError(
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


def _remove_staging(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    if (
        not path.is_dir()
        or path.is_symlink()
        or not path.name.startswith(".lfm2-shared-candidate-")
    ):
        raise Lfm2ParameterSharingRunnerError(
            "Refusing to remove an unauthenticated staging path."
        )
    shutil.rmtree(path)


def _remove_published_checkpoint(path: Path, *, created: bool) -> None:
    if not created or not path.exists():
        return
    if (
        not path.is_dir()
        or path.is_symlink()
        or not (
            (path / "neural_forge_lfm2_shared_manifest.json").is_file()
        )
    ):
        raise Lfm2ParameterSharingRunnerError(
            "Refusing to remove an unauthenticated published checkpoint path."
        )
    shutil.rmtree(path)


def _authenticated_rollback(
    installation: InstalledLfm2ParameterSharing,
    model: torch.nn.Module,
    *,
    parent_state_sha256: str,
    parent_counts: Mapping[str, int],
    parent_topology_sha256: str,
) -> dict[str, Any]:
    rolled_back = installation.rollback()
    observed_state = residual.model_state_sha256(model)
    observed_counts = parameter_counts(model)
    observed_topology = topology_signature(model)
    gradients = gradient_audit(model)
    exact = bool(
        rolled_back
        and observed_state == parent_state_sha256
        and observed_counts == dict(parent_counts)
        and observed_topology["topology_sha256"] == parent_topology_sha256
        and gradients["zero_gradient"]
    )
    if not exact:
        raise Lfm2ParameterSharingRunnerError(
            "Rejected candidate did not roll back to the exact source model."
        )
    return {
        "performed": True,
        "exact": True,
        "source_state_sha256_restored": observed_state,
        "source_parameter_counts_restored": observed_counts,
        "source_topology_sha256_restored": observed_topology["topology_sha256"],
        "gradient_audit": gradients,
    }


def _publish_report(
    config: ParameterSharingRunConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = _sealed_report(report)
    _atomic_write_new_json(config.report_path, sealed)
    return sealed


def run_experiment(config: ParameterSharingRunConfig) -> dict[str, Any]:
    """Run one complete fail-closed sharing experiment and publish its receipt."""

    if not isinstance(config, ParameterSharingRunConfig):
        raise TypeError("config must be ParameterSharingRunConfig.")
    config.work_root.mkdir(parents=True, exist_ok=True)
    config.checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    installation: InstalledLfm2ParameterSharing | None = None
    staging: Path | None = None
    checkpoint_created = False
    model: torch.nn.Module | None = None
    parent_state_sha256: str | None = None
    parent_counts: dict[str, int] | None = None
    parent_topology: dict[str, Any] | None = None
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "initializing",
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
    }
    try:
        cache = Lfm2CircuitTraceCache.open(config.parent_cache, verify=True)
        capture_config = capture_config_from_parent_cache(cache)
        parent_rows = residual.extract_trace_rows(cache)
        selected_rows, selected_indices = select_post_swiglu_rows(
            parent_rows,
            maximum_rows=config.max_evidence_rows,
        )
        dtype = residual._torch_dtype(config.dtype)
        model = residual.load_local_lfm2_model(
            config.parent_model,
            device=config.device,
            dtype=dtype,
        )
        model.eval()
        parent_counts = parameter_counts(model)
        parent_topology = topology_signature(model)
        parent_state_sha256 = residual.model_state_sha256(model)
        gradients_before = gradient_audit(model)
        if not gradients_before["zero_gradient"]:
            raise Lfm2ParameterSharingRunnerError(
                "The local source model already contains gradients."
            )
        if (
            cache.parent_fingerprint
            != fingerprint_lfm2_parent(model).identity_sha256
        ):
            raise Lfm2ParameterSharingRunnerError(
                "The sealed cache does not belong to the exact local source model."
            )
        if (
            parent_counts["unique"] != config.expected_parent_unique_parameters
            or parent_topology["layer_count"] != config.expected_layer_count
            or any(
                row["contract"]["hidden_size"] != config.expected_hidden_size
                or row["contract"]["ffn_intermediate_size"]
                != config.expected_intermediate_size
                for row in parent_topology["layers"]
            )
        ):
            raise Lfm2ParameterSharingRunnerError(
                "The local source does not match the authenticated LFM2 target "
                "contract."
            )
        tokenizer_fingerprint = _tokenizer_fingerprint(
            config.tokenizer_directory
        )
        tokenizer = residual.load_local_tokenizer(config.tokenizer_directory)
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
            plan = plan_lfm2_activation_aligned_parameter_sharing(
                model,
                evidence,
                pair_count=config.pair_count,
                layer_pairs=config.layer_pairs,
                pair_score_weights=config.pair_score_weights,
                merge_weights=config.merge_weights,
                exact_assignment_limit=config.exact_assignment_limit,
                chunk_size=config.correlation_chunk_size,
                variance_floor=config.variance_floor,
            )
            if (
                plan.projected_parameters_saved != config.expected_pair_savings
                or plan.projected_unique_parameters_after
                != config.expected_child_unique_parameters
                or plan.layer_count != config.expected_layer_count
            ):
                raise Lfm2ParameterSharingRunnerError(
                    "Sharing plan does not meet exact physical-count accounting."
                )
            installation = install_lfm2_parameter_sharing_plan(
                model,
                evidence,
                plan,
            )
        if source_guard.attempted_forward_calls != 0:
            raise Lfm2ParameterSharingRunnerError(
                "Source-model forward prohibition recorded an attempted call."
            )
        candidate_state_sha256 = residual.model_state_sha256(model)
        candidate_counts = parameter_counts(model)
        candidate_topology = topology_signature(model)
        candidate_aliases = audit_shared_aliases(model, plan.selected_pairs)
        gradients_candidate = gradient_audit(model)
        if (
            candidate_counts["unique"] != config.expected_child_unique_parameters
            or candidate_counts["logical"] != parent_counts["logical"]
            or candidate_topology["topology_sha256"]
            != parent_topology["topology_sha256"]
            or not gradients_candidate["zero_gradient"]
        ):
            raise Lfm2ParameterSharingRunnerError(
                "Installed child failed count, topology, or gradient authentication."
            )
        in_memory_trajectory = evaluate_cached_trajectory(
            model,
            parent_cache=cache,
            parent_rows=parent_rows,
            capture_config=capture_config,
            cache_root=config.work_root / "candidate-trajectory",
            minimum_residual_cosine=config.minimum_residual_cosine,
            phase="accepted_in_memory_child",
        )
        in_memory_behavior = _behavior_event(
            behavior_evaluator,
            model,
            phase="accepted_in_memory_child",
        )
        report.update(
            {
                "status": "candidate_evaluated",
                "local_io": residual.direct_local_io_audit(config.parent_model),
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
                    "physical_parameters_saved": (
                        parent_counts["unique"] - candidate_counts["unique"]
                    ),
                    "compression_fraction": (
                        (parent_counts["unique"] - candidate_counts["unique"])
                        / parent_counts["unique"]
                    ),
                    "topology": candidate_topology,
                    "aliases": candidate_aliases,
                    "gradient_audit": gradients_candidate,
                    "all_decoder_layers_retained": True,
                    "attention_and_recurrent_operators_retained": True,
                },
                "in_memory_evaluation": {
                    "trajectory": in_memory_trajectory,
                    "whole_output": in_memory_behavior,
                    "accepted": candidate_accepted(
                        in_memory_trajectory,
                        in_memory_behavior,
                    ),
                },
                "tokenizer_fingerprint": tokenizer_fingerprint,
            }
        )
        if not candidate_accepted(in_memory_trajectory, in_memory_behavior):
            report["rollback"] = _authenticated_rollback(
                installation,
                model,
                parent_state_sha256=parent_state_sha256,
                parent_counts=parent_counts,
                parent_topology_sha256=parent_topology["topology_sha256"],
            )
            report.update(
                {
                    "status": "rejected",
                    "rejection_stage": "in_memory_gates",
                    "checkpoint_published": False,
                    "diagnostics_only": True,
                }
            )
            return _publish_report(config, report)

        staging = (
            config.checkpoint_destination.parent
            / f".lfm2-shared-candidate-{uuid.uuid4().hex}"
        )
        manifest = export_lfm2_shared_checkpoint(
            model,
            staging,
            shared_ffn_layer_groups=plan.selected_pairs,
            parameter_budget=config.expected_child_unique_parameters,
            tensor_format=config.tensor_format,
        )
        staged_files = _checkpoint_files(staging)
        reloaded = load_lfm2_shared_checkpoint(
            staging,
            device=config.device,
            max_parameter_count=config.expected_child_unique_parameters,
        )
        reloaded.eval()
        reloaded_counts = parameter_counts(reloaded)
        reloaded_state_sha256 = residual.model_state_sha256(reloaded)
        reloaded_topology = topology_signature(reloaded)
        reloaded_aliases = audit_shared_aliases(reloaded, plan.selected_pairs)
        reloaded_plan = plan_lfm2_shared_checkpoint(
            reloaded,
            plan.selected_pairs,
            parameter_budget=config.expected_child_unique_parameters,
        )
        reloaded_gradients = gradient_audit(reloaded)
        reload_authentic = bool(
            reloaded_counts == candidate_counts
            and reloaded_state_sha256 == candidate_state_sha256
            and reloaded_topology["topology_sha256"]
            == candidate_topology["topology_sha256"]
            and reloaded_aliases["exact"]
            and reloaded_plan.unique_parameter_count
            == config.expected_child_unique_parameters
            and reloaded_plan.ffn_shared_parameter_savings
            == config.expected_pair_savings
            and reloaded_gradients["zero_gradient"]
        )
        if not reload_authentic:
            raise Lfm2ParameterSharingRunnerError(
                "Fresh standalone reload failed state/count/alias authentication."
            )
        reload_trajectory = evaluate_cached_trajectory(
            reloaded,
            parent_cache=cache,
            parent_rows=parent_rows,
            capture_config=capture_config,
            cache_root=config.work_root / "reload-trajectory",
            minimum_residual_cosine=config.minimum_residual_cosine,
            phase="freshly_reloaded_child",
        )
        reload_behavior = _behavior_event(
            behavior_evaluator,
            reloaded,
            phase="freshly_reloaded_child",
        )
        trajectory_equal = trajectory_equivalence(
            in_memory_trajectory,
            reload_trajectory,
        )
        behavior_equal = behavior_equivalence(
            in_memory_behavior,
            reload_behavior,
        )
        reload_accepted = candidate_accepted(
            reload_trajectory,
            reload_behavior,
        )
        reload_equivalent = bool(
            trajectory_equal["gate_decision_equivalent"]
            and trajectory_equal["cache_key_equal"]
            and behavior_equal["gate_decision_equivalent"]
        )
        report["standalone_reload"] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "staged_files": staged_files,
            "state_sha256": reloaded_state_sha256,
            "parameter_counts": reloaded_counts,
            "checkpoint_plan": reloaded_plan.as_dict(),
            "topology": reloaded_topology,
            "aliases": reloaded_aliases,
            "gradient_audit": reloaded_gradients,
            "authenticated": reload_authentic,
            "trajectory": reload_trajectory,
            "whole_output": reload_behavior,
            "trajectory_equivalence": trajectory_equal,
            "whole_output_equivalence": behavior_equal,
            "reload_gate_equivalent": reload_equivalent,
            "accepted": reload_accepted,
            "parent_model_required": False,
        }
        if not reload_accepted or not reload_equivalent:
            _remove_staging(staging)
            staging = None
            report["rollback"] = _authenticated_rollback(
                installation,
                model,
                parent_state_sha256=parent_state_sha256,
                parent_counts=parent_counts,
                parent_topology_sha256=parent_topology["topology_sha256"],
            )
            report.update(
                {
                    "status": "rejected",
                    "rejection_stage": "fresh_reload_gates_or_equivalence",
                    "checkpoint_published": False,
                    "diagnostics_only": True,
                }
            )
            return _publish_report(config, report)

        os.rename(staging, config.checkpoint_destination)
        staging = None
        checkpoint_created = True
        report.update(
            {
                "status": "accepted",
                "checkpoint_published": True,
                "checkpoint": {
                    "path": str(config.checkpoint_destination),
                    "manifest_sha256": manifest["manifest_sha256"],
                    **_checkpoint_files(config.checkpoint_destination),
                },
                "rollback": {
                    "performed": False,
                    "reason": "candidate and fresh reload both accepted",
                },
                "diagnostics_only": False,
                "source_parent_forward_calls": 0,
            }
        )
        return _publish_report(config, report)
    except Exception as exc:
        cleanup_errors: list[str] = []
        try:
            _remove_staging(staging)
        except Exception as cleanup_exc:  # pragma: no cover - filesystem failure
            cleanup_errors.append(
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
        try:
            _remove_published_checkpoint(
                config.checkpoint_destination,
                created=checkpoint_created,
            )
        except Exception as cleanup_exc:  # pragma: no cover - filesystem failure
            cleanup_errors.append(
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
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
                    parent_topology_sha256=parent_topology["topology_sha256"],
                )
            except Exception as rollback_exc:  # pragma: no cover - catastrophic
                cleanup_errors.append(
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
        report.update(
            {
                "status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "rollback": rollback,
                "cleanup_errors": cleanup_errors,
                "checkpoint_published": False,
                "diagnostics_only": True,
            }
        )
        return _publish_report(config, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Activation-align and physically share disjoint LFM2 FFN pairs "
            "without training or new parent forwards."
        )
    )
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--cached-parent-outputs", type=Path, required=True)
    parser.add_argument("--cached-parent-outputs-sha256", required=True)
    parser.add_argument("--tokenizer-directory", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint-destination", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
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
    parser.add_argument(
        "--merge-weights",
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
        default=28,
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
    parser.add_argument(
        "--tensor-format",
        choices=("safetensors", "raw"),
        default="safetensors",
    )
    return parser


def _config_from_namespace(arguments: argparse.Namespace) -> ParameterSharingRunConfig:
    return ParameterSharingRunConfig(**vars(arguments))


def main(argv: Sequence[str] | None = None) -> int:
    config = _config_from_namespace(build_parser().parse_args(argv))
    report = run_experiment(config)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    if report["status"] == "accepted":
        return 0
    if report["status"] == "rejected":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
