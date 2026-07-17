#!/usr/bin/env python3
"""Run a large, zero-training activation-factorization fold on local LFM2.

The source is an arbitrary already-folded local LFM2 child.  In the intended
frontier run it is the 206,726,912-parameter producer-first child.  This runner
never loads or executes the original parent.  It reuses an authenticated
source-child circuit cache when supplied, or builds that cache solely by
replaying token batches already sealed in the original-parent cache.

Every selected dense FFN projection is replaced physically by two ordinary
Linear factors.  The default selects only ``w2``: its input is the observed
post-SwiGLU activation and its output writes linearly into the residual stream,
while the nonlinear ``w1``/``w3`` producers remain exact.  Ranks are allocated
globally from disjoint activation-fit and held-out-validation paths.  The
default primary budget is 183,760,640 unique
learned scalars (almost exactly 20% below the 229,693,184-parameter original
parent); a 195,239,206 budget is available as the roughly-15% fallback.

Projection reconstruction error is only a local diagnostic.  Promotion also
requires trajectories against both the original parent and incumbent child,
then candidate generation against the sealed whole-output baseline.  The
candidate is always rolled back.  No checkpoint is emitted until behavior
passes and a custom authenticated factorized loader exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

try:
    from experiments import run_lfm2_partial_neuron_bank_fold as bank_runner
    from experiments import run_lfm2_residual_attention_fold as fold_runtime
    from experiments import (
        run_lfm2_residual_parameter_sharing_fold as shared_runner,
    )
except ImportError:  # Direct execution from the experiments directory.
    import run_lfm2_partial_neuron_bank_fold as bank_runner
    import run_lfm2_residual_attention_fold as fold_runtime
    import run_lfm2_residual_parameter_sharing_fold as shared_runner

from neural_forge.hf_lfm2_activation_factorization import (
    InstalledLfm2ActivationFactorization,
    Lfm2ActivationFactorizationEvidence,
    Lfm2ActivationFactorizationPlan,
    Lfm2ActivationFactorizedLinear,
    install_lfm2_activation_factorization_plan,
    plan_lfm2_activation_factorization,
    seal_lfm2_activation_factorization_evidence,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
)


RUN_SCHEMA = "neural-forge-lfm2-activation-factorization-fold-run-v1"
SPLIT_SCHEMA = "neural-forge-lfm2-activation-factorization-path-split-v1"
TRACE_SCHEMA = "neural-forge-lfm2-activation-factorization-trace-rows-v1"
PUBLICATION_BLOCK_STATUS = "behavior_passed_factorized_loader_deferred"

EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS = 229_693_184
STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS = 206_726_912
DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS = 183_760_640
DEFAULT_FALLBACK_TARGET_TOTAL_UNIQUE_PARAMETERS = 195_239_206
DEFAULT_MAX_FIT_ROWS = 512
DEFAULT_MAX_VALIDATION_ROWS = 128
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_MINIMUM_TOTAL_COMPRESSION_FRACTION = 0.149
DEFAULT_MINIMUM_PARENT_RESIDUAL_COSINE = 0.80
DEFAULT_MINIMUM_INCUMBENT_RESIDUAL_COSINE = 0.95
DEFAULT_MINIMUM_TRAJECTORY_SCORE = 0.80
DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION = 0.90
DEFAULT_MAXIMUM_CASE_REGRESSIONS = 2
EXPECTED_WHOLE_OUTPUT_CASE_COUNT = 28
_ROLE_ORDER = ("w1", "w3", "w2")
_PATH_PATTERN = re.compile(
    r"^batch\.(?P<batch>\d{6})/"
    r"(?P<phase>prefill|decode\.(?P<step>\d{3}))/"
    r"row\.(?P<row>\d{4})$"
)


class Lfm2ActivationFactorizationRunnerError(RuntimeError):
    """A local input, cache binding, target, gate, or rollback failed closed."""


@dataclass(frozen=True, slots=True)
class FactorizationBackend:
    """Injectable physical factorization primitive for the shared evaluator."""

    name: str
    plan: Any
    install: Any
    factorized_linear_type: type[torch.nn.Module]
    role_diagnostics: Any
    phase_prefix: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not callable(self.plan)
            or not callable(self.install)
            or not isinstance(self.factorized_linear_type, type)
            or not issubclass(
                self.factorized_linear_type,
                torch.nn.Module,
            )
            or not callable(self.role_diagnostics)
            or not isinstance(self.phase_prefix, str)
            or not self.phase_prefix
        ):
            raise TypeError("FactorizationBackend fields are invalid.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "phase_prefix": self.phase_prefix,
            "factorized_linear_type": (
                f"{self.factorized_linear_type.__module__}."
                f"{self.factorized_linear_type.__qualname__}"
            ),
        }


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
        raise Lfm2ActivationFactorizationRunnerError(
            "Run evidence must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"lfm2-activation-factorization-runner-tensor-v1\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(tensor.shape)))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2ActivationFactorizationRunnerError(
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
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} must be a finite float."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} must be a finite float."
        ) from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} is outside its valid range."
        )
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} must be an explicit local directory, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} must be an explicit local file, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2ActivationFactorizationRunnerError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def resolve_target_accounting(
    *,
    original_parent_unique_parameters: int,
    source_child_unique_parameters: int,
    target_total_unique_parameters: int,
) -> dict[str, Any]:
    """Resolve an explicit physical target without pretending it is exact rank."""

    parent = _positive_int(
        original_parent_unique_parameters,
        "original_parent_unique_parameters",
    )
    source = _positive_int(
        source_child_unique_parameters,
        "source_child_unique_parameters",
    )
    target = _positive_int(
        target_total_unique_parameters,
        "target_total_unique_parameters",
    )
    if source >= parent:
        raise Lfm2ActivationFactorizationRunnerError(
            "The source child must already be smaller than the original parent."
        )
    if target >= source:
        raise Lfm2ActivationFactorizationRunnerError(
            "The factorization target must be below the incumbent source child."
        )
    return {
        "original_parent_unique_parameters": parent,
        "source_child_unique_parameters": source,
        "requested_target_unique_parameter_budget": target,
        "requested_additional_parameters_removed": source - target,
        "requested_total_parameters_removed_from_original_parent": parent - target,
        "requested_additional_compression_fraction_of_incumbent": (
            (source - target) / source
        ),
        "requested_total_compression_fraction_of_original_parent": (
            (parent - target) / parent
        ),
        "rank_granularity_note": (
            "The planner must finish at or below this budget; per-role rank "
            "increments can make the installed count slightly smaller."
        ),
    }


@dataclass(frozen=True, slots=True)
class FactorizationPathSplit:
    """A deterministic, row-disjoint activation fit/validation split."""

    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    fit_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    domain_source: str
    domain_counts: Mapping[str, Mapping[str, int]]
    long_position_counts: Mapping[str, int]
    ordered_path_sha256: str
    domain_assignment_sha256: str
    split_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SPLIT_SCHEMA,
            "fit_indices": list(self.fit_indices),
            "validation_indices": list(self.validation_indices),
            "fit_row_ids": list(self.fit_row_ids),
            "validation_row_ids": list(self.validation_row_ids),
            "fit_count": len(self.fit_indices),
            "validation_count": len(self.validation_indices),
            "row_disjoint": not bool(
                set(self.fit_indices).intersection(self.validation_indices)
            ),
            "domain_source": self.domain_source,
            "domain_counts": {
                domain: dict(counts)
                for domain, counts in sorted(self.domain_counts.items())
            },
            "long_position_counts": dict(self.long_position_counts),
            "ordered_path_sha256": self.ordered_path_sha256,
            "domain_assignment_sha256": self.domain_assignment_sha256,
            "split_sha256": self.split_sha256,
        }


def _path_descriptor(path: str) -> dict[str, Any]:
    match = _PATH_PATTERN.fullmatch(path)
    if match is None:
        raise Lfm2ActivationFactorizationRunnerError(
            f"Unsupported exact circuit path ID: {path!r}."
        )
    phase = match.group("phase")
    step = match.group("step")
    return {
        "path": path,
        "batch": f"batch.{match.group('batch')}",
        "phase": phase,
        "decode_step": None if step is None else int(step),
        "row": int(match.group("row")),
    }


def _round_robin_strata(
    strata: Mapping[tuple[str, str], Sequence[int]],
    *,
    salt: str,
) -> tuple[int, ...]:
    buckets = {
        key: sorted(
            values,
            key=lambda index: hashlib.sha256(
                f"{salt}\0{key}\0{index}".encode("utf-8")
            ).hexdigest(),
        )
        for key, values in strata.items()
        if values
    }
    keys = sorted(
        buckets,
        key=lambda key: hashlib.sha256(
            f"{salt}\0{key}".encode("utf-8")
        ).hexdigest(),
    )
    result: list[int] = []
    cursor = 0
    while keys:
        key = keys[cursor % len(keys)]
        result.append(buckets[key].pop(0))
        if not buckets[key]:
            keys.remove(key)
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor += 1
    return tuple(result)


def deterministic_factorization_split(
    paths: Sequence[str],
    *,
    path_domains: Sequence[str] | None = None,
    max_fit_rows: int = DEFAULT_MAX_FIT_ROWS,
    max_validation_rows: int = DEFAULT_MAX_VALIDATION_ROWS,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> FactorizationPathSplit:
    """Split exact path IDs with deterministic domain and long-horizon coverage."""

    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise Lfm2ActivationFactorizationRunnerError(
            "paths must be a sequence of exact circuit path IDs."
        )
    normalized = tuple(paths)
    if (
        len(normalized) < 2
        or any(not isinstance(path, str) or not path for path in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise Lfm2ActivationFactorizationRunnerError(
            "At least two unique nonempty circuit path IDs are required."
        )
    fit_limit = _positive_int(max_fit_rows, "max_fit_rows")
    validation_limit = _positive_int(
        max_validation_rows,
        "max_validation_rows",
    )
    fraction = _finite_float(
        validation_fraction,
        "validation_fraction",
        minimum=torch.finfo(torch.float64).eps,
        maximum=1.0 - torch.finfo(torch.float64).eps,
    )
    descriptors = tuple(_path_descriptor(path) for path in normalized)
    if path_domains is None:
        domains = tuple(row["batch"] for row in descriptors)
        domain_source = "authenticated_batch_case_surrogate"
    else:
        if (
            isinstance(path_domains, (str, bytes))
            or not isinstance(path_domains, Sequence)
            or len(path_domains) != len(normalized)
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "path_domains must align one-for-one with exact path IDs."
            )
        domains = tuple(str(value).strip() for value in path_domains)
        if any(not value for value in domains):
            raise Lfm2ActivationFactorizationRunnerError(
                "path_domains cannot contain empty labels."
            )
        domain_source = "caller_supplied_authenticated_semantic_domains"

    decode_steps = sorted(
        {
            int(row["decode_step"])
            for row in descriptors
            if row["decode_step"] is not None
        }
    )
    decode_threshold = (
        None
        if not decode_steps
        else decode_steps[max(0, len(decode_steps) // 2)]
    )
    maximum_prefill_row: dict[str, int] = {}
    for row in descriptors:
        if row["phase"] == "prefill":
            maximum_prefill_row[row["batch"]] = max(
                maximum_prefill_row.get(row["batch"], -1),
                int(row["row"]),
            )

    def horizon_band(index: int) -> str:
        row = descriptors[index]
        step = row["decode_step"]
        if step is not None:
            return (
                "long_position"
                if decode_threshold is not None and int(step) >= decode_threshold
                else "early_position"
            )
        maximum = maximum_prefill_row.get(row["batch"], 0)
        return (
            "long_position"
            if maximum > 0 and int(row["row"]) * 2 >= maximum
            else "early_position"
        )

    strata: dict[tuple[str, str], list[int]] = {}
    for index, domain in enumerate(domains):
        strata.setdefault((domain, horizon_band(index)), []).append(index)
    target_validation = max(1, round(len(normalized) * fraction))
    target_validation = min(
        validation_limit,
        target_validation,
        len(normalized) - 1,
    )
    # When capacity permits, reserve at least one path from each observed
    # domain/horizon stratum before filling the remainder.
    target_validation = min(
        validation_limit,
        len(normalized) - 1,
        max(target_validation, min(len(strata), validation_limit)),
    )
    ordered = _round_robin_strata(
        strata,
        salt="lfm2-activation-factorization-validation-v1",
    )
    validation = tuple(ordered[:target_validation])
    validation_set = set(validation)
    remaining_strata = {
        key: [index for index in values if index not in validation_set]
        for key, values in strata.items()
    }
    fit_order = _round_robin_strata(
        remaining_strata,
        salt="lfm2-activation-factorization-fit-v1",
    )
    fit = tuple(fit_order[: min(fit_limit, len(fit_order))])
    if not fit or not validation or set(fit).intersection(validation):
        raise Lfm2ActivationFactorizationRunnerError(
            "Could not build nonempty disjoint fit and validation rows."
        )
    counts: dict[str, dict[str, int]] = {}
    for domain in sorted(set(domains)):
        counts[domain] = {
            "fit": sum(domains[index] == domain for index in fit),
            "validation": sum(
                domains[index] == domain for index in validation
            ),
        }
    long_counts = {
        "fit": sum(horizon_band(index) == "long_position" for index in fit),
        "validation": sum(
            horizon_band(index) == "long_position" for index in validation
        ),
    }
    body = {
        "schema": SPLIT_SCHEMA,
        "fit_indices": list(fit),
        "validation_indices": list(validation),
        "fit_row_ids": [normalized[index] for index in fit],
        "validation_row_ids": [normalized[index] for index in validation],
        "fit_count": len(fit),
        "validation_count": len(validation),
        "row_disjoint": True,
        "domain_source": domain_source,
        "domain_counts": counts,
        "long_position_counts": long_counts,
        "ordered_path_sha256": _json_sha256(list(normalized)),
        "domain_assignment_sha256": _json_sha256(list(domains)),
    }
    return FactorizationPathSplit(
        fit_indices=fit,
        validation_indices=validation,
        fit_row_ids=tuple(normalized[index] for index in fit),
        validation_row_ids=tuple(normalized[index] for index in validation),
        domain_source=domain_source,
        domain_counts=MappingProxyType(
            {
                domain: MappingProxyType(dict(values))
                for domain, values in counts.items()
            }
        ),
        long_position_counts=MappingProxyType(long_counts),
        ordered_path_sha256=body["ordered_path_sha256"],
        domain_assignment_sha256=body["domain_assignment_sha256"],
        split_sha256=_json_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class FactorizationTraceRows:
    """Only exact FFN input geometries needed by the factorization primitive."""

    paths: tuple[str, ...]
    layer_inputs: Mapping[int, torch.Tensor]
    post_swiglu: Mapping[int, torch.Tensor]
    cache_key: str
    corpus_fingerprint: str
    trace_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "path_count": len(self.paths),
            "layer_count": len(self.layer_inputs),
            "cache_key": self.cache_key,
            "corpus_fingerprint": self.corpus_fingerprint,
            "ordered_path_sha256": _json_sha256(list(self.paths)),
            "fields": ["mlp.input", "mlp.post_swiglu"],
            "trace_sha256": self.trace_sha256,
        }


def extract_factorization_trace_rows(
    cache: Lfm2CircuitTraceCache,
) -> FactorizationTraceRows:
    """Extract path-aligned ``mlp.input`` and post-SwiGLU rows from one cache."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    reference = fold_runtime.extract_trace_rows(cache)
    layer_count = len(reference.layer_kinds)
    input_pieces: dict[int, list[torch.Tensor]] = {
        index: [] for index in range(layer_count)
    }
    post_pieces: dict[int, list[torch.Tensor]] = {
        index: [] for index in range(layer_count)
    }
    paths: list[str] = []
    for batch in cache.iter_batches():
        tensors = batch.tensors
        phases = sorted(
            {
                match.group(1)
                for name in tensors
                if (match := fold_runtime._PHASE_PATTERN.match(name)) is not None
                and not name.startswith("cached_prefill.")
            },
            key=fold_runtime._phase_sort_key,
        )
        for phase in phases:
            anchor_name = f"{phase}.layer.000.mlp.input"
            if anchor_name not in tensors:
                continue
            anchor = fold_runtime._row_major_trace(
                tensors[anchor_name],
                "mlp.input",
            )
            phase_paths = [
                f"batch.{batch.index:06d}/{phase}/row.{row:04d}"
                for row in range(int(anchor.shape[0]))
            ]
            paths.extend(phase_paths)
            for layer_index in range(layer_count):
                names = (
                    f"{phase}.layer.{layer_index:03d}.mlp.input",
                    f"{phase}.layer.{layer_index:03d}.mlp.post_swiglu",
                )
                if any(name not in tensors for name in names):
                    raise Lfm2ActivationFactorizationRunnerError(
                        f"Cache phase {phase}, layer {layer_index} lacks "
                        "mlp.input or mlp.post_swiglu."
                    )
                layer_input = fold_runtime._row_major_trace(
                    tensors[names[0]],
                    "mlp.input",
                )
                post = fold_runtime._row_major_trace(
                    tensors[names[1]],
                    "mlp.post_swiglu",
                )
                if (
                    int(layer_input.shape[0]) != len(phase_paths)
                    or int(post.shape[0]) != len(phase_paths)
                ):
                    raise Lfm2ActivationFactorizationRunnerError(
                        "Activation-factor rows are not aligned to exact paths."
                    )
                input_pieces[layer_index].append(layer_input)
                post_pieces[layer_index].append(post)
    if tuple(paths) != reference.paths:
        raise Lfm2ActivationFactorizationRunnerError(
            "Factorization fields and incumbent trajectory paths differ."
        )
    materialized_inputs: dict[int, torch.Tensor] = {}
    materialized_post: dict[int, torch.Tensor] = {}
    descriptors = []
    for layer_index in range(layer_count):
        if not input_pieces[layer_index] or not post_pieces[layer_index]:
            raise Lfm2ActivationFactorizationRunnerError(
                f"Layer {layer_index} has no activation-factor rows."
            )
        layer_input = torch.cat(input_pieces[layer_index], dim=0).contiguous()
        post = torch.cat(post_pieces[layer_index], dim=0).contiguous()
        if int(layer_input.shape[0]) != len(paths) or int(post.shape[0]) != len(
            paths
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "Joined activation-factor rows lost path alignment."
            )
        materialized_inputs[layer_index] = layer_input
        materialized_post[layer_index] = post
        descriptors.extend(
            [
                {
                    "layer": layer_index,
                    "field": "mlp.input",
                    "shape": list(layer_input.shape),
                    "dtype": str(layer_input.dtype),
                    "sha256": _tensor_sha256(layer_input),
                },
                {
                    "layer": layer_index,
                    "field": "mlp.post_swiglu",
                    "shape": list(post.shape),
                    "dtype": str(post.dtype),
                    "sha256": _tensor_sha256(post),
                },
            ]
        )
    body = {
        "schema": TRACE_SCHEMA,
        "cache_key": cache.cache_key,
        "corpus_fingerprint": cache.corpus_fingerprint,
        "paths": paths,
        "tensors": descriptors,
    }
    return FactorizationTraceRows(
        paths=tuple(paths),
        layer_inputs=MappingProxyType(materialized_inputs),
        post_swiglu=MappingProxyType(materialized_post),
        cache_key=cache.cache_key,
        corpus_fingerprint=cache.corpus_fingerprint,
        trace_sha256=_json_sha256(body),
    )


def authenticate_factorization_cache(
    *,
    source_child_identity_sha256: str,
    source_child_cache: Lfm2CircuitTraceCache,
    parent_cache: Lfm2CircuitTraceCache,
    factor_rows: FactorizationTraceRows,
) -> dict[str, Any]:
    """Bind exact model, token corpus, capture protocol, paths, and FFN fields."""

    base = bank_runner.authenticate_source_child_trace(
        source_child_identity_sha256=source_child_identity_sha256,
        source_child_cache=source_child_cache,
        parent_cache=parent_cache,
    )
    if source_child_cache.manifest["capture"] != parent_cache.manifest["capture"]:
        raise Lfm2ActivationFactorizationRunnerError(
            "Source-child and parent caches used different capture protocols."
        )
    incumbent_rows = fold_runtime.extract_trace_rows(source_child_cache)
    parent_rows = fold_runtime.extract_trace_rows(parent_cache)
    if incumbent_rows.paths != parent_rows.paths:
        raise Lfm2ActivationFactorizationRunnerError(
            "Source-child and parent caches do not share exact path IDs."
        )
    if factor_rows.paths != incumbent_rows.paths:
        raise Lfm2ActivationFactorizationRunnerError(
            "Factorization fields do not bind to the incumbent trajectory paths."
        )
    if (
        factor_rows.cache_key != source_child_cache.cache_key
        or factor_rows.corpus_fingerprint
        != source_child_cache.corpus_fingerprint
    ):
        raise Lfm2ActivationFactorizationRunnerError(
            "Factorization fields do not bind to the exact source-child cache."
        )
    return {
        **base,
        "capture_protocol_exact": True,
        "trajectory_paths_exact": True,
        "factorization_fields_exact": True,
        "factorization_trace_sha256": factor_rows.trace_sha256,
        "original_parent_model_loaded": False,
        "new_parent_forward_calls": 0,
        "new_parent_response_generation_calls": 0,
    }


def _load_domain_map(
    path: Path | None,
    expected_sha256: str | None,
    paths: Sequence[str],
) -> tuple[tuple[str, ...] | None, dict[str, Any]]:
    if path is None:
        if expected_sha256 is not None:
            raise Lfm2ActivationFactorizationRunnerError(
                "path_domains_sha256 was supplied without path_domains_json."
            )
        return None, {
            "supplied": False,
            "semantic_domains_authenticated": False,
            "fallback": "exact batch IDs are used as case/domain strata",
        }
    if expected_sha256 is None:
        raise Lfm2ActivationFactorizationRunnerError(
            "path_domains_json requires path_domains_sha256."
        )
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise Lfm2ActivationFactorizationRunnerError(
            "The supplied path-domain map changed identity."
        )
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2ActivationFactorizationRunnerError(
            "The path-domain map is not valid JSON."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise Lfm2ActivationFactorizationRunnerError(
            "The path-domain map must be an object."
        )
    labels = []
    for exact_path in paths:
        batch = exact_path.split("/", 1)[0]
        value = parsed.get(exact_path, parsed.get(batch))
        if not isinstance(value, str) or not value.strip():
            raise Lfm2ActivationFactorizationRunnerError(
                f"The path-domain map has no label for {exact_path!r}."
            )
        labels.append(value.strip())
    return tuple(labels), {
        "supplied": True,
        "semantic_domains_authenticated": True,
        "path": str(path),
        "file_sha256": observed,
        "mapping_sha256": _json_sha256(dict(parsed)),
    }


@dataclass(frozen=True, slots=True)
class ActivationFactorizationRunConfig:
    """Authenticated local inputs and one meaningful physical frontier."""

    source_child_model: Path
    parent_cache: Path
    cached_parent_outputs: Path
    cached_parent_outputs_sha256: str
    work_root: Path
    checkpoint_destination: Path
    report_path: Path
    source_child_cache: Path | None = None
    whole_output_parent_cache: Path | None = None
    tokenizer_directory: Path | None = None
    path_domains_json: Path | None = None
    path_domains_sha256: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    target_total_unique_parameters: int = (
        DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS
    )
    fallback_target_total_unique_parameters: int | None = (
        DEFAULT_FALLBACK_TARGET_TOTAL_UNIQUE_PARAMETERS
    )
    expected_original_parent_unique_parameters: int = (
        EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS
    )
    expected_source_child_unique_parameters: int | None = None
    expected_whole_output_case_count: int = (
        EXPECTED_WHOLE_OUTPUT_CASE_COUNT
    )
    roles: tuple[str, ...] = ("w2",)
    max_fit_rows: int = DEFAULT_MAX_FIT_ROWS
    max_validation_rows: int = DEFAULT_MAX_VALIDATION_ROWS
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    minimum_total_compression_fraction: float = (
        DEFAULT_MINIMUM_TOTAL_COMPRESSION_FRACTION
    )
    minimum_parent_residual_cosine: float = (
        DEFAULT_MINIMUM_PARENT_RESIDUAL_COSINE
    )
    minimum_incumbent_residual_cosine: float = (
        DEFAULT_MINIMUM_INCUMBENT_RESIDUAL_COSINE
    )
    minimum_parent_trajectory_score: float = (
        DEFAULT_MINIMUM_TRAJECTORY_SCORE
    )
    minimum_incumbent_trajectory_score: float = (
        DEFAULT_MINIMUM_TRAJECTORY_SCORE
    )
    whole_output_max_new_tokens: int = 64
    whole_output_maximum_prompt_tokens: int = 512
    whole_output_fragment_minimum_characters: int = 12
    whole_output_maximum_new_empty_responses: int = 2
    whole_output_maximum_new_fragment_responses: int = 2
    whole_output_minimum_parent_nonempty_retention: float = (
        DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION
    )
    whole_output_minimum_task_score_retention: float = (
        DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION
    )
    whole_output_maximum_parent_task_score_regressions: int = (
        DEFAULT_MAXIMUM_CASE_REGRESSIONS
    )
    whole_output_maximum_new_generation_limit_saturations: int = 2
    whole_output_maximum_repeated_ngram_share_regressions: int = 2
    whole_output_maximum_repeated_bigram_share_delta: float = 0.10

    def __post_init__(self) -> None:
        source = _local_directory(self.source_child_model, "source_child_model")
        parent_cache = _local_directory(self.parent_cache, "parent_cache")
        outputs = _local_file(
            self.cached_parent_outputs,
            "cached_parent_outputs",
        )
        tokenizer = (
            source
            if self.tokenizer_directory is None
            else _local_directory(
                self.tokenizer_directory,
                "tokenizer_directory",
            )
        )
        child_cache = (
            None
            if self.source_child_cache is None
            else _local_directory(
                self.source_child_cache,
                "source_child_cache",
            )
        )
        whole_output_cache = (
            parent_cache
            if self.whole_output_parent_cache is None
            else _local_directory(
                self.whole_output_parent_cache,
                "whole_output_parent_cache",
            )
        )
        domains = (
            None
            if self.path_domains_json is None
            else _local_file(self.path_domains_json, "path_domains_json")
        )
        object.__setattr__(self, "source_child_model", source)
        object.__setattr__(self, "parent_cache", parent_cache)
        object.__setattr__(self, "cached_parent_outputs", outputs)
        object.__setattr__(self, "tokenizer_directory", tokenizer)
        object.__setattr__(self, "source_child_cache", child_cache)
        object.__setattr__(
            self,
            "whole_output_parent_cache",
            whole_output_cache,
        )
        object.__setattr__(self, "path_domains_json", domains)
        for name in ("work_root", "checkpoint_destination", "report_path"):
            path = Path(getattr(self, name)).expanduser().absolute()
            object.__setattr__(self, name, path)
        protected = [source, parent_cache, whole_output_cache, tokenizer]
        if child_cache is not None:
            protected.append(child_cache)
        for directory in protected:
            for output in (
                self.work_root,
                self.checkpoint_destination,
                self.report_path,
            ):
                if _is_within(output, directory):
                    raise Lfm2ActivationFactorizationRunnerError(
                        "Work, checkpoint, and report paths must stay outside "
                        "source model/cache/evidence directories."
                    )
        if domains is not None and domains in (
            self.work_root,
            self.checkpoint_destination,
            self.report_path,
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "Output paths cannot replace the path-domain evidence file."
            )
        if self.checkpoint_destination == self.report_path:
            raise Lfm2ActivationFactorizationRunnerError(
                "checkpoint_destination and report_path must differ."
            )
        if self.checkpoint_destination.exists():
            raise Lfm2ActivationFactorizationRunnerError(
                "checkpoint_destination exists; refusing replacement."
            )
        if self.report_path.exists():
            raise Lfm2ActivationFactorizationRunnerError(
                "report_path exists; refusing replacement."
            )
        for name in (
            "target_total_unique_parameters",
            "expected_original_parent_unique_parameters",
            "expected_whole_output_case_count",
            "max_fit_rows",
            "max_validation_rows",
            "whole_output_max_new_tokens",
            "whole_output_maximum_prompt_tokens",
            "whole_output_fragment_minimum_characters",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        for name in (
            "whole_output_maximum_new_empty_responses",
            "whole_output_maximum_new_fragment_responses",
            "whole_output_maximum_parent_task_score_regressions",
            "whole_output_maximum_new_generation_limit_saturations",
            "whole_output_maximum_repeated_ngram_share_regressions",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        if self.fallback_target_total_unique_parameters is not None:
            fallback = _positive_int(
                self.fallback_target_total_unique_parameters,
                "fallback_target_total_unique_parameters",
            )
            if fallback <= self.target_total_unique_parameters:
                raise Lfm2ActivationFactorizationRunnerError(
                    "The fallback target must be larger (less aggressive) "
                    "than the primary target."
                )
            object.__setattr__(
                self,
                "fallback_target_total_unique_parameters",
                fallback,
            )
        if self.expected_source_child_unique_parameters is not None:
            source_count = _positive_int(
                self.expected_source_child_unique_parameters,
                "expected_source_child_unique_parameters",
            )
            object.__setattr__(
                self,
                "expected_source_child_unique_parameters",
                source_count,
            )
            self._validate_targets_for_source(source_count)
        if (
            isinstance(self.roles, (str, bytes))
            or not isinstance(self.roles, Sequence)
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "roles must be a sequence drawn from w1,w3,w2."
            )
        roles = tuple(self.roles)
        if (
            not roles
            or len(set(roles)) != len(roles)
            or any(role not in _ROLE_ORDER for role in roles)
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "roles must be unique values drawn from w1,w3,w2."
            )
        object.__setattr__(self, "roles", roles)
        for name in (
            "validation_fraction",
            "minimum_total_compression_fraction",
            "minimum_parent_residual_cosine",
            "minimum_incumbent_residual_cosine",
            "minimum_parent_trajectory_score",
            "minimum_incumbent_trajectory_score",
            "whole_output_minimum_parent_nonempty_retention",
            "whole_output_minimum_task_score_retention",
            "whole_output_maximum_repeated_bigram_share_delta",
        ):
            minimum = -1.0 if "residual_cosine" in name else 0.0
            maximum = 1.0
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        if not 0.0 < self.validation_fraction < 1.0:
            raise Lfm2ActivationFactorizationRunnerError(
                "validation_fraction must be strictly between zero and one."
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise Lfm2ActivationFactorizationRunnerError(
                "device must be nonempty."
            )
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise Lfm2ActivationFactorizationRunnerError(
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
        if self.path_domains_sha256 is not None:
            object.__setattr__(
                self,
                "path_domains_sha256",
                _digest(
                    self.path_domains_sha256,
                    "path_domains_sha256",
                ),
            )
        if (domains is None) != (self.path_domains_sha256 is None):
            raise Lfm2ActivationFactorizationRunnerError(
                "path_domains_json and path_domains_sha256 must be supplied together."
            )

    def _validate_targets_for_source(self, source_count: int) -> None:
        targets = [self.target_total_unique_parameters]
        if (
            self.fallback_target_total_unique_parameters is not None
            and self.fallback_target_total_unique_parameters < source_count
        ):
            targets.append(self.fallback_target_total_unique_parameters)
        for target in targets:
            accounting = resolve_target_accounting(
                original_parent_unique_parameters=(
                    self.expected_original_parent_unique_parameters
                ),
                source_child_unique_parameters=source_count,
                target_total_unique_parameters=target,
            )
            if (
                accounting[
                    "requested_total_compression_fraction_of_original_parent"
                ]
                < self.minimum_total_compression_fraction
            ):
                raise Lfm2ActivationFactorizationRunnerError(
                    "The requested target is a tiny experiment; this runner "
                    "requires roughly 15% or greater total compression."
                )

    def target_schedule(self, source_count: int) -> tuple[int, ...]:
        self._validate_targets_for_source(source_count)
        values = [self.target_total_unique_parameters]
        if (
            self.fallback_target_total_unique_parameters is not None
            and self.fallback_target_total_unique_parameters < source_count
        ):
            values.append(self.fallback_target_total_unique_parameters)
        return tuple(values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_child_model": str(self.source_child_model),
            "source_child_cache": (
                None
                if self.source_child_cache is None
                else str(self.source_child_cache)
            ),
            "parent_cache": str(self.parent_cache),
            "whole_output_parent_cache": str(
                self.whole_output_parent_cache
            ),
            "cached_parent_outputs": str(self.cached_parent_outputs),
            "cached_parent_outputs_sha256": (
                self.cached_parent_outputs_sha256
            ),
            "tokenizer_directory": str(self.tokenizer_directory),
            "path_domains_json": (
                None
                if self.path_domains_json is None
                else str(self.path_domains_json)
            ),
            "path_domains_sha256": self.path_domains_sha256,
            "work_root": str(self.work_root),
            "checkpoint_destination": str(self.checkpoint_destination),
            "report_path": str(self.report_path),
            "device": self.device,
            "dtype": self.dtype,
            "primary_target_total_unique_parameters": (
                self.target_total_unique_parameters
            ),
            "fallback_target_total_unique_parameters": (
                self.fallback_target_total_unique_parameters
            ),
            "expected_original_parent_unique_parameters": (
                self.expected_original_parent_unique_parameters
            ),
            "expected_source_child_unique_parameters": (
                self.expected_source_child_unique_parameters
            ),
            "activation_split": {
                "max_fit_rows": self.max_fit_rows,
                "max_validation_rows": self.max_validation_rows,
                "validation_fraction": self.validation_fraction,
                "long_position_coverage_required_when_available": True,
                "domain_coverage_required_when_capacity_allows": True,
            },
            "factorized_roles": list(self.roles),
            "default_role_rationale": (
                "w2 is linear from the observed post-SwiGLU activation into "
                "the residual stream; w1/w3 nonlinear producers remain exact"
            ),
            "trajectory_gates": {
                "minimum_parent_residual_cosine": (
                    self.minimum_parent_residual_cosine
                ),
                "minimum_incumbent_residual_cosine": (
                    self.minimum_incumbent_residual_cosine
                ),
                "minimum_parent_score": (
                    self.minimum_parent_trajectory_score
                ),
                "minimum_incumbent_score": (
                    self.minimum_incumbent_trajectory_score
                ),
            },
            "whole_output_gate": {
                "expected_case_count": self.expected_whole_output_case_count,
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
            "original_parent_model_loaded": False,
            "new_parent_forwards_allowed": False,
            "new_parent_response_generation_allowed": False,
            "candidate_generation_required_for_coherence_gate": True,
            "training_allowed": False,
            "standalone_checkpoint_available": False,
        }


def _index_rows(
    values: Mapping[int, torch.Tensor],
    indices: Sequence[int],
) -> dict[int, torch.Tensor]:
    index = torch.tensor(tuple(indices), dtype=torch.long)
    return {
        layer_index: rows.index_select(0, index).contiguous()
        for layer_index, rows in values.items()
    }


def build_factorization_evidence(
    model: torch.nn.Module,
    rows: FactorizationTraceRows,
    split: FactorizationPathSplit,
) -> Lfm2ActivationFactorizationEvidence:
    """Seal exact child/cache-bound fit and held-out activation geometries."""

    evidence = seal_lfm2_activation_factorization_evidence(
        model,
        _index_rows(rows.layer_inputs, split.fit_indices),
        _index_rows(rows.layer_inputs, split.validation_indices),
        _index_rows(rows.post_swiglu, split.fit_indices),
        _index_rows(rows.post_swiglu, split.validation_indices),
        fit_row_ids=split.fit_row_ids,
        validation_row_ids=split.validation_row_ids,
        cache_key=rows.cache_key,
        heldout_corpus_sha256=rows.corpus_fingerprint,
    )
    if (
        evidence.cache_key != rows.cache_key
        or evidence.heldout_corpus_sha256 != rows.corpus_fingerprint
        or set(evidence.fit_row_ids).intersection(evidence.validation_row_ids)
    ):
        raise Lfm2ActivationFactorizationRunnerError(
            "The sealed evidence lost cache binding or split disjointness."
        )
    return evidence


def _checkpoint_absence_audit(
    config: ActivationFactorizationRunConfig,
) -> dict[str, Any]:
    if config.checkpoint_destination.exists():
        raise Lfm2ActivationFactorizationRunnerError(
            "A checkpoint appeared before a factorized loader was authorized."
        )
    return {
        "requested_destination": str(config.checkpoint_destination),
        "exists": False,
        "files_written": 0,
        "python_object_serialization_used": False,
        "dense_fallback_used": False,
        "custom_factorized_loader_available": False,
    }


def _authenticated_rollback(
    installation: InstalledLfm2ActivationFactorization,
    model: torch.nn.Module,
    *,
    source_state_sha256: str,
    source_counts: Mapping[str, int],
    source_topology_sha256: str,
) -> dict[str, Any]:
    """Restore and independently authenticate the exact incumbent child."""

    try:
        rolled_back = installation.rollback()
        state = fold_runtime.model_state_sha256(model)
        counts = shared_runner.parameter_counts(model)
        topology = shared_runner.topology_signature(model)
        gradients = shared_runner.gradient_audit(model)
    except BaseException as exc:
        raise Lfm2ActivationFactorizationRunnerError(
            "Activation-factor candidate could not restore the incumbent child."
        ) from exc
    exact = bool(
        rolled_back
        and state == source_state_sha256
        and counts == dict(source_counts)
        and topology["topology_sha256"] == source_topology_sha256
        and gradients["zero_gradient"]
    )
    if not exact:
        raise Lfm2ActivationFactorizationRunnerError(
            "Activation-factor rollback did not restore the exact incumbent child."
        )
    return {
        "performed": True,
        "exact": True,
        "restored_role": "incumbent_source_child",
        "source_state_sha256_restored": state,
        "source_parameter_counts_restored": counts,
        "source_topology_sha256_restored": topology["topology_sha256"],
        "gradient_audit": gradients,
    }


def _role_diagnostics(
    plan: Lfm2ActivationFactorizationPlan,
) -> list[dict[str, Any]]:
    return [
        {
            "layer_index": role.layer_index,
            "role": role.role,
            "rank": role.rank,
            "method": role.factorization.method,
            "fit_relative_rmse": role.factorization.fit_relative_rmse,
            "heldout_validation_relative_rmse": (
                role.factorization.validation_relative_rmse
            ),
            "raw_svd_heldout_validation_relative_rmse": (
                role.factorization.raw_svd_validation_relative_rmse
            ),
            "dense_parameters": role.dense_parameters,
            "factor_parameters": role.factor_parameters,
            "parameters_saved": role.parameters_saved,
            "local_projection_reconstruction_only": True,
            "evidence_of_coherent_language_behavior": False,
        }
        for role in plan.role_plans
    ]


def _default_factorization_backend() -> FactorizationBackend:
    return FactorizationBackend(
        name="independent_activation_weighted_factors",
        plan=plan_lfm2_activation_factorization,
        install=install_lfm2_activation_factorization_plan,
        factorized_linear_type=Lfm2ActivationFactorizedLinear,
        role_diagnostics=_role_diagnostics,
        phase_prefix="activation_factorization",
    )


def _candidate_accepted(
    parent_trajectory: Mapping[str, Any],
    incumbent_trajectory: Mapping[str, Any],
    whole_output: Mapping[str, Any],
) -> bool:
    return bank_runner.candidate_accepted(
        parent_trajectory,
        incumbent_trajectory,
        whole_output,
    )


def _sealed_report(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("run_sha256", None)
    return {**body, "run_sha256": _json_sha256(body)}


def _publish_report(
    config: ActivationFactorizationRunConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = _sealed_report(report)
    try:
        shared_runner._atomic_write_new_json(config.report_path, sealed)
    except BaseException as exc:
        raise Lfm2ActivationFactorizationRunnerError(
            "Could not atomically publish the sealed factorization report."
        ) from exc
    return sealed


def _evaluate_target(
    *,
    model: torch.nn.Module,
    target: int,
    evidence: Lfm2ActivationFactorizationEvidence,
    config: ActivationFactorizationRunConfig,
    parent_cache: Lfm2CircuitTraceCache,
    parent_rows: Any,
    source_cache: Lfm2CircuitTraceCache,
    source_rows: Any,
    capture_config: Any,
    behavior_evaluator: Any,
    source_state_sha256: str,
    source_counts: Mapping[str, int],
    source_topology_sha256: str,
    operator_objects: Sequence[Any],
    backend: FactorizationBackend,
) -> tuple[dict[str, Any], bool]:
    """Plan, install, evaluate, and always roll back one physical target."""

    installation: Any | None = None
    attempt: dict[str, Any] = {
        "requested_target_unique_parameter_budget": target,
        "status": "planning",
        "accepted": False,
    }
    try:
        plan = backend.plan(
            model,
            evidence,
            target_unique_parameter_budget=target,
            roles=config.roles,
        )
        if (
            plan.projected_unique_parameters_after > target
            or plan.unique_parameters_before != source_counts["unique"]
            or plan.evidence_sha256 != evidence.evidence_sha256
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "The sealed factorization plan missed its physical target."
            )
        installation = backend.install(
            model,
            evidence,
            plan,
        )
        candidate_counts = shared_runner.parameter_counts(model)
        candidate_gradients = shared_runner.gradient_audit(model)
        observed_operators = tuple(
            (
                layer.self_attn
                if layer.is_attention_layer
                else layer.conv
            )
            for layer in shared_runner._model_layers(model)
        )
        factorized_roles = tuple(
            (layer_index, role)
            for layer_index, layer in enumerate(
                shared_runner._model_layers(model)
            )
            for role in _ROLE_ORDER
            if isinstance(
                getattr(layer.feed_forward, role),
                backend.factorized_linear_type,
            )
        )
        if (
            candidate_counts["unique"]
            != plan.projected_unique_parameters_after
            or tuple(plan.selected_roles) != factorized_roles
            or not candidate_gradients["zero_gradient"]
            or any(
                before is not after
                for before, after in zip(
                    operator_objects,
                    observed_operators,
                )
            )
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "Installed candidate failed physical count, role, gradient, "
                "or attention/recurrent identity checks."
            )
        candidate_capture = bank_runner.capture_live_candidate_trajectory(
            model,
            token_cache=parent_cache,
            capture_config=capture_config,
        )
        parent_trajectory = bank_runner.evaluate_live_trajectory(
            reference_name="original_parent",
            reference_cache=parent_cache,
            reference_rows=parent_rows,
            candidate=candidate_capture,
            minimum_residual_cosine=(
                config.minimum_parent_residual_cosine
            ),
            minimum_score=config.minimum_parent_trajectory_score,
        )
        incumbent_trajectory = bank_runner.evaluate_live_trajectory(
            reference_name="incumbent_source_child",
            reference_cache=source_cache,
            reference_rows=source_rows,
            candidate=candidate_capture,
            minimum_residual_cosine=(
                config.minimum_incumbent_residual_cosine
            ),
            minimum_score=config.minimum_incumbent_trajectory_score,
        )
        whole_output = shared_runner._behavior_event(
            behavior_evaluator,
            model,
            phase=f"{backend.phase_prefix}_target_{target}",
        )
        accepted = _candidate_accepted(
            parent_trajectory,
            incumbent_trajectory,
            whole_output,
        )
        attempt.update(
            {
                "status": "evaluated",
                "accepted": accepted,
                "target_accounting": resolve_target_accounting(
                    original_parent_unique_parameters=(
                        config.expected_original_parent_unique_parameters
                    ),
                    source_child_unique_parameters=source_counts["unique"],
                    target_total_unique_parameters=target,
                ),
                "plan": plan.as_dict(),
                "installation": installation.report.as_dict(),
                "candidate": {
                    "identity_sha256": (
                        installation.report.model_identity_after
                    ),
                    "parameter_counts": candidate_counts,
                    "physical_unique_parameters": candidate_counts["unique"],
                    "physical_parameters_removed_from_incumbent": (
                        source_counts["unique"] - candidate_counts["unique"]
                    ),
                    "physical_parameters_removed_from_original_parent": (
                        config.expected_original_parent_unique_parameters
                        - candidate_counts["unique"]
                    ),
                    "physical_total_compression_fraction": (
                        (
                            config.expected_original_parent_unique_parameters
                            - candidate_counts["unique"]
                        )
                        / config.expected_original_parent_unique_parameters
                    ),
                    "factorized_role_count": len(factorized_roles),
                    "factorized_roles": [
                        [layer_index, role]
                        for layer_index, role in factorized_roles
                    ],
                    "attention_and_recurrent_operator_objects_retained": True,
                    "gradient_audit": candidate_gradients,
                    "training_steps": 0,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                },
                "local_projection_reconstruction": {
                    "interpretation": (
                        "Held-out matrix-output reconstruction only; these "
                        "numbers do not prove coherence or intelligence."
                    ),
                    "roles": backend.role_diagnostics(plan),
                },
                "behavioral_evaluation": {
                    "interpretation": (
                        "Only the trajectory and whole-output gates determine "
                        "whether local reconstruction composes into coherent "
                        "model behavior."
                    ),
                    "live_capture": {
                        "capture_sha256": candidate_capture.capture_sha256,
                        "candidate_forward_calls": (
                            candidate_capture.forward_calls
                        ),
                        "parent_forward_calls": 0,
                    },
                    "original_parent_trajectory": parent_trajectory,
                    "incumbent_child_trajectory": incumbent_trajectory,
                    "cached_parent_whole_output": whole_output,
                    "candidate_generation_invocations": whole_output[
                        "child_generate_invocations"
                    ],
                    "new_parent_generation_invocations": 0,
                    "accepted": accepted,
                    "exact_token_match_is_diagnostic_only": True,
                    "exact_generated_text_is_diagnostic_only": True,
                },
            }
        )
        attempt["rollback"] = _authenticated_rollback(
            installation,
            model,
            source_state_sha256=source_state_sha256,
            source_counts=source_counts,
            source_topology_sha256=source_topology_sha256,
        )
        installation = None
        return attempt, accepted
    except BaseException as exc:
        rollback = None
        rollback_error = None
        if installation is not None and installation.active:
            try:
                rollback = _authenticated_rollback(
                    installation,
                    model,
                    source_state_sha256=source_state_sha256,
                    source_counts=source_counts,
                    source_topology_sha256=source_topology_sha256,
                )
            except BaseException as rollback_exc:
                rollback_error = (
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
        if rollback_error is not None:
            raise Lfm2ActivationFactorizationRunnerError(
                "Candidate failed and exact rollback also failed: "
                f"{rollback_error}"
            ) from exc
        attempt.update(
            {
                "status": "error",
                "accepted": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "rollback": rollback,
            }
        )
        return attempt, False


def run_experiment(
    config: ActivationFactorizationRunConfig,
    *,
    backend: FactorizationBackend | None = None,
    run_schema: str = RUN_SCHEMA,
) -> dict[str, Any]:
    """Run the aggressive frontier and optional fallback, preserving source."""

    if not isinstance(config, ActivationFactorizationRunConfig):
        raise TypeError("config must be ActivationFactorizationRunConfig.")
    if backend is None:
        backend = _default_factorization_backend()
    if not isinstance(backend, FactorizationBackend):
        raise TypeError("backend must be a FactorizationBackend.")
    if not isinstance(run_schema, str) or not run_schema:
        raise TypeError("run_schema must be a nonempty string.")
    config.work_root.mkdir(parents=True, exist_ok=True)
    config.checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    active_installation: InstalledLfm2ActivationFactorization | None = None
    model: torch.nn.Module | None = None
    source_state_sha256: str | None = None
    source_counts: dict[str, int] | None = None
    source_topology: dict[str, Any] | None = None
    report: dict[str, Any] = {
        "schema": run_schema,
        "status": "initializing",
        "publication_status": "not_evaluated",
        "config": config.as_dict(),
        "factorization_backend": backend.as_dict(),
        "training": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "optimizer_constructed": False,
            "gradients_used": False,
        },
        "original_parent": {
            "model_loaded": False,
            "forward_calls": 0,
            "response_generation_calls": 0,
            "role": "sealed_cache_reference_only",
        },
        "checkpoint_published": False,
        "checkpoint_absence": _checkpoint_absence_audit(config),
        "promotion_policy": {
            "original_parent_trajectory_required": True,
            "incumbent_child_trajectory_required": True,
            "cached_whole_output_coherence_required": True,
            "local_projection_reconstruction_is_not_behavior": True,
            "exact_token_match_required": False,
            "exact_generated_text_required": False,
            "standalone_checkpoint_deferred_until_behavior_passes": True,
        },
    }
    try:
        parent_cache = Lfm2CircuitTraceCache.open(
            config.parent_cache,
            verify=True,
        )
        whole_output_parent_cache = Lfm2CircuitTraceCache.open(
            config.whole_output_parent_cache,
            verify=True,
        )
        if (
            whole_output_parent_cache.parent_fingerprint
            != parent_cache.parent_fingerprint
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "Trajectory and whole-output caches do not belong to the "
                "same exact original parent."
            )
        capture_config = shared_runner.capture_config_from_parent_cache(
            parent_cache
        )
        parent_rows = fold_runtime.extract_trace_rows(parent_cache)
        model = fold_runtime.load_local_lfm2_model(
            config.source_child_model,
            device=config.device,
            dtype=fold_runtime._torch_dtype(config.dtype),
        )
        model.eval()
        source_counts = shared_runner.parameter_counts(model)
        source_topology = shared_runner.topology_signature(model)
        source_state_sha256 = fold_runtime.model_state_sha256(model)
        source_identity = fingerprint_lfm2_parent(model).identity_sha256
        gradients_before = shared_runner.gradient_audit(model)
        if not gradients_before["zero_gradient"]:
            raise Lfm2ActivationFactorizationRunnerError(
                "The source child already contains gradients."
            )
        if (
            config.expected_source_child_unique_parameters is not None
            and source_counts["unique"]
            != config.expected_source_child_unique_parameters
        ):
            raise Lfm2ActivationFactorizationRunnerError(
                "The source child unique-parameter count differs from the "
                "explicit configured expectation."
            )
        schedule = config.target_schedule(source_counts["unique"])
        operator_objects = tuple(
            (
                layer.self_attn
                if layer.is_attention_layer
                else layer.conv
            )
            for layer in shared_runner._model_layers(model)
        )
        if config.source_child_cache is None:
            source_cache, source_rows, source_cache_audit = (
                bank_runner.build_source_child_trace_cache(
                    parent_cache=parent_cache,
                    source_child=model,
                    cache_root=config.work_root / "source-child-trace",
                    capture_config=capture_config,
                )
            )
            source_cache_origin = "built_from_existing_parent_cache_token_batches"
        else:
            source_cache = Lfm2CircuitTraceCache.open(
                config.source_child_cache,
                verify=True,
            )
            source_rows = fold_runtime.extract_trace_rows(source_cache)
            source_cache_audit = {
                "cache_reused": True,
                "source_child_forward_calls": 0,
                "parent_forward_calls": 0,
                "trace_rows_sha256": source_rows.trace_rows_sha256,
            }
            source_cache_origin = "caller_supplied_authenticated_cache"
        factor_rows = extract_factorization_trace_rows(source_cache)
        cache_binding = authenticate_factorization_cache(
            source_child_identity_sha256=source_identity,
            source_child_cache=source_cache,
            parent_cache=parent_cache,
            factor_rows=factor_rows,
        )
        domains, domain_audit = _load_domain_map(
            config.path_domains_json,
            config.path_domains_sha256,
            factor_rows.paths,
        )
        split = deterministic_factorization_split(
            factor_rows.paths,
            path_domains=domains,
            max_fit_rows=config.max_fit_rows,
            max_validation_rows=config.max_validation_rows,
            validation_fraction=config.validation_fraction,
        )
        evidence = build_factorization_evidence(model, factor_rows, split)
        tokenizer_fingerprint = shared_runner._tokenizer_fingerprint(
            config.tokenizer_directory
        )
        tokenizer = fold_runtime.load_local_tokenizer(
            config.tokenizer_directory
        )
        behavior_evaluator = shared_runner._behavior_evaluator(
            config,
            cache=whole_output_parent_cache,
            tokenizer=tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        attempts = []
        accepted = False
        accepted_target: int | None = None
        for attempt_index, target in enumerate(schedule):
            attempt, attempt_accepted = _evaluate_target(
                model=model,
                target=target,
                evidence=evidence,
                config=config,
                parent_cache=parent_cache,
                parent_rows=parent_rows,
                source_cache=source_cache,
                source_rows=source_rows,
                capture_config=capture_config,
                behavior_evaluator=behavior_evaluator,
                source_state_sha256=source_state_sha256,
                source_counts=source_counts,
                source_topology_sha256=(
                    source_topology["topology_sha256"]
                ),
                operator_objects=operator_objects,
                backend=backend,
            )
            attempt["attempt_index"] = attempt_index
            attempt["frontier"] = (
                "primary_approximately_20_percent"
                if attempt_index == 0
                else "fallback_approximately_15_percent"
            )
            attempts.append(attempt)
            if attempt_accepted:
                accepted = True
                accepted_target = target
                break
        report.update(
            {
                "status": "candidate_evaluated",
                "local_io": fold_runtime.direct_local_io_audit(
                    config.source_child_model
                ),
                "parent_cache": {
                    "path": str(parent_cache.path),
                    "cache_key": parent_cache.cache_key,
                    "parent_fingerprint": (
                        parent_cache.parent_fingerprint
                    ),
                    "corpus_fingerprint": (
                        parent_cache.corpus_fingerprint
                    ),
                    "verified": True,
                    "parent_model_loaded": False,
                    "new_parent_forward_calls": 0,
                    "new_parent_response_generation_calls": 0,
                },
                "whole_output_parent_cache": {
                    "path": str(whole_output_parent_cache.path),
                    "cache_key": whole_output_parent_cache.cache_key,
                    "parent_fingerprint": (
                        whole_output_parent_cache.parent_fingerprint
                    ),
                    "verified": True,
                    "same_exact_parent_as_trajectory_cache": True,
                    "used_for_cached_whole_output_binding_only": True,
                    "new_parent_forward_calls": 0,
                    "new_parent_response_generation_calls": 0,
                },
                "source_child_cache": {
                    "path": str(source_cache.path),
                    "origin": source_cache_origin,
                    "cache_key": source_cache.cache_key,
                    "source_child_fingerprint": (
                        source_cache.parent_fingerprint
                    ),
                    "corpus_fingerprint": (
                        source_cache.corpus_fingerprint
                    ),
                    "verified": True,
                    **source_cache_audit,
                    "binding": cache_binding,
                },
                "source_child": {
                    "identity_sha256": source_identity,
                    "state_sha256": source_state_sha256,
                    "parameter_counts": source_counts,
                    "topology": source_topology,
                    "gradient_audit": gradients_before,
                    "strong_10pct_incumbent_count_match": (
                        source_counts["unique"]
                        == STRONG_SOURCE_CHILD_UNIQUE_PARAMETERS
                    ),
                },
                "activation_evidence": {
                    "factor_rows": factor_rows.as_dict(),
                    "path_domain_audit": domain_audit,
                    "split": split.as_dict(),
                    "sealed_evidence": evidence.as_dict(),
                    "source_fields": [
                        "mlp.input",
                        "mlp.post_swiglu",
                    ],
                    "new_parent_forward_calls": 0,
                    "new_parent_response_generation_calls": 0,
                },
                "attempts": attempts,
                "accepted": accepted,
                "accepted_target_unique_parameter_budget": accepted_target,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "candidate_generation_invocations": sum(
                    int(
                        attempt.get("behavioral_evaluation", {}).get(
                            "candidate_generation_invocations",
                            0,
                        )
                    )
                    for attempt in attempts
                ),
                "new_parent_forward_calls": 0,
                "new_parent_response_generation_calls": 0,
            }
        )
        report["checkpoint_absence"] = _checkpoint_absence_audit(config)
        if not accepted:
            report.update(
                {
                    "status": "rejected",
                    "publication_status": "rejected_by_conservative_gates",
                    "candidate_gate_accepted": False,
                    "checkpoint_published": False,
                    "diagnostics_only": True,
                }
            )
            return _publish_report(config, report)
        report.update(
            {
                "status": PUBLICATION_BLOCK_STATUS,
                "publication_status": PUBLICATION_BLOCK_STATUS,
                "candidate_gate_accepted": True,
                "checkpoint_published": False,
                "diagnostics_only": False,
                "publication_blocker": {
                    "code": PUBLICATION_BLOCK_STATUS,
                    "reason": (
                        "Behavior passed, but an authenticated standalone "
                        "factorized-LFM2 serializer/reloader is the next "
                        "required component."
                    ),
                    "dense_fallback_permitted": False,
                    "python_object_serialization_permitted": False,
                },
            }
        )
        return _publish_report(config, report)
    except BaseException as exc:
        cleanup_errors: list[str] = []
        rollback = None
        if (
            active_installation is not None
            and active_installation.active
            and model is not None
            and source_state_sha256 is not None
            and source_counts is not None
            and source_topology is not None
        ):
            try:
                rollback = _authenticated_rollback(
                    active_installation,
                    model,
                    source_state_sha256=source_state_sha256,
                    source_counts=source_counts,
                    source_topology_sha256=(
                        source_topology["topology_sha256"]
                    ),
                )
            except BaseException as rollback_exc:
                cleanup_errors.append(
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
        try:
            checkpoint_absence = _checkpoint_absence_audit(config)
        except BaseException as checkpoint_exc:
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
            "Physically factorize all local LFM2 FFN projections at a real "
            "~20% total target, with an optional ~15% fallback."
        )
    )
    parser.add_argument("--source-child-model", type=Path, required=True)
    parser.add_argument("--source-child-cache", type=Path)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument(
        "--whole-output-parent-cache",
        type=Path,
        help=(
            "Optional original-parent cache named by the sealed whole-output "
            "baseline when trajectory evidence uses a separate horizon corpus."
        ),
    )
    parser.add_argument("--cached-parent-outputs", type=Path, required=True)
    parser.add_argument(
        "--cached-parent-outputs-sha256",
        required=True,
    )
    parser.add_argument("--tokenizer-directory", type=Path)
    parser.add_argument("--path-domains-json", type=Path)
    parser.add_argument("--path-domains-sha256")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-destination",
        type=Path,
        required=True,
        help="Reserved path; this experiment never writes a checkpoint.",
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--target-total-unique-parameters",
        type=int,
        default=DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS,
    )
    parser.add_argument(
        "--fallback-target-total-unique-parameters",
        type=int,
        default=DEFAULT_FALLBACK_TARGET_TOTAL_UNIQUE_PARAMETERS,
        help="Set to 0 to disable the less aggressive fallback.",
    )
    parser.add_argument(
        "--expected-original-parent-unique-parameters",
        type=int,
        default=EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS,
    )
    parser.add_argument("--expected-source-child-unique-parameters", type=int)
    parser.add_argument(
        "--expected-whole-output-case-count",
        type=int,
        default=EXPECTED_WHOLE_OUTPUT_CASE_COUNT,
    )
    parser.add_argument(
        "--roles",
        type=lambda value: tuple(
            item.strip() for item in value.split(",") if item.strip()
        ),
        default=("w2",),
        help=(
            "Comma-separated FFN projections. Defaults to w2 only so the "
            "nonlinear w1/w3 producers stay exact."
        ),
    )
    parser.add_argument(
        "--max-fit-rows",
        type=int,
        default=DEFAULT_MAX_FIT_ROWS,
    )
    parser.add_argument(
        "--max-validation-rows",
        type=int,
        default=DEFAULT_MAX_VALIDATION_ROWS,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--minimum-total-compression-fraction",
        type=float,
        default=DEFAULT_MINIMUM_TOTAL_COMPRESSION_FRACTION,
    )
    parser.add_argument(
        "--minimum-parent-residual-cosine",
        type=float,
        default=DEFAULT_MINIMUM_PARENT_RESIDUAL_COSINE,
    )
    parser.add_argument(
        "--minimum-incumbent-residual-cosine",
        type=float,
        default=DEFAULT_MINIMUM_INCUMBENT_RESIDUAL_COSINE,
    )
    parser.add_argument(
        "--minimum-parent-trajectory-score",
        type=float,
        default=DEFAULT_MINIMUM_TRAJECTORY_SCORE,
    )
    parser.add_argument(
        "--minimum-incumbent-trajectory-score",
        type=float,
        default=DEFAULT_MINIMUM_TRAJECTORY_SCORE,
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
        default=DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION,
    )
    parser.add_argument(
        "--whole-output-minimum-task-score-retention",
        type=float,
        default=DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION,
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
) -> ActivationFactorizationRunConfig:
    values = vars(arguments).copy()
    if values["fallback_target_total_unique_parameters"] == 0:
        values["fallback_target_total_unique_parameters"] = None
    return ActivationFactorizationRunConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_experiment(_config_from_namespace(arguments))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == PUBLICATION_BLOCK_STATUS:
        return 0
    return 2 if report["status"] == "rejected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
