#!/usr/bin/env python3
"""Blank-first, dense, full-source structural compression for local LFM2.

This runner is deliberately separate from the packed/quantized experiments.
Its compression claim is only the reduction in unique learned ``Parameter``
elements of a physically smaller, ordinary dense LFM2 architecture.  It never
loads from the Hub, never runs the parent, and never trains.  All folding
evidence comes from an already sealed LFM2 circuit cache.

The first production target is the full-depth h640/Q10/KV5/d64/FF1920 child:
114,067,328 unique logical parameters for the canonical 229,693,184 parameter
LFM2-230M parent (50.339263% structural reduction).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from neural_forge.dynamic_target_synthesis import (
    StructuralBufferPolicy,
    _create_compiler_owned_blank,
    _reject_parent_child_aliases,
    _verify_compiler_owned_blank,
    validate_structural_runtime_buffers,
)
from neural_forge.dynamic_sequence_horizon import (
    assert_model_context_preserved,
    build_sequence_horizon_contract,
)
from neural_forge.hf_activation_cluster import build_gqa_activation_cluster_plan
from neural_forge.hf_attention_fold import build_gqa_attention_fold_maps
from neural_forge.hf_feature_fold import FeatureFoldMap, build_feature_fold_map
from neural_forge.hf_lfm2_cached_causal_evidence import (
    verify_lfm2_cached_causal_frontier_document,
)
from neural_forge.hf_lfm2_canonical import (
    inspect_lfm2_layer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_seed_erasure import (
    SeedCompileOutcome,
    audit_seed_erasure,
)
from neural_forge.hf_transport_math import (
    CoordinateMap,
    build_parent_residual_basis,
)


RUN_SCHEMA = "neural-forge-lfm2-dynamic-structural-compression-v1"
FOLD_SCHEMA = "neural-forge-lfm2-full-source-residual-width-fold-v1"
EXPECTED_PARENT_UNIQUE_PARAMETERS = 229_693_184
FIXED_HEAD_DIM = 64
FIXED_LAYER_COUNT = 14
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}
_FORBIDDEN_DENSE_MARKERS = (
    "packed",
    "quant",
    "crevf",
    "bitlinear",
    "int4",
    "int8",
)
MINIMUM_GLOBAL_RESIDUAL_RETAINED_ENERGY_FRACTION = 0.94
MINIMUM_GLOBAL_OBJECTIVE_RETAINED_ENERGY_FRACTIONS = {
    "residual": 0.95,
    "observable": 0.85,
    "teacher_margin": 0.98,
}
GLOBAL_RESIDUAL_OBJECTIVE_PRIORITIES = {
    "residual": 1.0,
    "observable": 1.0,
    "teacher_margin": 1.0,
}
REFERENCE_H640_DENSE_PCA_RETAINED_ENERGY_UPPER_BOUND = 0.96837
_DENSE_PCA_THREAD_LOCK = threading.RLock()


class Lfm2StructuralCompressionError(RuntimeError):
    """The dense blank-first structural compression contract failed."""


def _build_deterministic_parent_residual_basis(
    calibration: torch.Tensor,
    target_width: int,
) -> CoordinateMap:
    """Run the exact PCA with a process-independent single-thread reduction.

    BLAS/LAPACK thread partitioning can change the last bits of singular
    vectors even when the represented subspace and retained energy are
    identical.  The basis bytes are part of the fold identity, so execute this
    offline analytical step under one Torch CPU thread and restore the caller's
    setting afterward.
    """

    with _DENSE_PCA_THREAD_LOCK:
        previous_threads = torch.get_num_threads()
        try:
            if previous_threads != 1:
                torch.set_num_threads(1)
            return build_parent_residual_basis(
                calibration,
                target_width,
                center=False,
            )
        finally:
            if torch.get_num_threads() != previous_threads:
                torch.set_num_threads(previous_threads)


class ResidualCoordinateFold(Protocol):
    """Common linear-coordinate API used by sparse and dense fold maps."""

    @property
    def source_width(self) -> int: ...

    @property
    def target_width(self) -> int: ...

    def encode(self, source_rows: torch.Tensor) -> torch.Tensor: ...

    def decode(self, target_rows: torch.Tensor) -> torch.Tensor: ...

    def fold_consumer_weight(self, source_weight: torch.Tensor) -> torch.Tensor: ...

    def fold_producer_weight(self, source_weight: torch.Tensor) -> torch.Tensor: ...

    def materialize_dense(
        self,
        *,
        max_elements: int,
        dtype: torch.dtype | None = None,
        device: Any = "cpu",
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class DenseCoordinateFoldMap:
    """Feature-fold-compatible wrapper around one orthonormal dense basis.

    ``coordinates.decode`` is ``B[source, target]``.  Row activations encode as
    ``x @ B``; linear consumers transport as ``W @ B``; and linear producers
    transport as ``B.T @ W``.  Nothing in this representation assigns a source
    channel to only one target channel.
    """

    coordinates: CoordinateMap
    map_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.coordinates, CoordinateMap):
            raise Lfm2StructuralCompressionError(
                "Dense coordinate folding requires a CoordinateMap."
            )
        if not _SHA256.fullmatch(self.map_sha256):
            raise Lfm2StructuralCompressionError(
                "Dense coordinate map SHA-256 is invalid."
            )
        basis = self.coordinates.decode
        if (
            not isinstance(basis, torch.Tensor)
            or tuple(basis.shape) != (self.source_width, self.target_width)
            or not bool(torch.isfinite(basis).all())
        ):
            raise Lfm2StructuralCompressionError(
                "Dense coordinate basis is missing, malformed, or non-finite."
            )

    @property
    def source_width(self) -> int:
        return int(self.coordinates.source_dim)

    @property
    def target_width(self) -> int:
        return int(self.coordinates.target_dim)

    @property
    def basis(self) -> torch.Tensor:
        return self.coordinates.decode

    def _validate_runtime(
        self,
        value: torch.Tensor,
        name: str,
        *,
        final_width: int,
        dimensions: int | None = None,
    ) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or not torch.is_floating_point(value)
            or torch.is_complex(value)
            or value.ndim < 1
            or (dimensions is not None and value.ndim != dimensions)
            or int(value.shape[-1]) != final_width
            or not bool(torch.isfinite(value.detach()).all())
        ):
            raise Lfm2StructuralCompressionError(
                f"{name} must be a finite dense floating tensor with final "
                f"width {final_width}."
            )
        return value

    def _runtime_basis(self, value: torch.Tensor) -> torch.Tensor:
        return self.basis.to(device=value.device, dtype=value.dtype)

    def encode(self, source_rows: torch.Tensor) -> torch.Tensor:
        source = self._validate_runtime(
            source_rows,
            "source_rows",
            final_width=self.source_width,
        )
        result = source @ self._runtime_basis(source)
        if not bool(torch.isfinite(result.detach()).all()):
            raise Lfm2StructuralCompressionError(
                "Dense coordinate encoding produced non-finite values."
            )
        return result

    def apply(self, source_rows: torch.Tensor) -> torch.Tensor:
        return self.encode(source_rows)

    def __call__(self, source_rows: torch.Tensor) -> torch.Tensor:
        return self.encode(source_rows)

    def decode(self, target_rows: torch.Tensor) -> torch.Tensor:
        target = self._validate_runtime(
            target_rows,
            "target_rows",
            final_width=self.target_width,
        )
        result = target @ self._runtime_basis(target).transpose(0, 1)
        if not bool(torch.isfinite(result.detach()).all()):
            raise Lfm2StructuralCompressionError(
                "Dense coordinate decoding produced non-finite values."
            )
        return result

    def pseudoinverse_decode(self, target_rows: torch.Tensor) -> torch.Tensor:
        return self.decode(target_rows)

    def fold_consumer_weight(self, source_weight: torch.Tensor) -> torch.Tensor:
        source = self._validate_runtime(
            source_weight,
            "source_weight",
            final_width=self.source_width,
            dimensions=2,
        )
        return self.encode(source)

    def fold_producer_weight(self, source_weight: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(source_weight, torch.Tensor)
            or source_weight.layout != torch.strided
            or source_weight.device.type == "meta"
            or not torch.is_floating_point(source_weight)
            or torch.is_complex(source_weight)
            or source_weight.ndim != 2
            or int(source_weight.shape[0]) != self.source_width
            or not bool(torch.isfinite(source_weight.detach()).all())
        ):
            raise Lfm2StructuralCompressionError(
                "source_weight must be a finite dense floating matrix whose "
                f"row count is {self.source_width}."
            )
        result = self._runtime_basis(source_weight).transpose(0, 1) @ source_weight
        if not bool(torch.isfinite(result.detach()).all()):
            raise Lfm2StructuralCompressionError(
                "Dense producer transport produced non-finite values."
            )
        return result

    def materialize_dense(
        self,
        *,
        max_elements: int,
        dtype: torch.dtype | None = None,
        device: Any = "cpu",
    ) -> torch.Tensor:
        if (
            isinstance(max_elements, bool)
            or not isinstance(max_elements, int)
            or max_elements < self.source_width * self.target_width
        ):
            raise Lfm2StructuralCompressionError(
                "max_elements is too small for the requested dense basis."
            )
        target_dtype = self.basis.dtype if dtype is None else dtype
        return self.basis.to(device=device, dtype=target_dtype).contiguous()


@dataclass(frozen=True, slots=True)
class StructuralCandidate:
    name: str
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    intermediate_size: int
    verified_unique_parameters: int
    verified_parent_reduction_fraction: float

    def __post_init__(self) -> None:
        values = (
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
            self.intermediate_size,
            self.verified_unique_parameters,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise Lfm2StructuralCompressionError("Candidate dimensions must be positive integers.")
        if self.query_heads % self.key_value_heads:
            raise Lfm2StructuralCompressionError("Candidate Q heads must divide into KV groups.")
        if self.query_heads * self.head_dim != self.hidden_size:
            raise Lfm2StructuralCompressionError(
                "Candidate attention width must equal its residual width."
            )
        expected = 1.0 - self.verified_unique_parameters / EXPECTED_PARENT_UNIQUE_PARAMETERS
        if not math.isclose(
            self.verified_parent_reduction_fraction,
            expected,
            rel_tol=0.0,
            abs_tol=5e-9,
        ):
            raise Lfm2StructuralCompressionError("Candidate verified reduction is inconsistent.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hidden_size": self.hidden_size,
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "verified_unique_parameters": self.verified_unique_parameters,
            "verified_parent_reduction_fraction": self.verified_parent_reduction_fraction,
            "full_depth": FIXED_LAYER_COUNT,
            "dense_logical_parameters_only": True,
        }


STRUCTURAL_CANDIDATES = (
    StructuralCandidate(
        "h640_q10_kv5_ff1920",
        640,
        10,
        5,
        64,
        1920,
        114_067_328,
        0.50339263,
    ),
    StructuralCandidate(
        "h640_q10_kv5_ff1088",
        640,
        10,
        5,
        64,
        1088,
        91_703_168,
        0.60075799,
    ),
    StructuralCandidate(
        "h512_q8_kv4_ff1568",
        512,
        8,
        4,
        64,
        1568,
        80_407_808,
        0.64993385,
    ),
    StructuralCandidate(
        "h512_q8_kv4_ff1024",
        512,
        8,
        4,
        64,
        1024,
        68_709_632,
        0.70086343,
    ),
)
_CANDIDATES_BY_NAME = {candidate.name: candidate for candidate in STRUCTURAL_CANDIDATES}


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2StructuralCompressionError("Value is not strict canonical JSON.") from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().resolve_conj().resolve_neg().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(_canonical_json(list(cpu.shape)))
    flat = cpu.reshape(-1)
    chunk_elements = max(1, (8 * 1024 * 1024) // max(1, cpu.element_size()))
    for start in range(0, flat.numel(), chunk_elements):
        digest.update(flat[start : start + chunk_elements].contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256(b"neural-forge-dense-structural-state-v1\0")
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def differing_state_tensors(left: nn.Module, right: nn.Module) -> tuple[str, ...]:
    """Return exact state mismatches for a failed seed-erasure audit."""

    left_state = left.state_dict()
    right_state = right.state_dict()
    if tuple(left_state) != tuple(right_state):
        return ("<state-key-set>",)
    result = []
    for name in left_state:
        left_value = left_state[name].detach().to(device="cpu")
        right_value = right_state[name].detach().to(device="cpu")
        if (
            tuple(left_value.shape) != tuple(right_value.shape)
            or left_value.dtype != right_value.dtype
            or not torch.equal(left_value, right_value)
        ):
            result.append(name)
    return tuple(result)


def _named_parameters_with_duplicates(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    try:
        values = tuple(model.named_parameters(recurse=True, remove_duplicate=False))
    except TypeError as exc:
        raise Lfm2StructuralCompressionError(
            "PyTorch 2.5+ duplicate-preserving parameter enumeration is required."
        ) from exc
    if not values:
        raise Lfm2StructuralCompressionError("A dense child must own learned Parameters.")
    return values


def _storage_view(value: torch.Tensor) -> tuple[Any, ...]:
    storage = value.untyped_storage()
    return (
        value.device.type,
        value.device.index,
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(value.storage_offset()),
        tuple(int(size) for size in value.shape),
        tuple(int(stride) for stride in value.stride()),
        str(value.dtype),
    )


def dense_logical_parameter_audit(
    model: nn.Module,
    *,
    require_any_nonzero: bool = False,
) -> dict[str, Any]:
    """Count unique dense learned elements and reject packing/quantization lanes."""

    suspicious_modules = []
    for path, module in model.named_modules():
        marker = f"{type(module).__module__}.{type(module).__qualname__}".lower()
        if any(token in marker for token in _FORBIDDEN_DENSE_MARKERS):
            suspicious_modules.append(path or "<root>")
    if suspicious_modules:
        raise Lfm2StructuralCompressionError(
            "Packed/quantized/runtime-codec modules are not dense structural compression: "
            + ", ".join(suspicious_modules)
        )

    unique: dict[tuple[Any, ...], tuple[str, nn.Parameter]] = {}
    object_to_view: dict[int, tuple[Any, ...]] = {}
    aliases: dict[tuple[Any, ...], list[str]] = {}
    any_nonzero = False
    for name, parameter in _named_parameters_with_duplicates(model):
        if not isinstance(parameter, nn.Parameter):
            raise Lfm2StructuralCompressionError(f"{name!r} is not an nn.Parameter.")
        if (
            parameter.layout != torch.strided
            or parameter.device.type == "meta"
            or parameter.dtype not in _ALLOWED_DTYPES
            or parameter.numel() <= 0
        ):
            raise Lfm2StructuralCompressionError(
                f"Dense learned Parameter {name!r} is packed, empty, meta, sparse, or non-FP."
            )
        view = _storage_view(parameter)
        previous = object_to_view.setdefault(id(parameter), view)
        if previous != view:
            raise Lfm2StructuralCompressionError(
                f"Parameter object {name!r} changed its logical storage view."
            )
        unique.setdefault(view, (name, parameter))
        aliases.setdefault(view, []).append(name)
        any_nonzero = any_nonzero or bool(torch.count_nonzero(parameter.detach()).item())
    if require_any_nonzero and not any_nonzero:
        raise Lfm2StructuralCompressionError("Folded dense model is still an all-zero blank.")
    count = sum(int(parameter.numel()) for _name, parameter in unique.values())
    return {
        "schema": "neural-forge-dense-logical-parameter-audit-v1",
        "unique_logical_parameters": count,
        "unique_parameter_views": len(unique),
        "logical_parameter_names": len(_named_parameters_with_duplicates(model)),
        "alias_groups": [
            sorted(names) for names in aliases.values() if len(names) > 1
        ],
        "all_parameters_dense_strided_fp": True,
        "packed_parameters": 0,
        "quantized_parameters": 0,
        "dtype_or_serialized_bytes_used_for_compression_claim": False,
    }


def parameter_topology_snapshot(model: nn.Module) -> dict[str, tuple[Any, ...]]:
    return {
        name: (
            id(parameter),
            tuple(int(size) for size in parameter.shape),
            str(parameter.dtype),
            str(parameter.device),
            _storage_view(parameter),
        )
        for name, parameter in _named_parameters_with_duplicates(model)
    }


def _config_payload(config: Any) -> dict[str, Any]:
    reporter = getattr(config, "to_dict", None)
    value = reporter() if callable(reporter) else vars(config)
    if not isinstance(value, Mapping):
        raise Lfm2StructuralCompressionError("LFM2 config must serialize to a mapping.")
    return dict(value)


def build_structural_lfm2_config(parent_config: Any, candidate: StructuralCandidate) -> Any:
    """Clone every LFM2 width alias consistently for a prebuilt target."""

    source = _config_payload(parent_config)
    if source.get("model_type") != "lfm2":
        raise Lfm2StructuralCompressionError("Parent config.model_type must be lfm2.")
    layer_types = tuple(source.get("layer_types", ()))
    if len(layer_types) != FIXED_LAYER_COUNT:
        raise Lfm2StructuralCompressionError("Structural lane keeps all fourteen layers.")
    result = copy.deepcopy(parent_config)
    assignments = {
        "hidden_size": candidate.hidden_size,
        "block_dim": candidate.hidden_size,
        "conv_dim": candidate.hidden_size,
        "num_attention_heads": candidate.query_heads,
        "num_heads": candidate.query_heads,
        "num_key_value_heads": candidate.key_value_heads,
        "head_dim": candidate.head_dim,
        "intermediate_size": candidate.intermediate_size,
        "block_ff_dim": candidate.intermediate_size,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": FIXED_LAYER_COUNT,
    }
    for name, value in assignments.items():
        setattr(result, name, value)
    if hasattr(result, "num_layers"):
        result.num_layers = FIXED_LAYER_COUNT
    result.layer_types = list(layer_types)
    result._name_or_path = ""
    if hasattr(result, "auto_map"):
        result.auto_map = None
    payload = _config_payload(result)
    for name, expected in assignments.items():
        if payload.get(name) != expected:
            raise Lfm2StructuralCompressionError(
                f"Serialized target config lost alias {name!r}: {payload.get(name)!r}."
            )
    if tuple(payload.get("layer_types", ())) != layer_types:
        raise Lfm2StructuralCompressionError("Target changed ordered layer/operator types.")
    try:
        assert_model_context_preserved(parent_config, result)
    except Exception as exc:
        raise Lfm2StructuralCompressionError(
            "Structural target changed the parent's context capacity or "
            "positional semantics."
        ) from exc
    return result


def _lfm2_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if layers is None:
        raise Lfm2StructuralCompressionError("Expected model.model.layers.")
    result = tuple(layers)
    if len(result) != FIXED_LAYER_COUNT:
        raise Lfm2StructuralCompressionError("Target must retain all fourteen layers.")
    return result


def _lfm2_final_norm(model: nn.Module) -> nn.Module:
    body = getattr(model, "model", None)
    if body is None:
        raise Lfm2StructuralCompressionError("Expected an LFM2 model body.")
    norm = getattr(body, "embedding_norm", None)
    if norm is None:
        # Compatibility with the small historical fixtures; the concrete
        # Transformers LFM2 runtime names this module ``embedding_norm``.
        norm = getattr(body, "norm", None)
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, nn.Parameter):
        raise Lfm2StructuralCompressionError(
            "Could not discover the concrete LFM2 final embedding norm."
        )
    return norm


def validate_live_structural_topology(
    parent: nn.Module,
    child: nn.Module,
    candidate: StructuralCandidate,
) -> dict[str, Any]:
    parent_layers = _lfm2_layers(parent)
    child_layers = _lfm2_layers(child)
    parent_kinds = []
    child_kinds = []
    for index, (source, target) in enumerate(zip(parent_layers, child_layers, strict=True)):
        source_contract = inspect_lfm2_layer(source)
        target_contract = inspect_lfm2_layer(target)
        parent_kinds.append(source_contract.kind)
        child_kinds.append(target_contract.kind)
        if target_contract.hidden_size != candidate.hidden_size:
            raise Lfm2StructuralCompressionError(f"Layer {index} target hidden width drifted.")
        if target_contract.ffn_intermediate_size != candidate.intermediate_size:
            raise Lfm2StructuralCompressionError(f"Layer {index} target FFN width drifted.")
        if target_contract.kind == "full_attention":
            operator = target_contract.operator
            observed = (
                operator.query_heads,
                operator.key_value_heads,
                operator.head_dim,
            )
            expected = (
                candidate.query_heads,
                candidate.key_value_heads,
                candidate.head_dim,
            )
            if observed != expected:
                raise Lfm2StructuralCompressionError(
                    f"Layer {index} live attention geometry {observed} != {expected}."
                )
    if parent_kinds != child_kinds:
        raise Lfm2StructuralCompressionError("Ordered operator types changed.")
    return {
        "layer_count": FIXED_LAYER_COUNT,
        "ordered_operator_types": child_kinds,
        "live_projection_shapes_verified": True,
        "config_aliases_verified": True,
    }


def _structural_buffer_policy(model: nn.Module) -> StructuralBufferPolicy:
    entries = []
    for name, value in model.named_buffers(recurse=True, remove_duplicate=False):
        if not torch.count_nonzero(value.detach()).item():
            continue
        lowered = name.lower()
        if (
            lowered.endswith(".inv_freq")
            or lowered.endswith(".original_inv_freq")
            or ("rotary" in lowered and lowered.endswith((".cos_cached", ".sin_cached")))
        ):
            entries.append((name, "rotary_frequency_geometry"))
        else:
            raise Lfm2StructuralCompressionError(
                f"Nonzero runtime buffer {name!r} is not an approved LFM2 structural buffer."
            )
    return StructuralBufferPolicy(
        policy_id="compiler.lfm2.dynamic-structural.rotary-geometry",
        version=1,
        adapter_id="neural_forge.lfm2.dynamic_structural",
        adapter_version=1,
        entries=tuple(entries),
        source="compiler-reviewed-runner-policy",
        compiler_reviewed=True,
    )


@dataclass(slots=True)
class BlankTarget:
    model: nn.Module
    seed: int
    random_state_sha256: str
    blank_state_sha256: str
    provenance: Mapping[str, Any]
    topology_before_fold: Mapping[str, tuple[Any, ...]]
    exact_unique_parameters: int


def instantiate_compiler_blank(
    parent: nn.Module,
    candidate: StructuralCandidate,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BlankTarget:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Lfm2StructuralCompressionError("Blank seed must be non-negative.")
    config = build_structural_lfm2_config(parent.config, candidate)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        child = type(parent)(copy.deepcopy(config))
    child.to(device=device, dtype=dtype).eval()
    tie = getattr(child, "tie_weights", None)
    if callable(tie):
        tie()
    _reject_parent_child_aliases(child, {"parent": parent})
    random_hash = model_state_sha256(child)
    preparation = _create_compiler_owned_blank(child)
    _verify_compiler_owned_blank(child, preparation)
    blank_hash = model_state_sha256(child)
    policy = _structural_buffer_policy(child)
    buffer_audit = validate_structural_runtime_buffers(
        child,
        policy,
        expected_adapter_id=policy.adapter_id,
        expected_adapter_version=policy.adapter_version,
    )
    provenance = preparation.provenance(blank_hash, policy, buffer_audit)
    dense = dense_logical_parameter_audit(child)
    if dense["unique_logical_parameters"] != candidate.verified_unique_parameters:
        raise Lfm2StructuralCompressionError(
            "Live target unique count differs from the verified dense candidate: "
            f"{dense['unique_logical_parameters']} != {candidate.verified_unique_parameters}."
        )
    return BlankTarget(
        model=child,
        seed=seed,
        random_state_sha256=random_hash,
        blank_state_sha256=blank_hash,
        provenance=provenance.as_dict(),
        topology_before_fold=parameter_topology_snapshot(child),
        exact_unique_parameters=dense["unique_logical_parameters"],
    )


def _phase_key(value: str) -> tuple[int, int]:
    if value == "prefill":
        return (0, 0)
    if value.startswith("decode."):
        return (1, int(value.split(".")[1]))
    return (2, 0)


def _row_major(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise Lfm2StructuralCompressionError("Circuit trace tensor must have a feature axis.")
    return value.detach().to(device="cpu", dtype=torch.float32).reshape(-1, int(value.shape[-1])).contiguous()


def _field_rows(field: str, value: torch.Tensor) -> torch.Tensor:
    """Keep one complete Q/K/V/context head bundle per causal token row."""

    if field in {
        "attention.q",
        "attention.k",
        "attention.v",
        "attention.context",
        "attention.q_linear",
        "attention.k_linear",
    }:
        if value.ndim != 4:
            raise Lfm2StructuralCompressionError(
                f"Cached {field} must have shape [batch, heads, sequence, head_dim]."
            )
        ordered = value.detach().to(device="cpu", dtype=torch.float32).transpose(1, 2)
        return ordered.reshape(
            int(ordered.shape[0]) * int(ordered.shape[1]),
            int(ordered.shape[2]) * int(ordered.shape[3]),
        ).contiguous()
    return _row_major(value)


def _bounded_rows(value: torch.Tensor, maximum: int) -> torch.Tensor:
    if value.shape[0] <= maximum:
        return value
    index = torch.arange(maximum, dtype=torch.long) * int(value.shape[0]) // maximum
    return value.index_select(0, index).contiguous()


@dataclass(slots=True)
class CachedFoldEvidence:
    residual_rows: torch.Tensor
    observable_rows: torch.Tensor
    margin_rows: torch.Tensor
    margin_weights: torch.Tensor
    layer_fields: tuple[dict[str, torch.Tensor], ...]
    cache_key: str
    corpus_sha256: str
    evidence_sha256: str


def extract_cached_fold_evidence(
    cache: Any,
    parent: nn.Module,
    *,
    max_rows: int = 4096,
    max_attention_rows: int = 512,
    max_vocab_rows: int = 2048,
) -> CachedFoldEvidence:
    """Read the sealed cache only; this function never invokes the parent."""

    layers = _lfm2_layers(parent)
    pieces: list[dict[str, list[torch.Tensor]]] = [dict() for _ in layers]
    residual_pieces: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    for batch in cache.iter_batches():
        tensors = batch.tensors
        phases = sorted(
            {
                name.split(".layer.", 1)[0]
                for name in tensors
                if ".layer." in name and not name.startswith("cached_prefill.")
            },
            key=_phase_key,
        )
        for phase in phases:
            for layer_index in range(len(layers)):
                prefix = f"{phase}.layer.{layer_index:03d}."
                found = {
                    name[len(prefix) :]: value
                    for name, value in tensors.items()
                    if name.startswith(prefix)
                }
                if not found:
                    raise Lfm2StructuralCompressionError(
                        f"Cache phase {phase} lacks layer {layer_index}."
                    )
                for field, value in found.items():
                    if not torch.is_floating_point(value) or value.ndim < 2:
                        continue
                    rows = _field_rows(field, value)
                    pieces[layer_index].setdefault(field, []).append(rows)
                for field in ("residual.pre", "residual.post_operator", "residual.post_layer"):
                    residual_pieces.append(_row_major(found[field]))
            logit_name = f"{phase}.logits"
            if logit_name in tensors:
                logits.append(_row_major(tensors[logit_name]))
    if not residual_pieces or not logits:
        raise Lfm2StructuralCompressionError("Sealed cache lacks residual/logit evidence.")
    residual = _bounded_rows(torch.cat(residual_pieces, dim=0), max_rows)
    joined: list[dict[str, torch.Tensor]] = []
    for fields in pieces:
        joined.append(
            {
                name: _bounded_rows(
                    torch.cat(values, dim=0),
                    max_attention_rows if name.startswith("attention.") else max_rows,
                )
                for name, values in fields.items()
            }
        )
    teacher_logits = _bounded_rows(torch.cat(logits, dim=0), max_vocab_rows)
    if teacher_logits.shape[1] < 2:
        raise Lfm2StructuralCompressionError("Teacher logits need at least two vocabulary rows.")
    top = torch.topk(teacher_logits, k=2, dim=1)
    embedding = parent.get_input_embeddings().weight.detach().to(device="cpu", dtype=torch.float32)
    winners = top.indices[:, 0]
    competitors = top.indices[:, 1]
    margins = embedding.index_select(0, winners) - embedding.index_select(0, competitors)
    margin_weights = (top.values[:, 0] - top.values[:, 1]).abs().clamp_min(1e-6)
    vocab = int(embedding.shape[0])
    sample_count = min(max_vocab_rows, vocab)
    sampled_ids = torch.arange(sample_count, dtype=torch.long) * vocab // sample_count
    observable = torch.cat(
        (
            embedding.index_select(0, sampled_ids),
            embedding.index_select(0, torch.unique(torch.cat((winners, competitors)))),
        ),
        dim=0,
    ).contiguous()
    descriptor = {
        "cache_key": cache.cache_key,
        "corpus_sha256": cache.corpus_fingerprint,
        "residual": _tensor_sha256(residual),
        "observable": _tensor_sha256(observable),
        "margins": _tensor_sha256(margins),
        "layers": [
            {name: _tensor_sha256(value) for name, value in sorted(fields.items())}
            for fields in joined
        ],
    }
    return CachedFoldEvidence(
        residual,
        observable,
        margins,
        margin_weights,
        tuple(joined),
        cache.cache_key,
        cache.corpus_fingerprint,
        _json_sha256(descriptor),
    )


def load_global_causal_guidance(
    path: Path,
    *,
    expected_sha256: str,
    parent_identity_sha256: str,
    cache_key: str,
    corpus_sha256: str,
) -> dict[str, Any]:
    if not _SHA256.fullmatch(expected_sha256):
        raise Lfm2StructuralCompressionError("Expected global frontier SHA-256 is invalid.")
    if _file_sha256(path) != expected_sha256:
        raise Lfm2StructuralCompressionError("Global causal frontier file SHA-256 differs.")
    document = json.loads(path.read_text(encoding="ascii"))
    verify_lfm2_cached_causal_frontier_document(document)
    binding = document["parent_binding"]
    if (
        binding["parent_identity_sha256"] != parent_identity_sha256
        or binding["parent_trace_cache_key"] != cache_key
        or binding["parent_corpus_sha256"] != corpus_sha256
    ):
        raise Lfm2StructuralCompressionError(
            "Global causal frontier is not bound to this parent/cache."
        )
    selected_ids = set(document["targets"]["0.35"]["selected_option_ids"])
    by_layer: dict[int, tuple[int, ...]] = {}
    for candidate in document["candidates"]:
        if candidate["option_id"] in selected_ids:
            by_layer[int(candidate["layer_index"])] = tuple(
                int(value) for value in candidate["retained_source_units"]
            )
    return {
        "frontier_sha256": document["frontier_sha256"],
        "file_sha256": expected_sha256,
        "ffn_only_35_plan_sha256": document["targets"]["0.35"]["plan_sha256"],
        "ffn_only_50_status": document["targets"]["0.50"]["status"],
        "causal_anchor_neurons_by_layer": by_layer,
        "exact_token_match_objective_used": False,
    }


def build_global_residual_map(
    evidence: CachedFoldEvidence,
    candidate: StructuralCandidate,
    *,
    minimum_retained_energy_fraction: float = (
        MINIMUM_GLOBAL_RESIDUAL_RETAINED_ENERGY_FRACTION
    ),
    minimum_objective_retained_energy_fractions: Mapping[str, float] = (
        MINIMUM_GLOBAL_OBJECTIVE_RETAINED_ENERGY_FRACTIONS
    ),
) -> tuple[DenseCoordinateFoldMap, dict[str, Any]]:
    """Build the optimal dense residual subspace from all cached evidence."""

    objective_names = ("residual", "observable", "teacher_margin")
    objective_matrices = {
        "residual": evidence.residual_rows,
        "observable": evidence.observable_rows,
        "teacher_margin": (
            evidence.margin_rows
            * evidence.margin_weights.sqrt().unsqueeze(1)
        ),
    }
    source_width = int(evidence.residual_rows.shape[1])
    if (
        isinstance(minimum_retained_energy_fraction, bool)
        or not isinstance(minimum_retained_energy_fraction, (int, float))
        or not math.isfinite(float(minimum_retained_energy_fraction))
        or not 0.0 <= float(minimum_retained_energy_fraction) <= 1.0
    ):
        raise Lfm2StructuralCompressionError(
            "minimum_retained_energy_fraction must be within [0, 1]."
        )
    if (
        not isinstance(minimum_objective_retained_energy_fractions, Mapping)
        or set(minimum_objective_retained_energy_fractions) != set(objective_names)
    ):
        raise Lfm2StructuralCompressionError(
            "Every global residual objective needs one retained-energy gate."
        )
    objective_gates: dict[str, float] = {}
    for name in objective_names:
        value = minimum_objective_retained_energy_fractions[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise Lfm2StructuralCompressionError(
                f"Retained-energy gate for {name!r} must be within [0, 1]."
            )
        objective_gates[name] = float(value)
    normalized_objectives: dict[str, torch.Tensor] = {}
    raw_objective_energies: dict[str, float] = {}
    for index, name in enumerate(objective_names):
        matrix = objective_matrices[name]
        if (
            not isinstance(matrix, torch.Tensor)
            or matrix.ndim != 2
            or int(matrix.shape[1]) != source_width
            or not torch.is_floating_point(matrix)
            or not bool(torch.isfinite(matrix).all())
        ):
            raise Lfm2StructuralCompressionError(
                f"Global residual calibration matrix {index} is malformed."
            )
        normalized = matrix.detach().to(
            device="cpu",
            dtype=torch.float64,
        ).contiguous()
        energy = float(normalized.square().sum().item())
        if not math.isfinite(energy) or energy <= 0.0:
            raise Lfm2StructuralCompressionError(
                f"Global residual objective {name!r} has no finite positive energy."
            )
        normalized_objectives[name] = normalized
        raw_objective_energies[name] = energy

    # Equalize total Frobenius energy before applying explicit semantic role
    # priorities.  Choosing residual energy as the numerical reference changes
    # no subspace: it is one common scalar after per-objective normalization.
    reference_energy = raw_objective_energies["residual"]
    objective_balance_scales = {
        name: math.sqrt(reference_energy / raw_objective_energies[name])
        for name in objective_names
    }
    effective_objective_scales = {
        name: (
            objective_balance_scales[name]
            * GLOBAL_RESIDUAL_OBJECTIVE_PRIORITIES[name]
        )
        for name in objective_names
    }
    weighted_objectives = {
        name: (
            normalized_objectives[name] * effective_objective_scales[name]
        ).contiguous()
        for name in objective_names
    }
    calibration = torch.cat(
        tuple(weighted_objectives[name] for name in objective_names),
        dim=0,
    ).contiguous()
    coordinates = _build_deterministic_parent_residual_basis(
        calibration,
        candidate.hidden_size,
    )
    singular_energy = coordinates.singular_values.square()
    weighted_input_energy = float(singular_energy.sum().item())
    weighted_projected_energy = float(
        singular_energy[: candidate.hidden_size].sum().item()
    )
    if weighted_input_energy <= 0.0 or not all(
        math.isfinite(value)
        for value in (weighted_input_energy, weighted_projected_energy)
    ):
        raise Lfm2StructuralCompressionError(
            "Global residual calibration energy is not finite and positive."
        )
    weighted_retained = weighted_projected_energy / weighted_input_energy
    weighted_reconstruction_error = math.sqrt(
        max(0.0, 1.0 - weighted_retained)
    )
    if not math.isclose(
        weighted_retained,
        coordinates.explained_variance_fraction,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise Lfm2StructuralCompressionError(
            "Coordinate-map and exact singular-energy retention disagree."
        )
    if weighted_retained + 1e-12 < float(minimum_retained_energy_fraction):
        raise Lfm2StructuralCompressionError(
            "Dense global residual PCA retained only "
            f"{weighted_retained:.12f} of exact weighted calibration energy, "
            "below required "
            f"{float(minimum_retained_energy_fraction):.12f}."
        )
    basis = coordinates.decode
    objective_reports: dict[str, dict[str, Any]] = {}
    for name in objective_names:
        rows = normalized_objectives[name]
        input_energy = raw_objective_energies[name]
        projected_energy = float((rows @ basis).square().sum().item())
        retained = projected_energy / input_energy
        reconstruction_error = math.sqrt(max(0.0, 1.0 - retained))
        gate = objective_gates[name]
        if retained + 1e-12 < gate:
            raise Lfm2StructuralCompressionError(
                f"Dense global residual PCA retained only {retained:.12f} "
                f"of raw {name} energy, below required {gate:.12f}."
            )
        balanced_energy = float(
            weighted_objectives[name].square().sum().item()
        )
        objective_reports[name] = {
            "rows": int(rows.shape[0]),
            "raw_input_energy": input_energy,
            "raw_projected_energy": projected_energy,
            "raw_retained_energy_fraction": retained,
            "raw_relative_reconstruction_error": reconstruction_error,
            "minimum_raw_retained_energy_fraction": gate,
            "raw_retained_energy_gate_passed": True,
            "frobenius_balance_scale": objective_balance_scales[name],
            "role_priority": GLOBAL_RESIDUAL_OBJECTIVE_PRIORITIES[name],
            "effective_scale": effective_objective_scales[name],
            "weighted_input_energy": balanced_energy,
            "teacher_margin_row_weighting_preserved": (
                name == "teacher_margin"
            ),
        }
    raw_combined_input_energy = sum(raw_objective_energies.values())
    raw_combined_projected_energy = sum(
        value["raw_projected_energy"] for value in objective_reports.values()
    )
    raw_combined_retained = (
        raw_combined_projected_energy / raw_combined_input_energy
    )
    row_loadings = basis.square().sum(dim=1)
    nonzero_source_axes = int((row_loadings > 0.0).sum().item())
    if nonzero_source_axes != source_width:
        raise Lfm2StructuralCompressionError(
            "Dense residual basis has a source axis with no target loading."
        )
    basis_sha256 = _tensor_sha256(basis)
    map_sha256 = _json_sha256(
        {
            "schema": "neural-forge-dense-orthonormal-coordinate-map-v1",
            "basis_sha256": basis_sha256,
            "source_width": source_width,
            "target_width": candidate.hidden_size,
            "centered_for_fit": False,
            "weighted_exact_retained_energy_fraction": weighted_retained,
            "objective_balance_scales": objective_balance_scales,
            "objective_priorities": GLOBAL_RESIDUAL_OBJECTIVE_PRIORITIES,
            "calibration_sha256": _tensor_sha256(calibration),
        }
    )
    mapping = DenseCoordinateFoldMap(coordinates, map_sha256)
    return mapping, {
        "kind": "reachable-observable-full-source-dense-orthonormal-residual-basis",
        "algorithm": "uncentered-exact-float64-pca-svd",
        "map_sha256": map_sha256,
        "basis_sha256": basis_sha256,
        "calibration_sha256": _tensor_sha256(calibration),
        "source_width": mapping.source_width,
        "target_width": mapping.target_width,
        "centered_for_fit": False,
        "dense_basis": True,
        "sparse_assignment_used": False,
        "source_coverage_fraction": 1.0,
        "source_axes_with_nonzero_loading": nonzero_source_axes,
        "dropped_source_units": 0,
        "conv_channel_locality_preserved": False,
        "locality_assumption_used": False,
        "reachable_rows": int(evidence.residual_rows.shape[0]),
        "observable_rows": int(evidence.observable_rows.shape[0]),
        "teacher_margin_rows": int(evidence.margin_rows.shape[0]),
        "margin_rows_weighted_by_sqrt_teacher_margin": True,
        "calibration_rows": int(calibration.shape[0]),
        "effective_rank": coordinates.effective_rank,
        "rank_tolerance": coordinates.rank_tolerance,
        "objective_weighting": {
            "method": "frobenius-energy-balance-then-role-priority",
            "reference_objective": "residual",
            "objectives": objective_reports,
        },
        "weighted_combined_exact_input_energy": weighted_input_energy,
        "weighted_combined_exact_projected_energy": weighted_projected_energy,
        "weighted_combined_exact_retained_energy_fraction": weighted_retained,
        "weighted_combined_exact_relative_reconstruction_error": (
            weighted_reconstruction_error
        ),
        "minimum_weighted_combined_retained_energy_fraction": float(
            minimum_retained_energy_fraction
        ),
        "weighted_combined_retained_energy_gate_passed": True,
        "raw_unweighted_combined_diagnostic": {
            "input_energy": raw_combined_input_energy,
            "projected_energy": raw_combined_projected_energy,
            "retained_energy_fraction": raw_combined_retained,
            "used_for_acceptance_gate": False,
            "warning": (
                "Raw combined retention can hide a weak objective and is not "
                "an acceptance claim."
            ),
        },
        "pca_is_optimal_rank_target_energy_upper_bound": True,
        "reference_real_h640_upper_bound": (
            REFERENCE_H640_DENSE_PCA_RETAINED_ENERGY_UPPER_BOUND
        ),
    }


def _map_vector_producer(
    mapping: ResidualCoordinateFold,
    value: torch.Tensor,
) -> torch.Tensor:
    return mapping.fold_producer_weight(
        value.detach().to(device="cpu", dtype=torch.float32).reshape(-1, 1)
    ).reshape(-1)


def _transport_square(
    mapping: ResidualCoordinateFold,
    weight: torch.Tensor,
) -> torch.Tensor:
    # CUDA index_add uses atomic accumulation and is not guaranteed to be
    # byte-deterministic.  Folding is an offline analytical compile, so keep
    # the reduction on CPU and copy only the completed target tensor to the
    # already-instantiated device Parameter.
    source = weight.detach().to(device="cpu", dtype=torch.float32)
    return mapping.fold_producer_weight(mapping.fold_consumer_weight(source))


def _materialize_coordinate_basis(
    mapping: ResidualCoordinateFold,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return mapping.materialize_dense(
        max_elements=mapping.source_width * mapping.target_width,
        dtype=dtype,
        device="cpu",
    )


def _transport_diagonal(
    mapping: ResidualCoordinateFold,
    value: torch.Tensor,
) -> torch.Tensor:
    """Return ``diag(B.T @ diag(value) @ B)`` without a locality premise."""

    basis = _materialize_coordinate_basis(mapping, dtype=torch.float32)
    source = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if int(source.numel()) != mapping.source_width:
        raise Lfm2StructuralCompressionError(
            "Diagonal source width differs from the global coordinate map."
        )
    return basis.square().transpose(0, 1) @ source


def _transport_depthwise_kernel(
    mapping: ResidualCoordinateFold,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Transport every depthwise FIR tap to the diagonal target operator."""

    source = kernel.detach().to(device="cpu", dtype=torch.float32)
    if source.ndim != 2 or int(source.shape[0]) != mapping.source_width:
        raise Lfm2StructuralCompressionError(
            "Depthwise FIR kernel must have shape [source_width, taps]."
        )
    basis = _materialize_coordinate_basis(mapping, dtype=torch.float32)
    result = basis.square().transpose(0, 1) @ source
    if not bool(torch.isfinite(result).all()):
        raise Lfm2StructuralCompressionError(
            "Depthwise FIR coordinate transport produced non-finite values."
        )
    return result.contiguous()


def _copy_parameter(
    target: nn.Parameter,
    value: torch.Tensor,
    name: str,
    written: set[int],
) -> None:
    if tuple(value.shape) != tuple(target.shape):
        raise Lfm2StructuralCompressionError(
            f"Folded shape mismatch for {name}: {tuple(value.shape)} != {tuple(target.shape)}."
        )
    if not torch.isfinite(value.detach()).all():
        raise Lfm2StructuralCompressionError(f"Folded value {name!r} is non-finite.")
    target.copy_(value.to(device=target.device, dtype=target.dtype))
    target.grad = None
    written.add(id(target))


def _attention_map_from_cluster(
    rows: torch.Tensor,
    head_assignment: Sequence[int],
    target_heads: int,
    head_dim: int,
) -> FeatureFoldMap:
    source_heads = len(head_assignment)
    shaped = rows.reshape(-1, source_heads, head_dim).to(dtype=torch.float64)
    energy = shaped.square().sum(dim=0).sqrt().clamp_min(1e-30)
    assignment = torch.empty(source_heads * head_dim, dtype=torch.long)
    coefficient = torch.empty(source_heads * head_dim, dtype=torch.float64)
    for source_head, target_head in enumerate(head_assignment):
        for channel in range(head_dim):
            index = source_head * head_dim + channel
            assignment[index] = int(target_head) * head_dim + channel
            coefficient[index] = energy[source_head, channel]
    return FeatureFoldMap(assignment, coefficient, target_heads * head_dim)


def build_attention_maps(
    fields: Mapping[str, torch.Tensor],
    *,
    source_query_heads: int,
    source_key_value_heads: int,
    target_query_heads: int,
    target_key_value_heads: int,
    head_dim: int,
) -> tuple[Any, dict[str, Any]]:
    topology_guard = build_gqa_attention_fold_maps(
        source_query_heads=source_query_heads,
        source_key_value_heads=source_key_value_heads,
        source_head_dim=head_dim,
        target_query_heads=target_query_heads,
        target_key_value_heads=target_key_value_heads,
        target_head_dim=head_dim,
    )
    query = fields["attention.q"].reshape(-1, source_query_heads, head_dim)
    key = fields["attention.k"].reshape(-1, source_key_value_heads, head_dim)
    value = fields["attention.v"].reshape(-1, source_key_value_heads, head_dim)
    context = fields["attention.context"].reshape(-1, source_query_heads, head_dim)
    cluster = build_gqa_activation_cluster_plan(
        query.permute(1, 0, 2),
        key.permute(1, 0, 2),
        value.permute(1, 0, 2),
        context.permute(1, 0, 2),
        context.permute(1, 0, 2),
        target_query_heads=target_query_heads,
        target_key_value_heads=target_key_value_heads,
        head_dim=head_dim,
        source_query_heads=source_query_heads,
        source_key_value_heads=source_key_value_heads,
    )
    maps = {
        "query": _attention_map_from_cluster(
            query, cluster.query_head_assignment, target_query_heads, head_dim
        ),
        "key": _attention_map_from_cluster(
            key, cluster.group_assignment, target_key_value_heads, head_dim
        ),
        "value": _attention_map_from_cluster(
            value, cluster.group_assignment, target_key_value_heads, head_dim
        ),
        "context": _attention_map_from_cluster(
            context, cluster.query_head_assignment, target_query_heads, head_dim
        ),
    }
    for mapping in maps.values():
        if sum(mapping.group_sizes) != mapping.source_width:
            raise Lfm2StructuralCompressionError("Attention map dropped source channels.")
    return maps, {
        "cluster": cluster.report.as_dict(),
        "topology_guard": {
            role: getattr(topology_guard, role).report.as_dict()
            for role in ("query", "key", "value", "context")
        },
        "actual_transport": "activation-cluster-assignment-all-source-weighted-fold",
        "source_coverage_fraction": 1.0,
        "dropped_source_heads": 0,
        "gqa_grouping_preserved": True,
        "rope_pairs_preserved": True,
    }


def _causal_weighted_mlp_fields(
    fields: Mapping[str, torch.Tensor],
    anchors: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    values = [
        fields["mlp.gate"].clone(),
        fields["mlp.up"].clone(),
        fields["mlp.post_swiglu"].clone(),
    ]
    if anchors:
        width = int(values[0].shape[1])
        index = torch.tensor(tuple(anchors), dtype=torch.long)
        if bool((index < 0).any()) or bool((index >= width).any()):
            raise Lfm2StructuralCompressionError("Global causal FFN anchors are out of range.")
        for value in values:
            value[:, index] *= 2.0
    return tuple(values)


def resolve_global_residual_map(
    evidence: CachedFoldEvidence,
    candidate: StructuralCandidate,
    *,
    global_map_override: ResidualCoordinateFold | None = None,
    global_map_report_override: Mapping[str, Any] | None = None,
) -> tuple[ResidualCoordinateFold, dict[str, Any]]:
    """Select the ordinary cached basis or an explicit whole-weight basis.

    The override is used by the child-closed-loop lane so the initial blank,
    vocabulary interface, every residual-facing operator, and later causal
    refinement all share exactly one prompt-independent coordinate system.
    """

    if global_map_override is None:
        if global_map_report_override is not None:
            raise Lfm2StructuralCompressionError(
                "A residual-map report override requires a map override."
            )
        return build_global_residual_map(evidence, candidate)
    if not isinstance(global_map_report_override, Mapping):
        raise Lfm2StructuralCompressionError(
            "A residual-map override requires its complete audit report."
        )
    mapping = global_map_override
    try:
        source_width = int(mapping.source_width)
        target_width = int(mapping.target_width)
        map_sha256 = str(mapping.map_sha256)
        basis = mapping.materialize_dense(
            max_elements=source_width * target_width,
            dtype=torch.float64,
            device="cpu",
        )
    except Exception as exc:
        raise Lfm2StructuralCompressionError(
            "Residual-map override does not implement the dense coordinate "
            "fold protocol."
        ) from exc
    if (
        source_width != int(evidence.residual_rows.shape[1])
        or target_width != candidate.hidden_size
        or not _SHA256.fullmatch(map_sha256)
        or tuple(basis.shape) != (source_width, target_width)
        or not bool(torch.isfinite(basis).all())
        or not torch.allclose(
            basis.T @ basis,
            torch.eye(target_width, dtype=basis.dtype),
            rtol=1e-8,
            atol=1e-10,
        )
        or int((basis.square().sum(dim=1) > 0.0).sum().item())
        != source_width
    ):
        raise Lfm2StructuralCompressionError(
            "Residual-map override failed width, hash, orthonormality, or "
            "full-source-axis coverage validation."
        )
    report = copy.deepcopy(dict(global_map_report_override))
    report.update(
        {
            "map_sha256": map_sha256,
            "basis_sha256": map_sha256,
            "source_width": source_width,
            "target_width": target_width,
            "dense_basis": True,
            "sparse_assignment_used": False,
            "source_coverage_fraction": 1.0,
            "source_axes_with_nonzero_loading": source_width,
            "dropped_source_units": 0,
            "prompt_independent_weight_basis_override": True,
            "activation_evidence_is_optional_bounded_supplement": True,
        }
    )
    return mapping, report


def fold_lfm2_into_blank(
    parent: nn.Module,
    blank: BlankTarget,
    evidence: CachedFoldEvidence,
    candidate: StructuralCandidate,
    *,
    global_guidance: Mapping[str, Any],
    global_map_override: ResidualCoordinateFold | None = None,
    global_map_report_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Overwrite a prebuilt blank in place using full-source cached folds."""

    child = blank.model
    parent_before = fingerprint_lfm2_parent(parent)
    global_map, residual_report = resolve_global_residual_map(
        evidence,
        candidate,
        global_map_override=global_map_override,
        global_map_report_override=global_map_report_override,
    )
    source_layers = _lfm2_layers(parent)
    target_layers = _lfm2_layers(child)
    source_q = int(parent.config.num_attention_heads)
    source_kv = int(parent.config.num_key_value_heads)
    source_head_dim_value = getattr(parent.config, "head_dim", None)
    source_head_dim = (
        int(source_layers[next(
            index
            for index, layer in enumerate(source_layers)
            if getattr(layer, "is_attention_layer", False)
        )].self_attn.head_dim)
        if source_head_dim_value is None
        else int(source_head_dim_value)
    )
    anchors_by_layer = global_guidance.get("causal_anchor_neurons_by_layer", {})
    written: set[int] = set()
    layer_reports = []

    with torch.no_grad():
        source_embedding = parent.get_input_embeddings().weight
        target_embedding = child.get_input_embeddings().weight
        _copy_parameter(
            target_embedding,
            global_map.fold_consumer_weight(
                source_embedding.detach().to(device="cpu", dtype=torch.float32)
            ),
            "model.embed_tokens.weight",
            written,
        )
        # Tied LFM2 uses the same Parameter object at both vocabulary interfaces.
        target_head = child.get_output_embeddings().weight
        if id(target_head) != id(target_embedding):
            _copy_parameter(
                target_head,
                global_map.fold_consumer_weight(
                    parent.get_output_embeddings()
                    .weight.detach()
                    .to(device="cpu", dtype=torch.float32)
                ),
                "lm_head.weight",
                written,
            )
        else:
            written.add(id(target_head))

        for layer_index, (source, target, fields) in enumerate(
            zip(source_layers, target_layers, evidence.layer_fields, strict=True)
        ):
            source_contract = inspect_lfm2_layer(source)
            target_contract = inspect_lfm2_layer(target)
            _copy_parameter(
                target.operator_norm.weight,
                _transport_diagonal(global_map, source.operator_norm.weight),
                f"model.layers.{layer_index}.operator_norm.weight",
                written,
            )
            _copy_parameter(
                target.ffn_norm.weight,
                _transport_diagonal(global_map, source.ffn_norm.weight),
                f"model.layers.{layer_index}.ffn_norm.weight",
                written,
            )

            mlp_map = build_feature_fold_map(
                _causal_weighted_mlp_fields(
                    fields,
                    anchors_by_layer.get(layer_index, ()),
                ),
                candidate.intermediate_size,
                source_width=int(source.feed_forward.w1.weight.shape[0]),
                chunk_size=128,
                sketch_size=64,
                balance_strength=0.35,
                # A SwiGLU neuron is not sign-equivariant:
                # silu(-gate) * (-up) != silu(gate) * up.  Negative hidden
                # feature coefficients therefore are not legal neuron
                # orientations even though signed residual coordinates are
                # legal.  Keep every neuron, but merge them with non-negative
                # coefficients.
                allow_sign_flip=False,
            )
            if mlp_map.report.negative_coefficient_count:
                raise Lfm2StructuralCompressionError(
                    f"Layer {layer_index} SwiGLU map contains an illegal "
                    "negative neuron coefficient."
                )
            residual_consumer_w1 = global_map.fold_consumer_weight(
                source.feed_forward.w1.weight.detach().to(
                    device="cpu", dtype=torch.float32
                )
            )
            residual_consumer_w3 = global_map.fold_consumer_weight(
                source.feed_forward.w3.weight.detach().to(
                    device="cpu", dtype=torch.float32
                )
            )
            _copy_parameter(
                target.feed_forward.w1.weight,
                mlp_map.fold_producer_weight(residual_consumer_w1),
                f"model.layers.{layer_index}.feed_forward.w1.weight",
                written,
            )
            _copy_parameter(
                target.feed_forward.w3.weight,
                mlp_map.fold_producer_weight(residual_consumer_w3),
                f"model.layers.{layer_index}.feed_forward.w3.weight",
                written,
            )
            _copy_parameter(
                target.feed_forward.w2.weight,
                global_map.fold_producer_weight(
                    mlp_map.fold_consumer_weight(
                        source.feed_forward.w2.weight.detach().to(
                            device="cpu", dtype=torch.float32
                        )
                    )
                ),
                f"model.layers.{layer_index}.feed_forward.w2.weight",
                written,
            )

            operator_report: dict[str, Any]
            if source_contract.kind == "full_attention":
                maps, operator_report = build_attention_maps(
                    fields,
                    source_query_heads=source_q,
                    source_key_value_heads=source_kv,
                    target_query_heads=candidate.query_heads,
                    target_key_value_heads=candidate.key_value_heads,
                    head_dim=source_head_dim,
                )
                for role, projection in (
                    ("query", "q_proj"),
                    ("key", "k_proj"),
                    ("value", "v_proj"),
                ):
                    source_weight = getattr(source.self_attn, projection).weight
                    target_weight = getattr(target.self_attn, projection).weight
                    _copy_parameter(
                        target_weight,
                        maps[role].fold_producer_weight(
                            global_map.fold_consumer_weight(
                                source_weight.detach().to(
                                    device="cpu", dtype=torch.float32
                                )
                            )
                        ),
                        f"model.layers.{layer_index}.self_attn.{projection}.weight",
                        written,
                    )
                _copy_parameter(
                    target.self_attn.out_proj.weight,
                    global_map.fold_producer_weight(
                        maps["context"].fold_consumer_weight(
                            source.self_attn.out_proj.weight.detach().to(
                                device="cpu", dtype=torch.float32
                            )
                        )
                    ),
                    f"model.layers.{layer_index}.self_attn.out_proj.weight",
                    written,
                )
                for norm_name in ("q_layernorm", "k_layernorm"):
                    _copy_parameter(
                        getattr(target.self_attn, norm_name).weight,
                        getattr(source.self_attn, norm_name).weight.detach(),
                        f"model.layers.{layer_index}.self_attn.{norm_name}.weight",
                        written,
                    )
            else:
                source_conv = source.conv
                target_conv = target.conv
                blocks = source_conv.in_proj.weight.chunk(3, dim=0)
                in_weight = torch.cat(
                    tuple(_transport_square(global_map, block) for block in blocks),
                    dim=0,
                )
                _copy_parameter(
                    target_conv.in_proj.weight,
                    in_weight,
                    f"model.layers.{layer_index}.conv.in_proj.weight",
                    written,
                )
                _copy_parameter(
                    target_conv.out_proj.weight,
                    _transport_square(global_map, source_conv.out_proj.weight),
                    f"model.layers.{layer_index}.conv.out_proj.weight",
                    written,
                )
                kernel = source_conv.conv.weight.detach().to(
                    device="cpu", dtype=torch.float32
                )[:, 0, :]
                folded_kernel = _transport_depthwise_kernel(global_map, kernel)
                _copy_parameter(
                    target_conv.conv.weight,
                    folded_kernel.unsqueeze(1),
                    f"model.layers.{layer_index}.conv.conv.weight",
                    written,
                )
                bias_names = ("in_proj", "out_proj")
                for projection_name in bias_names:
                    source_bias = getattr(source_conv, projection_name).bias
                    target_bias = getattr(target_conv, projection_name).bias
                    if source_bias is not None:
                        if projection_name == "in_proj":
                            folded_bias = torch.cat(
                                tuple(
                                    _map_vector_producer(global_map, block)
                                    for block in source_bias.chunk(3)
                                )
                            )
                        else:
                            folded_bias = _map_vector_producer(global_map, source_bias)
                        _copy_parameter(
                            target_bias,
                            folded_bias,
                            f"model.layers.{layer_index}.conv.{projection_name}.bias",
                            written,
                        )
                if source_conv.conv.bias is not None:
                    _copy_parameter(
                        target_conv.conv.bias,
                        _map_vector_producer(global_map, source_conv.conv.bias),
                        f"model.layers.{layer_index}.conv.conv.bias",
                        written,
                    )
                operator_report = {
                    "kind": "short-conv-dense-global-coordinate-transport",
                    "source_coverage_fraction": 1.0,
                    "dropped_source_channels": 0,
                    "input_projection_transport": "B.T @ W_block @ B",
                    "output_projection_transport": "B.T @ W @ B",
                    "bias_transport": "B.T @ b",
                    "depthwise_fir_transport": (
                        "diag(B.T @ diag(k_tap) @ B) "
                        "= sum_s B[s,t]^2 * k[s,tap]"
                    ),
                    "sparse_assignment_assumed": False,
                    "channel_locality_assumed": False,
                    "nonlinear_exactness": "approximate",
                    "approximation_reason": (
                        "B*x and C*v are channelwise multiplicative and a dense "
                        "basis does not preserve that product exactly; each FIR "
                        "tap installs the diagonal of the transported operator."
                    ),
                }
            layer_reports.append(
                {
                    "layer_index": layer_index,
                    "kind": target_contract.kind,
                    "mlp": {
                        "source_width": mlp_map.source_width,
                        "target_width": mlp_map.target_width,
                        "source_coverage_fraction": 1.0,
                        "dropped_source_neurons": 0,
                        "causal_anchor_count": len(anchors_by_layer.get(layer_index, ())),
                        "report": mlp_map.report.as_dict(),
                    },
                    "operator": operator_report,
                }
            )

        _copy_parameter(
            _lfm2_final_norm(child).weight,
            _transport_diagonal(
                global_map,
                _lfm2_final_norm(parent).weight,
            ),
            "model.embedding_norm.weight",
            written,
        )

    target_unique_objects = {id(parameter) for parameter in child.parameters()}
    missing = target_unique_objects - written
    if missing:
        names = [
            name
            for name, parameter in _named_parameters_with_duplicates(child)
            if id(parameter) in missing
        ]
        raise Lfm2StructuralCompressionError(
            "Fold did not overwrite every unique target Parameter: " + ", ".join(names)
        )
    if parameter_topology_snapshot(child) != blank.topology_before_fold:
        raise Lfm2StructuralCompressionError(
            "Target Parameter objects/shapes/dtypes changed after blank instantiation."
        )
    parent_after = fingerprint_lfm2_parent(parent)
    if parent_after != parent_before:
        raise Lfm2StructuralCompressionError("Parent changed during structural folding.")
    dense = dense_logical_parameter_audit(child, require_any_nonzero=True)
    coverage = {
        "layers": FIXED_LAYER_COUNT,
        "vocabulary_rows": int(parent.get_input_embeddings().weight.shape[0]),
        "hidden_coordinates": global_map.source_width,
        "ffn_neurons_per_layer": int(source_layers[0].feed_forward.w1.weight.shape[0]),
        "query_heads_per_attention_layer": source_q,
        "kv_groups_per_attention_layer": source_kv,
        "conv_channels_per_conv_layer": global_map.source_width,
        "source_coverage_fraction": 1.0,
        "dropped_source_units": 0,
        "all_parent_layers_assigned": True,
        "all_parent_spans_assigned": True,
        "all_parent_vocab_rows_transported": True,
        "no_top_k_or_subset_copy": True,
    }
    return {
        "schema": FOLD_SCHEMA,
        "candidate": candidate.as_dict(),
        "global_residual_map": residual_report,
        "layers": layer_reports,
        "source_coverage": coverage,
        "dense_parameter_audit": dense,
        "folded_state_sha256": model_state_sha256(child),
        "complete_target_parameter_overwrite": True,
        "parameter_objects_shapes_dtypes_unchanged": True,
        "parent_immutable": True,
        "parent_forward_calls": 0,
        "parent_trace_regeneration_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2StructuralCompressionError(f"Could not read JSON file {path}.") from exc
    if not isinstance(value, Mapping):
        raise Lfm2StructuralCompressionError(f"JSON file {path} must contain an object.")
    return dict(value)


def _local_model_directory(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2StructuralCompressionError("Model must be an explicit local directory.")
    path = Path(raw).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise Lfm2StructuralCompressionError(f"Local model directory is missing: {path}.")
    return path


def _local_safetensor_files(directory: Path) -> tuple[Path, ...]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Lfm2StructuralCompressionError("Safetensors index has no weight map.")
        result = tuple(directory / name for name in sorted(set(weight_map.values())))
    else:
        single = directory / "model.safetensors"
        result = (single,) if single.is_file() else tuple(sorted(directory.glob("model-*.safetensors")))
    if not result or any(not path.is_file() for path in result):
        raise Lfm2StructuralCompressionError("Local safetensors are incomplete.")
    return result


def load_local_lfm2(
    directory: str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Direct JSON+safetensors load; no Auto class, Hub, or from_pretrained."""

    from safetensors import safe_open
    from transformers import Lfm2Config, Lfm2ForCausalLM

    root = _local_model_directory(directory)
    payload = _read_json(root / "config.json")
    if payload.get("model_type") != "lfm2":
        raise Lfm2StructuralCompressionError("Local config is not LFM2.")
    model = Lfm2ForCausalLM(Lfm2Config(**payload)).to(dtype=dtype)
    model.tie_weights()
    state = model.state_dict()
    loaded = set()
    with torch.no_grad():
        for shard in _local_safetensor_files(root):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name not in state:
                        raise Lfm2StructuralCompressionError(
                            f"Unexpected local checkpoint tensor {name!r}."
                        )
                    source = handle.get_tensor(name)
                    if tuple(source.shape) != tuple(state[name].shape):
                        raise Lfm2StructuralCompressionError(
                            f"Local checkpoint shape mismatch for {name!r}."
                        )
                    state[name].copy_(source.to(dtype=state[name].dtype))
                    loaded.add(name)
    loaded_views = {_storage_view(state[name]) for name in loaded}
    missing = [name for name, value in state.items() if name not in loaded and _storage_view(value) not in loaded_views]
    if missing:
        raise Lfm2StructuralCompressionError(
            "Local checkpoint is missing non-aliased tensors: " + ", ".join(missing)
        )
    return model.to(device=device, dtype=dtype).eval()


def save_dense_local_checkpoint(
    model: nn.Module,
    source_directory: Path,
    destination: Path,
    report: Mapping[str, Any],
) -> None:
    from safetensors.torch import save_file

    if destination.exists():
        raise Lfm2StructuralCompressionError(
            f"Output checkpoint directory already exists: {destination}."
        )
    destination.mkdir(parents=True)
    config = _config_payload(model.config)
    config["_name_or_path"] = ""
    config.pop("auto_map", None)
    (destination / "config.json").write_bytes(_canonical_json(config) + b"\n")
    tensors: dict[str, torch.Tensor] = {}
    seen_views = set()
    for name, value in sorted(model.state_dict().items()):
        view = _storage_view(value)
        if view in seen_views:
            continue
        seen_views.add(view)
        tensors[name] = value.detach().to(device="cpu").contiguous()
    save_file(tensors, str(destination / "model.safetensors"))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        source = source_directory / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    (destination / "structural-fold-report.json").write_bytes(
        _canonical_json(dict(report)) + b"\n"
    )


def save_circuit_balanced_basis_artifact(
    output_directory: Path,
    basis_result: Any,
) -> dict[str, Any]:
    """Persist and immediately reauthenticate the full balanced realization."""

    from safetensors.torch import load_file, save_file

    basis_path = output_directory / "residual-coordinate-basis.safetensors"
    tensors = {
        "basis": basis_result.basis,
        "balanced_encoder": basis_result.balanced_encoder,
        "balanced_decoder": basis_result.balanced_decoder,
        "full_weight_gram": basis_result.full_weight_gram,
        "controllability_gram": basis_result.controllability_gram,
        "observability_gram": basis_result.observability_gram,
        "heldout_gram": basis_result.heldout_gram,
    }
    materialized = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in tensors.items()
    }
    save_file(materialized, str(basis_path))
    reloaded = load_file(str(basis_path), device="cpu")
    if set(reloaded) != set(materialized):
        raise Lfm2StructuralCompressionError(
            "Fresh circuit-balanced basis reload changed the tensor inventory."
        )
    mismatches = [
        name
        for name in sorted(materialized)
        if not torch.equal(reloaded[name], materialized[name])
    ]
    if mismatches:
        raise Lfm2StructuralCompressionError(
            "Fresh circuit-balanced basis reload changed tensors: "
            + ", ".join(mismatches)
            + "."
        )
    return {
        "file": basis_path.name,
        "file_sha256": _file_sha256(basis_path),
        "basis_sha256": basis_result.map_sha256,
        "reloaded_basis_tensor_sha256": _tensor_sha256(reloaded["basis"]),
        "source_width": basis_result.source_width,
        "target_width": basis_result.target_width,
        "tensor_names": sorted(materialized),
        "tensor_sha256": {
            name: _tensor_sha256(reloaded[name])
            for name in sorted(reloaded)
        },
        "contains_complete_objective_grams": True,
        "contains_balanced_encoder_decoder": True,
        "fresh_tensor_reload_verified": True,
    }


def _parse_dtype(value: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("dtype must be float16, bfloat16, or float32") from exc


def execute(args: argparse.Namespace) -> dict[str, Any]:
    candidate = _CANDIDATES_BY_NAME[args.candidate]
    device = torch.device(args.device)
    parent = load_local_lfm2(args.parent_model, device=device, dtype=args.dtype)
    parent_fingerprint = fingerprint_lfm2_parent(parent)
    parent_dense = dense_logical_parameter_audit(parent, require_any_nonzero=True)
    if parent_dense["unique_logical_parameters"] != EXPECTED_PARENT_UNIQUE_PARAMETERS:
        raise Lfm2StructuralCompressionError(
            "This verified ladder is bound to the canonical 229,693,184-parameter "
            "LFM2-230M parent; the loaded parent has "
            f"{parent_dense['unique_logical_parameters']} unique parameters."
        )
    cache = open_lfm2_circuit_trace_cache(args.trace_cache, verify=True)
    if cache.parent_fingerprint != parent_fingerprint.identity_sha256:
        raise Lfm2StructuralCompressionError("Trace cache is not bound to the local parent.")
    evidence = extract_cached_fold_evidence(
        cache,
        parent,
        max_rows=args.max_rows,
        max_attention_rows=args.max_attention_rows,
        max_vocab_rows=args.max_vocab_rows,
    )
    residual_basis_mode = getattr(
        args,
        "residual_basis",
        "cached-balanced",
    )
    if residual_basis_mode not in {
        "cached-balanced",
        "weight-complete",
        "circuit-balanced",
    }:
        raise Lfm2StructuralCompressionError(
            "residual_basis must be cached-balanced, weight-complete, or "
            "circuit-balanced."
        )
    weight_complete_basis = None
    circuit_balanced_basis = None
    global_map_override = None
    global_map_report_override = None
    if residual_basis_mode == "weight-complete":
        from neural_forge.hf_lfm2_weight_complete_basis import (
            build_lfm2_weight_complete_basis,
        )

        activation_weight = float(
            getattr(args, "weight_basis_activation_weight", 0.0)
        )
        activation_gram = None
        if activation_weight:
            activation_rows = torch.cat(
                (
                    evidence.residual_rows.to(torch.float64),
                    evidence.observable_rows.to(torch.float64),
                    evidence.margin_rows.to(torch.float64)
                    * evidence.margin_weights.to(torch.float64)
                    .clamp_min(0.0)
                    .sqrt()
                    .unsqueeze(1),
                ),
                dim=0,
            )
            activation_gram = (
                activation_rows.T @ activation_rows
            ).contiguous()
        weight_complete_basis = build_lfm2_weight_complete_basis(
            parent,
            candidate.hidden_size,
            activation_gram=activation_gram,
            activation_weight=activation_weight,
            maximum_activation_weight=float(
                getattr(
                    args,
                    "weight_basis_maximum_activation_weight",
                    0.25,
                )
            ),
            minimum_full_weight_retained_energy_fraction=float(
                getattr(
                    args,
                    "weight_basis_minimum_retained_energy",
                    0.0,
                )
            ),
        )
        global_map_override = weight_complete_basis
        global_map_report_override = weight_complete_basis.as_dict()
    elif residual_basis_mode == "circuit-balanced":
        from neural_forge.hf_lfm2_balanced_circuit_basis import (
            build_lfm2_circuit_balanced_basis,
        )

        circuit_balanced_basis = build_lfm2_circuit_balanced_basis(
            parent,
            cache,
            candidate.hidden_size,
            validation_fraction=float(
                getattr(
                    args,
                    "circuit_basis_validation_fraction",
                    0.25,
                )
            ),
            split_seed=str(
                getattr(
                    args,
                    "circuit_basis_split_seed",
                    "lfm2-circuit-balanced-v1",
                )
            ),
            max_rows_per_block=int(
                getattr(
                    args,
                    "circuit_basis_max_rows_per_block",
                    512,
                )
            ),
            structural_controllability_weight=float(
                getattr(
                    args,
                    "circuit_basis_structural_weight",
                    0.35,
                )
            ),
            cached_controllability_weight=float(
                getattr(
                    args,
                    "circuit_basis_cached_weight",
                    0.65,
                )
            ),
            gram_regularization=float(
                getattr(
                    args,
                    "circuit_basis_gram_regularization",
                    1e-10,
                )
            ),
            rank_tolerance=float(
                getattr(
                    args,
                    "circuit_basis_rank_tolerance",
                    1e-12,
                )
            ),
            max_condition_number=float(
                getattr(
                    args,
                    "circuit_basis_max_condition_number",
                    1e12,
                )
            ),
            minimum_heldout_retained_energy_fraction=float(
                getattr(
                    args,
                    "circuit_basis_minimum_heldout_retained_energy",
                    0.0,
                )
            ),
            vocabulary_margin_observability=bool(
                getattr(
                    args,
                    "circuit_basis_vocabulary_margin_observability",
                    False,
                )
            ),
            vocabulary_margin_top_k=int(
                getattr(
                    args,
                    "circuit_basis_vocabulary_margin_top_k",
                    8,
                )
            ),
            vocabulary_margin_floor_fraction=float(
                getattr(
                    args,
                    "circuit_basis_vocabulary_margin_floor_fraction",
                    0.05,
                )
            ),
        )
        global_map_override = circuit_balanced_basis
        global_map_report_override = circuit_balanced_basis.as_dict()
    if args.global_causal_frontier is None:
        guidance = {
            "status": "not-supplied",
            "causal_anchor_neurons_by_layer": {},
            "used_as_weighting_evidence_only": True,
            "source_units_dropped": 0,
        }
    else:
        if args.expected_global_causal_sha256 is None:
            raise Lfm2StructuralCompressionError(
                "--expected-global-causal-sha256 is required when a frontier is supplied."
            )
        guidance = load_global_causal_guidance(
            Path(args.global_causal_frontier).resolve(strict=True),
            expected_sha256=args.expected_global_causal_sha256,
            parent_identity_sha256=parent_fingerprint.identity_sha256,
            cache_key=cache.cache_key,
            corpus_sha256=cache.corpus_fingerprint,
        )
        guidance["used_as_weighting_evidence_only"] = True
        guidance["source_units_dropped"] = 0
    first = instantiate_compiler_blank(
        parent,
        candidate,
        seed=args.blank_seeds[0],
        device=device,
        dtype=args.dtype,
    )
    validate_live_structural_topology(parent, first.model, candidate)
    try:
        sequence_horizon_contract = build_sequence_horizon_contract(
            parent_config=parent.config,
            child_config=first.model.config,
            trace_manifest=cache.manifest,
            evaluation_maximum_new_tokens=None,
        )
    except Exception as exc:
        raise Lfm2StructuralCompressionError(
            "Could not prove independent model-capacity and folding-evidence "
            "horizons."
        ) from exc
    first_fold = fold_lfm2_into_blank(
        parent,
        first,
        evidence,
        candidate,
        global_guidance=guidance,
        global_map_override=global_map_override,
        global_map_report_override=global_map_report_override,
    )
    second = instantiate_compiler_blank(
        parent,
        candidate,
        seed=args.blank_seeds[1],
        device=device,
        dtype=args.dtype,
    )
    validate_live_structural_topology(parent, second.model, candidate)
    second_fold = fold_lfm2_into_blank(
        parent,
        second,
        evidence,
        candidate,
        global_guidance=guidance,
        global_map_override=global_map_override,
        global_map_report_override=global_map_report_override,
    )
    if first.random_state_sha256 == second.random_state_sha256:
        raise Lfm2StructuralCompressionError("Independent random pre-blanks are not distinct.")
    if first.blank_state_sha256 != second.blank_state_sha256:
        raise Lfm2StructuralCompressionError("Compiler-zeroed blanks differ by seed.")
    if first_fold["folded_state_sha256"] != second_fold["folded_state_sha256"]:
        mismatches = differing_state_tensors(first.model, second.model)
        raise Lfm2StructuralCompressionError(
            "Folded states retain blank seed information or used a "
            "nondeterministic transport: "
            + ", ".join(mismatches[:32])
        )
    common_compiler_report = {
        "schema": FOLD_SCHEMA,
        "candidate": candidate.as_dict(),
        "evidence_sha256": evidence.evidence_sha256,
        "folded_state_sha256": first_fold["folded_state_sha256"],
    }
    seed_report = audit_seed_erasure(
        (
            SeedCompileOutcome(
                args.blank_seeds[0],
                first.model,
                first.exact_unique_parameters,
                common_compiler_report,
                label="blank-seed-a",
            ),
            SeedCompileOutcome(
                args.blank_seeds[1],
                second.model,
                second.exact_unique_parameters,
                common_compiler_report,
                label="blank-seed-b",
            ),
        ),
        lambda _outcome: {
            "folded-parameter-state": first_fold["folded_state_sha256"]
        },
    )
    realized_reduction = 1.0 - (
        first.exact_unique_parameters / parent_dense["unique_logical_parameters"]
    )
    report = {
        "schema": RUN_SCHEMA,
        "candidate": candidate.as_dict(),
        "parent": {
            **parent_fingerprint.as_dict(),
            "dense_parameter_audit": parent_dense,
        },
        "child": {
            "dense_parameter_audit": first_fold["dense_parameter_audit"],
            "exact_unique_logical_parameters": first.exact_unique_parameters,
            "realized_logical_parameter_reduction_fraction": realized_reduction,
            "compression_metric": "unique-learned-Parameter-elements-only",
        },
        "cache": {
            "cache_key": cache.cache_key,
            "corpus_sha256": cache.corpus_fingerprint,
            "cached_evidence_sha256": evidence.evidence_sha256,
            "parent_forward_calls": 0,
            "parent_trace_regeneration_calls": 0,
        },
        "sequence_horizon_contract": sequence_horizon_contract,
        "residual_basis_selection": {
            "mode": residual_basis_mode,
            "prompt_independent_whole_weight_basis": (
                residual_basis_mode
                in {"weight-complete", "circuit-balanced"}
            ),
            "cached_residual_controllability": (
                residual_basis_mode == "circuit-balanced"
            ),
            "downstream_observability_balancing": (
                residual_basis_mode == "circuit-balanced"
            ),
            "activation_supplement_weight": (
                0.0
                if weight_complete_basis is None
                else weight_complete_basis.report[
                    "bounded_activation_supplement"
                ]["actual_weight"]
            ),
            "weight_complete_report": (
                None
                if weight_complete_basis is None
                else weight_complete_basis.as_dict()
            ),
            "circuit_balanced_report": (
                None
                if circuit_balanced_basis is None
                else circuit_balanced_basis.as_dict()
            ),
        },
        "global_causal_guidance": guidance,
        "blank_first": {
            "seeds": list(args.blank_seeds),
            "random_preblank_states_distinct": True,
            "zero_blank_states_identical": True,
            "folded_states_identical": True,
            "first_provenance": dict(first.provenance),
            "second_provenance": dict(second.provenance),
            "seed_erasure": seed_report.as_dict(),
        },
        "fold": first_fold,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "hub_loader_used": False,
        "network_allowed": False,
        "dtype_or_packing_reduction_claimed": False,
    }
    report["run_sha256"] = _json_sha256(report)
    output_directory = Path(args.output).expanduser().absolute()
    save_dense_local_checkpoint(
        first.model,
        _local_model_directory(args.parent_model),
        output_directory,
        report,
    )
    if weight_complete_basis is not None:
        from safetensors.torch import save_file

        basis_path = output_directory / "residual-coordinate-basis.safetensors"
        save_file(
            {
                "basis": weight_complete_basis.basis,
                "full_weight_gram": weight_complete_basis.full_weight_gram,
                "balanced_structural_gram": (
                    weight_complete_basis.balanced_structural_gram
                ),
                "combined_gram": weight_complete_basis.combined_gram,
            },
            str(basis_path),
        )
        report["residual_basis_selection"]["artifact"] = {
            "file": basis_path.name,
            "file_sha256": _file_sha256(basis_path),
            "basis_sha256": weight_complete_basis.map_sha256,
            "source_width": weight_complete_basis.source_width,
            "target_width": weight_complete_basis.target_width,
            "contains_complete_objective_grams": True,
        }
    elif circuit_balanced_basis is not None:
        report["residual_basis_selection"]["artifact"] = (
            save_circuit_balanced_basis_artifact(
                output_directory,
                circuit_balanced_basis,
            )
        )
    # A checkpoint is not accepted merely because the in-memory object passed.
    # Release the second proof model, reconstruct a fresh concrete LFM2 from the
    # local JSON+safetensors bytes, and reauthenticate topology/count/content.
    del second
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = load_local_lfm2(output_directory, device=device, dtype=args.dtype)
    reload_topology = validate_live_structural_topology(parent, reloaded, candidate)
    reload_dense = dense_logical_parameter_audit(reloaded, require_any_nonzero=True)
    reload_hash = model_state_sha256(reloaded)
    if reload_dense["unique_logical_parameters"] != first.exact_unique_parameters:
        raise Lfm2StructuralCompressionError(
            "Fresh local reload changed the exact unique logical parameter count."
        )
    if reload_hash != first_fold["folded_state_sha256"]:
        raise Lfm2StructuralCompressionError(
            "Fresh local reload changed folded dense Parameter/buffer content."
        )
    report["fresh_local_reload"] = {
        "verified": True,
        "state_sha256": reload_hash,
        "exact_unique_logical_parameters": reload_dense[
            "unique_logical_parameters"
        ],
        "topology": reload_topology,
        "from_pretrained_used": False,
        "hub_loader_used": False,
    }
    report["run_sha256"] = _json_sha256(
        {key: value for key, value in report.items() if key != "run_sha256"}
    )
    (output_directory / "structural-fold-report.json").write_bytes(
        _canonical_json(report) + b"\n"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--trace-cache", required=True)
    parser.add_argument("--global-causal-frontier")
    parser.add_argument("--expected-global-causal-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidate",
        choices=tuple(_CANDIDATES_BY_NAME),
        default=STRUCTURAL_CANDIDATES[0].name,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--blank-seeds", type=int, nargs=2, default=(1701, 2903))
    parser.add_argument(
        "--residual-basis",
        choices=(
            "cached-balanced",
            "weight-complete",
            "circuit-balanced",
        ),
        default="cached-balanced",
        help=(
            "weight-complete selects residual coordinates from every parent "
            "weight; circuit-balanced joins complete weights, disjoint sealed "
            "cache controllability, and circuit-IR observability; "
            "cached-balanced preserves the historical trace-selected baseline"
        ),
    )
    parser.add_argument(
        "--weight-basis-activation-weight",
        type=float,
        default=0.0,
        help=(
            "Optional bounded trace-Gram supplement; 0 keeps the coordinate "
            "basis fully prompt-independent"
        ),
    )
    parser.add_argument(
        "--weight-basis-maximum-activation-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--weight-basis-minimum-retained-energy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--circuit-basis-validation-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--circuit-basis-split-seed",
        default="lfm2-circuit-balanced-v1",
    )
    parser.add_argument(
        "--circuit-basis-max-rows-per-block",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--circuit-basis-structural-weight",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--circuit-basis-cached-weight",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--circuit-basis-gram-regularization",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--circuit-basis-rank-tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--circuit-basis-max-condition-number",
        type=float,
        default=1e12,
    )
    parser.add_argument(
        "--circuit-basis-minimum-heldout-retained-energy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--circuit-basis-vocabulary-margin-observability",
        action="store_true",
        help=(
            "Add fit-only sealed parent top-1/top-k vocabulary-margin "
            "directions as a separately balanced observability role."
        ),
    )
    parser.add_argument(
        "--circuit-basis-vocabulary-margin-top-k",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--circuit-basis-vocabulary-margin-floor-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--max-attention-rows", type=int, default=512)
    parser.add_argument("--max-vocab-rows", type=int, default=2048)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = execute(args)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
