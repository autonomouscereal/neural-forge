"""No-parent-forward long-horizon terminal refolding for an LFM2 child.

This module closes one deliberately narrow gap between the legacy LFM2
circuit cache and the pure terminal-action residual compiler:

* authenticate a sealed legacy cache, using bounded strict-row reads or one
  complete authenticated batch at a time for semantic retokenization;
* either invert an authenticated strict one-row tokenizer bridge or build an
  exact semantic full-prefix schedule from direct-local donor/target tokenizers;
* reconstruct the parent's final RMSNorm output analytically from the cached
  final-layer residual and a separately sealed final-normalization contract;
* replay each strict trajectory once, or each maximal target-native semantic
  branch once when suffix rewrites make one final token trajectory impossible,
  without calling ``model.forward`` or ``generate``;
* construct whole-trajectory FIT, FIT-CHECK, SELECTION, and VALIDATION banks;
* invoke the existing closed-form terminal-action residual primitive; and
* install the selected final ``w2`` writer in place only after its frozen
  validation gate accepts it.

There is no parent execution, gradient, optimizer, training loop, pruning,
quantization, parameter creation, or model-level child forward in this path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .hf_lfm2_circuit_trace_cache import (
    CACHED_VERIFIED_TRACE_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    _tensor_sha256 as _cache_tensor_sha256,
    fingerprint_lfm2_parent,
)
from .hf_lfm2_cpcf_macrocell_replay import replay_lfm2_cpcf_macrocell
from .hf_lfm2_cpcf_model_compiler import _rope_from_target
from .hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionEvidence,
    Lfm2CpcfTerminalActionPolicy,
    Lfm2CpcfTerminalActionResidualResult,
    Lfm2CpcfTerminalActionSeed,
    _evaluate_split,
    _prepare_split,
    compile_lfm2_cpcf_terminal_action_residual,
)
from .hf_lfm2_direct_local import (
    Lfm2DirectLocalError,
    _local_directory,
    _local_safetensor_shards,
    _read_json,
)
from .hf_lfm2_semantic_span_phase_bridge import (
    Lfm2SemanticSpanPhaseBridge,
    Lfm2SemanticSpanPhaseBridgeError,
    build_lfm2_semantic_span_phase_bridge,
)
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2TargetTokenizedRecord,
    authenticate_lfm2_bridge_input_pair,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from .hf_tied_causal_interface_fold import TiedCausalInterfaceFold
from .hf_tokenizer_bridge import StrictTokenizerBridgeArtifact
from .hf_tokenizer_bridge import StrictTokenizerBridgeError


LFM2_LONG_HORIZON_PARENT_FINAL_NORM_SCHEMA = (
    "neural-forge-lfm2-long-horizon-parent-final-norm-v1"
)
LFM2_LONG_HORIZON_SPLIT_PLAN_SCHEMA = "neural-forge-lfm2-long-horizon-split-plan-v1"
LFM2_LONG_HORIZON_TERMINAL_REFOLD_SCHEMA = (
    "neural-forge-lfm2-long-horizon-terminal-refold-v2"
)

TerminalSplit = Literal["fit", "fit_check", "selection", "validation"]
TokenizerReplayMode = Literal["strict_one_row", "semantic_span"]
_SPLITS: tuple[TerminalSplit, ...] = (
    "fit",
    "fit_check",
    "selection",
    "validation",
)
_HORIZON_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0-7", 0, 7),
    ("8-31", 8, 31),
    ("32+", 32, None),
)
_DECODE_FINAL_RESIDUAL = re.compile(
    r"^decode\.(\d{3})\.layer\.(\d{3})\.residual\.post_layer$"
)


class Lfm2LongHorizonTerminalRefoldError(ValueError):
    """Cache, bridge, replay, evidence, or installation failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "PyTorch is required for long-horizon LFM2 terminal refolding."
        ) from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Long-horizon metadata must be finite canonical JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _dtype_name(value: Any) -> str:
    return str(value.dtype).removeprefix("torch.")


def _tensor_sha256(value: Any, name: str = "tensor") -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            f"{name} must be a materialized dense tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": _dtype_name(tensor),
        "shape": [int(size) for size in tensor.shape],
        "stride": [int(size) for size in tensor.stride()],
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _interface_tensor_sha256(value: Any) -> str:
    """Match the authenticated hash contract of the tied-interface module."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _owned_floating_vector(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 1
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            f"{name} must be a nonempty dense real floating vector."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2LongHorizonTerminalRefoldError(f"{name} contains NaN or infinity.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2LongHorizonTerminalRefoldError(
            f"{name} must be a positive finite number."
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2LongHorizonTerminalRefoldError(
            f"{name} must be a positive finite number."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2SealedParentFinalRmsNorm:
    """Parent final RMSNorm bytes bound to one immutable circuit cache."""

    cache_key: str
    parent_fingerprint: str
    layer_topology_fingerprint: str
    weight: Any
    epsilon: float
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        cache_key = _digest(self.cache_key, "cache_key")
        parent = _digest(self.parent_fingerprint, "parent_fingerprint")
        topology = _digest(
            self.layer_topology_fingerprint,
            "layer_topology_fingerprint",
        )
        weight = _owned_floating_vector(self.weight, "weight")
        epsilon = _positive_finite(self.epsilon, "epsilon")
        body = {
            "schema": LFM2_LONG_HORIZON_PARENT_FINAL_NORM_SCHEMA,
            "cache_key": cache_key,
            "parent_fingerprint": parent,
            "layer_topology_fingerprint": topology,
            "weight_sha256": _tensor_sha256(weight, "parent_final_norm.weight"),
            "source_hidden_size": int(weight.numel()),
            "epsilon": epsilon,
            "parent_model_forward_calls": 0,
        }
        expected = _canonical_sha256(body)
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2LongHorizonTerminalRefoldError(
                "Parent final RMSNorm authentication is invalid."
            )
        object.__setattr__(self, "cache_key", cache_key)
        object.__setattr__(self, "parent_fingerprint", parent)
        object.__setattr__(self, "layer_topology_fingerprint", topology)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "authentication_sha256", expected)

    @property
    def source_hidden_size(self) -> int:
        return int(self.weight.numel())

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_LONG_HORIZON_PARENT_FINAL_NORM_SCHEMA,
            "cache_key": self.cache_key,
            "parent_fingerprint": self.parent_fingerprint,
            "layer_topology_fingerprint": self.layer_topology_fingerprint,
            "weight_sha256": _tensor_sha256(
                self.weight,
                "parent_final_norm.weight",
            ),
            "source_hidden_size": self.source_hidden_size,
            "epsilon": self.epsilon,
            "parent_model_forward_calls": 0,
        }

    def verify_authentication(self) -> bool:
        return self.authentication_sha256 == _canonical_sha256(
            self.authentication_body()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2LocalParentFinalRmsNormLoad:
    """Direct-local safetensors load bound to the cache's full parent identity."""

    final_norm: Lfm2SealedParentFinalRmsNorm
    report: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.final_norm, Lfm2SealedParentFinalRmsNorm)
            or not self.final_norm.verify_authentication()
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Direct-local parent final norm is not authenticated."
            )
        body = dict(self.report)
        supplied = body.pop("result_sha256", None)
        audit = body.get("operation_audit")
        if (
            supplied != self.result_sha256
            or _canonical_sha256(body) != supplied
            or body.get("cache_key") != self.final_norm.cache_key
            or body.get("cache_parent_fingerprint")
            != self.final_norm.parent_fingerprint
            or body.get("final_norm_authentication_sha256")
            != self.final_norm.authentication_sha256
            or not isinstance(audit, Mapping)
            or audit.get("parent_model_objects_constructed") != 0
            or audit.get("parent_model_forward_calls") != 0
            or audit.get("parent_model_generate_calls") != 0
            or audit.get("network_access_performed") is not False
            or audit.get("direct_local_json_and_safetensors_only") is not True
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Direct-local parent final-norm load receipt is invalid."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "result_sha256": supplied}),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024**2):
                digest.update(block)
    except OSError as exc:
        raise Lfm2LongHorizonTerminalRefoldError(
            f"Could not authenticate local checkpoint file {path.name!r}."
        ) from exc
    return digest.hexdigest()


def _direct_local_parent_runtime_buffers(config: Any, count: int) -> dict[str, Any]:
    torch = _torch()
    if count == 0:
        return {}
    rope = getattr(config, "rope_parameters", None)
    if (
        count != 2
        or not isinstance(rope, Mapping)
        or rope.get("rope_type") != "default"
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Direct-local parent identity reconstruction supports either no "
            "runtime buffers or the authenticated default LFM2 RoPE buffer pair."
        )
    base = _positive_finite(rope.get("rope_theta"), "parent rope_theta")
    hidden = int(getattr(config, "hidden_size", 0))
    heads = int(getattr(config, "num_attention_heads", 0))
    head_dim = int(getattr(config, "head_dim", 0) or (hidden // heads if heads else 0))
    if head_dim <= 0 or head_dim % 2 != 0:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Parent config cannot reconstruct the LFM2 RoPE buffers."
        )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64).to(torch.float32) / head_dim
        )
    )
    return {
        "model.rotary_emb.inv_freq": inv_freq,
        "model.rotary_emb.original_inv_freq": inv_freq.clone(),
    }


def load_lfm2_parent_final_rmsnorm_from_local_safetensors(
    cache: Lfm2CircuitTraceCache,
    model_directory: str | os.PathLike[str],
    *,
    tensor_key: str,
    epsilon_config_key: str,
) -> Lfm2LocalParentFinalRmsNormLoad:
    """Load one final norm without constructing or executing a parent model.

    The function scans the direct-local safetensors once to reconstruct the same
    state/config/runtime-buffer fingerprint sealed by the circuit cache.  The
    requested tensor and epsilon field are explicit caller inputs; no norm key or
    epsilon is guessed in core code.
    """

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(tensor_key, str) or not tensor_key.strip():
        raise Lfm2LongHorizonTerminalRefoldError("tensor_key must be nonempty text.")
    if not isinstance(epsilon_config_key, str) or not epsilon_config_key.strip():
        raise Lfm2LongHorizonTerminalRefoldError(
            "epsilon_config_key must be nonempty text."
        )
    try:
        from safetensors import safe_open
        from transformers import Lfm2Config
    except ImportError as exc:  # pragma: no cover - optional native dependency
        raise RuntimeError(
            "Direct-local final-norm authentication requires safetensors and "
            "the concrete LFM2 config class."
        ) from exc
    try:
        directory = _local_directory(model_directory, "model_directory")
        config_payload = _read_json(directory / "config.json", "LFM2 config")
        shards = _local_safetensor_shards(directory)
    except Lfm2DirectLocalError as exc:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Direct-local parent checkpoint layout is invalid."
        ) from exc
    if config_payload.get("model_type") != "lfm2":
        raise Lfm2LongHorizonTerminalRefoldError(
            "Direct-local parent config is not LFM2."
        )
    config = Lfm2Config(**config_payload)
    normalized_config = dict(config.to_dict())
    normalized_config.pop("_name_or_path", None)
    config_sha256 = _canonical_sha256(normalized_config)
    manifest = cache.manifest
    parent = manifest.get("parent")
    if not isinstance(parent, Mapping) or parent.get("config_sha256") != config_sha256:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Direct-local parent config differs from the cache's model identity."
        )

    tensor_locations: dict[str, Path] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensor_locations:
                    raise Lfm2LongHorizonTerminalRefoldError(
                        f"Duplicate direct-local parent tensor {name!r}."
                    )
                tensor_locations[name] = shard
    if tensor_key not in tensor_locations:
        raise Lfm2LongHorizonTerminalRefoldError(
            f"Direct-local parent lacks requested final norm tensor {tensor_key!r}."
        )

    tensor_hashes: dict[str, str] = {}
    selected_weight = None
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                value = handle.get_tensor(name)
                tensor_hashes[name] = _cache_tensor_sha256(_torch(), value)
                if name == tensor_key:
                    selected_weight = (
                        value.detach().to(device="cpu").contiguous().clone()
                    )
    expected_state_count = int(parent.get("state_tensor_count", -1))
    if (
        bool(getattr(config, "tie_word_embeddings", False))
        and len(tensor_hashes) + 1 == expected_state_count
    ):
        embedding_key = "model.embed_tokens.weight"
        output_key = "lm_head.weight"
        if embedding_key in tensor_hashes and output_key not in tensor_hashes:
            tensor_hashes[output_key] = tensor_hashes[embedding_key]
        elif output_key in tensor_hashes and embedding_key not in tensor_hashes:
            tensor_hashes[embedding_key] = tensor_hashes[output_key]
    if len(tensor_hashes) != expected_state_count:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Direct-local safetensors cannot reconstruct every state-dict entry "
            "sealed by the parent cache."
        )
    state_digest = hashlib.sha256(b"neural-forge-lfm2-parent-state-v1\0")
    for name in sorted(tensor_hashes):
        state_digest.update(name.encode("utf-8"))
        state_digest.update(b"\0")
        state_digest.update(tensor_hashes[name].encode("ascii"))
        state_digest.update(b"\0")
    state_sha256 = state_digest.hexdigest()

    expected_buffer_count = int(parent.get("runtime_buffer_count", -1))
    buffers = _direct_local_parent_runtime_buffers(config, expected_buffer_count)
    buffer_digest = hashlib.sha256(b"neural-forge-lfm2-parent-runtime-buffers-v1\0")
    for name in sorted(buffers):
        buffer_digest.update(name.encode("utf-8"))
        buffer_digest.update(b"\0")
        buffer_digest.update(
            _cache_tensor_sha256(_torch(), buffers[name]).encode("ascii")
        )
        buffer_digest.update(b"\0")
    buffer_sha256 = buffer_digest.hexdigest()
    identity_body = {
        "schema": "neural-forge-lfm2-parent-fingerprint-v2",
        "model_class": parent.get("model_class"),
        "config_sha256": config_sha256,
        "state_dict_sha256": state_sha256,
        "state_tensor_count": len(tensor_hashes),
        "runtime_buffer_sha256": buffer_sha256,
        "runtime_buffer_count": len(buffers),
    }
    if (
        state_sha256 != parent.get("state_dict_sha256")
        or buffer_sha256 != parent.get("runtime_buffer_sha256")
        or _canonical_sha256(identity_body) != cache.parent_fingerprint
        or selected_weight is None
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Direct-local safetensors do not reproduce the cache's full parent "
            "model identity."
        )
    epsilon = _positive_finite(
        normalized_config.get(epsilon_config_key),
        f"parent config {epsilon_config_key}",
    )
    hidden = int(getattr(config, "hidden_size", 0))
    if selected_weight.ndim != 1 or int(selected_weight.numel()) != hidden:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Requested direct-local final norm does not match parent hidden size."
        )
    final_norm = Lfm2SealedParentFinalRmsNorm(
        cache_key=cache.cache_key,
        parent_fingerprint=cache.parent_fingerprint,
        layer_topology_fingerprint=cache.layer_topology_fingerprint,
        weight=selected_weight,
        epsilon=epsilon,
    )
    body = {
        "schema": "neural-forge-lfm2-local-parent-final-rmsnorm-load-v1",
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "source_directory": str(directory),
        "config_file_sha256": _file_sha256(directory / "config.json"),
        "checkpoint_files": [
            {
                "name": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": _file_sha256(shard),
            }
            for shard in shards
        ],
        "tensor_key": tensor_key,
        "tensor_sha256": _cache_tensor_sha256(_torch(), selected_weight),
        "epsilon_config_key": epsilon_config_key,
        "epsilon": epsilon,
        "reconstructed_parent_identity": identity_body,
        "final_norm_authentication_sha256": final_norm.authentication_sha256,
        "operation_audit": {
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
            "parent_model_generate_calls": 0,
            "network_access_performed": False,
            "direct_local_json_and_safetensors_only": True,
            "training_steps": 0,
        },
    }
    seal = _canonical_sha256(body)
    return Lfm2LocalParentFinalRmsNormLoad(
        final_norm=final_norm,
        report={**body, "result_sha256": seal},
        result_sha256=seal,
    )


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonTrajectorySplitPlan:
    """One sealed whole-trajectory split assignment for a cache."""

    cache_key: str
    assignments: tuple[TerminalSplit, ...]
    assignment_method: str = "canonical-batch-index-modulo-four-v1"
    authentication_sha256: str = ""

    def __post_init__(self) -> None:
        cache_key = _digest(self.cache_key, "cache_key")
        assignments = tuple(self.assignments)
        if not assignments or any(value not in _SPLITS for value in assignments):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Split assignments must contain only the four terminal splits."
            )
        if set(assignments) != set(_SPLITS):
            raise Lfm2LongHorizonTerminalRefoldError(
                "FIT, FIT-CHECK, SELECTION, and VALIDATION each need at "
                "least one complete trajectory."
            )
        if (
            not isinstance(self.assignment_method, str)
            or not self.assignment_method.strip()
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "assignment_method must be nonempty text."
            )
        method = self.assignment_method.strip()
        body = {
            "schema": LFM2_LONG_HORIZON_SPLIT_PLAN_SCHEMA,
            "cache_key": cache_key,
            "batch_count": len(assignments),
            "assignments": list(assignments),
            "assignment_method": method,
            "whole_trajectory_isolation": True,
            "split_counts": {split: assignments.count(split) for split in _SPLITS},
        }
        expected = _canonical_sha256(body)
        supplied = str(self.authentication_sha256).strip().lower()
        if supplied and supplied != expected:
            raise Lfm2LongHorizonTerminalRefoldError(
                "Long-horizon split-plan authentication is invalid."
            )
        object.__setattr__(self, "cache_key", cache_key)
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "assignment_method", method)
        object.__setattr__(self, "authentication_sha256", expected)

    @classmethod
    def deterministic_round_robin(
        cls,
        cache: Lfm2CircuitTraceCache,
    ) -> "Lfm2LongHorizonTrajectorySplitPlan":
        if not isinstance(cache, Lfm2CircuitTraceCache):
            raise TypeError("cache must be Lfm2CircuitTraceCache.")
        if cache.batch_count < len(_SPLITS):
            raise Lfm2LongHorizonTerminalRefoldError(
                "At least four cache trajectories are required for isolated "
                "FIT, FIT-CHECK, SELECTION, and VALIDATION evidence."
            )
        return cls(
            cache_key=cache.cache_key,
            assignments=tuple(
                _SPLITS[index % len(_SPLITS)] for index in range(cache.batch_count)
            ),
        )

    def verify_for(self, cache: Lfm2CircuitTraceCache) -> None:
        if (
            self.cache_key != cache.cache_key
            or len(self.assignments) != cache.batch_count
            or self.authentication_sha256
            != _canonical_sha256(self.authentication_body())
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "The split plan is not authenticated for this cache."
            )

    def authentication_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_LONG_HORIZON_SPLIT_PLAN_SCHEMA,
            "cache_key": self.cache_key,
            "batch_count": len(self.assignments),
            "assignments": list(self.assignments),
            "assignment_method": self.assignment_method,
            "whole_trajectory_isolation": True,
            "split_counts": {split: self.assignments.count(split) for split in _SPLITS},
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.authentication_body(),
            "authentication_sha256": self.authentication_sha256,
        }


@dataclass(frozen=True, slots=True)
class _CachedTrajectory:
    batch_index: int
    input_sha256: str
    split: TerminalSplit
    target_record: Lfm2TargetTokenizedRecord | None
    semantic_bridge: Lfm2SemanticSpanPhaseBridge | None
    parent_final_norm_rows: Any
    horizon_indices: tuple[int, ...]
    selected_payload_sha256: str
    parent_final_norm_endpoint: Mapping[str, Any]
    parent_action_projection: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ChildTrajectoryRows:
    mlp_input: Any
    pre_mlp_residual: Any
    macrocell_replay_sha256: tuple[str, ...]
    attention_rope_materializations: int
    embedding_seed_lookup_calls: int = 1
    full_prefix_replay_calls: int = 1
    replay_strategy: str = "single-causal-trajectory-v1"
    replay_branch_phase_sha256: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class Lfm2LongHorizonTerminalRefoldResult:
    """Authenticated evidence, pure compiler result, and install receipt."""

    split_evidence: tuple[Lfm2CpcfTerminalActionEvidence, ...]
    terminal_result: Lfm2CpcfTerminalActionResidualResult
    accepted: bool
    report: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        evidence = tuple(self.split_evidence)
        if (
            len(evidence) != len(_SPLITS)
            or tuple(value.split for value in evidence) != _SPLITS
            or not all(value.verify_authentication() for value in evidence)
            or not isinstance(
                self.terminal_result,
                Lfm2CpcfTerminalActionResidualResult,
            )
            or not self.terminal_result.verify_authentication()
            or not isinstance(self.accepted, bool)
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Long-horizon result contains invalid typed evidence."
            )
        body = dict(self.report)
        supplied = body.pop("result_sha256", None)
        if (
            supplied != self.result_sha256
            or _canonical_sha256(body) != self.result_sha256
            or body.get("accepted") is not self.accepted
            or body.get("terminal_result_sha256") != self.terminal_result.result_sha256
            or body.get("split_evidence_sha256")
            != {value.split: value.authentication_sha256 for value in evidence}
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Long-horizon terminal-refold result seal is invalid."
            )
        object.__setattr__(self, "split_evidence", evidence)
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**body, "result_sha256": supplied}),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _verify_parent_norm_binding(
    cache: Lfm2CircuitTraceCache,
    seal: Lfm2SealedParentFinalRmsNorm,
) -> None:
    if not isinstance(seal, Lfm2SealedParentFinalRmsNorm):
        raise TypeError("parent_final_norm must be Lfm2SealedParentFinalRmsNorm.")
    if (
        not seal.verify_authentication()
        or seal.cache_key != cache.cache_key
        or seal.parent_fingerprint != cache.parent_fingerprint
        or seal.layer_topology_fingerprint != cache.layer_topology_fingerprint
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Parent final RMSNorm bytes are not bound to this sealed cache."
        )


def _verify_interface_report(
    interface: TiedCausalInterfaceFold,
    source_selected_tied_table: Any,
) -> dict[str, Any]:
    torch = _torch()
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    report = dict(interface.report)
    supplied = report.pop("report_sha256", None)
    if not isinstance(supplied, str) or supplied != _canonical_sha256(report):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Tied causal interface report authentication is invalid."
        )
    expected_target = report.get("target_weight_sha256")
    expected_encoder = report.get("producer_encoder_sha256")
    expected_source = report.get(
        "source_table_sha256",
        report.get("source_tied_table_sha256"),
    )
    source = source_selected_tied_table
    if (
        not isinstance(source, torch.Tensor)
        or source.layout != torch.strided
        or source.device.type == "meta"
        or source.ndim != 2
        or source.numel() == 0
        or source.dtype != torch.bfloat16
        or not bool(torch.isfinite(source).all().item())
        or expected_target != _interface_tensor_sha256(interface.target_weight)
        or expected_encoder != _interface_tensor_sha256(interface.producer_encoder)
        or expected_source
        != _interface_tensor_sha256(
            source.detach().to(device="cpu", dtype=torch.float64)
        )
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Tied interface/source selected-table tensors differ from their "
            "authenticated lineage."
        )
    return {**report, "report_sha256": supplied}


def _target_runtime_seed(
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    source_selected_tied_table: Any,
) -> tuple[Lfm2CpcfTerminalActionSeed, Any, tuple[Any, ...], float]:
    torch = _torch()
    core = getattr(target_model, "model", None)
    config = getattr(target_model, "config", None)
    layers = tuple(getattr(core, "layers", ()))
    if core is None or getattr(config, "model_type", None) != "lfm2" or not layers:
        raise Lfm2LongHorizonTerminalRefoldError(
            "target_model must be a materialized LFM2 causal language model."
        )
    final_layer = layers[-1]
    feed_forward = getattr(final_layer, "feed_forward", None)
    final_norm_module = getattr(core, "embedding_norm", None)
    final_norm = getattr(final_norm_module, "weight", None)
    target_input = getattr(target_model.get_input_embeddings(), "weight", None)
    target_output = getattr(target_model.get_output_embeddings(), "weight", None)
    gate = getattr(getattr(feed_forward, "w1", None), "weight", None)
    up = getattr(getattr(feed_forward, "w3", None), "weight", None)
    down = getattr(getattr(feed_forward, "w2", None), "weight", None)
    parameters = (gate, up, down, final_norm, target_input)
    if not all(isinstance(value, torch.nn.Parameter) for value in parameters):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Target final SwiGLU, final RMSNorm, or tied table is not materialized."
        )
    if (
        target_output is not target_input
        or any(value.dtype != torch.bfloat16 for value in parameters)
        or not torch.equal(
            target_input.detach().to(device="cpu"),
            interface.target_weight.to(torch.bfloat16),
        )
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Target must expose the exact installed native tied BF16 interface."
        )
    epsilon = _positive_finite(
        getattr(final_norm_module, "variance_epsilon", None),
        "target final RMSNorm variance_epsilon",
    )
    seed = Lfm2CpcfTerminalActionSeed(
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
        final_norm_weight=final_norm,
        producer_encoder=interface.producer_encoder,
        target_tied_weight=target_input,
        source_tied_table=source_selected_tied_table,
        norm_epsilon=epsilon,
    )
    return seed, down, layers, epsilon


def _strict_inverse_mapping(
    bridge: StrictTokenizerBridgeArtifact,
    target_rows: int,
) -> tuple[int, ...]:
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    try:
        bridge.verify_files()
    except Exception as exc:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Strict tokenizer bridge file authentication failed."
        ) from exc
    profile = bridge.transport_profile
    mapping = tuple(profile.target_to_donor_row_indices)
    if profile.target_row_capacity != target_rows or len(mapping) != target_rows:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Tokenizer bridge target row capacity differs from the installed "
            "tied child table."
        )
    inverse = [-1] * int(profile.donor_row_capacity)
    for target_id, donor_id in enumerate(mapping):
        if donor_id < 0 or donor_id >= len(inverse):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Tokenizer bridge donor row lies outside its capacity."
            )
        inverse[donor_id] = target_id if inverse[donor_id] == -1 else -2
    return tuple(inverse)


def _map_donor_ids_to_unique_target(
    value: Any,
    inverse: tuple[int, ...],
    *,
    name: str,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.numel() == 0
        or value.device.type != "cpu"
        or value.dtype
        not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        or bool((value < 0).any().item())
        or bool((value >= len(inverse)).any().item())
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            f"{name} is not a valid donor token-ID matrix."
        )
    lookup = torch.tensor(inverse, dtype=torch.long)
    mapped = lookup[value.to(torch.long)]
    if bool((mapped == -1).any().item()):
        raise Lfm2LongHorizonTerminalRefoldError(
            f"The strict bridge does not cover every cached {name} row."
        )
    if bool((mapped == -2).any().item()):
        raise Lfm2LongHorizonTerminalRefoldError(
            f"The strict bridge maps a cached {name} donor row to more than "
            "one target row; inverse replay would be ambiguous."
        )
    return mapped.contiguous().clone()


def _runtime_parent_rmsnorm(
    residual_rows: Any,
    seal: Lfm2SealedParentFinalRmsNorm,
) -> Any:
    """Replay the concrete LFM2 RMSNorm arithmetic without a model object."""

    torch = _torch()
    if (
        not isinstance(residual_rows, torch.Tensor)
        or residual_rows.ndim != 2
        or int(residual_rows.shape[1]) != seal.source_hidden_size
        or not torch.is_floating_point(residual_rows)
        or not bool(torch.isfinite(residual_rows).all().item())
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cached final residual rows differ from the sealed parent norm."
        )
    rows = residual_rows.detach().to(device="cpu").contiguous()
    input_dtype = rows.dtype
    hidden = rows.to(torch.float32)
    variance = hidden.square().mean(dim=1, keepdim=True)
    normalized = hidden * torch.rsqrt(variance + seal.epsilon)
    result = seal.weight * normalized.to(input_dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Runtime-equivalent parent final RMSNorm replay is nonfinite."
        )
    return result.contiguous().clone()


def _verify_parent_final_norm_endpoint(
    tensors: Mapping[str, Any],
    residual_rows: Any,
    seal: Lfm2SealedParentFinalRmsNorm,
    *,
    decode_steps: int,
) -> tuple[Any, dict[str, Any]]:
    """Bind direct-local norm bytes to every authenticated cache endpoint."""

    torch = _torch()
    phases = ("prefill",) + tuple(f"decode.{step:03d}" for step in range(decode_steps))
    norm_inputs = []
    norm_outputs = []
    head_inputs = []
    endpoint_names = tuple(
        f"{phase}.output.{field}"
        for phase in phases
        for field in (
            "final_norm_input",
            "final_norm_output",
            "lm_head_input",
        )
    )
    present_endpoint_names = tuple(name for name in endpoint_names if name in tensors)
    if not present_endpoint_names:
        replayed = _runtime_parent_rmsnorm(residual_rows, seal)
        body = {
            "method": "exact-parent-identity-bound-runtime-rmsnorm-replay-v1",
            "phase_order": list(phases),
            "phase_count": len(phases),
            "parent_final_norm_authentication_sha256": seal.authentication_sha256,
            "cached_final_residual_sha256": _tensor_sha256(residual_rows),
            "cached_output_endpoint_available": False,
            "runtime_replay_to_cached_output_verified": False,
            "runtime_replay_used_as_compiler_target": True,
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
        }
        return replayed.to(torch.bfloat16).contiguous().clone(), {
            **body,
            "authentication_sha256": _canonical_sha256(body),
        }
    if len(present_endpoint_names) != len(endpoint_names):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cache final RMSNorm endpoint inventory is partial or phase-incomplete."
        )
    for phase in phases:
        values = tuple(
            tensors.get(f"{phase}.output.{field}")
            for field in (
                "final_norm_input",
                "final_norm_output",
                "lm_head_input",
            )
        )
        if (
            any(not isinstance(value, torch.Tensor) for value in values)
            or any(value.ndim != 3 for value in values)
            or len({tuple(value.shape) for value in values}) != 1
            or int(values[0].shape[0]) != 1
            or int(values[0].shape[1]) < 1
            or int(values[0].shape[2]) != seal.source_hidden_size
            or any(not torch.is_floating_point(value) for value in values)
            or any(not bool(torch.isfinite(value).all().item()) for value in values)
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                f"Cache phase {phase!r} has no aligned final RMSNorm endpoint."
            )
        norm_inputs.append(values[0][:, -1, :].reshape(1, -1))
        norm_outputs.append(values[1][:, -1, :].reshape(1, -1))
        head_inputs.append(values[2][:, -1, :].reshape(1, -1))

    cached_input = torch.cat(norm_inputs, dim=0).to(device="cpu").contiguous()
    cached_output = torch.cat(norm_outputs, dim=0).to(device="cpu").contiguous()
    cached_head = torch.cat(head_inputs, dim=0).to(device="cpu").contiguous()
    residual = residual_rows.detach().to(device="cpu").contiguous()
    if not torch.equal(cached_input, residual):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cached final-layer residual does not exactly equal the final "
            "RMSNorm input."
        )
    if not torch.equal(cached_output, cached_head):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cached final RMSNorm output does not exactly equal LM-head input."
        )

    replayed = _runtime_parent_rmsnorm(cached_input, seal)
    if tuple(replayed.shape) != tuple(cached_output.shape):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Runtime-equivalent final RMSNorm replay changed endpoint shape."
        )
    replay64 = replayed.to(torch.float64)
    cached64 = cached_output.to(torch.float64)
    difference = (replay64 - cached64).abs()
    dtype = cached_output.dtype
    epsilon = float(torch.finfo(dtype).eps)
    # The checkpoint may have been traced on a different device, whose FP32
    # reduction order can differ.  Keep the allowance scale-relative (never a
    # denominator clamped to one): two storage-precision bins for BF16/FP16,
    # or 64 FP32/FP64 epsilons for the reduction itself.
    multiplier = 2.0 if dtype in {torch.bfloat16, torch.float16} else 64.0
    scale = torch.maximum(replay64.abs(), cached64.abs())
    tolerance = scale * (multiplier * epsilon)
    exact_zero = scale == 0
    equivalent = (difference <= tolerance) | (exact_zero & (difference == 0))
    if not bool(equivalent.all().item()):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Direct-local parent final RMSNorm does not reproduce the "
            "authenticated cache endpoint."
        )
    reference_rms = float(cached64.square().mean().sqrt().item())
    relative_rmse = float(difference.square().mean().sqrt().item()) / max(
        reference_rms,
        torch.finfo(torch.float64).tiny,
    )
    body = {
        "method": "direct-local-rmsnorm-runtime-replay-to-authenticated-cache-v1",
        "phase_order": list(phases),
        "phase_count": len(phases),
        "parent_final_norm_authentication_sha256": seal.authentication_sha256,
        "cached_final_residual_sha256": _tensor_sha256(residual),
        "cached_final_norm_input_sha256": _tensor_sha256(cached_input),
        "cached_final_norm_output_sha256": _tensor_sha256(cached_output),
        "cached_lm_head_input_sha256": _tensor_sha256(cached_head),
        "runtime_replayed_final_norm_output_sha256": _tensor_sha256(replayed),
        "final_residual_to_norm_input_exact": True,
        "final_norm_output_to_lm_head_input_exact": True,
        "runtime_replay_exact": bool(torch.equal(replayed, cached_output)),
        "runtime_replay_all_elements_verified": True,
        "runtime_replay_relative_rmse": relative_rmse,
        "runtime_replay_maximum_absolute_error": float(difference.max().item()),
        "scale_relative_tolerance_multiplier": multiplier,
        "storage_dtype_epsilon": epsilon,
        "absolute_denominator_floor_used": False,
        "cached_output_endpoint_available": True,
        "runtime_replay_to_cached_output_verified": True,
        "runtime_replay_used_as_compiler_target": False,
        "authenticated_cached_output_used_as_compiler_target": True,
        "parent_model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
    }
    # Use the authenticated cache output itself as the folding target.  BF16 is
    # the established terminal compiler evidence dtype; this removes the old
    # FP64 analytical approximation from the target path.
    target_rows = cached_output.to(torch.bfloat16).contiguous().clone()
    return target_rows, {**body, "authentication_sha256": _canonical_sha256(body)}


def _cache_tensor_names(
    reference: Any,
    *,
    final_layer_index: int,
    decode_steps: int,
) -> tuple[str, ...]:
    inventory = set(reference.ordered_tensor_names)
    suffixes = (
        "mlp.input",
        "residual.post_operator",
        "residual.post_layer",
    )
    names = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
    }
    if "input.position_ids" in inventory:
        names.add("input.position_ids")
    phases = ("prefill",) + tuple(f"decode.{step:03d}" for step in range(decode_steps))
    for phase in phases:
        names.add(f"{phase}.logits")
        prefix = f"{phase}.layer.{final_layer_index:03d}."
        names.update(prefix + suffix for suffix in suffixes)
        if phase != "prefill":
            names.add(f"{phase}.forced_token_ids")
    endpoint_names = {
        f"{phase}.output.{field}"
        for phase in phases
        for field in (
            "final_norm_input",
            "final_norm_output",
            "lm_head_input",
        )
    }
    present_endpoint_names = endpoint_names.intersection(inventory)
    if present_endpoint_names and present_endpoint_names != endpoint_names:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Legacy cache has partial or phase-incomplete final RMSNorm evidence."
        )
    names.update(present_endpoint_names)
    missing = sorted(names.difference(inventory))
    if missing:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Legacy cache is missing required final-layer/basic endpoint "
            f"evidence: {missing[:3]}."
        )
    return tuple(sorted(names))


def _validate_cache_trajectory_geometry(
    tensors: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    final_layer_index: int,
    source_hidden_size: int,
) -> Any:
    torch = _torch()
    prompt = tensors["input.input_ids"]
    mask = tensors["input.attention_mask"]
    forced = tensors["input.forced_decode_ids"]
    if (
        tuple(prompt.shape) != tuple(mask.shape)
        or prompt.ndim != 2
        or int(prompt.shape[0]) != 1
        or forced.ndim != 2
        or int(forced.shape[0]) != 1
        or int(forced.shape[1]) <= 0
        or not torch.equal(mask, torch.ones_like(mask))
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cache trajectory is not one complete unpadded causal sequence."
        )
    capture = record.get("capture")
    if not isinstance(capture, Mapping) or capture.get("decode_steps") != int(
        forced.shape[1]
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cache record and forced-decode trajectory geometry differ."
        )
    mode = capture.get("decode_trace_mode")
    if mode not in {
        CACHED_VERIFIED_TRACE_MODE,
        FULL_PREFIX_CANONICAL_TRACE_MODE,
    }:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cache decode trace mode is unsupported."
        )
    if mode == CACHED_VERIFIED_TRACE_MODE and (
        capture.get("all_cached_logits_verified") is not True
        or not (
            capture.get("all_cached_top1_decisions_verified") is True
            or capture.get("all_cached_top1_verified") is True
        )
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cached one-token decode was not authenticated against full-prefix "
            "causal execution."
        )

    observed_decode = sorted(
        int(match.group(1))
        for name in record.get("tensors", {})
        for match in (_DECODE_FINAL_RESIDUAL.fullmatch(str(name)),)
        if match is not None and int(match.group(2)) == final_layer_index
    )
    expected_decode = list(range(int(forced.shape[1])))
    if observed_decode != expected_decode:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Final-layer decode endpoints are missing, duplicated, or reordered."
        )

    residual_rows = []
    phases = ("prefill",) + tuple(f"decode.{step:03d}" for step in expected_decode)
    for phase_index, phase in enumerate(phases):
        prefix = f"{phase}.layer.{final_layer_index:03d}."
        values = tuple(
            tensors[prefix + suffix]
            for suffix in (
                "mlp.input",
                "residual.post_operator",
                "residual.post_layer",
            )
        )
        if (
            any(value.ndim != 3 for value in values)
            or len({tuple(value.shape) for value in values}) != 1
            or int(values[0].shape[0]) != 1
            or int(values[0].shape[2]) != source_hidden_size
            or (phase_index > 0 and int(values[0].shape[1]) != 1)
            or (phase_index == 0 and int(values[0].shape[1]) < 1)
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                f"Cache phase {phase!r} does not expose one aligned final "
                "basic endpoint row."
            )
        if phase_index > 0:
            step = phase_index - 1
            if not torch.equal(
                tensors[f"{phase}.forced_token_ids"],
                forced[:, step : step + 1],
            ):
                raise Lfm2LongHorizonTerminalRefoldError(
                    f"Cache phase {phase!r} forced token binding changed."
                )
        residual_rows.append(values[-1][:, -1, :].reshape(1, -1))
    return torch.cat(residual_rows, dim=0).contiguous()


def _verify_parent_action_projection(
    tensors: Mapping[str, Any],
    parent_final_norm_rows: Any,
    bridge: StrictTokenizerBridgeArtifact,
    source_selected_tied_table: Any,
    *,
    decode_steps: int,
) -> dict[str, Any]:
    """Bind cached parent actions to the selected tied-interface row order."""

    torch = _torch()
    profile = bridge.transport_profile
    mapping = tuple(profile.target_to_donor_row_indices)
    source = source_selected_tied_table
    if (
        not isinstance(source, torch.Tensor)
        or source.ndim != 2
        or source.dtype != torch.bfloat16
        or tuple(source.shape)
        != (profile.target_row_capacity, int(parent_final_norm_rows.shape[1]))
        or len(mapping) != profile.target_row_capacity
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "The authenticated source tied table does not implement the "
            "bridge's target-row action interface."
        )
    donor_rows = torch.tensor(mapping, dtype=torch.long)
    selected_rows = []
    phases = ("prefill",) + tuple(f"decode.{step:03d}" for step in range(decode_steps))
    for phase in phases:
        logits = tensors.get(f"{phase}.logits")
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or int(logits.shape[0]) != 1
            or int(logits.shape[1]) < 1
            or int(logits.shape[2]) != profile.donor_row_capacity
            or not torch.is_floating_point(logits)
            or not bool(torch.isfinite(logits).all().item())
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                f"Cached parent phase {phase!r} has no authenticated full-vocabulary "
                "action row."
            )
        selected_rows.append(
            logits.detach()
            .to(device="cpu")[:, -1, :]
            .index_select(1, donor_rows)
            .reshape(1, -1)
        )
    cached = torch.cat(selected_rows, dim=0).contiguous()
    reconstructed = (
        parent_final_norm_rows.to(device="cpu", dtype=torch.float64)
        @ source.to(device="cpu", dtype=torch.float64).T
    ).contiguous()
    if tuple(cached.shape) != tuple(reconstructed.shape):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cached parent logits and reconstructed selected-row actions are not "
            "phase aligned."
        )
    cached64 = cached.to(torch.float64)
    difference = reconstructed - cached64
    reference_rms = float(cached64.square().mean().sqrt().item())
    relative_rmse = float(difference.square().mean().sqrt().item()) / max(
        reference_rms,
        torch.finfo(torch.float64).tiny,
    )
    top1 = cached64.argmax(dim=1)
    reconstructed_top1 = reconstructed.argmax(dim=1)
    agreements = top1 == reconstructed_top1
    row_error = difference.abs().amax(dim=1)
    row_index = torch.arange(int(cached64.shape[0]), dtype=torch.long)
    cached_gap = cached64[row_index, top1] - cached64[row_index, reconstructed_top1]
    reconstructed_gap = (
        reconstructed[row_index, reconstructed_top1] - reconstructed[row_index, top1]
    )
    tie_envelope = 2.0 * row_error + torch.finfo(torch.float64).eps
    tie_verified = agreements | (
        (cached_gap <= tie_envelope) & (reconstructed_gap <= tie_envelope)
    )
    if (
        not math.isfinite(relative_rmse)
        or relative_rmse > 0.02
        or not bool(tie_verified.all().item())
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cached parent logits do not authenticate the selected tied-table "
            "action projection at every phase endpoint."
        )
    body = {
        "method": "cached-full-logits-selected-by-sealed-bridge-v1",
        "phase_order": list(phases),
        "phase_count": len(phases),
        "target_to_donor_row_indices_sha256": _canonical_sha256(list(mapping)),
        "cached_selected_logits_sha256": _tensor_sha256(cached),
        "reconstructed_selected_logits_sha256": _tensor_sha256(reconstructed),
        "cached_selected_top1_sha256": _tensor_sha256(top1),
        "reconstructed_selected_top1_sha256": _tensor_sha256(reconstructed_top1),
        "exact_selected_top1_agreement": bool(agreements.all().item()),
        "exact_selected_top1_agreement_count": int(agreements.sum().item()),
        "numeric_tie_verified_disagreement_count": int(
            ((~agreements) & tie_verified).sum().item()
        ),
        "all_selected_actions_verified": True,
        "relative_rmse": relative_rmse,
        "maximum_relative_rmse": 0.02,
        "numeric_token_id_equality_required": False,
    }
    return {**body, "authentication_sha256": _canonical_sha256(body)}


def _collect_cached_trajectories(
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    inverse: tuple[int, ...],
    parent_final_norm: Lfm2SealedParentFinalRmsNorm,
    source_selected_tied_table: Any,
    split_plan: Lfm2LongHorizonTrajectorySplitPlan,
) -> tuple[
    tuple[_CachedTrajectory, ...],
    Lfm2StrictTokenizerBridgeTraceAdapter,
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    manifest = cache.manifest
    topology = manifest.get("layer_topology", {}).get("layers")
    if not isinstance(topology, list) or not topology:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cache has no authenticated LFM2 layer topology."
        )
    final_layer_index = len(topology) - 1
    records: list[Lfm2TargetTokenizedRecord] = []
    cached: list[_CachedTrajectory] = []
    parent_batches: list[dict[str, Any]] = []
    payload_receipts: list[dict[str, Any]] = []

    with cache.selective_authenticated_session() as session:
        references = session.batch_references
        if len(references) != cache.batch_count:
            raise Lfm2LongHorizonTerminalRefoldError(
                "Authenticated cache batch references are incomplete."
            )
        for reference in references:
            record = reference.record
            capture = record.get("capture")
            if not isinstance(capture, Mapping):
                raise Lfm2LongHorizonTerminalRefoldError(
                    f"Cache batch {reference.index} has no capture record."
                )
            decode_steps = capture.get("decode_steps")
            if (
                isinstance(decode_steps, bool)
                or not isinstance(decode_steps, int)
                or decode_steps <= 0
            ):
                raise Lfm2LongHorizonTerminalRefoldError(
                    f"Cache batch {reference.index} has no forced horizon."
                )
            names = _cache_tensor_names(
                reference,
                final_layer_index=final_layer_index,
                decode_steps=decode_steps,
            )
            payload = session.load_selected_tensors(reference.index, names)
            try:
                tensors = payload.tensors
                residual = _validate_cache_trajectory_geometry(
                    tensors,
                    record,
                    final_layer_index=final_layer_index,
                    source_hidden_size=parent_final_norm.source_hidden_size,
                )
                parent_rows, norm_endpoint = _verify_parent_final_norm_endpoint(
                    tensors,
                    residual,
                    parent_final_norm,
                    decode_steps=decode_steps,
                )
                parent_action = _verify_parent_action_projection(
                    tensors,
                    parent_rows,
                    bridge,
                    source_selected_tied_table,
                    decode_steps=decode_steps,
                )
                target_prompt = _map_donor_ids_to_unique_target(
                    tensors["input.input_ids"],
                    inverse,
                    name="prompt",
                )
                target_forced = _map_donor_ids_to_unique_target(
                    tensors["input.forced_decode_ids"],
                    inverse,
                    name="forced-decode",
                )
                position_ids = tensors.get("input.position_ids")
                target_record = Lfm2TargetTokenizedRecord(
                    record_id=f"cache-batch-{reference.index:06d}",
                    split=split_plan.assignments[reference.index],
                    input_ids=target_prompt,
                    attention_mask=tensors["input.attention_mask"],
                    forced_decode_ids=target_forced,
                    position_ids=position_ids,
                )
                records.append(target_record)
                cached.append(
                    _CachedTrajectory(
                        batch_index=reference.index,
                        input_sha256=reference.input_sha256,
                        split=split_plan.assignments[reference.index],
                        target_record=target_record,
                        semantic_bridge=None,
                        parent_final_norm_rows=parent_rows,
                        horizon_indices=tuple(range(decode_steps + 1)),
                        selected_payload_sha256=payload.selection_sha256,
                        parent_final_norm_endpoint=MappingProxyType(norm_endpoint),
                        parent_action_projection=MappingProxyType(parent_action),
                    )
                )
                parent_batch = {
                    "input_ids": tensors["input.input_ids"].clone(),
                    "attention_mask": tensors["input.attention_mask"].clone(),
                    "forced_decode_ids": tensors["input.forced_decode_ids"].clone(),
                }
                if position_ids is not None:
                    parent_batch["position_ids"] = position_ids.clone()
                parent_batches.append(parent_batch)
                payload_receipts.append(
                    {
                        "batch_index": reference.index,
                        "input_sha256": reference.input_sha256,
                        "selection_sha256": payload.selection_sha256,
                        "selected_tensor_count": len(names),
                        "selected_logical_bytes": payload.logical_bytes,
                    }
                )
            finally:
                session.release_selected_tensors(payload)
        session.verify()
        cache_receipt = session.as_dict()

    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        tuple(records),
        expected_artifact_sha256=bridge.artifact_sha256,
    )
    pair_receipts = []
    for trajectory, parent_batch, aligned in zip(
        cached,
        parent_batches,
        adapter.records,
        strict=True,
    ):
        if (
            aligned.ordinal != trajectory.batch_index
            or aligned.donor_cache_input_sha256 != trajectory.input_sha256
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Tokenizer adapter does not bind to the authenticated cache input."
            )
        paired = authenticate_lfm2_bridge_input_pair(
            adapter,
            record_id=aligned.record_id,
            parent_cache_batch=parent_batch,
            child_replay_batch=aligned.as_child_replay_batch(),
        )
        pair_receipts.append(paired.to_dict())
    adapter.verify()
    return (
        tuple(cached),
        adapter,
        tuple(pair_receipts),
        {
            "session": cache_receipt,
            "selected_payloads": payload_receipts,
        },
    )


def _collect_cached_semantic_trajectories(
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    parent_final_norm: Lfm2SealedParentFinalRmsNorm,
    source_selected_tied_table: Any,
    split_plan: Lfm2LongHorizonTrajectorySplitPlan,
) -> tuple[
    tuple[_CachedTrajectory, ...],
    tuple[Lfm2SemanticSpanPhaseBridge, ...],
    dict[str, Any],
]:
    """Authenticate complete cache batches and retokenize every parent phase."""

    try:
        bridge.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    except StrictTokenizerBridgeError as exc:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Live semantic tokenizers differ from the sealed child bridge."
        ) from exc
    manifest = cache.manifest
    topology = manifest.get("layer_topology", {}).get("layers")
    if not isinstance(topology, list) or not topology:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Cache has no authenticated LFM2 layer topology."
        )
    final_layer_index = len(topology) - 1
    semantic_bridges: list[Lfm2SemanticSpanPhaseBridge] = []
    cached: list[_CachedTrajectory] = []
    payload_receipts: list[dict[str, Any]] = []
    session = cache.authenticated_session()
    for batch in session.iter_batches():
        record = batch.record
        capture = record.get("capture")
        decode_steps = (
            None if not isinstance(capture, Mapping) else capture.get("decode_steps")
        )
        if (
            isinstance(decode_steps, bool)
            or not isinstance(decode_steps, int)
            or decode_steps <= 0
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                f"Cache batch {batch.index} has no forced horizon."
            )
        tensors = batch.tensors
        residual = _validate_cache_trajectory_geometry(
            tensors,
            record,
            final_layer_index=final_layer_index,
            source_hidden_size=parent_final_norm.source_hidden_size,
        )
        parent_rows, norm_endpoint = _verify_parent_final_norm_endpoint(
            tensors,
            residual,
            parent_final_norm,
            decode_steps=decode_steps,
        )
        parent_action = _verify_parent_action_projection(
            tensors,
            parent_rows,
            bridge,
            source_selected_tied_table,
            decode_steps=decode_steps,
        )
        try:
            semantic = build_lfm2_semantic_span_phase_bridge(
                batch,
                manifest,
                donor_tokenizer,
                target_tokenizer,
                expected_source_cache_key=cache.cache_key,
                expected_source_manifest_sha256=manifest["manifest_sha256"],
                expected_source_input_sha256=batch.input_sha256,
                expected_donor_tokenizer_fingerprint=(
                    bridge.donor_tokenizer_semantics_sha256
                ),
                expected_target_tokenizer_fingerprint=(
                    bridge.target_tokenizer_semantics_sha256
                ),
            )
        except Lfm2SemanticSpanPhaseBridgeError as exc:
            raise Lfm2LongHorizonTerminalRefoldError(
                f"Cache batch {batch.index} failed semantic-span authentication."
            ) from exc
        if (
            semantic.source_batch_index != batch.index
            or semantic.source_input_sha256 != batch.input_sha256
            or len(semantic.phases) != decode_steps + 1
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Semantic phase endpoints differ from the authenticated parent "
                "trajectory."
            )
        semantic_bridges.append(semantic)
        cached.append(
            _CachedTrajectory(
                batch_index=batch.index,
                input_sha256=batch.input_sha256,
                split=split_plan.assignments[batch.index],
                target_record=None,
                semantic_bridge=semantic,
                parent_final_norm_rows=parent_rows,
                horizon_indices=tuple(range(decode_steps + 1)),
                selected_payload_sha256=semantic.source_record_sha256,
                parent_final_norm_endpoint=MappingProxyType(norm_endpoint),
                parent_action_projection=MappingProxyType(parent_action),
            )
        )
        payload_receipts.append(
            {
                "batch_index": batch.index,
                "input_sha256": batch.input_sha256,
                "record_sha256": semantic.source_record_sha256,
                "semantic_bridge_sha256": semantic.bridge_sha256,
                "authenticated_tensor_count": int(record["tensor_count"]),
                "authenticated_logical_bytes": int(capture["capture_bytes"]),
            }
        )
    session.verify()
    return (
        tuple(cached),
        tuple(semantic_bridges),
        {
            "session": session.as_dict(),
            "selected_payloads": payload_receipts,
        },
    )


def _layer_runtime(layer: Any) -> tuple[Any, Any]:
    torch = _torch()
    parameters = tuple(layer.parameters())
    if (
        not parameters
        or any(
            parameter.device.type == "meta"
            or parameter.dtype != torch.bfloat16
            or not torch.is_floating_point(parameter)
            for parameter in parameters
        )
        or len({parameter.device for parameter in parameters}) != 1
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Every target macrocell must use one materialized BF16 device."
        )
    return parameters[0].device, parameters[0].dtype


def _additive_causal_mask(attention_mask: Any, residual: Any) -> Any:
    torch = _torch()
    sequence = int(attention_mask.shape[1])
    allowed = torch.tril(
        torch.ones(
            (sequence, sequence),
            device=residual.device,
            dtype=torch.bool,
        )
    )
    active_keys = attention_mask.to(
        device=residual.device,
        dtype=torch.bool,
    )[:, None, None, :]
    allowed = allowed[None, None, :, :] & active_keys
    minimum = torch.finfo(residual.dtype).min
    return torch.where(
        allowed,
        torch.zeros((), device=residual.device, dtype=residual.dtype),
        torch.full((), minimum, device=residual.device, dtype=residual.dtype),
    )


def _full_target_trajectory(record: Any) -> tuple[Any, Any, Any, int]:
    torch = _torch()
    prompt = record.target_input_ids
    forced = record.target_forced_decode_ids
    prompt_mask = record.attention_mask
    if not torch.equal(prompt_mask, torch.ones_like(prompt_mask)):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Long-horizon child replay currently requires the cache's unpadded "
            "one-sequence geometry."
        )
    full_ids = torch.cat((prompt, forced), dim=1).to(torch.long).contiguous()
    full_mask = torch.ones_like(full_ids, dtype=torch.long)
    positions = record.position_ids
    if positions is None:
        prompt_positions = (
            torch.arange(int(prompt.shape[1]), dtype=torch.long)
            .unsqueeze(0)
            .expand(int(prompt.shape[0]), -1)
        )
    else:
        prompt_positions = positions.to(torch.long)
    start = prompt_positions[:, -1:] + 1
    offsets = torch.arange(int(forced.shape[1]), dtype=torch.long).unsqueeze(0)
    full_positions = torch.cat(
        (prompt_positions, start + offsets),
        dim=1,
    ).contiguous()
    return full_ids, full_mask, full_positions, int(prompt.shape[1])


def _replay_child_sequence_once(
    target_model: Any,
    layers: tuple[Any, ...],
    full_ids: Any,
    full_mask: Any,
    positions: Any,
    row_indices: Sequence[int],
    *,
    replay_strategy: str,
) -> _ChildTrajectoryRows:
    torch = _torch()
    if (
        not isinstance(full_ids, torch.Tensor)
        or not isinstance(full_mask, torch.Tensor)
        or not isinstance(positions, torch.Tensor)
        or full_ids.ndim != 2
        or tuple(full_mask.shape) != tuple(full_ids.shape)
        or tuple(positions.shape) != tuple(full_ids.shape)
        or int(full_ids.shape[0]) != 1
        or int(full_ids.shape[1]) <= 0
        or not torch.equal(full_mask, torch.ones_like(full_mask))
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Target-native replay requires one complete unpadded token sequence."
        )
    selected_rows = tuple(int(value) for value in row_indices)
    if not selected_rows or any(
        value < 0 or value >= int(full_ids.shape[1]) for value in selected_rows
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Target-native replay endpoint rows lie outside the token sequence."
        )
    embedding = getattr(target_model.get_input_embeddings(), "weight", None)
    if not isinstance(embedding, torch.nn.Parameter):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Target input embedding is not materialized."
        )
    if bool((full_ids >= int(embedding.shape[0])).any().item()):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Mapped target trajectory exceeds the installed tied table."
        )
    with torch.no_grad():
        ids = full_ids.to(device=embedding.device)
        residual = (
            embedding.index_select(0, ids.reshape(-1))
            .reshape(
                int(full_ids.shape[0]),
                int(full_ids.shape[1]),
                int(embedding.shape[1]),
            )
            .detach()
        )
    replay_sha = []
    final_mlp = None
    final_pre_mlp = None
    rope_calls = 0
    for layer_index, layer in enumerate(layers):
        device, dtype = _layer_runtime(layer)
        residual = residual.to(device=device, dtype=dtype, non_blocking=False)
        if bool(getattr(layer, "is_attention_layer", False)):
            position_embeddings = _rope_from_target(
                target_model,
                residual,
                positions,
            )
            causal_mask = _additive_causal_mask(full_mask, residual)
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                position_embeddings=position_embeddings,
                additive_causal_mask=causal_mask,
            )
            rope_calls += 1
        else:
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                padding_mask=full_mask.to(device=device, dtype=dtype),
            )
        replay_sha.append(replay.replay_sha256)
        if layer_index == len(layers) - 1:
            index = torch.tensor(
                selected_rows,
                dtype=torch.long,
            )
            final_mlp = replay.mlp_input.index_select(1, index)[0]
            final_pre_mlp = replay.post_operator_residual.index_select(
                1,
                index,
            )[0]
        residual = replay.post_layer_residual
    if final_mlp is None or final_pre_mlp is None:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Complete child macrocell replay produced no final-layer endpoints."
        )
    if (
        final_mlp.dtype != torch.bfloat16
        or final_pre_mlp.dtype != torch.bfloat16
        or tuple(final_mlp.shape) != tuple(final_pre_mlp.shape)
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Child final-layer endpoint rows are not aligned exact BF16 tensors."
        )
    return _ChildTrajectoryRows(
        mlp_input=final_mlp.contiguous().clone(),
        pre_mlp_residual=final_pre_mlp.contiguous().clone(),
        macrocell_replay_sha256=tuple(replay_sha),
        attention_rope_materializations=rope_calls,
        replay_strategy=replay_strategy,
    )


def _replay_child_trajectory_once(
    target_model: Any,
    layers: tuple[Any, ...],
    aligned_record: Any,
) -> _ChildTrajectoryRows:
    full_ids, full_mask, positions, prompt_width = _full_target_trajectory(
        aligned_record
    )
    return _replay_child_sequence_once(
        target_model,
        layers,
        full_ids,
        full_mask,
        positions,
        (prompt_width - 1, *range(prompt_width, int(full_ids.shape[1]))),
        replay_strategy="strict-one-row-single-causal-trajectory-v1",
    )


def _is_token_prefix(prefix: tuple[int, ...], value: tuple[int, ...]) -> bool:
    return len(prefix) <= len(value) and value[: len(prefix)] == prefix


def _replay_child_semantic_span_once(
    target_model: Any,
    layers: tuple[Any, ...],
    semantic: Lfm2SemanticSpanPhaseBridge,
) -> _ChildTrajectoryRows:
    """Replay each maximal target-native branch once and collect phase endpoints."""

    torch = _torch()
    phase_sequences = tuple(
        phase.target_full_prefix_token_ids for phase in semantic.phases
    )
    unique_sequences = tuple(dict.fromkeys(phase_sequences))
    maxima = tuple(
        sequence
        for sequence in unique_sequences
        if not any(
            len(other) > len(sequence) and _is_token_prefix(sequence, other)
            for other in unique_sequences
        )
    )
    if not maxima:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Semantic phase schedule has no complete target-native trajectory."
        )
    assignments: list[int] = []
    for sequence in phase_sequences:
        matches = tuple(
            index
            for index, maximum in enumerate(maxima)
            if _is_token_prefix(sequence, maximum)
        )
        if not matches:
            raise Lfm2LongHorizonTerminalRefoldError(
                "Semantic phase endpoint is not represented by a complete "
                "target-native trajectory."
            )
        assignments.append(matches[0])

    mlp_rows: list[Any | None] = [None] * len(semantic.phases)
    residual_rows: list[Any | None] = [None] * len(semantic.phases)
    replay_sha: list[str] = []
    rope_calls = 0
    branch_phase_sha: list[tuple[str, ...]] = []
    for branch_index, sequence in enumerate(maxima):
        phase_indices = tuple(
            index
            for index, assignment in enumerate(assignments)
            if assignment == branch_index
        )
        ids = torch.tensor([sequence], dtype=torch.long)
        mask = torch.ones_like(ids, dtype=torch.long)
        positions = torch.arange(
            semantic.target_position_origin,
            semantic.target_position_origin + len(sequence),
            dtype=torch.long,
        ).unsqueeze(0)
        replay = _replay_child_sequence_once(
            target_model,
            layers,
            ids,
            mask,
            positions,
            tuple(len(phase_sequences[index]) - 1 for index in phase_indices),
            replay_strategy="semantic-maximal-target-prefix-branch-v1",
        )
        replay_sha.extend(replay.macrocell_replay_sha256)
        rope_calls += replay.attention_rope_materializations
        branch_phase_sha.append(
            tuple(semantic.phases[index].phase_sha256 for index in phase_indices)
        )
        for row, phase_index in enumerate(phase_indices):
            mlp_rows[phase_index] = replay.mlp_input[row : row + 1]
            residual_rows[phase_index] = replay.pre_mlp_residual[row : row + 1]
    if any(value is None for value in (*mlp_rows, *residual_rows)):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Semantic target-native replay omitted a parent phase endpoint."
        )
    return _ChildTrajectoryRows(
        mlp_input=torch.cat(tuple(mlp_rows), dim=0),
        pre_mlp_residual=torch.cat(tuple(residual_rows), dim=0),
        macrocell_replay_sha256=tuple(replay_sha),
        attention_rope_materializations=rope_calls,
        embedding_seed_lookup_calls=len(maxima),
        full_prefix_replay_calls=len(maxima),
        replay_strategy="semantic-maximal-target-prefix-branch-cover-v1",
        replay_branch_phase_sha256=tuple(branch_phase_sha),
    )


def _build_split_evidence(
    cached: tuple[_CachedTrajectory, ...],
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter | None,
    target_model: Any,
    layers: tuple[Any, ...],
) -> tuple[
    tuple[Lfm2CpcfTerminalActionEvidence, ...],
    dict[str, tuple[int, ...]],
    tuple[dict[str, Any], ...],
    int,
]:
    torch = _torch()
    rows: dict[str, list[dict[str, Any]]] = {split: [] for split in _SPLITS}
    replay_receipts = []
    rope_calls = 0
    for trajectory in cached:
        if trajectory.target_record is not None:
            if adapter is None or trajectory.semantic_bridge is not None:
                raise Lfm2LongHorizonTerminalRefoldError(
                    "Strict target trajectory lost its authenticated adapter."
                )
            aligned = adapter.records[trajectory.batch_index]
            child = _replay_child_trajectory_once(target_model, layers, aligned)
            record_id = aligned.record_id
            semantic_sha256 = None
        else:
            semantic = trajectory.semantic_bridge
            if semantic is None or adapter is not None:
                raise Lfm2LongHorizonTerminalRefoldError(
                    "Semantic target trajectory has an ambiguous replay adapter."
                )
            child = _replay_child_semantic_span_once(
                target_model,
                layers,
                semantic,
            )
            record_id = f"semantic-cache-batch-{trajectory.batch_index:06d}"
            semantic_sha256 = semantic.bridge_sha256
        rope_calls += child.attention_rope_materializations
        expected_rows = len(trajectory.horizon_indices)
        if (
            int(child.mlp_input.shape[0]) != expected_rows
            or int(trajectory.parent_final_norm_rows.shape[0]) != expected_rows
        ):
            raise Lfm2LongHorizonTerminalRefoldError(
                "Parent and child complete-trajectory terminal row counts differ."
            )
        row_ids = tuple(
            f"{trajectory.input_sha256}:batch-{trajectory.batch_index:06d}:"
            f"horizon-{horizon:05d}"
            for horizon in trajectory.horizon_indices
        )
        rows[trajectory.split].append(
            {
                "row_ids": row_ids,
                "horizons": trajectory.horizon_indices,
                "mlp_input": child.mlp_input,
                "pre_mlp_residual": child.pre_mlp_residual,
                "parent_final_norm_output": (trajectory.parent_final_norm_rows),
            }
        )
        replay_receipts.append(
            {
                "batch_index": trajectory.batch_index,
                "record_id": record_id,
                "split": trajectory.split,
                "trajectory_rows": expected_rows,
                "horizon_minimum": min(trajectory.horizon_indices),
                "horizon_maximum": max(trajectory.horizon_indices),
                "embedding_seed_lookup_calls": child.embedding_seed_lookup_calls,
                "full_prefix_replay_calls": child.full_prefix_replay_calls,
                "macrocell_replay_calls": len(child.macrocell_replay_sha256),
                "attention_rope_materializations": (
                    child.attention_rope_materializations
                ),
                "replay_strategy": child.replay_strategy,
                "semantic_bridge_sha256": semantic_sha256,
                "replay_branch_phase_sha256": [
                    list(value) for value in child.replay_branch_phase_sha256
                ],
                "parent_action_projection_sha256": (
                    trajectory.parent_action_projection["authentication_sha256"]
                ),
                "parent_final_norm_endpoint_sha256": (
                    trajectory.parent_final_norm_endpoint["authentication_sha256"]
                ),
                "macrocell_replay_sha256": list(child.macrocell_replay_sha256),
            }
        )

    evidence = []
    horizons_by_split: dict[str, tuple[int, ...]] = {}
    for split in _SPLITS:
        trajectories = rows[split]
        if not trajectories:
            raise Lfm2LongHorizonTerminalRefoldError(
                f"Held-out split {split!r} has no complete trajectory."
            )
        row_ids = tuple(
            row_id for trajectory in trajectories for row_id in trajectory["row_ids"]
        )
        horizons = tuple(
            horizon for trajectory in trajectories for horizon in trajectory["horizons"]
        )
        weights = torch.cat(
            tuple(
                torch.full(
                    (len(trajectory["row_ids"]),),
                    1.0 / (len(trajectories) * len(trajectory["row_ids"])),
                    dtype=torch.float64,
                )
                for trajectory in trajectories
            )
        )
        bank = Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=row_ids,
            mlp_input=torch.cat(
                tuple(value["mlp_input"] for value in trajectories),
                dim=0,
            ),
            pre_mlp_residual=torch.cat(
                tuple(value["pre_mlp_residual"] for value in trajectories),
                dim=0,
            ),
            parent_final_norm_output=torch.cat(
                tuple(value["parent_final_norm_output"] for value in trajectories),
                dim=0,
            ),
            sample_weights=weights,
        )
        evidence.append(bank)
        horizons_by_split[split] = horizons
    return tuple(evidence), horizons_by_split, tuple(replay_receipts), rope_calls


def _evidence_subset(
    evidence: Lfm2CpcfTerminalActionEvidence,
    indices: Sequence[int],
) -> Lfm2CpcfTerminalActionEvidence:
    torch = _torch()
    index = torch.tensor(tuple(indices), dtype=torch.long)
    return Lfm2CpcfTerminalActionEvidence(
        split=evidence.split,
        row_ids=tuple(evidence.row_ids[value] for value in indices),
        mlp_input=evidence.mlp_input.index_select(0, index),
        pre_mlp_residual=evidence.pre_mlp_residual.index_select(0, index),
        parent_final_norm_output=evidence.parent_final_norm_output.index_select(
            0,
            index,
        ),
        sample_weights=evidence.sample_weights.index_select(0, index),
    )


def _combined_held_out_evidence(
    evidence: Mapping[str, Lfm2CpcfTerminalActionEvidence],
    horizons: Mapping[str, tuple[int, ...]],
) -> tuple[Lfm2CpcfTerminalActionEvidence, tuple[int, ...]]:
    torch = _torch()
    selected = (evidence["selection"], evidence["validation"])
    return (
        Lfm2CpcfTerminalActionEvidence(
            split="validation",
            row_ids=tuple(row for bank in selected for row in bank.row_ids),
            mlp_input=torch.cat(tuple(bank.mlp_input for bank in selected), dim=0),
            pre_mlp_residual=torch.cat(
                tuple(bank.pre_mlp_residual for bank in selected),
                dim=0,
            ),
            parent_final_norm_output=torch.cat(
                tuple(bank.parent_final_norm_output for bank in selected),
                dim=0,
            ),
            sample_weights=torch.cat(
                tuple(bank.sample_weights * 0.5 for bank in selected),
                dim=0,
            ),
        ),
        horizons["selection"] + horizons["validation"],
    )


def _horizon_metrics(
    evidence: Mapping[str, Lfm2CpcfTerminalActionEvidence],
    horizons: Mapping[str, tuple[int, ...]],
    seed: Lfm2CpcfTerminalActionSeed,
    terminal: Lfm2CpcfTerminalActionResidualResult,
    policy: Lfm2CpcfTerminalActionPolicy,
) -> dict[str, Any]:
    held_out, held_out_horizons = _combined_held_out_evidence(
        evidence,
        horizons,
    )
    result = {}
    for label, lower, upper in _HORIZON_BUCKETS:
        indices = tuple(
            index
            for index, horizon in enumerate(held_out_horizons)
            if horizon >= lower and (upper is None or horizon <= upper)
        )
        if not indices:
            result[label] = {
                "available": False,
                "rows": 0,
                "incumbent": None,
                "selected": None,
            }
            continue
        subset = _evidence_subset(held_out, indices)
        prepared = _prepare_split(subset, seed)
        incumbent = _evaluate_split(
            prepared,
            seed,
            terminal.incumbent_down_weight,
            metric_chunk_rows=policy.metric_chunk_rows,
        )
        selected = _evaluate_split(
            prepared,
            seed,
            terminal.selected_down_weight,
            metric_chunk_rows=policy.metric_chunk_rows,
        )
        result[label] = {
            "available": True,
            "rows": subset.rows,
            "row_ids_sha256": _canonical_sha256(list(subset.row_ids)),
            "incumbent": incumbent.as_dict(),
            "selected": selected.as_dict(),
        }
    return {
        "scope": "held_out_selection_plus_validation",
        "horizon_zero_is_prompt_endpoint": True,
        "forced_decode_horizon_starts_at_one": True,
        "selected_is_frozen_prepublication_candidate": True,
        "published_state": (
            "selected" if terminal.accepted and terminal.changed else "incumbent"
        ),
        "buckets": result,
    }


def _parameter_identity(parameter: Any) -> dict[str, Any]:
    return {
        "python_id": id(parameter),
        "data_ptr": int(parameter.data_ptr()),
        "shape": [int(size) for size in parameter.shape],
        "dtype": _dtype_name(parameter),
        "device": str(parameter.device),
        "requires_grad": bool(parameter.requires_grad),
    }


def _unique_parameter_count(model: Any) -> tuple[int, int]:
    seen: set[int] = set()
    tensors = 0
    elements = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        tensors += 1
        elements += int(parameter.numel())
    return tensors, elements


def _install_if_accepted(
    parameter: Any,
    terminal: Lfm2CpcfTerminalActionResidualResult,
) -> dict[str, Any]:
    torch = _torch()
    identity = _parameter_identity(parameter)
    incumbent = parameter.detach().to(device="cpu").contiguous().clone()
    if not torch.equal(incumbent, terminal.incumbent_down_weight):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Live final writer changed after the authenticated terminal seed."
        )
    install = bool(terminal.accepted and terminal.changed)
    rollback_performed = False
    if install:
        try:
            with torch.no_grad():
                parameter.copy_(
                    terminal.published_down_weight.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
            if _parameter_identity(parameter) != identity or not torch.equal(
                parameter.detach().to(device="cpu"),
                terminal.published_down_weight,
            ):
                raise Lfm2LongHorizonTerminalRefoldError(
                    "Installed terminal writer changed Parameter identity or bytes."
                )
        except BaseException as exc:
            rollback_performed = True
            with torch.no_grad():
                parameter.copy_(incumbent.to(device=parameter.device))
            rollback_exact = bool(
                _parameter_identity(parameter) == identity
                and torch.equal(parameter.detach().to(device="cpu"), incumbent)
            )
            if not rollback_exact:
                raise Lfm2LongHorizonTerminalRefoldError(
                    "Failed terminal installation could not restore the exact "
                    "incumbent writer."
                ) from exc
            raise Lfm2LongHorizonTerminalRefoldError(
                "Transactional terminal writer installation failed and was "
                "rolled back exactly."
            ) from exc
    elif _parameter_identity(parameter) != identity or not torch.equal(
        parameter.detach().to(device="cpu"), incumbent
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Rejected terminal selection mutated the live child writer."
        )
    return {
        "primitive_accepted": terminal.accepted,
        "primitive_changed": terminal.changed,
        "installed": install,
        "parameter_identity_before": identity,
        "parameter_identity_preserved": _parameter_identity(parameter) == identity,
        "incumbent_sha256": _tensor_sha256(incumbent),
        "published_sha256": _tensor_sha256(parameter.detach().to(device="cpu")),
        "rollback_performed": rollback_performed,
        "new_parameter_created": False,
    }


def compile_install_lfm2_long_horizon_terminal_refold(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    source_selected_tied_table: Any,
    parent_final_norm: (Lfm2SealedParentFinalRmsNorm | Lfm2LocalParentFinalRmsNormLoad),
    split_plan: Lfm2LongHorizonTrajectorySplitPlan | None = None,
    policy: Lfm2CpcfTerminalActionPolicy | None = None,
    tokenizer_replay_mode: TokenizerReplayMode = "strict_one_row",
    donor_tokenizer: Any | None = None,
    target_tokenizer: Any | None = None,
) -> Lfm2LongHorizonTerminalRefoldResult:
    """Compile and conditionally install one long-horizon terminal refold."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if isinstance(parent_final_norm, Lfm2LocalParentFinalRmsNormLoad):
        selected_parent_final_norm = parent_final_norm.final_norm
        parent_final_norm_source = parent_final_norm.as_dict()
    elif isinstance(parent_final_norm, Lfm2SealedParentFinalRmsNorm):
        selected_parent_final_norm = parent_final_norm
        parent_final_norm_source = None
    else:
        raise TypeError(
            "parent_final_norm must be a sealed norm or authenticated direct-local "
            "load."
        )
    _verify_parent_norm_binding(cache, selected_parent_final_norm)
    interface_report = _verify_interface_report(
        interface,
        source_selected_tied_table,
    )
    selected_policy = policy or Lfm2CpcfTerminalActionPolicy()
    if (
        not isinstance(selected_policy, Lfm2CpcfTerminalActionPolicy)
        or not selected_policy.verify_authentication()
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "policy must be an authenticated terminal-action policy."
        )
    plan = split_plan or (
        Lfm2LongHorizonTrajectorySplitPlan.deterministic_round_robin(cache)
    )
    if not isinstance(plan, Lfm2LongHorizonTrajectorySplitPlan):
        raise TypeError("split_plan must be Lfm2LongHorizonTrajectorySplitPlan.")
    plan.verify_for(cache)

    if tokenizer_replay_mode not in {"strict_one_row", "semantic_span"}:
        raise Lfm2LongHorizonTerminalRefoldError(
            "tokenizer_replay_mode must be strict_one_row or semantic_span."
        )
    if tokenizer_replay_mode == "semantic_span":
        if donor_tokenizer is None or target_tokenizer is None:
            raise Lfm2LongHorizonTerminalRefoldError(
                "semantic_span replay requires direct-local donor and target "
                "tokenizers."
            )
    elif donor_tokenizer is not None or target_tokenizer is not None:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Direct-local tokenizers are accepted only by explicit semantic_span "
            "replay."
        )

    seed, down_parameter, layers, target_norm_epsilon = _target_runtime_seed(
        target_model,
        interface,
        source_selected_tied_table,
    )
    child_before = fingerprint_lfm2_parent(target_model)
    parameter_count_before = _unique_parameter_count(target_model)
    semantic_bridges: tuple[Lfm2SemanticSpanPhaseBridge, ...] = ()
    if tokenizer_replay_mode == "strict_one_row":
        inverse = _strict_inverse_mapping(bridge, seed.vocabulary_size)
        cached, adapter, pair_receipts, cache_receipt = _collect_cached_trajectories(
            cache,
            bridge,
            inverse,
            selected_parent_final_norm,
            source_selected_tied_table,
            plan,
        )
    else:
        if bridge.target_vocabulary_size != seed.vocabulary_size:
            raise Lfm2LongHorizonTerminalRefoldError(
                "Semantic target tokenizer capacity differs from the installed "
                "tied child interface."
            )
        cached, semantic_bridges, cache_receipt = _collect_cached_semantic_trajectories(
            cache,
            bridge,
            donor_tokenizer,
            target_tokenizer,
            selected_parent_final_norm,
            source_selected_tied_table,
            plan,
        )
        adapter = None
        pair_receipts = ()
    authenticated_norm_endpoints = all(
        value.parent_final_norm_endpoint.get("runtime_replay_to_cached_output_verified")
        is True
        for value in cached
    )
    evidence_tuple, horizons, replay_receipts, rope_calls = _build_split_evidence(
        cached,
        adapter,
        target_model,
        layers,
    )
    evidence = {value.split: value for value in evidence_tuple}
    child_after_replay = fingerprint_lfm2_parent(target_model)
    if child_after_replay.identity_sha256 != child_before.identity_sha256:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Child state changed during no-forward macrocell replay."
        )

    terminal = compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=evidence["fit"],
        fit_check_evidence=evidence["fit_check"],
        selection_evidence=evidence["selection"],
        validation_evidence=evidence["validation"],
        policy=selected_policy,
    )
    if not terminal.verify_authentication():
        raise Lfm2LongHorizonTerminalRefoldError(
            "Closed-form terminal-action result failed authentication."
        )
    horizon_report = _horizon_metrics(
        evidence,
        horizons,
        seed,
        terminal,
        selected_policy,
    )
    installation = _install_if_accepted(down_parameter, terminal)
    child_after_install = fingerprint_lfm2_parent(target_model)
    parameter_count_after = _unique_parameter_count(target_model)
    accepted = bool(installation["installed"])
    if (
        parameter_count_after != parameter_count_before
        or (
            accepted
            and child_after_install.identity_sha256 == child_before.identity_sha256
        )
        or (
            not accepted
            and child_after_install.identity_sha256 != child_before.identity_sha256
        )
    ):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Terminal publication changed child topology or contradicted its "
            "installation receipt."
        )

    macrocell_calls = sum(
        int(value["full_prefix_replay_calls"]) * len(layers)
        for value in replay_receipts
    )
    observed_macrocell_calls = sum(
        int(value["macrocell_replay_calls"]) for value in replay_receipts
    )
    if observed_macrocell_calls != macrocell_calls:
        raise Lfm2LongHorizonTerminalRefoldError(
            "Macrocell replay audit count is inconsistent."
        )
    body = {
        "schema": LFM2_LONG_HORIZON_TERMINAL_REFOLD_SCHEMA,
        "method": (
            "sealed_cache_runtime_verified_parent_final_norm_endpoint_plus_"
            + (
                "one_pass_complete_child_macrocell_trajectory"
                if tokenizer_replay_mode == "strict_one_row"
                else "semantic_maximal_target_prefix_branch_cover"
            )
            + "_plus_closed_form_terminal_writer_residual"
        ),
        "tokenizer_replay_mode": tokenizer_replay_mode,
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "cache_layer_topology_fingerprint": cache.layer_topology_fingerprint,
        "cache_authentication_sha256": cache_receipt["session"]["session_sha256"],
        "cache_authentication": cache_receipt,
        "parent_final_norm": selected_parent_final_norm.as_dict(),
        "parent_final_norm_direct_local_source": parent_final_norm_source,
        "parent_final_norm_reconstruction": {
            "method": (
                "direct_local_runtime_equivalent_rmsnorm_replay_then_"
                "authenticated_cached_output"
            ),
            "final_norm_module_or_parent_model_called": False,
            "reconstructed_output_dtype": "bfloat16",
            "prompt_endpoint_and_every_forced_position": True,
            "cached_final_residual_to_norm_input_exact": (authenticated_norm_endpoints),
            "cached_final_norm_output_to_lm_head_input_exact": (
                authenticated_norm_endpoints
            ),
            "authenticated_cached_output_used_as_compiler_target": (
                authenticated_norm_endpoints
            ),
            "legacy_cache_exact_parent_identity_fallback_used": (
                not authenticated_norm_endpoints
            ),
        },
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_transport_profile_fingerprint": bridge.profile_fingerprint,
        "bridge_trace_adapter_sha256": (
            None if adapter is None else adapter.evidence_sha256
        ),
        "semantic_span_bridge_sha256": [
            value.bridge_sha256 for value in semantic_bridges
        ],
        "semantic_span_bridges": [
            value.to_evidence_dict() for value in semantic_bridges
        ],
        "bridge_pair_authentication_sha256": [
            value["evidence_sha256"] for value in pair_receipts
        ],
        "bridge_inverse_contract": {
            "every_cached_donor_prompt_and_forced_id_covered": (
                tokenizer_replay_mode == "strict_one_row"
            ),
            "every_used_donor_row_has_exactly_one_target_row": (
                tokenizer_replay_mode == "strict_one_row"
            ),
            "exact_full_prefix_text_roundtrip_used": (
                tokenizer_replay_mode == "semantic_span"
            ),
            "suffix_rewrite_supported": tokenizer_replay_mode == "semantic_span",
            "numeric_token_id_equality_required": False,
        },
        "parent_action_projection_sha256": [
            value.parent_action_projection["authentication_sha256"] for value in cached
        ],
        "parent_final_norm_endpoint_sha256": [
            value.parent_final_norm_endpoint["authentication_sha256"]
            for value in cached
        ],
        "split_plan": plan.as_dict(),
        "split_evidence_sha256": {
            value.split: value.authentication_sha256 for value in evidence_tuple
        },
        "split_rows": {value.split: value.rows for value in evidence_tuple},
        "split_horizon_maximum": {
            split: max(values) for split, values in horizons.items()
        },
        "whole_trajectory_split_isolation": True,
        "interface_report_sha256": interface_report["report_sha256"],
        "terminal_seed_sha256": seed.authentication_sha256,
        "target_final_norm_epsilon": target_norm_epsilon,
        "terminal_result_sha256": terminal.result_sha256,
        "terminal_result": terminal.as_dict(),
        "horizon_metrics": horizon_report,
        "trajectory_replay_receipts": list(replay_receipts),
        "child_state": {
            "before_sha256": child_before.identity_sha256,
            "after_replay_sha256": child_after_replay.identity_sha256,
            "after_install_sha256": child_after_install.identity_sha256,
            "replay_state_unchanged": True,
            "parameter_tensor_count_before": parameter_count_before[0],
            "parameter_tensor_count_after": parameter_count_after[0],
            "parameter_element_count_before": parameter_count_before[1],
            "parameter_element_count_after": parameter_count_after[1],
            "parameter_count_changed": False,
        },
        "installation": installation,
        "accepted": accepted,
        "operation_audit": {
            "parent_model_forward_calls": 0,
            "parent_model_generate_calls": 0,
            "parent_final_norm_module_calls": 0,
            "parent_cache_generation_calls": 0,
            "parent_trace_recaptures": 0,
            "authenticated_cache_full_batch_loads": (
                cache.batch_count if tokenizer_replay_mode == "semantic_span" else 0
            ),
            "authenticated_cache_selected_payload_loads": (
                cache.batch_count if tokenizer_replay_mode == "strict_one_row" else 0
            ),
            "parent_cached_action_rows_authenticated": sum(
                len(value.horizon_indices) for value in cached
            ),
            "child_model_forward_calls": 0,
            "child_model_generate_calls": 0,
            "child_embedding_seed_lookups": sum(
                int(value["embedding_seed_lookup_calls"]) for value in replay_receipts
            ),
            "child_target_native_full_prefix_replay_calls": sum(
                int(value["full_prefix_replay_calls"]) for value in replay_receipts
            ),
            "child_macrocell_replay_calls": observed_macrocell_calls,
            "expected_child_macrocell_replay_calls": macrocell_calls,
            "child_attention_rope_materializations": rope_calls,
            "child_layer_forward_calls": 0,
            "closed_form_terminal_compiler_calls": 1,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "new_parameters": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
        "scope_limits": {
            "changes_only_existing_final_layer_w2_if_accepted": True,
            "does_not_refold_upstream_layers": True,
            "does_not_create_tokenizer_rows": True,
            "semantic_evidence_is_parent_phase_terminal_only": (
                tokenizer_replay_mode == "semantic_span"
            ),
            "fabricated_intermediate_parent_rows": False,
            "reconstructs_full_attention_histories": False,
            "claims_inserted_target_subtoken_parent_actions": False,
            "does_not_claim_multimodal_coverage": True,
        },
    }
    seal = _canonical_sha256(body)
    return Lfm2LongHorizonTerminalRefoldResult(
        split_evidence=evidence_tuple,
        terminal_result=terminal,
        accepted=accepted,
        report={**body, "result_sha256": seal},
        result_sha256=seal,
    )


__all__ = [
    "LFM2_LONG_HORIZON_PARENT_FINAL_NORM_SCHEMA",
    "LFM2_LONG_HORIZON_SPLIT_PLAN_SCHEMA",
    "LFM2_LONG_HORIZON_TERMINAL_REFOLD_SCHEMA",
    "Lfm2LongHorizonTerminalRefoldError",
    "Lfm2LongHorizonTerminalRefoldResult",
    "Lfm2LongHorizonTrajectorySplitPlan",
    "Lfm2LocalParentFinalRmsNormLoad",
    "Lfm2SealedParentFinalRmsNorm",
    "TokenizerReplayMode",
    "compile_install_lfm2_long_horizon_terminal_refold",
    "load_lfm2_parent_final_rmsnorm_from_local_safetensors",
]
