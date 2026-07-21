"""No-parent-forward long-horizon terminal refolding for an LFM2 child.

This module closes one deliberately narrow gap between the legacy LFM2
circuit cache and the pure terminal-action residual compiler:

* authenticate a sealed legacy cache while loading only bounded endpoints;
* invert an authenticated strict one-row tokenizer bridge for every cached
  prompt and forced token;
* reconstruct the parent's final RMSNorm output analytically from the cached
  final-layer residual and a separately sealed final-normalization contract;
* replay each complete target-token trajectory through every installed child
  macrocell exactly once, without calling ``model.forward`` or ``generate``;
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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from .hf_lfm2_circuit_trace_cache import (
    CACHED_VERIFIED_TRACE_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
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
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2TargetTokenizedRecord,
    authenticate_lfm2_bridge_input_pair,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from .hf_tied_causal_interface_fold import TiedCausalInterfaceFold
from .hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_LONG_HORIZON_PARENT_FINAL_NORM_SCHEMA = (
    "neural-forge-lfm2-long-horizon-parent-final-norm-v1"
)
LFM2_LONG_HORIZON_SPLIT_PLAN_SCHEMA = "neural-forge-lfm2-long-horizon-split-plan-v1"
LFM2_LONG_HORIZON_TERMINAL_REFOLD_SCHEMA = (
    "neural-forge-lfm2-long-horizon-terminal-refold-v1"
)

TerminalSplit = Literal["fit", "fit_check", "selection", "validation"]
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
    target_record: Lfm2TargetTokenizedRecord
    parent_final_norm_rows: Any
    horizon_indices: tuple[int, ...]
    selected_payload_sha256: str


@dataclass(frozen=True, slots=True)
class _ChildTrajectoryRows:
    mlp_input: Any
    pre_mlp_residual: Any
    macrocell_replay_sha256: tuple[str, ...]
    attention_rope_materializations: int


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


def _analytical_parent_rmsnorm(
    residual_rows: Any,
    seal: Lfm2SealedParentFinalRmsNorm,
) -> Any:
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
    rows = residual_rows.to(device="cpu", dtype=torch.float64)
    result = (
        rows
        * (rows.square().mean(dim=1, keepdim=True) + seal.epsilon).rsqrt()
        * seal.weight.to(torch.float64)
    ).to(torch.bfloat16)
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2LongHorizonTerminalRefoldError(
            "Analytical parent final RMSNorm reconstruction is nonfinite."
        )
    return result.contiguous().clone()


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
        prefix = f"{phase}.layer.{final_layer_index:03d}."
        names.update(prefix + suffix for suffix in suffixes)
        if phase != "prefill":
            names.add(f"{phase}.forced_token_ids")
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


def _collect_cached_trajectories(
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    inverse: tuple[int, ...],
    parent_final_norm: Lfm2SealedParentFinalRmsNorm,
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
                parent_rows = _analytical_parent_rmsnorm(
                    residual,
                    parent_final_norm,
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
                        parent_final_norm_rows=parent_rows,
                        horizon_indices=tuple(range(decode_steps + 1)),
                        selected_payload_sha256=payload.selection_sha256,
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


def _replay_child_trajectory_once(
    target_model: Any,
    layers: tuple[Any, ...],
    aligned_record: Any,
) -> _ChildTrajectoryRows:
    torch = _torch()
    full_ids, full_mask, positions, prompt_width = _full_target_trajectory(
        aligned_record
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
            row_indices = torch.tensor(
                [prompt_width - 1] + list(range(prompt_width, int(full_ids.shape[1]))),
                dtype=torch.long,
            )
            final_mlp = replay.mlp_input.index_select(1, row_indices)[0]
            final_pre_mlp = replay.post_operator_residual.index_select(
                1,
                row_indices,
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
    )


def _build_split_evidence(
    cached: tuple[_CachedTrajectory, ...],
    adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
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
        aligned = adapter.records[trajectory.batch_index]
        child = _replay_child_trajectory_once(target_model, layers, aligned)
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
                "record_id": aligned.record_id,
                "split": trajectory.split,
                "trajectory_rows": expected_rows,
                "horizon_minimum": min(trajectory.horizon_indices),
                "horizon_maximum": max(trajectory.horizon_indices),
                "embedding_seed_lookup_calls": 1,
                "macrocell_replay_calls": len(layers),
                "attention_rope_materializations": (
                    child.attention_rope_materializations
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
    parent_final_norm: Lfm2SealedParentFinalRmsNorm,
    split_plan: Lfm2LongHorizonTrajectorySplitPlan | None = None,
    policy: Lfm2CpcfTerminalActionPolicy | None = None,
) -> Lfm2LongHorizonTerminalRefoldResult:
    """Compile and conditionally install one long-horizon terminal refold."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    _verify_parent_norm_binding(cache, parent_final_norm)
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

    seed, down_parameter, layers, target_norm_epsilon = _target_runtime_seed(
        target_model,
        interface,
        source_selected_tied_table,
    )
    inverse = _strict_inverse_mapping(bridge, seed.vocabulary_size)
    child_before = fingerprint_lfm2_parent(target_model)
    parameter_count_before = _unique_parameter_count(target_model)

    cached, adapter, pair_receipts, cache_receipt = _collect_cached_trajectories(
        cache,
        bridge,
        inverse,
        parent_final_norm,
        plan,
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

    macrocell_calls = cache.batch_count * len(layers)
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
            "sealed_legacy_cache_analytical_parent_final_norm_plus_"
            "one_pass_complete_child_macrocell_trajectory_plus_closed_form_"
            "terminal_writer_residual"
        ),
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "cache_layer_topology_fingerprint": cache.layer_topology_fingerprint,
        "cache_authentication_sha256": cache_receipt["session"]["session_sha256"],
        "cache_authentication": cache_receipt,
        "parent_final_norm": parent_final_norm.as_dict(),
        "parent_final_norm_reconstruction": {
            "method": "analytical_rmsnorm_from_cached_final_residual",
            "final_norm_module_or_parent_model_called": False,
            "reconstructed_output_dtype": "bfloat16",
            "prompt_endpoint_and_every_forced_position": True,
        },
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_transport_profile_fingerprint": bridge.profile_fingerprint,
        "bridge_trace_adapter_sha256": adapter.evidence_sha256,
        "bridge_pair_authentication_sha256": [
            value["evidence_sha256"] for value in pair_receipts
        ],
        "bridge_inverse_contract": {
            "every_cached_donor_prompt_and_forced_id_covered": True,
            "every_used_donor_row_has_exactly_one_target_row": True,
            "numeric_token_id_equality_required": False,
        },
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
            "child_model_forward_calls": 0,
            "child_model_generate_calls": 0,
            "child_embedding_seed_lookups": cache.batch_count,
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
    "Lfm2SealedParentFinalRmsNorm",
    "compile_install_lfm2_long_horizon_terminal_refold",
]
