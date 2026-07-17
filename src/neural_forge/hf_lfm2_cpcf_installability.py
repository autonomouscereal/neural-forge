"""Strict parent-side CPCF compiler inputs from an opt-in LFM2 trace cache.

The legacy predictive-rank adapter intentionally accepts older sampled caches
for measurement.  This module does not.  It requires the opt-in causal-circuit
capture contract and functionally authenticates every installability endpoint
before exposing it as a streaming compiler input:

* normalized pre-RoPE Q/K, RoPE terms, complete K/V history, exact QK scores,
  additive masks, probabilities, and reconstructed attention context;
* complete B/C/x and B*x history, the pre-update convolution state, exact FIR
  kernel/bias, filtered path, and reconstructed gated context; and
* final RMSNorm input/output with an exactly equal language-head input.

It only opens a caller-supplied sealed cache.  It has no parent-model or target
model argument and cannot execute, regenerate, train, prune, or quantize one.
The result proves parent-side compiler-input readiness, not that a target child
has been compiled or preserves intelligence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA = (
    "neural-forge-lfm2-cpcf-installable-layer-v1"
)
LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA = (
    "neural-forge-lfm2-cpcf-installable-phase-v1"
)
LFM2_CPCF_INSTALLABILITY_AUDIT_SCHEMA = (
    "neural-forge-lfm2-cpcf-installability-audit-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CAPTURE_FLAGS = (
    "capture_causal_circuit_evidence",
    "attention_pre_rope_qk",
    "attention_exact_causal_scores_and_probabilities",
    "attention_complete_kv_history",
    "short_conv_exact_fir_state_and_response",
    "final_normalization_boundary",
)
_COMMON_FIELDS = (
    "residual.pre",
    "operator.input",
    "operator.output",
    "residual.post_operator",
    "mlp.input",
    "mlp.gate",
    "mlp.up",
    "mlp.post_swiglu",
    "mlp.down",
    "residual.post_layer",
)
_ATTENTION_FIELDS = (
    "attention.q_linear",
    "attention.q",
    "attention.k_linear",
    "attention.k",
    "attention.v",
    "attention.context",
    "attention.o",
    "attention.q_normalized_pre_rope",
    "attention.k_normalized_pre_rope",
    "attention.rope_cos",
    "attention.rope_sin",
    "attention.key_history",
    "attention.value_history",
    "attention.qk_scores",
    "attention.additive_mask",
    "attention.probabilities",
    "attention.context_from_probabilities",
)
_SHORT_CONV_FIELDS = (
    "short_conv.b",
    "short_conv.c",
    "short_conv.x",
    "short_conv.bx",
    "short_conv.context",
    "short_conv.o",
    "short_conv.b_history",
    "short_conv.c_history",
    "short_conv.x_history",
    "short_conv.bx_history",
    "short_conv.pre_bx_state",
    "short_conv.fir_kernel",
    "short_conv.fir_bias",
    "short_conv.fir_has_bias",
    "short_conv.filtered_bx",
    "short_conv.context_from_fir",
)
_OUTPUT_FIELDS = (
    "output.final_norm_input",
    "output.final_norm_output",
    "output.lm_head_input",
)


class Lfm2CpcfInstallabilityError(ValueError):
    """A cache cannot be trusted as an actual CPCF compiler input."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF installability evidence."
        ) from exc
    return torch


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
        raise Lfm2CpcfInstallabilityError(
            "Installability metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfInstallabilityError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfInstallabilityError(f"{name} must be positive.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfInstallabilityError(f"{name} must be nonnegative.")
    return value


def _tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
    floating: bool = True,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
    ):
        raise Lfm2CpcfInstallabilityError(
            f"{name} must be a nonempty dense rank-{dimensions} tensor."
        )
    if floating and (
        not torch.is_floating_point(value) or torch.is_complex(value)
    ):
        raise Lfm2CpcfInstallabilityError(
            f"{name} must be a real floating tensor."
        )
    result = value.detach().to(device="cpu").contiguous()
    if (
        torch.is_floating_point(result) or torch.is_complex(result)
    ) and not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfInstallabilityError(f"{name} contains NaN or infinity.")
    return result


def _tensor_sha256(value: Any) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(_torch().uint8).numpy().tobytes())
    return digest.hexdigest()


def _field(
    tensors: Mapping[str, Any],
    name: str,
    *,
    batch_index: int,
    phase: str,
) -> Any:
    try:
        return tensors[name]
    except KeyError as exc:
        raise Lfm2CpcfInstallabilityError(
            f"Batch {batch_index} phase {phase!r} is missing required "
            f"installability field {name!r}."
        ) from exc


def _phase_names(record: Mapping[str, Any], batch_index: int) -> tuple[str, ...]:
    capture = record.get("capture")
    if not isinstance(capture, Mapping):
        raise Lfm2CpcfInstallabilityError(
            f"Cache batch {batch_index} has no capture record."
        )
    decode_steps = capture.get("decode_steps")
    if (
        isinstance(decode_steps, bool)
        or not isinstance(decode_steps, int)
        or decode_steps < 0
    ):
        raise Lfm2CpcfInstallabilityError(
            f"Cache batch {batch_index} has an invalid decode count."
        )
    return (
        "prefill",
        *(f"decode.{step:03d}" for step in range(decode_steps)),
    )


def _rotate_half(value: Any) -> Any:
    midpoint = int(value.shape[-1]) // 2
    return _torch().cat(
        (-value[..., midpoint:], value[..., :midpoint]),
        dim=-1,
    )


def _close(left: Any, right: Any, name: str, *, exact: bool = False) -> None:
    torch = _torch()
    left_value = left.to(dtype=torch.float64)
    right_value = right.to(dtype=torch.float64)
    if tuple(left_value.shape) != tuple(right_value.shape):
        raise Lfm2CpcfInstallabilityError(
            f"{name} changed shape across its functional identity."
        )
    if exact:
        passed = bool(torch.equal(left_value, right_value))
    else:
        passed = bool(
            torch.allclose(
                left_value,
                right_value,
                rtol=2e-2,
                atol=2e-3,
            )
        )
    if not passed:
        maximum = float((left_value - right_value).abs().max().item())
        raise Lfm2CpcfInstallabilityError(
            f"{name} failed its functional identity "
            f"(maximum absolute error {maximum:.6g})."
        )


def _copy_fields(
    tensors: Mapping[str, Any],
    prefix: str,
    names: Sequence[str],
    *,
    batch_index: int,
    phase: str,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            name: _field(
                tensors,
                prefix + name,
                batch_index=batch_index,
                phase=phase,
            )
            .detach()
            .to(device="cpu")
            .contiguous()
            .clone()
            for name in names
        }
    )


def _field_inventory(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": [int(size) for size in value.shape],
            "sha256": _tensor_sha256(value),
        }
        for name, value in sorted(values.items())
    }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfInstallableLayerEvidence:
    """One functionally authenticated operator/MLP compiler input."""

    layer_index: int
    operator_kind: str
    source_width: int
    tensors: Mapping[str, Any]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise Lfm2CpcfInstallabilityError(
                "layer_index must be nonnegative."
            )
        if self.operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfInstallabilityError(
                "operator_kind is not canonical LFM2."
            )
        _positive_int(self.source_width, "source_width")
        _digest(self.evidence_sha256, "evidence_sha256")
        object.__setattr__(
            self,
            "tensors",
            MappingProxyType(dict(self.tensors)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA,
            "layer_index": self.layer_index,
            "operator_kind": self.operator_kind,
            "source_width": self.source_width,
            "field_inventory": _field_inventory(self.tensors),
            "functional_identities_verified": True,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfInstallablePhaseEvidence:
    """One cache batch/phase ready to feed parent-side CPCF solvers."""

    cache_key: str
    batch_index: int
    input_sha256: str
    phase: str
    layers: tuple[Lfm2CpcfInstallableLayerEvidence, ...]
    final_norm_tensors: Mapping[str, Any]
    evidence_sha256: str

    def __post_init__(self) -> None:
        _digest(self.cache_key, "cache_key")
        _digest(self.input_sha256, "input_sha256")
        _digest(self.evidence_sha256, "evidence_sha256")
        if not self.layers:
            raise Lfm2CpcfInstallabilityError(
                "Installable phase must contain all model layers."
            )
        if tuple(layer.layer_index for layer in self.layers) != tuple(
            range(len(self.layers))
        ):
            raise Lfm2CpcfInstallabilityError(
                "Installable phase layers are incomplete or unordered."
            )
        object.__setattr__(
            self,
            "final_norm_tensors",
            MappingProxyType(dict(self.final_norm_tensors)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA,
            "cache_key": self.cache_key,
            "batch_index": self.batch_index,
            "input_sha256": self.input_sha256,
            "phase": self.phase,
            "layers": [layer.as_dict() for layer in self.layers],
            "final_norm_field_inventory": _field_inventory(
                self.final_norm_tensors
            ),
            "parent_side_compiler_inputs_ready": True,
            "evidence_sha256": self.evidence_sha256,
        }


def _installability_audit_body(
    *,
    cache_key: str,
    layer_topology_sha256: str,
    capture_selection_sha256: str,
    batch_count: int,
    phase_count: int,
    layer_count: int,
    source_width: int,
    attention_phase_count: int,
    short_conv_phase_count: int,
    phase_evidence_sha256: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": LFM2_CPCF_INSTALLABILITY_AUDIT_SCHEMA,
        "cache_key": cache_key,
        "layer_topology_sha256": layer_topology_sha256,
        "capture_selection_sha256": capture_selection_sha256,
        "batch_count": batch_count,
        "phase_count": phase_count,
        "layer_count": layer_count,
        "source_width": source_width,
        "attention_phase_count": attention_phase_count,
        "short_conv_phase_count": short_conv_phase_count,
        "phase_evidence_sha256": list(phase_evidence_sha256),
        "required_capture_flags": list(_CAPTURE_FLAGS),
        "functional_identities": {
            "pre_rope_qk_rotary_reconstruction": True,
            "qk_score_reconstruction": True,
            "masked_softmax_probability_reconstruction": True,
            "attention_context_reconstruction": True,
            "short_conv_bx_reconstruction": True,
            "short_conv_fir_reconstruction": True,
            "short_conv_gated_context_reconstruction": True,
            "final_norm_to_lm_head_identity": True,
        },
        "parent_side_compiler_inputs_ready": True,
        "measurement_only": False,
        "target_child_compiled": False,
        "parameter_compaction_proven": False,
        "intelligence_retention_measured": False,
        "external_requirements": [
            "target-width boundary coordinates",
            "child-only closed-loop replay",
            "exact blank-target parameter inventory",
            "tokenizer bridge when token semantics differ",
        ],
        "operation_audit": {
            "parent_forward_count": 0,
            "parent_generation_count": 0,
            "parent_cache_regeneration_count": 0,
            "training_step_count": 0,
            "backward_call_count": 0,
            "optimizer_step_count": 0,
        },
    }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfInstallabilityAudit:
    """Sealed readiness result over every batch, phase, and LFM2 layer."""

    cache_key: str
    layer_topology_sha256: str
    capture_selection_sha256: str
    batch_count: int
    phase_count: int
    layer_count: int
    source_width: int
    attention_phase_count: int
    short_conv_phase_count: int
    phase_evidence_sha256: tuple[str, ...]
    report_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.cache_key, "cache_key"),
            (self.layer_topology_sha256, "layer_topology_sha256"),
            (self.capture_selection_sha256, "capture_selection_sha256"),
            (self.report_sha256, "report_sha256"),
        ):
            _digest(value, name)
        for value, name in (
            (self.batch_count, "batch_count"),
            (self.phase_count, "phase_count"),
            (self.layer_count, "layer_count"),
            (self.source_width, "source_width"),
        ):
            _positive_int(value, name)
        if len(self.phase_evidence_sha256) != self.phase_count:
            raise Lfm2CpcfInstallabilityError(
                "Phase evidence seals do not cover every phase."
            )
        for digest in self.phase_evidence_sha256:
            _digest(digest, "phase evidence SHA-256")
        attention = _nonnegative_int(
            self.attention_phase_count,
            "attention_phase_count",
        )
        short_conv = _nonnegative_int(
            self.short_conv_phase_count,
            "short_conv_phase_count",
        )
        if attention + short_conv != self.phase_count * self.layer_count:
            raise Lfm2CpcfInstallabilityError(
                "Operator-phase coverage does not match every phase/layer."
            )
        if _sha(self._body()) != self.report_sha256:
            raise Lfm2CpcfInstallabilityError(
                "Installability audit changed after sealing."
            )

    def _body(self) -> dict[str, Any]:
        return _installability_audit_body(
            cache_key=self.cache_key,
            layer_topology_sha256=self.layer_topology_sha256,
            capture_selection_sha256=self.capture_selection_sha256,
            batch_count=self.batch_count,
            phase_count=self.phase_count,
            layer_count=self.layer_count,
            source_width=self.source_width,
            attention_phase_count=self.attention_phase_count,
            short_conv_phase_count=self.short_conv_phase_count,
            phase_evidence_sha256=self.phase_evidence_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "report_sha256": self.report_sha256}


class Lfm2CpcfInstallabilityAdapter:
    """Authenticate opt-in cache evidence and stream compiler inputs."""

    def __init__(self, cache: Any) -> None:
        if cache is None or not callable(getattr(cache, "verify", None)):
            raise Lfm2CpcfInstallabilityError(
                "A verified sealed circuit cache is required."
            )
        cache.verify()
        self.cache = cache
        self.cache_key = _digest(
            getattr(cache, "cache_key", None),
            "cache_key",
        )
        self.batch_count = _positive_int(
            getattr(cache, "batch_count", None),
            "batch_count",
        )
        manifest = getattr(cache, "manifest", None)
        if not isinstance(manifest, Mapping):
            raise Lfm2CpcfInstallabilityError(
                "Verified cache manifest is unavailable."
            )
        self.capture_selection_sha256 = self._verify_capture_contract(
            manifest.get("capture")
        )
        topology = manifest.get("layer_topology")
        corpus = manifest.get("corpus")
        if not isinstance(topology, Mapping) or not isinstance(
            corpus,
            Mapping,
        ):
            raise Lfm2CpcfInstallabilityError(
                "Cache manifest lacks topology or corpus identity."
            )
        self.layer_topology_sha256 = _digest(
            topology.get("layer_topology_sha256"),
            "layer_topology_sha256",
        )
        layers = topology.get("layers")
        if not isinstance(layers, Sequence) or isinstance(
            layers,
            (str, bytes),
        ) or not layers:
            raise Lfm2CpcfInstallabilityError(
                "Cache topology has no ordered layers."
            )
        self.topology = tuple(layers)
        widths = set()
        for index, entry in enumerate(self.topology):
            if not isinstance(entry, Mapping):
                raise Lfm2CpcfInstallabilityError(
                    f"Layer topology {index} is invalid."
                )
            contract = entry.get("contract")
            if (
                not isinstance(contract, Mapping)
                or contract.get("layer_index") != index
                or contract.get("kind")
                not in {"full_attention", "short_conv"}
            ):
                raise Lfm2CpcfInstallabilityError(
                    f"Layer topology {index} is not canonical LFM2."
                )
            widths.add(
                _positive_int(
                    contract.get("hidden_size"),
                    f"layer {index} hidden_size",
                )
            )
        if len(widths) != 1:
            raise Lfm2CpcfInstallabilityError(
                "Installability requires one LFM2 residual width."
            )
        self.source_width = next(iter(widths))
        ordered = corpus.get("ordered_batch_input_sha256")
        if (
            not isinstance(ordered, Sequence)
            or isinstance(ordered, (str, bytes))
            or len(ordered) != self.batch_count
        ):
            raise Lfm2CpcfInstallabilityError(
                "Cache corpus batch identities are incomplete."
            )
        self.ordered_input_sha256 = tuple(
            _digest(value, f"batch {index} input identity")
            for index, value in enumerate(ordered)
        )

    @staticmethod
    def _verify_capture_contract(value: Any) -> str:
        if not isinstance(value, Mapping):
            raise Lfm2CpcfInstallabilityError(
                "Cache has no sealed capture selection."
            )
        for flag in _CAPTURE_FLAGS:
            if value.get(flag) is not True:
                raise Lfm2CpcfInstallabilityError(
                    "Actual compilation requires "
                    "capture_causal_circuit_evidence=True and every "
                    f"installability flag; {flag!r} is not authenticated."
                )
        if (
            value.get("forced_decode_use_cache") is not True
            or value.get("cached_prefill_use_cache") is not True
            or value.get("full_prefix_verification_use_cache") is not False
            or value.get("gradients_used") is not False
            or value.get("training_steps") != 0
        ):
            raise Lfm2CpcfInstallabilityError(
                "Installability evidence is not the canonical cached-verified "
                "zero-training capture mode."
            )
        observed = _digest(
            value.get("selection_sha256"),
            "capture selection_sha256",
        )
        body = dict(value)
        body.pop("selection_sha256")
        if _sha(body) != observed:
            raise Lfm2CpcfInstallabilityError(
                "Capture selection changed after sealing."
            )
        return observed

    def _validate_capture_record(
        self,
        record: Mapping[str, Any],
        *,
        batch_index: int,
        phases: Sequence[str],
    ) -> Mapping[str, Any]:
        capture = record.get("capture")
        if not isinstance(capture, Mapping):
            raise Lfm2CpcfInstallabilityError(
                f"Cache batch {batch_index} has no capture record."
            )
        if (
            capture.get("decode_trace_mode") != "cached_verified"
            or capture.get("canonical_trace_source")
            != "verified_one_token_cache"
            or capture.get("cached_logits_accepted") is not True
            or capture.get("all_cached_logits_verified") is not True
            or (
                capture.get(
                    "all_cached_top1_decisions_verified",
                    capture.get("all_cached_top1_verified"),
                )
                is not True
            )
            or capture.get("gradients_used") is not False
            or capture.get("training_steps") != 0
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Cache batch {batch_index} is not an authenticated "
                "cached-verified zero-training trace."
            )
        diagnostics = capture.get("phase_diagnostics")
        if (
            not isinstance(diagnostics, Mapping)
            or tuple(diagnostics) != tuple(phases)
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Cache batch {batch_index} phase diagnostics are incomplete "
                "or reordered."
            )
        return diagnostics

    def _attention_layer(
        self,
        values: Mapping[str, Any],
        contract: Mapping[str, Any],
        *,
        batch_index: int,
        phase: str,
        layer_index: int,
    ) -> None:
        torch = _torch()
        hidden = self.source_width
        operator = contract.get("operator")
        if not isinstance(operator, Mapping):
            raise Lfm2CpcfInstallabilityError(
                f"Attention layer {layer_index} has no operator contract."
            )
        query_heads = _positive_int(
            operator.get("query_heads"),
            f"layer {layer_index} query_heads",
        )
        kv_heads = _positive_int(
            operator.get("key_value_heads"),
            f"layer {layer_index} key_value_heads",
        )
        head_dim = _positive_int(
            operator.get("head_dim"),
            f"layer {layer_index} head_dim",
        )
        if (
            query_heads % kv_heads != 0
            or query_heads * head_dim != hidden
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Attention layer {layer_index} topology is not installable."
            )
        common = values["residual.pre"]
        batch, sampled = int(common.shape[0]), int(common.shape[1])
        q_norm = _tensor(
            values["attention.q_normalized_pre_rope"],
            "attention.q_normalized_pre_rope",
            dimensions=4,
        )
        k_norm = _tensor(
            values["attention.k_normalized_pre_rope"],
            "attention.k_normalized_pre_rope",
            dimensions=4,
        )
        cos = _tensor(
            values["attention.rope_cos"],
            "attention.rope_cos",
            dimensions=3,
        )
        sin = _tensor(
            values["attention.rope_sin"],
            "attention.rope_sin",
            dimensions=3,
        )
        current = int(q_norm.shape[2])
        if (
            tuple(q_norm.shape)
            != (batch, query_heads, current, head_dim)
            or tuple(k_norm.shape) != (batch, kv_heads, current, head_dim)
            or tuple(cos.shape) != (batch, current, head_dim)
            or tuple(sin.shape) != tuple(cos.shape)
            or sampled > current
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Attention layer {layer_index} pre-RoPE evidence shape changed."
            )
        q_rotary = (
            q_norm * cos.unsqueeze(1)
            + _rotate_half(q_norm) * sin.unsqueeze(1)
        )
        k_rotary = (
            k_norm * cos.unsqueeze(1)
            + _rotate_half(k_norm) * sin.unsqueeze(1)
        )
        _close(
            q_rotary[:, :, -sampled:, :],
            _tensor(
                values["attention.q"],
                "attention.q",
                dimensions=4,
            ),
            f"layer {layer_index} sampled post-RoPE Q",
        )
        key_history = _tensor(
            values["attention.key_history"],
            "attention.key_history",
            dimensions=4,
        )
        value_history = _tensor(
            values["attention.value_history"],
            "attention.value_history",
            dimensions=4,
        )
        key_count = int(key_history.shape[2])
        if (
            tuple(key_history.shape)
            != (batch, kv_heads, key_count, head_dim)
            or tuple(value_history.shape) != tuple(key_history.shape)
            or key_count < current
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Attention layer {layer_index} complete K/V history changed."
            )
        _close(
            key_history[:, :, -current:, :],
            k_rotary,
            f"layer {layer_index} current post-RoPE K history",
        )
        groups = query_heads // kv_heads
        repeated_key = key_history.repeat_interleave(groups, dim=1)
        repeated_value = value_history.repeat_interleave(groups, dim=1)
        scores = torch.matmul(
            q_rotary[:, :, -sampled:, :].to(torch.float64),
            repeated_key.transpose(2, 3).to(torch.float64),
        ) / math.sqrt(head_dim)
        captured_scores = _tensor(
            values["attention.qk_scores"],
            "attention.qk_scores",
            dimensions=4,
        )
        _close(
            scores,
            captured_scores,
            f"layer {layer_index} exact QK scores",
        )
        additive_mask = _tensor(
            values["attention.additive_mask"],
            "attention.additive_mask",
            dimensions=4,
        )
        if (
            int(additive_mask.shape[0]) != batch
            or int(additive_mask.shape[1]) not in {1, query_heads}
            or tuple(additive_mask.shape[2:]) != (sampled, key_count)
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Attention layer {layer_index} additive mask shape changed."
            )
        probabilities = torch.nn.functional.softmax(
            scores + additive_mask.to(torch.float64),
            dim=-1,
        )
        captured_probabilities = _tensor(
            values["attention.probabilities"],
            "attention.probabilities",
            dimensions=4,
        )
        _close(
            probabilities,
            captured_probabilities,
            f"layer {layer_index} masked-softmax probabilities",
        )
        context = torch.matmul(
            probabilities,
            repeated_value.to(torch.float64),
        )
        captured_from_probabilities = _tensor(
            values["attention.context_from_probabilities"],
            "attention.context_from_probabilities",
            dimensions=4,
        )
        captured_context = _tensor(
            values["attention.context"],
            "attention.context",
            dimensions=4,
        )
        _close(
            context,
            captured_from_probabilities,
            f"layer {layer_index} probability-weighted V context",
        )
        _close(
            captured_from_probabilities,
            captured_context,
            f"layer {layer_index} executed pre-O context",
        )

    def _short_conv_layer(
        self,
        values: Mapping[str, Any],
        contract: Mapping[str, Any],
        *,
        layer_index: int,
    ) -> None:
        torch = _torch()
        operator = contract.get("operator")
        if not isinstance(operator, Mapping):
            raise Lfm2CpcfInstallabilityError(
                f"Short-conv layer {layer_index} has no operator contract."
            )
        hidden = self.source_width
        fir_length = _positive_int(
            operator.get("fir_length"),
            f"layer {layer_index} fir_length",
        )
        b_history = _tensor(
            values["short_conv.b_history"],
            "short_conv.b_history",
            dimensions=3,
        )
        c_history = _tensor(
            values["short_conv.c_history"],
            "short_conv.c_history",
            dimensions=3,
        )
        x_history = _tensor(
            values["short_conv.x_history"],
            "short_conv.x_history",
            dimensions=3,
        )
        batch, sequence = int(b_history.shape[0]), int(b_history.shape[1])
        expected = (batch, sequence, hidden)
        if (
            tuple(b_history.shape) != expected
            or tuple(c_history.shape) != expected
            or tuple(x_history.shape) != expected
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Short-conv layer {layer_index} complete history changed."
            )
        bx_history = b_history * x_history
        _close(
            bx_history,
            _tensor(
                values["short_conv.bx_history"],
                "short_conv.bx_history",
                dimensions=3,
            ),
            f"layer {layer_index} B*x history",
        )
        pre_state = _tensor(
            values["short_conv.pre_bx_state"],
            "short_conv.pre_bx_state",
            dimensions=3,
        )
        kernel = _tensor(
            values["short_conv.fir_kernel"],
            "short_conv.fir_kernel",
            dimensions=2,
        )
        bias = _tensor(
            values["short_conv.fir_bias"],
            "short_conv.fir_bias",
            dimensions=1,
        )
        has_bias = _tensor(
            values["short_conv.fir_has_bias"],
            "short_conv.fir_has_bias",
            dimensions=1,
            floating=False,
        )
        if (
            tuple(pre_state.shape) != (batch, hidden, fir_length)
            or tuple(kernel.shape) != (hidden, fir_length)
            or tuple(bias.shape) != (hidden,)
            or int(has_bias.numel()) != 1
            or int(has_bias.item()) not in {0, 1}
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Short-conv layer {layer_index} FIR state/kernel shape changed."
            )
        if int(has_bias.item()) == 0 and not bool(
            torch.equal(bias, torch.zeros_like(bias))
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Short-conv layer {layer_index} declares no bias but stores one."
            )
        bx_channels = bx_history.transpose(1, 2).contiguous()
        if sequence == 1:
            updated = torch.cat(
                (pre_state[..., 1:], bx_channels),
                dim=-1,
            )
            filtered = (
                updated.to(torch.float64)
                * kernel.to(torch.float64).unsqueeze(0)
            ).sum(dim=-1, keepdim=True)
            filtered = filtered + bias.to(torch.float64)[None, :, None]
        else:
            if not bool(
                torch.equal(pre_state, torch.zeros_like(pre_state))
            ):
                raise Lfm2CpcfInstallabilityError(
                    f"Short-conv layer {layer_index} multi-token prefill has "
                    "a nonempty pre-update state."
                )
            filtered = torch.nn.functional.conv1d(
                bx_channels.to(torch.float64),
                kernel.to(torch.float64)[:, None, :],
                bias=bias.to(torch.float64),
                padding=fir_length - 1,
                groups=hidden,
            )[..., :sequence]
        filtered_tokens = filtered.transpose(1, 2).contiguous()
        captured_filtered = _tensor(
            values["short_conv.filtered_bx"],
            "short_conv.filtered_bx",
            dimensions=3,
        )
        _close(
            filtered_tokens,
            captured_filtered,
            f"layer {layer_index} exact FIR response",
        )
        gated_context = c_history.to(torch.float64) * filtered_tokens
        captured_gated = _tensor(
            values["short_conv.context_from_fir"],
            "short_conv.context_from_fir",
            dimensions=3,
        )
        _close(
            gated_context,
            captured_gated,
            f"layer {layer_index} gated FIR context",
        )
        sampled = int(values["residual.pre"].shape[1])
        _close(
            captured_gated[:, -sampled:, :],
            _tensor(
                values["short_conv.context"],
                "short_conv.context",
                dimensions=3,
            ),
            f"layer {layer_index} executed short-conv context",
        )

    def _layer_evidence(
        self,
        tensors: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        *,
        batch_index: int,
        phase: str,
        layer_index: int,
    ) -> Lfm2CpcfInstallableLayerEvidence:
        contract = self.topology[layer_index]["contract"]
        operator_kind = str(contract["kind"])
        names = (
            *_COMMON_FIELDS,
            *(
                _ATTENTION_FIELDS
                if operator_kind == "full_attention"
                else _SHORT_CONV_FIELDS
            ),
        )
        prefix = f"{phase}.layer.{layer_index:03d}."
        values = _copy_fields(
            tensors,
            prefix,
            names,
            batch_index=batch_index,
            phase=phase,
        )
        common_shapes = {
            name: _tensor(values[name], name, dimensions=3)
            for name in _COMMON_FIELDS
        }
        batch, sampled = (
            int(common_shapes["residual.pre"].shape[0]),
            int(common_shapes["residual.pre"].shape[1]),
        )
        intermediate = _positive_int(
            contract.get("ffn_intermediate_size"),
            f"layer {layer_index} ffn_intermediate_size",
        )
        for name in (
            "residual.pre",
            "operator.input",
            "operator.output",
            "residual.post_operator",
            "mlp.input",
            "mlp.down",
            "residual.post_layer",
        ):
            if tuple(common_shapes[name].shape) != (
                batch,
                sampled,
                self.source_width,
            ):
                raise Lfm2CpcfInstallabilityError(
                    f"Layer {layer_index} field {name!r} changed shape."
                )
        for name in ("mlp.gate", "mlp.up", "mlp.post_swiglu"):
            if tuple(common_shapes[name].shape) != (
                batch,
                sampled,
                intermediate,
            ):
                raise Lfm2CpcfInstallabilityError(
                    f"Layer {layer_index} field {name!r} changed shape."
                )
        layer_diagnostics = diagnostics.get(f"layer.{layer_index:03d}")
        if not isinstance(layer_diagnostics, Mapping):
            raise Lfm2CpcfInstallabilityError(
                f"Layer {layer_index} has no phase diagnostics."
            )
        if operator_kind == "full_attention":
            if (
                layer_diagnostics.get(
                    "attention_causal_circuit_evidence_captured"
                )
                is not True
                or layer_diagnostics.get(
                    "attention_context_reconstructed_from_probabilities"
                )
                is not True
            ):
                raise Lfm2CpcfInstallabilityError(
                    f"Attention layer {layer_index} diagnostics do not "
                    "authenticate causal-circuit evidence."
                )
            self._attention_layer(
                values,
                contract,
                batch_index=batch_index,
                phase=phase,
                layer_index=layer_index,
            )
        else:
            if (
                layer_diagnostics.get(
                    "short_conv_causal_circuit_evidence_captured"
                )
                is not True
                or layer_diagnostics.get(
                    "short_conv_context_reconstructed_from_fir"
                )
                is not True
            ):
                raise Lfm2CpcfInstallabilityError(
                    f"Short-conv layer {layer_index} diagnostics do not "
                    "authenticate causal-circuit evidence."
                )
            self._short_conv_layer(
                values,
                contract,
                layer_index=layer_index,
            )
        inventory = _field_inventory(values)
        body = {
            "schema": LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA,
            "cache_key": self.cache_key,
            "batch_index": batch_index,
            "phase": phase,
            "layer_index": layer_index,
            "operator_kind": operator_kind,
            "source_width": self.source_width,
            "field_inventory": inventory,
            "functional_identities_verified": True,
        }
        return Lfm2CpcfInstallableLayerEvidence(
            layer_index=layer_index,
            operator_kind=operator_kind,
            source_width=self.source_width,
            tensors=values,
            evidence_sha256=_sha(body),
        )

    def _phase_evidence(
        self,
        tensors: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        *,
        batch_index: int,
        input_sha256: str,
        phase: str,
    ) -> Lfm2CpcfInstallablePhaseEvidence:
        layers = tuple(
            self._layer_evidence(
                tensors,
                diagnostics,
                batch_index=batch_index,
                phase=phase,
                layer_index=layer_index,
            )
            for layer_index in range(len(self.topology))
        )
        output_diagnostics = diagnostics.get("output")
        if (
            not isinstance(output_diagnostics, Mapping)
            or output_diagnostics.get("final_norm_boundary_captured")
            is not True
            or output_diagnostics.get("lm_head_input_exactly_verified")
            is not True
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Batch {batch_index} phase {phase!r} final normalization "
                "diagnostics are not authenticated."
            )
        output = _copy_fields(
            tensors,
            f"{phase}.",
            _OUTPUT_FIELDS,
            batch_index=batch_index,
            phase=phase,
        )
        normalized = {
            name: _tensor(output[name], name, dimensions=3)
            for name in _OUTPUT_FIELDS
        }
        expected_shape = tuple(
            layers[-1].tensors["residual.post_layer"].shape
        )
        if any(
            tuple(value.shape) != expected_shape
            for value in normalized.values()
        ):
            raise Lfm2CpcfInstallabilityError(
                f"Batch {batch_index} phase {phase!r} final norm boundary "
                "is not aligned to the final residual."
            )
        _close(
            normalized["output.final_norm_output"],
            normalized["output.lm_head_input"],
            "final RMSNorm output to LM-head input",
            exact=True,
        )
        body = {
            "schema": LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA,
            "cache_key": self.cache_key,
            "batch_index": batch_index,
            "input_sha256": input_sha256,
            "phase": phase,
            "layer_evidence_sha256": [
                layer.evidence_sha256 for layer in layers
            ],
            "final_norm_field_inventory": _field_inventory(output),
            "parent_side_compiler_inputs_ready": True,
        }
        return Lfm2CpcfInstallablePhaseEvidence(
            cache_key=self.cache_key,
            batch_index=batch_index,
            input_sha256=input_sha256,
            phase=phase,
            layers=layers,
            final_norm_tensors=output,
            evidence_sha256=_sha(body),
        )

    def iter_compiler_inputs(
        self,
    ) -> Iterator[Lfm2CpcfInstallablePhaseEvidence]:
        """Stream exact parent-side inputs without retaining every cache batch."""

        for batch_index in range(self.batch_count):
            batch = self.cache.load_batch(batch_index)
            expected_identity = self.ordered_input_sha256[batch_index]
            if getattr(batch, "input_sha256", None) != expected_identity:
                raise Lfm2CpcfInstallabilityError(
                    f"Cache batch {batch_index} identity changed."
                )
            tensors = getattr(batch, "tensors", None)
            record = getattr(batch, "record", None)
            if not isinstance(tensors, Mapping) or not isinstance(
                record,
                Mapping,
            ):
                raise Lfm2CpcfInstallabilityError(
                    f"Cache batch {batch_index} is not a verified trace batch."
                )
            phases = _phase_names(record, batch_index)
            diagnostics = self._validate_capture_record(
                record,
                batch_index=batch_index,
                phases=phases,
            )
            for phase in phases:
                phase_diagnostics = diagnostics.get(phase)
                if not isinstance(phase_diagnostics, Mapping):
                    raise Lfm2CpcfInstallabilityError(
                        f"Cache batch {batch_index} phase {phase!r} has no "
                        "diagnostics."
                    )
                yield self._phase_evidence(
                    tensors,
                    phase_diagnostics,
                    batch_index=batch_index,
                    input_sha256=expected_identity,
                    phase=phase,
                )

    def audit(self) -> Lfm2CpcfInstallabilityAudit:
        phase_sha: list[str] = []
        attention_count = 0
        short_conv_count = 0
        for phase in self.iter_compiler_inputs():
            phase_sha.append(phase.evidence_sha256)
            attention_count += sum(
                layer.operator_kind == "full_attention"
                for layer in phase.layers
            )
            short_conv_count += sum(
                layer.operator_kind == "short_conv"
                for layer in phase.layers
            )
        if not phase_sha:
            raise Lfm2CpcfInstallabilityError(
                "Installability audit found no cache phases."
            )
        body = _installability_audit_body(
            cache_key=self.cache_key,
            layer_topology_sha256=self.layer_topology_sha256,
            capture_selection_sha256=self.capture_selection_sha256,
            batch_count=self.batch_count,
            phase_count=len(phase_sha),
            layer_count=len(self.topology),
            source_width=self.source_width,
            attention_phase_count=attention_count,
            short_conv_phase_count=short_conv_count,
            phase_evidence_sha256=phase_sha,
        )
        self.cache.verify()
        if getattr(self.cache, "cache_key", None) != self.cache_key:
            raise Lfm2CpcfInstallabilityError(
                "Cache identity changed during installability audit."
            )
        return Lfm2CpcfInstallabilityAudit(
            cache_key=self.cache_key,
            layer_topology_sha256=self.layer_topology_sha256,
            capture_selection_sha256=self.capture_selection_sha256,
            batch_count=self.batch_count,
            phase_count=len(phase_sha),
            layer_count=len(self.topology),
            source_width=self.source_width,
            attention_phase_count=attention_count,
            short_conv_phase_count=short_conv_count,
            phase_evidence_sha256=tuple(phase_sha),
            report_sha256=_sha(body),
        )


def iter_lfm2_cpcf_compiler_inputs(
    cache: Any,
) -> Iterator[Lfm2CpcfInstallablePhaseEvidence]:
    """Yield strict installability evidence; legacy caches fail immediately."""

    yield from Lfm2CpcfInstallabilityAdapter(cache).iter_compiler_inputs()


def audit_lfm2_cpcf_installability(
    cache: Any,
) -> Lfm2CpcfInstallabilityAudit:
    """Authenticate every opt-in parent-side CPCF compiler input."""

    return Lfm2CpcfInstallabilityAdapter(cache).audit()


__all__ = [
    "LFM2_CPCF_INSTALLABILITY_AUDIT_SCHEMA",
    "LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA",
    "LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA",
    "Lfm2CpcfInstallabilityAdapter",
    "Lfm2CpcfInstallabilityAudit",
    "Lfm2CpcfInstallabilityError",
    "Lfm2CpcfInstallableLayerEvidence",
    "Lfm2CpcfInstallablePhaseEvidence",
    "audit_lfm2_cpcf_installability",
    "iter_lfm2_cpcf_compiler_inputs",
]
