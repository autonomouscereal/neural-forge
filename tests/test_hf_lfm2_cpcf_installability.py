from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallabilityAdapter,
    Lfm2CpcfInstallabilityError,
    audit_lfm2_cpcf_installability,
    iter_lfm2_cpcf_compiler_inputs,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class _Batch:
    input_sha256: str
    tensors: dict[str, Any]
    record: dict[str, Any]


class _Cache:
    def __init__(self, batch: _Batch, *, opt_in: bool = True) -> None:
        self.cache_key = _digest("installability-cache")
        self.batch_count = 1
        self._batch = batch
        self.verify_count = 0
        self.load_count = 0
        self.parent_forward_count = 0
        capture = {
            "sample_last_tokens": 2,
            "max_decode_steps": 1,
            "max_capture_bytes": 100_000_000,
            "logit_rtol": 1e-5,
            "logit_atol": 1e-6,
            "require_top1_agreement": True,
            "inference_mode": True,
            "model_eval_mode": True,
            "gradients_used": False,
            "training_steps": 0,
            "prefill_use_cache": False,
            "cached_prefill_use_cache": True,
            "forced_decode_use_cache": True,
            "full_prefix_verification_use_cache": False,
        }
        if opt_in:
            capture.update(
                {
                    "capture_causal_circuit_evidence": True,
                    "attention_pre_rope_qk": True,
                    "attention_exact_causal_scores_and_probabilities": True,
                    "attention_complete_kv_history": True,
                    "short_conv_exact_fir_state_and_response": True,
                    "final_normalization_boundary": True,
                }
            )
        capture["selection_sha256"] = _canonical_sha(capture)
        self.manifest = {
            "capture": capture,
            "corpus": {
                "ordered_batch_input_sha256": [batch.input_sha256],
            },
            "layer_topology": {
                "layer_topology_sha256": _digest("topology"),
                "layers": [
                    {
                        "contract": {
                            "layer_index": 0,
                            "kind": "short_conv",
                            "hidden_size": 4,
                            "ffn_intermediate_size": 6,
                            "operator": {
                                "fir_length": 3,
                                "input_projection_chunk_order": [
                                    "b",
                                    "c",
                                    "x",
                                ],
                                "not_attention_qkv": True,
                            },
                        }
                    },
                    {
                        "contract": {
                            "layer_index": 1,
                            "kind": "full_attention",
                            "hidden_size": 4,
                            "ffn_intermediate_size": 6,
                            "operator": {
                                "query_heads": 2,
                                "key_value_heads": 1,
                                "head_dim": 2,
                                "qk_rmsnorm": True,
                            },
                        }
                    },
                ],
            },
        }

    def verify(self) -> "_Cache":
        self.verify_count += 1
        return self

    def load_batch(self, index: int) -> _Batch:
        assert index == 0
        self.load_count += 1
        return self._batch

    def forward(self, *_args: Any, **_kwargs: Any) -> None:
        self.parent_forward_count += 1
        raise AssertionError("Installability adapter executed a parent.")


def _common(
    *,
    batch: int,
    sampled: int,
    hidden: int,
    intermediate: int,
    offset: float,
) -> dict[str, Any]:
    residual = (
        torch.arange(
            batch * sampled * hidden,
            dtype=torch.float32,
        ).reshape(batch, sampled, hidden)
        * 0.01
        + offset
    )
    operator = residual * 0.1
    post_operator = residual + operator
    gate = torch.ones(
        (batch, sampled, intermediate),
        dtype=torch.float32,
    ) * (0.5 + offset)
    up = torch.ones_like(gate) * 0.75
    post_swiglu = torch.nn.functional.silu(gate) * up
    down = residual * 0.05
    return {
        "residual.pre": residual,
        "operator.input": residual.clone(),
        "operator.output": operator,
        "residual.post_operator": post_operator,
        "mlp.input": post_operator.clone(),
        "mlp.gate": gate,
        "mlp.up": up,
        "mlp.post_swiglu": post_swiglu,
        "mlp.down": down,
        "residual.post_layer": post_operator + down,
    }


def _conv_fields(
    *,
    common: dict[str, Any],
    history: int,
) -> dict[str, Any]:
    batch, sampled, hidden = common["residual.pre"].shape
    b_history = (
        torch.arange(
            int(batch * history * hidden),
            dtype=torch.float32,
        ).reshape(int(batch), history, int(hidden))
        * 0.01
        + 0.2
    )
    c_history = torch.ones_like(b_history) * 0.5
    x_history = torch.ones_like(b_history) * 0.8
    bx_history = b_history * x_history
    kernel = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.3, 0.2, 0.1],
            [0.1, 0.4, 0.2],
        ],
        dtype=torch.float32,
    )
    bias = torch.zeros(int(hidden), dtype=torch.float32)
    pre_state = torch.zeros(
        (int(batch), int(hidden), 3),
        dtype=torch.float32,
    )
    filtered = torch.nn.functional.conv1d(
        bx_history.transpose(1, 2),
        kernel[:, None, :],
        bias=bias,
        padding=2,
        groups=int(hidden),
    )[..., :history].transpose(1, 2)
    context = c_history * filtered
    return {
        "short_conv.b": b_history[:, -sampled:, :].clone(),
        "short_conv.c": c_history[:, -sampled:, :].clone(),
        "short_conv.x": x_history[:, -sampled:, :].clone(),
        "short_conv.bx": bx_history[:, -sampled:, :].clone(),
        "short_conv.context": context[:, -sampled:, :].clone(),
        "short_conv.o": common["operator.output"].clone(),
        "short_conv.b_history": b_history,
        "short_conv.c_history": c_history,
        "short_conv.x_history": x_history,
        "short_conv.bx_history": bx_history,
        "short_conv.pre_bx_state": pre_state,
        "short_conv.fir_kernel": kernel,
        "short_conv.fir_bias": bias,
        "short_conv.fir_has_bias": torch.tensor([0], dtype=torch.int8),
        "short_conv.filtered_bx": filtered,
        "short_conv.context_from_fir": context,
    }


def _attention_fields(
    *,
    common: dict[str, Any],
    current: int,
    prior: int,
) -> dict[str, Any]:
    batch, sampled, _hidden = common["residual.pre"].shape
    query_heads, kv_heads, head_dim = 2, 1, 2
    q_norm = (
        torch.arange(
            int(batch * query_heads * current * head_dim),
            dtype=torch.float32,
        ).reshape(int(batch), query_heads, current, head_dim)
        * 0.03
        + 0.1
    )
    k_norm = (
        torch.arange(
            int(batch * kv_heads * current * head_dim),
            dtype=torch.float32,
        ).reshape(int(batch), kv_heads, current, head_dim)
        * 0.02
        + 0.2
    )
    cos = torch.ones((int(batch), current, head_dim))
    sin = torch.zeros_like(cos)
    q_rotary = q_norm.clone()
    k_rotary = k_norm.clone()
    if prior:
        prior_key = torch.ones(
            (int(batch), kv_heads, prior, head_dim),
            dtype=torch.float32,
        ) * 0.15
        prior_value = torch.ones_like(prior_key) * 0.25
        key_history = torch.cat((prior_key, k_rotary), dim=2)
    else:
        prior_value = None
        key_history = k_rotary
    current_value = k_norm * 0.7
    value_history = (
        current_value
        if prior_value is None
        else torch.cat((prior_value, current_value), dim=2)
    )
    repeated_key = key_history.repeat_interleave(2, dim=1)
    repeated_value = value_history.repeat_interleave(2, dim=1)
    scores = torch.matmul(
        q_rotary[:, :, -int(sampled) :, :],
        repeated_key.transpose(2, 3),
    ) / (head_dim**0.5)
    additive_mask = torch.zeros(
        (int(batch), 1, int(sampled), int(key_history.shape[2])),
        dtype=torch.float32,
    )
    probabilities = torch.softmax(scores + additive_mask, dim=-1)
    context = torch.matmul(probabilities, repeated_value)
    q_sample = q_rotary[:, :, -int(sampled) :, :]
    k_sample = k_rotary[:, :, -int(sampled) :, :]
    v_sample = current_value[:, :, -int(sampled) :, :]
    return {
        "attention.q_linear": q_sample.clone(),
        "attention.q": q_sample,
        "attention.k_linear": k_sample.clone(),
        "attention.k": k_sample,
        "attention.v": v_sample,
        "attention.context": context.clone(),
        "attention.o": common["operator.output"].clone(),
        "attention.q_normalized_pre_rope": q_norm,
        "attention.k_normalized_pre_rope": k_norm,
        "attention.rope_cos": cos,
        "attention.rope_sin": sin,
        "attention.key_history": key_history,
        "attention.value_history": value_history,
        "attention.qk_scores": scores,
        "attention.additive_mask": additive_mask,
        "attention.probabilities": probabilities,
        "attention.context_from_probabilities": context,
    }


def _phase(
    phase: str,
    *,
    sampled: int,
    current: int,
    prior: int,
    conv_history: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensors: dict[str, Any] = {}
    conv_common = _common(
        batch=1,
        sampled=sampled,
        hidden=4,
        intermediate=6,
        offset=0.1,
    )
    conv = {**conv_common, **_conv_fields(
        common=conv_common,
        history=conv_history,
    )}
    for name, value in conv.items():
        tensors[f"{phase}.layer.000.{name}"] = value

    attention_common = _common(
        batch=1,
        sampled=sampled,
        hidden=4,
        intermediate=6,
        offset=0.3,
    )
    attention = {
        **attention_common,
        **_attention_fields(
            common=attention_common,
            current=current,
            prior=prior,
        ),
    }
    for name, value in attention.items():
        tensors[f"{phase}.layer.001.{name}"] = value

    final_input = attention_common["residual.post_layer"].clone()
    final_output = final_input * 0.9
    tensors[f"{phase}.output.final_norm_input"] = final_input
    tensors[f"{phase}.output.final_norm_output"] = final_output
    tensors[f"{phase}.output.lm_head_input"] = final_output.clone()
    diagnostics = {
        "layer.000": {
            "short_conv_causal_circuit_evidence_captured": True,
            "short_conv_context_reconstructed_from_fir": True,
        },
        "layer.001": {
            "attention_causal_circuit_evidence_captured": True,
            "attention_context_reconstructed_from_probabilities": True,
        },
        "output": {
            "final_norm_boundary_captured": True,
            "lm_head_input_exactly_verified": True,
        },
    }
    return tensors, diagnostics


def _cache(*, opt_in: bool = True) -> _Cache:
    tensors: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    prefill, prefill_diagnostics = _phase(
        "prefill",
        sampled=2,
        current=3,
        prior=0,
        conv_history=3,
    )
    decode, decode_diagnostics = _phase(
        "decode.000",
        sampled=1,
        current=1,
        prior=3,
        conv_history=1,
    )
    tensors.update(prefill)
    tensors.update(decode)
    diagnostics["prefill"] = prefill_diagnostics
    diagnostics["decode.000"] = decode_diagnostics
    identity = _digest("batch-zero")
    batch = _Batch(
        input_sha256=identity,
        tensors=tensors,
        record={
            "capture": {
                "decode_steps": 1,
                "phase_diagnostics": diagnostics,
                "decode_trace_mode": "cached_verified",
                "canonical_trace_source": "verified_one_token_cache",
                "cached_logits_accepted": True,
                "all_cached_logits_verified": True,
                "all_cached_top1_verified": True,
                "gradients_used": False,
                "training_steps": 0,
            }
        },
    )
    return _Cache(batch, opt_in=opt_in)


def test_strict_installability_audit_passes_complete_opt_in_cache() -> None:
    cache = _cache()

    result = audit_lfm2_cpcf_installability(cache)
    report = result.as_dict()

    assert result.batch_count == 1
    assert result.phase_count == 2
    assert result.layer_count == 2
    assert result.attention_phase_count == 2
    assert result.short_conv_phase_count == 2
    assert report["parent_side_compiler_inputs_ready"] is True
    assert report["measurement_only"] is False
    assert report["target_child_compiled"] is False
    assert report["operation_audit"]["parent_forward_count"] == 0
    assert cache.parent_forward_count == 0
    assert cache.verify_count == 2
    assert cache.load_count == 1


def test_canonical_json_phase_key_order_is_not_treated_as_causal_order() -> None:
    cache = _cache()
    capture = cache._batch.record["capture"]
    diagnostics = capture["phase_diagnostics"]
    capture["phase_diagnostics"] = {
        name: diagnostics[name] for name in sorted(diagnostics)
    }

    phases = tuple(iter_lfm2_cpcf_compiler_inputs(cache))

    assert [phase.phase for phase in phases] == ["prefill", "decode.000"]
    assert cache.parent_forward_count == 0


def test_missing_phase_diagnostic_still_fails_closed() -> None:
    cache = _cache()
    del cache._batch.record["capture"]["phase_diagnostics"]["decode.000"]

    with pytest.raises(
        Lfm2CpcfInstallabilityError,
        match="incomplete or contain unexpected",
    ):
        tuple(iter_lfm2_cpcf_compiler_inputs(cache))

    assert cache.parent_forward_count == 0


def test_attention_softmax_replays_authenticated_execution_scores() -> None:
    cache = _cache()
    values = cache._batch.tensors
    prefix = "prefill.layer.001."
    q_norm = values[prefix + "attention.q_normalized_pre_rope"]
    k_norm = values[prefix + "attention.k_normalized_pre_rope"]
    q_norm.fill_(8.5)
    k_norm.fill_(8.5)
    sampled = int(values[prefix + "attention.q"].shape[2])
    q = q_norm[:, :, -sampled:, :].clone()
    k = k_norm[:, :, -sampled:, :].clone()
    values[prefix + "attention.q"] = q
    values[prefix + "attention.q_linear"] = q.clone()
    values[prefix + "attention.k"] = k
    values[prefix + "attention.k_linear"] = k.clone()
    values[prefix + "attention.key_history"] = k_norm.clone()
    repeated_key = k_norm.repeat_interleave(2, dim=1)
    captured_scores = torch.matmul(
        q,
        repeated_key.transpose(2, 3),
    ) / (2**0.5)
    captured_scores[..., 0] += 1.5
    values[prefix + "attention.qk_scores"] = captured_scores
    probabilities = torch.nn.functional.softmax(
        captured_scores,
        dim=-1,
        dtype=torch.float32,
    )
    values[prefix + "attention.probabilities"] = probabilities
    repeated_value = values[
        prefix + "attention.value_history"
    ].repeat_interleave(2, dim=1)
    context = torch.matmul(probabilities, repeated_value)
    values[prefix + "attention.context_from_probabilities"] = context
    values[prefix + "attention.context"] = context.clone()

    phases = tuple(iter_lfm2_cpcf_compiler_inputs(cache))

    assert [phase.phase for phase in phases] == ["prefill", "decode.000"]
    assert cache.parent_forward_count == 0


def test_compiler_input_stream_exposes_exact_installability_tensors() -> None:
    cache = _cache()

    phases = tuple(iter_lfm2_cpcf_compiler_inputs(cache))

    assert [phase.phase for phase in phases] == ["prefill", "decode.000"]
    assert "short_conv.fir_kernel" in phases[0].layers[0].tensors
    assert (
        "attention.q_normalized_pre_rope"
        in phases[0].layers[1].tensors
    )
    assert "attention.probabilities" in phases[1].layers[1].tensors
    assert "output.final_norm_input" in phases[0].final_norm_tensors
    assert cache.parent_forward_count == 0


def test_legacy_measurement_cache_fails_installability_immediately() -> None:
    cache = _cache(opt_in=False)

    with pytest.raises(
        Lfm2CpcfInstallabilityError,
        match="capture_causal_circuit_evidence",
    ):
        Lfm2CpcfInstallabilityAdapter(cache)

    assert cache.load_count == 0
    assert cache.parent_forward_count == 0


def test_missing_required_installability_field_fails_closed() -> None:
    cache = _cache()
    del cache._batch.tensors[
        "prefill.layer.001.attention.rope_cos"
    ]

    with pytest.raises(
        Lfm2CpcfInstallabilityError,
        match="attention.rope_cos",
    ):
        audit_lfm2_cpcf_installability(cache)

    assert cache.parent_forward_count == 0


@pytest.mark.parametrize(
    ("field", "match"),
    (
        (
            "prefill.layer.001.attention.probabilities",
            "masked-softmax probabilities",
        ),
        (
            "prefill.layer.000.short_conv.filtered_bx",
            "exact FIR response",
        ),
        (
            "prefill.output.final_norm_output",
            "final RMSNorm output",
        ),
    ),
)
def test_corrupted_functional_installability_field_fails_closed(
    field: str,
    match: str,
) -> None:
    cache = _cache()
    cache._batch.tensors[field].add_(0.5)

    with pytest.raises(Lfm2CpcfInstallabilityError, match=match):
        audit_lfm2_cpcf_installability(cache)

    assert cache.parent_forward_count == 0
