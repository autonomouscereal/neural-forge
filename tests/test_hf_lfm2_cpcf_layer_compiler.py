from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_cpcf_layer_compiler as layer_compiler  # noqa: E402
from neural_forge.hf_closed_form_causal_circuit_fold import (  # noqa: E402
    PredictiveBoundaryMap,
)
from neural_forge.hf_cpcf_boundary_chart import CpcfBoundaryChart  # noqa: E402
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (  # noqa: E402
    LFM2_CPCF_LAYER_CHARTS_SCHEMA,
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (  # noqa: E402
    build_lfm2_cpcf_target_phases,
    install_lfm2_cpcf_neutral_norms,
    seed_lfm2_cpcf_phase_states,
)


def _sha(value) -> str:
    if isinstance(value, str):
        value = value.encode()
    else:
        value = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    return hashlib.sha256(value).hexdigest()


def _rms(module, value):
    direction = value.to(torch.float32)
    direction = direction * torch.rsqrt(
        direction.square().mean(dim=-1, keepdim=True)
        + float(module.variance_epsilon)
    )
    return module.weight * direction.to(value.dtype)


def _rotate_half(value):
    half = int(value.shape[-1]) // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _rope(model, positions, dtype):
    inv = model.model.rotary_emb.inv_freq.to(torch.float32)
    frequency = positions.to(torch.float32).unsqueeze(-1) * inv.reshape(1, 1, -1)
    angle = torch.cat((frequency, frequency), dim=-1)
    scale = float(model.model.rotary_emb.attention_scaling)
    return (
        (angle.cos() * scale).to(dtype),
        (angle.sin() * scale).to(dtype),
    )


def _config(kind: str, *, hidden: int, intermediate: int):
    heads = hidden // 2
    return transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=max(1, heads // 2),
        head_dim=2,
        layer_types=[kind],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )


def _models(kind: str):
    layer_type = "conv" if kind == "short_conv" else kind
    torch.manual_seed(811)
    source = transformers.Lfm2ForCausalLM(
        _config(layer_type, hidden=8, intermediate=12)
    ).eval()
    torch.manual_seed(977)
    target = transformers.Lfm2ForCausalLM(
        _config(layer_type, hidden=4, intermediate=6)
    ).eval()
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            if "embed_tokens" not in name and "lm_head" not in name:
                parameter.zero_()
    install_lfm2_cpcf_neutral_norms(target)
    return source, target


def _mlp_fields(layer, post_operator, sampled):
    mlp_input = _rms(layer.ffn_norm, post_operator)
    gate = torch.nn.functional.linear(mlp_input, layer.feed_forward.w1.weight)
    up = torch.nn.functional.linear(mlp_input, layer.feed_forward.w3.weight)
    post = torch.nn.functional.silu(gate) * up
    down = torch.nn.functional.linear(post, layer.feed_forward.w2.weight)
    return {
        "residual.pre": None,
        "residual.post_operator": post_operator[:, -sampled:, :].clone(),
        "mlp.input": mlp_input[:, -sampled:, :].clone(),
        "mlp.gate": gate[:, -sampled:, :].clone(),
        "mlp.up": up[:, -sampled:, :].clone(),
        "mlp.post_swiglu": post[:, -sampled:, :].clone(),
        "mlp.down": down[:, -sampled:, :].clone(),
        "residual.post_layer": (
            post_operator + down
        )[:, -sampled:, :].clone(),
    }


def _attention_evidence(model, residual, sampled):
    layer = model.model.layers[0]
    attention = layer.self_attn
    batch, sequence, hidden = residual.shape
    operator_input = _rms(layer.operator_norm, residual)
    q_linear = torch.nn.functional.linear(operator_input, attention.q_proj.weight)
    k_linear = torch.nn.functional.linear(operator_input, attention.k_proj.weight)
    v_linear = torch.nn.functional.linear(operator_input, attention.v_proj.weight)
    q_heads = int(attention.q_proj.weight.shape[0]) // int(attention.head_dim)
    kv_heads = int(attention.k_proj.weight.shape[0]) // int(attention.head_dim)
    q_linear_heads = q_linear.view(batch, sequence, q_heads, 2)
    k_linear_heads = k_linear.view(batch, sequence, kv_heads, 2)
    value = v_linear.view(batch, sequence, kv_heads, 2).transpose(1, 2)
    q_norm = _rms(attention.q_layernorm, q_linear_heads).transpose(1, 2)
    k_norm = _rms(attention.k_layernorm, k_linear_heads).transpose(1, 2)
    positions = torch.arange(sequence).unsqueeze(0).expand(batch, -1)
    cosine, sine = _rope(model, positions, residual.dtype)
    query = q_norm * cosine.unsqueeze(1) + _rotate_half(q_norm) * sine.unsqueeze(1)
    key = k_norm * cosine.unsqueeze(1) + _rotate_half(k_norm) * sine.unsqueeze(1)
    repeated_key = key.repeat_interleave(q_heads // kv_heads, dim=1)
    repeated_value = value.repeat_interleave(q_heads // kv_heads, dim=1)
    scores = torch.matmul(
        query[:, :, -sampled:, :],
        repeated_key.transpose(2, 3),
    ) * float(attention.scaling)
    query_positions = torch.arange(sequence - sampled, sequence)
    key_positions = torch.arange(sequence)
    allowed = key_positions.reshape(1, -1) <= query_positions.reshape(-1, 1)
    additive = torch.where(
        allowed,
        torch.zeros((), dtype=residual.dtype),
        torch.full((), torch.finfo(residual.dtype).min, dtype=residual.dtype),
    ).reshape(1, 1, sampled, sequence).expand(batch, -1, -1, -1)
    probabilities = torch.softmax(
        scores + additive,
        dim=-1,
        dtype=torch.float32,
    ).to(residual.dtype)
    context = torch.matmul(probabilities, repeated_value)
    output = torch.nn.functional.linear(
        context.transpose(1, 2).reshape(batch, sampled, hidden),
        attention.out_proj.weight,
    )
    post_operator = residual.clone()
    post_operator[:, -sampled:, :] += output
    # Earlier full-prefix output is needed only to produce the correct full
    # MLP path in this synthetic parent evidence.
    full_scores = torch.matmul(
        query,
        repeated_key.transpose(2, 3),
    ) * float(attention.scaling)
    full_allowed = torch.tril(torch.ones(sequence, sequence, dtype=torch.bool))
    full_additive = torch.where(
        full_allowed,
        torch.zeros((), dtype=residual.dtype),
        torch.full((), torch.finfo(residual.dtype).min, dtype=residual.dtype),
    ).reshape(1, 1, sequence, sequence)
    full_probabilities = torch.softmax(
        full_scores + full_additive,
        dim=-1,
        dtype=torch.float32,
    ).to(residual.dtype)
    full_context = torch.matmul(full_probabilities, repeated_value)
    full_output = torch.nn.functional.linear(
        full_context.transpose(1, 2).reshape(batch, sequence, hidden),
        attention.out_proj.weight,
    )
    full_post_operator = residual + full_output
    tensors = _mlp_fields(layer, full_post_operator, sampled)
    tensors.update(
        {
            "residual.pre": residual[:, -sampled:, :].clone(),
            "operator.input": operator_input[:, -sampled:, :].clone(),
            "operator.output": output.clone(),
            "residual.post_operator": full_post_operator[
                :, -sampled:, :
            ].clone(),
            "attention.q_linear": q_linear_heads[
                :, -sampled:, :, :
            ].transpose(1, 2).contiguous(),
            "attention.q": query[:, :, -sampled:, :].clone(),
            "attention.k_linear": k_linear_heads[
                :, -sampled:, :, :
            ].transpose(1, 2).contiguous(),
            "attention.k": key[:, :, -sampled:, :].clone(),
            "attention.v": value[:, :, -sampled:, :].clone(),
            "attention.context": context.clone(),
            "attention.o": output.clone(),
            "attention.q_normalized_pre_rope": q_norm.clone(),
            "attention.k_normalized_pre_rope": k_norm.clone(),
            "attention.rope_cos": cosine.clone(),
            "attention.rope_sin": sine.clone(),
            "attention.key_history": key.clone(),
            "attention.value_history": value.clone(),
            "attention.qk_scores": scores.clone(),
            "attention.additive_mask": additive.clone(),
            "attention.probabilities": probabilities.clone(),
            "attention.context_from_probabilities": context.clone(),
        }
    )
    return tensors


def _conv_evidence(model, residual, sampled):
    layer = model.model.layers[0]
    conv = layer.conv
    batch, sequence, _hidden = residual.shape
    operator_input = _rms(layer.operator_norm, residual)
    projected = torch.nn.functional.linear(
        operator_input,
        conv.in_proj.weight,
        conv.in_proj.bias,
    )
    b_value, c_value, x_value = projected.chunk(3, dim=-1)
    filtered = torch.nn.functional.conv1d(
        (b_value * x_value).transpose(1, 2),
        conv.conv.weight,
        conv.conv.bias,
        padding=int(conv.conv.weight.shape[-1]) - 1,
        groups=int(residual.shape[-1]),
    )[..., :sequence]
    context = c_value * filtered.transpose(1, 2)
    output = torch.nn.functional.linear(
        context,
        conv.out_proj.weight,
        conv.out_proj.bias,
    )
    post_operator = residual + output
    tensors = _mlp_fields(layer, post_operator, sampled)
    tensors.update(
        {
            "residual.pre": residual[:, -sampled:, :].clone(),
            "operator.input": operator_input[:, -sampled:, :].clone(),
            "operator.output": output[:, -sampled:, :].clone(),
            "residual.post_operator": post_operator[:, -sampled:, :].clone(),
            "short_conv.b_history": b_value.clone(),
            "short_conv.c_history": c_value.clone(),
            "short_conv.x_history": x_value.clone(),
        }
    )
    return tensors


class _Adapter:
    def __init__(self, records):
        self.records = tuple(records)

    def verify(self):
        return self


def _chart(kind, source_width=8, target_width=4):
    generator = torch.Generator().manual_seed(1441)
    encoder = torch.randn(
        source_width,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    decoder = torch.linalg.pinv(encoder)
    mapping = PredictiveBoundaryMap(
        encoder=encoder,
        decoder=decoder,
        feature_predictors={},
        report={},
    )
    post_operator = CpcfBoundaryChart(
        name="layer.000.residual.post_operator",
        mapping=mapping,
        report={"report_sha256": _sha("post-operator")},
    )
    post_layer = CpcfBoundaryChart(
        name="layer.000.residual.post_layer",
        mapping=mapping,
        report={"report_sha256": _sha("post-layer")},
    )
    body = {
        "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
        "layer_index": 0,
        "operator_kind": kind,
        "source_width": source_width,
        "target_width": target_width,
        "post_operator_report_sha256": post_operator.report["report_sha256"],
        "post_layer_report_sha256": post_layer.report["report_sha256"],
        "independent_boundary_coordinates": True,
    }
    return Lfm2CpcfLayerBoundaryCharts(
        layer_index=0,
        operator_kind=kind,
        post_operator=post_operator,
        post_layer=post_layer,
        layer_charts_sha256=_sha(body),
    )


def _fixture(kind):
    source, target = _models(kind)
    generator = torch.Generator().manual_seed(2027)
    records = []
    evidence = []
    splits = ("selection", "fit", "validation")
    for ordinal, split in enumerate(splits):
        residual = torch.randn(2, 6, 8, generator=generator) + ordinal * 0.05
        tensors = (
            _attention_evidence(source, residual, 4)
            if kind == "full_attention"
            else _conv_evidence(source, residual, 4)
        )
        donor_sha = _sha(f"donor-{ordinal}")
        layer = Lfm2CpcfInstallableLayerEvidence(
            layer_index=0,
            operator_kind=kind,
            source_width=8,
            tensors=tensors,
            evidence_sha256=_sha(f"layer-{ordinal}"),
        )
        evidence.append(
            Lfm2CpcfInstallablePhaseEvidence(
                cache_key=_sha("cache"),
                batch_index=ordinal,
                input_sha256=donor_sha,
                phase="prefill",
                layers=(layer,),
                final_norm_tensors={},
                evidence_sha256=_sha(f"phase-{ordinal}"),
            )
        )
        ids = torch.tensor(
            [
                [2 + ordinal, 4, 5, 6, 7, 8],
                [3 + ordinal, 5, 6, 7, 8, 9],
            ]
        )
        records.append(
            SimpleNamespace(
                ordinal=ordinal,
                record_id=f"row-{ordinal}",
                split=split,
                donor_cache_input_sha256=donor_sha,
                alignment_sha256=_sha(f"alignment-{ordinal}"),
                target_input_ids=ids,
                attention_mask=torch.ones_like(ids),
                target_forced_decode_ids=torch.full((2, 1), 10),
                position_ids=None,
            )
        )
    phases = build_lfm2_cpcf_target_phases(_Adapter(records), tuple(evidence))
    bank = seed_lfm2_cpcf_phase_states(target, phases)
    return source, target, bank, _chart(kind)


def _state_sha(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy())
    return digest.hexdigest()


@pytest.mark.parametrize("kind", ["full_attention", "short_conv"])
def test_real_lfm2_layer_compiler_replays_actual_child_and_holds_out_validation(
    kind,
    monkeypatch,
):
    source, target, bank, charts = _fixture(kind)
    source_before = _state_sha(source.model.layers[0])
    target_ids = {
        name: id(parameter)
        for name, parameter in target.model.layers[0].named_parameters()
    }
    observed = {}
    original_mlp = layer_compiler.compile_lfm2_structural_residual_mlp_fold

    def mlp_spy(**kwargs):
        observed["mlp_case_count"] = len(kwargs["cases"])
        return original_mlp(**kwargs)

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_structural_residual_mlp_fold",
        mlp_spy,
    )
    if kind == "full_attention":
        original_operator = (
            layer_compiler.compile_lfm2_structural_residual_attention_fold
        )

        def attention_spy(**kwargs):
            observed["operator_case_count"] = len(kwargs["cases"])
            return original_operator(**kwargs)

        monkeypatch.setattr(
            layer_compiler,
            "compile_lfm2_structural_residual_attention_fold",
            attention_spy,
        )
    else:
        original_operator = layer_compiler.compile_lfm2_cpcf_short_conv_fold

        def conv_spy(**kwargs):
            observed["operator_splits"] = tuple(
                case.split for case in kwargs["cases"]
            )
            return original_operator(**kwargs)

        monkeypatch.setattr(
            layer_compiler,
            "compile_lfm2_cpcf_short_conv_fold",
            conv_spy,
        )

    result = layer_compiler.compile_and_install_lfm2_cpcf_layer(
        source,
        target,
        layer_index=0,
        input_bank=bank,
        boundary_charts=charts,
    )

    assert result.next_bank.layer_index == 1
    assert result.input_bank_sha256 == bank.state_bank_sha256
    assert set(result.split_metrics) == {"fit", "validation"}
    assert observed["mlp_case_count"] == 1
    if kind == "full_attention":
        assert observed["operator_case_count"] == 1
    else:
        assert observed["operator_splits"] == ("fit", "validation")
        assert (
            result.operator_artifact.report["behavior_gate"][
                "heldout_validation_used_for_fitting"
            ]
            is False
        )
    policy = result.report["solver_evidence_policy"]
    assert policy["validation_used_for_attention_fit"] is False
    assert policy["validation_used_for_mlp_fit"] is False
    assert policy["validation_used_for_short_conv_fit"] is False
    assert _state_sha(source.model.layers[0]) == source_before
    assert {
        name: id(parameter)
        for name, parameter in target.model.layers[0].named_parameters()
    } == target_ids
    assert result.report["execution"]["training_steps"] == 0
    assert result.report["execution"]["target_model_forward_calls"] == 0


def test_layer_transaction_rolls_back_operator_when_mlp_compile_fails(
    monkeypatch,
):
    source, target, bank, charts = _fixture("full_attention")
    target_before = copy.deepcopy(target.model.layers[0].state_dict())
    target_ids = {
        name: id(parameter)
        for name, parameter in target.model.layers[0].named_parameters()
    }

    def fail_mlp(**_kwargs):
        raise RuntimeError("forced post-operator failure")

    monkeypatch.setattr(
        layer_compiler,
        "compile_lfm2_structural_residual_mlp_fold",
        fail_mlp,
    )
    with pytest.raises(RuntimeError, match="forced post-operator failure"):
        layer_compiler.compile_and_install_lfm2_cpcf_layer(
            source,
            target,
            layer_index=0,
            input_bank=bank,
            boundary_charts=charts,
        )

    for name, value in target.model.layers[0].state_dict().items():
        torch.testing.assert_close(value, target_before[name], rtol=0, atol=0)
    assert {
        name: id(parameter)
        for name, parameter in target.model.layers[0].named_parameters()
    } == target_ids


def test_inverse_rope_and_full_key_recovery_are_functional() -> None:
    generator = torch.Generator().manual_seed(31337)
    linear = torch.randn(2, 1, 5, 4, generator=generator)
    weight = torch.tensor([0.9, 1.1, 0.8, 1.2])
    normalized = layer_compiler._runtime_rms_norm(  # noqa: SLF001
        linear,
        weight.reshape(1, 1, 1, -1),
        1e-6,
    )
    positions = torch.arange(5, dtype=torch.float32)
    frequencies = positions.reshape(1, 5, 1) * torch.tensor([0.7, 0.2])
    cosine = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1)
    sine = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1)
    cosine = cosine.expand(2, -1, -1).contiguous()
    sine = sine.expand(2, -1, -1).contiguous()
    rotated = (
        normalized * cosine.unsqueeze(1)
        + _rotate_half(normalized) * sine.unsqueeze(1)
    )

    inverted = layer_compiler.invert_lfm2_rope(rotated, cosine, sine)
    torch.testing.assert_close(inverted, normalized, rtol=1e-5, atol=1e-6)
    recovered = layer_compiler.recover_lfm2_full_key_linear_history(
        parent_key_history=rotated,
        parent_key_norm_weight=weight,
        full_cosine=cosine,
        full_sine=sine,
        norm_epsilon=1e-6,
        cached_current_key_linear=linear[:, :, -2:, :],
        cached_current_normalized_pre_rope=normalized,
    )
    reconstructed = layer_compiler._runtime_rms_norm(  # noqa: SLF001
        recovered,
        weight.reshape(1, 1, 1, -1),
        1e-6,
    )
    reconstructed = (
        reconstructed * cosine.unsqueeze(1)
        + _rotate_half(reconstructed) * sine.unsqueeze(1)
    )
    torch.testing.assert_close(reconstructed, rotated, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(
        recovered[:, :, -2:, :],
        linear[:, :, -2:, :],
        rtol=0,
        atol=0,
    )
